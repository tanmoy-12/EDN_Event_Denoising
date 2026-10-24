from __future__ import annotations

import hashlib
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

# ---- ED24 csv column layout ----
T_COL, X_COL, Y_COL, P_COL, LABEL_COL = 0, 1, 2, 3, 4

# DAVIS346 by default; other corpora use other sensors (DVSCLEAN is 1280x720,
# recorded with a CeleX-V), so this is settable at run time.  It matters for
# more than bookkeeping: normalize_window maps pixels to [-1,1] using these,
# and the density descriptor's "expected count under uniform background
# activity" divides by the sensor area.  Leaving it at 346x260 on a 1280x720
# recording silently rescales the input geometry and biases every density
# feature.
SENSOR_W, SENSOR_H = 346, 260


def set_sensor_geometry(w: int, h: int):
    """Set the sensor dimensions for this process, in this module and in
    edn_model, which keeps its own copy."""
    global SENSOR_W, SENSOR_H
    SENSOR_W, SENSOR_H = int(w), int(h)
    import edn_model
    edn_model.SENSOR_W, edn_model.SENSOR_H = int(w), int(h)

# ED24 ships 41 attenuator-voltage levels per scene (0.0 to 4.0 in steps of
# 0.1).  The paper annotates and releases 21 of them, 1.5V to 3.5V:
#   "we employed an inverted microscope and a DAVIS346 event camera to
#    respectively capture pure BA noise under 21 different illumination
#    conditions, ranging from 1.5V to 3.5V for attenuator voltage (BA noise
#    below 1.5V was too sparse, and noise distribution above 3.5V was nearly
#    uniform)"
# Higher voltage = more attenuation = lower illuminance = MORE BA noise, so
# the noise fraction is monotonically increasing in this list.  That
# monotonicity is used as a second, independent convention check below.
ED24_VOLTAGES = ['1.5', '1.6', '1.7', '1.8', '1.9', '2.0', '2.1', '2.2', '2.3',
                 '2.4', '2.5', '2.6', '2.7', '2.8', '2.9', '3.0', '3.1', '3.2',
                 '3.3', '3.4', '3.5']

POSITIVE_CLASS_IS_NOISE = True  # label==1 is noise; see module docstring


# ==============================================================================
# 1. RAW FILE READING (with an on-disk cache)
# ==============================================================================
def _read_events_txt(path: str, skip_first_row: bool = True) -> np.ndarray:
    """Read one ED24 / AUC_test file into an (n, 5) int32 array.

    Header detection is now automatic rather than assumed.  ED24 csvs carry a
    real text header ("timestamp x y polarity label"); the AUC_test
    *_result.txt files are the DND21-derived ones and may or may not.  Guessing
    wrong silently drops one event (harmless) or crashes the int cast (not),
    so we look at the first line instead of trusting a flag.
    """
    import pandas as pd

    with open(path, "r") as fh:
        first = fh.readline()
    tok = first.replace(",", " ").split()
    has_header = not all(_is_number(v) for v in tok) if tok else False
    header = 0 if (has_header and skip_first_row) else None

    df = pd.read_csv(path, sep=r"\s+", header=header, engine="c")
    arr = df.to_numpy()
    if arr.dtype.kind not in "iu":
        arr = arr.astype(np.float64)
        arr = arr[~np.isnan(arr).any(axis=1)]
    arr = arr.astype(np.int64)

    if arr.shape[1] < 4:
        raise ValueError(f"{path}: expected >=4 columns, got {arr.shape[1]}")
    if arr.shape[1] == 4:  # unlabelled stream
        arr = np.concatenate([arr, np.zeros((len(arr), 1), np.int64)], axis=1)
    arr = arr[:, :5]

    # ED24 timestamps are absolute Unix microseconds (~1.7e15), which forces
    # int64 and would make the .npy cache ~40 bytes/event.  Nothing downstream
    # uses absolute time, so rebase to the file start and store int32.
    arr[:, T_COL] -= arr[:, T_COL].min()
    if arr[:, T_COL].max() > np.iinfo(np.int32).max:
        raise ValueError(f"{path}: recording longer than 35 min; widen the cache dtype")

    # Polarity is {0,1} in ED24 and may be {-1,1} elsewhere.  Store canonically
    # as {0,1} so one model handles both without a dead embedding lookup.
    arr[:, P_COL] = (arr[:, P_COL] > 0).astype(np.int64)

    # Events must be time-ordered: the banded k-NN in the model assumes a
    # spatiotemporal neighbour is nearby in index.  ED24 is sorted; check
    # rather than hope.
    if not np.all(np.diff(arr[:, T_COL]) >= 0):
        arr = arr[np.argsort(arr[:, T_COL], kind="stable")]
    return arr.astype(np.int32)


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def _cache_path(src: str, cache_dir: str) -> str:
    h = hashlib.md5(os.path.abspath(src).encode()).hexdigest()[:16]
    return os.path.join(cache_dir, f"{os.path.basename(src)}.{h}.npy")


def load_events_cached(path: str, cache_dir: Optional[str] = None,
                       skip_first_row: bool = True) -> np.ndarray:
    """Parse once, memmap thereafter.  ~15 GB of ED24 text is hours to reparse."""
    if cache_dir is None:
        return _read_events_txt(path, skip_first_row)
    os.makedirs(cache_dir, exist_ok=True)
    cp = _cache_path(path, cache_dir)
    if os.path.exists(cp):
        try:
            return np.load(cp, mmap_mode="r")
        except Exception:
            pass
    arr = _read_events_txt(path, skip_first_row)
    tmp = cp + f".{os.getpid()}.tmp.npy"
    np.save(tmp, arr)
    os.replace(tmp, cp)
    return np.load(cp, mmap_mode="r")


# ==============================================================================
# 2. PER-WINDOW NORMALIZATION
# ==============================================================================
def normalize_window(ev: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(N, >=4) raw [t_us, x, y, p] -> ((N,4) float32 normalized, (2,) meta).

    feat : [x_n, y_n, t_n, p]
             x_n, y_n in [-1,1] from FIXED sensor dimensions, so a pixel
                 distance means the same thing in every window;
             t_n in [-1,1] from THIS WINDOW's own [t_min, t_max];
             p in {0,1}.
    meta : [duration_us, n_events] -- the scale that t_n normalized away.
    """
    t = ev[:, T_COL].astype(np.float64)
    t0, t1 = float(t.min()), float(t.max())
    dur = max(t1 - t0, 1.0)

    out = np.empty((ev.shape[0], 4), dtype=np.float32)
    out[:, 0] = (ev[:, X_COL].astype(np.float32) / (SENSOR_W - 1)) * 2.0 - 1.0
    out[:, 1] = (ev[:, Y_COL].astype(np.float32) / (SENSOR_H - 1)) * 2.0 - 1.0
    out[:, 2] = ((t - t0) / dur).astype(np.float32) * 2.0 - 1.0
    out[:, 3] = (ev[:, P_COL] > 0).astype(np.float32)

    meta = np.array([dur, ev.shape[0]], dtype=np.float32)
    return out, meta


# ==============================================================================
# 3. FILE LISTING AND SPLITS
# ==============================================================================
_VOLT_RE = re.compile(r"_(\d+\.\d+)\.csv$", re.IGNORECASE)


def list_ed24_files(root: str, voltages: Sequence[str] = ED24_VOLTAGES
                    ) -> List[Tuple[str, str, str]]:
    """-> [(scene, voltage, filepath), ...] for the 21 annotated levels.

    Returns the voltage too, so training can report validation AUC PER NOISE
    LEVEL.  A single pooled AUC over all 21 levels is close to meaningless
    here: the noise fraction sweeps from ~0.5% to most of the stream, and
    pooling windows with wildly different base rates lets between-level score
    offsets inflate the number (Simpson's paradox on the ROC).  A model can
    post a good pooled AUC while being at chance within every single level.
    """
    out = []
    if not os.path.isdir(root):
        raise SystemExit(f"ED24 root not found: {root}")
    want = set(voltages)
    for scene in sorted(os.listdir(root)):
        sdir = os.path.join(root, scene)
        if not os.path.isdir(sdir):
            continue
        for fn in sorted(os.listdir(sdir)):
            m = _VOLT_RE.search(fn)
            if m and m.group(1) in want:
                out.append((scene, m.group(1), os.path.join(sdir, fn)))
    return out


def scene_split(triples: List[Tuple[str, str, str]], val_frac: float = 0.15,
                seed: int = 0):
    """Split by SCENE, never by window.

    Consecutive windows of the same recording at the same voltage are
    near-duplicates, so a window-level split leaks and reports a validation
    number that looks great and predicts nothing about AUC_test.
    """
    scenes = sorted({s for s, _, _ in triples})
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(scenes))
    n_val = max(1, int(round(len(scenes) * val_frac)))
    val_scenes = {scenes[i] for i in perm[:n_val]}
    train = [(v, f) for s, v, f in triples if s not in val_scenes]
    val = [(v, f) for s, v, f in triples if s in val_scenes]
    return train, val


# ==============================================================================
# 3b. SHOT-NOISE AUGMENTATION -- covers the ED24 -> AUC_test noise-type gap
# ==============================================================================
def inject_shot_noise(ev: np.ndarray, rate_hz_px: float, N: int,
                      rng: np.random.Generator, fpn_decades: float = 0.5
                      ) -> np.ndarray:
    """Build an N-event window at a target v2e-style shot-noise rate.

    WHY THIS EXISTS
    ---------------
    ED24's noise is *measured* DAVIS346 background activity: hot pixels, and a
    frequency spectrum the paper describes as spanning four orders of
    magnitude.  AUC_test's noise is v2e-simulated shot noise at a uniform
    per-pixel rate with 0.5 decades of fixed-pattern noise.  Those are
    different distributions, and the fraction of test events drawn from the
    one you never trained on rises with the sweep: 11% at 1Hz, 56% at 10Hz.
    That is what the residual AUC slope tracks -- the learned margin over the
    unlearned BAF baseline halves from +0.136 to +0.072 across the sweep while
    the baseline itself stays flat, which is the signature of a coverage hole
    rather than underfitting.

    So we do to ED24 exactly what the DND21 protocol did to hotel-bar and
    driving: take real events, add synthetic shot noise at a chosen per-pixel
    rate, merge by timestamp.  The injected events are labelled 1 (noise),
    which is correct and consistent with the rest of the pipeline.

    HOW THE COUNT IS CHOSEN
    -----------------------
    The window must still hold exactly N events or the protocol changes.  So
    we take only M real events and let synthetic noise supply the remaining
    N - M, with M solved so the realized noise rate lands on the target:

        M = N / (1 + rate_hz_px * W * H / real_event_rate)

    Because M real events span less wall-clock time than N of them, the window
    also gets SHORTER as the injected rate rises -- which is what happens on
    AUC_test too, and it exercises the short-duration regime where several
    density scales saturate.
    """
    W, H = SENSOR_W, SENSOR_H
    n_avail = ev.shape[0]
    t_all = ev[:, T_COL].astype(np.float64)

    # Solve M + rate*W*H*T(M)/1e6 = N for the number of real events to keep.
    # A closed form needs T(M) = M / mean_rate, but the event rate inside a
    # window is far from constant -- assuming it costs ~30% on the realized
    # noise rate -- so binary-search the true M instead.  The left side is
    # increasing in M, so this is exact in ~12 lookups.
    per_us = rate_hz_px * W * H / 1e6
    lo, hi = 1, min(N, n_avail)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        total = mid + per_us * (t_all[mid - 1] - t_all[0])
        if total <= N:
            lo = mid
        else:
            hi = mid - 1
    M = int(np.clip(lo, N // 16, min(N, n_avail)))
    n_noise = N - M
    if n_noise <= 0:
        return ev[:N]

    real = ev[:M]
    t0 = float(real[0, T_COL]); t1 = float(real[-1, T_COL])
    if t1 <= t0:
        return ev[:N]

    # Fixed-pattern noise: per-pixel rate multipliers, log-normal with the
    # 0.5-decade spread the AUC_test noise was generated with.  Without it the
    # synthetic noise is *too* uniform and is trivially separable, which would
    # teach the model the wrong invariance.
    if fpn_decades > 0:
        w = 10.0 ** rng.normal(0.0, fpn_decades, size=W * H)
        w /= w.sum()
        pix = rng.choice(W * H, size=n_noise, p=w)
    else:
        pix = rng.integers(0, W * H, size=n_noise)

    noise = np.empty((n_noise, 5), dtype=ev.dtype)
    noise[:, T_COL] = rng.uniform(t0, t1, size=n_noise).astype(ev.dtype)
    noise[:, X_COL] = (pix % W).astype(ev.dtype)
    noise[:, Y_COL] = (pix // W).astype(ev.dtype)
    noise[:, P_COL] = rng.integers(0, 2, size=n_noise).astype(ev.dtype)
    noise[:, LABEL_COL] = 1

    out = np.concatenate([real, noise], axis=0)
    out = out[np.argsort(out[:, T_COL], kind="stable")]
    return out[:N]

def augment_window(ev: np.ndarray, rng) -> np.ndarray:
    """Label-preserving geometric / polarity augmentation.

    None of EDformer, EDmamba or WedNet augments at all -- EDformer says so in
    the paper and its loader has no randomness in __getitem__.  ED24 is 100
    scenes from one sensor and the train/val gap here widened from +0.025 to
    +0.088 in six epochs, so the corpus is being memorised.  All three
    operations are exact symmetries of the descriptor: the density channels are
    box counts (reflection-invariant), the velocity bank is symmetric about
    zero, and the polarity channels are defined by agreement, not by sign.
    """
    ev = ev.copy()
    if rng.random() < 0.5:
        ev[:, X_COL] = (SENSOR_W - 1) - ev[:, X_COL]
    if rng.random() < 0.5:
        ev[:, Y_COL] = (SENSOR_H - 1) - ev[:, Y_COL]
    if rng.random() < 0.5:
        ev[:, P_COL] = 1 - ev[:, P_COL]
    return ev

def quantize_timestamps(ev: np.ndarray, q_us: float,
                        min_levels: int = 64) -> np.ndarray:
    """Round timestamps onto a q_us readout grid, as a coarser sensor would.

    Floor is monotone non-decreasing, so a t-sorted array stays sorted and no
    re-sort is needed.  The grid is capped so a window keeps at least
    `min_levels` distinct times: at 60 Hz/pixel a 4096-event window can be
    under a millisecond, and an uncapped grid would collapse it to a constant,
    after which normalize_window's t_n is degenerate.
    """
    t = ev[:, T_COL]
    dur = float(t[-1] - t[0])
    if dur <= 0.0:
        return ev
    q = int(max(1.0, min(float(q_us), dur / min_levels)))
    if q <= 1:
        return ev
    ev = ev.copy()
    ev[:, T_COL] = (t // q) * q
    return ev

class ED24Windows(Dataset):
    """Non-overlapping N-event windows, indexed lazily, sampled across the file.

    Two changes from the previous version:

      * windows are drawn EVENLY ACROSS the recording, not taken as the first
        `max_windows_per_file` of it.  Capping at the head meant every capped
        file contributed only its opening ~0.5 s, which for a driving or
        walking scene is a single motion state.
      * each item carries its voltage id, so validation can be split by level.
    """

    def __init__(self, items: List[Tuple[str, str]], N: int = 4096,
                 cache_dir: Optional[str] = None,
                 max_windows_per_file: Optional[int] = None,
                 seed: int = 0,
                 shot_noise_prob: float = 0.0,
                 shot_noise_range: Tuple[float, float] = (0.5, 20.0),
                 shot_noise_fixed: Optional[float] = None,
                 augment: bool = False,
                 ts_quant_prob: float = 0.0,
                 ts_quant_range: Tuple[float, float] = (2.0, 200.0)):
        self.files = [f for _, f in items]
        self.volts = [v for v, _ in items]
        # stable integer id per voltage level, so a collated batch can carry it
        self.voltage_levels = sorted({v for v, _ in items}, key=float)
        self._vid = {v: i for i, v in enumerate(self.voltage_levels)}
        self.N = N
        self.cache_dir = cache_dir
        self.index: List[Tuple[int, int]] = []
        self._open: Dict[int, np.ndarray] = {}
        # shot-noise augmentation (train) / fixed-rate probe (val)
        self.shot_noise_prob = shot_noise_prob
        self.shot_noise_range = shot_noise_range
        self.shot_noise_fixed = shot_noise_fixed
        self.augment = augment
        self.ts_quant_prob = ts_quant_prob
        self.ts_quant_range = ts_quant_range
        self._rng: Optional[np.random.Generator] = None

        rng = np.random.default_rng(seed)
        for fi, path in enumerate(self.files):
            arr = load_events_cached(path, cache_dir)
            n_win = arr.shape[0] // N
            if n_win == 0:
                continue
            if max_windows_per_file is not None and n_win > max_windows_per_file:
                # evenly spaced, jittered, deterministic
                base = np.linspace(0, n_win - 1, max_windows_per_file)
                jit = rng.integers(0, max(1, n_win // max_windows_per_file),
                                   size=max_windows_per_file)
                sel = np.unique(np.clip(base.astype(int) + jit, 0, n_win - 1))
            else:
                sel = np.arange(n_win)
            self.index.extend((fi, int(w) * N) for w in sel)

    def __len__(self):
        return len(self.index)

    def voltage_of(self, i: int) -> str:
        return self.volts[self.index[i][0]]

    _MAX_OPEN = 48   # LRU cap; memmaps are cheap but not free, and an
                     # uncapped dict in a persistent worker grows all run

    def _get_array(self, fi: int) -> np.ndarray:
        a = self._open.get(fi)
        if a is None:
            if len(self._open) >= self._MAX_OPEN:
                self._open.pop(next(iter(self._open)))  # dicts keep insertion order
            a = load_events_cached(self.files[fi], self.cache_dir)
            self._open[fi] = a
        return a

    def __getitem__(self, i: int):
        fi, start = self.index[i]
        arr = self._get_array(fi)

        if self.shot_noise_fixed is not None:
            # deterministic per index, so the validation probe is identical
            # between epochs and between runs
            rng = np.random.default_rng(1000003 + i)
            ev = inject_shot_noise(np.asarray(arr[start:start + self.N]),
                                   self.shot_noise_fixed, self.N, rng)
        elif self.shot_noise_prob > 0.0:
            if self._rng is None:
                # torch reseeds each worker every epoch, so this varies
                self._rng = np.random.default_rng(torch.initial_seed() % (2 ** 32))
            rng = self._rng
            if rng.random() < self.shot_noise_prob:
                lo, hi = self.shot_noise_range
                lam = float(10.0 ** rng.uniform(np.log10(lo), np.log10(hi)))
                ev = inject_shot_noise(np.asarray(arr[start:start + self.N]),
                                       lam, self.N, rng)
            else:
                ev = np.asarray(arr[start:start + self.N])
        else:
            ev = np.asarray(arr[start:start + self.N])

        # After injection, so real and synthetic events share one grid -- a
        # real sensor quantises everything it reads out, not just the signal.
        # Skipped for the fixed-rate val probe so the per-epoch number stays
        # comparable with previous runs.
        # if self.ts_quant_prob > 0.0 and self.shot_noise_fixed is None:
        #     if self._rng is None:
        #         self._rng = np.random.default_rng(torch.initial_seed() % (2 ** 32))
        #     if self._rng.random() < self.ts_quant_prob:
        #         qlo, qhi = self.ts_quant_range
        #         q = 10.0 ** self._rng.uniform(np.log10(qlo), np.log10(qhi))
        #         ev = quantize_timestamps(ev, q)

        if self.augment and self.shot_noise_fixed is None:
            if self._rng is None:
                self._rng = np.random.default_rng(torch.initial_seed() % (2 ** 32))
            ev = augment_window(ev, self._rng)

        feat, meta = normalize_window(ev)
        label = ev[:, LABEL_COL].astype(np.float32)  # 1 == NOISE
        vid = self._vid[self.volts[fi]]
        return (torch.from_numpy(feat), torch.from_numpy(label),
                torch.from_numpy(meta), vid)


class EventFileWindows(Dataset):
    """Same windowing for a single AUC_test file (no scene folders)."""

    def __init__(self, path: str, N: int = 4096, cache_dir: Optional[str] = None,
                 has_label: bool = True):
        self.arr = load_events_cached(path, cache_dir)
        self.N = N
        self.has_label = has_label
        self.n_win = self.arr.shape[0] // N

    def __len__(self):
        return self.n_win

    def __getitem__(self, i: int):
        ev = np.asarray(self.arr[i * self.N:(i + 1) * self.N])
        feat, meta = normalize_window(ev)
        label = (ev[:, LABEL_COL].astype(np.float32) if self.has_label
                 else np.zeros(ev.shape[0], np.float32))
        return (torch.from_numpy(feat), torch.from_numpy(label),
                torch.from_numpy(meta))


# ==============================================================================
# 4. CONVENTION CHECK -- spatiotemporal, not spatial
# ==============================================================================
def _neighbour_counts(ev: np.ndarray, r_px: int = 3, tau_us: int = 5000
                      ) -> np.ndarray:
    """Per-event count of other events within +-r_px and +-tau_us.

    Exact, via a time-ordered two-pointer sweep -- O(N * events-in-tau) rather
    than the O(N^2) pairwise matrix, so this stays usable at N=4096 on CPU.
    """
    t = ev[:, T_COL].astype(np.int64)
    x = ev[:, X_COL].astype(np.int64)
    y = ev[:, Y_COL].astype(np.int64)
    n = len(t)
    cnt = np.zeros(n, dtype=np.int64)
    lo = 0
    for i in range(n):
        while t[i] - t[lo] > tau_us:
            lo += 1
        hi = np.searchsorted(t, t[i] + tau_us, side="right")
        sl = slice(lo, hi)
        m = (np.abs(x[sl] - x[i]) <= r_px) & (np.abs(y[sl] - y[i]) <= r_px)
        cnt[i] = m.sum() - 1
    return cnt


def verify_label_convention(paths, N: int = 4096, n_windows: int = 8,
                            cache_dir: Optional[str] = None,
                            r_px: int = 3, tau_us: int = 5000) -> dict:
    """Decide which label is noise from SPATIOTEMPORAL correlation.

    Real events come from a moving edge, so they arrive in dense
    spatiotemporal bursts; BA noise -- shot, dark-current, leakage and hot
    pixel -- is uncorrelated with its spatial neighbourhood.  The class with
    the LOWER mean neighbour count is noise.

    This is a mean of a per-event statistic, so it is unbiased with respect to
    class size and stays correct at the 1.5V level where one class is ~0.5% of
    the stream.  It is also immune to hot pixels, which defeated the old
    spatial-clustering test: a hot pixel fires at a single pixel at ~100 Hz, so
    consecutive events from it are ~10 ms apart and it has no spatial
    neighbours at all.

    Pass several files (ideally spanning low and high voltage); the vote is
    taken over files, and the per-file evidence is returned.
    """
    if isinstance(paths, str):
        paths = [paths]

    per_file, votes = [], []
    for path in paths:
        arr = load_events_cached(path, cache_dir)
        n_win = min(n_windows, arr.shape[0] // N)
        s0 = s1 = 0.0
        n0 = n1 = 0
        for w in range(n_win):
            ev = np.asarray(arr[w * N:(w + 1) * N])
            lab = ev[:, LABEL_COL].astype(np.int64)
            cnt = _neighbour_counts(ev, r_px, tau_us)
            s0 += cnt[lab == 0].sum(); n0 += int((lab == 0).sum())
            s1 += cnt[lab == 1].sum(); n1 += int((lab == 1).sum())
        if n0 == 0 or n1 == 0:
            per_file.append({"file": os.path.basename(path),
                             "frac_label_1": n1 / max(n0 + n1, 1),
                             "note": "single-class file, no vote"})
            continue
        m0, m1 = s0 / n0, s1 / n1
        vote = 1 if m1 < m0 else 0
        votes.append(vote)
        per_file.append({
            "file": os.path.basename(path),
            "frac_label_1": n1 / (n0 + n1),
            "mean_neighbours_label_0": m0,
            "mean_neighbours_label_1": m1,
            "separation_ratio": max(m0, m1) / max(min(m0, m1), 1e-6),
            "vote_noise_label": vote,
        })

    if not votes:
        raise SystemExit("No usable file for the convention check "
                         "(every sampled file was single-class).")

    inferred = int(round(float(np.mean(votes))))
    res = {
        "test": f"spatiotemporal correlation (+-{r_px}px, +-{tau_us/1000:.0f}ms); "
                f"noise is the class with FEWER neighbours",
        "per_file": per_file,
        "files_voting_noise_is_1": int(sum(votes)),
        "files_voting_noise_is_0": int(len(votes) - sum(votes)),
        "inferred_noise_label": inferred,
        "config_says_noise_label": 1 if POSITIVE_CLASS_IS_NOISE else 0,
        "unanimous": len(set(votes)) == 1,
    }
    res["agrees"] = res["inferred_noise_label"] == res["config_says_noise_label"]
    return res


def noise_fraction_report(triples, cache_dir: Optional[str] = None,
                          per_level: int = 2, N: int = 4096) -> Dict[str, float]:
    """Mean fraction of label==1 per voltage level.

    A third, independent convention check, and a useful sanity table in its own
    right: the ED24 voltage is attenuator voltage, so higher voltage means
    lower illuminance and MORE BA noise.  The noise class's fraction must
    therefore INCREASE monotonically from 1.5V to 3.5V.  If the fraction of
    label==1 decreases with voltage, label 1 is the signal class and the
    convention is flipped, no matter what any density test says.
    """
    by_level: Dict[str, List[str]] = {}
    for s, v, f in triples:
        by_level.setdefault(v, []).append(f)
    out = {}
    for v in sorted(by_level, key=float):
        fr = []
        for f in by_level[v][:per_level]:
            arr = load_events_cached(f, cache_dir)
            nw = min(4, arr.shape[0] // N)
            for w in range(nw):
                fr.append(float((np.asarray(arr[w * N:(w + 1) * N, LABEL_COL]) == 1).mean()))
        if fr:
            out[v] = float(np.mean(fr))
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="ED24 loader checks")
    ap.add_argument("path", nargs="+", help="one or more ED24 csv / AUC_test txt")
    ap.add_argument("--N", type=int, default=4096)
    ap.add_argument("--cache_dir", type=str, default=None)
    args = ap.parse_args()

    info = verify_label_convention(args.path, args.N, cache_dir=args.cache_dir)
    print(info["test"])
    for d in info["per_file"]:
        if "note" in d:
            print(f"  {d['file']:28s} frac1={d['frac_label_1']:.5f}  {d['note']}")
        else:
            print(f"  {d['file']:28s} frac1={d['frac_label_1']:.5f}  "
                  f"nbrs(label0)={d['mean_neighbours_label_0']:6.2f}  "
                  f"nbrs(label1)={d['mean_neighbours_label_1']:6.2f}  "
                  f"ratio={d['separation_ratio']:5.2f}x  -> noise is label "
                  f"{d['vote_noise_label']}")
    print(f"\n  inferred_noise_label {info['inferred_noise_label']}  "
          f"(config says {info['config_says_noise_label']})  "
          f"unanimous={info['unanimous']}  agrees={info['agrees']}")
    if not info["agrees"]:
        print("\n*** flip POSITIVE_CLASS_IS_NOISE in ed24_data.py ***")

    ds = EventFileWindows(args.path[0], args.N, cache_dir=args.cache_dir)
    f, l, m = ds[0]
    print(f"\nwindow feat {tuple(f.shape)}  label {tuple(l.shape)}  meta {m.tolist()}")
    print(f"x range [{f[:,0].min():.3f}, {f[:,0].max():.3f}]")
    print(f"y range [{f[:,1].min():.3f}, {f[:,1].max():.3f}]")
    print(f"t range [{f[:,2].min():.3f}, {f[:,2].max():.3f}]   <- must be ~[-1, 1]")
    print(f"window duration {m[0]/1000:.2f} ms, rate {m[1]/(m[0]/1e6):.0f} ev/s, "
          f"noise fraction {l.mean():.4f}")
