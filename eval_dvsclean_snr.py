from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

import ed24_data
from ed24_data import normalize_window, set_sensor_geometry
from edn_model import baf_baseline_score
from edn_ablate import (AblConfig as EDNConfig, EDNetAbl as EDNet,
                        config_from_checkpoint)


# ==============================================================================
# loading
# ==============================================================================
# Candidate key names seen in event HDF5 files.  DVSCLEAN ships one file per
# (scene, noise ratio) and different releases name the contents differently, so
# we probe rather than assume.  Run with --inspect to print the actual tree.
_H5_EVENT_KEYS = ("events", "event", "data", "raw", "ev", "stream")
_H5_LABEL_KEYS = ("labels", "label", "gt", "y", "target", "mask", "noise")
_H5_FIELD_KEYS = {"t": ("t", "ts", "time", "timestamp", "timestamps"),
                  "x": ("x", "xs"),
                  "y": ("y", "ys"),
                  "p": ("p", "ps", "pol", "polarity", "polarities"),
                  "l": ("l", "label", "labels", "gt", "y_label", "noise")}


def h5_tree(path: str) -> str:
    """Human-readable dump of an HDF5 file's datasets."""
    import h5py
    lines = []

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            lines.append(f"    /{name:30s} shape={str(obj.shape):18s} "
                         f"dtype={obj.dtype}")
    with h5py.File(path, "r") as f:
        f.visititems(visit)
        if f.attrs:
            lines.append(f"    attrs: {dict(f.attrs)}")
    return "\n".join(lines) or "    (no datasets found)"


def _first_match(keys, candidates):
    """Find a dataset whose LEAF name matches one of `candidates`.

    HDF5 paths are often nested -- DVSCLEAN stores events/timestamp, events/x,
    events/y, events/polarity, events/label -- so matching the full path
    against a bare name like "timestamp" never fires.  We index by leaf name
    and return the full path, preferring the shallowest match when the same
    leaf appears in more than one group.
    """
    by_leaf = {}
    for k in sorted(keys, key=lambda q: (q.count("/"), len(q))):
        by_leaf.setdefault(k.rsplit("/", 1)[-1].lower(), k)
    for c in candidates:
        if c in by_leaf:
            return by_leaf[c]
    return None


def load_any(path: str, cols: str, noise_label: int,
             events_key=None, label_key=None, time_unit: str = "auto",
             polarity: str = "auto", verbose: bool = False) -> np.ndarray:
    """Read one DVSCLEAN file. Delegates to event_io -- see the note below.

    WHY THE LOCAL LOADER WAS REMOVED
    --------------------------------
    It ended with

        out = out[~np.isnan(out).any(axis=1)].astype(np.int64)

    DVSCLEAN stores events/timestamp as float32 SECONDS (0.0533 .. 0.6028 on
    MAH00455_50), so that cast truncated every timestamp to 0. Nothing raised.
    x, y and the labels survived, so the model still produced a plausible AUC
    -- computed with a dead time axis. Every DVSCLEAN number produced before
    this change is affected.

    The same line also did `(p > 0)` on events/polarity, which is not a
    polarity column at all: it holds a per-event uniform random draw in (0,1)
    (noise-event mean 0.4981, median 0.4984) that correlates weakly with the
    label, AUC 0.564 on its own. Every value being positive, it collapsed to a
    constant 1 -- so the leak never reached the model, but neither did any
    polarity information. event_io now drops that column explicitly and logs
    that it did.
    """
    import event_io
    _require_event_io(event_io)
    return event_io.load_events(path, noise_label=noise_label,
                                time_unit=time_unit, polarity=polarity,
                                verbose=verbose)


_EVENT_IO_MIN = 4


def _require_event_io(mod) -> None:
    """Fail immediately, and legibly, on a stale event_io.py.

    Without this a mismatched copy on the path surfaces as
    `TypeError: load_events() got an unexpected keyword argument 'polarity'`
    from inside the first file of a long run, which says nothing useful about
    what to do next.
    """
    v = getattr(mod, "__version__", 0)
    if v < _EVENT_IO_MIN:
        raise SystemExit(
            f"\nSTALE event_io.py: found version {v}, need >= {_EVENT_IO_MIN}.\n"
            f"  located at: {getattr(mod, '__file__', '?')}\n"
            f"Copy the latest event_io.py next to this script, then verify with:\n"
            f"  python event_io.py --self_check\n")


SCENE_FILTER = None


def find_files(root: str, level: str) -> list:
    """Files for one noise ratio.

    DVSCLEAN names them `<scene>_<ratio>.hdf5`, e.g. MAH00444_50.hdf5 and
    MAH00444_100.hdf5 side by side in one directory.  Substring matching is
    unsafe here -- "50" occurs inside scene ids like MAH00450 -- so we parse
    the trailing `_<digits>` off the stem and compare numerically.
    """
    cand = []
    for ext in ("h5", "hdf5", "hdf", "npy", "npz", "txt", "csv", "dat"):
        cand += glob.glob(os.path.join(root, "**", f"*.{ext}"), recursive=True)

    want = int(level)
    hits = []
    for p in cand:
        if SCENE_FILTER is not None:
            stem = os.path.splitext(os.path.basename(p))[0]
            if re.sub(r"_\d+$", "", stem) not in SCENE_FILTER:
                continue
        stem = os.path.splitext(os.path.basename(p))[0]
        m = re.search(r"_(\d+)$", stem)
        if m and int(m.group(1)) == want:
            hits.append(p); continue
        # fall back to directory-name conventions (.../50%noise/, .../noise_50/)
        parts = p.replace("/", os.sep).split(os.sep)[:-1]
        if any(re.fullmatch(rf"(noise[_-]?)?{want}%?(_?noise)?", q, re.I)
               for q in parts):
            hits.append(p)
    return sorted(set(hits))


# ==============================================================================
# metrics
# ==============================================================================
def snr_db(n_real_kept: float, n_noise_kept: float, k: float) -> float:
    if n_noise_kept <= 0:
        return float("inf")
    if n_real_kept <= 0:
        return float("-inf")
    return k * np.log10(n_real_kept / n_noise_kept)


def confusion(y_noise: np.ndarray, pred_noise: np.ndarray) -> dict:
    """Positives are REAL events retained, following DVSCLEAN's definitions:
    TP real kept, FN real removed, TN noise removed, FP noise kept."""
    real = y_noise == 0
    kept = ~pred_noise
    tp = int((real & kept).sum())
    fn = int((real & ~kept).sum())
    tn = int((~real & ~kept).sum())
    fp = int((~real & kept).sum())
    rer = tp / max(tp + fn, 1)
    nsr = tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    f1 = 2 * prec * rer / max(prec + rer, 1e-12)
    return dict(TP=tp, FN=fn, TN=tn, FP=fp, RER=rer, NSR=nsr,
                precision=prec, F1=f1)


# ==============================================================================
# scoring
# ==============================================================================
class _Windows(torch.utils.data.Dataset):
    def __init__(self, ev, N):
        self.ev, self.N, self.n = ev, N, ev.shape[0] // N

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        w = self.ev[i * self.N:(i + 1) * self.N]
        f, m = normalize_window(w)
        return torch.from_numpy(f), torch.from_numpy(m), i


@torch.no_grad()
def score_file(model, ev, N, device, batch_size, amp):
    ds = _Windows(ev, N)
    if len(ds) == 0:
        return None
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    probs, bases = [], []
    for feat, meta, _ in loader:
        feat = feat.to(device); meta = meta.to(device)
        with torch.autocast("cuda", enabled=amp):
            logits = model(feat, meta)
        probs.append(torch.sigmoid(logits.float()).cpu().numpy().ravel())
        bases.append(baf_baseline_score(feat, meta).float().cpu().numpy().ravel())
    n = len(ds) * N
    return np.concatenate(probs), np.concatenate(bases), ev[:n, 4]


def evaluate_level(model, files, level, args, device, amp):
    """-> (per-file results, pooled probs, pooled baseline, pooled labels).

    Per-file results are kept because the AUC comparison in the literature is
    reported PER SCENE, not pooled: pooling scenes with different noise
    fractions lets between-scene score offsets carry the ranking, which can
    move the number either way relative to a per-scene mean.
    """
    per_file, P, B, Y = [], [], [], []
    for path in tqdm(files, desc=f"  {level}% noise", leave=False):
        ev = load_any(path, args.cols, args.noise_label,
                      args.h5_events_key, args.h5_label_key,
                      time_unit=args.time_unit, polarity=args.polarity,
                      verbose=args.verbose_load)
        if args.force_polarity is not None:
            ev = np.array(ev, copy=True)
            if args.force_polarity == 2:
                # Seeded on the file name so the assignment is identical on
                # every rerun; a reviewer re-running this must get our numbers.
                # rng = np.random.default_rng(
                #     abs(hash(os.path.basename(path))) % (2**32))
                # in evaluate_level, replace the seeding line
                import zlib
                rng = np.random.default_rng(zlib.crc32(os.path.basename(path).encode()))
                ev[:, 3] = rng.integers(0, 2, size=len(ev))
            else:
                ev[:, 3] = args.force_polarity
        out = score_file(model, ev, args.N, device, args.batch_size, amp)
        if out is None:
            continue
        p, b, y = out
        scene = re.sub(r"_\d+$", "", os.path.splitext(os.path.basename(path))[0])
        if 0 < y.sum() < len(y):
            per_file.append(dict(scene=scene,
                                 auc=roc_auc_score(y, p),
                                 auc_base=roc_auc_score(y, b),
                                 noise=float(y.mean()), n=len(y)))
        P.append(p); B.append(b); Y.append(y)
    if not P:
        return None
    return per_file, np.concatenate(P), np.concatenate(B), np.concatenate(Y)


def report_auc(per_file, probs, base, y, top):
    """Per-scene AUC, the way the comparison tables report it."""
    print(f"\n  -- AUC (noise = positive class) --")
    if not per_file:
        print("    no scene had both classes present")
        return
    per_file = sorted(per_file, key=lambda r: r["auc"])
    show = per_file if len(per_file) <= top else per_file[:top // 2] + per_file[-(top - top // 2):]
    print(f"    {'scene':28s} {'AUC':>8} {'BAF':>8} {'noise':>7} {'events':>10}")
    prev_gap = False
    for r in show:
        if len(per_file) > top and r is show[top // 2] and not prev_gap:
            print(f"    {'... ' + str(len(per_file) - top) + ' scenes omitted':28s}")
            prev_gap = True
        print(f"    {r['scene'][:28]:28s} {r['auc']:8.4f} {r['auc_base']:8.4f} "
              f"{r['noise']:7.3f} {r['n']:10d}")
    a = np.array([r["auc"] for r in per_file])
    b = np.array([r["auc_base"] for r in per_file])
    print(f"    {'MEAN over ' + str(len(per_file)) + ' scenes':28s} "
          f"{a.mean():8.4f} {b.mean():8.4f}")
    print(f"    {'  std / min / max':28s} {a.std():8.4f} {a.min():8.4f} {a.max():8.4f}")
    print(f"    {'POOLED (all events)':28s} {roc_auc_score(y, probs):8.4f} "
          f"{roc_auc_score(y, base):8.4f}")


def report(name, scores, y, thresholds, ks, invert=False):
    """Print the SNR / retention table for one scorer across thresholds."""
    s = -scores if invert else scores
    rows = []
    for thr in thresholds:
        pred_noise = s >= thr
        c = confusion(y, pred_noise)
        row = dict(thr=thr, **c)
        for k in ks:
            row[f"SNR{int(k)}"] = snr_db(c["TP"], c["FP"], k)
        rows.append(row)
    hdr = (f"  {'thr':>7} {'RER':>7} {'NSR':>7} {'prec':>7} {'F1':>7} "
           + " ".join(f"{'SNR(k='+str(int(k))+')':>13}" for k in ks))
    print(f"\n  -- {name} --")
    print(hdr)
    for r in rows:
        print(f"  {r['thr']:7.3f} {r['RER']:7.4f} {r['NSR']:7.4f} "
              f"{r['precision']:7.4f} {r['F1']:7.4f} "
              + " ".join(f"{r['SNR'+str(int(k))]:13.2f}" for k in ks))
    return rows


def main():
    ap = argparse.ArgumentParser(description="DVSCLEAN evaluation for EDNet")
    ap.add_argument("--ckpt", default=None,
                    help="required unless --inspect")
    ap.add_argument("--root", required=True,
                    help="DVSCLEAN simulated partition root")
    ap.add_argument("--sensor", nargs=2, type=int, default=[1280, 720],
                    metavar=("W", "H"))
    ap.add_argument("--cols", type=str, default="txypl",
                    help="column order in the files, a permutation of t,x,y,p,l")
    ap.add_argument("--time_unit", default="auto",
                    choices=["auto", "us", "ms", "s"],
                    help="unit of the raw timestamp column")
    ap.add_argument("--polarity", default="auto",
                    choices=["auto", "binary", "threshold", "raw", "constant"],
                    help="auto uses the column only if it is two-state")
    ap.add_argument("--force_polarity", type=int, default=None, choices=[0, 1, 2],
                help="overwrite the polarity column after loading: 0 or 1 for a "
                     "constant, 2 for a balanced random assignment seeded per "
                     "file. Diagnostic for the DVSCLEAN polarity field, which "
                     "is not a polarity.")
    ap.add_argument("--verbose_load", action="store_true", default=True)
    ap.add_argument("--noise_label", type=int, default=1,
                    help="value in the label column that means NOISE")
    ap.add_argument("--N", type=int, default=None,
                    help="window size in events. Default: whatever the checkpoint "
                         "was TRAINED with, since a train/test window mismatch "
                         "silently changes what the density ladder measures.")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--thresholds", nargs="+", type=float,
                    default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--inspect", action="store_true",
                    help="print the HDF5 structure of the first file per level "
                         "and exit, without loading a checkpoint")
    ap.add_argument("--h5_events_key", type=str, default=None)
    ap.add_argument("--h5_label_key", type=str, default=None)
    ap.add_argument("--aednet_k", type=float, default=20.0,
                    help="coefficient for the iso-RER table: 20 for the "
                         "AEDNet/SAGE lineage (derived, see snr_aednet.py), "
                         "10 for the EDformer/EDmamba lineage")
    ap.add_argument("--snr_threshold", type=float, default=0.5,
                    help="fixed operating point for the AEDNet-style SNR table")
    ap.add_argument("--auc_only", action="store_true",
                    help="report only AUC, skipping the SNR/threshold sweep")
    ap.add_argument("--top_scenes", type=int, default=12,
                    help="how many per-scene rows to print (worst and best)")
    ap.add_argument("--scenes", nargs="+", default=None,
                    help="restrict to these scene stems, e.g. MAH00455 MAH00456 "
                         "MAH00461 MAH00467 MAH00470 -- the five-scene test set "
                         "used by the published comparison table (taken from "
                         "EDmamba's denoise_dvsclean.py test split)")
    args = ap.parse_args()

    if args.inspect:
        for level in ("50", "100"):
            files = find_files(args.root, level)
            print(f"=== {level}% noise: {len(files)} files ===")
            if files:
                print(f"  {files[0]}")
                print(h5_tree(files[0]) if files[0].endswith((".h5", ".hdf5", ".hdf"))
                      else "    (not HDF5)")
        return

    import event_io
    _require_event_io(event_io)
    print(f"event_io v{event_io.__version__} from {event_io.__file__}")

    global SCENE_FILTER
    SCENE_FILTER = set(args.scenes) if args.scenes else None
    set_sensor_geometry(*args.sensor)
    print(f"sensor set to {args.sensor[0]}x{args.sensor[1]}")

    if args.ckpt is None:
        raise SystemExit("--ckpt is required (or use --inspect)")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    amp = (not args.no_amp) and device.type == "cuda"
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    if args.N is None:
        args.N = int(ck.get("args", {}).get("N", 4096))
        print(f"  N = {args.N} (from checkpoint)")
    cfg = config_from_checkpoint(ck, args.N)
    model = EDNet(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"loaded {args.ckpt} (epoch {ck.get('epoch','?')})")
    print("NOTE: trained on DAVIS346 data only -- these are zero-shot "
          "cross-sensor numbers.\n")

    ks = [10.0, 20.0]
    summary = {}
    for level in ("50", "100"):
        files = find_files(args.root, level)
        if not files:
            print(f"=== {level}% noise: no files found under {args.root} ===")
            print("    (check --root, or adjust the patterns in find_files)\n")
            continue
        print(f"=== {level}% noise ratio | {len(files)} files"
              + (f" | restricted to {len(SCENE_FILTER)} scenes" if SCENE_FILTER else "")
              + " ===")
        out = evaluate_level(model, files, level, args, device, amp)
        if out is None:
            print("    no window reached the required length\n")
            continue
        per_file, probs, base, y = out

        # Control first: the undenoised stream.  This is what tells you which
        # k the published table you are comparing against actually used.
        n_real, n_noise = int((y == 0).sum()), int((y == 1).sum())
        print(f"  raw stream: {n_real} real / {n_noise} noise "
              f"(ratio {n_real/max(n_noise,1):.3f})")
        print(f"  raw-data control SNR: "
              + "  ".join(f"k={int(k)}: {snr_db(n_real, n_noise, k):.2f} dB"
                          for k in ks))
        print("  published control for this level: "
              + ("3 dB" if level == "50" else "0 dB")
              + "  -> whichever k reproduces it is the convention in use")

        summary[level] = (probs, base, y)
        report_auc(per_file, probs, base, y, args.top_scenes)
        if not args.auc_only:
            report("EDNet", probs, y, args.thresholds, ks)
        if not args.auc_only:
            bt = np.quantile(base, [0.5, 0.7, 0.9])
            report("BAF baseline", base, y, list(bt), ks)
        print()

    if len(summary) == 2:
        snr_summary(summary, ks, args.snr_threshold, aednet_k=args.aednet_k)


def snr_summary(summary, ks, fixed_thr, aednet_k=20.0):
    """SNR laid out as the source tables lay it out: 50% / 100% / Average.

    THE CONVENTION PROBLEM, NOW RESOLVED PER LINEAGE
    ------------------------------------------------
    The metric itself is not in doubt.  With the signal class positive --
    TP real kept, FN real removed, FP noise kept, TN noise removed --

        SNR_before = k * log10( (TP + FN) / (TN + FP) )
        SNR_after  = k * log10(  TP / FP  )

    which is exactly what snr_db() computes.  Only k was open, and no paper
    states it.  It is recoverable, because SAGE publishes an SNR table and an
    RER/NSR table over the SAME 49 scenes, so one predicts the other:

        TP/FP = [ RER / (1 - NSR) ] * (R / Nz)

    Measured (see snr_aednet.reconcile, which prints the full residual table):

        50% noise, learned methods   k=10 off by 13.0 dB    k=20 off by 0.59 dB
        100% noise, learned methods  k=10 off by 14.8 dB    k=20 off by 6.00 dB

    At the 50% level k=20 reproduces AEDNet to 0.08 dB, TS to 0.11, SAGE to
    0.39.  So the AEDNet/SAGE lineage uses k = 20.

    The EDformer/EDmamba lineage uses k = 10: their control row of 3 dB / 0 dB
    is reproduced by the raw stream under k=10 (measured 2.97 / -0.04) and not
    under k=20 (5.93 / -0.09).  Two lineages, two conventions, a factor of two
    apart -- wider than the gaps either table reports as progress.  Both k are
    printed; quote the one matching whichever table you put your row in, and
    say so.

    NOT RECONCILED, AND SAID SO: the classical filters fit neither k, and at
    the 100% level every method needs a reference ratio near 1.45 where the
    files give 0.99.  Part of the mapping is not recoverable from what is
    published, so the 100% SNR column should be treated as indicative.

    THE OPERATING-POINT PROBLEM
    ---------------------------
    Unlike AUC, SNR depends on the decision threshold, and M/N rises
    monotonically as more events are deleted -- a method can raise SNR without
    denoising better simply by cutting harder, up to the point where nothing is
    left and the ratio is undefined.  None of the source tables state the
    threshold used.  Three operating points are therefore reported, and RER is
    printed beside each so the reader can see what was paid for the number.
    """
    rows = []

    def add(name, scorer, pick):
        cells, rer = {}, {}
        for level, (probs, base, y) in summary.items():
            s_ = probs if scorer == "model" else base
            if pick == "fixed":
                thr = fixed_thr
            else:
                grid = np.quantile(s_, np.linspace(0.02, 0.98, 49))
                best, thr = -1, grid[0]
                for t in grid:
                    c = confusion(y, s_ >= t)
                    if c["F1"] > best:
                        best, thr = c["F1"], t
            c = confusion(y, s_ >= thr)
            cells[level] = {int(k): snr_db(c["TP"], c["FP"], k) for k in ks}
            rer[level] = c["RER"]
        rows.append((name, cells, rer))

    raw, raw_ratio = {}, {}
    for level, (_, _, y) in summary.items():
        nr, nn = int((y == 0).sum()), int((y == 1).sum())
        raw[level] = {int(k): snr_db(nr, nn, k) for k in ks}
        raw_ratio[level] = nr / max(nn, 1)
    rows.append(("Raw data (control)", raw, {l: 1.0 for l in summary}))

    add("Density baseline (best F1)", "base", "f1")
    add(f"EDNet (thr={fixed_thr:g})", "model", "fixed")
    add("EDNet (best F1)", "model", "f1")

    # Which k the AEDNet/SAGE tables use, derived from their own two tables
    # rather than assumed, and our model at their operating points. Both are
    # in snr_aednet.py; a missing copy must not kill the run that produced the
    # AUC numbers above.
    try:
        import snr_aednet
        snr_aednet.reconcile(raw_ratio, ks=tuple(float(k) for k in ks))
        snr_aednet.iso_rer_report(summary, k=float(aednet_k))
    except ImportError:
        print("\n  [snr_aednet.py not found -- skipping the convention check "
              "and the iso-RER comparison]")

    for k in ks:
        ki = int(k)
        print(f"\n=== SNR summary, k = {ki}  "
              f"(SNR = {ki} * log10(M/N)) ===")
        print(f"  {'method':30s} {'50%':>8} {'100%':>8} {'Average':>9}   "
              f"{'RER@50':>7} {'RER@100':>8}")
        for name, cells, rer in rows:
            if "50" not in cells or "100" not in cells:
                continue
            a, b = cells["50"][ki], cells["100"][ki]
            print(f"  {name:30s} {a:8.2f} {b:8.2f} {(a+b)/2:9.2f}   "
                  f"{rer['50']:7.3f} {rer['100']:8.3f}")
        if ki == 10:
            print("  (the published control is 3 / 0 / 1.5 -- the k whose control "
                  "row reproduces that\n   is the convention those tables actually used)")


if __name__ == "__main__":
    main()
