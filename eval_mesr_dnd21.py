from __future__ import annotations

import argparse
import os
import sys

import numpy as np

SP = np.spacing(1)
PUBLISHED_DND21_RAW = 0.869          # EDmamba Table 2, "Raw" row, DND21 column


# ==============================================================================
# 1.  THE METRIC
# ==============================================================================
def esr_slice(x, y, W, H, variant="v1"):
    """ESR of one slice. Transcription of E-MLB `_calc_esr`."""
    N = int(len(x))
    if N < 2:
        return np.nan
    K = W * H
    idx = y.astype(np.int64) * W + x.astype(np.int64)
    n = np.bincount(idx, minlength=K).astype(np.float64)

    if variant == "v2":
        # EventStructuralRatioV2 -- median-filtered, 0.5**n, ln/K, x1000.
        # Provided only for comparison; not what the benchmark reports.
        from scipy.ndimage import median_filter
        n = median_filter(n.reshape(H, W), size=3).ravel()
        ntss = (n * n).sum() / (N * N)
        ln = (K - (0.5 ** n).sum()) / K
        return float(1000.0 * np.sqrt(ntss * ln))

    M = int(N * 2 / 3)
    ntss = (n * (n - 1.0)).sum() / (N + SP) / (N - 1 + SP)
    ln = K - ((1.0 - M / N) ** n).sum()
    return float(np.sqrt(ntss * ln))


def mesr(x, y, W, H, interval=30000, variant="v1", keep_tail=False):
    """Mean ESR over fixed-event-count slices of the (already denoised) stream.

    dv's `doEveryNumberOfElements` fires only on a full slice, so the remainder
    is dropped. `keep_tail` deviates from the reference on purpose, for testing.
    """
    n_ev = len(x)
    n_full = n_ev // interval
    scores = []
    for i in range(n_full):
        sl = slice(i * interval, (i + 1) * interval)
        scores.append(esr_slice(x[sl], y[sl], W, H, variant))
    if keep_tail and n_ev % interval >= 2:
        sl = slice(n_full * interval, n_ev)
        scores.append(esr_slice(x[sl], y[sl], W, H, variant))
    if not scores:
        return np.nan, 0
    return float(np.nanmean(scores)), len(scores)


# ==============================================================================
# 2.  BASELINE DENOISER
# ==============================================================================
def baf_keep(t_us, x, y, W, H, dt_us=2000):
    """Background-activity filter: keep an event with a recent 8-neighbour.

    Fully vectorised. Events are grouped by pixel; because the group key is
    primary and time is secondary, (pixel, t) is lexicographically sorted, so
    "last event at neighbour pixel q strictly before time t" is a single global
    searchsorted on the packed key pixel*BIG + t. Eight offsets, eight
    searchsorteds -- no Python loop over events, which matters because these
    recordings run to millions of events.
    """
    t_us = t_us.astype(np.int64)
    n = len(t_us)
    if n == 0:
        return np.zeros(0, bool)
    NEG = -(10 ** 18)
    BIG = int(t_us.max()) + 2

    pix = y.astype(np.int64) * W + x.astype(np.int64)
    order = np.lexsort((t_us, pix))
    pix_s, t_s = pix[order], t_us[order]
    key = pix_s * BIG + t_s
    starts = np.searchsorted(pix_s, np.arange(W * H + 1), side="left")

    best = np.full(n, NEG, np.int64)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue                      # neighbours only, not self
            nx, ny = x.astype(np.int64) + dx, y.astype(np.int64) + dy
            ok = (nx >= 0) & (nx < W) & (ny >= 0) & (ny < H)
            if not ok.any():
                continue
            q = ny[ok] * W + nx[ok]
            j = np.searchsorted(key, q * BIG + t_us[ok], side="left") - 1
            good = j >= starts[q]             # stayed inside that pixel's run
            cand = np.full(ok.sum(), NEG, np.int64)
            cand[good] = t_s[j[good]]
            np.maximum(best[ok], cand, out=cand)
            best[ok] = cand
    return (t_us - best) <= dt_us


# ==============================================================================
# 3.  EDNet INFERENCE -- every event scored, none dropped
# ==============================================================================
def ednet_probs(model, ev, N, device, batch_size, amp):
    """Per-event P(noise) for the whole file.

    The file is cut into non-overlapping N-event windows. The remainder is
    covered by one extra window anchored at the END of the file, from which
    only the previously unscored tail is taken -- so the returned array has
    exactly len(ev) entries and the event count is preserved. That matters
    here: ESR is computed on a fixed event count, so silently dropping a
    remainder would change the slicing of every method differently.
    """
    import torch
    from ed24_data import normalize_window

    n = ev.shape[0]
    if n < N:
        return None

    starts = [i * N for i in range(n // N)]
    tail = n % N
    if tail:
        starts.append(n - N)

    probs = np.empty(n, np.float32)
    probs[:] = np.nan

    with torch.no_grad():
        for b0 in range(0, len(starts), batch_size):
            chunk = starts[b0:b0 + batch_size]
            feats, metas = [], []
            for s in chunk:
                f, m = normalize_window(np.asarray(ev[s:s + N]))
                feats.append(f)
                metas.append(m)
            feat = torch.from_numpy(np.stack(feats)).to(device)
            meta = torch.from_numpy(np.stack(metas)).to(device)
            with torch.autocast("cuda", enabled=amp):
                logits = model(feat, meta)
            p = torch.sigmoid(logits.float()).cpu().numpy()
            for j, s in enumerate(chunk):
                if s == n - N and tail and s != (n // N) * N:
                    probs[n - tail:] = p[j][-tail:]      # tail only
                else:
                    probs[s:s + N] = p[j]

    assert not np.isnan(probs).any(), "some events went unscored"
    return probs


# ==============================================================================
# 3b. LOADER -- with a diagnosed bypass of event_io's collapsed-time guard
# ==============================================================================
def load_with_grid(path, relax=True):
    """load_events, but able to survive a coarsely quantised timestamp column.

    event_io refuses a file whose timestamps average more than 100 events per
    distinct value, on the grounds that a collapsed time column silently turns
    every spatiotemporal method into a constant score. That guard is right in
    general but it is a threshold on a ratio, and adding noise to a file with a
    QUANTISED timestamp grid raises the event count without adding any new
    distinct timestamps -- so the ratio degrades with noise level and eventually
    trips on a file that is perfectly usable.

    Here we retry without the guard and report the actual grid, so the coarse
    resolution becomes a stated property of the data rather than a crash or,
    worse, a silent pass.

    Returns (ev, grid_info).
    """
    from event_io import load_events
    try:
        ev = load_events(path)
        relaxed = False
    except ValueError as e:
        if not relax or "distinct values" not in str(e):
            raise
        ev = _load_unguarded(path)
        relaxed = True

    t = ev[:, 0]
    uniq = np.unique(t)
    span = float(t[-1] - t[0])
    d = np.diff(uniq)
    grid = {
        "n_events": int(len(t)),
        "n_distinct_t": int(len(uniq)),
        "events_per_stamp": float(len(t) / max(len(uniq), 1)),
        "span_us": span,
        "median_step_us": float(np.median(d)) if len(d) else float("nan"),
        "min_step_us": float(d.min()) if len(d) else float("nan"),
        "relaxed": relaxed,
    }
    return ev, grid


def _load_unguarded(path):
    """The event_io conversion, minus the collapsed-time check.

    Deliberately built from event_io's own primitives rather than reimplemented,
    so it cannot drift from the loader every other evaluation in this project
    uses. Only the guard is skipped.
    """
    from event_io import _UNIT_TO_US, infer_time_unit, read_raw

    cols = read_raw(path)
    for c in "txyp":
        if c not in cols:
            raise ValueError(f"{path}: no '{c}' column found")
    t_raw = np.asarray(cols["t"])
    unit = infer_time_unit(t_raw)

    n = len(t_raw)
    out = np.empty((n, 5), np.float64)
    tf = t_raw.astype(np.float64)
    out[:, 0] = (tf - np.nanmin(tf)) * _UNIT_TO_US[unit]
    out[:, 1] = np.asarray(cols["x"]).astype(np.float64)
    out[:, 2] = np.asarray(cols["y"]).astype(np.float64)

    p_raw = np.asarray(cols["p"]).astype(np.float64)
    uniq = np.unique(p_raw[np.isfinite(p_raw)])
    out[:, 3] = ((p_raw > 0.5 * (uniq[0] + uniq[-1])).astype(np.float64)
                 if len(uniq) == 2 else 0.0)
    out[:, 4] = (np.asarray(cols["l"]).astype(np.float64)
                 if "l" in cols else 0.0)

    out = out[~np.isnan(out).any(axis=1)]
    ev = np.empty(out.shape, np.int64)
    ev[:, 0] = np.rint(out[:, 0])
    ev[:, 1] = np.rint(out[:, 1])
    ev[:, 2] = np.rint(out[:, 2])
    ev[:, 3] = out[:, 3] > 0.5
    ev[:, 4] = out[:, 4] == 1
    if not np.all(np.diff(ev[:, 0]) >= 0):
        ev = ev[np.argsort(ev[:, 0], kind="stable")]
    return ev


# ==============================================================================
# 4.  SELF-TEST
# ==============================================================================
def verify():
    print("ESR self-test\n" + "-" * 66)
    rng = np.random.default_rng(0)
    W, H = 346, 260

    # (a) hand-computable case: every event on one pixel
    x = np.zeros(30000, np.int64); y = np.zeros(30000, np.int64)
    N = 30000; M = int(N * 2 / 3); K = W * H
    ntss = (N * (N - 1.0)) / (N + SP) / (N - 1 + SP)
    ln = K - ((K - 1) * 1.0 + (1 - M / N) ** N)
    print(f"  single-pixel    expected {np.sqrt(ntss*ln):.6f}   "
          f"got {esr_slice(x, y, W, H):.6f}")

    # (b) uniform noise -- the floor of the metric
    xu = rng.integers(0, W, 30000); yu = rng.integers(0, H, 30000)
    e_unif = esr_slice(xu, yu, W, H)

    # (c) structured: events on a few moving edges
    xs, ys = [], []
    for k in range(30000):
        cx = int(50 + (k / 30000) * 200)
        xs.append(np.clip(cx + rng.integers(-2, 3), 0, W - 1))
        ys.append(rng.integers(40, 220))
    e_struct = esr_slice(np.array(xs), np.array(ys), W, H)
    print(f"  uniform noise   {e_unif:.4f}")
    print(f"  structured edge {e_struct:.4f}   "
          f"({'OK, structure scores higher' if e_struct > e_unif else 'FAIL'})")

    # (d) monotonicity, through the real pipeline (delete, then RE-SLICE at
    #     30k). This is the only honest version of the test: slices are cut by
    #     event count, so removing events changes how much wall-clock time each
    #     slice spans, and that is where the metric's behaviour actually lives.
    def synth(n_sig, n_noi, halfwidth=1, rows=120, dur=2e6):
        ts = np.sort(rng.uniform(0, dur, n_sig))
        cx = 40 + (ts / dur) * 260
        xs = np.clip(cx + rng.integers(-halfwidth, halfwidth + 1, n_sig),
                     0, W - 1).astype(np.int64)
        ys = rng.integers(130 - rows // 2, 130 + rows // 2, n_sig)
        tn = np.sort(rng.uniform(0, dur, n_noi))
        xn = rng.integers(0, W, n_noi)
        yn = rng.integers(0, H, n_noi)
        t = np.concatenate([ts, tn])
        xx = np.concatenate([xs, xn])
        yy = np.concatenate([ys, yn])
        ll = np.concatenate([np.zeros(n_sig, bool), np.ones(n_noi, bool)])
        o = np.argsort(t, kind="stable")
        return xx[o], yy[o], ll[o]

    print("\n  monotonicity check -- a moving edge plus uniform BA noise,")
    print("  scored through the real pipeline (remove events, then re-slice):\n")
    print(f"    {'scene':<22}{'raw':>9}{'oracle':>9}{'keep 50% at random':>22}")
    for tag, a in (("30% noise", (420000, 180000)),
                   ("50% noise", (300000, 300000)),
                   ("70% noise", (180000, 420000))):
        xx, yy, ll = synth(*a)
        raw_v, _ = mesr(xx, yy, W, H)
        orc_v, _ = mesr(xx[~ll], yy[~ll], W, H)
        m = rng.random(len(xx)) < 0.5
        rnd_v, _ = mesr(xx[m], yy[m], W, H)
        flag = "  <-- oracle WORSE than raw" if orc_v < raw_v else ""
        print(f"    {tag:<22}{raw_v:9.4f}{orc_v:9.4f}{rnd_v:22.4f}{flag}")

    print("""
  Read that table before trusting any MESR number. Perfect, ground-truth
  denoising does not reliably raise ESR: at a low noise fraction it LOWERS it.
  The mechanism is the fixed 30,000-event slice. Remove events and each slice
  must span more wall-clock time to fill its budget, so the moving edge sweeps
  further and spreads over more pixels -- the structured content de-concentrates
  even though only noise was removed. Whether that loss outweighs the gain from
  dropping noise depends on the noise fraction, i.e. on the scene.

  So MESR is not monotone in denoising quality, and a method's rank under it
  is not by itself evidence about denoising. This is why the run below always
  prints raw, the oracle, and the retained fraction alongside every score.
""")


# ==============================================================================
# 5.  MAIN
# ==============================================================================
def main():
    ap = argparse.ArgumentParser(
        description="MESR on DND21, faithful to the E-MLB reference code")
    ap.add_argument("--ckpt")
    ap.add_argument("--auc_root",
                    help="AUC_test root: <root>/<hz>hz/<name>_result.txt")
    ap.add_argument("--files", nargs="+",
                    help="explicit event files instead of --auc_root")
    ap.add_argument("--datasets", nargs="+", default=["mix", "driving_mix"])
    ap.add_argument("--freqs", nargs="+", type=int, default=list(range(1, 11)))
    ap.add_argument("--width", type=int, default=346)
    ap.add_argument("--height", type=int, default=260)
    ap.add_argument("--interval", type=int, default=30000)
    ap.add_argument("--variant", choices=["v1", "v2"], default="v1")
    ap.add_argument("--keep_tail", action="store_true")
    ap.add_argument("--thresholds", nargs="+", type=float,
                    default=[0.3, 0.5, 0.7])
    ap.add_argument("--baf_dt_us", type=int, default=2000)
    ap.add_argument("--N", type=int, default=4096)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--no_baf", action="store_true")
    ap.add_argument("--strict_time", action="store_true",
                    help="do not bypass event_io's collapsed-timestamp guard")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--json", help="write per-file results here")
    args = ap.parse_args()

    if args.verify:
        verify()
        if not (args.ckpt or args.files or args.auc_root):
            return

    # ---- gather files ------------------------------------------------------
    files = []
    if args.files:
        files = [(os.path.basename(f), f) for f in args.files]
    elif args.auc_root:
        for name in args.datasets:
            for hz in args.freqs:
                p = os.path.join(args.auc_root, f"{hz}hz", f"{name}_result.txt")
                if os.path.exists(p):
                    files.append((f"{name}@{hz}Hz", p))
    if not files:
        print("no input files; pass --auc_root or --files")
        return

    # ---- model -------------------------------------------------------------
    model = device = None
    amp = False
    if args.ckpt:
        import torch
        from edn_model import EDNet, config_from_checkpoint
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        amp = (not args.no_amp) and device.type == "cuda"
        ck = torch.load(args.ckpt, map_location=device, weights_only=False)
        model = EDNet(config_from_checkpoint(ck, args.N)).to(device)
        model.load_state_dict(ck["model"])
        model.eval()
        print(f"loaded {args.ckpt} (epoch {ck.get('epoch','?')}, "
              f"val_auc {ck.get('val_auc','?')})")

    W, H = args.width, args.height
    print(f"\nESR: variant {args.variant}, slices of {args.interval} events "
          f"(tail {'kept' if args.keep_tail else 'dropped'}), sensor {W}x{H}, "
          f"M = int(2N/3)\n")

    methods = ["raw", "oracle"] + ([] if args.no_baf else ["baf"])
    methods += [f"ednet@{t:g}" for t in args.thresholds] if model else []

    rows, results = [], {}
    hdr = f"  {'sequence':<18}" + "".join(f"{m:>16}" for m in methods)
    print(hdr)
    print("  " + "-" * (18 + 16 * len(methods)))

    grids = {}
    for label, path in files:
        try:
            ev, grid = load_with_grid(path, relax=not args.strict_time)
        except ValueError as e:
            print(f"  ! {label}: {e}")
            continue
        grids[label] = grid
        if grid["relaxed"]:
            print(f"  ~ {label}: timestamps are quantised to a "
                  f"{grid['median_step_us']:.0f} us grid "
                  f"({grid['n_distinct_t']} distinct stamps, "
                  f"{grid['events_per_stamp']:.0f} events each); event_io's "
                  f"collapsed-time guard was bypassed for this file")
        t, x, y, lab = ev[:, 0], ev[:, 1], ev[:, 2], ev[:, 4]

        if x.max() >= W or y.max() >= H:
            print(f"  ! {label}: coords exceed {W}x{H} "
                  f"(max {x.max()},{y.max()}) -- pass --width/--height")
            continue

        if lab.sum() == 0:
            # No label column (the original DND21 release is unlabelled). The
            # oracle row would silently equal the raw row and look like a
            # result, so refuse it rather than print a duplicate.
            print(f"  ! {label}: no noise labels in this file -- the oracle "
                  f"row is meaningless here and is reported as nan")
            keeps = {"raw": np.ones(len(t), bool),
                     "oracle": np.zeros(len(t), bool)}
        else:
            keeps = {"raw": np.ones(len(t), bool),
                     "oracle": lab == 0}
        if not args.no_baf:
            keeps["baf"] = baf_keep(t, x, y, W, H, args.baf_dt_us)
        if model is not None:
            p = ednet_probs(model, ev, args.N, device, args.batch_size, amp)
            if p is None:
                print(f"  ! {label}: fewer than {args.N} events, skipped")
                continue
            for th in args.thresholds:
                keeps[f"ednet@{th:g}"] = p < th   # label 1 == noise

        cells, rec = [], {}
        for m in methods:
            k = keeps[m]
            v, nsl = mesr(x[k], y[k], W, H, args.interval,
                          args.variant, args.keep_tail)
            frac = k.mean()
            rec[m] = {"mesr": v, "keep": float(frac), "slices": nsl,
                      "n_kept": int(k.sum())}
            cells.append(f"{v:8.3f}/{frac*100:4.0f}%")
        results[label] = rec
        rows.append(rec)
        print(f"  {label:<18}" + "".join(f"{c:>16}" for c in cells))

    if not rows:
        return

    print("  " + "-" * (18 + 16 * len(methods)))
    means = {m: float(np.nanmean([r[m]["mesr"] for r in rows])) for m in methods}
    kmean = {m: float(np.nanmean([r[m]["keep"] for r in rows])) for m in methods}
    print(f"  {'MEAN':<18}" +
          "".join(f"{means[m]:8.3f}/{kmean[m]*100:4.0f}%".rjust(16)
                  for m in methods))
    print("\n  (each cell is  MESR / retained-fraction)")

    # ---- the calibration check --------------------------------------------
    raw = means["raw"]
    print("\n" + "=" * 70)
    print(f"CALIBRATION: our raw MESR = {raw:.3f}   "
          f"published DND21 raw = {PUBLISHED_DND21_RAW:.3f}   "
          f"ratio = {raw / PUBLISHED_DND21_RAW:.2f}x")
    if abs(raw - PUBLISHED_DND21_RAW) <= 0.05:
        print("  -> matches. Rows above are on the published scale.")
    else:
        print("  -> DOES NOT MATCH. Our DND21 copy is not the population the")
        print("     published table was computed on (the AUC_test build is a")
        print("     v2e-renoised derivative, not the original DND21 release).")
        print("     Do not put these numbers in the same table as the paper's.")
    if "oracle" in means:
        print(f"\nORACLE CHECK: ground-truth denoising scores {means['oracle']:.3f} "
              f"at {kmean['oracle']*100:.0f}% retention")
        beat = [m for m in methods
                if m not in ("raw", "oracle") and means[m] > means["oracle"]]
        if beat:
            print(f"  -> {', '.join(beat)} BEAT the oracle. MESR is rewarding")
            print("     event removal here, not denoising quality.")
        else:
            print("  -> no method beats the oracle; ordering is at least sane.")
    print("=" * 70)

    if args.json:
        import json
        with open(args.json, "w") as f:
            json.dump({"config": vars(args), "per_file": results,
                       "timestamp_grid": grids,
                       "mean": means, "keep": kmean}, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    sys.exit(main())
