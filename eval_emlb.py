from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import numpy as np

try:
    from eval_mesr_dnd21 import baf_keep, ednet_probs, esr_slice, mesr
except ImportError:
    sys.exit("eval_mesr_dnd21.py must sit next to this file (ESR lives there).")

FNAME = re.compile(r"^(?P<scene>.+)-ND(?P<nd>\d+)-(?P<rep>\d+)\.aedat4$", re.I)
ND_LABEL = {0: "ND1", 4: "ND4", 16: "ND16", 64: "ND64"}
ND_ORDER = [0, 4, 16, 64]
SUBSET_LABEL = {"D-END": "Daylight", "N-END": "Night"}

# EDmamba Table 2, the "Raw" row. This is the calibration anchor: it is the one
# published number that depends on the dataset and the metric convention alone,
# not on any denoiser, so if our raw row does not land near it we are not
# computing what that table computed and no row of ours belongs beside it.
# SAGE's Table IV has no Raw row and therefore cannot be calibrated at all.
PUBLISHED_RAW = {
    ("D-END", 0): 0.821, ("D-END", 4): 0.824,
    ("D-END", 16): 0.815, ("D-END", 64): 0.786,
    ("N-END", 0): 0.890, ("N-END", 4): 0.824,
    ("N-END", 16): 0.786, ("N-END", 64): 0.768,
}


# ==============================================================================
# 1.  AEDAT4 READER
# ==============================================================================
_BACKEND = None


def _pick_backend():
    global _BACKEND
    if _BACKEND:
        return _BACKEND
    try:
        import dv_processing  # noqa: F401
        _BACKEND = "dv_processing"
    except ImportError:
        try:
            import aedat  # noqa: F401
            _BACKEND = "aedat"
        except ImportError:
            sys.exit("No AEDAT4 reader. Run:  pip install dv-processing   "
                     "(or: pip install aedat)")
    return _BACKEND


def read_aedat4(path):
    """-> (t_us, x, y, p) int64 arrays, time-sorted, and (W, H).

    Resolution comes from the file's own stream header, never assumed: a
    hardcoded 346x260 would silently fold any differently-sized recording into
    the wrong K in the ESR denominator.
    """
    be = _pick_backend()
    if be == "dv_processing":
        import dv_processing as dv
        r = dv.io.MonoCameraRecording(path)
        W, H = r.getEventResolution()
        parts = []
        while r.isRunning():
            b = r.getNextEventBatch()
            if b is not None:
                parts.append(b.numpy())
        if not parts:
            return None
        a = np.concatenate(parts)
        t = a["timestamp"].astype(np.int64)
        x = a["x"].astype(np.int64)
        y = a["y"].astype(np.int64)
        p = a["polarity"].astype(np.int64)
    else:
        import aedat
        d = aedat.Decoder(path)
        info = d.id_to_stream()
        ev_ids = [k for k, v in info.items() if v.get("type") == "events"]
        if not ev_ids:
            return None
        W, H = info[ev_ids[0]]["width"], info[ev_ids[0]]["height"]
        parts = [pk["events"] for pk in d if "events" in pk]
        if not parts:
            return None
        a = np.concatenate(parts)
        t = a["t"].astype(np.int64)
        x = a["x"].astype(np.int64)
        y = a["y"].astype(np.int64)
        p = a["on"].astype(np.int64)

    if len(t) == 0:
        return None
    if not np.all(np.diff(t) >= 0):
        o = np.argsort(t, kind="stable")
        t, x, y, p = t[o], x[o], y[o], p[o]
    return (t - t[0], x, y, p), (int(W), int(H))


def mesr_time(t, x, y, W, H, interval_us, variant="v1"):
    """Mean ESR over fixed-DURATION slices -- dv's doEveryTimeInterval.

    E-MLB's metric.py exposes this alongside the event-count slicer and the
    papers do not say which they used. It is not a cosmetic difference: under
    count slicing a denoised stream stretches each slice over more wall-clock
    time, under time slicing it simply puts fewer events in a fixed window.
    """
    if len(t) < 2:
        return float("nan"), 0
    b = (t - t[0]) // interval_us
    edges = np.searchsorted(b, np.arange(b[-1] + 2))
    scores = []
    for i in range(len(edges) - 2):          # drop the trailing partial slice
        s_, e_ = edges[i], edges[i + 1]
        if e_ - s_ >= 2:
            scores.append(esr_slice(x[s_:e_], y[s_:e_], W, H, variant))
    if not scores:
        return float("nan"), 0
    return float(np.nanmean(scores)), len(scores)


# Candidate conventions. Each paper's ESR number depends on all of these and
# none of them states any. The raw stream is the probe: it involves no
# denoiser, so whichever convention reproduces the published Raw row is the
# one that table was computed under.
CONVENTIONS = [
    ("v1 / 30000 events", "v1", "count", 30000),
    ("v1 / 10000 events", "v1", "count", 10000),
    ("v1 / 50000 events", "v1", "count", 50000),
    ("v1 / 100000 events", "v1", "count", 100000),
    ("v1 / 33 ms", "v1", "time", 33000),
    ("v1 / 100 ms", "v1", "time", 100000),
    ("v2 / 30000 events", "v2", "count", 30000),
    ("v2 / 33 ms", "v2", "time", 33000),
]


def convention_sweep(args):
    """Find which ESR convention reproduces the published Raw row.

    Raw only -- no model, no thresholds. If our raw row is on the wrong
    convention then our EDNet row is wrong by the same factor, and no amount
    of work on the model can fix it. So this runs before anything else.
    """
    files, _bad, subsets = scan(args.root)
    if args.limit_scenes:
        keep = set()
        for sub in subsets:
            sc = sorted({f["scene"] for f in files if f["subset"] == sub})
            keep |= {(sub, s) for s in sc[:args.limit_scenes]}
        files = [f for f in files if (f["subset"], f["scene"]) in keep]
    files.sort(key=lambda f: (f["subset"], f["scene"], f["nd"], f["rep"]))

    n_sc = {s: len({f["scene"] for f in files if f["subset"] == s})
            for s in subsets}
    print(f"convention sweep over {len(files)} files "
          f"({n_sc}) -- raw stream only\n")
    if min(n_sc.values() or [0]) < 8:
        print("  ! Few scenes. Between-scene spread in this metric is large;")
        print("    with a handful of scenes a convention can look wrong purely")
        print("    from which scenes were drawn. Use at least 8-10 per subset,")
        print("    ideally all of them -- this pass has no model in it and is")
        print("    limited by disk read.")


    acc = {}
    for i, f in enumerate(files, 1):
        out = read_aedat4(f["path"])
        if out is None:
            continue
        (t, x, y, _p), (W, H) = out
        for name, var, mode, size in CONVENTIONS:
            v = (mesr(x, y, W, H, size, var)[0] if mode == "count"
                 else mesr_time(t, x, y, W, H, size, var)[0])
            acc.setdefault((f["subset"], f["nd"], name), {}).setdefault(
                f["scene"], []).append(v)
        if i % 5 == 0 or i == len(files):
            print(f"  [{i}/{len(files)}] {f['name']}")

    def cellval(sub, nd, name):
        d = acc.get((sub, nd, name))
        if not d:
            return float("nan")
        per_scene = [np.nanmean(v) for v in d.values()]
        per_scene = [v for v in per_scene if np.isfinite(v)]
        return float(np.mean(per_scene)) if per_scene else float("nan")

    print("\n" + "=" * 84)
    print("RAW MESR BY CONVENTION, against EDmamba Table 2 'Raw'")
    print("=" * 84)
    cells = [(s, n) for s in ("D-END", "N-END") for n in ND_ORDER
             if (s, n) in {(k[0], k[1]) for k in acc}]
    head = f"  {'convention':<20}"
    for s, n in cells:
        head += f"{SUBSET_LABEL[s][0]}-{ND_LABEL[n]:<4}".rjust(9)
    print(head + "      mean|err|   max|err|")
    print("  " + "-" * 82)
    print(f"  {'PUBLISHED':<20}" +
          "".join(f"{PUBLISHED_RAW[c]:9.3f}" for c in cells))
    print("  " + "-" * 82)
    ranked = []
    for name, _v, _m, _s in CONVENTIONS:
        vals = [cellval(s, n, name) for s, n in cells]
        err = [abs(v - PUBLISHED_RAW[c])
               for v, c in zip(vals, cells) if np.isfinite(v)]
        row = f"  {name:<20}" + "".join(
            f"{v:9.3f}" if np.isfinite(v) else "      n/a" for v in vals)
        if err:
            ranked.append((np.mean(err), np.max(err), name))
            row += f"{np.mean(err):12.3f}{np.max(err):11.3f}"
        print(row)
    print("  " + "-" * 82)
    if ranked:
        ranked.sort()
        best = ranked[0]
        print(f"\n  closest: {best[2]}  (mean error {best[0]:.3f}, "
              f"max {best[1]:.3f})")
        if best[0] <= 0.03:
            print("  -> that is the convention. Re-run the benchmark with it")
            print("     and our rows are on the published scale.")
        else:
            print("  -> no candidate reproduces the published Raw row. Either")
            print("     the convention is outside this list, or the published")
            print("     table was computed on different data (note the .Calib")
            print("     folder, and that the two source papers disagree with")
            print("     each other). Report MESR only within our own run.")
    print("=" * 84)


# ==============================================================================
# 2.  DATASET WALK
# ==============================================================================
def scan(root):
    """-> list of dicts, one per file, plus a structural report."""
    files, bad = [], []
    subsets = [d for d in sorted(os.listdir(root))
               if os.path.isdir(os.path.join(root, d)) and not d.startswith(".")]
    for sub in subsets:
        subdir = os.path.join(root, sub)
        for scene in sorted(os.listdir(subdir)):
            sdir = os.path.join(subdir, scene)
            if scene.startswith(".") or not os.path.isdir(sdir):
                continue
            for fn in sorted(os.listdir(sdir)):
                if not fn.lower().endswith(".aedat4"):
                    continue
                m = FNAME.match(fn)
                if not m:
                    bad.append(os.path.join(sub, scene, fn))
                    continue
                files.append({
                    "subset": sub, "scene": scene,
                    "nd": int(m.group("nd")), "rep": int(m.group("rep")),
                    "path": os.path.join(sdir, fn), "name": fn,
                })
    return files, bad, subsets


def structure_report(files, bad, subsets):
    print(f"subsets: {', '.join(subsets)}")
    for sub in subsets:
        fs = [f for f in files if f["subset"] == sub]
        scenes = sorted({f["scene"] for f in fs})
        nds = sorted({f["nd"] for f in fs})
        print(f"\n  {sub}  ({SUBSET_LABEL.get(sub, sub)})")
        print(f"    scenes {len(scenes)}   files {len(fs)}   "
              f"ND levels {['ND%02d' % n for n in nds]}")
        # completeness: every scene should have every ND at every rep
        reps = sorted({f["rep"] for f in fs})
        missing = []
        for sc in scenes:
            for nd in nds:
                have = {f["rep"] for f in fs
                        if f["scene"] == sc and f["nd"] == nd}
                for r in reps:
                    if r not in have:
                        missing.append(f"{sc}-ND{nd:02d}-{r}")
        if missing:
            print(f"    ! {len(missing)} missing combination(s): "
                  f"{', '.join(missing[:8])}{' ...' if len(missing) > 8 else ''}")
            print("      (repeats are averaged per scene, so a gap re-weights "
                  "that scene rather than dropping it -- check these)")
        else:
            print(f"    complete: {len(scenes)} scenes x {len(nds)} ND x "
                  f"{len(reps)} repeats = {len(scenes)*len(nds)*len(reps)}")
    if bad:
        print(f"\n  ! {len(bad)} file(s) did not match "
              f"<scene>-ND<nn>-<rep>.aedat4 and are EXCLUDED:")
        for b in bad[:10]:
            print(f"      {b}")


def inspect(root, n_sample, seed=0):
    files, bad, subsets = scan(root)
    if not files:
        sys.exit(f"no .aedat4 files found under {root}")
    print(f"backend: {_pick_backend()}\nroot: {root}\n")
    structure_report(files, bad, subsets)

    # sample across subsets and ND levels rather than the first n files, which
    # would all be one scene at one illuminance
    rng = np.random.default_rng(seed)
    buckets = {}
    for f in files:
        buckets.setdefault((f["subset"], f["nd"]), []).append(f)
    keys = sorted(buckets)
    picks = []
    for i in range(n_sample):
        k = keys[i % len(keys)]
        picks.append(buckets[k][rng.integers(len(buckets[k]))])

    print(f"\n\nsample of {len(picks)} files")
    print("  " + "-" * 104)
    print(f"  {'file':<28}{'res':>10}{'events':>11}{'ms':>9}{'ev/s':>11}"
          f"{'pol+':>7}{'dt_us':>8}{'uniq_t':>9}{'ev/stamp':>10}")
    print("  " + "-" * 104)
    res_seen, tot = set(), []
    for f in picks:
        t0 = time.time()
        out = read_aedat4(f["path"])
        if out is None:
            print(f"  {f['name'][:27]:<28}  EMPTY / no event stream")
            continue
        (t, x, y, p), (W, H) = out
        res_seen.add((W, H))
        dur = (t[-1] - t[0]) / 1e6
        uniq = np.unique(t)
        d = np.diff(uniq)
        tot.append((len(t), time.time() - t0))
        print(f"  {f['name'][:27]:<28}{f'{W}x{H}':>10}{len(t):>11,}"
              f"{dur*1e3:>9.0f}{len(t)/max(dur,1e-9):>11,.0f}"
              f"{p.mean():>7.2f}{np.median(d) if len(d) else 0:>8.0f}"
              f"{len(uniq):>9,}{len(t)/max(len(uniq),1):>10.1f}")
        if x.max() >= W or y.max() >= H:
            print(f"      ! coordinates exceed the declared resolution "
                  f"(max {x.max()},{y.max()}) -- ESR's K would be wrong")

    print("  " + "-" * 104)
    if len(res_seen) > 1:
        print(f"  ! MIXED RESOLUTIONS across the sample: {sorted(res_seen)}.")
        print("    ESR divides by K = W*H, so scores from different sensor")
        print("    sizes are not on one scale. Check before aggregating.")
    else:
        W, H = res_seen.pop()
        print(f"  resolution {W}x{H} (K = {W*H:,}) consistent across the sample")

    if tot:
        mean_ev = np.mean([n for n, _ in tot])
        mean_rd = np.mean([s for _, s in tot])
        print(f"\n  mean {mean_ev:,.0f} events/file, {mean_rd:.2f}s to read")
        print(f"  {len(files)} files -> roughly {mean_ev*len(files)/1e9:.1f}G "
              f"events, ~{mean_rd*len(files)/60:.0f} min of reading alone")
        wins = mean_ev / 4096
        print(f"  EDNet inference adds ~{wins:,.0f} windows/file; budget hours, "
              f"not minutes, and use --resume")


# ==============================================================================
# 3.  BENCHMARK
# ==============================================================================
def evaluate(args):
    files, bad, subsets = scan(args.root)
    if not files:
        sys.exit(f"no .aedat4 files found under {args.root}")

    if args.limit_scenes:
        keep = set()
        for sub in subsets:
            sc = sorted({f["scene"] for f in files if f["subset"] == sub})
            keep |= {(sub, s) for s in sc[:args.limit_scenes]}
        files = [f for f in files if (f["subset"], f["scene"]) in keep]
    if args.subsets:
        files = [f for f in files if f["subset"] in args.subsets]
    files.sort(key=lambda f: (f["subset"], f["scene"], f["nd"], f["rep"]))

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
        print(f"loaded {args.ckpt} (epoch {ck.get('epoch','?')})")

    methods = ["raw"] + (["baf"] if args.baf else [])
    methods += [f"ednet@{t:g}" for t in args.thresholds] if model else []

    done = {}
    if args.resume and args.json and os.path.exists(args.json):
        done = json.load(open(args.json)).get("per_file", {})
        print(f"resuming: {len(done)} file(s) already scored")

    print(f"\n{len(files)} files, methods: {', '.join(methods)}\n")
    t_start = time.time()
    for i, f in enumerate(files, 1):
        key = f"{f['subset']}/{f['scene']}/{f['name']}"
        if key in done and all(m in done[key] for m in methods):
            continue
        out = read_aedat4(f["path"])
        if out is None:
            print(f"  [{i}/{len(files)}] {key}  EMPTY, skipped")
            continue
        (t, x, y, p), (W, H) = out
        if args.max_events and len(t) > args.max_events:
            t, x, y, p = (a[:args.max_events] for a in (t, x, y, p))

        keeps = {"raw": np.ones(len(t), bool)}
        if args.baf:
            keeps["baf"] = baf_keep(t, x, y, W, H, args.baf_dt_us)
        if model is not None:
            ev = np.stack([t, x, y, p, np.zeros(len(t), np.int64)], 1)
            pr = ednet_probs(model, ev, args.N, device, args.batch_size, amp)
            if pr is None:
                print(f"  [{i}/{len(files)}] {key}  <{args.N} events, skipped")
                continue
            for th in args.thresholds:
                keeps[f"ednet@{th:g}"] = pr < th

        rec = {"_meta": {"W": W, "H": H, "n": int(len(t)),
                         "subset": f["subset"], "scene": f["scene"],
                         "nd": f["nd"], "rep": f["rep"]}}
        for m in methods:
            k = keeps[m]
            v, ns = mesr(x[k], y[k], W, H, args.interval)
            rec[m] = {"mesr": v, "keep": float(k.mean()), "slices": ns}
        done[key] = rec

        el = time.time() - t_start
        print(f"  [{i}/{len(files)}] {key[:56]:<56} "
              + "  ".join(f"{m}={rec[m]['mesr']:.3f}/{rec[m]['keep']*100:.0f}%"
                          for m in methods)
              + f"   [{el/60:.0f}m, eta {el/i*(len(files)-i)/60:.0f}m]")

        if args.json and i % 10 == 0:
            json.dump({"config": vars(args), "per_file": done},
                      open(args.json, "w"), indent=1)

    if args.json:
        json.dump({"config": vars(args), "per_file": done},
                  open(args.json, "w"), indent=1)
    tables(done, methods, args)


# ==============================================================================
# 4.  AGGREGATION AND TABLES
# ==============================================================================
def aggregate(done, methods):
    """-> cell[(subset, nd)][method] = (mesr, keep, n_scenes), coverage.

    Repeats are averaged per scene first, then scenes are averaged, so a scene
    with a missing repeat still counts once rather than three-fifths of a scene.

    A file whose denoised stream is too short to fill even one 30,000-event
    slice yields no ESR at all. Averaging over whatever is left would quietly
    compute the cell from a subset selected BY the method's own aggressiveness
    -- the files it emptied are exactly the ones it did worst on. So those are
    counted and reported rather than dropped in silence.
    """
    per_scene, dropped = {}, {}
    for rec in done.values():
        md = rec["_meta"]
        k = (md["subset"], md["nd"], md["scene"])
        for m in methods:
            if m not in rec:
                continue
            v = rec[m]["mesr"]
            ns = rec[m].get("slices", 0)
            tot, bad, thin = dropped.get((md["subset"], md["nd"], m), (0, 0, 0))
            dropped[(md["subset"], md["nd"], m)] = (
                tot + 1,
                bad + (1 if not np.isfinite(v) else 0),
                thin + (1 if np.isfinite(v) and ns < 10 else 0))
            per_scene.setdefault(k, {}).setdefault(m, []).append(
                (v, rec[m]["keep"]))

    def _mean(vals):
        a = np.asarray(vals, float)
        a = a[np.isfinite(a)]
        return float(a.mean()) if a.size else float("nan")

    cell = {}
    for (sub, nd, _sc), d in per_scene.items():
        for m, vals in d.items():
            a = np.array(vals, float)
            cell.setdefault((sub, nd), {}).setdefault(m, []).append(
                (_mean(a[:, 0]), _mean(a[:, 1])))
    out = {}
    for key, d in cell.items():
        out[key] = {m: (_mean([v[0] for v in vals]),
                        _mean([v[1] for v in vals]),
                        len(vals))
                    for m, vals in d.items()}
    return out, dropped


def tables(done, methods, args):
    cell, dropped = aggregate(done, methods)
    subs = [s for s in ("D-END", "N-END") if any(k[0] == s for k in cell)]
    nds = [n for n in ND_ORDER if any(k[1] == n for k in cell)]

    print("\n\n" + "=" * 78)
    print("E-MLB, MESR / retained-fraction     (mean over scenes; repeats "
          "averaged first)")
    print("=" * 78)
    hdr = f"  {'method':<14}"
    for s in subs:
        for n in nds:
            hdr += f"{SUBSET_LABEL[s][0]}-{ND_LABEL[n]:<5}".rjust(15)
    print(hdr)
    print("  " + "-" * (14 + 15 * len(subs) * len(nds)))
    for m in methods:
        row = f"  {m:<14}"
        for s in subs:
            for n in nds:
                c = cell.get((s, n), {}).get(m)
                if not c or not np.isfinite(c[0]):
                    row += "n/a".rjust(15)
                else:
                    row += f"{c[0]:7.3f}/{c[1]*100:3.0f}%".rjust(15)
        print(row)
    ns = {(s, n): cell.get((s, n), {}).get(methods[0], (0, 0, 0))[2]
          for s in subs for n in nds}
    print("  " + "-" * (14 + 15 * len(subs) * len(nds)))
    print(f"  scenes per cell: "
          f"{dict((f'{s}-ND{n:02d}', v) for (s, n), v in ns.items())}")

    holes = {k: v for k, v in dropped.items() if v[1]}
    if holes:
        print("\n  ! SCORELESS FILES -- the denoised stream did not fill one "
              f"{args.interval}-event slice:")
        for (s, n, m), (tot, bad, thin) in sorted(holes.items()):
            print(f"      {s}-ND{n:02d} {m:<14} {bad} of {tot} files")
        print("    Those cells are computed from the remaining files only, and")
        print("    the files that vanished are the ones that method cut hardest")
        print("    -- so the cell flatters it. Treat any cell with a hole here")
        print("    as not reportable rather than as a slightly noisier number.")

    thin_cells = {k: v for k, v in dropped.items() if v[2]}
    if thin_cells:
        print("\n  ! THIN FILES -- fewer than 10 ESR slices:")
        for (s, n, m), (tot, bad, thin) in sorted(thin_cells.items()):
            print(f"      {s}-ND{n:02d} {m:<14} {thin} of {tot} files")
        print("    A one-slice MESR is finite and passes every check above, so")
        print("    it enters the cell as if it were an ordinary measurement.")
        print("    On the night probe this hit 31 of 144 files at tau=0.3, with")
        print("    Bridge-ND64 scored from a single slice at 5% retention.")
        print("    A cell with many thin files is not comparable with one")
        print("    without, in either direction.")

    if any(k[0] == "N-END" for k in cell):
        print("\n  Night-only average over ND4/ND16/ND64 (the SAGE Table IV "
              "column):")
        for m in methods:
            vals = [cell.get(("N-END", n), {}).get(m) for n in (4, 16, 64)]
            vals = [v for v in vals if v and np.isfinite(v[0])]
            if len(vals) == 3:
                print(f"    {m:<14} MESR {np.mean([v[0] for v in vals]):.3f}   "
                      f"keep {np.mean([v[1] for v in vals])*100:.0f}%")

    # ---- calibration against the published Raw row --------------------
    print("\n  CALIBRATION -- our raw row against EDmamba Table 2 'Raw':")
    devs = []
    for s_ in subs:
        for n in nds:
            c = cell.get((s_, n), {}).get("raw")
            pub = PUBLISHED_RAW.get((s_, n))
            if c and np.isfinite(c[0]) and pub:
                devs.append(abs(c[0] - pub))
                print(f"    {SUBSET_LABEL[s_]:<9}{ND_LABEL[n]:<6} "
                      f"ours {c[0]:.3f}   published {pub:.3f}   "
                      f"diff {c[0]-pub:+.3f}")
    if devs:
        if max(devs) <= 0.05:
            print("    -> matches. Our rows are on that table's scale.")
        else:
            print(f"    -> MAX DEVIATION {max(devs):.3f}. We are not computing "
                  "what that")
            print("       table computed -- convention, slice length or ESR")
            print("       variant differs. Do not place our row in it.")
        print("    (Note the two published sources disagree with each other: "
              "for")
        print("     EDformer at Night ND64, EDmamba's table reports 1.099 and")
        print("     SAGE's Table IV reports 0.807, and the trend with ND runs "
              "in")
        print("     opposite directions. At most one convention can be matched.)")

    print("\n  Every cell above is MESR / retained fraction. E-MLB has no")
    print("  ground-truth labels, so nothing here separates denoising from")
    print("  deletion; compare cells only at comparable retention.")
    print("=" * 78)

    if args.latex:
        write_latex(cell, methods, subs, nds, args)


def write_latex(cell, methods, subs, nds, args):
    def fmt(s, n, m):
        c = cell.get((s, n), {}).get(m)
        return f"{c[0]:.3f}" if c and np.isfinite(c[0]) else "---"

    L = ["% Generated by eval_emlb.py -- EDNet rows only.",
         "% Paste the published baseline rows above \\midrule and check that",
         "% your raw row is near theirs before trusting the comparison.",
         "\\begin{table}[t]", "\\centering",
         "\\caption{MESR on E-MLB. Higher is better. Retained fractions in "
         "Table~\\ref{tab:emlb-retention}.}",
         "\\label{tab:emlb}", "\\small",
         "\\begin{tabular}{l" + "c" * (len(subs) * len(nds)) + "}", "\\toprule"]
    head = "Method"
    for s in subs:
        for n in nds:
            head += f" & {ND_LABEL[n]}"
    L += [head + r" \\", "\\midrule"]
    for m in methods:
        L.append(m.replace("ednet@", r"\ourmethod{} $\tau=") .replace("raw", "Raw")
                 + ("$" if m.startswith("ednet@") else "")
                 + " & " + " & ".join(fmt(s, n, m) for s in subs for n in nds)
                 + r" \\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    open(args.latex, "w").write("\n".join(L) + "\n")
    print(f"\nwrote {args.latex}")


# ==============================================================================
def main():
    ap = argparse.ArgumentParser(description="E-MLB (ESR/MESR) benchmark")
    ap.add_argument("--root", required=True, help="the EMLB folder")
    ap.add_argument("--inspect", action="store_true",
                    help="check dataset structure and sample files, then stop")
    ap.add_argument("--inspect_n", type=int, default=8)
    ap.add_argument("--random_curve", action="store_true",
                    help="MESR of random deletion at matched retention -- the "
                         "control that separates denoising from deletion")
    ap.add_argument("--convention_sweep", action="store_true",
                    help="raw-only: find which ESR convention reproduces the "
                         "published Raw row. Run this before trusting any "
                         "absolute MESR.")
    ap.add_argument("--ckpt")
    ap.add_argument("--subsets", nargs="+", help="e.g. N-END")
    ap.add_argument("--limit_scenes", type=int,
                    help="first N scenes per subset, for a smoke run")
    ap.add_argument("--max_events", type=int,
                    help="truncate each file to this many events")
    ap.add_argument("--thresholds", nargs="+", type=float,
                    default=[0.3, 0.5, 0.7])
    ap.add_argument("--baf", action="store_true",
                    help="also run the unlearned background-activity filter")
    ap.add_argument("--baf_dt_us", type=int, default=2000)
    ap.add_argument("--interval", type=int, default=30000)
    ap.add_argument("--N", type=int, default=None,
                    help="window size in events. Default: whatever the "
                         "checkpoint was trained with.")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--json")
    ap.add_argument("--latex")
    args = ap.parse_args()
    if args.N is None:
        args.N = 4096
        if args.ckpt:
            import torch
            _ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
            args.N = int(_ck.get("args", {}).get("N", 4096))
            del _ck
        print(f"  N = {args.N}")
    if args.inspect:
        inspect(args.root, args.inspect_n)
        return
    if args.convention_sweep:
        convention_sweep(args)
        return
    if args.random_curve:
        random_curve(args)
        return
    evaluate(args)




# ==============================================================================
# 6.  RANDOM-DELETION REFERENCE CURVE
# ==============================================================================
#
# E-MLB has no labels, so "is this score denoising or deletion?" cannot be asked
# directly. It CAN be asked against a control: what does the same stream score
# when the same fraction of events is removed AT RANDOM? Random deletion keeps
# no structure whatsoever, so it is a floor. A method whose MESR sits at or
# below that floor at its own retention has demonstrated deletion, not
# denoising; the distance above the floor is a retention-controlled quality
# measure that needs no ground truth.
#
# This is the E-MLB analogue of the iso-RER analysis used on DVSCLEAN, and it
# has one large advantage over re-implementing competing methods: random
# deletion has no implementation to get wrong, so the control cannot be accused
# of being a weakened version of somebody's algorithm.

RETENTIONS = (0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95)


def random_curve(args):
    files, _bad, subsets = scan(args.root)
    if args.subsets:
        files = [f for f in files if f["subset"] in args.subsets]
    if args.limit_scenes:
        keep = set()
        for sub in {f["subset"] for f in files}:
            sc = sorted({f["scene"] for f in files if f["subset"] == sub})
            keep |= {(sub, s) for s in sc[:args.limit_scenes]}
        files = [f for f in files if (f["subset"], f["scene"]) in keep]
    files.sort(key=lambda f: (f["subset"], f["scene"], f["nd"], f["rep"]))

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

    acc = {}
    print(f"{len(files)} files; random-deletion control at "
          f"{', '.join('%.0f%%' % (r*100) for r in RETENTIONS)}\n")
    for i, f in enumerate(files, 1):
        out = read_aedat4(f["path"])
        if out is None:
            continue
        (t, x, y, p), (W, H) = out
        rng = np.random.default_rng(abs(hash(f["name"])) % (2 ** 32))
        rec = {}
        for r in RETENTIONS:
            m = rng.random(len(t)) < r
            rec[f"rand@{r:g}"] = (mesr(x[m], y[m], W, H, args.interval)[0],
                                  float(m.mean()))
        rec["raw"] = (mesr(x, y, W, H, args.interval)[0], 1.0)
        if args.baf:
            k = baf_keep(t, x, y, W, H, args.baf_dt_us)
            rec["baf"] = (mesr(x[k], y[k], W, H, args.interval)[0],
                          float(k.mean()))
        if model is not None:
            ev = np.stack([t, x, y, p, np.zeros(len(t), np.int64)], 1)
            pr = ednet_probs(model, ev, args.N, device, args.batch_size, amp)
            if pr is not None:
                for th in args.thresholds:
                    k = pr < th
                    rec[f"ednet@{th:g}"] = (
                        mesr(x[k], y[k], W, H, args.interval)[0],
                        float(k.mean()))
        for mm, v in rec.items():
            acc.setdefault((f["subset"], f["nd"], mm), {}).setdefault(
                f["scene"], []).append(v)
        if i % 5 == 0 or i == len(files):
            print(f"  [{i}/{len(files)}] {f['name']}")

    def cell(sub, nd, mm):
        d = acc.get((sub, nd, mm))
        if not d:
            return None
        per = [(np.nanmean([a for a, _ in v]), np.nanmean([b for _, b in v]))
               for v in d.values()]
        per = [q for q in per if np.isfinite(q[0])]
        if not per:
            return None
        return (float(np.mean([a for a, _ in per])),
                float(np.mean([b for _, b in per])))

    keys = sorted({(k[0], k[1]) for k in acc})
    tested = ([f"ednet@{t:g}" for t in args.thresholds] if model else []) \
        + (["baf"] if args.baf else [])

    print("\n" + "=" * 76)
    print("MESR ABOVE THE RANDOM-DELETION FLOOR, at each method's own retention")
    print("=" * 76)
    for sub, nd in keys:
        pts = [(cell(sub, nd, f"rand@{r:g}")) for r in RETENTIONS]
        pts = [q for q in pts if q]
        if len(pts) < 2:
            continue
        xs = np.array([q[1] for q in pts])
        ys = np.array([q[0] for q in pts])
        o = np.argsort(xs)
        xs, ys = xs[o], ys[o]
        raw = cell(sub, nd, "raw")
        print(f"\n  {SUBSET_LABEL.get(sub, sub)} {ND_LABEL.get(nd, nd)}"
              f"   raw {raw[0]:.3f}")
        print(f"    random floor:  " +
              "  ".join(f"{b*100:.0f}%:{a:.3f}" for a, b in zip(ys, xs)))
        for mm in tested:
            c = cell(sub, nd, mm)
            if not c:
                continue
            floor = float(np.interp(c[1], xs, ys))
            print(f"    {mm:<12} {c[0]:.3f} @ {c[1]*100:4.1f}%   "
                  f"floor {floor:.3f}   excess {c[0]-floor:+.3f}"
                  + ("   <-- AT OR BELOW THE FLOOR" if c[0] <= floor else ""))
    print("\n  A method at or below the floor scored no better than deleting")
    print("  the same number of events at random. Excess above the floor is")
    print("  the part of its score that deletion alone does not explain.")
    print("=" * 76)
    if args.json:
        json.dump({"config": vars(args),
                   "cells": {f"{s}|{n}|{m}": cell(s, n, m)
                             for (s, n, m) in acc}},
                  open(args.json, "w"), indent=1)
        print(f"\nwrote {args.json}")

if __name__ == "__main__":
    sys.exit(main())
