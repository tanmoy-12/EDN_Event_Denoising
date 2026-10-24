from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ND_COLS = [4, 16, 64]
ND_LABEL = {4: "ND04", 16: "ND16", 64: "ND64"}
SUBSET = "N-END"

# SAGE Table IV, nighttime E-MLB, ESR. Higher is better.
# The BAF row is the calibration anchor; the rest is the field to enter.
PUBLISHED_SAGE_NIGHT = {
    "TS":        (0.812, 0.760, 0.706, 0.759),
    "DWF":       (0.603, 0.568, 0.522, 0.564),
    "NNb":       (0.766, 0.721, 0.681, 0.723),
    "BAF":       (0.832, 0.785, 0.733, 0.783),
    "KNoise":    (0.801, 0.753, 0.701, 0.752),
    "MLPF":      (0.782, 0.731, 0.680, 0.731),
    "GEF":       (0.906, 0.848, 0.796, 0.850),
    "EBF":       (0.914, 0.856, 0.801, 0.857),
    "EventZoom": (0.965, 0.902, 0.842, 0.903),
    "EDformer":  (0.921, 0.862, 0.807, 0.863),
    "EDmamba":   (1.018, 0.956, 0.895, 0.956),
    "AEDNet":    (1.032, 0.970, 0.909, 0.970),
    "SAGE":      (1.161, 1.094, 1.021, 1.092),
}
PUB_BAF = PUBLISHED_SAGE_NIGHT["BAF"]
CALIB_TOL = 0.030          # mean |error| on the BAF row to call it a match


# ==============================================================================
# aggregation -- repeats averaged per scene, then scenes averaged
# ==============================================================================
def _mean(vals):
    a = np.asarray(list(vals), float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else float("nan")


def load_cells(path):
    """-> cells[(nd, method)] = (esr, keep, n_scenes, n_dropped, n_thin)"""
    if not os.path.exists(path):
        sys.exit(f"no such file: {path}")
    blob = json.load(open(path))
    per_file = blob.get("per_file")
    if not per_file:
        sys.exit(f"{path} has no 'per_file' block -- is it a --random_curve "
                 "json? pass that with --random_json instead.")

    methods, per_scene, health = set(), {}, {}
    for rec in per_file.values():
        md = rec.get("_meta")
        if not md or md.get("subset") != SUBSET or md.get("nd") not in ND_COLS:
            continue
        for m, v in rec.items():
            if m == "_meta" or not isinstance(v, dict):
                continue
            methods.add(m)
            esr, keep = v.get("mesr"), v.get("keep")
            ns = v.get("slices", 0)
            tot, bad, thin = health.get((md["nd"], m), (0, 0, 0))
            ok = esr is not None and np.isfinite(esr)
            health[(md["nd"], m)] = (tot + 1,
                                     bad + (0 if ok else 1),
                                     thin + (1 if ok and ns < 10 else 0))
            per_scene.setdefault((md["nd"], md["scene"]), {}).setdefault(
                m, []).append((esr, keep))

    bucket = {}
    for (nd, _scene), d in per_scene.items():
        for m, vals in d.items():
            bucket.setdefault((nd, m), []).append(
                (_mean(v[0] for v in vals), _mean(v[1] for v in vals)))

    cells = {}
    for (nd, m), vals in bucket.items():
        good = [v for v in vals if np.isfinite(v[0])]
        tot, bad, thin = health.get((nd, m), (0, 0, 0))
        cells[(nd, m)] = (_mean(v[0] for v in good),
                          _mean(v[1] for v in good),
                          len(good), bad, thin)
    return cells, sorted(methods), blob.get("config", {})


def row(cells, m):
    """-> (three esr values, three keeps, avg esr, avg keep)"""
    e = [cells.get((nd, m), (np.nan,) * 5)[0] for nd in ND_COLS]
    k = [cells.get((nd, m), (np.nan,) * 5)[1] for nd in ND_COLS]
    return e, k, _mean(e), _mean(k)


def label_of(m):
    if m == "raw":
        return "Raw"
    if m == "baf":
        return "BAF (ours)"
    if m.startswith("ednet@"):
        return f"EDNet tau={m.split('@')[1]}"
    return m


# ==============================================================================
# random-deletion floor (optional overlay)
# ==============================================================================
def load_floor(path):
    """eval_emlb.py --random_curve json -> floor[nd] = (keeps, esrs) sorted."""
    if not path:
        return {}
    blob = json.load(open(path))
    cells = blob.get("cells")
    if not cells:
        sys.exit(f"{path} has no 'cells' block -- that is not a "
                 "--random_curve json.")
    pts = {}
    for key, val in cells.items():
        if val is None:
            continue
        sub, nd, m = key.split("|")
        if sub != SUBSET or not m.startswith("rand@"):
            continue
        nd = int(nd)
        if nd not in ND_COLS:
            continue
        pts.setdefault(nd, []).append((float(val[1]), float(val[0])))
    out = {}
    for nd, ps in pts.items():
        ps = sorted(p for p in ps if np.isfinite(p[0]) and np.isfinite(p[1]))
        if len(ps) >= 2:
            out[nd] = (np.array([p[0] for p in ps]),
                       np.array([p[1] for p in ps]))
    return out


def floor_at(floor, nd, keep):
    f = floor.get(nd)
    if not f or not np.isfinite(keep):
        return float("nan")
    return float(np.interp(keep, f[0], f[1]))


# ==============================================================================
# the table
# ==============================================================================
def print_table(cells, methods, floor, args):
    ours = ([m for m in methods if m == "raw"]
            + [m for m in methods if m == "baf"]
            + sorted(m for m in methods if m.startswith("ednet@")))
    if not ours:
        sys.exit("no scored methods found for N-END at ND04/ND16/ND64.")

    W = 78
    print("\n" + "=" * W)
    print("NIGHTTIME E-MLB, ESR    (SAGE Table IV format; higher is better)")
    print("=" * W)
    print(f"  {'Method':<22}{'ND04':>9}{'ND16':>9}{'ND64':>9}{'Avg.':>9}")
    print("  " + "-" * (22 + 36 - 2))
    for name, v in PUBLISHED_SAGE_NIGHT.items():
        star = "  *" if name == "BAF" else ""
        print(f"  {name:<22}" + "".join(f"{x:9.3f}" for x in v) + star)
    print("  " + "-" * (22 + 36 - 2))
    for m in ours:
        e, k, ea, _ka = row(cells, m)
        print(f"  {label_of(m):<22}"
              + "".join(f"{x:9.3f}" if np.isfinite(x) else "      n/a"
                        for x in e)
              + (f"{ea:9.3f}" if np.isfinite(ea) else "      n/a"))
    print("  " + "-" * (22 + 36 - 2))
    print("  * calibration anchor -- see below")

    # ---- retention, always beside the score ----------------------------
    print(f"\n  RETAINED FRACTION (ESR rises as a stream is thinned, so no")
    print(f"  cell above is readable without the cell below it)")
    print(f"  {'Method':<22}{'ND04':>9}{'ND16':>9}{'ND64':>9}{'Avg.':>9}")
    print("  " + "-" * (22 + 36 - 2))
    for m in ours:
        _e, k, _ea, ka = row(cells, m)
        print(f"  {label_of(m):<22}"
              + "".join(f"{x*100:8.1f}%" if np.isfinite(x) else "      n/a"
                        for x in k)
              + (f"{ka*100:8.1f}%" if np.isfinite(ka) else "      n/a"))

    # ---- random-deletion floor ----------------------------------------
    if floor:
        print(f"\n  EXCESS OVER THE RANDOM-DELETION FLOOR at each method's own")
        print(f"  retention (what the score is not explained by deletion alone)")
        print(f"  {'Method':<22}{'ND04':>9}{'ND16':>9}{'ND64':>9}{'Avg.':>9}")
        print("  " + "-" * (22 + 36 - 2))
        for m in ours:
            if m == "raw":
                continue
            e, k, _ea, _ka = row(cells, m)
            ex = [a - floor_at(floor, nd, b)
                  for nd, a, b in zip(ND_COLS, e, k)]
            flag = "   <-- AT OR BELOW FLOOR" if any(
                np.isfinite(v) and v <= 0 for v in ex) else ""
            print(f"  {label_of(m):<22}"
                  + "".join(f"{v:+9.3f}" if np.isfinite(v) else "      n/a"
                            for v in ex)
                  + (f"{_mean(ex):+9.3f}" if np.isfinite(_mean(ex))
                     else "      n/a") + flag)
    else:
        print("\n  (no --random_json given; run  eval_emlb.py --random_curve")
        print("   --subsets N-END --baf --ckpt ...  to add the floor overlay)")

    # ---- data health ---------------------------------------------------
    bad = {(nd, m): cells[(nd, m)] for (nd, m) in cells
           if m in ours and (cells[(nd, m)][3] or cells[(nd, m)][4])}
    if bad:
        print("\n  ! DATA HEALTH")
        for (nd, m), c in sorted(bad.items()):
            if c[3]:
                print(f"      {ND_LABEL[nd]} {label_of(m):<20} {c[3]} file(s) "
                      f"scoreless -- cell computed without them, which")
                print(f"           flatters it: those are the files it cut "
                      f"hardest")
            if c[4]:
                print(f"      {ND_LABEL[nd]} {label_of(m):<20} {c[4]} file(s) "
                      f"scored from fewer than 10 slices")

    ns = sorted({cells[(nd, m)][2] for (nd, m) in cells if m in ours})
    print(f"\n  scenes per cell: {ns[0] if len(ns) == 1 else ns}")

    return ours


# ==============================================================================
# calibration verdict
# ==============================================================================
def verdict(cells, ours, floor):
    W = 78
    print("\n" + "=" * W)
    print("CALIBRATION -- our BAF against SAGE Table IV's BAF row")
    print("=" * W)
    if ("baf" not in ours):
        print("  BAF was not scored. Re-run eval_emlb.py with --baf.")
        print("  Without it, SAGE's table cannot be calibrated at all: it")
        print("  carries no Raw row, so BAF is the only anchor it offers.")
        print("=" * W)
        return
    e, k, ea, _ka = row(cells, "baf")
    errs = []
    for nd, ours_v, pub in zip(ND_COLS, e, PUB_BAF[:3]):
        if np.isfinite(ours_v):
            errs.append(abs(ours_v - pub))
            print(f"  {ND_LABEL[nd]:<8} ours {ours_v:.3f}   published {pub:.3f}"
                  f"   diff {ours_v - pub:+.3f}")
    if np.isfinite(ea):
        print(f"  {'Avg.':<8} ours {ea:.3f}   published {PUB_BAF[3]:.3f}"
              f"   diff {ea - PUB_BAF[3]:+.3f}")
    if not errs:
        print("  no finite BAF cells.")
        print("=" * W)
        return

    me, mx = float(np.mean(errs)), float(np.max(errs))
    print(f"\n  mean |error| {me:.3f}   max {mx:.3f}")
    if me <= CALIB_TOL:
        print("\n  -> CALIBRATED. Our rows are on SAGE Table IV's scale and")
        print("     may be placed in it. Report the retention column too.")
    else:
        print("\n  -> NOT CALIBRATED. We are not computing what that table")
        print("     computed. Our ESR and theirs are different quantities,")
        print("     so an EDNet row placed in Table IV would be meaningless")
        print("     whether it landed above 1.092 or below it.")
        print("     Try --baf_sweep to find the dt they used; if nothing")
        print("     reproduces the row, report E-MLB only within our own run")
        print("     (raw / BAF / EDNet / random floor, one convention, one")
        print("     script) and say so explicitly.")

    # ---- where do we actually sit --------------------------------------
    best = None
    for m in ours:
        if not m.startswith("ednet@"):
            continue
        _e, _k, ea_, ka_ = row(cells, m)
        if np.isfinite(ea_) and (best is None or ea_ > best[1]):
            best = (m, ea_, ka_)
    if best:
        m, ea_, ka_ = best
        print(f"\n  Best EDNet row: {label_of(m)}  avg ESR {ea_:.3f} "
              f"at {ka_*100:.1f}% retention")
        print(f"    vs SAGE    {PUBLISHED_SAGE_NIGHT['SAGE'][3]:.3f}"
              f"   ({ea_ - PUBLISHED_SAGE_NIGHT['SAGE'][3]:+.3f})")
        print(f"    vs AEDNet  {PUBLISHED_SAGE_NIGHT['AEDNet'][3]:.3f}"
              f"   ({ea_ - PUBLISHED_SAGE_NIGHT['AEDNet'][3]:+.3f})")
        print(f"    vs EDmamba {PUBLISHED_SAGE_NIGHT['EDmamba'][3]:.3f}"
              f"   ({ea_ - PUBLISHED_SAGE_NIGHT['EDmamba'][3]:+.3f})")
        if me > CALIB_TOL:
            print("    (all three comparisons void while uncalibrated)")
        if floor:
            e_, k_, _, _ = row(cells, m)
            ex = _mean([a - floor_at(floor, nd, b)
                        for nd, a, b in zip(ND_COLS, e_, k_)])
            print(f"    excess over random-deletion floor: {ex:+.3f}")
            if np.isfinite(ex) and ex <= 0:
                print("    -> at or below the floor: this score is deletion,")
                print("       not denoising, and must not be reported as a win.")
    print("=" * W)


# ==============================================================================
# LaTeX
# ==============================================================================
def write_latex(cells, ours, path, calibrated):
    def f3(v):
        return f"{v:.3f}" if np.isfinite(v) else "---"

    L = ["% Generated by emlb_night_table.py",
         "% Published rows are SAGE Table IV. The BAF row is the calibration",
         "% anchor: our BAF must reproduce it for our row to belong here.",
         "\\begin{table}[t]",
         "\\centering",
         "\\caption{Extreme low-light evaluation on nighttime E-MLB using ESR. "
         "Higher values indicate better denoising performance. Retained "
         "fractions for our rows are given in Table~\\ref{tab:emlb-retention}; "
         "ESR is reference-free and increases as a stream is thinned, so the "
         "two must be read together.}",
         "\\label{tab:emlb-night}",
         "\\begin{tabular}{lcccc}",
         "\\toprule",
         "Method & ND04 & ND16 & ND64 & Avg. \\\\",
         "\\midrule"]
    for name, v in PUBLISHED_SAGE_NIGHT.items():
        L.append(f"{name} & " + " & ".join(f"{x:.3f}" for x in v) + r" \\")
    L.append("\\midrule")
    for m in ours:
        e, _k, ea, _ka = row(cells, m)
        nm = ("Raw" if m == "raw" else
              "BAF (reproduced)" if m == "baf" else
              r"\ourmethod{} ($\tau=" + m.split("@")[1] + r"$)")
        L.append(f"{nm} & " + " & ".join(f3(x) for x in e)
                 + f" & {f3(ea)}" + r" \\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    if not calibrated:
        L.insert(0, "% !! NOT CALIBRATED against the published BAF row. Do not")
        L.insert(1, "%    submit this table until the anchor reproduces.")

    # retention companion table
    L += ["", "\\begin{table}[t]", "\\centering",
          "\\caption{Retained event fraction for the rows of "
          "Table~\\ref{tab:emlb-night}.}",
          "\\label{tab:emlb-retention}",
          "\\begin{tabular}{lcccc}", "\\toprule",
          "Method & ND04 & ND16 & ND64 & Avg. \\\\", "\\midrule"]
    for m in ours:
        _e, k, _ea, ka = row(cells, m)
        nm = ("Raw" if m == "raw" else
              "BAF (reproduced)" if m == "baf" else
              r"\ourmethod{} ($\tau=" + m.split("@")[1] + r"$)")
        L.append(f"{nm} & " + " & ".join(
            f"{x*100:.1f}\\%" if np.isfinite(x) else "---" for x in k)
            + (f" & {ka*100:.1f}\\%" if np.isfinite(ka) else " & ---") + r" \\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]

    open(path, "w").write("\n".join(L) + "\n")
    print(f"\nwrote {path}"
          + ("" if calibrated else "  (marked NOT CALIBRATED at the top)"))


# ==============================================================================
# BAF dt sweep -- which time constant reproduces SAGE's BAF row?
# ==============================================================================
DT_CANDIDATES = (500, 1000, 2000, 5000, 10000, 20000, 50000)


def baf_sweep(args):
    """Re-reads night files; BAF only, no model, no GPU."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(args.emlb_py)) or ".")
    try:
        from eval_emlb import read_aedat4, scan
        from eval_mesr_dnd21 import baf_keep, mesr
    except ImportError as exc:
        sys.exit(f"could not import from eval_emlb.py / eval_mesr_dnd21.py: "
                 f"{exc}")

    files, _bad, _subs = scan(args.root)
    files = [f for f in files
             if f["subset"] == SUBSET and f["nd"] in ND_COLS]
    if args.limit_scenes:
        sc = sorted({f["scene"] for f in files})[:args.limit_scenes]
        files = [f for f in files if f["scene"] in set(sc)]
    files.sort(key=lambda f: (f["scene"], f["nd"], f["rep"]))
    if not files:
        sys.exit("no nighttime ND04/ND16/ND64 files found under --root")

    n_scenes = len({f["scene"] for f in files})
    print(f"BAF dt sweep: {len(files)} files, {n_scenes} scenes, "
          f"dt in {DT_CANDIDATES} us\n")
    if n_scenes < 8:
        print("  ! few scenes -- between-scene spread in ESR is large and a dt")
        print("    can look right or wrong purely from which scenes were "
              "drawn.\n")

    acc = {}
    for i, f in enumerate(files, 1):
        out = read_aedat4(f["path"])
        if out is None:
            continue
        (t, x, y, _p), (W, H) = out
        for dt in DT_CANDIDATES:
            k = baf_keep(t, x, y, W, H, dt)
            v = mesr(x[k], y[k], W, H, args.interval)[0]
            acc.setdefault((f["nd"], dt), {}).setdefault(
                f["scene"], []).append((v, float(k.mean())))
        if i % 5 == 0 or i == len(files):
            print(f"  [{i}/{len(files)}] {f['name']}")

    def cell(nd, dt):
        d = acc.get((nd, dt))
        if not d:
            return float("nan"), float("nan")
        per = [(_mean(a for a, _ in v), _mean(b for _, b in v))
               for v in d.values()]
        per = [q for q in per if np.isfinite(q[0])]
        if not per:
            return float("nan"), float("nan")
        return _mean(q[0] for q in per), _mean(q[1] for q in per)

    print("\n" + "=" * 78)
    print("BAF BY dt, against SAGE Table IV's BAF row")
    print("=" * 78)
    print(f"  {'dt (us)':<10}{'ND04':>9}{'ND16':>9}{'ND64':>9}{'Avg.':>9}"
          f"{'keep':>9}{'mean|err|':>11}")
    print("  " + "-" * 74)
    print(f"  {'PUBLISHED':<10}" + "".join(f"{v:9.3f}" for v in PUB_BAF))
    print("  " + "-" * 74)
    ranked = []
    for dt in DT_CANDIDATES:
        vals = [cell(nd, dt) for nd in ND_COLS]
        e = [v[0] for v in vals]
        k = _mean(v[1] for v in vals)
        ea = _mean(e)
        errs = [abs(a - p) for a, p in zip(e, PUB_BAF[:3]) if np.isfinite(a)]
        me = float(np.mean(errs)) if errs else float("nan")
        if errs:
            ranked.append((me, dt))
        print(f"  {dt:<10}" + "".join(f"{v:9.3f}" if np.isfinite(v) else
                                      "      n/a" for v in e)
              + (f"{ea:9.3f}" if np.isfinite(ea) else "      n/a")
              + (f"{k*100:8.1f}%" if np.isfinite(k) else "      n/a")
              + (f"{me:11.3f}" if np.isfinite(me) else "        n/a"))
    print("  " + "-" * 74)
    if ranked:
        ranked.sort()
        me, dt = ranked[0]
        print(f"\n  closest: dt = {dt} us  (mean |error| {me:.3f})")
        if me <= CALIB_TOL:
            print(f"  -> re-run eval_emlb.py with --baf_dt_us {dt} and the")
            print(f"     whole table is on SAGE's scale.")
        else:
            print("  -> no dt reproduces the published BAF row. The mismatch")
            print("     is not the time constant: it is the ESR convention,")
            print("     the slice length, or the data itself. Run")
            print("     eval_emlb.py --convention_sweep next, and if that also")
            print("     fails, do not place our rows in SAGE's table.")
    print("=" * 78)


# ==============================================================================
def main():
    ap = argparse.ArgumentParser(
        description="Nighttime E-MLB table in SAGE Table IV format")
    ap.add_argument("--json", help="the json written by eval_emlb.py")
    ap.add_argument("--random_json",
                    help="json from eval_emlb.py --random_curve (floor overlay)")
    ap.add_argument("--latex", help="write the LaTeX tables here")
    ap.add_argument("--baf_sweep", action="store_true",
                    help="sweep BAF dt against SAGE's published BAF row")
    ap.add_argument("--root", help="the EMLB folder (only for --baf_sweep)")
    ap.add_argument("--emlb_py", default="eval_emlb.py",
                    help="path to eval_emlb.py (only for --baf_sweep)")
    ap.add_argument("--limit_scenes", type=int,
                    help="first N night scenes, for a quick sweep")
    ap.add_argument("--interval", type=int, default=30000)
    args = ap.parse_args()

    if args.baf_sweep:
        if not args.root:
            sys.exit("--baf_sweep needs --root")
        baf_sweep(args)
        return
    if not args.json:
        sys.exit("give --json (or --baf_sweep --root ...)")

    cells, methods, cfg = load_cells(args.json)
    if cfg:
        print(f"from {args.json}:  interval={cfg.get('interval')}  "
              f"N={cfg.get('N')}  baf_dt_us={cfg.get('baf_dt_us')}  "
              f"ckpt={cfg.get('ckpt')}")
    floor = load_floor(args.random_json)
    ours = print_table(cells, methods, floor, args)
    verdict(cells, ours, floor)

    if args.latex:
        e, _k, ea, _ka = row(cells, "baf") if "baf" in ours else ([], [],
                                                                  np.nan, 0)
        errs = [abs(a - p) for a, p in zip(e, PUB_BAF[:3]) if np.isfinite(a)]
        calibrated = bool(errs) and float(np.mean(errs)) <= CALIB_TOL
        write_latex(cells, ours, args.latex, calibrated)


if __name__ == "__main__":
    sys.exit(main())
