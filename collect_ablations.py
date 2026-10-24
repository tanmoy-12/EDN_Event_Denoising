from __future__ import annotations
import argparse, json, os, re

ORDER = ["full", "a1_rawdens", "a2_idxgraph", "a3_nofilm", "a3b_constfilm",
         "a4_nomotion", "a5_nopixel", "a6_noaug"]
LABEL = {
    "full":        r"\ourmethod{} (full)",
    "a1_rawdens":  r"raw counts (no rate normalisation)",
    "a2_idxgraph": r"event-index neighbourhood",
    "a3_nofilm":   r"no window conditioner / FiLM",
    "a3b_constfilm": r"conditioner fed constants (capacity-matched)",
    "a4_nomotion": r"no motion-adaptive channels",
    "a5_nopixel":  r"no pixel-recurrence channels",
    "a6_noaug":    r"no shot-noise augmentation",
}

# which component of the paper's claim each row tests, for the caption
CLAIM = {
    "a1_rawdens":  "rate-normalised density descriptor",
    "a2_idxgraph": "physical-time neighbourhood",
    "a3_nofilm":   "window-level rate conditioning",
    "a3b_constfilm": "window-level rate conditioning, capacity-matched",
    "a4_nomotion": "motion-adaptive cues",
    "a5_nopixel":  "pixel-recurrence cues",
    "a6_noaug":    "synthetic shot-noise augmentation",
}



def read_text(path):
    """Read a log whatever encoding it landed in.

    Logs written by these scripts are explicit UTF-8, but a log captured with
    PowerShell's `>` redirect is UTF-16LE -- with or without a BOM -- and read
    as UTF-8 it becomes text with a NUL between every character, so every
    regex below silently fails and the row shows as "--". Sniff the first
    bytes and decode accordingly rather than guessing from the extension.
    """
    raw = open(path, "rb").read()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16", errors="ignore")
    if raw[:4] in (b"\xff\xfe\x00\x00", b"\x00\x00\xfe\xff"):
        return raw.decode("utf-32", errors="ignore")
    # No BOM: UTF-16LE still shows as NULs in the odd byte positions.
    head = raw[:4000]
    if head.count(b"\x00") > len(head) // 4:
        le = raw.decode("utf-16-le", errors="ignore")
        be = raw.decode("utf-16-be", errors="ignore")
        return le if le.count("\x00") <= be.count("\x00") else be
    return raw.decode("utf-8", errors="ignore")

def read_run(d):
    h = os.path.join(d, "history.json")
    if not os.path.exists(h):
        return None
    hist = json.load(open(h))
    if not hist:
        return None
    best = max(hist, key=lambda e: e.get("selection", e.get("val_auc", 0)))
    lv = [v for v in best.get("per_level", {}).values() if v == v]
    params = None
    try:
        import torch
        ck = torch.load(os.path.join(d, "best.pth"), map_location="cpu",
                        weights_only=False)
        params = sum(v.numel() for v in ck["model"].values()) / 1e6
    except Exception:
        pass
    return dict(epoch=best["epoch"], val=best.get("val_auc"),
                shot=best.get("shot_auc"), worst=min(lv) if lv else None,
                params=params, epochs=len(hist))


def scrape(path, pat, n=2):
    """Pull the first n matches of pat out of an eval log, or None.

    A match with groups returns a tuple of floats per match.
    """
    if not path or not os.path.exists(path):
        return None
    txt = read_text(path)
    m = re.findall(pat, txt)
    if len(m) < n:
        return None          # a partial log is a crashed run, not a result
    m = m[:n]
    if isinstance(m[0], tuple):
        return [tuple(float(x) for x in g) for g in m]
    return [float(x) for x in m]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/abl")
    ap.add_argument("--dvsclean", default=None,
                    help="dir of <variant>.txt from eval_dvsclean_snr.py")
    ap.add_argument("--dnd21", default=None,
                    help="dir of <variant>.txt from eval_auc_dnd21.py")
    ap.add_argument("--tex", default=None, help="write the LaTeX table here")
    ap.add_argument("--mode", choices=["delta", "abs"], default="delta")
    a = ap.parse_args()

    rows, ref = [], None
    for v in ORDER:
        r = read_run(os.path.join(a.runs, v))
        if r is None:
            print(f"  {v:14s} MISSING ({os.path.join(a.runs, v)})")
            continue
        r["name"] = v
        # DVSCLEAN prints 50% then 100%; DND21 is run with --datasets mix
        # driving_mix, so hotel-bar then driving.
        dvsp = os.path.join(a.dvsclean, f"{v}.txt") if a.dvsclean else None
        dndp = os.path.join(a.dnd21, f"{v}.txt") if a.dnd21 else None
        r["dvs"] = scrape(dvsp, r"MEAN over \d+ scenes\s+([0-9.]+)")
        # The dispersion line, which is where the discriminating signal is:
        # the mean AUC is saturated on all three benchmarks, the worst-case
        # sequence is not. Order on the line is std, min, max.
        r["disp"] = scrape(dvsp, r"std / min / max\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)")
        r["dnd"] = scrape(dndp, r"mean over \d+ levels:\s+([0-9.]+)\s+worst:\s+([0-9.]+)")
        r["dvs50"], r["dvs100"] = (r["dvs"] or [None, None])
        r["sd50"], r["worst50"] = ((r["disp"][0][0], r["disp"][0][1])
                                   if r["disp"] else (None, None))
        r["sd100"], r["worst100"] = ((r["disp"][1][0], r["disp"][1][1])
                                     if r["disp"] else (None, None))
        r["hotel"], r["hotel_w"] = (r["dnd"][0] if r["dnd"] else (None, None))
        r["driving"], r["driving_w"] = (r["dnd"][1] if r["dnd"] else (None, None))
        if v == "full":
            ref = r
        rows.append(r)

    if not rows:
        raise SystemExit("  no runs found; check --runs")
    if ref is None:
        raise SystemExit("  the 'full' run is missing; there is nothing to "
                         "take deltas against.")

    budget = {r["epochs"] for r in rows}
    print(f"\n  epochs per run: {sorted(budget)}"
          + ("   <-- UNEQUAL BUDGET, the table is not controlled"
             if len(budget) > 1 else "   (equal, as required)"))

    sel = {r["name"]: r["epoch"] for r in rows}
    print(f"  selected epoch: " + "  ".join(f"{k}={v}" for k, v in sel.items()))

    COLS = [("ED24", "val"), ("shot10", "shot"),
            ("DVS50", "dvs50"), ("DVS100", "dvs100"),
            ("DVS100w", "worst100"), ("DVS100sd", "sd100"),
            ("DVS50w", "worst50"), ("DVS50sd", "sd50"),
            ("DNDhot", "hotel"), ("DNDhotW", "hotel_w"),
            ("DNDdrv", "driving"), ("DNDdrvW", "driving_w")]
    # Columns that go into the paper's table; the rest are diagnostics.
    TEXCOLS = [("ED24", "val"), ("shot10", "shot"),
               ("DVS50", "dvs50"), ("DVS100", "dvs100"),
               ("DVS100w", "worst100"), ("DVS100sd", "sd100"),
               ("DNDhot", "hotel"), ("DNDdrv", "driving")]

    print("\n  absolute")
    print(f"  {'variant':14s} {'par(M)':>7s} " +
          " ".join(f"{h:>8s}" for h, _ in COLS))
    for r in rows:
        cells = " ".join(f"{r[k]:8.4f}" if r[k] is not None else f"{'--':>8s}"
                         for _, k in COLS)
        print(f"  {r['name']:14s} {(r['params'] or float('nan')):7.3f} {cells}")

    print("\n  delta vs full")
    print(f"  {'variant':14s} " + " ".join(f"{h:>9s}" for h, _ in COLS))
    for r in rows:
        if r["name"] == "full":
            continue
        cells = " ".join(
            f"{r[k] - ref[k]:+9.4f}" if (r[k] is not None and ref[k] is not None)
            else f"{'--':>9s}" for _, k in COLS)
        print(f"  {r['name']:14s} {cells}")

    # ------------------------------------------------------------------ LaTeX
    def cell(r, k, nd=4):
        v, rv = r[k], ref[k]
        if v is None:
            return "--"
        if a.mode == "abs" or r["name"] == "full":
            return f"${v:.{nd}f}$"
        if rv is None:
            return f"${v:.{nd}f}$"
        d = v - rv
        return f"${d:+.{nd}f}$" if abs(d) >= 0.5 * 10 ** (-nd) else "$0.0000$"

    head = (r"Variant & Params (M) & ED24 & shot@$10$~Hz "
            r"& $50\%$ & $100\%$ & worst seq. & SD "
            r"& \emph{hotel-bar} & \emph{driving} \\")
    tex = [r"\begin{table*}[t]", r"\centering",
           r"\caption{Component ablation. Every variant is trained with the "
           r"identical recipe --- same $28$ epochs, scene split, augmentation "
           r"policy, optimiser, schedule, seed and checkpoint-selection rule "
           r"--- so the only difference between a row and the first row is the "
           r"component named in it. ED24 is the mean per-level validation AUC "
           r"at the selected epoch and shot@$10$~Hz the held-out shot-noise "
           r"probe, both in-domain; the DVSCLEAN and DND21 columns are "
           r"zero-shot transfer under the protocols of "
           r"Sections~\ref{sec:dvsclean_eval} and~\ref{sec:dnd21}. ``worst seq.'' "
           r"and ``SD'' are the lowest per-sequence AUC and the standard "
           r"deviation across the $49$ DVSCLEAN sequences at the $100\%$ "
           r"ratio. The first row gives absolute values; every other row gives "
           r"its signed difference from it, so for AUC a negative entry is a "
           r"loss caused by removing the component, while for SD a "
           r"\emph{positive} entry is a loss. $0.0000$ is a tie to four "
           r"decimals. Parameter counts differ because removing a component "
           r"also removes the weights that read it. All variants share one "
           r"seed; see the text on what sizes of difference we read as real.}",
           r"\label{tab:ablation}", r"\footnotesize",
           r"\setlength{\tabcolsep}{4pt}",
           r"\begin{tabular}{lccccccccc}", r"\toprule",
           r" & & \multicolumn{2}{c}{ED24 (in-domain)}"
           r" & \multicolumn{4}{c}{DVSCLEAN (zero-shot, cross-sensor)}"
           r" & \multicolumn{2}{c}{DND21 (zero-shot, same sensor)} \\",
           r"\cmidrule(lr){3-4}\cmidrule(lr){5-8}\cmidrule(lr){9-10}",
           head, r"\midrule"]
    for r in rows:
        par = f"${r['params']:.3f}$" if r["params"] is not None else "--"
        tex.append(f"{LABEL[r['name']]} & {par} & "
                   + " & ".join(cell(r, k) for _, k in TEXCOLS) + r" \\")
        if r["name"] == "full":
            tex.append(r"\midrule")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    out = "\n".join(tex)

    # ---- sign test across the six ablations, printed not typeset.
    # With one seed per variant there is no within-variant error bar, so the
    # only inferential leverage is whether the six independent removals move
    # a statistic in the same direction. Six of six in one direction has
    # p = 1/64 = 0.0156 one-sided under a null of random sign.
    print("\n  consistency across the six ablations (sign test, one seed each)")
    FAM = [("DVSCLEAN 100% mean", "dvs100", -1),
           ("DVSCLEAN 100% worst sequence", "worst100", -1),
           ("DVSCLEAN 100% cross-sequence SD", "sd100", +1),
           ("DVSCLEAN 50% mean", "dvs50", -1),
           ("DVSCLEAN 50% worst sequence", "worst50", -1),
           ("DVSCLEAN 50% cross-sequence SD", "sd50", +1),
           ("DND21 hotel-bar mean", "hotel", -1),
           ("DND21 driving mean", "driving", -1),
           ("ED24 mean", "val", -1),
           ("shot@10 Hz probe", "shot", -1)]
    TIE = 0.5e-4          # anything that prints as 0.0000 is a tie
    for label, k, worse_sign in FAM:
        raw = [r[k] - ref[k] for r in rows
               if r["name"] != "full" and r[k] is not None and ref[k] is not None]
        if not raw:
            continue
        ties = sum(1 for d in raw if abs(d) < TIE)
        ds = [d for d in raw if abs(d) >= TIE]
        n = len(ds)
        if n == 0:
            print(f"    {label:34s} all {ties} differences are ties")
            continue
        worse = sum(1 for d in ds if d * worse_sign > 0)
        p = 2.0 ** (-n) if worse in (0, n) else None
        tag = (f"p = {p:.4f} one-sided" if p is not None else "not consistent")
        note = f"   ({ties} tie{'s' if ties != 1 else ''} excluded)" if ties else ""
        print(f"    {label:34s} worse in {worse}/{n}   {tag}{note}")
    print("    NOTE: the worst-sequence and SD families are not independent --"
          "\n    one low sequence lowers the minimum and raises the SD. Treat"
          "\n    them as one finding, not two.")

    missing = [r["name"] for r in rows if r["dvs"] is None or r["dnd"] is None]
    if missing:
        print(f"\n  NOTE: transfer columns incomplete for: {', '.join(missing)}")
        print("  Run eval_ablations.py for those variants before using this table.")

    if a.tex:
        open(a.tex, "w").write(out + "\n")
        print(f"\n  wrote {a.tex}")
    else:
        print("\n" + out)


if __name__ == "__main__":
    main()
