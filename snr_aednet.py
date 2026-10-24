from __future__ import annotations

import numpy as np

# SAGE Table II -- DVSCLEAN SNR (dB), 49-scene synthetic partition
PUB_SNR = {
    "50":  {"TS": 14.52, "DWF": 19.68, "NNb": 23.56, "BAF": 23.50,
            "KNoise": 23.61, "MLPF": 23.38, "GEF": 24.17, "EBF": 24.46,
            "EventZoom": 24.57, "EDformer": 25.48, "EDmamba": 25.82,
            "AEDNet": 26.07, "SAGE": 29.60},
    "100": {"TS": 13.46, "DWF": 16.04, "NNb": 18.52, "BAF": 19.12,
            "KNoise": 20.77, "MLPF": 21.56, "GEF": 22.87, "EBF": 23.18,
            "EventZoom": 20.85, "EDformer": 22.96, "EDmamba": 23.64,
            "AEDNet": 25.09, "SAGE": 28.87},
}

# SAGE Table III -- (RER, NSR, precision, F1) on the same partition
PUB_RER = {
    "50":  {"TS": (0.953, 0.641), "DWF": (0.811, 0.917), "NNb": (0.887, 0.829),
            "BAF": (0.892, 0.836), "KNoise": (0.884, 0.851),
            "MLPF": (0.876, 0.872), "GEF": (0.902, 0.881), "EBF": (0.913, 0.887),
            "EventZoom": (0.914, 0.879), "EDformer": (0.925, 0.892),
            "EDmamba": (0.931, 0.900), "AEDNet": (0.936, 0.907),
            "SAGE": (0.962, 0.934)},
    "100": {"TS": (0.941, 0.618), "DWF": (0.768, 0.902), "NNb": (0.846, 0.801),
            "BAF": (0.858, 0.817), "KNoise": (0.862, 0.833),
            "MLPF": (0.854, 0.858), "GEF": (0.881, 0.866), "EBF": (0.892, 0.872),
            "EventZoom": (0.872, 0.861), "EDformer": (0.903, 0.881),
            "EDmamba": (0.913, 0.889), "AEDNet": (0.921, 0.896),
            "SAGE": (0.948, 0.926)},
}

DEEP = ("MLPF", "GEF", "EBF", "EventZoom", "EDformer", "EDmamba", "AEDNet", "SAGE")


# ==============================================================================
# 1. which k does the published table use?
# ==============================================================================
def reconcile(raw_ratio: dict, ks=(10.0, 20.0)) -> None:
    """raw_ratio: {"50": R/Nz, "100": R/Nz} measured from the files actually read.

    Predicts the published SNR table from the published RER/NSR table under
    each k and prints the residuals. Nothing here is fitted to our model -- it
    is entirely a statement about the source papers' internal consistency.
    """
    print("\n" + "=" * 78)
    print("SNR CONVENTION CHECK -- predicting the published SNR table from the")
    print("published RER/NSR table of the same paper, same 49 scenes")
    print("=" * 78)
    for lvl in ("50", "100"):
        if lvl not in raw_ratio:
            continue
        ratio = raw_ratio[lvl]
        print(f"\n  {lvl}% noise   measured real/noise in the raw stream = "
              f"{ratio:.4f}")
        print(f"    {'method':11s} {'published':>10}" +
              "".join(f"{'k=' + str(int(k)):>9}" for k in ks) +
              f"{'resid':>9}")
        err = {k: [] for k in ks}
        for m in PUB_SNR[lvl]:
            r, s = PUB_RER[lvl][m]
            tpfp = (r / max(1.0 - s, 1e-12)) * ratio
            cells = {k: k * np.log10(tpfp) for k in ks}
            for k in ks:
                err[k].append(abs(cells[k] - PUB_SNR[lvl][m]))
            best = max(ks)
            print(f"    {m:11s} {PUB_SNR[lvl][m]:10.2f}" +
                  "".join(f"{cells[k]:9.2f}" for k in ks) +
                  f"{cells[best] - PUB_SNR[lvl][m]:+9.2f}"
                  + ("" if m in DEEP else "   (classical)"))
        print("    " + "-" * 62)
        for k in ks:
            deep_err = [abs(k * np.log10((PUB_RER[lvl][m][0] /
                                          max(1 - PUB_RER[lvl][m][1], 1e-12))
                                         * ratio) - PUB_SNR[lvl][m])
                        for m in DEEP]
            print(f"    k={int(k):<3d} mean |resid| all {np.mean(err[k]):6.2f} dB"
                  f"   learned-only {np.mean(deep_err):6.2f} dB"
                  f"   max {np.max(deep_err):5.2f} dB")
    print("\n  The k whose learned-only residual is small is the convention that")
    print("  table used. Quote it explicitly in the paper; a bare 'SNR' on")
    print("  DVSCLEAN is ambiguous by a factor of two between paper lineages.")


# ==============================================================================
# 2. iso-RER: our model at somebody else's operating point
# ==============================================================================
def _orient(score: np.ndarray, y_noise: np.ndarray) -> np.ndarray:
    """-> a score where HIGHER means MORE LIKELY NOISE, whatever came in.

    Decided from the data, not from a flag, so a baseline whose sign convention
    is the other way round cannot silently invert the whole table.
    """
    real_mean = score[y_noise == 0].mean()
    noise_mean = score[y_noise == 1].mean()
    return score if noise_mean >= real_mean else -score


def threshold_for_rer(score: np.ndarray, y_noise: np.ndarray,
                      target_rer: float) -> float:
    """Threshold t such that keeping {score < t} retains exactly `target_rer`
    of the real events. Exact by construction: it is the target_rer quantile
    of the real events' own scores."""
    real = score[y_noise == 0]
    return float(np.quantile(real, np.clip(target_rer, 0.0, 1.0)))


def at_rer(score: np.ndarray, y_noise: np.ndarray, target_rer: float,
           k: float) -> dict:
    s = _orient(np.asarray(score, np.float64), y_noise)
    thr = threshold_for_rer(s, y_noise, target_rer)
    kept = s < thr
    tp = int((kept & (y_noise == 0)).sum())
    fp = int((kept & (y_noise == 1)).sum())
    fn = int((~kept & (y_noise == 0)).sum())
    tn = int((~kept & (y_noise == 1)).sum())
    rer = tp / max(tp + fn, 1)
    nsr = tn / max(tn + fp, 1)
    snr = k * np.log10(tp / fp) if fp > 0 and tp > 0 else float("inf")
    return dict(thr=thr, RER=rer, NSR=nsr, SNR=snr, TP=tp, FP=fp)


def iso_rer_report(summary: dict, k: float = 20.0,
                   methods=("EDformer", "EDmamba", "AEDNet", "SAGE")) -> None:
    """summary: {"50": (probs, base, y), "100": (...)} as eval_dvsclean_snr builds it.

    For each competitor, runs our model at the threshold where its RER equals
    that competitor's published RER, and reports the SNR there. Equal retention
    means the only thing left varying is noise suppression, so the comparison
    does not depend on a threshold anyone picked.
    """
    print("\n" + "=" * 78)
    print(f"ISO-RER COMPARISON  (SNR = {int(k)} * log10(TP/FP), matched retention)")
    print("=" * 78)
    print("  Our model is thresholded so that its real-event retention EQUALS the")
    print("  competitor's published RER. At equal RER a higher SNR is strictly")
    print("  better noise suppression -- no operating point has been chosen to")
    print("  favour either side.\n")

    for lvl in ("50", "100"):
        if lvl not in summary:
            continue
        probs, _base, y = summary[lvl]
        y = np.asarray(y)
        print(f"  --- {lvl}% noise ---")
        print(f"    {'competitor':11s} {'their RER':>10} {'their NSR':>10} "
              f"{'their SNR':>10} | {'our NSR':>9} {'our SNR':>9} {'delta':>8}")
        for m in methods:
            if m not in PUB_RER[lvl]:
                continue
            r_pub, s_pub = PUB_RER[lvl][m]
            snr_pub = PUB_SNR[lvl][m]
            ours = at_rer(probs, y, r_pub, k)
            d = ours["SNR"] - snr_pub
            print(f"    {m:11s} {r_pub:10.3f} {s_pub:10.3f} {snr_pub:10.2f} | "
                  f"{ours['NSR']:9.3f} {ours['SNR']:9.2f} {d:+8.2f}")
        print()
    print("  Report the delta column. It is the one quantity here that is not")
    print("  sensitive to the threshold, and it survives the k ambiguity as a")
    print("  RATIO even if the absolute dB values do not: doubling k doubles")
    print("  both columns.")
