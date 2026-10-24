from __future__ import annotations

import os

import numpy as np

# Bumped whenever the public signature changes. Consumers check it, because a
# stale copy of this file on the path otherwise surfaces as a bare TypeError
# deep inside a long evaluation run.
__version__ = 4

T, X, Y, P, L = 0, 1, 2, 3, 4

_FIELDS = {
    "t": ("t", "ts", "time", "timestamp", "timestamps"),
    "x": ("x", "xs"),
    "y": ("y", "ys"),
    "p": ("p", "ps", "pol", "polarity", "polarities"),
    "l": ("l", "label", "labels", "gt", "noise", "y_label"),
}
_TABLE_KEYS = ("events", "event", "data", "raw", "ev")
_UNIT_TO_US = {"us": 1.0, "ms": 1e3, "s": 1e6}


def _leaf_match(keys, candidates):
    """Match on the LEAF name so nested paths (events/timestamp) are found."""
    by_leaf = {}
    for k in sorted(keys, key=lambda q: (q.count("/"), len(q))):
        by_leaf.setdefault(k.rsplit("/", 1)[-1].lower(), k)
    for c in candidates:
        if c in by_leaf:
            return by_leaf[c]
    return None


def read_raw(path: str, with_names: bool = False):
    """-> dict canonical letter -> raw 1-D array, before ANY conversion.

    With with_names=True also returns the source dataset name chosen for each
    letter.  That mapping is a guess based on name matching, and a wrong guess
    is invisible in the values alone, so diagnostics must be able to show it.

    Keeping the raw arrays is the point: their dtype is what decides the time
    unit, and the dtype is exactly what a premature int cast destroys.
    """
    ext = os.path.splitext(path)[1].lower()

    if ext in (".h5", ".hdf5", ".hdf"):
        import h5py
        with h5py.File(path, "r") as f:
            flat = {}
            f.visititems(lambda n, o: flat.__setitem__(n, o)
                         if isinstance(o, h5py.Dataset) else None)
            if not flat:
                raise ValueError(f"{path}: no datasets inside")

            tbl = _leaf_match(flat, _TABLE_KEYS)
            if tbl is not None:
                probe = np.asarray(flat[tbl])
                if probe.dtype.names:                       # compound dtype
                    names = {n.lower(): n for n in probe.dtype.names}
                    cols, src = {}, {}
                    for c in "txypl":
                        nm = _leaf_match(names, _FIELDS[c])
                        if nm is not None:
                            cols[c] = np.asarray(probe[names[nm]]).reshape(-1)
                            src[c] = f"{tbl}.{names[nm]}"
                    return (cols, src, sorted(names.values())) if with_names else cols
                if probe.ndim == 2 and probe.shape[1] >= 4:  # plain table
                    cols = {c: probe[:, j]
                            for j, c in enumerate("txypl"[:probe.shape[1]])}
                    src = {c: f"{tbl}[:,{j}]" for j, c in enumerate(cols)}
                    return (cols, src, [tbl]) if with_names else cols

            cols, src = {}, {}
            for c in "txypl":
                nm = _leaf_match(flat, _FIELDS[c])
                if nm is not None:
                    cols[c] = np.asarray(flat[nm]).reshape(-1)
                    src[c] = nm
            return (cols, src, sorted(flat)) if with_names else cols

    if ext in (".npy", ".npz"):
        if ext == ".npy":
            arr = np.load(path)
        else:
            z = np.load(path)
            arr = np.asarray(z[list(z.keys())[0]])
        cols = {c: arr[:, j] for j, c in enumerate("txypl"[:arr.shape[1]])}
        src = {c: f"column {j}" for j, c in enumerate(cols)}
        return (cols, src, ["<array>"]) if with_names else cols

    import pandas as pd
    with open(path) as fh:
        first = fh.readline()
    tok = first.replace(",", " ").split()

    def _num(v):
        try:
            float(v)
            return True
        except ValueError:
            return False
    hdr = 0 if (tok and not all(_num(v) for v in tok)) else None
    arr = pd.read_csv(path, sep=r"[\s,]+", header=hdr, engine="python").to_numpy()
    cols = {c: arr[:, j] for j, c in enumerate("txypl"[:arr.shape[1]])}
    src = {c: f"column {j}" for j, c in enumerate(cols)}
    return (cols, src, ["<text table>"]) if with_names else cols


def infer_time_unit(t_raw: np.ndarray) -> str:
    """-> 'us' | 'ms' | 's'.

    Span alone cannot decide this: a 900 ms clip has a raw span of 0.9, 900 or
    900000 depending on the unit, and a 900 SECOND clip has a span of 900 in
    seconds -- identical to the 900 ms clip measured in milliseconds.  What
    breaks the tie is the implied event rate, which for any real event camera
    sits in roughly 1e3 to 1e8 events per second.  We try each unit and keep
    the one whose implied rate is closest, in log space, to a typical 3e5.

    Two shortcuts come first because they are near-certain:
      * an integer dtype is microseconds, the event-data convention;
      * float values that are all integral are microseconds written as float.
    """
    if np.issubdtype(t_raw.dtype, np.integer):
        return "us"

    finite = t_raw[np.isfinite(t_raw)]
    if finite.size == 0:
        return "us"
    span = float(np.nanmax(finite) - np.nanmin(finite))
    if span <= 0:
        return "us"
    if np.all(finite == np.rint(finite)):
        return "us"

    n = finite.size
    target = np.log10(3e5)
    best, best_cost = "us", np.inf
    for unit, mult in _UNIT_TO_US.items():
        rate = n / (span * mult / 1e6)          # events per second
        if rate <= 0:
            continue
        cost = abs(np.log10(rate) - target)
        if cost < best_cost:
            best, best_cost = unit, cost
    return best


def load_events(path: str, noise_label: int = 1, time_unit: str = "auto",
                polarity: str = "auto", verbose: bool = False) -> np.ndarray:
    """-> (n,5) int64 [t_us, x, y, p, label]; label 1 == noise; time sorted.

    Raises rather than returning a silently degenerate array: a timestamp
    column that collapsed to a handful of distinct values cannot support any
    spatiotemporal method, and failing loudly here is the whole point.
    """
    cols = read_raw(path)
    for c in "txyp":
        if c not in cols:
            raise ValueError(f"{path}: no '{c}' column found (got {sorted(cols)})")

    t_raw = np.asarray(cols["t"])
    unit = infer_time_unit(t_raw) if time_unit == "auto" else time_unit
    if unit not in _UNIT_TO_US:
        raise ValueError(f"time_unit must be auto/us/ms/s, got {time_unit!r}")

    n = len(t_raw)
    out = np.empty((n, 5), np.float64)
    tf = t_raw.astype(np.float64)
    out[:, T] = (tf - np.nanmin(tf)) * _UNIT_TO_US[unit]
    for j, c in zip((X, Y), "xy"):
        out[:, j] = np.asarray(cols[c]).astype(np.float64)

    # ---- polarity ----------------------------------------------------------
    # A polarity column has two states.  DVSCLEAN's events/polarity does not:
    # it holds a per-event uniform random draw in (0, 1) -- noise events have
    # mean 0.4981 and median 0.4984, which is uniform[0,1] to three decimals --
    # and it weakly tracks the label (AUC 0.564 on its own).  That is an
    # artefact of how the file was generated, not a property of the sensor, so
    # feeding it to a model hands over a little free label information that
    # will not transfer anywhere.  When the column is not two-state we drop it
    # and say so, rather than letting `p > 0` silently turn it into a constant.
    p_raw = np.asarray(cols["p"]).astype(np.float64)
    uniq = np.unique(p_raw[np.isfinite(p_raw)])
    pol_note = ""
    if polarity == "auto":
        mode = "binary" if len(uniq) <= 2 else "constant"
    else:
        mode = polarity
    if mode == "binary":
        if len(uniq) == 2:
            out[:, P] = (p_raw > 0.5 * (uniq[0] + uniq[-1])).astype(np.float64)
        else:
            out[:, P] = 0.0
        pol_note = f"binary from {len(uniq)} distinct"
    elif mode == "threshold":
        out[:, P] = (p_raw > 0.5).astype(np.float64)
        pol_note = "thresholded at 0.5 (NOT recommended for a non-polarity column)"
    elif mode == "raw":
        out[:, P] = p_raw
        pol_note = "raw values passed through (caller beware)"
    else:                                     # constant
        out[:, P] = 0.0
        pol_note = (f"DROPPED to a constant: source had {len(uniq)} distinct "
                    f"values, which is not a polarity")
    out[:, L] = (np.asarray(cols["l"]).astype(np.float64)
                 if "l" in cols else 0.0)

    keep = ~np.isnan(out).any(axis=1)
    out = out[keep]
    if len(out) == 0:
        raise ValueError(f"{path}: every row was dropped as non-finite")

    ev = np.empty(out.shape, np.int64)
    ev[:, T] = np.rint(out[:, T])                  # round, never truncate
    ev[:, X] = np.rint(out[:, X])
    ev[:, Y] = np.rint(out[:, Y])
    ev[:, P] = np.rint(out[:, P]) if mode == "raw" else (out[:, P] > 0.5)
    ev[:, L] = (out[:, L] == noise_label)

    n_unique = len(np.unique(ev[:, T]))
    if n_unique < max(2, len(ev) // 100):
        raise ValueError(
            f"{path}: after conversion the timestamp column has only "
            f"{n_unique} distinct values across {len(ev)} events. The time "
            f"unit was read as '{unit}' from dtype {t_raw.dtype} with raw span "
            f"{float(np.nanmax(t_raw) - np.nanmin(t_raw)):.6g}. Pass an "
            f"explicit time_unit; a collapsed timestamp column silently turns "
            f"every spatiotemporal method into a constant score.")

    if not np.all(np.diff(ev[:, T]) >= 0):
        ev = ev[np.argsort(ev[:, T], kind="stable")]

    if verbose:
        dur_s = (ev[:, T].max() - ev[:, T].min()) / 1e6
        print(f"    {os.path.basename(path)}: {len(ev)} events, "
              f"t dtype {t_raw.dtype} read as '{unit}', "
              f"{dur_s * 1000:.1f} ms, {len(ev) / max(dur_s, 1e-9):.0f} ev/s, "
              f"noise {ev[:, L].mean():.3f}; polarity {pol_note}")
    return ev


def probe_column(path: str, col: str = "p") -> None:
    """Work out what an unexpected column actually contains.

    A column can only be interpreted by its RELATIONSHIPS: to time, to the
    label, to the event index.  Raw min/max/distinct cannot distinguish a
    normalised timestamp from a confidence weight from a mislabelled field.
    """
    cols, src, _ = read_raw(path, with_names=True)
    if col not in cols:
        print(f"  no '{col}' column"); return
    v = np.asarray(cols[col]).astype(np.float64)
    t = np.asarray(cols["t"]).astype(np.float64)
    lab = np.asarray(cols["l"]).astype(np.float64) if "l" in cols else None
    n = len(v)
    print(f"\n  --- probing '{col}' (from {src.get(col)}) ---")
    print(f"    quantiles 0/1/25/50/75/99/100: " +
          " ".join(f"{q:.4g}" for q in np.percentile(v, [0, 1, 25, 50, 75, 99, 100])))
    print(f"    exactly 0: {(v == 0).sum()}   exactly 1: {(v == 1).sum()}   "
          f"in (0,1): {((v > 0) & (v < 1)).sum()}   negative: {(v < 0).sum()}")

    from scipy.stats import spearmanr
    idx = np.arange(n)
    print(f"    Spearman vs timestamp   : {spearmanr(v, t).statistic:+.4f}")
    print(f"    Spearman vs event index : {spearmanr(v, idx).statistic:+.4f}")
    tn = (t - t.min()) / max(t.max() - t.min(), 1e-12)
    print(f"    max |v - normalised t|  : {np.abs(v - tn).max():.4g}   "
          f"(near 0 would mean v IS the normalised timestamp)")
    if lab is not None:
        for name, m in (("label 0 (signal)", lab == 0), ("label 1 (noise)", lab == 1)):
            if m.sum():
                print(f"    {name}: mean {v[m].mean():.4g}  median "
                      f"{np.median(v[m]):.4g}  n={m.sum()}")
        try:
            from sklearn.metrics import roc_auc_score
            print(f"    AUC of this column alone as a noise score: "
                  f"{max(roc_auc_score(lab, v), roc_auc_score(lab, -v)):.4f}")
        except Exception:
            pass
    print("    READ: |v - normalised t| near 0 -> it is a rescaled timestamp.\n"
          "          Spearman vs index near +1 -> it is monotone in arrival order.\n"
          "          A high standalone AUC -> it leaks the label and must not be\n"
          "          fed to a model as a feature.")


def describe(path: str, time_unit: str = "auto") -> None:
    """Print what is actually in a file, before and after conversion."""
    print(f"\n=== {path} ===")
    cols, src, every = read_raw(path, with_names=True)
    print(f"  datasets in file: {every}")
    print(f"  mapping chosen by name matching:")
    for c in "txypl":
        if c not in cols:
            print(f"    {c} <- (none found)")
            continue
        a = np.asarray(cols[c])
        print(f"    {c} <- {src.get(c, '?'):24s} dtype={str(a.dtype):9s} "
              f"n={a.size:9d} min={np.nanmin(a):<14.6g} max={np.nanmax(a):<14.6g} "
              f"distinct={len(np.unique(a)) if a.size < 4_000_000 else '?'}")
    if "p" in cols:
        a = np.asarray(cols["p"]); u = np.unique(a)
        if len(u) > 2:
            print(f"\n  !! the column mapped to POLARITY has {len(u)} distinct "
                  f"values, not 2.\n     It is probably not polarity, or the "
                  f"name match is wrong. First 8: {u[:8]}\n     Old loaders did "
                  f"(p > 0), which collapses all of these to a single value.")
    if "l" in cols:
        u = np.unique(np.asarray(cols["l"]))
        print(f"  label column distinct values: {u[:8]}")

    t_raw = np.asarray(cols["t"])
    unit = infer_time_unit(t_raw) if time_unit == "auto" else time_unit
    span = float(np.nanmax(t_raw) - np.nanmin(t_raw))
    print(f"\n  time unit: read as '{unit}' (raw span {span:.6g}, "
          f"dtype {t_raw.dtype})")
    print(f"  NAIVE int64 cast would give "
          f"{len(np.unique(t_raw.astype(np.int64)))} distinct timestamps "
          f"-- this is the bug to watch for")
    try:
        ev = load_events(path, time_unit=time_unit)
        dur_s = (ev[:, T].max() - ev[:, T].min()) / 1e6
        print(f"  loaded OK: {len(ev)} events, {dur_s * 1000:.2f} ms, "
              f"{len(ev) / max(dur_s, 1e-9):.0f} ev/s, "
              f"{len(np.unique(ev[:, T]))} distinct timestamps")
        print(f"  x in [{ev[:, X].min()}, {ev[:, X].max()}], "
              f"y in [{ev[:, Y].min()}, {ev[:, Y].max()}], "
              f"polarity {np.unique(ev[:, P]).tolist()}, "
              f"noise fraction {ev[:, L].mean():.4f}")
        print("  SANITY: is the duration plausible for this clip, and is the "
              "event rate\n          in the 1e5-1e7 range? If not, the unit is "
              "wrong -- pass --time_unit.")
    except ValueError as e:
        print(f"  REJECTED: {e}")


def self_check() -> None:
    """Confirm this file is the version the other scripts expect."""
    import inspect
    sig = inspect.signature(load_events)
    print(f"event_io __version__ = {__version__}")
    print(f"load_events{sig}")
    need = {"noise_label", "time_unit", "polarity", "verbose"}
    have = set(sig.parameters) - {"path"}
    missing = need - have
    if missing:
        raise SystemExit(f"  STALE FILE: missing {sorted(missing)}")
    print("  OK -- this copy supports time_unit and polarity")


if __name__ == "__main__":
    import argparse
    import glob
    ap = argparse.ArgumentParser(description="Inspect event files")
    ap.add_argument("paths", nargs="*", help="files or globs")
    ap.add_argument("--time_unit", default="auto",
                    choices=["auto", "us", "ms", "s"])
    ap.add_argument("--probe", default=None,
                    help="analyse one column in depth, e.g. --probe p")
    ap.add_argument("--self_check", action="store_true",
                    help="print the version and signature, then exit")
    args = ap.parse_args()
    if args.self_check:
        self_check(); raise SystemExit(0)
    files = []
    for p in args.paths:
        files += sorted(glob.glob(p)) or [p]
    for f in files[:10]:
        describe(f, args.time_unit)
        if args.probe:
            probe_column(f, args.probe)
