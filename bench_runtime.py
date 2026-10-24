from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)) or "."
if HERE not in sys.path:
    sys.path.insert(0, HERE)

ED24_SENSOR = (346, 260)
DVSCLEAN_SENSOR = (1280, 720)


# ----------------------------------------------------------------- the windows

def _windows_from(ev, N, k):
    """Up to k consecutive N-event windows as (feat, meta) tensor pairs."""
    n = ev.shape[0] // N
    if n == 0:
        raise SystemExit(f"  file has {ev.shape[0]} events, need at least {N}")
    import ed24_data as D
    out = []
    for i in range(min(k, n)):
        w = np.ascontiguousarray(ev[i * N:(i + 1) * N])
        feat, meta = D.normalize_window(w)
        out.append((torch.from_numpy(feat).unsqueeze(0),
                    torch.from_numpy(meta).unsqueeze(0)))
    return out


def load_real_windows(a, N):
    """-> ([(feat, meta), ...], source label). Preprocessed exactly as the
    evaluation scripts do."""
    import ed24_data as D
    D.set_sensor_geometry(*a.sensor)

    if a.dvsclean_root or (a.file and a.from_dvsclean):
        from eval_dvsclean_snr import find_files, load_any
        if a.file:
            path = a.file
        else:
            files = (find_files(a.dvsclean_root, "100")
                     or find_files(a.dvsclean_root, "50"))
            if not files:
                raise SystemExit(f"  no DVSCLEAN files under {a.dvsclean_root}")
            path = files[0]
        ev = load_any(path, a.cols, a.noise_label, a.h5_events_key,
                      a.h5_label_key, time_unit=a.time_unit,
                      polarity=a.polarity, verbose=a.verbose_load)
        if a.force_polarity is not None:
            ev = np.array(ev, copy=True)
            if a.force_polarity == 2:
                import zlib
                rng = np.random.default_rng(
                    zlib.crc32(os.path.basename(path).encode()))
                ev[:, 3] = rng.integers(0, 2, size=len(ev))
            else:
                ev[:, 3] = a.force_polarity
    else:
        if a.file:
            path = a.file
        else:
            trips = D.list_ed24_files(a.ed24_root)
            if not trips:
                raise SystemExit(f"  no ED24 files under {a.ed24_root}")
            # middle of the voltage sweep rather than the first entry, which is
            # the lowest noise level and the cheapest descriptor.
            path = trips[len(trips) // 2][2]
        ev = D.load_events_cached(path, a.cache_dir)

    return _windows_from(ev, N, a.windows), os.path.basename(path)


def synthetic_windows(a, N, dur_us=2.0e4):
    import ed24_data as D
    W, H = a.sensor
    D.set_sensor_geometry(W, H)
    rng = np.random.default_rng(0)
    total = N * a.windows
    ev = np.empty((total, 4), dtype=np.float32)
    ev[:, 0] = np.sort(rng.uniform(0, dur_us * a.windows, total))
    ev[:, 1] = rng.integers(0, W, total)
    ev[:, 2] = rng.integers(0, H, total)
    ev[:, 3] = rng.integers(0, 2, total)
    return _windows_from(ev, N, a.windows), "SYNTHETIC"


# ------------------------------------------------------------------- the clock

def _ctx(amp):
    return torch.autocast("cuda", dtype=torch.float16) if amp \
        else torch.autocast("cuda", enabled=False)


def time_calls(fn, nwin, warmup, iters, amp):
    """fn(i) runs window i. -> median / mean / sd / p5 / p95 GPU milliseconds."""
    with torch.inference_mode(), _ctx(amp):
        for i in range(warmup):
            fn(i % nwin)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    ms = []
    with torch.inference_mode(), _ctx(amp):
        for i in range(iters):
            start.record()
            fn(i % nwin)
            end.record()
            torch.cuda.synchronize()
            ms.append(start.elapsed_time(end))
    v = np.asarray(ms)
    return dict(median=float(np.median(v)), mean=float(v.mean()),
                std=float(v.std(ddof=1)), p5=float(np.percentile(v, 5)),
                p95=float(np.percentile(v, 95)), n=iters, windows=nwin)


def peak_mb_of(fn, nwin, amp):
    """Worst-case peak over the loaded windows, and its increment."""
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated() / 2**20
    with torch.inference_mode(), _ctx(amp):
        for i in range(nwin):
            fn(i)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 2**20
    return peak, peak - before


# ------------------------------------------------- network-only and the stages

class _Capture:
    """Grabs the tensors entering the first block, so the learned part can be
    replayed without redoing the descriptor and the graphs."""

    def __init__(self, model):
        self.args = None
        self.cond = None
        self.h = [model.blocks[0].register_forward_pre_hook(self._blk)]
        if hasattr(model, "cond"):
            self.h.append(model.cond.register_forward_hook(self._cond))

    def _blk(self, mod, args):
        if self.args is None:
            self.args = tuple(args)

    def _cond(self, mod, args, out):
        if self.cond is None:
            self.cond = out

    def close(self):
        for h in self.h:
            h.remove()


def make_network_only(model, wins, amp):
    """-> fn(i) running only blocks + head on cached per-window inputs."""
    use_film = getattr(model.cfg, "use_film", True)
    cached = []
    for feat, meta in wins:
        cap = _Capture(model)
        try:
            with torch.inference_mode(), _ctx(amp):
                model(feat, meta)
        finally:
            cap.close()
        if cap.args is None:
            return None
        cached.append((cap.args, cap.cond))

    def run(i):
        (x, c_fine, c_coarse, pol, nbr_f, nbr_c, _film), cond = cached[i]
        for j, blk in enumerate(model.blocks):
            if use_film:
                scale, shift = model.film[j](cond).chunk(2, dim=-1)
            else:
                z = x.new_zeros(x.shape[0], model.cfg.hidden_dim)
                scale, shift = z, z
            x = blk(x, c_fine, c_coarse, pol, nbr_f, nbr_c, (scale, shift))
        return model.head(model.norm(x)).squeeze(-1)

    return run


def stage_breakdown(model, wins, amp, iters):
    """Split end-to-end into descriptor+stem / graph / blocks+head / other.

    Hooks on the real modules and a wrapper around the real physical_knn (and
    index_knn, for A2), so nothing here duplicates model code. Restored in a
    finally block, because leaving a timed wrapper installed would silently
    slow every later measurement.
    """
    import edn_ablate as A
    import edn_model as M

    acc = {k: [] for k in ("total", "desc", "graph", "net")}
    ev, gspan = {}, []

    def mk():
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        return e

    def pre_desc(mod, args):
        ev["t0"] = mk()

    def post_stem(mod, args, out):
        ev["desc_end"] = mk()

    def pre_blocks(mod, args):
        ev["net_start"] = mk()

    def post_head(mod, args, out):
        ev["net_end"] = mk()

    hs = [model.density.register_forward_pre_hook(pre_desc),
          model.stem.register_forward_hook(post_stem),
          model.blocks[0].register_forward_pre_hook(pre_blocks),
          model.head.register_forward_hook(post_head)]

    def timed(fn):
        def wrapper(*args, **kw):
            s = mk()
            out = fn(*args, **kw)
            gspan.append((s, mk()))
            return out
        return wrapper

    real_knn = M.physical_knn
    real_idx = A.index_knn
    # EDNetAbl.forward resolves these in edn_ablate's globals; edn_model is
    # patched too so that the unablated EDNet path is covered as well.
    A.physical_knn = M.physical_knn = timed(real_knn)
    A.index_knn = timed(real_idx)

    nwin = len(wins)
    try:
        with torch.inference_mode(), _ctx(amp):
            for i in range(min(20, iters)):
                model(*wins[i % nwin])
            torch.cuda.synchronize()
            for i in range(iters):
                ev.clear()
                gspan.clear()
                t0 = mk()
                model(*wins[i % nwin])
                t1 = mk()
                torch.cuda.synchronize()
                acc["total"].append(t0.elapsed_time(t1))
                acc["desc"].append(ev["t0"].elapsed_time(ev["desc_end"]))
                acc["graph"].append(
                    sum(s.elapsed_time(e) for s, e in gspan))
                acc["net"].append(
                    ev["net_start"].elapsed_time(ev["net_end"]))
    finally:
        for h in hs:
            h.remove()
        A.physical_knn = M.physical_knn = real_knn
        A.index_knn = real_idx

    out = {k: float(np.median(v)) for k, v in acc.items()}
    out["other"] = out["total"] - out["desc"] - out["graph"] - out["net"]
    return out


# --------------------------------------------------------------------- reports

def write_tex(path, rows, gpu, iters, windows):
    card = gpu.replace("NVIDIA ", "")
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Inference efficiency of \ourmethod{} on an NVIDIA " + card
         + r", batch size one. Latency is the median over $" + str(iters)
         + r"$ timed forward passes over $" + str(windows) + r"$ event "
         r"windows, and covers the whole inference path: the multi-scale "
         r"density descriptor, both physical-time neighborhood graphs, the "
         r"four dual-timescale blocks and the prediction head. Throughput is "
         r"$N$ divided by that latency.}",
         r"\label{tab:runtime}", r"\small",
         r"\begin{tabular}{lccc}", r"\toprule",
         r"$N$ & Latency (ms) & Throughput (Mev/s) & Peak GPU (MB) \\",
         r"\midrule"]
    for r in rows:
        L.append(f"${r['N']:,}$".replace(",", "{,}")
                 + f" & ${r['e2e']['median']:.2f}$"
                 + f" & ${r['throughput_mevs']:.2f}$"
                 + f" & ${r['peak_mb']:.0f}$ \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(path, "w", encoding="utf-8").write("\n".join(L) + "\n")


def sentence(row, gpu):
    card = gpu.replace("NVIDIA ", "")
    b = row.get("batch", 1)
    if b > 1:
        return (f"On an NVIDIA {card}, batching {b} windows of {row['N']:,} "
                f"events costs {row['e2e']['median']:.1f}~ms, i.e. "
                f"{row['e2e']['median']/b:.1f}~ms per window and "
                f"{row['throughput_mevs']:.2f}~Mev/s, with a peak GPU "
                f"allocation of {row['peak_mb']:.0f}~MB. The timing covers "
                f"the density descriptor, both physical-time graphs, all "
                f"four dual-timescale blocks and the prediction head.")
    return (f"On an NVIDIA {card}, end-to-end inference for a "
            f"{row['N']:,}-event window takes "
            f"{row['e2e']['median']:.1f}~ms, corresponding to "
            f"{row['throughput_mevs']:.2f}~Mev/s, with a peak GPU allocation "
            f"of {row['peak_mb']:.0f}~MB. The timing covers the density "
            f"descriptor, both physical-time graphs, all four dual-timescale "
            f"blocks and the prediction head.")


# ---------------------------------------------------------------------- driver

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--Ns", nargs="+", type=int, default=[4096, 10240, 20480])
    ap.add_argument("--ed24_root", default=None)
    ap.add_argument("--dvsclean_root", default=None)
    ap.add_argument("--file", default=None,
                    help="benchmark one explicit event file instead of "
                         "picking from a root")
    ap.add_argument("--from_dvsclean", action="store_true",
                    help="with --file: read it with the DVSCLEAN loader")
    ap.add_argument("--synthetic", action="store_true",
                    help="benchmark uniform random events (NOT for the paper)")
    ap.add_argument("--sensor", nargs=2, type=int, default=None,
                    metavar=("W", "H"),
                    help="default 346 260 for ED24, 1280 720 for DVSCLEAN")
    ap.add_argument("--windows", type=int, default=8,
                    help="distinct event windows to cycle through")
    ap.add_argument("--batch", type=int, default=1,
                    help="windows per forward pass. 1 is the deployment "
                         "latency; larger amortizes the fixed per-window "
                         "cost and is what offline denoising would use")
    ap.add_argument("--cache_dir", default="./cache")
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--stage_iters", type=int, default=100)
    ap.add_argument("--no_amp", action="store_true",
                    help="time in fp32; by default AMP fp16 is on, matching "
                         "the evaluation scripts")
    ap.add_argument("--no_network_only", action="store_true")
    ap.add_argument("--no_stages", action="store_true")
    # DVSCLEAN loader flags, same names and defaults as eval_dvsclean_snr.py
    ap.add_argument("--cols", default="txypl")
    ap.add_argument("--time_unit", default="s")
    ap.add_argument("--force_polarity", type=int, default=2)
    ap.add_argument("--polarity", default="auto")
    ap.add_argument("--noise_label", type=int, default=1)
    ap.add_argument("--h5_events_key", default=None)
    ap.add_argument("--h5_label_key", default=None)
    ap.add_argument("--verbose_load", action="store_true")
    ap.add_argument("--json", default="runtime.json")
    ap.add_argument("--tex", default=None)
    a = ap.parse_args()

    if not (a.ed24_root or a.dvsclean_root or a.file or a.synthetic):
        raise SystemExit(
            "  pass --ed24_root, --dvsclean_root or --file for a real window, "
            "or --synthetic to accept unrepresentative timings")
    if a.sensor is None:
        a.sensor = list(DVSCLEAN_SENSOR if (a.dvsclean_root or a.from_dvsclean)
                        else ED24_SENSOR)
    if a.windows < 1 or a.batch < 1:
        raise SystemExit("  --windows and --batch must be at least 1")
    # a batch of B needs at least B distinct windows to be built from, and at
    # least two batches so the timed loop is not re-running one tensor.
    a.windows = max(a.windows, 2 * a.batch)
    if not torch.cuda.is_available():
        raise SystemExit("  no CUDA device visible; these numbers are "
                         "meaningless on CPU")

    dev = torch.device("cuda")
    gpu = torch.cuda.get_device_name(0)
    amp = not a.no_amp
    print(f"  device    : {gpu}")
    print(f"  precision : {'AMP fp16' if amp else 'fp32'}")
    print(f"  torch     : {torch.__version__}  cuda {torch.version.cuda}")
    print(f"  sensor    : {a.sensor[0]} x {a.sensor[1]}")
    if a.synthetic:
        print("  WARNING: synthetic uniform events. The descriptor hash and "
              "the graph\n           are data-dependent, so these timings are "
              "UNREPRESENTATIVE.")

    from edn_ablate import EDNetAbl, config_from_checkpoint
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    print(f"  checkpoint: {a.ckpt}  (epoch {ck.get('epoch', '?')})")

    rows = []
    for N in a.Ns:
        cfg = config_from_checkpoint(ck, seq_len=N)
        model = EDNetAbl(cfg).to(dev)
        model.load_state_dict(ck["model"])
        model.eval()
        params = sum(p.numel() for p in model.parameters())

        wins, src = (synthetic_windows(a, N) if a.synthetic
                     else load_real_windows(a, N))
        wins = [(f.to(dev), m.to(dev)) for f, m in wins]
        durs = [float(m[0, 0]) / 1e3 for _, m in wins]
        nloaded = len(wins)
        if a.batch > 1:
            if nloaded < a.batch:
                raise SystemExit(f"  only {nloaded} windows of {N} events "
                                 f"available; --batch {a.batch} needs more")
            grouped = []
            for s in range(0, nloaded - a.batch + 1, a.batch):
                chunk = wins[s:s + a.batch]
                grouped.append((torch.cat([f for f, _ in chunk], 0),
                                torch.cat([m for _, m in chunk], 0)))
            wins = grouped
        nwin = len(wins)
        per_call = N * a.batch

        tag = "   [UNREPRESENTATIVE: synthetic]" if a.synthetic else ""
        print(f"\n  === N = {N:,}   {src}   {nloaded} window(s), "
              f"batch {a.batch}{tag} ===")
        print(f"  parameters      : {params/1e6:.3f} M")
        print(f"  window duration : {min(durs):.2f} - {max(durs):.2f} ms")

        e2e = time_calls(lambda i: model(*wins[i]), nwin,
                         a.warmup, a.iters, amp)
        peak, inc = peak_mb_of(lambda i: model(*wins[i]), nwin, amp)
        thr = per_call / (e2e["median"] / 1e3) / 1e6

        label = "end-to-end" if a.batch == 1 else f"per batch of {a.batch}"
        print(f"  {label:15s} : {e2e['median']:8.3f} ms   median   "
              f"[p5 {e2e['p5']:.3f}, p95 {e2e['p95']:.3f}]")
        print(f"                    {e2e['mean']:8.3f} ms   mean "
              f"+/- {e2e['std']:.3f}")
        if a.batch > 1:
            print(f"  per window      : "
                  f"{e2e['median']/a.batch:8.3f} ms   amortized")
        print(f"  throughput      : {thr:8.3f} Mev/s")
        print(f"  peak GPU memory : {peak:8.1f} MB   "
              f"(increment over resident model {inc:.1f} MB)")

        net = None
        if not a.no_network_only:
            fn = make_network_only(model, wins, amp)
            if fn is None:
                print("  network-only    : unavailable (no block inputs "
                      "captured)")
            else:
                net = time_calls(fn, nwin, a.warmup, a.iters, amp)
                print(f"  network only    : {net['median']:8.3f} ms   "
                      f"({100*net['median']/e2e['median']:.0f}% of end-to-end)")

        st = None
        if not a.no_stages:
            st = stage_breakdown(model, wins, amp, a.stage_iters)
            print(f"  stage split     : instrumented pass, median of "
                  f"{a.stage_iters}, total {st['total']:.3f} ms")
            for k, lab in (("desc", "density descriptor + stem"),
                           ("graph", "physical-time graphs"),
                           ("net", "dual-timescale blocks + head"),
                           ("other", "rescaling, conditioner, coords")):
                print(f"      {lab:32s} {st[k]:8.3f} ms   "
                      f"{100*st[k]/st['total']:5.1f}%")
            if st["other"] < -0.05 * st["total"]:
                print("      NOTE: residual is negative beyond rounding; the "
                      "hook placement no longer matches edn_model.forward.")

        rows.append(dict(N=N, params=params, source=src, synthetic=a.synthetic,
                         gpu=gpu, amp=amp, sensor=list(a.sensor),
                         batch=a.batch, events_per_call=per_call,
                         n_windows=nloaded, n_batches=nwin,
                         e2e=e2e, network=net,
                         throughput_mevs=thr, peak_mb=peak, inc_mb=inc,
                         stages=st))
        del model, wins
        torch.cuda.empty_cache()

    with open(a.json, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=1)
    print(f"\n  wrote {a.json}")

    if a.tex:
        write_tex(a.tex, rows, gpu, a.iters, a.windows)
        print(f"  wrote {a.tex}")

    main_row = next((r for r in rows if r["N"] == 10240), rows[0])
    print("\n  sentence for the text:\n")
    print("    " + sentence(main_row, gpu))
    if a.synthetic:
        print("\n  DO NOT PUT THE ABOVE IN THE PAPER: synthetic window.")


if __name__ == "__main__":
    main()
