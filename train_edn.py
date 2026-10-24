from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from ed24_data import (ED24Windows, list_ed24_files, noise_fraction_report,
                       scene_split, verify_label_convention)
from edn_model import EDNConfig, EDNet, baf_baseline_score, denoise_loss
from edn_ablate import VARIANTS, build_config, EDNetAbl


class _Tee:
    """Duplicate stdout to a file, so no shell redirection is needed.

    WHY THIS EXISTS
    ---------------
    On Windows PowerShell, `python ... 2>&1 | Tee-Object` merges the process's
    STDERR into the pipeline as ErrorRecord objects. tqdm writes its progress
    bar to stderr, so the first progress update is turned into a
    NativeCommandError and the run dies at the start of training with

        python :
        At line:1 char:1
            + CategoryInfo : NotSpecified: (:String) [], RemoteException
            + FullyQualifiedErrorId : NativeCommandError

    Teeing inside the process avoids the shell entirely. It also produces a
    CLEANER log than the shell would: every informative line goes to stdout and
    is captured, while tqdm stays on stderr and therefore on the console only,
    so the log file has no carriage-return progress-bar spam in it.
    """

    def __init__(self, stream, path):
        self.stream = stream
        self.fh = open(path, "a", encoding="utf-8", buffering=1)

    def write(self, s):
        self.stream.write(s)
        self.fh.write(s)
        return len(s)

    def flush(self):
        self.stream.flush()
        self.fh.flush()

    def isatty(self):
        return getattr(self.stream, "isatty", lambda: False)()

    def close(self):
        try:
            self.fh.close()
        except Exception:
            pass


def setup_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("high")


def _auc(y, s):
    return roc_auc_score(y, s) if 0 < y.sum() < len(y) else float("nan")


def write_provenance(args) -> dict:
    """Record how this run was invoked, next to what it produces.

    Without this a checkpoint states its architecture but not its recipe, and
    the learning rate, batch size, seed and augmentation range become
    unrecoverable the moment the terminal is closed.
    """
    prov = {
        "argv": sys.argv,
        "args": vars(args),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": (torch.cuda.get_device_name(0)
                   if torch.cuda.is_available() else "cpu"),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    try:
        prov["git"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        prov["git"] = None
    path = os.path.join(args.out_dir, "run_config.json")
    with open(path, "w") as fh:
        json.dump(prov, fh, indent=2)
    print(f"  provenance -> {path}")
    return prov


@torch.no_grad()
def evaluate(model, loader, device, cfg, amp: bool, levels):
    """-> (per-level model AUC, per-level baseline AUC, mean val loss)."""
    model.eval()
    sc, bs, ys = defaultdict(list), defaultdict(list), defaultdict(list)
    losses = []
    for feat, label, meta, vid in tqdm(loader, desc="  [val]", leave=False):
        feat = feat.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        meta = meta.to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=amp):
            logits = model(feat, meta)
        losses.append(denoise_loss(logits.float(), label, cfg).item())
        s = torch.sigmoid(logits.float()).cpu().numpy()
        b = baf_baseline_score(feat, meta).float().cpu().numpy()
        y = label.cpu().numpy()
        for j, v in enumerate(vid.tolist()):
            sc[v].append(s[j]); bs[v].append(b[j]); ys[v].append(y[j])

    model_auc, base_auc = {}, {}
    for v in sorted(sc):
        y = np.concatenate(ys[v]).ravel()
        model_auc[levels[v]] = _auc(y, np.concatenate(sc[v]).ravel())
        base_auc[levels[v]] = _auc(y, np.concatenate(bs[v]).ravel())
    return model_auc, base_auc, float(np.mean(losses))


def main():
    ap = argparse.ArgumentParser(description="Train EDNet on ED24")
    ap.add_argument("root", type=str, help="path to the ED24 directory")
    ap.add_argument("--N", type=int, default=4096)
    ap.add_argument("--cache_dir", type=str, default="./cache")
    ap.add_argument("--batch_size", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--max_windows_per_file", type=int, default=24)
    ap.add_argument("--out_dir", type=str, default="runs/ednet")
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--seed", type=int, default=230086)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--init_from", type=str, default=None,
                    help="checkpoint to initialise weights from (fine-tune). "
                         "Only 'model' is loaded; optimiser and schedule start "
                         "fresh, so --lr and --epochs describe THIS run.")
    ap.add_argument("--shot_noise_prob", type=float, default=0.6,
                    help="fraction of training windows given synthetic v2e-style "
                         "shot noise; 0 disables the augmentation")
    ap.add_argument("--shot_noise_min_hz", type=float, default=0.5)
    ap.add_argument("--shot_noise_max_hz", type=float, default=12.0,
                    help="upper end of the log-uniform training range. Rates "
                         "inside [min,max] are interpolation at test time; "
                         "beyond it, extrapolation. State this in the paper.")
    ap.add_argument("--ts_quant_prob", type=float, default=0.0,
                    help="fraction of train windows rounded onto a random "
                         "readout grid; 0 reproduces the original recipe")
    ap.add_argument("--ts_quant_min_us", type=float, default=2.0)
    ap.add_argument("--ts_quant_max_us", type=float, default=200.0)
    ap.add_argument("--val_shot_hz", type=float, default=10.0,
                    help="second validation pass with shot noise injected at "
                         "this rate; a per-epoch proxy for transfer to a noise "
                         "model absent from ED24. 0 disables it")
    ap.add_argument("--scale_mode", default="absolute",
                    choices=["relative", "absolute"])
    ap.add_argument("--save_every_epoch", action="store_true",
                    help="write epochNNN.pth as well as best/latest. Recommended: "
                         "in-domain validation and out-of-domain transfer have "
                         "been observed to move in OPPOSITE directions, so no "
                         "criterion computed on ED24 can identify the best "
                         "transferring epoch after the fact.")
    ap.add_argument("--skip_checks", action="store_true",
                    help="skip the convention / noise-fraction audit (not advised)")
    ap.add_argument("--no_log", action="store_true",
                    help="do not write <out_dir>/train_log.txt. The log is "
                         "written from inside the process, so do NOT pipe this "
                         "command through Tee-Object -- see the _Tee docstring.")
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--variant", default="full",
                    help="ablation variant: " + ", ".join(VARIANTS)
                         + ". 'full' reproduces the published model exactly.")
    ap.add_argument("--resume", default=None,
                    help="'auto' to continue from <out_dir>/latest.pth, or an "
                         "explicit checkpoint path. Restores model, optimiser, "
                         "schedule, AMP scaler, history and best-so-far, so the "
                         "run continues on the same recipe instead of "
                         "restarting it.")
    args = ap.parse_args()

    setup_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    tee = None
    if not args.no_log:
        tee = _Tee(sys.stdout, os.path.join(args.out_dir, "train_log.txt"))
        sys.stdout = tee
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    amp = (not args.no_amp) and device.type == "cuda"

    print("===================== run =============================")
    if tee is not None:
        print(f"  logging to {os.path.join(args.out_dir, 'train_log.txt')}")
    prov = write_provenance(args)
    print(f"  torch {prov['torch']} / cuda {prov['cuda']} on {prov['device']}")

    triples = list_ed24_files(args.root)
    if not triples:
        raise SystemExit(f"No ED24 files found under {args.root}")
    print(f"  {len(triples)} files across "
          f"{len({s for s, _, _ in triples})} scenes and "
          f"{len({v for _, v, _ in triples})} voltage levels")

    if not args.skip_checks:
        print("\n============ label convention check (spatiotemporal) ============")
        by_v = defaultdict(list)
        for s, v, f in triples:
            by_v[v].append(f)
        vs = sorted(by_v, key=float)
        probe = [by_v[v][0] for v in (vs[0], vs[len(vs) // 2], vs[-1])]
        info = verify_label_convention(probe, args.N, n_windows=6,
                                       cache_dir=args.cache_dir)
        print(f"  {info['test']}")
        for d in info["per_file"]:
            if "note" in d:
                print(f"    {d['file']:26s} frac1={d['frac_label_1']:.5f}  {d['note']}")
            else:
                print(f"    {d['file']:26s} frac1={d['frac_label_1']:.5f}  "
                      f"nbrs0={d['mean_neighbours_label_0']:6.2f}  "
                      f"nbrs1={d['mean_neighbours_label_1']:6.2f}  "
                      f"ratio={d['separation_ratio']:5.2f}x -> noise={d['vote_noise_label']}")
        print(f"  inferred_noise_label={info['inferred_noise_label']} "
              f"config={info['config_says_noise_label']} "
              f"unanimous={info['unanimous']} agrees={info['agrees']}")

        print("\n  noise fraction vs attenuator voltage "
              "(higher V = dimmer = more BA noise, so this must increase):")
        frac = noise_fraction_report(triples, cache_dir=args.cache_dir)
        keys = list(frac)
        print("    " + "  ".join(f"{k}V:{frac[k]:.3f}" for k in keys))
        rho = np.corrcoef([float(k) for k in keys], [frac[k] for k in keys])[0, 1]
        mono_says_1 = rho > 0
        print(f"    corr(voltage, fraction of label==1) = {rho:+.3f}  -> "
              f"monotonicity says noise is label {1 if mono_says_1 else 0}")

        if (not info["agrees"]) and (int(mono_says_1) != info["config_says_noise_label"]):
            raise SystemExit(
                "Both independent tests say the convention is flipped.\n"
                "Set POSITIVE_CLASS_IS_NOISE = False in ed24_data.py and invert "
                "the label column, then rerun.")
        if not info["agrees"]:
            print("  ! the two tests disagree with each other; proceeding with the "
                  "configured convention, but inspect a few files by hand.")

    print("\n=================== data loading =====================")
    train_items, val_items = scene_split(triples, args.val_frac, args.seed)
    print(f"  {len(train_items)} train / {len(val_items)} val files "
          f"(split by scene, so no window leakage)")
    train_ds = ED24Windows(train_items, args.N, args.cache_dir,
                           max_windows_per_file=args.max_windows_per_file,
                           seed=args.seed,
                           shot_noise_prob=args.shot_noise_prob,
                           shot_noise_range=(args.shot_noise_min_hz,
                                             args.shot_noise_max_hz),
                           ts_quant_prob=args.ts_quant_prob,
                           augment=args.augment,
                           ts_quant_range=(args.ts_quant_min_us,
                                           args.ts_quant_max_us))
    n_val_win = max(4, args.max_windows_per_file // 4)
    val_ds = ED24Windows(val_items, args.N, args.cache_dir,
                         max_windows_per_file=n_val_win, seed=args.seed)
    levels = val_ds.voltage_levels
    if args.shot_noise_prob > 0:
        print(f"  shot-noise augmentation on {args.shot_noise_prob:.0%} of train "
              f"windows, {args.shot_noise_min_hz}-{args.shot_noise_max_hz} Hz/pixel "
              f"(log-uniform)")
    # Held out CLEAN and at a fixed injected rate.  The clean number says
    # whether ED24 itself is still being learned; the injected one is the
    # early-warning proxy for transfer, which the clean number cannot see since
    # ED24 contains no v2e shot noise anywhere.
    shot_val_ds = None
    if args.val_shot_hz > 0:
        shot_val_ds = ED24Windows(val_items, args.N, args.cache_dir,
                                  max_windows_per_file=n_val_win, seed=args.seed,
                                  shot_noise_fixed=args.val_shot_hz)
    print(f"  {len(train_ds)} train windows / {len(val_ds)} val windows")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True,
                              pin_memory=True,
                              persistent_workers=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    shot_loader = (DataLoader(shot_val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
                   if shot_val_ds is not None else None)

    print("\n===================== model ==========================")
    cfg = build_config(args.variant, seq_len=args.N, dropout=args.dropout,
                       scale_mode=args.scale_mode)
    model = EDNetAbl(cfg).to(device)
    print(f"  variant: {args.variant} -- {VARIANTS[args.variant][2]}")
    if args.init_from:
        _ck = torch.load(args.init_from, map_location=device, weights_only=False)
        model.load_state_dict(_ck["model"] if "model" in _ck else _ck, strict=True)
        print(f"  init_from: {args.init_from} (epoch {_ck.get('epoch', '?')})")
    print(f"  parameters: {sum(p.numel() for p in model.parameters())/1e6:.3f}M"
          f"  | density channels {model.density.out_dim}"
          f"  | stem width {model.stem[0].in_features}"
          f"  | dropout {args.dropout}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    total = args.epochs * max(1, len(train_loader))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr,
                                                total_steps=total, pct_start=0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    best = -1.0
    history = []
    start_epoch = 1
    if args.resume:
        rp = (os.path.join(args.out_dir, "latest.pth")
              if args.resume == "auto" else args.resume)
        if not os.path.exists(rp):
            print(f"  --resume given but {rp} not found; starting fresh.")
        else:
            _rk = torch.load(rp, map_location=device, weights_only=False)
            if "opt" not in _rk:
                raise SystemExit(
                    f"\n{rp} carries no optimiser state -- it was written "
                    f"before this patch.\nResuming would restart AdamW and "
                    f"the OneCycle schedule mid-run, which is NOT the same "
                    f"recipe as the other variants.\nDelete {args.out_dir} "
                    f"and run that variant again from scratch.\n")
            if _rk.get("variant", args.variant) != args.variant:
                raise SystemExit(
                    f"\n{rp} is variant {_rk.get('variant')!r} but --variant "
                    f"is {args.variant!r}. Refusing to resume across "
                    f"variants.\n")
            model.load_state_dict(_rk["model"])
            opt.load_state_dict(_rk["opt"])
            sched.load_state_dict(_rk["sched"])
            scaler.load_state_dict(_rk["scaler"])
            start_epoch = int(_rk["epoch"]) + 1
            hp = os.path.join(args.out_dir, "history.json")
            if os.path.exists(hp):
                with open(hp) as fh:
                    history = [e for e in json.load(fh) if e["epoch"] < start_epoch]
            if history:
                best = max(e.get("selection", -1.0) for e in history)
            else:
                best = float(_rk.get("selection", -1.0))
            print(f"  RESUMED from {rp}: next epoch {start_epoch}/{args.epochs}, "
                  f"best so far {best:.4f}, lr {sched.get_last_lr()[0]:.2e}")
            if start_epoch > args.epochs:
                print("  this run is already complete; nothing to do.")
                return

    print("\n==================== training ========================")
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        t0, tot = time.time(), []
        pbar = tqdm(train_loader, desc=f"epoch {epoch:03d}")
        for feat, label, meta, _ in pbar:
            feat = feat.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            meta = meta.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=amp):
                logits = model(feat, meta)
                loss = denoise_loss(logits.float(), label, cfg)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        m_auc, b_auc, val_loss = evaluate(model, val_loader, device, cfg, amp, levels)
        vals = [v for v in m_auc.values() if np.isfinite(v)]
        mean_auc = float(np.mean(vals)) if vals else float("nan")
        worst_v = min(m_auc, key=lambda k: m_auc[k]) if vals else None

        shot_mean = float("nan")
        if shot_loader is not None:
            s_auc, s_base, _ = evaluate(model, shot_loader, device, cfg, amp, levels)
            sv = [v for v in s_auc.values() if np.isfinite(v)]
            shot_mean = float(np.mean(sv)) if sv else float("nan")
            sb = [v for v in s_base.values() if np.isfinite(v)]
            print(f"   shot-noise probe @{args.val_shot_hz:g}Hz/px: "
                  f"AUC={shot_mean:.4f} (baseline {np.mean(sb):.4f})")

        print(f"== epoch {epoch:03d} | train_loss={np.mean(tot):.4f} "
              f"val_loss={val_loss:.4f} mean_AUC={mean_auc:.4f} "
              f"worst={m_auc[worst_v]:.4f}@{worst_v}V "
              f"lr={sched.get_last_lr()[0]:.2e} ({time.time()-t0:.0f}s) ==")
        print("   per level  " + "  ".join(
            f"{v}V:{m_auc[v]:.3f}({b_auc[v]:.3f})" for v in sorted(m_auc, key=float)))
        below = [v for v in m_auc if np.isfinite(m_auc[v]) and m_auc[v] < b_auc[v]]
        if below:
            print(f"   ! loses to the unlearned baseline at {', '.join(below)}V")

        # Select on both when the probe is available: a checkpoint that is
        # excellent on clean ED24 and mediocre under injected shot noise is
        # precisely the one that transfers badly.
        sel = mean_auc if np.isnan(shot_mean) else 0.5 * (mean_auc + shot_mean)
        history.append({"epoch": epoch, "train_loss": float(np.mean(tot)),
                        "val_loss": val_loss, "val_auc": mean_auc,
                        "shot_auc": shot_mean, "selection": sel,
                        "per_level": m_auc})

        # ONE dict, reused for every save, so latest/best/epochNNN cannot drift.
        ckpt = {"model": model.state_dict(), "cfg": cfg, "epoch": epoch,
                "variant": args.variant,
                "opt": opt.state_dict(), "sched": sched.state_dict(),
                "scaler": scaler.state_dict(),
                "val_auc": mean_auc, "shot_auc": shot_mean, "selection": sel,
                "per_level": m_auc, "args": vars(args), "prov": prov}
        torch.save(ckpt, os.path.join(args.out_dir, "latest.pth"))
        if args.save_every_epoch:
            torch.save(ckpt, os.path.join(args.out_dir, f"epoch{epoch:03d}.pth"))
        if sel > best:
            best = sel
            torch.save(ckpt, os.path.join(args.out_dir, "best.pth"))
            print(f"   new best selection score {best:.4f} -> best.pth")

        with open(os.path.join(args.out_dir, "history.json"), "w") as fh:
            json.dump(history, fh, indent=2)

    print(f"\n[done] best selection score {best:.4f}")
    print(f"  artefacts in {args.out_dir}: best.pth, latest.pth, "
          f"run_config.json, history.json"
          + (", train_log.txt" if tee is not None else "")
          + (", epochNNN.pth" if args.save_every_epoch else ""))
    if tee is not None:
        sys.stdout = tee.stream
        tee.close()


if __name__ == "__main__":
    main()
