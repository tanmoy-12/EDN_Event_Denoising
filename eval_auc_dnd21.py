from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from ed24_data import EventFileWindows, verify_label_convention
from edn_model import baf_baseline_score
from edn_ablate import (AblConfig as EDNConfig, EDNetAbl as EDNet,
                        config_from_checkpoint)


@torch.no_grad()
def score_file(model, path, N, device, batch_size, cache_dir, amp):
    ds = EventFileWindows(path, N, cache_dir=cache_dir)
    if len(ds) == 0:
        return None
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)

    s_model, s_base, ys = [], [], []
    for feat, label, meta in tqdm(loader, desc=f"    {os.path.basename(path)}",
                                  leave=False):
        feat = feat.to(device); meta = meta.to(device)
        with torch.autocast("cuda", enabled=amp):
            logits = model(feat, meta)
        s_model.append(torch.sigmoid(logits.float()).cpu().numpy().ravel())
        s_base.append(baf_baseline_score(feat, meta).float().cpu().numpy().ravel())
        ys.append(label.numpy().ravel())
    return (np.concatenate(s_model), np.concatenate(s_base), np.concatenate(ys))


def main():
    ap = argparse.ArgumentParser(description="AUC_test sweep for EDNet")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--auc_root", required=True)
    ap.add_argument("--datasets", nargs="+", default=["driving_mix", "mix"])
    ap.add_argument("--freqs", nargs="+", type=int, default=list(range(1, 11)))
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--cache_dir", type=str, default="./cache")
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--skip_checks", action="store_true")
    ap.add_argument("--N", type=int, default=None,
                    help="window size in events. Default: whatever the checkpoint "
                         "was TRAINED with, since a train/test window mismatch "
                         "silently changes what the density ladder measures.")
    args = ap.parse_args()

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
    print(f"loaded {args.ckpt} (epoch {ck.get('epoch','?')}, "
          f"val_auc {ck.get('val_auc','?')})\n")

    # AUC_test is a different corpus from ED24 (DND21 sequences with v2e shot
    # noise), so its label column is worth verifying separately rather than
    # assumed to match.  A silent flip here inverts every number below.
    if not args.skip_checks:
        probe = [os.path.join(args.auc_root, f"{hz}hz", f"{args.datasets[0]}_result.txt")
                 for hz in (args.freqs[0], args.freqs[-1])]
        probe = [p for p in probe if os.path.exists(p)]
        if probe:
            info = verify_label_convention(probe, args.N, n_windows=4,
                                           cache_dir=args.cache_dir)
            print("label convention on AUC_test:")
            for d in info["per_file"]:
                if "note" not in d:
                    print(f"  frac1={d['frac_label_1']:.4f}  "
                          f"nbrs0={d['mean_neighbours_label_0']:6.2f} "
                          f"nbrs1={d['mean_neighbours_label_1']:6.2f} "
                          f"-> noise={d['vote_noise_label']}")
            if not info["agrees"]:
                print("  ! AUC_test convention looks flipped relative to ED24; "
                      "every AUC below would be 1-AUC.\n")
            else:
                print("  matches ED24 (noise is label 1)\n")

    table = {}
    for name in args.datasets:
        print(f"=== {name} ===")
        for hz in args.freqs:
            path = os.path.join(args.auc_root, f"{hz}hz", f"{name}_result.txt")
            if not os.path.exists(path):
                print(f"  {hz:2d}Hz  (missing: {path})")
                continue
            out = score_file(model, path, args.N, device,
                             args.batch_size, args.cache_dir, amp)
            if out is None:
                print(f"  {hz:2d}Hz  (fewer than {args.N} events)")
                continue
            sm, sb, y = out
            if not (0 < y.sum() < len(y)):
                print(f"  {hz:2d}Hz  (single-class labels; AUC undefined)")
                continue

            auc_m = roc_auc_score(y, sm)
            auc_b = roc_auc_score(y, sb)
            table[(name, hz)] = auc_m
            flag = ""
            if auc_m < 0.5:
                flag = (f"   <-- BELOW CHANCE (inverted = {1-auc_m:.4f}); "
                        "check label convention")
            elif auc_m < auc_b:
                flag = "   <-- loses to the unlearned baseline"
            print(f"  {hz:2d}Hz  AUC={auc_m:.4f}  baseline={auc_b:.4f}  "
                  f"noise_frac={y.mean():.3f}  score_std={sm.std():.4f}{flag}")
        vals = [v for (n, _), v in table.items() if n == name]
        if vals:
            print(f"  mean over {len(vals)} levels: {np.mean(vals):.4f}  "
                  f"worst: {np.min(vals):.4f}\n")


if __name__ == "__main__":
    main()
