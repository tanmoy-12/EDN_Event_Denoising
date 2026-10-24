#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

from edn_ablate import VARIANTS

BUDGET = dict(N=10240, batch_size=8, epochs=28, lr="1e-3", weight_decay="1e-2",
              max_windows_per_file=20, seed=230086, val_shot_hz=10,
              shot_noise_prob=0.6, shot_noise_min_hz=0.5, shot_noise_max_hz=12,
              scale_mode="absolute", num_workers=6)


def done_epochs(out_dir: str) -> int:
    """Highest epoch recorded in history.json; 0 if the run has not started."""
    hp = os.path.join(out_dir, "history.json")
    if not os.path.exists(hp):
        return 0
    try:
        with open(hp) as fh:
            return max((int(e["epoch"]) for e in json.load(fh)), default=0)
    except Exception:
        return 0


def command(variant, root, cache, out_dir, epochs, smoke=False):
    _, over, _ = VARIANTS[variant]
    b = dict(BUDGET)
    b["epochs"] = epochs
    b.update(over)
    cmd = [sys.executable, "train_edn.py", root, "--cache_dir", cache,
           "--variant", variant, "--out_dir", out_dir,
           "--augment", "--save_every_epoch", "--resume", "auto"]
    for k, v in b.items():
        cmd += [f"--{k}", str(v)]
    if smoke:
        cmd += ["--no_amp", "--skip_checks"]
    return cmd


class Log:
    def __init__(self, path):
        self.fh = open(path, "a", encoding="utf-8", buffering=1)

    def __call__(self, msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.fh.write(line + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--cache_dir", default="./cache")
    ap.add_argument("--out_root", default="runs/abl")
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--epochs", type=int, default=BUDGET["epochs"])
    ap.add_argument("--max_stall", type=int, default=3)
    ap.add_argument("--max_tries", type=int, default=25)
    ap.add_argument("--cooldown", type=int, default=60,
                    help="seconds between attempts; lets the GPU settle and "
                         "keeps a crash-loop from spinning the disk")
    ap.add_argument("--log", default="supervisor.log")
    ap.add_argument("--smoke", action="store_true",
                    help="TEST MODE ONLY. Shrinks the budget (N=512, batch 4, "
                         "4 windows/file, 0 workers, no AMP, no data checks) so "
                         "the supervisor itself can be verified in minutes on "
                         "the synthetic tree. Never produces a table row.")
    a = ap.parse_args()

    if a.smoke:
        BUDGET.update(N=512, batch_size=4, max_windows_per_file=4,
                      num_workers=0)
        log_warn = ("  *** SMOKE MODE: budget shrunk, results are meaningless. "
                    "***")
    else:
        log_warn = None

    bad = [v for v in a.variants if v not in VARIANTS]
    if bad:
        raise SystemExit(f"unknown variant(s): {bad}")

    log = Log(a.log)
    log(f"supervising {len(a.variants)} variant(s): {' '.join(a.variants)}")
    log(f"budget {a.epochs} epochs each; max_stall={a.max_stall} "
        f"max_tries={a.max_tries} cooldown={a.cooldown}s")
    if log_warn:
        log(log_warn)

    summary = {}
    for v in a.variants:
        out = os.path.join(a.out_root, v)
        tries = stalls = 0
        while True:
            before = done_epochs(out)
            if before >= a.epochs:
                log(f"{v}: complete at epoch {before}")
                summary[v] = f"done ({before} epochs, {tries} attempt(s))"
                break
            if tries >= a.max_tries:
                log(f"{v}: ABANDONED after {tries} attempts at epoch {before}")
                summary[v] = f"abandoned at epoch {before} (max_tries)"
                break
            tries += 1
            log(f"{v}: attempt {tries}, resuming from epoch {before}")
            t0 = time.time()
            rc = subprocess.run(command(v, a.root, a.cache_dir, out,
                                        a.epochs, a.smoke)).returncode
            after = done_epochs(out)
            gained = after - before
            log(f"{v}: attempt {tries} exited {rc} after "
                f"{(time.time()-t0)/3600:.2f} h; epochs {before} -> {after} "
                f"(+{gained})")
            if rc == 0 and after >= a.epochs:
                log(f"{v}: complete at epoch {after}")
                summary[v] = f"done ({after} epochs, {tries} attempt(s))"
                break
            stalls = 0 if gained > 0 else stalls + 1
            if stalls >= a.max_stall:
                log(f"{v}: ABANDONED -- {stalls} consecutive attempts made no "
                    f"progress, so this is a deterministic failure, not a "
                    f"transient one. Read {os.path.join(out,'train_log.txt')}.")
                summary[v] = f"abandoned at epoch {after} (no progress)"
                break
            log(f"{v}: cooling down {a.cooldown}s before retry "
                f"(stall streak {stalls}/{a.max_stall})")
            time.sleep(a.cooldown)

    log("=" * 60)
    for v in a.variants:
        log(f"  {v:14s} {summary.get(v, 'not reached')}")
    unfinished = [v for v in a.variants if not summary.get(v, "").startswith("done")]
    log(f"finished {len(a.variants)-len(unfinished)}/{len(a.variants)} variant(s)")
    return 1 if unfinished else 0


if __name__ == "__main__":
    sys.exit(main())
