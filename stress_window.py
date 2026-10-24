from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__)) or "."

DVSCLEAN_FLAGS = ["--sensor", "1280", "720", "--time_unit", "s",
                  "--force_polarity", "2"]
DND21_DATASETS = ["mix", "driving_mix"]
DND21_FREQS = ["1", "3", "5", "7", "10"]

PAT = {
    "dvsclean": r"MEAN over \d+ scenes\s+([0-9.]+)",
    "dnd21":    r"mean over \d+ levels:\s+([0-9.]+)",
}
HEADS = {
    "dvsclean": ("50%", "100%"),
    "dnd21":    ("hotel-bar", "driving"),
}


def build_cmd(task, ckpt, N, a):
    if task == "dvsclean":
        return [sys.executable, "-u", os.path.join(HERE, "eval_dvsclean_snr.py"),
                "--ckpt", ckpt, "--root", a.dvsclean_root,
                "--N", str(N), *DVSCLEAN_FLAGS]
    return [sys.executable, "-u", os.path.join(HERE, "eval_auc_dnd21.py"),
            "--ckpt", ckpt, "--auc_root", a.dnd21_root, "--N", str(N),
            "--datasets", *DND21_DATASETS, "--freqs", *DND21_FREQS,
            "--cache_dir", a.cache_dir]



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

def read_log(path, task):
    if not os.path.exists(path):
        return None
    m = re.findall(PAT[task], read_text(path))
    return [float(x) for x in m[:2]] if len(m) >= 2 else None


# tqdm redraws its bar on every iteration, and with stderr merged into stdout
# each redraw is a separate line. One DND21 run emits a few thousand of them,
# which floods the terminal scrollback and bloats the log by two orders of
# magnitude -- the first ablation console log was 396 KB, almost all bars.
# They carry no information once the run is over, so they are dropped on the
# way to both the screen and the file.
_BAR = re.compile(r"\d+%\|[#\s|]*\||\d+/\d+\s*\[[\d:]+<|it/s\]|s/it\]|^\s*$")


def keep(line: str) -> bool:
    return not _BAR.search(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/abl")
    ap.add_argument("--dvsclean_root", default="DVSCLEAN/simulated_data")
    ap.add_argument("--dnd21_root",
                    default="ECCV2024_datasets/ECCV2024_datasets/AUC_test")
    ap.add_argument("--logs", default="logs/stress")
    ap.add_argument("--variants", nargs="+",
                    default=["full", "a1_rawdens", "a2_idxgraph"])
    ap.add_argument("--Ns", nargs="+", type=int, default=[4096, 10240, 20480])
    ap.add_argument("--task", choices=["dvsclean", "dnd21", "both"],
                    default="both")
    ap.add_argument("--ckpt_name", default="best.pth")
    ap.add_argument("--trained_N", type=int, default=10240,
                    help="the window the models were trained at; slopes are "
                         "reported relative to this column")
    ap.add_argument("--max_tries", type=int, default=3)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--cache_dir", default="./cache")
    ap.add_argument("--report_only", action="store_true",
                    help="run nothing; rebuild the tables from the logs "
                         "already on disk. Use this when the terminal "
                         "scrollback ate the summary.")
    ap.add_argument("--console", default="stress_window_console.log",
                    help="tee the filtered output here as well")
    ap.add_argument("--csv", default="stress_window_results.csv",
                    help="machine-readable dump of every cell in the tables")
    ap.add_argument("--keep_bars", action="store_true",
                    help="do not filter tqdm progress bars out of the logs")
    a = ap.parse_args()

    tasks = ["dvsclean", "dnd21"] if a.task == "both" else [a.task]

    con = open(a.console, "a", encoding="utf-8") if a.console else None

    def say(s=""):
        print(s)
        if con:
            con.write(s + "\n")
            con.flush()

    jobs = []
    for task in tasks:
        for v in a.variants:
            ck = os.path.join(a.runs, v, a.ckpt_name)
            if not os.path.exists(ck) and not a.report_only:
                raise SystemExit(f"  missing {ck}")
            for N in a.Ns:
                log = os.path.join(a.logs, task, f"{v}_N{N}.txt")
                if read_log(log, task) and not a.force:
                    if not a.report_only:
                        say(f"  [{task}/{v}/N={N}] already complete (skipping)")
                    continue
                jobs.append((task, v, N, ck, log))

    if a.dry_run:
        for task, v, N, ck, _ in jobs:
            print("  $ " + " ".join(build_cmd(task, ck, N, a)))
        return 0

    if a.report_only:
        if jobs:
            say(f"  report_only: {len(jobs)} cell(s) have no complete log and "
                f"will show as --")
        jobs = []

    for task, v, N, ck, log in jobs:
        os.makedirs(os.path.dirname(log), exist_ok=True)
        cmd = build_cmd(task, ck, N, a)
        for attempt in range(1, a.max_tries + 1):
            t0 = time.time()
            say(f"\n  [{task}/{v}/N={N}] attempt {attempt}/{a.max_tries}")
            with open(log, "w", encoding="utf-8") as fh:
                fh.write("# " + " ".join(cmd) + "\n")
                p = subprocess.Popen(cmd, cwd=HERE, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True,
                                     encoding="utf-8", errors="replace",
                                     bufsize=1)
                for line in p.stdout:
                    if not a.keep_bars and not keep(line):
                        continue
                    fh.write(line)
                    sys.stdout.write("    " + line)
                    if con:
                        con.write("    " + line)
                rc = p.wait()
            if con:
                con.flush()
            ok = read_log(log, task) is not None
            say(f"  [{task}/{v}/N={N}] rc={rc}  "
                f"{(time.time()-t0)/60:.1f} min  "
                f"{'COMPLETE' if ok else 'INCOMPLETE'}")
            if ok:
                break

    # ------------------------------------------------------------- the table
    csv_rows = [("task", "variant", "N", "series", "auc")]
    for task in tasks:
        h1, h2 = HEADS[task]
        say(f"\n  === {task}: AUC against window size ===")
        say(f"  {'variant':14s} " +
            " ".join(f"{'N=' + str(N):>17s}" for N in a.Ns))
        say(f"  {'':14s} " +
            " ".join(f"{h1 + '/' + h2:>17s}" for _ in a.Ns))
        got = {}
        for v in a.variants:
            cells = []
            for N in a.Ns:
                r = read_log(os.path.join(a.logs, task, f"{v}_N{N}.txt"), task)
                got[(v, N)] = r
                cells.append("/".join(f"{x:.4f}" for x in r) if r else "--")
                if r:
                    csv_rows.append((task, v, str(N), h1, f"{r[0]:.4f}"))
                    csv_rows.append((task, v, str(N), h2, f"{r[1]:.4f}"))
            say(f"  {v:14s} " + " ".join(f"{c:>17s}" for c in cells))

        # The quantity the claim is about: how much each variant loses when the
        # window moves off the trained one. Reported per sequence/ratio, and
        # relative to the full model's own loss, since only the DIFFERENCE in
        # degradation speaks to the mechanism.
        base = a.trained_N
        if base not in a.Ns:
            continue
        say(f"\n  degradation from N={base}, and relative to the full model")
        for j, lab in enumerate((h1, h2)):
            say(f"    -- {lab} --")
            fullref = got.get(("full", base))
            for v in a.variants:
                at0 = got.get((v, base))
                if not at0:
                    continue
                parts = []
                for N in a.Ns:
                    if N == base:
                        continue
                    r = got.get((v, N))
                    if not r:
                        parts.append(f"N={N}: --")
                        continue
                    drop = r[j] - at0[j]
                    extra = ""
                    if v != "full" and fullref and got.get(("full", N)):
                        fdrop = got[("full", N)][j] - fullref[j]
                        extra = f"  ({drop - fdrop:+.4f} vs full)"
                    parts.append(f"N={N}: {drop:+.4f}{extra}")
                say(f"      {v:14s} " + "   ".join(parts))
        say("\n    A negative number in parentheses means that variant lost "
            "MORE than\n    the full model did when the window moved -- "
            "which is the result the\n    rate-normalisation and "
            "physical-time claims predict. A positive or\n    near-zero "
            "number means the mechanism did not buy window invariance."
            "\n    Read the two directions separately: a mechanism can help"
            "\n    when the window grows and hurt when it shrinks.")

    if a.csv:
        with open(a.csv, "w", encoding="utf-8", newline="") as fh:
            for row in csv_rows:
                fh.write(",".join(row) + "\n")
        say(f"\n  wrote {a.csv} ({len(csv_rows)-1} rows)")
    if con:
        say(f"  full filtered output appended to {a.console}")
        con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
