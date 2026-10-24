import inspect
import os
import sys

SEP = "=" * 72


def banner(s):
    print("\n" + SEP)
    print(s)
    print(SEP)


# ---------------------------------------------------------------- 1
banner("1.  ednet_probs -- signature and source")
try:
    import eval_mesr_dnd21 as m
    fn = m.ednet_probs
    print(f"  defined in: {inspect.getsourcefile(fn)}")
    print(f"  signature : ednet_probs{inspect.signature(fn)}\n")
    src = inspect.getsource(fn)
    print(src if len(src) < 6000 else src[:6000] + "\n  ... (truncated)")
except Exception as exc:
    print(f"  could not read ednet_probs: {exc!r}")

# ---------------------------------------------------------------- 2
banner("2.  event_io -- public surface")
try:
    import event_io as io
    print(f"  file: {getattr(io, '__file__', '?')}")
    ver = [n for n in dir(io) if "version" in n.lower() or n == "__version__"]
    for n in ver:
        print(f"  {n} = {getattr(io, n)!r}")

    print("\n  callables:")
    for n in sorted(dir(io)):
        if n.startswith("_"):
            continue
        v = getattr(io, n)
        if callable(v):
            try:
                sig = str(inspect.signature(v))
            except (TypeError, ValueError):
                sig = "(?)"
            flag = "  <-- sensor-ish" if any(
                k in n.lower() for k in
                ("sensor", "resol", "size", "shape", "dim", "wh")) else ""
            print(f"    {n}{sig}{flag}")

    print("\n  module-level values (non-callable, non-dunder):")
    for n in sorted(dir(io)):
        if n.startswith("_"):
            continue
        v = getattr(io, n)
        if callable(v) or inspect.ismodule(v):
            continue
        r = repr(v)
        if len(r) > 90:
            r = r[:90] + "..."
        flag = ""
        if isinstance(v, (tuple, list)) and len(v) == 2 and \
                all(isinstance(q, int) for q in v):
            flag = "  <-- looks like (W, H)"
        elif any(k in n.lower() for k in
                 ("sensor", "resol", "size", "width", "height", "wh")):
            flag = "  <-- sensor-ish"
        print(f"    {n} = {r}{flag}")
except Exception as exc:
    print(f"  could not import event_io: {exc!r}")

# ---------------------------------------------------------------- 3
banner("3.  where 'sensor' appears in the existing eval scripts")
for fname in ("eval_dvsclean_snr.py", "eval_mesr_dnd21.py", "event_io.py",
              "eval_emlb.py"):
    if not os.path.exists(fname):
        print(f"\n  -- {fname}: not found")
        continue
    hits = []
    with open(fname, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f, 1):
            low = line.lower()
            if any(k in low for k in ("sensor", "set_res", "resolution",
                                      "width", "height", " w, h", "(w, h)")):
                hits.append((i, line.rstrip()))
    print(f"\n  -- {fname}: {len(hits)} matching line(s)")
    for i, line in hits[:40]:
        print(f"     {i:>5}: {line[:110]}")
    if len(hits) > 40:
        print(f"     ... {len(hits) - 40} more")

# ---------------------------------------------------------------- 4
banner("4.  does the model config carry a resolution?")
try:
    import torch
    from edn_model import config_from_checkpoint
    ck_path = sys.argv[1] if len(sys.argv) > 1 else r"runs\edn_abs28\best.pth"
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    cfg = config_from_checkpoint(ck)
    print(f"  checkpoint: {ck_path}")
    fields = [n for n in dir(cfg) if not n.startswith("_")
              and not callable(getattr(cfg, n))]
    for n in sorted(fields):
        v = getattr(cfg, n)
        flag = "  <-- sensor-ish" if any(
            k in n.lower() for k in
            ("sensor", "resol", "width", "height", "size", "wh")) else ""
        print(f"    {n} = {v!r}{flag}")
    print(f"\n  ck['args'] keys: "
          f"{sorted(ck.get('args', {}).keys()) if isinstance(ck.get('args'), dict) else ck.get('args')}")
except Exception as exc:
    print(f"  could not read the checkpoint config: {exc!r}")

print("\n" + SEP)
print("Paste all of the above back.")
print(SEP)
