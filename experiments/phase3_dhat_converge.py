"""d̂ convergence RUNNER (thin; logic in directions.convergence_cos) — the H-extract check.

Does the diff-of-means refusal direction STABILIZE as extraction n grows? Extract d̂ at the best cell
(L18/pos−1) from nested subsets n∈{50,100,200} and report cos to the largest-n d̂. cos→1 ⇒ the direction
has converged (more data doesn't move it) ⇒ Llama's floor is NOT a too-few-examples artifact (H-extract
ruled out); persistently low cos ⇒ still under-extracted. Extraction-only (no generation/judge) → cheap.

    python experiments/phase3_dhat_converge.py --ckpt <hf> --base <hf> --device cuda --no-processing \
        --layer 18 --ns 50 100 200 --out results/phase3_dhat_converge.json
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # MPS fallback before torch import (clean cmd)

import argparse
import json
import random
from functools import partial
from pathlib import Path

from mech_security import track1_splits as ts
from mech_security.directions import convergence_cos, extract_d_hat, unit
from mech_security.model import _auto_device, format_prompt_for_bundle
from mech_security.phase3_loaders import DEFAULT_BASE, load_defended_model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--advbench", default="data/advbench_harmful_behaviors.csv")
    ap.add_argument("--harmless", default="data/alpaca_harmless.jsonl")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--position", type=int, default=-1)
    ap.add_argument("--ns", type=int, nargs="+", default=[50, 100, 200])
    # --n-extract is the max n actually used; named so the orchestrator pre-flight can size the data need.
    ap.add_argument("--n-extract", type=int, default=200)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-processing", action="store_true")
    ap.add_argument("--seed", type=int, default=20260606)
    ap.add_argument("--out", default="results/phase3_dhat_converge.json")
    args = ap.parse_args()

    device = args.device or _auto_device()
    goals = ts._advbench_goals(args.advbench)
    harmless = ts._harmless(args.harmless)
    random.Random(args.seed).shuffle(goals)
    ns = sorted(n for n in args.ns if n <= min(len(goals), len(harmless)) and n <= args.n_extract)
    if len(ns) < 2:
        raise SystemExit(f"need ≥2 usable n values; got {ns} from {len(goals)} goals / {len(harmless)} harmless")

    print(f"[converge] {args.ckpt} device={device} layer={args.layer} pos={args.position} ns={ns}")
    b = load_defended_model(args.ckpt, base=args.base, device=device, no_processing=args.no_processing)
    fmt = partial(format_prompt_for_bundle, b)
    dhats = {}
    for n in ns:  # NESTED subsets goals[:n] vs harmless[:n] — measures stability as n grows
        d_hat, _, _, _ = extract_d_hat(b, goals[:n], harmless[:n], layer=args.layer,
                                       position=args.position, format_fn=fmt)
        dhats[n] = unit(d_hat).detach().cpu().float()

    res = convergence_cos(dhats)
    res.update({"ckpt": args.ckpt, "layer": args.layer, "position": args.position, "device": device})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"[converge] wrote {out}")
    print("[converge] CONVERGE_RESULT:", {"cos_to_largest": res["cos_to_largest"],
                                          "cos_consecutive": res["cos_consecutive"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
