"""Probe-after-ablation RUNNER (thin; logic in mech_security.probe_ablation).

The Goal-B centerpiece (PHASE3_PLAN 2026-06-06): after ablating the diff-of-means k-subspace at the best
cell, is the leftover refusal still linearly readable? Distinguishes H-dim (AUC high, still refuses:
"linearly present, not low-k-ablatable") from H-nonlinear (AUC ≈ chance, still refuses: refusal isn't a
linear residual feature → why diff-of-means underperforms). Run on Llama-8B (the question) and on Qwen-3B
(the positive control: post-ablation refusal gone ⇒ ~single-class ⇒ nothing to read = full collapse).

    python experiments/phase3_probe_ablation.py --base <hf> --vanilla <hf> --device cuda --no-processing \
        --layer 18 --k 3 --n-extract 60 --n-probe 120 --out results/phase3_probe_llama.json

Disjoint sets (seeded): extract_harmful (d̂) ⊥ probe harmful (labelled + probed). Reports RAW numbers.
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # MPS fallback before torch import (clean cmd)

import argparse
import json
import random
from pathlib import Path

from dotenv import load_dotenv

from mech_security import track1_splits as ts
from mech_security.model import _auto_device
from mech_security.phase3_loaders import DEFAULT_BASE, load_defended_model
from mech_security.probe_ablation import probe_after_ablation


def main() -> int:
    load_dotenv()  # ANTHROPIC_API_KEY (judge labels) from .env
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="model under test (e.g. the Llama vanilla, or the defense)")
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--advbench", default="data/advbench_harmful_behaviors.csv")
    ap.add_argument("--harmless", default="data/alpaca_harmless.jsonl")
    ap.add_argument("--layer", type=int, required=True, help="best-cell layer to ablate (e.g. Llama L18)")
    ap.add_argument("--position", type=int, default=-1)
    ap.add_argument("--k", type=int, default=3, help="low-k subspace to ablate (clean regime)")
    ap.add_argument("--n-extract", type=int, default=60, help="harmful prompts for d̂ extraction")
    ap.add_argument("--n-probe", type=int, default=120, help="harmful prompts to label + probe (disjoint)")
    ap.add_argument("--n-harmless", type=int, default=60)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-processing", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=128,
                    help="HELD CONSTANT (DEVLOG §9: judge S is length-entangled); 128 to match the attack run")
    ap.add_argument("--seed", type=int, default=20260606)
    ap.add_argument("--out", default="results/phase3_probe_ablation.json")
    args = ap.parse_args()

    device = args.device or _auto_device()
    goals = ts._advbench_goals(args.advbench)
    harmless = ts._harmless(args.harmless)
    rng = random.Random(args.seed)
    rng.shuffle(goals)
    extract_harmful = goals[: args.n_extract]
    probe_prompts = goals[args.n_extract : args.n_extract + args.n_probe]  # DISJOINT from extraction
    harmless_extract = harmless[: args.n_harmless]
    if len(probe_prompts) < args.n_probe or len(harmless_extract) < args.n_harmless:
        raise SystemExit(f"insufficient data: {len(goals)} goals / {len(harmless)} harmless for "
                         f"extract {args.n_extract} + probe {args.n_probe}, harmless {args.n_harmless}")

    print(f"[probe] {args.ckpt} device={device} layer={args.layer} k={args.k} "
          f"extract={len(extract_harmful)} probe={len(probe_prompts)} harmless={len(harmless_extract)}")
    b = load_defended_model(args.ckpt, base=args.base, device=device, no_processing=args.no_processing)
    res = probe_after_ablation(
        b, probe_prompts, harmless_extract, layer=args.layer, position=args.position, k=args.k,
        extract_harmful=extract_harmful, seed=args.seed, max_new_tokens=args.max_new_tokens,
    )
    res["ckpt"] = args.ckpt
    res["device"] = device
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, default=str))
    print(f"[probe] wrote {out}")
    print("[probe] PROBE_RESULT:", {k: v for k, v in res.items() if k != "probe"},
          "| probe:", res.get("probe") if res.get("probe") is None else
          {kk: res["probe"][kk] for kk in ("best_readout_layer", "max_test_auc", "shuffled_auc_at_best")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
