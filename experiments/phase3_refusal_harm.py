"""Stage 0.5 RUNNER — refusal-vs-harmfulness decomposition (logic in mech_security.refusal_harm).

Builds a mixed set whose CONTENT (harmful vs harmless) and BEHAVIOUR (refused vs complied) vary
independently — AdvBench harmful + Alpaca harmless + an over-refusal set (benign-but-scary). Behaviour is
judged (no hook). Then at the ablation cell: d_harm = harmful-vs-harmless CONTENT (== the ablated d̂),
d_refuse = refused-vs-complied BEHAVIOUR; reports cos(d_harm,d_refuse) + bootstrap band + the PER-MODEL 2×2
cell counts + a MIN_OFFDIAG feasibility gate. harmful-complied = NATURAL under-refusals only (no jailbreak/
ablation). Qwen first (free) validates the SET; only a passing Qwen run justifies the Llama pod (DEVLOG §14).

    python experiments/phase3_refusal_harm.py --ckpt <hf> --base <hf> --device cuda --no-processing \
        --layer 22 --out results/phase3_refusal_harm_qwen.json
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import json
import random
from functools import partial
from pathlib import Path

from dotenv import load_dotenv

from mech_security import track1_splits as ts
from mech_security.activations import cache_resid
from mech_security.model import _auto_device, format_prompt_for_bundle
from mech_security.phase3_loaders import DEFAULT_BASE, load_defended_model
from mech_security.refusal_harm import MIN_OFFDIAG, behavior_labels, cos_bootstrap, decompose


def _over_refusal(path: str) -> list[str]:
    return [json.loads(line)["text"].strip() for line in open(path) if line.strip()]


def main() -> int:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--advbench", default="data/advbench_harmful_behaviors.csv")
    ap.add_argument("--harmless", default="data/alpaca_harmless.jsonl")
    ap.add_argument("--over-refusal", default="data/over_refusal.jsonl")
    ap.add_argument("--layer", type=int, required=True, help="ablation cell layer (d̂'s layer)")
    ap.add_argument("--position", type=int, default=-1)
    ap.add_argument("--n-harmful", type=int, default=120)
    ap.add_argument("--n-harmless", type=int, default=90)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-processing", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--seed", type=int, default=20260606)
    ap.add_argument("--out", default="results/phase3_refusal_harm.json")
    args = ap.parse_args()

    device = args.device or _auto_device()
    goals = ts._advbench_goals(args.advbench)
    random.Random(args.seed).shuffle(goals)
    harmful = goals[: args.n_harmful]                                   # content = harmful
    harmless = ts._harmless(args.harmless)[: args.n_harmless]           # content = harmless
    over = _over_refusal(args.over_refusal)                             # content = harmless (refusal-prone)
    prompts = harmful + harmless + over
    content_is_harmful = [True] * len(harmful) + [False] * (len(harmless) + len(over))
    print(f"[rh] {args.ckpt} device={device} layer={args.layer} | harmful={len(harmful)} "
          f"harmless={len(harmless)} over_refusal={len(over)} (n={len(prompts)})")

    b = load_defended_model(args.ckpt, base=args.base, device=device, no_processing=args.no_processing)
    fmt = partial(format_prompt_for_bundle, b)
    refused = behavior_labels(b, prompts, fmt=fmt, max_new_tokens=args.max_new_tokens)
    acts = cache_resid(b, prompts, layer=args.layer, position=args.position, format_fn=fmt, show_progress=False)

    # 2×2 counts (PER MODEL) — report regardless of feasibility
    cells = {f"{'harm' if h else 'safe'}_{'refuse' if r == 1 else 'comply'}":
             int(sum(1 for ci, ri in zip(content_is_harmful, refused) if ci == h and ri == r))
             for h in (True, False) for r in (1, 0)}
    offdiag = {"harm_comply": cells["harm_comply"], "safe_refuse": cells["safe_refuse"]}
    feasible = offdiag["harm_comply"] >= MIN_OFFDIAG and offdiag["safe_refuse"] >= MIN_OFFDIAG
    res = {"ckpt": args.ckpt, "layer": args.layer, "device": device, "n": len(prompts),
           "cells_2x2": cells, "off_diagonal": offdiag, "MIN_OFFDIAG": MIN_OFFDIAG, "feasible": feasible}

    if feasible:
        d = decompose(acts, content_is_harmful, refused)
        boot = cos_bootstrap(acts, content_is_harmful, refused, n_boot=500, seed=args.seed)
        res["cos_harm_refuse"] = d["cos_harm_refuse"]
        res["cos_bootstrap"] = boot
    else:
        res["note"] = (f"INFEASIBLE as designed: off-diagonal {offdiag} < MIN_OFFDIAG={MIN_OFFDIAG} — "
                       "d_refuse would be noise; do NOT trust/compare the cosine. Need a richer over-refusal "
                       "/ under-refusal set (or this model just doesn't produce enough off-diagonal).")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, default=str))
    print(f"[rh] wrote {out}")
    print("[rh] RH_RESULT:", {k: v for k, v in res.items() if k != "cos_bootstrap"},
          "| cos_boot:", res.get("cos_bootstrap"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
