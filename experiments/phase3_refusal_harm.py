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
from mech_security.refusal_harm import (
    LEX_TOL,
    MIN_OFFDIAG,
    behavior_labels,
    cos_bootstrap,
    decompose,
    lexical_overlap_check,
)


def _over_refusal(path: str) -> list[str]:
    return [json.loads(line)["text"].strip() for line in open(path) if line.strip()]


def main() -> int:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--advbench", default="data/advbench_harmful_behaviors.csv")
    ap.add_argument("--harmless", default="data/alpaca_harmless.jsonl")
    ap.add_argument("--over-refusal", default="data/xstest_safe.jsonl",
                    help="XSTest safe prompts (canonical over-refusal benchmark); the lexical gate filters vocab-driven ones")
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

    # 2×2 counts (PER MODEL) + the prompt list per cell (for the lexical-overlap gate)
    def cell_of(h, r):
        return [p for p, ci, ri in zip(prompts, content_is_harmful, refused) if ci == h and ri == r]
    cell_texts = {"harm_refuse": cell_of(True, 1), "harm_comply": cell_of(True, 0),
                  "safe_refuse": cell_of(False, 1), "safe_comply": cell_of(False, 0)}
    cells = {k: len(v) for k, v in cell_texts.items()}
    offdiag = {"harm_comply": cells["harm_comply"], "safe_refuse": cells["safe_refuse"]}
    counts_ok = offdiag["harm_comply"] >= MIN_OFFDIAG and offdiag["safe_refuse"] >= MIN_OFFDIAG

    # LEXICAL-OVERLAP GATE (DEVLOG §14b): train TF-IDF harm-vs-harmless on ORDINARY content (advbench vs
    # alpaca), score each cell. Off-diagonal must be lexically like its CONTENT class, not harm-vocab.
    lex = lexical_overlap_check(harmful, harmless, cell_texts, seed=args.seed)
    cp = lex["cell_mean_p_harmful"]
    safe_delta = cp.get("safe_refuse", float("nan")) - cp.get("safe_comply", float("nan"))  # >TOL ⇒ vocab confound
    harm_delta = cp.get("harm_refuse", float("nan")) - cp.get("harm_comply", float("nan"))  # >TOL ⇒ severity skew
    lexical_ok = bool(safe_delta <= LEX_TOL and harm_delta <= LEX_TOL)  # nan compares False → not ok
    feasible = counts_ok and lexical_ok

    res = {"ckpt": args.ckpt, "layer": args.layer, "device": device, "n": len(prompts),
           "cells_2x2": cells, "off_diagonal": offdiag, "MIN_OFFDIAG": MIN_OFFDIAG, "counts_ok": counts_ok,
           "lexical": {**lex, "safe_delta": safe_delta, "harm_delta": harm_delta, "LEX_TOL": LEX_TOL,
                       "lexical_ok": lexical_ok}, "feasible": feasible}

    if feasible:
        d = decompose(acts, content_is_harmful, refused)
        res["cos_harm_refuse"] = d["cos_harm_refuse"]
        res["cos_bootstrap"] = cos_bootstrap(acts, content_is_harmful, refused, n_boot=500, seed=args.seed)
    else:
        why = []
        if not counts_ok:
            why.append(f"off-diagonal {offdiag} < MIN_OFFDIAG={MIN_OFFDIAG}")
        if not lexical_ok:
            why.append(f"lexical gate FAILED (safe_delta={safe_delta:.2f}, harm_delta={harm_delta:.2f} vs "
                       f"LEX_TOL={LEX_TOL}; cell P_harmful={cp}) — off-diagonal is harm-VOCAB-driven, "
                       "d_refuse would correlate with d_harm through lexicon (Phase-1 confound)")
        res["note"] = "INFEASIBLE as designed: " + "; ".join(why) + ". Cosine NOT computed/trusted."

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, default=str))
    print(f"[rh] wrote {out}")
    print("[rh] RH_RESULT:", {k: v for k, v in res.items() if k != "cos_bootstrap"},
          "| cos_boot:", res.get("cos_bootstrap"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
