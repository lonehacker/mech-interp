"""Phase 2 Part 2 — follow-up specificity control at the PRIOR-COMMITTED cells.

The main sweep (phase2_part2_dim_bypass_gap_sweep.py) auto-runs a
random-direction ablation only at its min-by-composite "best" cell. If
the pre-committed prior cell (from Step 4b: L20-L24 × pos {-4, -1}) is
NOT the auto-selected one, we still need random-direction completions
at the prior cell to decide:

  Reading A — coherent d̂-specific bypass at the prior cell (random
              ablation at the same cell does NOT also collapse refusal,
              and even if it drops the rate the output is degraded/noisy).
              Layer-selection-artifact confirmed.

  Reading B — random ablation at the SAME cell also produces 10/10
              "refusal-free" output AND that output is incoherent / off-
              answer / repetitive. Then ablation at this layer disrupts
              the model into not-refusing rather than removing a refusal
              direction. KL guard was right.

This script runs ablation with a fresh random unit vector at each of
[L22_pos-1, L22_pos-4, L23_pos-1] (the cells the main run flagged at
10/10 compliance — see live log) and dumps full completions for the
coherence read. mean_chars vs baseline is computed for the disruption
flag.

Greedy (T=0) explicitly. Three random seeds per cell to bound the
random-direction noise floor.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import dotenv
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
dotenv.load_dotenv()

from experiments._runner import (  # noqa: E402
    generate_batch,
    get_logger,
    get_model,
    new_run_dir,
    write_json,
)
from src.directions import bypass_gap, random_unit_vector  # noqa: E402
from src.eval import is_refusal  # noqa: E402
from src.eval_llm import JUDGE_MODEL, judge_many, refusal_rate_llm  # noqa: E402

log = get_logger("phase2_part2_random_at_prior")

# Pre-committed cells from Step 4b prior (Wollschläger DIM-matched).
# Anchor specifically on the L22+L23 region the main run lit up.
PRIOR_CELLS = [(22, -1), (22, -4), (23, -1)]
RANDOM_SEEDS = [42, 1337, 0xBEEF]


def split_matched(jsonl_path: Path, seed: int = 1, n_test: int = 10):
    import random
    recs = [json.loads(l) for l in jsonl_path.open()]
    harmful = [r["text"] for r in recs if r["label"] == "harmful"]
    harmless = [r["text"] for r in recs if r["label"] == "harmless"]
    rng = random.Random(seed)
    rng.shuffle(harmful); rng.shuffle(harmless)
    return harmful[n_test:], harmful[:n_test], harmless[n_test:], harmless[:n_test]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/code_contrastive_matched.jsonl")
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--max_new_tokens", type=int, default=160)
    args = ap.parse_args()

    run_dir = new_run_dir("phase2_part2_random_at_prior")
    log.info("run_dir: %s", run_dir)

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = Path(__file__).resolve().parent.parent / data_path
    _, h_test, _, _ = split_matched(data_path)
    log.info("loaded matched test set: %d harmful held-out prompts", len(h_test))

    log.info("loading model: %s", args.model)
    bundle = get_model(args.model)
    d_model = bundle.model.cfg.d_model

    log.info("generating baseline (no hook)...")
    base = generate_batch(bundle, h_test, max_new_tokens=args.max_new_tokens, temperature=0.0)
    base_substr = sum(is_refusal(g) for g in base) / len(base)
    base_mc = float(np.mean([len(g) for g in base]))
    log.info("baseline substr refusal=%.3f mean_chars=%.0f", base_substr, base_mc)
    base_verdicts = judge_many(list(zip(h_test, base)), show_progress=False)
    base_judge = refusal_rate_llm(base_verdicts)
    log.info("baseline judge refused=%d/%d partial=%d complied=%d",
             base_judge["refused"], base_judge["n"],
             base_judge["partial_comply"], base_judge["complied"])

    out_cells = []
    for (layer, pos) in PRIOR_CELLS:
        cell_label = f"L{layer}_pos{pos}"
        log.info("=== %s ===", cell_label)
        for seed in RANDOM_SEEDS:
            log.info("  random unit vec seed=%d at %s ...", seed, cell_label)
            rand_dir = random_unit_vector(d_model, seed=seed).to(bundle.model.cfg.device).to(bundle.model.cfg.dtype)
            # bypass_gap: substring-scorer gap + completions + mean_chars in one call.
            # Pass the precomputed baseline so we don't regenerate it per seed.
            gap = bypass_gap(
                bundle, rand_dir, h_test,
                baseline_completions=base, max_new_tokens=args.max_new_tokens,
            )
            gens = gap["ablated_completions"]
            substr = gap["ablated_refusal"]
            mc = gap["mean_chars_ablated"]
            verdicts = judge_many(list(zip(h_test, gens)), show_progress=False)
            j = refusal_rate_llm(verdicts)
            log.info("  seed=%d: substr=%.3f gap=%.3f mean_chars=%.0f | judge refused=%d partial=%d complied=%d",
                     seed, substr, gap["gap"], mc, j["refused"], j["partial_comply"], j["complied"])
            out_cells.append({
                "cell": cell_label, "layer": layer, "position": pos, "seed": seed,
                "substr_refusal": substr, "substr_gap": gap["gap"],
                "judge": j, "mean_chars": mc, "completions": gens,
            })

    result = {
        "model": args.model, "data": str(data_path),
        "prior_cells": PRIOR_CELLS, "random_seeds": RANDOM_SEEDS,
        "judge_model": JUDGE_MODEL,
        "baseline": {
            "substr_refusal": base_substr, "mean_chars": base_mc,
            "judge": base_judge, "completions": base,
        },
        "random_ablations": out_cells,
    }
    write_json(run_dir / "result.json", result)
    log.info("wrote %s", run_dir / "result.json")

    print("\n=== Random-direction ablation at prior cells (3 seeds each) ===")
    print(f"baseline: substr={base_substr:.3f} judge_complied={base_judge['complied']}/{base_judge['n']} mean_chars={base_mc:.0f}")
    print(f"\n{'cell':<10} {'seed':>8} {'substr':>7} {'refused':>8} {'partial':>8} {'complied':>9} {'mc':>5}")
    for c in out_cells:
        j = c["judge"]
        print(f"{c['cell']:<10} {c['seed']:>8} {c['substr_refusal']:>7.3f} "
              f"{j['refused']:>8} {j['partial_comply']:>8} {j['complied']:>9} {c['mean_chars']:>5.0f}")

    # Read disruption flag: if random ablation at any L22/L23 cell ALSO produces
    # 10/10 refusal-free + similar mean_chars, that's the smoking gun for Reading B.
    return 0


if __name__ == "__main__":
    sys.exit(main())
