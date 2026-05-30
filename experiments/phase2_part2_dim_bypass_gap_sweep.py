"""Phase 2 Part 2 — diff-of-means + bypass-gap layer/position sweep.

The untested 2×2 cell from PROJECT_STATE.md: extraction method = diff-of-means,
layer-selection criterion = bypass-gap. Prior from Step 4b (DIM-matched in
Wollschläger's harness) flags L20–L24 around positions {−4, −1} as the
candidate region.

Procedure:
  - Standard matched-set split (30/10, seed=1) — byte-identical to step3e.
  - For each (layer, pos) in [L19..L25] × [{-1, -4}]:
      1. Cache resid_post at (layer, pos) for harmful_train (30) + harmless_train (30).
      2. d̂_(L, pos) = unit(mean(H) - mean(L)).
      3. Ablate d̂ via TransformerLens hook during greedy generation on 10
         held-out matched-harmful prompts.
      4. Score completions: substring + Haiku 4.5 judge. Coherence = mean_chars.
  - Identify best cell (lowest judge refusal rate). Run two controls:
      - Random-direction at matched magnitude (specificity).
      - Read 3 completions verbatim (coherence sentinel — load-bearing
        because Step 4b's KL guard flagged these cells as potentially
        disruptive).

Interpretation:
  - Coherent refusal→compliance + d̂-specificity → layer-selection-artifact
    confirmed: plain diff-of-means at the bypass-gap-selected layer IS causal;
    L14 inertness was the AUC-saturation artifact.
  - Degraded/incoherent → KL guard was right, "bypass" was disruption.

Outputs:
  - artifacts/runs/phase2_part2/<timestamp>/result.json — per-cell scores +
    completions.
  - results/phase2_part2_dim_bypass_gap_sweep.md — written by hand after the
    run; this script just emits the JSON.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import dotenv
import numpy as np

dotenv.load_dotenv()  # for ANTHROPIC_API_KEY

from experiments._runner import (  # noqa: E402
    RESULTS,
    generate_batch,
    get_logger,
    get_model,
    new_run_dir,
    write_json,
)
from mech_security.activations import cache_resid  # noqa: E402
from mech_security.directions import ablate_dir, diff_of_means, random_unit_vector, unit  # noqa: E402
from mech_security.eval import is_refusal  # noqa: E402
from mech_security.eval_llm import JUDGE_MODEL, judge_many, refusal_rate_llm  # noqa: E402
from mech_security.model import format_prompt_for_bundle  # noqa: E402

log = get_logger("phase2_part2")


def split_matched(jsonl_path: Path, seed: int = 1, n_test: int = 10):
    """Byte-faithful to experiments/phase2_step3e_matched_set_sweep._split."""
    import random
    recs = [json.loads(l) for l in jsonl_path.open()]
    harmful = [r["text"] for r in recs if r["label"] == "harmful"]
    harmless = [r["text"] for r in recs if r["label"] == "harmless"]
    rng = random.Random(seed)
    rng.shuffle(harmful)
    rng.shuffle(harmless)
    return harmful[n_test:], harmful[:n_test], harmless[n_test:], harmless[:n_test]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/code_contrastive_matched.jsonl")
    ap.add_argument("--layers", type=int, nargs="+", default=[19, 20, 21, 22, 23, 24, 25],
                    help="Layers to sweep (default L19-L25 from Step 4b prior)")
    ap.add_argument("--positions", type=int, nargs="+", default=[-1, -4],
                    help="Sequence positions to extract at")
    ap.add_argument("--split_seed", type=int, default=1)
    ap.add_argument("--n_test", type=int, default=10)
    ap.add_argument("--max_new_tokens", type=int, default=160)
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--judge", action="store_true", default=True,
                    help="Run dual-judge (Haiku 4.5). Default ON.")
    ap.add_argument("--no_judge", action="store_false", dest="judge")
    args = ap.parse_args()

    run_dir = new_run_dir("phase2_part2")
    log.info("run_dir: %s", run_dir)

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = Path(__file__).resolve().parent.parent / data_path
    h_train, h_test, l_train, l_test = split_matched(data_path, args.split_seed, args.n_test)
    log.info("matched split (seed=%d): harm train=%d test=%d, "
             "harml train=%d test=%d",
             args.split_seed, len(h_train), len(h_test), len(l_train), len(l_test))

    log.info("loading model: %s", args.model)
    bundle = get_model(args.model)
    fmt = lambda msg: format_prompt_for_bundle(bundle, msg)  # noqa: E731

    log.info("=== generating baseline (no hook) on %d harmful_test ===", len(h_test))
    base_gens = generate_batch(bundle, h_test, max_new_tokens=args.max_new_tokens)
    base_substr = sum(is_refusal(g) for g in base_gens) / len(base_gens)
    base_mean_chars = float(np.mean([len(g) for g in base_gens]))
    log.info("baseline: substr refusal=%.3f mean_chars=%.0f", base_substr, base_mean_chars)

    base_judge_rate = None
    if args.judge:
        log.info("judging baseline (%d completions, Haiku 4.5)...", len(base_gens))
        base_verdicts = judge_many(list(zip(h_test, base_gens)), show_progress=False)
        base_judge = refusal_rate_llm(base_verdicts)
        base_judge_rate = (base_judge["refused"] + 0.5 * base_judge["partial_comply"]) / base_judge["n"]
        log.info("baseline judge: refused=%d/%d partial=%d complied=%d → score=%.3f",
                 base_judge["refused"], base_judge["n"], base_judge["partial_comply"],
                 base_judge["complied"], base_judge_rate)

    cells = []
    total_cells = len(args.layers) * len(args.positions)
    cell_idx = 0

    for L in args.layers:
        for pos in args.positions:
            cell_idx += 1
            cell_name = f"L{L}_pos{pos}"
            log.info("[%d/%d] === cell %s: cache resid at L%d pos%d ===",
                     cell_idx, total_cells, cell_name, L, pos)
            H = cache_resid(bundle, h_train, layer=L, position=pos, format_fn=fmt, show_progress=False)
            Lact = cache_resid(bundle, l_train, layer=L, position=pos, format_fn=fmt, show_progress=False)
            d_raw = diff_of_means(H, Lact)
            d_hat = unit(d_raw)
            natural_scale = float((H @ d_hat).mean().item())
            log.info("[%d/%d] %s: natural_scale=%.3f d_hat.norm=%.3f",
                     cell_idx, total_cells, cell_name, natural_scale, d_hat.norm().item())

            d_hat_dev = d_hat.to(bundle.model.cfg.device).to(bundle.model.cfg.dtype)
            log.info("[%d/%d] generating with ablate(d_hat_%s)...",
                     cell_idx, total_cells, cell_name)
            with ablate_dir(bundle.model, d_hat_dev):
                abl_gens = generate_batch(bundle, h_test, max_new_tokens=args.max_new_tokens)
            abl_substr = sum(is_refusal(g) for g in abl_gens) / len(abl_gens)
            abl_mean_chars = float(np.mean([len(g) for g in abl_gens]))
            log.info("[%d/%d] %s: substr refusal=%.3f mean_chars=%.0f",
                     cell_idx, total_cells, cell_name, abl_substr, abl_mean_chars)

            abl_judge_rate = None
            judge_breakdown = None
            if args.judge:
                verdicts = judge_many(list(zip(h_test, abl_gens)), show_progress=False)
                judge = refusal_rate_llm(verdicts)
                abl_judge_rate = (judge["refused"] + 0.5 * judge["partial_comply"]) / judge["n"]
                judge_breakdown = judge
                log.info("[%d/%d] %s: judge refused=%d/%d partial=%d complied=%d → %.3f",
                         cell_idx, total_cells, cell_name,
                         judge["refused"], judge["n"], judge["partial_comply"],
                         judge["complied"], abl_judge_rate)

            cells.append({
                "name": cell_name, "layer": L, "position": pos,
                "natural_scale": natural_scale,
                "substr_refusal": abl_substr,
                "judge_refusal": abl_judge_rate,
                "judge_breakdown": judge_breakdown,
                "mean_chars": abl_mean_chars,
                "completions": abl_gens,
                "d_hat_cpu": d_hat.tolist(),  # for cosine analyses later
            })

    # Best cell by judge refusal (lower = better bypass), tiebreak by substring
    sortable = [c for c in cells if c["judge_refusal"] is not None] or cells
    best = min(sortable, key=lambda c: (c["judge_refusal"] or 1.0, c["substr_refusal"]))
    log.info("=== best cell: %s | judge=%s substr=%.3f ===",
             best["name"], best["judge_refusal"], best["substr_refusal"])

    log.info("=== specificity: random unit vector at best cell %s ===", best["name"])
    rand_dir = random_unit_vector(bundle.model.cfg.d_model, seed=42)
    rand_dir = rand_dir.to(bundle.model.cfg.device).to(bundle.model.cfg.dtype)
    with ablate_dir(bundle.model, rand_dir):
        rand_gens = generate_batch(bundle, h_test, max_new_tokens=args.max_new_tokens)
    rand_substr = sum(is_refusal(g) for g in rand_gens) / len(rand_gens)
    rand_mean_chars = float(np.mean([len(g) for g in rand_gens]))
    log.info("random control: substr=%.3f mean_chars=%.0f", rand_substr, rand_mean_chars)
    rand_judge_rate = None
    rand_judge_breakdown = None
    if args.judge:
        rand_verdicts = judge_many(list(zip(h_test, rand_gens)), show_progress=False)
        rand_judge_breakdown = refusal_rate_llm(rand_verdicts)
        rand_judge_rate = (rand_judge_breakdown["refused"] + 0.5 * rand_judge_breakdown["partial_comply"]) / rand_judge_breakdown["n"]
        log.info("random control judge: refused=%d/%d → %.3f",
                 rand_judge_breakdown["refused"], rand_judge_breakdown["n"], rand_judge_rate)

    result = {
        "model": args.model,
        "data": str(data_path),
        "split_seed": args.split_seed,
        "n_test": args.n_test,
        "layers": args.layers,
        "positions": args.positions,
        "judge_model": JUDGE_MODEL if args.judge else None,
        "max_new_tokens": args.max_new_tokens,
        "baseline": {
            "substr_refusal": base_substr,
            "judge_refusal": base_judge_rate,
            "mean_chars": base_mean_chars,
            "completions": base_gens,
        },
        "cells": cells,
        "best_cell": best["name"],
        "specificity_random": {
            "substr_refusal": rand_substr,
            "judge_refusal": rand_judge_rate,
            "mean_chars": rand_mean_chars,
            "judge_breakdown": rand_judge_breakdown,
            "completions": rand_gens,
        },
    }
    write_json(run_dir / "result.json", result)
    log.info("wrote %s", run_dir / "result.json")

    # console summary
    print("\n=== Phase 2 Part 2 summary ===")
    print(f"baseline substr refusal: {base_substr:.3f} | judge: {base_judge_rate}")
    print(f"\n{'cell':<10} {'natscale':>9} {'substr':>7} {'judge':>7} {'mean_chars':>11}")
    for c in cells:
        j = f"{c['judge_refusal']:.3f}" if c["judge_refusal"] is not None else "  N/A"
        print(f"{c['name']:<10} {c['natural_scale']:>9.3f} {c['substr_refusal']:>7.3f} {j:>7} {c['mean_chars']:>11.0f}")
    print(f"\nbest cell: {best['name']}")
    print(f"random control: substr={rand_substr:.3f} judge={rand_judge_rate}")
    print(f"\n3 completions at best cell ({best['name']}) for coherence read:")
    for i, g in enumerate(best["completions"][:3]):
        print(f"\n  [{i+1}] {g[:300]}{'...' if len(g) > 300 else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
