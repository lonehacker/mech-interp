"""Phase 2 Part 2 PIECE 1 — firm the headline N at the pre-committed cells.

Reruns the pre-committed prior cells (L22_pos-1, L22_pos-4, L23_pos-1) at
n_test=30 held-out matched-harmful so the headline number is X/30 instead
of X/10. Same harness, dual-judge, greedy. Random-direction specificity
control at every cell × 3 seeds at the SAME N so the d̂-vs-random contrast
stays clean.

Split: from the 40-prompt matched-harmful set (seed=1 shuffle), test =
harmful[:30] (was [:10]), train = harmful[30:] (was [10:]). d̂ is therefore
re-extracted from a smaller (n=10) train slice than the original Part 2
run's n=30 train — the d̂ direction is in the same neighborhood but not
byte-identical to the original. The writeup notes this explicitly; the
n=10/30-test result is the headline number, the n=30/10-test result from
the original Part 2 sweep stays as confirmation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import dotenv
import numpy as np

dotenv.load_dotenv()

from experiments._runner import (  # noqa: E402
    generate_batch,
    get_logger,
    get_model,
    new_run_dir,
    write_json,
)
from mech_security.activations import cache_resid  # noqa: E402
from mech_security.directions import (  # noqa: E402
    ablate_dir,
    bypass_gap,
    diff_of_means,
    random_unit_vector,
    unit,
)
from mech_security.eval import is_refusal  # noqa: E402
from mech_security.eval_llm import JUDGE_MODEL, judge_many, refusal_rate_llm  # noqa: E402
from mech_security.model import format_prompt_for_bundle  # noqa: E402

log = get_logger("phase2_part2_firm_n30")

PRIOR_CELLS = [(22, -1), (22, -4), (23, -1)]
RANDOM_SEEDS = [42, 1337, 0xBEEF]


def split_matched(jsonl_path: Path, seed: int = 1, n_test: int = 30):
    """Same shuffle as step3e (seed=1), bigger test slice. With n_test=30,
    train = harmful[30:] (n=10), test = harmful[:30]."""
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
    ap.add_argument("--n_test", type=int, default=30,
                    help="Test split size (planner: 30-50; matched set max 40).")
    ap.add_argument("--max_new_tokens", type=int, default=160)
    args = ap.parse_args()

    run_dir = new_run_dir("phase2_part2_firm_n30")
    log.info("run_dir: %s  n_test=%d", run_dir, args.n_test)

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = Path(__file__).resolve().parent.parent / data_path
    h_train, h_test, l_train, l_test = split_matched(data_path, seed=1, n_test=args.n_test)
    log.info("matched split (seed=1, n_test=%d): harm train=%d test=%d, harml train=%d test=%d",
             args.n_test, len(h_train), len(h_test), len(l_train), len(l_test))

    log.info("loading model: %s", args.model)
    bundle = get_model(args.model)
    fmt = lambda msg: format_prompt_for_bundle(bundle, msg)  # noqa: E731

    log.info("=== baseline generation on %d harmful_test ===", len(h_test))
    base = generate_batch(bundle, h_test, max_new_tokens=args.max_new_tokens, temperature=0.0)
    base_substr = sum(is_refusal(g) for g in base) / len(base)
    base_mc = float(np.mean([len(g) for g in base]))
    log.info("baseline: substr=%.3f mean_chars=%.0f", base_substr, base_mc)
    base_verdicts = judge_many(list(zip(h_test, base)), show_progress=False)
    base_judge = refusal_rate_llm(base_verdicts)
    log.info("baseline judge: refused=%d/%d partial=%d complied=%d",
             base_judge["refused"], base_judge["n"],
             base_judge["partial_comply"], base_judge["complied"])

    cells_out = []
    randoms_out = []
    for (L, pos) in PRIOR_CELLS:
        cell_name = f"L{L}_pos{pos}"
        log.info("=== %s: extract d̂ from %d train ===", cell_name, len(h_train))
        H = cache_resid(bundle, h_train, layer=L, position=pos, format_fn=fmt, show_progress=False)
        Lact = cache_resid(bundle, l_train, layer=L, position=pos, format_fn=fmt, show_progress=False)
        d_hat = unit(diff_of_means(H, Lact))
        natural_scale = float((H @ d_hat).mean().item())
        log.info("%s: natural_scale=%.3f d_hat.norm=%.3f", cell_name, natural_scale, d_hat.norm().item())

        # d̂ ablation via bypass_gap (which handles baseline reuse + completion capture)
        d_hat_dev = d_hat.to(bundle.model.cfg.device).to(bundle.model.cfg.dtype)
        log.info("%s: ablate(d̂) on %d test ...", cell_name, len(h_test))
        gap = bypass_gap(
            bundle, d_hat_dev, h_test,
            baseline_completions=base, max_new_tokens=args.max_new_tokens,
        )
        gens = gap["ablated_completions"]
        substr = gap["ablated_refusal"]
        mc = gap["mean_chars_ablated"]
        verdicts = judge_many(list(zip(h_test, gens)), show_progress=False)
        j = refusal_rate_llm(verdicts)
        log.info("%s: d̂ substr=%.3f gap=%.3f mc=%.0f | judge R=%d P=%d C=%d",
                 cell_name, substr, gap["gap"], mc,
                 j["refused"], j["partial_comply"], j["complied"])
        cells_out.append({
            "cell": cell_name, "layer": L, "position": pos,
            "natural_scale": natural_scale,
            "substr_refusal": substr, "substr_gap": gap["gap"],
            "judge": j, "mean_chars": mc, "completions": gens,
            "d_hat_cpu": d_hat.tolist(),
        })

        # Random control at SAME cell, same N, 3 seeds
        for seed in RANDOM_SEEDS:
            log.info("%s: random unit vec seed=%d ...", cell_name, seed)
            rand_dir = random_unit_vector(bundle.model.cfg.d_model, seed=seed).to(
                bundle.model.cfg.device).to(bundle.model.cfg.dtype)
            r_gap = bypass_gap(
                bundle, rand_dir, h_test,
                baseline_completions=base, max_new_tokens=args.max_new_tokens,
            )
            r_gens = r_gap["ablated_completions"]
            r_substr = r_gap["ablated_refusal"]
            r_mc = r_gap["mean_chars_ablated"]
            r_verdicts = judge_many(list(zip(h_test, r_gens)), show_progress=False)
            rj = refusal_rate_llm(r_verdicts)
            log.info("%s seed=%d: random substr=%.3f mc=%.0f | judge R=%d P=%d C=%d",
                     cell_name, seed, r_substr, r_mc,
                     rj["refused"], rj["partial_comply"], rj["complied"])
            randoms_out.append({
                "cell": cell_name, "layer": L, "position": pos, "seed": seed,
                "substr_refusal": r_substr, "substr_gap": r_gap["gap"],
                "judge": rj, "mean_chars": r_mc, "completions": r_gens,
            })

    result = {
        "model": args.model, "data": str(data_path),
        "split_seed": 1, "n_test": args.n_test,
        "prior_cells": PRIOR_CELLS, "random_seeds": RANDOM_SEEDS,
        "judge_model": JUDGE_MODEL, "max_new_tokens": args.max_new_tokens,
        "baseline": {
            "substr_refusal": base_substr, "judge": base_judge,
            "mean_chars": base_mc, "completions": base,
        },
        "d_hat_ablations": cells_out,
        "random_ablations": randoms_out,
    }
    write_json(run_dir / "result.json", result)
    log.info("wrote %s", run_dir / "result.json")

    print("\n=== Phase 2 Part 2 firmed (n_test=%d) ===" % args.n_test)
    print(f"baseline: substr={base_substr:.3f} judge R={base_judge['refused']}/{base_judge['n']} "
          f"P={base_judge['partial_comply']} C={base_judge['complied']} mc={base_mc:.0f}")
    print(f"\n{'cell':<10} {'method':<10} {'substr':>7} {'refused':>8} {'partial':>8} {'complied':>9} {'mc':>5}")
    for c in cells_out:
        j = c["judge"]
        print(f"{c['cell']:<10} {'d̂':<10} {c['substr_refusal']:>7.3f} "
              f"{j['refused']:>8} {j['partial_comply']:>8} {j['complied']:>9} {c['mean_chars']:>5.0f}")
    print()
    for c in randoms_out:
        j = c["judge"]
        method = f"rand_{c['seed']}"
        print(f"{c['cell']:<10} {method:<10} {c['substr_refusal']:>7.3f} "
              f"{j['refused']:>8} {j['partial_comply']:>8} {j['complied']:>9} {c['mean_chars']:>5.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
