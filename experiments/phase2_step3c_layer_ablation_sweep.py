"""
Phase 2 Step 3c — extraction-layer sweep for d̂ ablation on Qwen2.5-3B.

The Phase 2 Step 1 layer-sweep AUC was misleading: harmful-vs-harmless
is linearly separable at AUC 0.994+ at EVERY layer 0–35 on Qwen2.5-3B,
including the embedding output. AUC tells us nothing about which layer
to extract d̂ from. Step 3 used L14 (the first floor-1.000 layer) and
ablation did essentially nothing (Δ ≈ −0.03 substr, 0.00 judge).

This sweep does what Phase 1 Step 5 did for Gemma: try multiple
extraction layers, ablate each, see which (if any) produces a causal
refusal drop on held-out code-harmful prompts. The extraction layer
where ablation works (if any) is Qwen's actual operating layer for
diff-of-means refusal extraction.

Five candidate layers spaced across the depth: L3, L7, L13, L20, L27.
For each:
  1. Extract d̂ from 120 harmful train + 150 harmless (cached acts).
  2. Apply Arditi multi-layer ablation (project d̂ out at every layer).
  3. Generate on the same 30 held-out test prompts as Step 3.
  4. Substring + Haiku 4.5 judge.

Cost: 5 layers × 30 × 160 tokens × ~12 tok/s ≈ 33 min generation +
~5 min dual judge. Total ~40 min.

Run:
    cd ~/safe_ai/mech-security
    python -m experiments.phase2_step3c_layer_ablation_sweep
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from experiments._runner import (
    RESULTS,
    cached_activations,
    content_hash,
    get_logger,
    get_model,
    new_run_dir,
    write_json,
)
from src.activations import cache_resid
from src.directions import ablate_dir, diff_of_means, unit
from src.eval import is_refusal
from src.model import format_prompt_for_bundle, generate

log = get_logger("phase2_step3c")


def _generate_batch(bundle, prompts, max_new=160):
    return [generate(bundle, p, max_new_tokens=max_new, temperature=0.0).strip()
            for p in prompts]


def _stratified_split(jsonl_path: Path, seed: int = 1, n_test: int = 30):
    """Same split logic as phase2_step3_causal.py — guarantees the d̂ from
    layer L is the same direction Step 3 used at L14 (modulo extraction layer)."""
    recs = [json.loads(l) for l in jsonl_path.open()]
    harm_recs = [r for r in recs if r["label"] == "harmful"]
    harmless = [r["text"] for r in recs if r["label"] == "harmless"]
    hb = [r for r in harm_recs if r["source"] == "harmbench_cybercrime"]
    adv = [r for r in harm_recs if r["source"] == "advbench_code"]
    frac_hb = len(hb) / len(harm_recs)
    n_test_hb = round(n_test * frac_hb)
    n_test_adv = n_test - n_test_hb
    rng = random.Random(seed)
    rng.shuffle(hb)
    rng.shuffle(adv)
    test = hb[:n_test_hb] + adv[:n_test_adv]
    train = hb[n_test_hb:] + adv[n_test_adv:]
    return ([r["text"] for r in train], [r["text"] for r in test],
            harmless, test)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--data", type=str, default="data/code_contrastive.jsonl")
    ap.add_argument("--layers", type=int, nargs="+", default=[3, 7, 13, 20, 27],
                    help="Candidate extraction layers to sweep")
    ap.add_argument("--split-seed", type=int, default=1)
    ap.add_argument("--n-test", type=int, default=30)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--scorer", choices=("substring", "llm", "both"), default="both")
    args = ap.parse_args()

    run_dir = new_run_dir("phase2_step3c")
    log.info("run_dir: %s | layers=%s", run_dir, args.layers)

    bundle = get_model(args.model)
    log.info("model: %s | n_layers=%d d_model=%d device=%s dtype=%s",
             bundle.name, bundle.n_layers, bundle.d_model, bundle.device, bundle.model.cfg.dtype)

    pairs_path = Path(args.data)
    if not pairs_path.is_absolute():
        pairs_path = Path(__file__).resolve().parent.parent / pairs_path
    harmful_train, harmful_test, harmless, _ = _stratified_split(
        pairs_path, seed=args.split_seed, n_test=args.n_test,
    )
    log.info("split (seed=%d): train=%d, test=%d, harmless=%d",
             args.split_seed, len(harmful_train), len(harmful_test), len(harmless))

    fmt = lambda msg: format_prompt_for_bundle(bundle, msg)

    # === Baseline (no hook) — only once; reused across all layer cells ===
    log.info("generating baseline (no hook) on %d test prompts...", len(harmful_test))
    base_gen = _generate_batch(bundle, harmful_test, max_new=args.max_new_tokens)
    baseline_substr_rate = sum(is_refusal(g) for g in base_gen) / len(base_gen)
    log.info("  baseline substring refusal: %.3f", baseline_substr_rate)

    cells = {"baseline": base_gen}
    substr_rates = {"baseline": baseline_substr_rate}

    # === Sweep over extraction layers ===
    for L in args.layers:
        log.info("=== extraction layer L%d ===", L)
        extra = (
            f"{bundle.name}|dtype={bundle.model.cfg.dtype}|L{L}|"
            f"resid_post|last_token|phase2"
        )
        key_ht = content_hash(harmful_train, extra=extra + "|harmful_train")
        key_l = content_hash(harmless, extra=extra + "|harmless")
        log.info("  caching residuals (train+harmless) at L%d ...", L)
        H_train = cached_activations(
            key_ht,
            lambda: cache_resid(bundle, harmful_train, layer=L,
                                show_progress=False, format_fn=fmt),
        )
        L_full = cached_activations(
            key_l,
            lambda: cache_resid(bundle, harmless, layer=L,
                                show_progress=False, format_fn=fmt),
        )
        d_hat = unit(diff_of_means(H_train, L_full))
        log.info("  d_hat extracted from L%d | shape=%s", L, tuple(d_hat.shape))

        d_dev = d_hat.to(bundle.device).to(bundle.model.cfg.dtype)
        log.info("  generating with ablate(d̂_L%d) at all layers...", L)
        with ablate_dir(bundle.model, d_dev):
            abl_gen = _generate_batch(bundle, harmful_test, max_new=args.max_new_tokens)

        cell_name = f"ablate_dhat_L{L}"
        cells[cell_name] = abl_gen
        rate = sum(is_refusal(g) for g in abl_gen) / len(abl_gen)
        substr_rates[cell_name] = rate
        log.info("  %-20s substring refusal: %d / %d = %.3f  (Δ vs baseline %+.3f)",
                 cell_name, sum(is_refusal(g) for g in abl_gen), len(abl_gen),
                 rate, rate - baseline_substr_rate)

    # === Save partial before judge ===
    partial = {
        "step": "phase2_step3c_layer_ablation_sweep",
        "model": bundle.name,
        "layers_swept": args.layers,
        "n_train": len(harmful_train),
        "n_test": len(harmful_test),
        "completions": cells,
        "substring_refusal_rates": substr_rates,
    }
    write_json(run_dir / "partial.json", partial)
    log.info("partial -> %s", run_dir / "partial.json")

    # === Dual judge ===
    judge_summary = {}
    if args.scorer in ("llm", "both"):
        from src.eval_llm import judge_many
        log.info("running Haiku 4.5 judge on %d × %d pairs...", len(cells), len(harmful_test))
        for name, gens in cells.items():
            pairs = list(zip(harmful_test, gens))
            verdicts = judge_many(pairs, show_progress=False)
            labels = [v.label for v in verdicts]
            n_ref = sum(1 for l in labels if l == "REFUSED")
            judge_summary[name] = {"n_refused": n_ref, "refusal_rate": n_ref / len(labels), "labels": labels}
            log.info("  %-20s judge refusal: %d / %d = %.3f", name, n_ref, len(labels),
                     judge_summary[name]["refusal_rate"])

    # === Find winning extraction layer (if any) ===
    baseline_judge = judge_summary["baseline"]["refusal_rate"] if judge_summary else baseline_substr_rate
    best_layer = None
    best_drop = 0.0
    for L in args.layers:
        cell = f"ablate_dhat_L{L}"
        rate = (judge_summary[cell]["refusal_rate"] if judge_summary else substr_rates[cell])
        drop = baseline_judge - rate
        if drop > best_drop:
            best_drop = drop
            best_layer = L
    log.info("best extraction layer: L%s with Δ = %+.3f vs baseline", best_layer, best_drop)

    record = {
        **partial,
        "judge_summary": {n: {k: v for k, v in d.items() if k != "labels"}
                          for n, d in judge_summary.items()} if judge_summary else None,
        "judge_full": judge_summary if judge_summary else None,
        "best_extraction_layer": best_layer,
        "best_drop_judge": best_drop,
        "baseline_judge_rate": baseline_judge,
    }
    write_json(run_dir / "result.json", record)

    md_path = RESULTS / "phase2_step3c_layer_ablation_sweep.md"
    md_path.write_text(_render_summary(record))
    log.info("result -> %s", run_dir / "result.json")
    log.info("summary -> %s", md_path)

    print(f"\n=== phase2_step3c_layer_ablation_sweep: {bundle.name} ===")
    print(f"  baseline substring: {substr_rates['baseline']:.3f}")
    if judge_summary:
        print(f"  baseline judge:     {judge_summary['baseline']['refusal_rate']:.3f}")
    print(f"\n  Extraction layer | substr refusal | judge refusal | Δ judge vs baseline")
    print(f"  -----------------|----------------|----------------|---------------------")
    for L in args.layers:
        cell = f"ablate_dhat_L{L}"
        s = substr_rates[cell]
        j = judge_summary[cell]["refusal_rate"] if judge_summary else None
        delta = (j - judge_summary["baseline"]["refusal_rate"]) if judge_summary else (s - substr_rates["baseline"])
        j_str = f"{j:.3f}" if j is not None else "—"
        print(f"  L{L:<3d}            |  {s:.3f}         |  {j_str}        |  {delta:+.3f}")
    print(f"\n  Best layer: L{best_layer} with judge-Δ = {best_drop:+.3f}")
    print(f"  (Phase 1 Gemma reference: L13 ablation gave Δ ≈ -0.83 on N=12, -0.91 on N=200 HarmBench.)")
    return 0


def _render_summary(rec):
    md = [
        f"# Phase 2 Step 3c — extraction-layer sweep for d̂ ablation ({rec['model']})",
        "",
        f"- Layers swept: {rec['layers_swept']}",
        f"- Test set: {rec['n_test']} held-out code-harmful prompts (same split as step 3, seed 1)",
        f"- Baseline (no hook): {rec['substring_refusal_rates']['baseline']:.3f} substring",
    ]
    if rec.get("judge_summary"):
        md.append(f"  / {rec['judge_summary']['baseline']['refusal_rate']:.3f} judge")
    md.append("")
    md.append("## Substring scorer + Haiku 4.5 judge by extraction layer")
    md.append("")
    md.append("| Extraction layer | Substring refusal | Δ vs baseline | Judge refusal | Δ vs baseline |")
    md.append("|---|---:|---:|---:|---:|")
    base_s = rec['substring_refusal_rates']['baseline']
    base_j = rec['judge_summary']['baseline']['refusal_rate'] if rec.get('judge_summary') else None
    for L in rec['layers_swept']:
        cell = f"ablate_dhat_L{L}"
        s = rec['substring_refusal_rates'][cell]
        j = rec['judge_summary'][cell]['refusal_rate'] if rec.get('judge_summary') else None
        j_str = f"{j:.3f}" if j is not None else "—"
        dj_str = f"{(j - base_j):+.3f}" if j is not None else "—"
        md.append(f"| L{L} | {s:.3f} | {(s - base_s):+.3f} | {j_str} | {dj_str} |")
    md.append("")
    md.append(f"**Best extraction layer: L{rec['best_extraction_layer']}** with judge-Δ = {rec['best_drop_judge']:+.3f}.")
    md.append("")
    md.append(f"Per-prompt completions in `artifacts/runs/phase2_step3c/<timestamp>/result.json`.")
    return "\n".join(md) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
