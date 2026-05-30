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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from experiments._runner import (
    RESULTS,
    extract_d_hat,
    generate_batch,
    get_logger,
    get_model,
    new_run_dir,
    stratified_split,
    write_json,
)
from src.directions import ablate_dir
from src.eval import is_refusal
from src.model import format_prompt_for_bundle

log = get_logger("phase2_step3c")


def _load_split(jsonl_path: Path, seed: int = 1, n_test: int = 30):
    """Same split logic as phase2_step3_causal.py — guarantees the d̂ from
    layer L is the same direction Step 3 used at L14 (modulo extraction layer)."""
    recs = [json.loads(l) for l in jsonl_path.open()]
    harm_recs = [r for r in recs if r["label"] == "harmful"]
    harmless = [r["text"] for r in recs if r["label"] == "harmless"]
    train, test = stratified_split(
        harm_recs, key_fn=lambda r: r["source"], seed=seed, n_test=n_test,
    )
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
    harmful_train, harmful_test, harmless, _ = _load_split(
        pairs_path, seed=args.split_seed, n_test=args.n_test,
    )
    log.info("split (seed=%d): train=%d, test=%d, harmless=%d",
             args.split_seed, len(harmful_train), len(harmful_test), len(harmless))

    fmt = lambda msg: format_prompt_for_bundle(bundle, msg)

    # === Baseline (no hook) — only once; reused across all layer cells ===
    log.info("generating baseline (no hook) on %d test prompts...", len(harmful_test))
    base_gen = generate_batch(bundle, harmful_test, max_new_tokens=args.max_new_tokens)
    baseline_substr_rate = sum(is_refusal(g) for g in base_gen) / len(base_gen)
    log.info("  baseline substring refusal: %.3f", baseline_substr_rate)

    cells = {"baseline": base_gen}
    substr_rates = {"baseline": baseline_substr_rate}

    # === Sweep over extraction layers ===
    for L in args.layers:
        log.info("=== extraction layer L%d ===", L)
        log.info("  caching residuals (train+harmless) at L%d ...", L)
        d_hat, _H, _L, _meta = extract_d_hat(
            bundle, harmful_train, harmless,
            layer=L, format_fn=fmt, extra_tag="phase2",
        )
        log.info("  d_hat extracted from L%d | shape=%s", L, tuple(d_hat.shape))

        d_dev = d_hat.to(bundle.device).to(bundle.model.cfg.dtype)
        log.info("  generating with ablate(d̂_L%d) at all layers...", L)
        with ablate_dir(bundle.model, d_dev):
            abl_gen = generate_batch(bundle, harmful_test, max_new_tokens=args.max_new_tokens)

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
    print("\n  Extraction layer | substr refusal | judge refusal | Δ judge vs baseline")
    print("  -----------------|----------------|----------------|---------------------")
    for L in args.layers:
        cell = f"ablate_dhat_L{L}"
        s = substr_rates[cell]
        j = judge_summary[cell]["refusal_rate"] if judge_summary else None
        delta = (j - judge_summary["baseline"]["refusal_rate"]) if judge_summary else (s - substr_rates["baseline"])
        j_str = f"{j:.3f}" if j is not None else "—"
        print(f"  L{L:<3d}            |  {s:.3f}         |  {j_str}        |  {delta:+.3f}")
    print(f"\n  Best layer: L{best_layer} with judge-Δ = {best_drop:+.3f}")
    print("  (Phase 1 Gemma reference: L13 ablation gave Δ ≈ -0.83 on N=12, -0.91 on N=200 HarmBench.)")
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
    md.append("Per-prompt completions in `artifacts/runs/phase2_step3c/<timestamp>/result.json`.")
    return "\n".join(md) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
