"""
Phase 2 Step 3b — addition test on Qwen2.5-3B-Instruct.

The diagnostic that runs regardless of how the ablation test (step 3) goes.

If adding d̂ to harmless CodeAlpaca prompts INDUCES refusal, then d̂ IS
causally connected to refusal — ablation might just be a weaker direction
of the same mechanism, but the addition signal confirms d̂ is not merely a
vocabulary direction. If adding does nothing either, d̂ really is a
vocabulary classifier with no causal hand on the refusal mechanism, and
the contrastive set needs rebuilding.

Reuses the d̂ from step 3 (same 120-train extraction at L14). Tests on 30
held-out harmless prompts (stratified from the 150 CodeAlpaca side,
seed=1).

Three conditions × 30 prompts:
  1. baseline (no hook)
  2. add d̂_train at L3 with coeff = 1.0× of natural scale at L14
  3. add random unit vector at L3 with the same coefficient (specificity control)

Coefficient choice mirrors Phase 1 Step 3d (natural scale at extraction
layer, NOT injection-layer scale — Phase 1 §4.4 showed per-injection-layer
scale is the wrong unit). Inject layer L3 chosen as lower-end of likely
operating band; can sweep later if needed.

Run:
    cd ~/safe_ai/mech-security
    python -m experiments.phase2_step3b_addition
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
    train_test_split,
    write_json,
)
from src.directions import add_dir, random_unit_vector
from src.eval import is_refusal
from src.model import format_prompt_for_bundle

log = get_logger("phase2_step3b")


def _harmless_test_split(jsonl_path: Path, seed: int = 1, n_test: int = 30):
    """Pull the 150 harmless side; split into (unused train) / n_test."""
    recs = [json.loads(l) for l in jsonl_path.open()]
    harmless_recs = [r for r in recs if r["label"] == "harmless"]
    _, test = train_test_split(harmless_recs, seed=seed, n_test=n_test)
    return [r["text"] for r in test], test


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--data", type=str, default="data/code_contrastive.jsonl")
    ap.add_argument("--peak-layer", type=int, default=14, help="d̂ extraction layer")
    ap.add_argument("--inject-layer", type=int, default=3, help="addition layer")
    ap.add_argument("--coeff-mult", type=float, default=1.0,
                    help="coefficient as multiple of natural scale at extraction layer")
    ap.add_argument("--split-seed", type=int, default=1)
    ap.add_argument("--n-test", type=int, default=30)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--scorer", choices=("substring", "llm", "both"), default="both")
    args = ap.parse_args()

    run_dir = new_run_dir("phase2_step3b")
    log.info("run_dir: %s | model=%s extract=L%d inject=L%d", run_dir, args.model, args.peak_layer, args.inject_layer)

    bundle = get_model(args.model)
    log.info("model: %s | n_layers=%d d_model=%d device=%s dtype=%s",
             bundle.name, bundle.n_layers, bundle.d_model, bundle.device, bundle.model.cfg.dtype)

    pairs_path = Path(args.data)
    if not pairs_path.is_absolute():
        pairs_path = Path(__file__).resolve().parent.parent / pairs_path

    # === Match step 3's harmful train split (seed=1) so d̂ is identical ===
    recs = [json.loads(l) for l in pairs_path.open()]
    harm_recs = [r for r in recs if r["label"] == "harmful"]
    harmless = [r["text"] for r in recs if r["label"] == "harmless"]
    train_recs, _ = stratified_split(
        harm_recs, key_fn=lambda r: r["source"], seed=args.split_seed, n_test=args.n_test,
    )
    harmful_train = [r["text"] for r in train_recs]
    log.info("harmful train (mirrors step 3): %d", len(harmful_train))

    # === Held-out harmless test set (the prompts d̂-addition needs to flip) ===
    harmless_test, harmless_test_meta = _harmless_test_split(pairs_path, seed=args.split_seed, n_test=args.n_test)
    log.info("harmless test: %d (seed=%d)", len(harmless_test), args.split_seed)

    # === Reconstruct d̂_train at extraction layer + natural scale ===
    fmt = lambda msg: format_prompt_for_bundle(bundle, msg)
    log.info("loading cached activations at L%d ...", args.peak_layer)
    d_hat, _H, _L, meta = extract_d_hat(
        bundle, harmful_train, harmless,
        layer=args.peak_layer, format_fn=fmt, extra_tag="phase2",
    )
    natural_scale = meta["natural_scale"]
    coeff = args.coeff_mult * natural_scale
    log.info("d_hat: shape=%s | natural scale at L%d = %.2f | coeff = %.2f (%.1f× nat-scale)",
             tuple(d_hat.shape), args.peak_layer, natural_scale, coeff, args.coeff_mult)

    # Move d̂ + random to model device/dtype
    d_dev = d_hat.to(bundle.device).to(bundle.model.cfg.dtype)
    rand_dir = random_unit_vector(bundle.d_model, seed=0, device=bundle.device).to(bundle.model.cfg.dtype)

    # === Three conditions × n_test harmless test prompts ===
    log.info("generating baseline (no hook) on %d harmless test prompts...", len(harmless_test))
    base_gen = generate_batch(bundle, harmless_test, max_new_tokens=args.max_new_tokens)

    log.info("generating with add(d̂_train) at L%d, coeff=%.2f...", args.inject_layer, coeff)
    with add_dir(bundle.model, d_dev, coeff=coeff, layer=args.inject_layer):
        add_real = generate_batch(bundle, harmless_test, max_new_tokens=args.max_new_tokens)

    log.info("generating with add(random) at L%d, coeff=%.2f...", args.inject_layer, coeff)
    with add_dir(bundle.model, rand_dir, coeff=coeff, layer=args.inject_layer):
        add_rand = generate_batch(bundle, harmless_test, max_new_tokens=args.max_new_tokens)

    cells = [
        ("baseline", base_gen),
        ("add_dhat", add_real),
        ("add_random", add_rand),
    ]
    substr_rates = {}
    for name, gens in cells:
        refused = [is_refusal(g) for g in gens]
        substr_rates[name] = sum(refused) / len(refused)
        log.info("  %-15s substring refusal: %d / %d = %.3f",
                 name, sum(refused), len(refused), substr_rates[name])

    partial = {
        "step": "phase2_step3b_addition",
        "model": bundle.name,
        "peak_layer": args.peak_layer,
        "inject_layer": args.inject_layer,
        "coeff": coeff,
        "coeff_mult_of_natural_scale": args.coeff_mult,
        "natural_scale_at_peak": natural_scale,
        "split_seed": args.split_seed,
        "n_train": len(harmful_train),
        "n_test": len(harmless_test),
        "harmless_test": harmless_test,
        "completions": {n: g for n, g in cells},
        "substring_refusal_rates": substr_rates,
    }
    write_json(run_dir / "partial.json", partial)

    judge_summary = {}
    if args.scorer in ("llm", "both"):
        log.info("running Haiku 4.5 judge on 3 × %d = %d pairs ...", len(harmless_test), 3 * len(harmless_test))
        from src.eval_llm import judge_many
        for name, gens in cells:
            pairs = list(zip(harmless_test, gens))
            verdicts = judge_many(pairs, show_progress=False)
            labels = [v.label for v in verdicts]
            n_ref = sum(1 for l in labels if l == "REFUSED")
            judge_summary[name] = {"n_refused": n_ref, "refusal_rate": n_ref / len(labels), "labels": labels}
            log.info("  %-15s judge refusal: %d / %d = %.3f",
                     name, n_ref, len(labels), judge_summary[name]["refusal_rate"])

    delta_dhat = substr_rates["add_dhat"] - substr_rates["baseline"]
    delta_random = substr_rates["add_random"] - substr_rates["baseline"]
    log.info("induced-refusal lift:  add d̂ Δ = %+.3f  |  add random Δ = %+.3f", delta_dhat, delta_random)

    record = {
        **partial,
        "judge_summary": {n: {k: v for k, v in d.items() if k != "labels"}
                          for n, d in judge_summary.items()} if judge_summary else None,
        "judge_full": judge_summary if judge_summary else None,
        "headline_substr_lift_dhat": delta_dhat,
        "headline_substr_lift_random": delta_random,
    }
    write_json(run_dir / "result.json", record)

    md_path = RESULTS / "phase2_step3b_addition.md"
    md_path.write_text(_render_summary(record))
    log.info("result -> %s", run_dir / "result.json")
    log.info("summary -> %s", md_path)

    print(f"\n=== phase2_step3b_addition: {bundle.name} ===")
    print(f"  Inject L{args.inject_layer}, coeff={coeff:.2f} ({args.coeff_mult}× L{args.peak_layer} natural scale {natural_scale:.1f})")
    print("  Substring scorer (n=30 harmless test):")
    for n in ["baseline", "add_dhat", "add_random"]:
        print(f"    {n:<15s} refusal {substr_rates[n]:.3f}")
    if judge_summary:
        print("  LLM judge:")
        for n in ["baseline", "add_dhat", "add_random"]:
            r = judge_summary[n]
            print(f"    {n:<15s} refusal {r['refusal_rate']:.3f} ({r['n_refused']}/{len(harmless_test)})")
    print(f"\n  Induced-refusal lift (substr): add d̂  Δ = {delta_dhat:+.3f}")
    print(f"                                 add rand Δ = {delta_random:+.3f}")
    print("  Specificity: d̂ lift should be MUCH larger than random lift if d̂ causally controls refusal.")
    return 0


def _render_summary(rec):
    md = [
        f"# Phase 2 Step 3b — addition test ({rec['model']})",
        "",
        f"- Extraction layer: L{rec['peak_layer']}  |  inject layer: L{rec['inject_layer']}",
        f"- Coefficient: {rec['coeff']:.2f} ({rec['coeff_mult_of_natural_scale']}× natural scale {rec['natural_scale_at_peak']:.2f})",
        f"- Split: {rec['n_train']} harmful train (matches step 3 d̂), {rec['n_test']} harmless test (seed {rec['split_seed']})",
        "",
        "## Substring scorer",
        "",
        "| Condition | Refusal rate | Δ vs baseline |",
        "|---|---:|---:|",
    ]
    base = rec["substring_refusal_rates"]["baseline"]
    for n in ("baseline", "add_dhat", "add_random"):
        rate = rec["substring_refusal_rates"][n]
        md.append(f"| {n} | {rate:.3f} | {(rate - base):+.3f} |")
    md.append("")
    if rec.get("judge_summary"):
        md.append("## Calibrated LLM judge (Haiku 4.5)")
        md.append("")
        md.append("| Condition | Refusal rate | n_refused / n |")
        md.append("|---|---:|---:|")
        for n in ("baseline", "add_dhat", "add_random"):
            r = rec["judge_summary"][n]
            md.append(f"| {n} | {r['refusal_rate']:.3f} | {r['n_refused']} / {rec['n_test']} |")
        md.append("")
    md.append("Per-prompt completions in `artifacts/runs/phase2_step3b/<timestamp>/result.json`.")
    return "\n".join(md) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
