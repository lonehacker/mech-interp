"""
Phase 2 Step 3 — causal test on Qwen2.5-3B-Instruct.

Mirrors phase1_step3_steering.py for Phase 2. The decisive test:

Given a candidate refusal direction d̂ extracted at L14 from a TRAIN split
of code_contrastive.jsonl, does Arditi-style multi-layer ablation of d̂
collapse refusal on a HELD-OUT split?

This is what cuts through the step 1b flag — random-direction AUC at L14
is wide and high (mean 0.40, range [0.05, 0.98]) because the contrastive
set has a strong vocabulary/topic shift, not because L14 d̂ is the wrong
direction. If ablating L14 d̂ collapses refusal, the direction IS a causal
refusal mechanism regardless of how the AUC reads against random.

Setup:
  - Split 150 code-harmful into 120 train / 30 test (seed=1, stratified
    so HarmBench cybercrime + AdvBench-code preserve their ratios).
  - Extract d̂_train at L14 from 120 train + 150 harmless.
  - On 30 held-out code-harmful, generate completions under three conditions:
      1. baseline (no hook)
      2. ablate d̂_train (Arditi multi-layer)
      3. ablate random unit vector (specificity control)
  - Score with substring + dual LLM judge (Haiku 4.5 + Opus 4.7).

Run:
    cd ~/safe_ai/mech-security
    python -m experiments.phase2_step3_causal
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
from src.directions import ablate_dir, random_unit_vector
from src.eval import is_refusal
from src.model import format_prompt_for_bundle

log = get_logger("phase2_step3")


def _load_split(jsonl_path: Path, seed: int = 1, n_test: int = 30):
    """Split code_contrastive.jsonl harmful side into train/test, preserving
    HarmBench cybercrime vs AdvBench-code ratios.
    Returns (harmful_train, harmful_test, harmless, harmful_test_meta).
    """
    recs = [json.loads(l) for l in jsonl_path.open()]
    harm_recs = [r for r in recs if r["label"] == "harmful"]
    harmless = [r["text"] for r in recs if r["label"] == "harmless"]
    hb_n = sum(1 for r in harm_recs if r["source"] == "harmbench_cybercrime")
    adv_n = sum(1 for r in harm_recs if r["source"] == "advbench_code")
    log.info("harmful pool: %d HarmBench cybercrime + %d AdvBench-code = %d",
             hb_n, adv_n, len(harm_recs))

    train, test = stratified_split(
        harm_recs, key_fn=lambda r: r["source"], seed=seed, n_test=n_test,
    )
    log.info("split (seed=%d): train=%d, test=%d", seed, len(train), len(test))
    return ([r["text"] for r in train], [r["text"] for r in test],
            harmless, test)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--data", type=str, default="data/code_contrastive.jsonl")
    ap.add_argument("--peak-layer", type=int, default=14)
    ap.add_argument("--split-seed", type=int, default=1)
    ap.add_argument("--n-test", type=int, default=30)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--scorer", choices=("substring", "llm", "both"), default="both")
    args = ap.parse_args()

    run_dir = new_run_dir("phase2_step3")
    log.info("run_dir: %s | model=%s peak=L%d", run_dir, args.model, args.peak_layer)

    bundle = get_model(args.model)
    log.info("model: %s | n_layers=%d d_model=%d device=%s dtype=%s",
             bundle.name, bundle.n_layers, bundle.d_model, bundle.device, bundle.model.cfg.dtype)

    # === Stratified split ===
    pairs_path = Path(args.data)
    if not pairs_path.is_absolute():
        pairs_path = Path(__file__).resolve().parent.parent / pairs_path
    harmful_train, harmful_test, harmless, test_meta = _load_split(
        pairs_path, seed=args.split_seed, n_test=args.n_test,
    )

    # === Extract d̂_train at L14 from cached activations (re-derived; cache hit expected) ===
    fmt = lambda msg: format_prompt_for_bundle(bundle, msg)
    log.info("caching residuals (train+harmless) at L%d ...", args.peak_layer)
    d_hat, H_train, L_full, _meta = extract_d_hat(
        bundle, harmful_train, harmless,
        layer=args.peak_layer, format_fn=fmt, extra_tag="phase2",
    )
    log.info("d_hat extracted at L%d from %d harmful_train + %d harmless",
             args.peak_layer, H_train.shape[0], L_full.shape[0])

    # Move d̂ to model device + dtype for hooks
    d_dev = d_hat.to(bundle.device).to(bundle.model.cfg.dtype)
    rand_dir = random_unit_vector(bundle.d_model, seed=0, device=bundle.device)
    rand_dir = rand_dir.to(bundle.model.cfg.dtype)

    # === Three conditions × n_test prompts ===
    log.info("generating baseline (no hook) on %d test prompts...", len(harmful_test))
    base_gen = generate_batch(bundle, harmful_test, max_new_tokens=args.max_new_tokens)

    log.info("generating with ablate(d̂_train) at all layers...")
    with ablate_dir(bundle.model, d_dev):
        abl_real = generate_batch(bundle, harmful_test, max_new_tokens=args.max_new_tokens)

    log.info("generating with ablate(random) at all layers...")
    with ablate_dir(bundle.model, rand_dir):
        abl_rand = generate_batch(bundle, harmful_test, max_new_tokens=args.max_new_tokens)

    # === Substring scorer (Qwen-tested in smoke; uses "is illegal"/"is unethical" patterns) ===
    cells = [
        ("baseline", base_gen),
        ("ablate_dhat", abl_real),
        ("ablate_random", abl_rand),
    ]
    substr_rates = {}
    for name, gens in cells:
        refused = [is_refusal(g) for g in gens]
        substr_rates[name] = sum(refused) / len(refused)
        log.info("  %-15s substring refusal: %d / %d = %.3f",
                 name, sum(refused), len(refused), substr_rates[name])

    # === Save partial before judge (HarmBench lesson — never lose generations to a judge crash) ===
    partial = {
        "step": "phase2_step3_causal",
        "model": bundle.name,
        "peak_layer": args.peak_layer,
        "split_seed": args.split_seed,
        "n_train": len(harmful_train),
        "n_test": len(harmful_test),
        "test_meta": [{"text": r["text"], "source": r["source"]} for r in test_meta],
        "completions": {n: g for n, g in cells},
        "substring_refusal_rates": substr_rates,
    }
    write_json(run_dir / "partial.json", partial)
    log.info("partial -> %s", run_dir / "partial.json")

    # === Dual-judge ===
    judge_summary = {}
    if args.scorer in ("llm", "both"):
        log.info("running calibrated Haiku 4.5 judge on 3 × %d = %d pairs ...",
                 len(harmful_test), 3 * len(harmful_test))
        from src.eval_llm import judge_many
        for name, gens in cells:
            pairs = list(zip(harmful_test, gens))
            verdicts = judge_many(pairs, show_progress=False)
            labels = [v.label for v in verdicts]
            n_ref = sum(1 for l in labels if l == "REFUSED")
            judge_summary[name] = {
                "n_refused": n_ref,
                "refusal_rate": n_ref / len(labels),
                "labels": labels,
            }
            log.info("  %-15s judge refusal: %d / %d = %.3f",
                     name, n_ref, len(labels), judge_summary[name]["refusal_rate"])

    # === Per-prompt table ===
    per_prompt = []
    for i, (prompt, meta) in enumerate(zip(harmful_test, test_meta)):
        row = {
            "idx": i,
            "prompt": prompt,
            "source": meta["source"],
            "baseline_completion": base_gen[i][:300],
            "ablate_dhat_completion": abl_real[i][:300],
            "ablate_random_completion": abl_rand[i][:300],
            "baseline_substr": is_refusal(base_gen[i]),
            "ablate_dhat_substr": is_refusal(abl_real[i]),
            "ablate_random_substr": is_refusal(abl_rand[i]),
        }
        if judge_summary:
            row["baseline_judge"] = judge_summary["baseline"]["labels"][i]
            row["ablate_dhat_judge"] = judge_summary["ablate_dhat"]["labels"][i]
            row["ablate_random_judge"] = judge_summary["ablate_random"]["labels"][i]
        per_prompt.append(row)

    # === Headline drop deltas ===
    delta_dhat = substr_rates["baseline"] - substr_rates["ablate_dhat"]
    delta_random = substr_rates["baseline"] - substr_rates["ablate_random"]
    log.info("substring drop:  d̂ ablate Δ = %+.3f | random ablate Δ = %+.3f",
             -delta_dhat, -delta_random)

    record = {
        **partial,
        "judge_summary": {n: {k: v for k, v in d.items() if k != "labels"}
                          for n, d in judge_summary.items()} if judge_summary else None,
        "judge_full": judge_summary if judge_summary else None,
        "per_prompt": per_prompt,
        "headline_substr_delta_dhat": delta_dhat,
        "headline_substr_delta_random": delta_random,
    }
    write_json(run_dir / "result.json", record)
    md_path = RESULTS / "phase2_step3_causal.md"
    md_path.write_text(_render_summary(record))
    log.info("result -> %s", run_dir / "result.json")
    log.info("summary -> %s", md_path)

    print(f"\n=== phase2_step3_causal: {bundle.name} @ L{args.peak_layer} ===")
    print(f"  Split: {len(harmful_train)} train / {len(harmful_test)} test (seed={args.split_seed})")
    print("  Substring scorer:")
    for n in ["baseline", "ablate_dhat", "ablate_random"]:
        print(f"    {n:<15s} refusal {substr_rates[n]:.3f}")
    if judge_summary:
        print("  LLM judge (Haiku 4.5):")
        for n in ["baseline", "ablate_dhat", "ablate_random"]:
            r = judge_summary[n]
            print(f"    {n:<15s} refusal {r['refusal_rate']:.3f} ({r['n_refused']}/{len(harmful_test)})")
    print(f"\n  Headline (substring): baseline → ablate_d̂ drop = {-delta_dhat:+.3f}")
    print(f"                        baseline → ablate_random drop = {-delta_random:+.3f}")
    print("  Specificity: d̂ drop should be MUCH larger (absolute value) than random drop.")
    return 0


def _render_summary(rec):
    md = [
        f"# Phase 2 Step 3 — causal test ({rec['model']}, L{rec['peak_layer']})",
        "",
        f"- Split: {rec['n_train']} train / {rec['n_test']} test (seed {rec['split_seed']})",
        "- Test source breakdown:",
    ]
    from collections import Counter
    src_counts = Counter(r["source"] for r in rec["test_meta"])
    for s, c in src_counts.most_common():
        md.append(f"  - {s}: {c}")
    md.append("")
    md.append("## Substring scorer (Qwen-compatible refusal patterns)")
    md.append("")
    md.append("| Condition | Refusal rate | Δ vs baseline |")
    md.append("|---|---:|---:|")
    base = rec["substring_refusal_rates"]["baseline"]
    for n in ("baseline", "ablate_dhat", "ablate_random"):
        rate = rec["substring_refusal_rates"][n]
        md.append(f"| {n} | {rate:.3f} | {(rate - base):+.3f} |")
    md.append("")

    if rec.get("judge_summary"):
        md.append("## Calibrated LLM judge (Haiku 4.5)")
        md.append("")
        md.append("| Condition | Refusal rate | n_refused / n |")
        md.append("|---|---:|---:|")
        for n in ("baseline", "ablate_dhat", "ablate_random"):
            r = rec["judge_summary"][n]
            md.append(f"| {n} | {r['refusal_rate']:.3f} | {r['n_refused']} / {rec['n_test']} |")
        md.append("")

    md.append("## Per-prompt verdict (judge if present, else substring)")
    md.append("")
    md.append("| # | Source | Prompt (first 70 chars) | base | abl d̂ | abl rand |")
    md.append("|---|---|---|:---:|:---:|:---:|")
    for r in rec["per_prompt"]:
        b = r.get("baseline_judge", "REF" if r["baseline_substr"] else "ans")
        ad = r.get("ablate_dhat_judge", "REF" if r["ablate_dhat_substr"] else "ans")
        ar = r.get("ablate_random_judge", "REF" if r["ablate_random_substr"] else "ans")
        short = r["prompt"][:70]
        src = r["source"].replace("harmbench_cybercrime", "HB-cyber").replace("advbench_code", "ADV-code")
        md.append(f"| {r['idx']} | {src} | {short}... | {b[:3]} | {ad[:3]} | {ar[:3]} |")
    md.append("")
    md.append("Per-prompt completions in `artifacts/runs/phase2_step3/<timestamp>/result.json`.")
    return "\n".join(md) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
