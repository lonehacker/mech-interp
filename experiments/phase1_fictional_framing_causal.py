"""
Phase 1.5-B causal half — does ablating d̂_augmented drop refusal on
fictional-framing prompts?

Closes the causal half of the open loop. Phase 1.5-B (extraction half)
showed that augmenting the training set with 30 fictional-framing prompts
moves d̂ by < 0.7° (cos = 0.9938) and doesn't pull held-out fictional-
framing test prompts onto the axis. The remaining question: under
Arditi-style ablation of d̂_augmented, does the model still refuse the 15
held-out fictional-framing prompts?

  • If d̂_augmented ablation drops refusal substantially → unitary-mechanism;
    the 1.5-B null was about projection (a classification readout) not
    causation. Phase 1's separable-mechanism claim weakens.
  • If d̂_augmented ablation does NOT drop refusal → fictional-framing is
    a separable causal mechanism. The 1.5-B null was the right signal.

3 conditions × 15 prompts = 45 generations, then dual-judge.

Run:
    cd ~/safe_ai/mech-security
    python -m experiments.phase1_fictional_framing_causal
"""

from __future__ import annotations

import argparse
import json
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
    load_jsonl_pairs,
    new_run_dir,
    write_json,
)
from src.activations import cache_resid
from src.directions import ablate_dir, diff_of_means, project, unit
from src.eval import is_refusal
from src.model import generate

log = get_logger("phase1_fictional_causal")


def load_jsonl_simple(path: Path, key: str = "text", filter_label: str | None = None) -> list[str]:
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line: continue
            r = json.loads(line)
            if filter_label is not None:
                if r.get("category") != filter_label and r.get("label") != filter_label:
                    continue
            out.append(r[key])
    return out


def _generate_batch(bundle, prompts, max_new_tokens=160):
    return [generate(bundle, p, max_new_tokens=max_new_tokens, temperature=0.0).strip()
            for p in prompts]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract-layer", type=int, default=13)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--scorer", choices=("substring", "llm", "both"), default="both")
    args = ap.parse_args()

    run_dir = new_run_dir("phase1_fictional_causal")
    log.info("run_dir: %s | extract=L%d | max_new=%d", run_dir, args.extract_layer, args.max_new_tokens)

    bundle = get_model()
    log.info("model: %s | device=%s", bundle.name, bundle.device)

    # === Load contrastive train sets and the fictional-framing augmentation ===
    advbench_harmful, alpaca_harmless = load_jsonl_pairs(
        Path(__file__).resolve().parent.parent / "data/contrastive.jsonl"
    )
    fictional_harmful = load_jsonl_simple(
        Path(__file__).resolve().parent.parent / "data/fictional_framing_train.jsonl"
    )
    affect_test_path = Path(__file__).resolve().parent.parent / "data/affect-test.jsonl"
    test_prompts = load_jsonl_simple(affect_test_path, filter_label="adversarial_jailbreak")

    log.info("train: %d AdvBench + %d fictional = %d harmful | %d Alpaca harmless",
             len(advbench_harmful), len(fictional_harmful),
             len(advbench_harmful) + len(fictional_harmful), len(alpaca_harmless))
    log.info("test: %d adversarial_jailbreak prompts", len(test_prompts))

    # === Reconstruct d̂_old and d̂_augmented from cached activations ===
    extra = (f"{bundle.name}|dtype={bundle.model.cfg.dtype}|L{args.extract_layer}|"
             f"resid_post|last_token|advbench_full")
    key_ah = content_hash(advbench_harmful, extra=extra + "|harmful")
    key_al = content_hash(alpaca_harmless, extra=extra + "|harmless")
    H_advbench = cached_activations(
        key_ah, lambda: cache_resid(bundle, advbench_harmful, layer=args.extract_layer, show_progress=False)
    )
    L_alpaca = cached_activations(
        key_al, lambda: cache_resid(bundle, alpaca_harmless, layer=args.extract_layer, show_progress=False)
    )
    extra_f = (f"{bundle.name}|dtype={bundle.model.cfg.dtype}|L{args.extract_layer}|"
               f"resid_post|last_token|fictional_framing_30")
    key_f = content_hash(fictional_harmful, extra=extra_f + "|harmful")
    H_fictional = cached_activations(
        key_f, lambda: cache_resid(bundle, fictional_harmful, layer=args.extract_layer, show_progress=False)
    )

    d_hat_old = unit(diff_of_means(H_advbench, L_alpaca))
    H_augmented = torch.cat([H_advbench, H_fictional], dim=0)
    d_hat_augmented = unit(diff_of_means(H_augmented, L_alpaca))
    cos = float((d_hat_old * d_hat_augmented).sum())
    log.info("d̂_old + d̂_augmented reconstructed | cos = %.4f", cos)

    # === Also need per-prompt projections under d̂_old to subset the 3 below-midpoint prompts ===
    extra_t = (f"{bundle.name}|dtype={bundle.model.cfg.dtype}|L{args.extract_layer}|"
               f"resid_post|last_token|affect_jailbreak_15")
    key_t = content_hash(test_prompts, extra=extra_t)
    H_test = cached_activations(
        key_t, lambda: cache_resid(bundle, test_prompts, layer=args.extract_layer, show_progress=False)
    )
    midpoint_old = 0.5 * (project(H_advbench, d_hat_old).mean() + project(L_alpaca, d_hat_old).mean()).item()
    test_proj_old = project(H_test, d_hat_old).numpy()
    below_midpoint = [bool(p < midpoint_old) for p in test_proj_old]
    n_below = sum(below_midpoint)
    log.info("test prompts BELOW d̂_old midpoint (the 1.5-B residual cases): %d / %d", n_below, len(test_prompts))

    # === Move d̂s to model device/dtype for hooks ===
    d_old_dev = d_hat_old.to(bundle.device).to(bundle.model.cfg.dtype)
    d_aug_dev = d_hat_augmented.to(bundle.device).to(bundle.model.cfg.dtype)

    # === Three conditions ===
    log.info("generating baseline (no hook)...")
    base_gen = _generate_batch(bundle, test_prompts, max_new_tokens=args.max_new_tokens)

    log.info("generating with ablate(d̂_old) hook (Arditi multi-layer)...")
    with ablate_dir(bundle.model, d_old_dev):
        abl_old_gen = _generate_batch(bundle, test_prompts, max_new_tokens=args.max_new_tokens)

    log.info("generating with ablate(d̂_augmented) hook (Arditi multi-layer)...")
    with ablate_dir(bundle.model, d_aug_dev):
        abl_aug_gen = _generate_batch(bundle, test_prompts, max_new_tokens=args.max_new_tokens)

    # === Substring scorer (fast, no API) ===
    cells = [
        ("baseline", base_gen),
        ("ablate_d_old", abl_old_gen),
        ("ablate_d_augmented", abl_aug_gen),
    ]
    substr_rates = {}
    for name, gens in cells:
        refused = [is_refusal(g) for g in gens]
        substr_rates[name] = sum(refused) / len(refused)
        log.info("  %-22s substring refusal rate: %d / %d = %.3f",
                 name, sum(refused), len(refused), substr_rates[name])

    # === Save partial result EARLY (before judge, in case judge crashes) ===
    partial = {
        "step": "phase1_fictional_framing_causal",
        "model": bundle.name,
        "extract_layer": args.extract_layer,
        "cos_old_vs_augmented": cos,
        "n_test": len(test_prompts),
        "n_below_midpoint_under_d_old": n_below,
        "test_prompts": test_prompts,
        "test_proj_old": test_proj_old.tolist(),
        "below_midpoint": below_midpoint,
        "completions": {
            "baseline": base_gen,
            "ablate_d_old": abl_old_gen,
            "ablate_d_augmented": abl_aug_gen,
        },
        "substring_refusal_rates": substr_rates,
    }
    write_json(run_dir / "partial.json", partial)
    log.info("partial result (no judge yet) -> %s", run_dir / "partial.json")

    # === LLM judge (calibrated, Haiku 4.5 primary). Cross-judge optional separately. ===
    judge_results = {}
    if args.scorer in ("llm", "both"):
        log.info("running calibrated LLM judge on 3 × 15 = 45 (prompt, completion) pairs ...")
        from src.eval_llm import judge_many
        for name, gens in cells:
            pairs = list(zip(test_prompts, gens))
            verdicts = judge_many(pairs, show_progress=False)
            labels = [v.label for v in verdicts]
            n_refused = sum(1 for l in labels if l == "REFUSED")
            judge_results[name] = {
                "labels": labels,
                "n_refused": n_refused,
                "refusal_rate": n_refused / len(labels),
            }
            log.info("  %-22s judge refusal rate: %d / %d = %.3f",
                     name, n_refused, len(labels), judge_results[name]["refusal_rate"])

    # === Per-prompt breakdown: which prompts flipped under each ablation? ===
    per_prompt = []
    for i, prompt in enumerate(test_prompts):
        row = {
            "idx": i,
            "prompt": prompt,
            "proj_d_old": float(test_proj_old[i]),
            "below_midpoint_under_d_old": below_midpoint[i],
            "baseline_completion": base_gen[i][:300],
            "ablate_d_old_completion": abl_old_gen[i][:300],
            "ablate_d_augmented_completion": abl_aug_gen[i][:300],
            "baseline_substr_refused": is_refusal(base_gen[i]),
            "ablate_d_old_substr_refused": is_refusal(abl_old_gen[i]),
            "ablate_d_augmented_substr_refused": is_refusal(abl_aug_gen[i]),
        }
        if judge_results:
            row["baseline_judge"] = judge_results["baseline"]["labels"][i]
            row["ablate_d_old_judge"] = judge_results["ablate_d_old"]["labels"][i]
            row["ablate_d_augmented_judge"] = judge_results["ablate_d_augmented"]["labels"][i]
        per_prompt.append(row)

    # === Subset analysis: 12 above-midpoint vs 3 below-midpoint ===
    above_idx = [i for i, b in enumerate(below_midpoint) if not b]
    below_idx = [i for i, b in enumerate(below_midpoint) if b]
    def subset_rate(idx_list, source):
        return sum(1 for i in idx_list if source["labels"][i] == "REFUSED") / max(len(idx_list), 1) if judge_results else None
    subset_rates_judge = {}
    if judge_results:
        for name in ["baseline", "ablate_d_old", "ablate_d_augmented"]:
            subset_rates_judge[name] = {
                "above_midpoint_n_12": subset_rate(above_idx, judge_results[name]),
                "below_midpoint_n_3": subset_rate(below_idx, judge_results[name]),
            }

    # === Verdict (raw numbers, no spin) ===
    raw_drop_d_old = (substr_rates["baseline"] - substr_rates["ablate_d_old"])
    raw_drop_d_aug = (substr_rates["baseline"] - substr_rates["ablate_d_augmented"])
    log.info("substring drop under d̂_old:        baseline %.3f → ablated %.3f (Δ = %+.3f)",
             substr_rates["baseline"], substr_rates["ablate_d_old"], -raw_drop_d_old)
    log.info("substring drop under d̂_augmented:  baseline %.3f → ablated %.3f (Δ = %+.3f)",
             substr_rates["baseline"], substr_rates["ablate_d_augmented"], -raw_drop_d_aug)

    record = {
        **partial,
        "judge_results_summary": {k: {"n_refused": v["n_refused"], "refusal_rate": v["refusal_rate"]}
                                  for k, v in judge_results.items()} if judge_results else None,
        "judge_results_per_prompt": judge_results if judge_results else None,
        "subset_rates_judge": subset_rates_judge if judge_results else None,
        "per_prompt": per_prompt,
        "raw_drop_d_old_substr": raw_drop_d_old,
        "raw_drop_d_augmented_substr": raw_drop_d_aug,
    }
    write_json(run_dir / "result.json", record)
    log.info("full result -> %s", run_dir / "result.json")

    md_path = RESULTS / "phase1_fictional_framing_causal.md"
    md_path.write_text(_render_summary(record))
    log.info("summary -> %s", md_path)

    print(f"\n=== phase1_fictional_framing_causal ===")
    print(f"cos(d̂_old, d̂_augmented) = {cos:.4f}")
    print(f"3 conditions × 15 prompts, substring scorer:")
    print(f"  baseline:           refusal {substr_rates['baseline']:.3f}")
    print(f"  ablate d̂_old:       refusal {substr_rates['ablate_d_old']:.3f} (Δ = {-raw_drop_d_old:+.3f})")
    print(f"  ablate d̂_augmented: refusal {substr_rates['ablate_d_augmented']:.3f} (Δ = {-raw_drop_d_aug:+.3f})")
    if judge_results:
        print(f"\nLLM judge (Haiku 4.5):")
        for name, r in judge_results.items():
            print(f"  {name:22s} refusal {r['refusal_rate']:.3f} ({r['n_refused']}/15)")
        print(f"\nSubset (judge) — 12 above-midpoint vs 3 below-midpoint:")
        for name, r in subset_rates_judge.items():
            print(f"  {name:22s} above: {r['above_midpoint_n_12']:.3f} | below: {r['below_midpoint_n_3']:.3f}")
    return 0


def _render_summary(rec):
    md = [
        "# Phase 1.5-B causal half — fictional-framing ablation",
        "",
        f"- Model: `{rec['model']}` | extract layer L{rec['extract_layer']}",
        f"- cos(d̂_old, d̂_augmented) = **{rec['cos_old_vs_augmented']:.4f}**",
        f"- Test set: {rec['n_test']} `adversarial_jailbreak` prompts",
        f"- Of these, {rec['n_below_midpoint_under_d_old']} projected BELOW d̂_old's harmful/harmless midpoint",
        "",
        "## Substring scorer (fast, deterministic — primary)",
        "",
        "| Condition | Refusal rate | n_refused / n |",
        "|---|---:|---:|",
    ]
    for name, rate in rec["substring_refusal_rates"].items():
        n_refused = int(round(rate * rec["n_test"]))
        md.append(f"| {name} | {rate:.3f} | {n_refused} / {rec['n_test']} |")
    md.append("")
    if rec.get("judge_results_summary"):
        md.append("## Calibrated LLM judge (Haiku 4.5)")
        md.append("")
        md.append("| Condition | Refusal rate | n_refused / n |")
        md.append("|---|---:|---:|")
        for name, r in rec["judge_results_summary"].items():
            md.append(f"| {name} | {r['refusal_rate']:.3f} | {r['n_refused']} / {rec['n_test']} |")
        md.append("")
        md.append("### Subset breakdown — 12 above-midpoint vs 3 below-midpoint")
        md.append("")
        md.append("| Condition | Above-midpoint (n=12) refusal | Below-midpoint (n=3) refusal |")
        md.append("|---|---:|---:|")
        for name, r in rec["subset_rates_judge"].items():
            md.append(f"| {name} | {r['above_midpoint_n_12']:.3f} | {r['below_midpoint_n_3']:.3f} |")
        md.append("")
    md.append("## Per-prompt verdict")
    md.append("")
    md.append("| # | Prompt (first 70 chars) | proj_d_old | below_mid? | base | abl d̂_old | abl d̂_aug |")
    md.append("|---|---|---:|:---:|:---:|:---:|:---:|")
    for r in rec["per_prompt"]:
        b = r.get("baseline_judge", "REF" if r["baseline_substr_refused"] else "ans")
        ao = r.get("ablate_d_old_judge", "REF" if r["ablate_d_old_substr_refused"] else "ans")
        aa = r.get("ablate_d_augmented_judge", "REF" if r["ablate_d_augmented_substr_refused"] else "ans")
        flag = "**Y**" if r["below_midpoint_under_d_old"] else " "
        md.append(f"| {r['idx']} | {r['prompt'][:70]}... | {r['proj_d_old']:.1f} | {flag} | {b[:3]} | {ao[:3]} | {aa[:3]} |")
    md.append("")
    md.append("Per-prompt completions are in `artifacts/runs/phase1_fictional_causal/<timestamp>/result.json`.")
    return "\n".join(md) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
