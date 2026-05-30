"""
Phase 2 Step 3e — matched-set causal sweep on Qwen2.5-3B.

Pre-registered at results/phase2_step3e_preregistration.md (committed
before this runner). Decides between (A-i) data-driven null and (A-ii)
model-driven null on the Qwen result, by re-running both the ablation
test and the operating-band addition sweep on the de-confounded
contrastive set `data/code_contrastive_matched.jsonl` (TF-IDF AUC 0.61
vs `code_contrastive` 0.99).

Reads the asymmetric pre-registered:
  - POSITIVE → (A-i) provisional, scale up to 100 pairs for hardened
    headline.
  - NULL → INCONCLUSIVE at n=40; scale up before any (A-ii) claim.

Run:
    cd ~/safe_ai/mech-security
    python -m experiments.phase2_step3e_matched_set_sweep
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


import matplotlib

matplotlib.use("Agg")
import numpy as np

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
from mech_security.directions import ablate_dir, add_dir, random_unit_vector
from mech_security.eval import is_refusal
from mech_security.model import format_prompt_for_bundle

log = get_logger("phase2_step3e")


def _split(jsonl_path, seed=1, n_test=10):
    """Simple 30/10 split on each side. The matched set has uniform sources
    (all harmful = harmbench_cybercrime, all harmless = defensive_matched_v2),
    so no stratification needed.

    The harmful + harmless shuffles share a single rng instance; this
    state-sharing is part of the cache key, so the inlined sequence is kept
    here rather than delegated to the generic _runner helper (which makes a
    fresh rng per call)."""
    recs = [json.loads(l) for l in jsonl_path.open()]
    harmful = [r["text"] for r in recs if r["label"] == "harmful"]
    harmless = [r["text"] for r in recs if r["label"] == "harmless"]
    rng = random.Random(seed)
    rng.shuffle(harmful); rng.shuffle(harmless)
    return (harmful[n_test:], harmful[:n_test],
            harmless[n_test:], harmless[:n_test])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--matched-data", type=str, default="data/code_contrastive_matched.jsonl")
    ap.add_argument("--code-data", type=str, default="data/code_contrastive.jsonl",
                    help="Original code_contrastive — for ablate-d̂_old comparator cell")
    ap.add_argument("--extract-layer", type=int, default=14)
    ap.add_argument("--inject-layers", type=int, nargs="+",
                    default=[3, 7, 11, 15, 19, 23, 27, 31])
    ap.add_argument("--coeff-mults", type=float, nargs="+",
                    default=[1.0, 2.0, 4.0])
    ap.add_argument("--split-seed", type=int, default=1)
    ap.add_argument("--n-test", type=int, default=10)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--judge-lift-threshold", type=float, default=0.10)
    args = ap.parse_args()

    run_dir = new_run_dir("phase2_step3e")
    log.info("run_dir: %s | extract=L%d", run_dir, args.extract_layer)

    bundle = get_model(args.model)
    log.info("model: %s | n_layers=%d d_model=%d dtype=%s",
             bundle.name, bundle.n_layers, bundle.d_model, bundle.model.cfg.dtype)

    matched_path = Path(args.matched_data)
    code_path = Path(args.code_data)
    if not matched_path.is_absolute():
        matched_path = Path(__file__).resolve().parent.parent / matched_path
    if not code_path.is_absolute():
        code_path = Path(__file__).resolve().parent.parent / code_path

    h_train_m, h_test_m, l_train_m, l_test_m = _split(
        matched_path, seed=args.split_seed, n_test=args.n_test)
    log.info("matched split (seed=%d): %d harmful train / %d test, %d harmless train / %d test",
             args.split_seed, len(h_train_m), len(h_test_m), len(l_train_m), len(l_test_m))

    fmt = lambda msg: format_prompt_for_bundle(bundle, msg)

    # === Extract d̂_matched at L14 ===
    log.info("caching d̂_matched activations at L%d...", args.extract_layer)
    d_hat_matched, _H_m, _L_m, meta_m = extract_d_hat(
        bundle, h_train_m, l_train_m,
        layer=args.extract_layer, format_fn=fmt,
        extra_tag="matched_v2", harmless_key_suffix="harmless_train",
    )
    nat_scale_matched = meta_m["natural_scale"]
    log.info("d̂_matched: natural scale at L%d = %.3f", args.extract_layer, nat_scale_matched)

    # === Reconstruct d̂_old from code_contrastive (already-cached) for direct comparison ===
    recs_code = [json.loads(l) for l in code_path.open()]
    harm_code = [r for r in recs_code if r["label"] == "harmful"]
    harmless_code = [r["text"] for r in recs_code if r["label"] == "harmless"]
    train_code_recs, _ = stratified_split(
        harm_code, key_fn=lambda r: r["source"], seed=args.split_seed, n_test=30,
    )
    h_train_code = [r["text"] for r in train_code_recs]
    d_hat_old, _H_c, _L_c, meta_o = extract_d_hat(
        bundle, h_train_code, harmless_code,
        layer=args.extract_layer, format_fn=fmt, extra_tag="phase2",
    )
    log.info("d̂_old (code_contrastive): natural scale at L%d = %.3f",
             args.extract_layer, meta_o["natural_scale"])

    cos_old_new = float((d_hat_matched * d_hat_old).sum())
    log.info("cos(d̂_matched, d̂_old) = %.4f", cos_old_new)

    d_m_dev = d_hat_matched.to(bundle.device).to(bundle.model.cfg.dtype)
    d_o_dev = d_hat_old.to(bundle.device).to(bundle.model.cfg.dtype)
    rand_dir = random_unit_vector(bundle.d_model, seed=0, device=bundle.device).to(bundle.model.cfg.dtype)

    # =====================================================
    # PART 1 — Ablation test (4 cells × 10 test prompts)
    # =====================================================
    log.info("=== PART 1: ablation cells on 10 held-out matched-harmful ===")
    log.info("generating baseline (no hook)...")
    abl_base = generate_batch(bundle, h_test_m, max_new_tokens=args.max_new_tokens)
    log.info("generating with ablate(d̂_matched)...")
    with ablate_dir(bundle.model, d_m_dev):
        abl_dnew = generate_batch(bundle, h_test_m, max_new_tokens=args.max_new_tokens)
    log.info("generating with ablate(d̂_old)...")
    with ablate_dir(bundle.model, d_o_dev):
        abl_dold = generate_batch(bundle, h_test_m, max_new_tokens=args.max_new_tokens)
    log.info("generating with ablate(random)...")
    with ablate_dir(bundle.model, rand_dir):
        abl_rand = generate_batch(bundle, h_test_m, max_new_tokens=args.max_new_tokens)

    abl_cells = {
        "baseline": abl_base,
        "ablate_dhat_matched": abl_dnew,
        "ablate_dhat_old": abl_dold,
        "ablate_random": abl_rand,
    }
    abl_rates = {n: sum(is_refusal(g) for g in gens) / len(gens) for n, gens in abl_cells.items()}
    for n, r in abl_rates.items():
        log.info("  %-22s refusal=%d/%d=%.2f", n, int(r * len(h_test_m)), len(h_test_m), r)

    # =====================================================
    # PART 2 — Operating-band addition sweep on harmless
    # =====================================================
    log.info("=== PART 2: addition sweep on %d harmless test prompts ===", len(l_test_m))
    log.info("generating addition-baseline (no hook)...")
    add_base = generate_batch(bundle, l_test_m, max_new_tokens=args.max_new_tokens)
    add_base_rate = sum(is_refusal(g) for g in add_base) / len(add_base)
    log.info("  addition baseline substring refusal: %.3f", add_base_rate)

    add_cells = {"baseline": add_base}
    add_meta = {}
    total_cells = len(args.inject_layers) * len(args.coeff_mults)
    cell_idx = 0
    for L_inj in args.inject_layers:
        for cmult in args.coeff_mults:
            cell_idx += 1
            coeff = cmult * nat_scale_matched
            name = f"add_L{L_inj}_c{cmult}x"
            log.info("  [%d/%d] %s: inject L%d coeff=%.3f", cell_idx, total_cells, name, L_inj, coeff)
            with add_dir(bundle.model, d_m_dev, coeff=coeff, layer=L_inj):
                gen = generate_batch(bundle, l_test_m, max_new_tokens=args.max_new_tokens)
            n_ref = sum(is_refusal(g) for g in gen)
            mean_len = float(np.mean([len(g) for g in gen]))
            add_cells[name] = gen
            add_meta[name] = {
                "inject_layer": L_inj, "coeff_mult": cmult, "coeff": coeff,
                "n_refused_substr": n_ref,
                "refusal_rate_substr": n_ref / len(gen),
                "lift_vs_baseline": n_ref / len(gen) - add_base_rate,
                "mean_chars": mean_len,
            }
            log.info("    substr refusal=%d/%d=%.2f (lift %+.2f), mean_chars=%.0f",
                     n_ref, len(gen), n_ref/len(gen), n_ref/len(gen) - add_base_rate, mean_len)

    # Save partial
    partial = {
        "step": "phase2_step3e_matched_set_sweep",
        "model": bundle.name,
        "matched_data": str(matched_path),
        "matched_data_hash": "ed5cfdeff469",
        "extract_layer": args.extract_layer,
        "natural_scale_matched": nat_scale_matched,
        "cos_dhat_matched_old": cos_old_new,
        "split_seed": args.split_seed,
        "n_train": len(h_train_m),
        "n_test": len(h_test_m),
        "h_test": h_test_m,
        "l_test": l_test_m,
        "ablation_cells": abl_cells,
        "ablation_substr_rates": abl_rates,
        "addition_baseline_rate": add_base_rate,
        "addition_cells": add_cells,
        "addition_meta": add_meta,
    }
    write_json(run_dir / "partial.json", partial)
    log.info("partial -> %s", run_dir / "partial.json")

    # =====================================================
    # PART 3 — Dual-judge lifted cells
    # =====================================================
    judge_summary = {}
    cells_to_judge = (
        ["baseline_ablation"] + list(abl_cells.keys()) +
        [name for name, m in add_meta.items() if m["lift_vs_baseline"] >= args.judge_lift_threshold]
    )
    # Always judge ablation cells; only judge addition cells with lift
    log.info("=== PART 3: judge ablation cells (always) + lifted addition cells ===")
    from mech_security.eval_llm import judge_many
    for name in abl_cells:
        pairs = list(zip(h_test_m, abl_cells[name]))
        verdicts = judge_many(pairs, show_progress=False)
        labels = [v.label for v in verdicts]
        n_ref = sum(1 for l in labels if l == "REFUSED")
        judge_summary[f"ablation_{name}"] = {
            "labels": labels,
            "n_refused": n_ref,
            "refusal_rate": n_ref / len(labels),
        }
        log.info("  ablation %-22s judge refusal=%d/%d=%.2f", name, n_ref, len(labels), n_ref/len(labels))

    lifted = [name for name, m in add_meta.items() if m["lift_vs_baseline"] >= args.judge_lift_threshold]
    if lifted:
        log.info("addition cells with lift >= %.2f: %d", args.judge_lift_threshold, len(lifted))
        # Judge baseline once
        pairs = list(zip(l_test_m, add_base))
        verdicts = judge_many(pairs, show_progress=False)
        base_lab = [v.label for v in verdicts]
        base_jrate = sum(1 for l in base_lab if l == "REFUSED") / len(base_lab)
        judge_summary["addition_baseline"] = {"labels": base_lab, "n_refused": sum(1 for l in base_lab if l == "REFUSED"), "refusal_rate": base_jrate}
        for name in lifted:
            pairs = list(zip(l_test_m, add_cells[name]))
            verdicts = judge_many(pairs, show_progress=False)
            labels = [v.label for v in verdicts]
            n_ref = sum(1 for l in labels if l == "REFUSED")
            judge_summary[f"addition_{name}"] = {
                "labels": labels, "n_refused": n_ref,
                "refusal_rate": n_ref / len(labels),
                "judge_lift": n_ref / len(labels) - base_jrate,
            }
            log.info("  addition %-22s judge refusal=%d/%d=%.2f (lift %+.2f)",
                     name, n_ref, len(labels), n_ref/len(labels), n_ref/len(labels) - base_jrate)
    else:
        log.info("no addition cells worth judging (no substring lift >= %.2f).", args.judge_lift_threshold)

    # =====================================================
    # Specificity control on best add cell (if lifted)
    # =====================================================
    spec_control = None
    if add_meta:
        best = max(add_meta, key=lambda k: add_meta[k]["lift_vs_baseline"])
        if add_meta[best]["lift_vs_baseline"] >= args.judge_lift_threshold:
            log.info("specificity control: random unit vector at %s", best)
            with add_dir(bundle.model, rand_dir, coeff=add_meta[best]["coeff"],
                          layer=add_meta[best]["inject_layer"]):
                rand_gen = generate_batch(bundle, l_test_m, max_new_tokens=args.max_new_tokens)
            n_ref = sum(is_refusal(g) for g in rand_gen)
            spec_control = {
                "cell": best, "completions": rand_gen,
                "n_refused_substr": n_ref,
                "refusal_rate_substr": n_ref / len(rand_gen),
                "lift_vs_baseline_substr": n_ref / len(rand_gen) - add_base_rate,
            }
            log.info("  random direction at %s: substr refusal=%d/%d=%.2f (lift %+.2f)",
                     best, n_ref, len(rand_gen), n_ref/len(rand_gen), n_ref/len(rand_gen) - add_base_rate)

    # Save final
    record = {
        **partial,
        "judge_summary": {k: {kk: vv for kk, vv in v.items() if kk != "labels"}
                          for k, v in judge_summary.items()} if judge_summary else None,
        "judge_full": judge_summary if judge_summary else None,
        "specificity_control": spec_control,
        "best_addition_cell": max(add_meta, key=lambda k: add_meta[k]["lift_vs_baseline"]) if add_meta else None,
    }
    write_json(run_dir / "result.json", record)
    md_path = RESULTS / "phase2_step3e_matched_set_sweep.md"
    md_path.write_text(_render_summary(record))
    log.info("result -> %s", run_dir / "result.json")
    log.info("summary -> %s", md_path)

    # Headline console output
    print("\n=== Phase 2 Step 3e — matched-set causal sweep ===")
    print(f"d̂_matched at L{args.extract_layer} | natural scale = {nat_scale_matched:.3f}")
    print(f"cos(d̂_matched, d̂_old from code_contrastive) = {cos_old_new:.4f}")
    print(f"\nAblation (n={len(h_test_m)} held-out matched-harmful):")
    for n, r in abl_rates.items():
        j = judge_summary.get(f"ablation_{n}", {}).get("refusal_rate") if judge_summary else None
        print(f"  {n:<22s}  substr {r:.2f}  judge {j if j is None else f'{j:.2f}'}")
    print(f"\nAddition sweep ({len(l_test_m)} harmless prompts × {total_cells} cells):")
    print(f"  baseline (no hook): {add_base_rate:.2f}")
    if lifted:
        for name in lifted:
            m = add_meta[name]
            print(f"  {name:<22s}  substr {m['refusal_rate_substr']:.2f}  lift {m['lift_vs_baseline']:+.2f}  mean_chars={m['mean_chars']:.0f}")
    else:
        print(f"  NO cells with substring lift >= {args.judge_lift_threshold} (i.e., no induced refusal anywhere).")
    if spec_control:
        print(f"\nSpecificity control at {spec_control['cell']}: random-vector lift = {spec_control['lift_vs_baseline_substr']:+.2f}")
    return 0


def _render_summary(rec):
    md = [
        f"# Phase 2 Step 3e — matched-set causal sweep ({rec['model']})",
        "",
        f"- Matched contrastive set: `{rec['matched_data']}` (hash `{rec['matched_data_hash']}`)",
        f"- Extract layer L{rec['extract_layer']}; natural scale d̂_matched = {rec['natural_scale_matched']:.3f}",
        f"- cos(d̂_matched, d̂_old from code_contrastive) = {rec['cos_dhat_matched_old']:.4f}",
        f"- Split seed {rec['split_seed']}: {rec['n_train']} train / {rec['n_test']} test per side",
        "",
        "## Part 1 — Ablation on 10 held-out matched-harmful prompts",
        "",
        "| Condition | Substring refusal | Judge refusal | Δ vs baseline (judge) |",
        "|---|---:|---:|---:|",
    ]
    base_j = rec["judge_summary"].get("ablation_baseline", {}).get("refusal_rate", None) if rec.get("judge_summary") else None
    for n in ("baseline", "ablate_dhat_matched", "ablate_dhat_old", "ablate_random"):
        s = rec["ablation_substr_rates"][n]
        j = rec["judge_summary"].get(f"ablation_{n}", {}).get("refusal_rate", None) if rec.get("judge_summary") else None
        dj = (j - base_j) if (j is not None and base_j is not None and n != "baseline") else None
        md.append(f"| {n} | {s:.3f} | {j if j is None else f'{j:.3f}'} | {dj if dj is None else f'{dj:+.3f}'} |")
    md.append("")
    md.append("## Part 2 — Operating-band addition sweep on 10 harmless")
    md.append("")
    md.append(f"Baseline (no hook) substring refusal: {rec['addition_baseline_rate']:.3f}")
    md.append("")
    md.append("| coeff × | " + " | ".join(f"L{L}" for L in sorted(set(m['inject_layer'] for m in rec['addition_meta'].values()))) + " |")
    md.append("|---|" + "---:|" * len(set(m['inject_layer'] for m in rec['addition_meta'].values())))
    cmults = sorted(set(m["coeff_mult"] for m in rec["addition_meta"].values()))
    layers = sorted(set(m["inject_layer"] for m in rec["addition_meta"].values()))
    for cmult in cmults:
        row = []
        for L in layers:
            name = f"add_L{L}_c{cmult}x"
            if name in rec["addition_meta"]:
                row.append(f"{rec['addition_meta'][name]['refusal_rate_substr']:.2f}")
            else:
                row.append("—")
        md.append(f"| {cmult}× | " + " | ".join(row) + " |")
    md.append("")
    if rec.get("specificity_control"):
        sc = rec["specificity_control"]
        md.append(f"## Specificity control (random at {sc['cell']}):")
        md.append(f"- Substring refusal: {sc['refusal_rate_substr']:.3f}, lift {sc['lift_vs_baseline_substr']:+.3f}")
        md.append("")
    md.append("Per-prompt completions in `artifacts/runs/phase2_step3e/<timestamp>/result.json`.")
    md.append("Pre-registration: `results/phase2_step3e_preregistration.md`.")
    return "\n".join(md) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
