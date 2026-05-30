"""
Phase 1 — SUBSPACE ABLATION (revised after bootstrap stability check).

Earlier analysis suggested L13 has a multi-D refusal subspace because LDA-top-1
and diff-of-means are nearly orthogonal yet both perfect classifiers. The
bootstrap stability check then showed that LDA-top-1's specific direction
varies wildly across resamples (cosines 0.08, 0.03, 0.45, 0.24 vs rep 0). So
the "perfect classifier subspace" exists in any sample but the directions
within it beyond diff-of-means are sample-dependent.

This experiment is the SHARPENED causal test (per planner): ablate LDA-top-1
from THREE DIFFERENT bootstrap resamples as separate cells. The three
outcomes distinguish:

  All three break refusal → multiple causal directions; subspace is causal.
  Only some break refusal → those were coincidence with the test prompts;
                            no single LDA-top-1 direction is causal.
  None break refusal       → LDA-top-1 is statistically discriminative but
                            never on the causal path. Diff-of-means is the
                            unique stably-recoverable causal direction.

PRE-REGISTERED PREDICTION: outcome (2) or (3). Diff-of-means is bootstrap-
stable and causally sufficient. Bootstrap-LDA directions are statistically
discriminative (high AUC by construction) but not consistently causal —
ablating one or the other gives inconsistent behavioral effects because the
specific direction was overfitting accidental in-sample correlations in a
2304-D residual stream.

If the prediction lands, the honest Phase 1 conclusion is:
  - Diff-of-means is the unique stably-recoverable refusal direction on Gemma-2-2b-it.
  - The Arditi single-direction picture HOLDS for behavior.
  - The "multi-D subspace" claim survives only at the CLASSIFICATION level
    (any sample admits multiple high-AUC orthogonal directions in 2304-D space —
    a statistical fact, not a mechanism).
  - This is consistent with the post-Arditi literature (Wollschläger et al.,
    Winninger 2025) which already describes polyhedral cones / multi-directional
    causal structure; our experiment doesn't add to that picture, it just
    measures it on a model where it was implicit.

Ablation cells (all Arditi multi-layer recipe):

  A. baseline (no hook)                          — refusal ≈ 1.00
  B. diff-of-means d_hat at L13                  — Step 3 confirmation
  C1. LDA-top-1 at L13 from BOOTSTRAP A          — causal? consistency test
  C2. LDA-top-1 at L13 from BOOTSTRAP B          — causal? consistency test
  C3. LDA-top-1 at L13 from BOOTSTRAP C          — causal? consistency test
  D. top-5 LDA orthogonal subspace from BOOTSTRAP A  — does multi-D ablation work?
  E. diff-of-means d_hat at L3                   — different layer, same population direction
  F. random unit vector                          — specificity control

Reuses cached L13 activations (no MPS for direction extraction).

Run:
    cd ~/safe_ai/mech-security
    python -m experiments.phase1_subspace_ablation
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


import numpy as np
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
from mech_security.activations import cache_resid
from mech_security.directions import (
    ablate_dir,
    ablate_subspace,
    diff_of_means,
    lda_directions,
    random_unit_vector,
    unit,
)
from mech_security.eval import coherence_ok, refusal_rate
from mech_security.model import generate

log = get_logger("phase1_subspace_ablation")


def _gen(bundle, prompts, max_new_tokens=96):
    return [generate(bundle, p, max_new_tokens=max_new_tokens, temperature=0.0)
            for p in prompts]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract-layer", type=int, default=13)
    ap.add_argument("--data", type=str, default="data/contrastive.jsonl")
    ap.add_argument("--n-test", type=int, default=12,
                    help="harmful prompts per cell (Step 3 used 12 → keep apples-to-apples)")
    ap.add_argument("--seed", type=int, default=0,
                    help="Same as Step 3's split seed for direct comparability.")
    ap.add_argument("--lda-k", type=int, default=5,
                    help="Number of LDA directions in the subspace ablation cell.")
    args = ap.parse_args()

    run_dir = new_run_dir("phase1_subspace_ablation")
    log.info("run_dir: %s | extract=L%d n_test=%d k=%d",
             run_dir, args.extract_layer, args.n_test, args.lda_k)

    bundle = get_model()
    log.info("model: %s | device=%s", bundle.name, bundle.device)

    pairs_path = Path(args.data)
    if not pairs_path.is_absolute():
        pairs_path = Path(__file__).resolve().parent.parent / pairs_path
    harmful_all, harmless_all = load_jsonl_pairs(pairs_path)

    # Match Step 3's split exactly so we can compare cells directly
    rng = np.random.default_rng(args.seed)
    test_h_idx = rng.choice(len(harmful_all), size=args.n_test, replace=False).tolist()
    train_h = [t for i, t in enumerate(harmful_all) if i not in set(test_h_idx)]
    train_l = harmless_all
    test_h = [harmful_all[i] for i in test_h_idx]
    log.info("train: %d harmful / %d harmless | test: %d harmful",
             len(train_h), len(train_l), len(test_h))

    # Extract activations at L13 and L3 (caches hit if Step 3b already ran)
    def get_acts(prompts, layer, label_tag):
        extra = (
            f"{bundle.name}|dtype={bundle.model.cfg.dtype}|L{layer}|"
            f"resid_post|last_token|seed{args.seed}|sweep"
        )
        key = content_hash(prompts, extra=extra + f"|{label_tag}")
        return cached_activations(key, lambda: cache_resid(bundle, prompts, layer=layer, show_progress=False))

    log.info("loading/caching residuals at L%d (extract for d_hat) ...", args.extract_layer)
    train_h_L13 = get_acts(train_h, args.extract_layer, "harmful")
    train_l_L13 = get_acts(train_l, args.extract_layer, "harmless")
    log.info("loading/caching residuals at L3 (for L3 d_hat) ...")
    train_h_L3 = get_acts(train_h, 3, "harmful")
    train_l_L3 = get_acts(train_l, 3, "harmless")

    # === Extract the directions ===
    d_hat_L13 = unit(diff_of_means(train_h_L13, train_l_L13))
    d_hat_L3 = unit(diff_of_means(train_h_L3, train_l_L3))
    rand = random_unit_vector(bundle.d_model, seed=7)

    # Three bootstrap LDA-top-1 directions — the sharpened causal consistency test
    bootstrap_seeds = [101, 202, 303]
    lda_top1_bootstraps = []
    for bs in bootstrap_seeds:
        d = lda_directions(train_h_L13, train_l_L13, k=1, bootstrap_seed=bs)[0]
        lda_top1_bootstraps.append(d)
    # Cosines: between bootstrap LDA dirs, and vs diff-of-means
    cos_lda_pairs = [
        ("bs101 vs bs202", float((lda_top1_bootstraps[0] * lda_top1_bootstraps[1]).sum())),
        ("bs101 vs bs303", float((lda_top1_bootstraps[0] * lda_top1_bootstraps[2]).sum())),
        ("bs202 vs bs303", float((lda_top1_bootstraps[1] * lda_top1_bootstraps[2]).sum())),
    ]
    cos_lda_vs_dhat = [float((d * d_hat_L13).sum()) for d in lda_top1_bootstraps]
    log.info("bootstrap LDA-top-1 pairwise cosines: %s", cos_lda_pairs)
    log.info("bootstrap LDA-top-1 vs diff-of-means cosines: %s",
             [(f"bs{bs}", round(c, 3)) for bs, c in zip(bootstrap_seeds, cos_lda_vs_dhat)])

    # Top-5 orthogonal subspace from a single bootstrap (the "subspace ablation" cell)
    log.info("extracting top-%d LDA orthogonal subspace from bootstrap seed=101 ...", args.lda_k)
    lda_subspace = lda_directions(train_h_L13, train_l_L13,
                                                  k=args.lda_k, bootstrap_seed=101)

    cos_d_l3 = float((d_hat_L13 * d_hat_L3).sum())
    log.info("cos(d_hat_L13, d_hat_L3) = %.3f", cos_d_l3)

    # === Baselines ===
    log.info("baseline (no hook) on n=%d harmful ...", len(test_h))
    base_gens = _gen(bundle, test_h)
    base_r = refusal_rate(base_gens)
    base_c = coherence_ok(base_gens)
    log.info("baseline harmful refusal=%.2f (%d/%d), p50_chars=%.0f",
             base_r.rate, base_r.n_refused, base_r.n, base_c.p50_chars)

    cells = []

    def _record(name, predicted, gens):
        r = refusal_rate(gens)
        c = coherence_ok(gens)
        cells.append({
            "name": name,
            "predicted_refusal": predicted,
            "refusal_rate": r.rate,
            "n_refused": r.n_refused,
            "n": r.n,
            "p50_chars": c.p50_chars,
            "sample_gens": [g.strip()[:200] for g in gens[:3]],
        })
        log.info("%s: refusal=%.2f (%d/%d), p50=%.0f",
                 name, r.rate, r.n_refused, r.n, c.p50_chars)

    # B: diff-of-means L13 (Step 3 confirmation; predicted causally sufficient)
    with ablate_dir(bundle.model, d_hat_L13):
        _record("B_ablate_d_hat_L13",
                "≈ 0.00 (bootstrap-stable direction; predicted causally sufficient)",
                _gen(bundle, test_h))

    # C1, C2, C3: bootstrap LDA-top-1 directions (the consistency test)
    # Pre-registered prediction: inconsistent across bootstraps. If all three
    # break refusal, multi-causal subspace. If 0 or only some, those that did
    # were coincidence with test prompts, not actual causal directions.
    for i, (bs_seed, d) in enumerate(zip(bootstrap_seeds, lda_top1_bootstraps), start=1):
        with ablate_dir(bundle.model, d):
            _record(
                f"C{i}_ablate_LDA_top1_bootstrap_seed_{bs_seed}",
                "inconsistent across bootstraps (predicted: only some or none break refusal)",
                _gen(bundle, test_h),
            )

    # D: top-k LDA orthogonal subspace from single bootstrap
    with ablate_subspace(bundle.model, lda_subspace):
        _record(
            f"D_ablate_LDA_top{args.lda_k}_subspace_seed_101",
            "subspace size > 1; works iff any of C1-C3 worked or if cumulative effect matters",
            _gen(bundle, test_h),
        )

    # E: L3 d_hat (different layer, same population direction → bootstrap-stable)
    with ablate_dir(bundle.model, d_hat_L3):
        _record(
            "E_ablate_d_hat_L3",
            "≤ 0.3 (different layer, same population direction; predicted causally sufficient)",
            _gen(bundle, test_h),
        )

    # F: random direction control (specificity)
    with ablate_dir(bundle.model, rand):
        _record("F_ablate_random", "≈ 1.00 (control)", _gen(bundle, test_h))

    record = {
        "step": "phase1_subspace_ablation",
        "model": bundle.name,
        "device": bundle.device,
        "extract_layer": args.extract_layer,
        "n_test": len(test_h),
        "lda_k": args.lda_k,
        "split_seed": args.seed,
        "cos_lda_pairs_bootstrap": cos_lda_pairs,
        "cos_lda_vs_dhat_bootstrap": dict(zip(
            [f"bs{bs}" for bs in bootstrap_seeds], cos_lda_vs_dhat)),
        "cos_d_hat_L13_vs_d_hat_L3": cos_d_l3,
        "baseline_harmful_refusal": base_r.rate,
        "baseline_harmful_p50": base_c.p50_chars,
        "cells": cells,
    }
    write_json(run_dir / "result.json", record)
    log.info("run record -> %s", run_dir / "result.json")

    md_path = RESULTS / "phase1_subspace_ablation.md"
    md_path.write_text(_render_summary(record))
    log.info("summary -> %s", md_path)

    print(
        f"\nphase1_subspace_ablation | baseline refusal={base_r.rate:.2f}\n"
        + "\n".join(f"  {c['name']:<40} refusal={c['refusal_rate']:.2f} (predicted: {c['predicted_refusal']})"
                    for c in cells)
    )
    return 0


def _render_summary(rec):
    cells = rec["cells"]
    base = rec["baseline_harmful_refusal"]
    md = [
        "# Phase 1 — Subspace ablation (causal vs readable directions)",
        "",
        "**Question:** the iterative-LDA analysis found ≥15 mutually-orthogonal "
        "perfect-classification directions in L13's residual stream. Which of "
        "them are *causally* on the refusal pathway, vs *correlationally* readable?",
        "",
        f"- Model: `{rec['model']}` on `{rec['device']}`",
        f"- L13 extraction layer; matched to Step 3's seed and test set (n={rec['n_test']})",
        f"- LDA orthogonal-subspace ablation uses k={rec['lda_k']} directions",
        f"- cos(d_hat_L13, LDA_top1) = {rec['cos_d_hat_L13_vs_LDA_top1']:.3f}",
        f"- cos(d_hat_L13, d_hat_L3)  = {rec['cos_d_hat_L13_vs_d_hat_L3']:.3f}",
        f"- Baseline harmful refusal: **{base:.2f}** (no hook), p50 chars: {rec['baseline_harmful_p50']:.0f}",
        "",
        "## Results",
        "",
        "| cell | direction(s) ablated | predicted | refusal rate | n_refused/n | p50 chars |",
        "|---|---|---|---:|---|---:|",
    ]
    for c in cells:
        md.append(
            f"| {c['name']} | {c['name'].replace('_', ' ')} | {c['predicted_refusal']} | "
            f"{c['refusal_rate']:.2f} | {c['n_refused']}/{c['n']} | {c['p50_chars']:.0f} |"
        )
    md.append("")
    md.append("## Logged predictions (made before running)")
    md.append("")
    md.append("- **C ≥ 0.7**: LDA top-1 is correlationally readable, not on the causal path.")
    md.append("- **D ≤ 0.3**: top-5 LDA subspace ablation works — multiple causal directions exist.")
    md.append("- **E ≤ 0.3**: L3 d_hat is also causal (same subspace as L13 d_hat, just a different basis).")
    md.append("")
    md.append("## Sample completions per cell (first 3)")
    md.append("")
    for c in cells:
        md.append(f"### {c['name']}")
        for g in c["sample_gens"]:
            md.append(f"  > {g}")
        md.append("")
    return "\n".join(md) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
