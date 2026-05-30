"""
Phase 1.5-B — Balanced fictional-framing test (≥30 prompts).

Closes the open-loop prediction from phase1_step3c_expanded_contrastive:
the previous test used 3 fictional-framing prompts added to a 138-prompt
train set — too statistically weak (cos=1.000 by construction; 0/3 moved).
This run uses 30 fictional-framing prompts and measures whether the
augmented d̂ pulls the held-out adversarial-jailbreak prompts onto its
axis.

Train set:
  - Harmful: 150 AdvBench + 30 fictional-framing = 180 prompts
  - Harmless: 150 Alpaca (unchanged — augmenting harmless would introduce
    a length-matching confound; the 17% imbalance doesn't break diff-of-means)

Test set (clean separation — never seen in any d̂ training):
  - 15 adversarial_jailbreak prompts from data/affect-test.jsonl
  - Includes the 3 originals from phase1_step3c (the ones that were the
    motivation for this test)

Pre-registered prediction (logged before any run):
  (A) cos(d̂_old, d̂_augmented) ≥ 0.85 AND mean projection of test prompts
      rises meaningfully under d̂_augmented vs d̂_old
      → unitary mechanism; the original d̂ had a fictional-framing coverage
         gap that re-extraction fixes.
  (B) cos diverges (< 0.85) OR test projections stay low under both
      → separable fictional-framing refusal mechanism;
         "one causal direction" has a known exception worth flagging.

Both outcomes are publishable; (B) is more interesting because it would
mean the unitary-mechanism claim has a real boundary.

Run:
    cd ~/safe_ai/mech-security
    python -m experiments.phase1_fictional_framing_balanced
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments._runner import (
    ARTIFACTS_FIGURES,
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
from mech_security.directions import diff_of_means, project, unit

log = get_logger("phase1_fictional_balanced")


def load_jsonl_simple(path: Path, key: str = "text", filter_label: str | None = None) -> list[str]:
    """Load a JSONL file and extract a field, optionally filtering by label/category."""
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract-layer", type=int, default=13)
    args = ap.parse_args()

    run_dir = new_run_dir("phase1_fictional_balanced")
    log.info("run_dir: %s | extract=L%d", run_dir, args.extract_layer)

    bundle = get_model()
    log.info("model: %s | device=%s", bundle.name, bundle.device)

    # === Load all three prompt sets ===
    advbench_harmful, alpaca_harmless = load_jsonl_pairs(
        Path(__file__).resolve().parent.parent / "data/contrastive.jsonl"
    )
    fictional_harmful = load_jsonl_simple(
        Path(__file__).resolve().parent.parent / "data/fictional_framing_train.jsonl"
    )
    affect_test_path = Path(__file__).resolve().parent.parent / "data/affect-test.jsonl"
    test_prompts = load_jsonl_simple(affect_test_path, filter_label="adversarial_jailbreak")

    log.info("train sets: %d AdvBench + %d fictional-framing = %d harmful | %d Alpaca harmless",
             len(advbench_harmful), len(fictional_harmful),
             len(advbench_harmful) + len(fictional_harmful), len(alpaca_harmless))
    log.info("test set: %d adversarial_jailbreak prompts (never seen in any d̂ training)",
             len(test_prompts))

    # === d̂_old — from cache ===
    extra = (f"{bundle.name}|dtype={bundle.model.cfg.dtype}|L{args.extract_layer}|"
             f"resid_post|last_token|advbench_full")
    key_ah = content_hash(advbench_harmful, extra=extra + "|harmful")
    key_al = content_hash(alpaca_harmless, extra=extra + "|harmless")
    log.info("loading d̂_old activations (cache hit expected) ...")
    H_advbench = cached_activations(
        key_ah, lambda: cache_resid(bundle, advbench_harmful, layer=args.extract_layer, show_progress=False)
    )
    L_alpaca = cached_activations(
        key_al, lambda: cache_resid(bundle, alpaca_harmless, layer=args.extract_layer, show_progress=False)
    )
    d_hat_old = unit(diff_of_means(H_advbench, L_alpaca))
    log.info("d̂_old extracted from %d harmful + %d harmless", len(H_advbench), len(L_alpaca))

    # === Cache fictional-framing activations ===
    extra_f = (f"{bundle.name}|dtype={bundle.model.cfg.dtype}|L{args.extract_layer}|"
               f"resid_post|last_token|fictional_framing_30")
    key_f = content_hash(fictional_harmful, extra=extra_f + "|harmful")
    log.info("caching activations for %d fictional-framing prompts ...", len(fictional_harmful))
    H_fictional = cached_activations(
        key_f, lambda: cache_resid(bundle, fictional_harmful, layer=args.extract_layer, show_progress=False)
    )
    log.info("fictional-framing activations cached | shape=%s", tuple(H_fictional.shape))

    # === Cache test-set activations ===
    extra_t = (f"{bundle.name}|dtype={bundle.model.cfg.dtype}|L{args.extract_layer}|"
               f"resid_post|last_token|affect_jailbreak_15")
    key_t = content_hash(test_prompts, extra=extra_t)
    log.info("caching activations for %d test (adversarial_jailbreak) prompts ...", len(test_prompts))
    H_test = cached_activations(
        key_t, lambda: cache_resid(bundle, test_prompts, layer=args.extract_layer, show_progress=False)
    )

    # === d̂_augmented from the augmented harmful set ===
    H_augmented = torch.cat([H_advbench, H_fictional], dim=0)
    d_hat_augmented = unit(diff_of_means(H_augmented, L_alpaca))
    log.info("d̂_augmented extracted from %d (=150 AdvBench + %d fictional) harmful + %d harmless",
             H_augmented.shape[0], len(H_fictional), L_alpaca.shape[0])

    # === The headline measurement: cos(d̂_old, d̂_augmented) ===
    cos = float((d_hat_old * d_hat_augmented).sum())
    log.info("cos(d̂_old, d̂_augmented) = %.4f", cos)

    # === Projections on the test prompts (the actual coverage-gap test) ===
    test_proj_old = project(H_test, d_hat_old).numpy()
    test_proj_new = project(H_test, d_hat_augmented).numpy()
    # For scale reference, also project the harmful + harmless TRAIN sets
    advbench_proj_old = project(H_advbench, d_hat_old).numpy()
    advbench_proj_new = project(H_advbench, d_hat_augmented).numpy()
    alpaca_proj_old = project(L_alpaca, d_hat_old).numpy()
    alpaca_proj_new = project(L_alpaca, d_hat_augmented).numpy()
    fictional_proj_old = project(H_fictional, d_hat_old).numpy()
    fictional_proj_new = project(H_fictional, d_hat_augmented).numpy()

    # "Moved onto the axis": for each test prompt, did its projection rise
    # from somewhere near alpaca_mean to somewhere near advbench_mean?
    # Define a threshold: midpoint between alpaca_mean and advbench_mean
    # under the OLD d̂. A prompt "on the axis" iff projection > midpoint.
    midpoint_old = 0.5 * (advbench_proj_old.mean() + alpaca_proj_old.mean())
    midpoint_new = 0.5 * (advbench_proj_new.mean() + alpaca_proj_new.mean())
    test_on_axis_old = int((test_proj_old > midpoint_old).sum())
    test_on_axis_new = int((test_proj_new > midpoint_new).sum())
    log.info("test prompts above harmful/harmless midpoint:  old d̂: %d/%d  |  new d̂: %d/%d",
             test_on_axis_old, len(test_prompts),
             test_on_axis_new, len(test_prompts))

    # === Per-prompt rise: how much each test prompt's projection moved (in natural-scale units) ===
    scale_old = float(advbench_proj_old.mean() - alpaca_proj_old.mean())
    scale_new = float(advbench_proj_new.mean() - alpaca_proj_new.mean())
    test_proj_old_normalized = (test_proj_old - alpaca_proj_old.mean()) / max(scale_old, 1e-6)
    test_proj_new_normalized = (test_proj_new - alpaca_proj_new.mean()) / max(scale_new, 1e-6)
    rise = test_proj_new_normalized - test_proj_old_normalized
    log.info("test-prompt normalized projection shift (fraction of natural scale): "
             "mean=%.3f, min=%.3f, max=%.3f", rise.mean(), rise.min(), rise.max())

    # === Verdict per the pre-registered prediction ===
    cos_passes = cos >= 0.85
    # "rises meaningfully" — mean test-prompt projection (normalized) increases by ≥ 0.10 of natural scale
    rise_passes = rise.mean() >= 0.10
    verdict = (
        "PREDICTION CONFIRMED — unitary mechanism, coverage gap"
        if cos_passes and rise_passes
        else "PREDICTION REFUTED — fictional-framing refusal looks separable"
        if not cos_passes
        else "MIXED — directions are aligned (cos ≥ 0.85) but test prompts didn't rise on the axis"
    )
    log.info("VERDICT: %s", verdict)

    # === Plot: projection distributions before and after ===
    _plot_projections(
        d_hat_label="d̂_old (AdvBench-only)",
        advbench_proj=advbench_proj_old, alpaca_proj=alpaca_proj_old,
        fictional_proj=fictional_proj_old, test_proj=test_proj_old,
        midpoint=midpoint_old, scale=scale_old,
        ax_title="d̂_old: 150 AdvBench + 150 Alpaca",
        save_to=ARTIFACTS_FIGURES / "phase1_fictional_balanced_old.png",
    )
    _plot_projections(
        d_hat_label="d̂_augmented (+30 fictional-framing)",
        advbench_proj=advbench_proj_new, alpaca_proj=alpaca_proj_new,
        fictional_proj=fictional_proj_new, test_proj=test_proj_new,
        midpoint=midpoint_new, scale=scale_new,
        ax_title="d̂_augmented: 150 AdvBench + 30 fictional-framing + 150 Alpaca",
        save_to=ARTIFACTS_FIGURES / "phase1_fictional_balanced_new.png",
    )

    # Combined figure
    fig_combined = ARTIFACTS_FIGURES / "phase1_fictional_balanced.png"
    _plot_combined(
        advbench_proj_old, alpaca_proj_old, fictional_proj_old, test_proj_old,
        advbench_proj_new, alpaca_proj_new, fictional_proj_new, test_proj_new,
        cos, fig_combined,
    )

    record = {
        "step": "phase1_fictional_framing_balanced",
        "model": bundle.name,
        "extract_layer": args.extract_layer,
        "n_advbench": len(advbench_harmful),
        "n_fictional_train": len(fictional_harmful),
        "n_alpaca": len(alpaca_harmless),
        "n_test_adversarial_jailbreak": len(test_prompts),
        "cos_old_vs_augmented": cos,
        "natural_scale_old": scale_old,
        "natural_scale_new": scale_new,
        "midpoint_threshold_old": midpoint_old,
        "midpoint_threshold_new": midpoint_new,
        "test_on_axis_old_count": test_on_axis_old,
        "test_on_axis_new_count": test_on_axis_new,
        "test_proj_old": test_proj_old.tolist(),
        "test_proj_new": test_proj_new.tolist(),
        "test_proj_old_normalized": test_proj_old_normalized.tolist(),
        "test_proj_new_normalized": test_proj_new_normalized.tolist(),
        "test_normalized_rise_mean": float(rise.mean()),
        "test_normalized_rise_min": float(rise.min()),
        "test_normalized_rise_max": float(rise.max()),
        "verdict": verdict,
        "pre_registered_prediction": {
            "cos_threshold": 0.85,
            "rise_threshold_fraction_of_natural_scale": 0.10,
            "cos_passes": cos_passes,
            "rise_passes": rise_passes,
        },
        "test_prompts": test_prompts,
    }
    write_json(run_dir / "result.json", record)
    log.info("run record -> %s", run_dir / "result.json")

    md_path = RESULTS / "phase1_fictional_framing_balanced.md"
    md_path.write_text(_render_summary(record))
    log.info("summary -> %s", md_path)

    print("\n=== phase1_fictional_framing_balanced ===")
    print(f"cos(d̂_old, d̂_augmented) = {cos:.4f}")
    print("test prompts above harmful/harmless midpoint:")
    print(f"  under d̂_old:        {test_on_axis_old}/{len(test_prompts)}")
    print(f"  under d̂_augmented:  {test_on_axis_new}/{len(test_prompts)}")
    print(f"mean test-prompt projection rise (fraction of natural scale): {rise.mean():+.3f}")
    print(f"VERDICT: {verdict}")
    return 0


def _plot_projections(d_hat_label, advbench_proj, alpaca_proj, fictional_proj,
                       test_proj, midpoint, scale, ax_title, save_to):
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(min(alpaca_proj.min(), test_proj.min()) - 5,
                        max(advbench_proj.max(), fictional_proj.max(), test_proj.max()) + 5, 50)
    ax.hist(alpaca_proj, bins=bins, alpha=0.6, label=f"Alpaca harmless (n={len(alpaca_proj)})", color="tab:blue")
    ax.hist(advbench_proj, bins=bins, alpha=0.6, label=f"AdvBench harmful (n={len(advbench_proj)})", color="tab:red")
    ax.hist(fictional_proj, bins=bins, alpha=0.5, label=f"fictional-framing train (n={len(fictional_proj)})", color="tab:orange", edgecolor="black", linewidth=0.5)
    # Test prompts as a rug
    ax.plot(test_proj, np.zeros_like(test_proj) - 1, "|", color="tab:green", markersize=14,
             markeredgewidth=2, label=f"test adversarial_jailbreak (n={len(test_proj)})")
    ax.axvline(midpoint, color="black", linestyle=":", linewidth=1, label=f"harmful/harmless midpoint = {midpoint:.1f}")
    ax.set_xlabel(f"projection on {d_hat_label}")
    ax.set_ylabel("count")
    ax.set_title(ax_title)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_to, dpi=140)
    plt.close(fig)


def _plot_combined(advbench_old, alpaca_old, fictional_old, test_old,
                    advbench_new, alpaca_new, fictional_new, test_new,
                    cos, save_to):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    for ax, (advb, alpa, fict, test, label) in zip(
        [ax1, ax2],
        [(advbench_old, alpaca_old, fictional_old, test_old, "d̂_old (150 AdvBench + 150 Alpaca)"),
         (advbench_new, alpaca_new, fictional_new, test_new, "d̂_augmented (+30 fictional-framing)")],
    ):
        bins = np.linspace(min(alpa.min(), test.min()) - 5,
                            max(advb.max(), fict.max(), test.max()) + 5, 50)
        ax.hist(alpa, bins=bins, alpha=0.55, label=f"Alpaca (n={len(alpa)})", color="tab:blue")
        ax.hist(advb, bins=bins, alpha=0.55, label=f"AdvBench (n={len(advb)})", color="tab:red")
        ax.hist(fict, bins=bins, alpha=0.55, label=f"fictional-framing train (n={len(fict)})", color="tab:orange", edgecolor="black", linewidth=0.5)
        ax.plot(test, np.zeros_like(test) - 1, "|", color="tab:green", markersize=14,
                 markeredgewidth=2, label=f"test adversarial_jailbreak (n={len(test)})")
        midpoint = 0.5 * (advb.mean() + alpa.mean())
        ax.axvline(midpoint, color="black", linestyle=":", linewidth=1, label="harmful/harmless midpoint")
        ax.set_xlabel(f"projection on {label}")
        ax.set_title(label)
        ax.legend(loc="upper left", fontsize=7)
    ax1.set_ylabel("count")
    fig.suptitle(f"Phase 1.5-B fictional-framing test — cos(d̂_old, d̂_augmented) = {cos:.3f}", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_to, dpi=140)
    plt.close(fig)


def _render_summary(rec):
    pr = rec["pre_registered_prediction"]
    md = [
        "# Phase 1.5-B — balanced fictional-framing test",
        "",
        f"**Headline: {rec['verdict']}**",
        "",
        f"- cos(d̂_old, d̂_augmented) = **{rec['cos_old_vs_augmented']:.4f}** "
        f"(threshold for unitary-mechanism: ≥ {pr['cos_threshold']})",
        f"- Mean test-prompt projection rise (fraction of natural scale) = "
        f"**{rec['test_normalized_rise_mean']:+.3f}** (threshold: ≥ {pr['rise_threshold_fraction_of_natural_scale']:.2f})",
        f"- Test prompts above harmful/harmless midpoint: "
        f"{rec['test_on_axis_old_count']}/{rec['n_test_adversarial_jailbreak']} → "
        f"{rec['test_on_axis_new_count']}/{rec['n_test_adversarial_jailbreak']} under augmented d̂",
        "",
        f"- Model: `{rec['model']}` | extract layer L{rec['extract_layer']}",
        f"- Train: {rec['n_advbench']} AdvBench harmful + {rec['n_fictional_train']} fictional-framing harmful + {rec['n_alpaca']} Alpaca harmless",
        f"- Test: {rec['n_test_adversarial_jailbreak']} `adversarial_jailbreak` prompts from `data/affect-test.jsonl` (never seen in any d̂ training)",
        f"- Natural scale: d̂_old = {rec['natural_scale_old']:.2f}, d̂_augmented = {rec['natural_scale_new']:.2f}",
        "",
        "## Pre-registered prediction (logged before running)",
        "",
        "> (A) cos(d̂_old, d̂_augmented) ≥ 0.85 AND mean test-prompt projection rises ≥ 0.10 of natural scale",
        ">     → unitary mechanism with coverage gap.",
        "> (B) cos < 0.85 OR test projections don't rise",
        ">     → separable fictional-framing refusal mechanism; \"one direction\" has a known exception.",
        "",
        "## Per-prompt test projections",
        "",
        "| Test prompt (first 80 chars) | proj under d̂_old | proj under d̂_aug | rise (frac of nat scale) |",
        "|---|---:|---:|---:|",
    ]
    for prompt, p_old, p_new in zip(
        rec["test_prompts"], rec["test_proj_old"], rec["test_proj_new"]
    ):
        # Normalize to natural-scale fraction
        rise = (
            p_new - rec["natural_scale_new"] * 0  # rebuild from normalized
        )
        # use the saved normalized arrays
        idx = rec["test_prompts"].index(prompt)
        n_old = rec["test_proj_old_normalized"][idx]
        n_new = rec["test_proj_new_normalized"][idx]
        md.append(f"| {prompt[:80]} | {p_old:+.2f} | {p_new:+.2f} | {n_new - n_old:+.3f} |")
    md.append("")
    return "\n".join(md) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
