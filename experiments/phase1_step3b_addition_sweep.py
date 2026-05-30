"""
Phase 1 — Step 3b: addition-coefficient × injection-layer sweep.

C4 of the steering experiment failed at coeff=4 and coeff=25 at L13. The
planner's correct critique: that's two coefficients at one layer, not a
swept hyperparameter. Before claiming the mechanism is asymmetric (ablation
works, addition doesn't), we must rule out the boring explanation: addition
needs the right coefficient and/or the right injection layer.

This script sweeps:
- coefficients: {1, 4, 10, 25, 50, 100, 200}
- injection layers: {7, 10, 13, 16, 20}

For each (coeff, layer) cell:
  - Add `coeff * d_hat` at `hook_resid_post` of the chosen layer during the
    forward pass of each harmless test prompt.
  - Generate the completion at temperature=0.
  - Score with the patched substring scorer.
  - Record refusal rate AND coherence (chars p50).

Pass criterion for C4 (per cell): refusal rate increases by ≥ 0.30
versus the baseline_harmless rate. Coherence remains within 60% of baseline.

If C4 passes anywhere in the grid: the asymmetry claim is wrong — addition
works when properly tuned. If C4 fails everywhere AND coherence holds:
asymmetry survives.

Run:
    python -m experiments.phase1_step3b_addition_sweep
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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
from src.activations import cache_resid
from src.directions import add_dir, diff_of_means, unit
from src.eval import coherence_ok, refusal_rate
from src.model import generate

log = get_logger("phase1_step3b")

# Coefficients are MULTIPLES of the natural scale = mean projection of the
# harmful cluster onto d_hat (computed at runtime). 1.0 means "one
# harmful-cluster-mean worth of d_hat". 0.0 is implicit via the no-hook baseline.
# Spans ~3 orders of magnitude so we can see (a) no effect, (b) effect window,
# (c) coherence breakdown — the diagnostic shape that distinguishes a real
# asymmetric mechanism from a tuning artifact.
COEFF_MULTIPLES = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]

# Injection layers span pre-, at-, and post-extraction layer (L13). Pre-L13
# matters because addition is typically most effective at or before the
# extraction layer — components downstream get to attenuate it.
LAYERS = [3, 7, 10, 13, 16, 20, 23]


def _gen(bundle, prompts, max_new_tokens=96):
    return [generate(bundle, p, max_new_tokens=max_new_tokens, temperature=0.0)
            for p in prompts]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract-layer", type=int, default=13,
                    help="Layer at which to extract d_hat (default 13).")
    ap.add_argument("--data", type=str, default="data/contrastive.jsonl")
    ap.add_argument("--n-test", type=int, default=10,
                    help="harmless prompts per cell (lower than steering to keep "
                         "the 35-cell grid tractable)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run_dir = new_run_dir("phase1_step3b")
    log.info("run_dir: %s | extract_layer=L%d | n_test=%d",
             run_dir, args.extract_layer, args.n_test)

    bundle = get_model()
    log.info("model: %s | device=%s", bundle.name, bundle.device)

    pairs_path = Path(args.data)
    if not pairs_path.is_absolute():
        pairs_path = Path(__file__).resolve().parent.parent / pairs_path
    harmful_all, harmless_all = load_jsonl_pairs(pairs_path)

    rng = np.random.default_rng(args.seed)
    test_l_idx = rng.choice(len(harmless_all), size=args.n_test, replace=False).tolist()
    train_h = harmful_all
    train_l = [t for i, t in enumerate(harmless_all) if i not in set(test_l_idx)]
    test_l = [harmless_all[i] for i in test_l_idx]
    log.info("train: %d/%d | test: %d harmless prompts", len(train_h), len(train_l), len(test_l))

    extra = (
        f"{bundle.name}|dtype={bundle.model.cfg.dtype}|L{args.extract_layer}|"
        f"resid_post|last_token|seed{args.seed}|sweep"
    )
    key_h = content_hash(train_h, extra=extra + "|harmful")
    key_l = content_hash(train_l, extra=extra + "|harmless")
    log.info("caching residuals to extract d_hat at L%d ...", args.extract_layer)
    train_h_acts = cached_activations(
        key_h, lambda: cache_resid(bundle, train_h, layer=args.extract_layer, show_progress=False)
    )
    train_l_acts = cached_activations(
        key_l, lambda: cache_resid(bundle, train_l, layer=args.extract_layer, show_progress=False)
    )
    d_hat = unit(diff_of_means(train_h_acts, train_l_acts))
    d_hat_dev = d_hat.to(bundle.model.cfg.device).to(bundle.model.W_E.dtype)

    # Reference: cluster-mean projection on train. This IS the natural scale
    # AT THE EXTRACTION LAYER. We use it as the unit for the absolute-coefficient
    # grid, because (a) it's a single scalar that defines the coefficient axis,
    # and (b) d_hat is a single vector regardless of where we inject it.
    #
    # BUT: the right interpretive unit for cross-model portability (Phase 2) is
    # the natural scale AT THE INJECTION LAYER. We log and persist both —
    # see `natural_scale_per_layer` in the result record below.
    from src.directions import project
    train_h_proj_mean = float(project(train_h_acts, d_hat).mean())
    train_l_proj_mean = float(project(train_l_acts, d_hat).mean())
    natural_scale = train_h_proj_mean - train_l_proj_mean  # harmful-vs-harmless gap
    log.info("d_hat extracted | train harmful mean=%.1f, harmless mean=%.1f, natural scale (at extract)=%.1f",
             train_h_proj_mean, train_l_proj_mean, natural_scale)

    # Convert multiples to absolute coefficients (using extract-layer scale; see
    # natural_scale_per_layer below for the per-injection-layer reinterpretation).
    coeffs_abs = [round(m * natural_scale, 2) for m in COEFF_MULTIPLES]
    log.info("coefficient multiples %s → absolute %s", COEFF_MULTIPLES, coeffs_abs)

    # Per-injection-layer natural scale: project the train activations at each
    # injection layer L onto d_hat and report harmful_mean − harmless_mean.
    # Persisted per-layer so the absolute coefficient → per-layer-multiple
    # reinterpretation is reproducible after the fact (and portable to Phase 2).
    natural_scale_per_layer = {}
    for inj_L in LAYERS:
        extra_inj = (
            f"{bundle.name}|dtype={bundle.model.cfg.dtype}|L{inj_L}|"
            f"resid_post|last_token|seed{args.seed}|sweep"
        )
        key_h_inj = content_hash(train_h, extra=extra_inj + "|harmful")
        key_l_inj = content_hash(train_l, extra=extra_inj + "|harmless")
        train_h_at_inj = cached_activations(
            key_h_inj, lambda L=inj_L: cache_resid(bundle, train_h, layer=L, show_progress=False)
        )
        train_l_at_inj = cached_activations(
            key_l_inj, lambda L=inj_L: cache_resid(bundle, train_l, layer=L, show_progress=False)
        )
        h_mean = float(project(train_h_at_inj, d_hat).mean())
        l_mean = float(project(train_l_at_inj, d_hat).mean())
        natural_scale_per_layer[inj_L] = {
            "harmful_proj_mean": h_mean,
            "harmless_proj_mean": l_mean,
            "natural_scale": h_mean - l_mean,
        }
        log.info("natural scale at inject L%d: %.2f (h=%.2f, l=%.2f)",
                 inj_L, h_mean - l_mean, h_mean, l_mean)

    log.info("generating baseline (no hook) on n=%d harmless...", len(test_l))
    base_gens = _gen(bundle, test_l)
    base_rate = refusal_rate(base_gens).rate
    base_p50 = coherence_ok(base_gens).p50_chars
    log.info("baseline_harmless: refusal=%.2f, p50_chars=%.0f", base_rate, base_p50)

    # Sweep
    grid = []
    for layer in LAYERS:
        row = []
        for coeff_mult, coeff in zip(COEFF_MULTIPLES, coeffs_abs):
            log.info("sweep cell: L=%d coeff=%.1f ...", layer, coeff)
            with add_dir(bundle.model, d_hat_dev, coeff=coeff, layer=layer):
                gens = _gen(bundle, test_l)
            r = refusal_rate(gens)
            c = coherence_ok(gens)
            delta = r.rate - base_rate
            coherence_kept = c.p50_chars >= max(80, 0.6 * base_p50)
            c4_pass = delta >= 0.30 and coherence_kept
            cell = {
                "layer": layer,
                "coeff_multiple": coeff_mult,  # multiple of natural scale
                "coeff": coeff,                # absolute coefficient
                "refusal_rate": r.rate,
                "delta_vs_baseline": delta,
                "p50_chars": c.p50_chars,
                "coherence_kept": coherence_kept,
                "c4_pass": c4_pass,
                "n_refused": r.n_refused,
                "n": r.n,
                "sample_gens": [g.strip()[:200] for g in gens[:3]],
            }
            row.append(cell)
            log.info("    refusal=%.2f (%d/%d), p50=%.0f, Δ=%+.2f, c4_pass=%s",
                     r.rate, r.n_refused, r.n, c.p50_chars, delta, c4_pass)
        grid.append(row)

    # Persist
    any_c4_pass = any(cell["c4_pass"] for row in grid for cell in row)
    record = {
        "step": "phase1_step3b",
        "extract_layer": args.extract_layer,
        "n_test": args.n_test,
        "baseline_harmless_refusal": base_rate,
        "baseline_harmless_p50": base_p50,
        "train_harmful_proj_mean": train_h_proj_mean,
        "train_harmless_proj_mean": train_l_proj_mean,
        "natural_scale": natural_scale,                      # at extract layer
        "natural_scale_per_layer": natural_scale_per_layer,  # the Phase-2 portable knob
        "coeff_multiples": COEFF_MULTIPLES,
        "coeffs_absolute": coeffs_abs,
        "layers": LAYERS,
        "grid": grid,
        "any_c4_pass": any_c4_pass,
    }
    write_json(run_dir / "result.json", record)

    # Plot 1: two heatmaps (full grid) — refusal rate and coherence
    refusal_grid = np.array([[c["refusal_rate"] for c in row] for row in grid])
    coherence_grid = np.array([[c["p50_chars"] for c in row] for row in grid])
    c4_grid = np.array([[c["c4_pass"] for c in row] for row in grid])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    im1 = ax1.imshow(refusal_grid, aspect="auto", cmap="RdYlGn_r",
                     vmin=0, vmax=1, origin="lower")
    ax1.set_title(f"Over-refusal rate after adding (mult × natural-scale) × d_hat\n"
                  f"(baseline harmless refusal: {base_rate:.2f} | natural scale={natural_scale:.1f})")
    ax1.set_xticks(range(len(COEFF_MULTIPLES)))
    ax1.set_xticklabels([f"{m:g}" for m in COEFF_MULTIPLES])
    ax1.set_yticks(range(len(LAYERS))); ax1.set_yticklabels([f"L{l}" for l in LAYERS])
    ax1.set_xlabel("addition coefficient (multiples of natural scale)")
    ax1.set_ylabel("injection layer")
    for i in range(len(LAYERS)):
        for j in range(len(COEFF_MULTIPLES)):
            mark = "★" if c4_grid[i, j] else ""
            ax1.text(j, i, f"{refusal_grid[i, j]:.2f}{mark}",
                     ha="center", va="center", fontsize=8,
                     color="white" if refusal_grid[i, j] > 0.5 else "black")
    plt.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(coherence_grid, aspect="auto", cmap="RdYlGn",
                     vmin=0, vmax=max(1, coherence_grid.max()), origin="lower")
    ax2.set_title(f"Coherence (p50 chars)\n(baseline: {base_p50:.0f} chars; near-zero = lobotomized)")
    ax2.set_xticks(range(len(COEFF_MULTIPLES)))
    ax2.set_xticklabels([f"{m:g}" for m in COEFF_MULTIPLES])
    ax2.set_yticks(range(len(LAYERS))); ax2.set_yticklabels([f"L{l}" for l in LAYERS])
    ax2.set_xlabel("addition coefficient (multiples of natural scale)")
    ax2.set_ylabel("injection layer")
    for i in range(len(LAYERS)):
        for j in range(len(COEFF_MULTIPLES)):
            ax2.text(j, i, f"{coherence_grid[i, j]:.0f}",
                     ha="center", va="center", fontsize=8,
                     color="white" if coherence_grid[i, j] < 200 else "black")
    plt.colorbar(im2, ax=ax2)
    fig.tight_layout()
    fig_path = ARTIFACTS_FIGURES / "phase1_step3b_addition_sweep.png"
    fig.savefig(fig_path, dpi=140)
    plt.close(fig)
    log.info("heatmap figure -> %s", fig_path)

    # Plot 2: THE diagnostic — refusal & coherence on the same x-axis,
    # at the extraction layer (the natural injection point). The shape of
    # these two curves is the asymmetry-vs-tuning-artifact answer:
    # - effect window opens before coherence collapses → artifact
    # - coherence collapses before any effect window → asymmetric mechanism
    extract_layer_idx = LAYERS.index(args.extract_layer) if args.extract_layer in LAYERS else len(LAYERS) // 2
    extract_row = grid[extract_layer_idx]
    refusal_curve  = [c["refusal_rate"] for c in extract_row]
    coherence_curve = [c["p50_chars"] for c in extract_row]

    fig2, ax = plt.subplots(figsize=(8, 4.5))
    ax.set_xscale("log")
    ax.set_xlabel("addition coefficient (multiples of natural scale)")
    ax.set_ylabel("refusal rate on harmless prompts", color="tab:red")
    line_r, = ax.plot(COEFF_MULTIPLES, refusal_curve, "o-", color="tab:red",
                      linewidth=2, label="refusal rate")
    ax.axhline(base_rate, color="tab:red", linestyle=":", alpha=0.4,
               label=f"baseline refusal ({base_rate:.2f})")
    ax.set_ylim(-0.05, 1.05)
    ax.tick_params(axis="y", labelcolor="tab:red")

    ax2c = ax.twinx()
    ax2c.set_ylabel("p50 chars (coherence proxy)", color="tab:blue")
    line_c, = ax2c.plot(COEFF_MULTIPLES, coherence_curve, "s--",
                        color="tab:blue", linewidth=2, alpha=0.8, label="p50 chars")
    ax2c.axhline(base_p50, color="tab:blue", linestyle=":", alpha=0.4,
                 label=f"baseline coherence ({base_p50:.0f})")
    ax2c.set_ylim(0, max(base_p50 * 1.2, max(coherence_curve) * 1.1))
    ax2c.tick_params(axis="y", labelcolor="tab:blue")

    ax.set_title(
        f"Addition sweep at L{args.extract_layer} — refusal vs coherence\n"
        f"asymmetry-vs-artifact diagnostic"
    )
    ax.legend(handles=[line_r, line_c], loc="center left")
    ax.grid(alpha=0.3)
    fig2.tight_layout()
    fig2_path = ARTIFACTS_FIGURES / "phase1_step3b_diagnostic.png"
    fig2.savefig(fig2_path, dpi=140)
    plt.close(fig2)
    log.info("diagnostic figure -> %s", fig2_path)
    record["figures"] = {"heatmaps": str(fig_path), "diagnostic": str(fig2_path)}

    # Summary
    summary_path = RESULTS / "phase1_step3b_addition_sweep.md"
    summary_path.write_text(_render_summary(record, fig_path))

    print(f"\nphase1_step3b | baseline harmless refusal={base_rate:.2f} | any C4 cell passes? {any_c4_pass}")
    return 0


def _render_summary(rec, fig_path):
    grid = rec["grid"]
    mults = rec["coeff_multiples"]
    coeffs_abs = rec["coeffs_absolute"]
    layers = rec["layers"]
    base = rec["baseline_harmless_refusal"]
    any_pass = rec["any_c4_pass"]
    nat = rec["natural_scale"]

    lines = [
        "# Phase 1 Step 3b — addition-coefficient × layer sweep",
        "",
        "**Purpose:** rule out the hyperparameter-artifact explanation for C4 failing in the L13 steering run. "
        "Sweep injection layer × addition coefficient (multiples of the natural ||x · d_hat|| scale); "
        "report refusal rate + coherence per cell.",
        "",
        f"- d_hat extracted from L{rec['extract_layer']} on `data/contrastive.jsonl` train split",
        f"- Natural scale (harmful_proj_mean − harmless_proj_mean) = **{nat:.1f}** — used as the unit for coefficient multiples",
        f"- Baseline harmless refusal rate: **{base:.2f}** (the floor; addition must push UP by ≥ 0.30 to pass C4)",
        f"- Baseline harmless p50 chars: **{rec['baseline_harmless_p50']:.0f}**",
        f"- n_test = {rec['n_test']} per cell",
        "",
        "## Headline",
        "",
        f"**Any cell where C4 passes (Δrefusal ≥ 0.30 AND coherence kept ≥ max(80, 0.6×baseline)):** "
        f"{'YES — addition CAN induce refusal at the right (coeff, layer); the L13 asymmetry was a tuning artifact.' if any_pass else 'NO — asymmetry survives the full sweep. Across 2+ orders of magnitude and pre/post extraction layers, addition cannot manufacture refusal before coherence collapses.'}",
        "",
        "## Over-refusal rate after addition (rows = injection layer, cols = coeff_multiple × natural_scale)",
        "",
        "| layer | " + " | ".join(f"{m:g}×" for m in mults) + " |",
        "|---|" + "---|" * len(mults),
    ]
    for row, layer in zip(grid, layers):
        cells = []
        for cell in row:
            marker = "★" if cell["c4_pass"] else ""
            cells.append(f"{cell['refusal_rate']:.2f}{marker}")
        lines.append(f"| L{layer} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("**★ = passes C4 (Δrefusal ≥ 0.30 AND coherence kept ≥ max(80, 0.6×baseline))**")
    lines.append("")
    lines.append(f"_Absolute coefficient values: {coeffs_abs}_")
    lines.append("")
    lines.append("## Coherence (p50 chars) after addition")
    lines.append("")
    lines.append("| layer | " + " | ".join(f"{m:g}×" for m in mults) + " |")
    lines.append("|---|" + "---|" * len(mults))
    for row, layer in zip(grid, layers):
        lines.append(f"| L{layer} | " + " | ".join(f"{c['p50_chars']:.0f}" for c in row) + " |")
    lines.append("")
    figs = rec.get("figures", {})
    if figs:
        lines.append(f"Figures: heatmaps `{figs.get('heatmaps')}`, diagnostic line plot `{figs.get('diagnostic')}`")
    else:
        lines.append(f"Figure: `{fig_path}`")
    lines.append("")
    lines.append("## Interpretation guide (the diagnostic figure is the centerpiece)")
    lines.append("")
    lines.append("Two curves on a shared log-x axis at the extraction layer (refusal rate in red; p50 chars in blue):")
    lines.append("")
    lines.append("- **Refusal rises before coherence collapses** → C4 was a tuning artifact. The mechanism is bidirectional; "
                 "we under-tuned the addition coefficient in the initial steering run.")
    lines.append("- **Coherence collapses before any refusal effect window opens** → the asymmetry is real. d_hat at L13 is "
                 "causally necessary for refusal but not causally sufficient — adding the direction without the broader "
                 "computational context just destabilizes the model.")
    lines.append("- **Refusal flat AND coherence flat** → addition is being attenuated downstream of the injection point. "
                 "Try injection at multiple pre-extraction layers (this script already includes layers pre-L13).")
    lines.append("")
    lines.append("## What this resolves")
    lines.append("- Whether to claim asymmetric vs symmetric mechanism in any future writeup.")
    lines.append("- Whether the runbook's C4 control was failed by the d_hat mechanism or by our tuning.")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
