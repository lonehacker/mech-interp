"""
Phase 2 Step 3d — operating-band addition sweep on Qwen2.5-3B.

Per planner protocol. Replicates Gemma's §4.4 addition × layer × coefficient
sweep, on the Qwen target, after the normalization gate confirmed the
12× natural-scale gap is genuine (not a hook mismatch).

Pre-registered design (see results/phase2_step3d_preregistration.md):
- d̂ from L14 (holding extraction constant; mirrors Phase 1 §4.4)
- Inject layers: L3, L7, L11, L15, L19, L23, L27, L31
- Coefficients: 0.5×, 1×, 2×, 4× of natural_scale_at_L14 (= 8.94, NOT
  per-injection-layer scale — Phase 1 §4.4 established this is the
  portable unit)
- Target: 10 fixed CodeAlpaca-harmless prompts (subset of the 30 held-out
  step 3b harmless test set, deterministic seed=1)
- Substring scorer primary; judge any cell with substring lift > 0.10
- Specificity control at the best cell

Outcome maps to the A/B/C tree:
  (A) No cell induces refusal → d̂ is causally inert
  (B) Some cell induces refusal → d̂ is causal, L14 ablation wasn't the
      right intervention (wrong layer/scale)
  (C) Distinguishes from (B) only after the ablation follow-up: if a
      single-layer ablation at the live layer collapses refusal → Phase 2
      replicated at a different depth than Gemma.

Run:
    cd ~/safe_ai/mech-security
    python -m experiments.phase2_step3d_operating_band_sweep
"""

from __future__ import annotations

import argparse
import json
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

log = get_logger("phase2_step3d")


def _harmless_subset(jsonl_path: Path, seed: int = 1, n_test: int = 30, n_subset: int = 10):
    """Take the same 30 held-out CodeAlpaca harmless from step 3b's split,
    then deterministically subset to n_subset for the sweep."""
    recs = [json.loads(l) for l in jsonl_path.open()]
    harmless_recs = [r for r in recs if r["label"] == "harmless"]
    _, test = train_test_split(harmless_recs, seed=seed, n_test=n_test)
    full_30 = [r["text"] for r in test]
    return full_30[:n_subset]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--data", type=str, default="data/code_contrastive.jsonl")
    ap.add_argument("--extract-layer", type=int, default=14,
                    help="Layer to extract d̂ from (held constant across cells)")
    ap.add_argument("--inject-layers", type=int, nargs="+",
                    default=[3, 7, 11, 15, 19, 23, 27, 31])
    ap.add_argument("--coeff-mults", type=float, nargs="+",
                    default=[0.5, 1.0, 2.0, 4.0])
    ap.add_argument("--split-seed", type=int, default=1)
    ap.add_argument("--n-subset", type=int, default=10,
                    help="N harmless prompts per cell")
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--judge-lift-threshold", type=float, default=0.10,
                    help="Only LLM-judge cells with substring lift above this")
    args = ap.parse_args()

    run_dir = new_run_dir("phase2_step3d")
    log.info("run_dir: %s", run_dir)

    bundle = get_model(args.model)
    log.info("model: %s | n_layers=%d d_model=%d device=%s dtype=%s",
             bundle.name, bundle.n_layers, bundle.d_model, bundle.device, bundle.model.cfg.dtype)

    pairs_path = Path(args.data)
    if not pairs_path.is_absolute():
        pairs_path = Path(__file__).resolve().parent.parent / pairs_path

    # === Reconstruct d̂ at extraction layer (cache hit) ===
    # Use the same 120 harmful train + 150 harmless as Step 3 + 3b — d̂ must be
    # the SAME direction.
    recs = [json.loads(l) for l in pairs_path.open()]
    harm_recs = [r for r in recs if r["label"] == "harmful"]
    harmless = [r["text"] for r in recs if r["label"] == "harmless"]
    train_recs, _ = stratified_split(
        harm_recs, key_fn=lambda r: r["source"], seed=args.split_seed, n_test=30,
    )
    harmful_train = [r["text"] for r in train_recs]
    log.info("harmful train (matches step 3/3b): %d", len(harmful_train))

    fmt = lambda m: format_prompt_for_bundle(bundle, m)
    log.info("loading cached activations at L%d ...", args.extract_layer)
    d_hat, _H, _L, meta = extract_d_hat(
        bundle, harmful_train, harmless,
        layer=args.extract_layer, format_fn=fmt, extra_tag="phase2",
    )
    natural_scale = meta["natural_scale"]
    log.info("d_hat from L%d (matches step 3/3b) | natural scale = %.3f",
             args.extract_layer, natural_scale)
    d_dev = d_hat.to(bundle.device).to(bundle.model.cfg.dtype)

    # === Target prompts ===
    harmless_subset = _harmless_subset(pairs_path, seed=args.split_seed,
                                        n_test=30, n_subset=args.n_subset)
    log.info("harmless test subset: %d (subset of step 3b's 30)", len(harmless_subset))

    # === Baseline (no hook) ===
    log.info("generating baseline (no hook) on %d harmless prompts...", len(harmless_subset))
    base_gen = generate_batch(bundle, harmless_subset, max_new_tokens=args.max_new_tokens)
    base_refused = sum(is_refusal(g) for g in base_gen)
    base_rate = base_refused / len(base_gen)
    log.info("  baseline substring refusal: %d / %d = %.3f", base_refused, len(base_gen), base_rate)

    # === Sweep ===
    cells = {}
    cell_meta = {}
    total_cells = len(args.inject_layers) * len(args.coeff_mults)
    log.info("sweep: %d cells × %d prompts = %d generations",
             total_cells, args.n_subset, total_cells * args.n_subset)
    cell_idx = 0
    for L_inject in args.inject_layers:
        for cmult in args.coeff_mults:
            cell_idx += 1
            coeff = cmult * natural_scale
            cell_name = f"add_L{L_inject}_c{cmult}x"
            log.info("[%d/%d] %s: inject L%d, coeff=%.3f (%.1fx natural scale)",
                     cell_idx, total_cells, cell_name, L_inject, coeff, cmult)
            with add_dir(bundle.model, d_dev, coeff=coeff, layer=L_inject):
                gen = generate_batch(bundle, harmless_subset, max_new_tokens=args.max_new_tokens)
            n_ref = sum(is_refusal(g) for g in gen)
            rate = n_ref / len(gen)
            mean_len = float(np.mean([len(g) for g in gen]))
            cells[cell_name] = gen
            cell_meta[cell_name] = {
                "inject_layer": L_inject,
                "coeff_mult": cmult,
                "coeff": coeff,
                "n_refused_substr": n_ref,
                "refusal_rate_substr": rate,
                "lift_vs_baseline": rate - base_rate,
                "mean_chars": mean_len,
            }
            log.info("       substr refusal=%d/%d=%.2f (lift %+.2f), mean_chars=%.0f",
                     n_ref, len(gen), rate, rate - base_rate, mean_len)

    # === Save partial before judge ===
    partial = {
        "step": "phase2_step3d_operating_band_sweep",
        "model": bundle.name,
        "extract_layer": args.extract_layer,
        "natural_scale": natural_scale,
        "inject_layers": args.inject_layers,
        "coeff_mults": args.coeff_mults,
        "n_subset": args.n_subset,
        "split_seed": args.split_seed,
        "harmless_subset": harmless_subset,
        "baseline_gen": base_gen,
        "baseline_substr_refusal_rate": base_rate,
        "completions": cells,
        "cell_meta": cell_meta,
    }
    write_json(run_dir / "partial.json", partial)
    log.info("partial -> %s", run_dir / "partial.json")

    # === Identify cells worth judging ===
    cells_to_judge = [name for name, meta in cell_meta.items()
                      if meta["lift_vs_baseline"] >= args.judge_lift_threshold]
    log.info("cells with substring lift >= %.2f: %d",
             args.judge_lift_threshold, len(cells_to_judge))
    judge_summary = {}
    if cells_to_judge:
        from src.eval_llm import judge_many
        log.info("running Haiku 4.5 judge on baseline + %d lifted cells...", len(cells_to_judge))
        pairs = list(zip(harmless_subset, base_gen))
        verdicts = judge_many(pairs, show_progress=False)
        base_labels = [v.label for v in verdicts]
        base_judge_rate = sum(1 for l in base_labels if l == "REFUSED") / len(base_labels)
        judge_summary["baseline"] = {
            "labels": base_labels,
            "n_refused": sum(1 for l in base_labels if l == "REFUSED"),
            "refusal_rate": base_judge_rate,
        }
        for name in cells_to_judge:
            pairs = list(zip(harmless_subset, cells[name]))
            verdicts = judge_many(pairs, show_progress=False)
            labels = [v.label for v in verdicts]
            n_ref = sum(1 for l in labels if l == "REFUSED")
            judge_summary[name] = {
                "labels": labels,
                "n_refused": n_ref,
                "refusal_rate": n_ref / len(labels),
                "judge_lift_vs_baseline": n_ref / len(labels) - base_judge_rate,
            }
            log.info("  %-25s judge refusal=%d/%d=%.2f (lift %+.2f)",
                     name, n_ref, len(labels), n_ref / len(labels),
                     n_ref / len(labels) - base_judge_rate)
    else:
        log.info("no cells worth judging; skipping LLM judge.")

    # === Specificity control at best cell (if any showed lift) ===
    spec_control = None
    if cell_meta:
        best_cell_name = max(cell_meta, key=lambda k: cell_meta[k]["lift_vs_baseline"])
        best_meta = cell_meta[best_cell_name]
        if best_meta["lift_vs_baseline"] >= args.judge_lift_threshold:
            log.info("running specificity control: random unit vector at %s",
                     best_cell_name)
            rand_dir = random_unit_vector(bundle.d_model, seed=0, device=bundle.device).to(bundle.model.cfg.dtype)
            with add_dir(bundle.model, rand_dir, coeff=best_meta["coeff"], layer=best_meta["inject_layer"]):
                rand_gen = generate_batch(bundle, harmless_subset, max_new_tokens=args.max_new_tokens)
            n_ref_rand = sum(is_refusal(g) for g in rand_gen)
            spec_control = {
                "cell": best_cell_name,
                "inject_layer": best_meta["inject_layer"],
                "coeff": best_meta["coeff"],
                "n_refused_substr": n_ref_rand,
                "refusal_rate_substr": n_ref_rand / len(rand_gen),
                "lift_vs_baseline_substr": n_ref_rand / len(rand_gen) - base_rate,
                "completions": rand_gen,
            }
            log.info("  random direction same cell: substr refusal=%d/%d=%.2f (lift %+.2f)",
                     n_ref_rand, len(rand_gen), n_ref_rand / len(rand_gen),
                     n_ref_rand / len(rand_gen) - base_rate)

    # === Heatmap figure ===
    n_layers = len(args.inject_layers)
    n_coeffs = len(args.coeff_mults)
    rate_grid = np.zeros((n_coeffs, n_layers))
    for ci, cmult in enumerate(args.coeff_mults):
        for li, L_inject in enumerate(args.inject_layers):
            cell_name = f"add_L{L_inject}_c{cmult}x"
            rate_grid[ci, li] = cell_meta[cell_name]["refusal_rate_substr"]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    im = ax.imshow(rate_grid, aspect="auto", cmap="Reds", vmin=0, vmax=1, origin="lower")
    ax.set_xticks(range(n_layers))
    ax.set_xticklabels([f"L{L}" for L in args.inject_layers])
    ax.set_yticks(range(n_coeffs))
    ax.set_yticklabels([f"{c}×" for c in args.coeff_mults])
    ax.set_xlabel("injection layer")
    ax.set_ylabel("coefficient (× natural scale at L%d = %.2f)" % (args.extract_layer, natural_scale))
    ax.set_title(f"Phase 2 §3d — induced-refusal on harmless prompts (Qwen2.5-3B), N={args.n_subset}/cell")
    for ci in range(n_coeffs):
        for li in range(n_layers):
            ax.text(li, ci, f"{rate_grid[ci, li]:.2f}",
                    ha="center", va="center",
                    color="white" if rate_grid[ci, li] > 0.5 else "black",
                    fontsize=10)
    fig.colorbar(im, ax=ax, label="substring refusal rate")
    fig.tight_layout()
    fig_path = ARTIFACTS_FIGURES / "phase2_step3d_operating_band_sweep.png"
    fig.savefig(fig_path, dpi=140)
    plt.close(fig)
    log.info("figure -> %s", fig_path)

    record = {
        **partial,
        "judge_summary": {k: {kk: vv for kk, vv in v.items() if kk != "labels"}
                          for k, v in judge_summary.items()} if judge_summary else None,
        "judge_full": judge_summary if judge_summary else None,
        "specificity_control": spec_control,
        "best_cell_name": max(cell_meta, key=lambda k: cell_meta[k]["lift_vs_baseline"]),
        "best_cell_meta": max((m for m in cell_meta.values()), key=lambda m: m["lift_vs_baseline"]),
        "figure": str(fig_path.relative_to(fig_path.parent.parent.parent)),
    }
    write_json(run_dir / "result.json", record)
    md_path = RESULTS / "phase2_step3d_operating_band_sweep.md"
    md_path.write_text(_render_summary(record))
    log.info("result -> %s", run_dir / "result.json")
    log.info("summary -> %s", md_path)

    # === Console headline ===
    print("\n=== Phase 2 Step 3d — operating-band addition sweep ===")
    print(f"Model: {bundle.name}  |  d̂ from L{args.extract_layer}  |  natural scale = {natural_scale:.3f}")
    print(f"Baseline (no hook): {base_rate:.3f}")
    print("\nSubstring refusal-rate heatmap (rows = coeff multiplier, cols = inject layer):")
    print("          " + "  ".join(f"L{L:>3d}" for L in args.inject_layers))
    for ci, cmult in enumerate(args.coeff_mults):
        row = "  ".join(f"{rate_grid[ci, li]:.2f}" for li in range(n_layers))
        print(f"  {cmult}×    {row}")
    print(f"\nBest cell: {record['best_cell_name']}  → lift {record['best_cell_meta']['lift_vs_baseline']:+.3f}")
    if judge_summary:
        print(f"Judge confirmed {len(cells_to_judge)} lifted cells.")
    if spec_control:
        print(f"Random-vector specificity at best cell: substring lift = {spec_control['lift_vs_baseline_substr']:+.3f}")
        print(f"  (compared to d̂ lift {record['best_cell_meta']['lift_vs_baseline']:+.3f} — "
              f"d̂ should be MUCH higher for specificity)")
    return 0


def _render_summary(rec):
    md = [
        f"# Phase 2 Step 3d — operating-band addition sweep ({rec['model']})",
        "",
        f"- d̂ extracted from L{rec['extract_layer']} (held constant)",
        f"- Natural scale at L{rec['extract_layer']}: {rec['natural_scale']:.3f}",
        f"- Inject layers: {rec['inject_layers']}",
        f"- Coefficients (× natural scale): {rec['coeff_mults']}",
        f"- Target: {rec['n_subset']} CodeAlpaca-harmless prompts (subset of step 3b's 30, seed {rec['split_seed']})",
        f"- Baseline (no hook) substring refusal: **{rec['baseline_substr_refusal_rate']:.3f}**",
        "",
        "## Substring-scorer refusal-rate heatmap",
        "",
        "| coeff × | " + " | ".join(f"L{L}" for L in rec["inject_layers"]) + " |",
        "|---|" + "---:|" * len(rec["inject_layers"]),
    ]
    for cmult in rec["coeff_mults"]:
        cells = [rec["cell_meta"][f"add_L{L}_c{cmult}x"]["refusal_rate_substr"]
                 for L in rec["inject_layers"]]
        md.append(f"| {cmult}× | " + " | ".join(f"{c:.2f}" for c in cells) + " |")
    md.append("")
    md.append(f"**Best cell:** `{rec['best_cell_name']}` "
              f"(lift {rec['best_cell_meta']['lift_vs_baseline']:+.3f} vs baseline)")
    md.append("")
    if rec.get("judge_summary"):
        md.append("## LLM judge (Haiku 4.5) on lifted cells")
        md.append("")
        md.append("| Cell | Judge refusal | Lift vs baseline |")
        md.append("|---|---:|---:|")
        for name, r in rec["judge_summary"].items():
            if name == "baseline": continue
            md.append(f"| {name} | {r['refusal_rate']:.3f} | "
                      f"{r.get('judge_lift_vs_baseline', 0):+.3f} |")
        md.append("")
    if rec.get("specificity_control"):
        sc = rec["specificity_control"]
        md.append("## Specificity control")
        md.append("")
        md.append(f"At the best cell ({sc['cell']}, L{sc['inject_layer']}, coeff {sc['coeff']:.2f}), "
                  f"injecting a random unit vector instead of d̂:")
        md.append(f"- Substring refusal rate: {sc['refusal_rate_substr']:.3f}")
        md.append(f"- Lift vs baseline: {sc['lift_vs_baseline_substr']:+.3f}")
        md.append("")
        md.append("If d̂ lift >> random lift, the induced refusal is direction-specific (not norm-perturbation).")
        md.append("")
    md.append(f"Heatmap figure: `{rec['figure']}`")
    md.append("")
    md.append("Per-prompt completions in `artifacts/runs/phase2_step3d/<timestamp>/result.json`.")
    md.append("Pre-registration: `results/phase2_step3d_preregistration.md`.")
    return "\n".join(md) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
