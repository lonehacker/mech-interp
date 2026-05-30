"""
Phase 1 — Cross-extraction matrix: is the refusal direction the same
whether extracted from AdvBench or from HarmBench, or distribution-specific?

Phase A (cheap, ~20 min): extract d_hat from HarmBench harmful prompts +
the same Alpaca harmless side as AdvBench; compute cos(d_hat_advbench,
d_hat_harmbench) and per-layer AUC profile. The cos similarity does most
of the heavy lifting:

  cos > 0.9   → essentially the same direction. Strong evidence the
                "refusal direction" is invariant to harmful-prompt source.
  cos 0.5-0.9 → related but distinct. Behavioral test needed to disambiguate.
  cos < 0.3   → distinct directions. Surprising. Would need behavioral test
                + careful interpretation.

Phase B (if Phase A warrants, separate run): ablate d_hat_harmbench on
both AdvBench held-out and HarmBench held-out, compare to d_hat_advbench
behavioral numbers. Closes the cross-extraction transfer matrix.

Method note (honest framing): HarmBench ships harmful-only; there's no
matched harmless side. This experiment uses the existing Alpaca harmless
set (the one used for AdvBench) as a *generic* harmless reference. We're
holding the harmless side constant and varying only the harmful source.
Length-matching between HarmBench and Alpaca isn't audited for this run;
some length-confound contamination of d_hat_harmbench is plausible. The
cos with d_hat_advbench tells us the magnitude of any such confound's
effect.

Run:
    cd ~/safe_ai/mech-security
    python -m experiments.phase1_cross_extraction
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


import numpy as np
from sklearn.metrics import roc_auc_score

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
from experiments.phase1_harmbench_eval import load_harmbench
from mech_security.activations import cache_resid, cache_resid_all_layers
from mech_security.directions import diff_of_means, project, unit

log = get_logger("phase1_cross_extract")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract-layer", type=int, default=13,
                    help="Layer at which to extract both d_hats. L13 is AdvBench's peak; "
                         "HarmBench's peak is computed and reported separately.")
    ap.add_argument("--n-harmbench", type=int, default=200,
                    help="Number of HarmBench harmful prompts to use (standard subset; "
                         "max 200).")
    ap.add_argument("--all-layers", action="store_true",
                    help="Also cache all-layers HarmBench activations to compute "
                         "per-layer AUC for the new d_hat_harmbench.")
    args = ap.parse_args()

    run_dir = new_run_dir("phase1_cross_extraction")
    log.info("run_dir: %s | extract_layer=L%d", run_dir, args.extract_layer)

    bundle = get_model()
    log.info("model: %s | device=%s", bundle.name, bundle.device)

    # === Load both harmful sources ===
    pairs_path = Path(__file__).resolve().parent.parent / "data/contrastive.jsonl"
    advbench_harmful, alpaca_harmless = load_jsonl_pairs(pairs_path)
    log.info("AdvBench harmful: %d  |  Alpaca harmless: %d (held constant across extractions)",
             len(advbench_harmful), len(alpaca_harmless))

    harmbench_rows = load_harmbench(n_prompts=args.n_harmbench, seed=0)
    harmbench_harmful = [r["prompt"] for r in harmbench_rows]
    log.info("HarmBench harmful: %d  | semantic categories: %s",
             len(harmbench_harmful), sorted(set(r["category"] for r in harmbench_rows)))

    # === Extract d_hat_advbench (cache hit) ===
    extra_a = (f"{bundle.name}|dtype={bundle.model.cfg.dtype}|L{args.extract_layer}|"
               f"resid_post|last_token|advbench_full")
    key_ah = content_hash(advbench_harmful, extra=extra_a + "|harmful")
    key_al = content_hash(alpaca_harmless, extra=extra_a + "|harmless")
    log.info("loading AdvBench d_hat activations at L%d (should hit cache) ...", args.extract_layer)
    H_adv = cached_activations(key_ah, lambda: cache_resid(bundle, advbench_harmful, layer=args.extract_layer, show_progress=False))
    L_alpaca = cached_activations(key_al, lambda: cache_resid(bundle, alpaca_harmless, layer=args.extract_layer, show_progress=False))
    d_hat_advbench = unit(diff_of_means(H_adv, L_alpaca))
    log.info("d_hat_advbench extracted | shape=%s", tuple(d_hat_advbench.shape))

    # === Extract d_hat_harmbench (new caching) ===
    extra_hb = (f"{bundle.name}|dtype={bundle.model.cfg.dtype}|L{args.extract_layer}|"
                f"resid_post|last_token|harmbench_n{args.n_harmbench}")
    key_hb_h = content_hash(harmbench_harmful, extra=extra_hb + "|harmful")
    # Reuse the same Alpaca harmless cache (the harmless side is held constant)
    log.info("caching HarmBench activations at L%d (n=%d, ~%d sec) ...",
             args.extract_layer, args.n_harmbench, args.n_harmbench * 2)
    H_hb = cached_activations(key_hb_h, lambda: cache_resid(bundle, harmbench_harmful, layer=args.extract_layer, show_progress=False))
    d_hat_harmbench = unit(diff_of_means(H_hb, L_alpaca))
    log.info("d_hat_harmbench extracted | shape=%s", tuple(d_hat_harmbench.shape))

    # === The core number: cosine similarity between the two directions ===
    cos = float((d_hat_advbench * d_hat_harmbench).sum())
    log.info("cos(d_hat_advbench, d_hat_harmbench) = %.4f", cos)

    # === Diagnostics: cluster-mean projections + AUC on both data sets ===
    # For each d_hat, project both AdvBench-H and HarmBench-H onto it, plus Alpaca-L.
    def proj_stats(name, d_hat):
        a_h = project(H_adv, d_hat).numpy()
        hb_h = project(H_hb, d_hat).numpy()
        a_l = project(L_alpaca, d_hat).numpy()
        # AUC vs Alpaca harmless on each harmful source
        auc_a = float(roc_auc_score([1]*len(a_h) + [0]*len(a_l),
                                     np.concatenate([a_h, a_l])))
        auc_hb = float(roc_auc_score([1]*len(hb_h) + [0]*len(a_l),
                                      np.concatenate([hb_h, a_l])))
        log.info("[%s] AdvBench h_mean=%.2f, HarmBench h_mean=%.2f, Alpaca l_mean=%.2f | "
                 "AUC(AdvBench): %.3f, AUC(HarmBench): %.3f",
                 name, a_h.mean(), hb_h.mean(), a_l.mean(), auc_a, auc_hb)
        return {
            "advbench_proj_mean": float(a_h.mean()),
            "advbench_proj_std": float(a_h.std()),
            "harmbench_proj_mean": float(hb_h.mean()),
            "harmbench_proj_std": float(hb_h.std()),
            "alpaca_proj_mean": float(a_l.mean()),
            "alpaca_proj_std": float(a_l.std()),
            "auc_advbench_vs_alpaca": auc_a,
            "auc_harmbench_vs_alpaca": auc_hb,
            "scale_advbench": float(a_h.mean() - a_l.mean()),
            "scale_harmbench": float(hb_h.mean() - a_l.mean()),
        }

    advbench_dhat_stats = proj_stats("d_hat_advbench", d_hat_advbench)
    harmbench_dhat_stats = proj_stats("d_hat_harmbench", d_hat_harmbench)

    # === Optional: per-layer AUC for HarmBench-derived d_hat ===
    per_layer_auc = None
    if args.all_layers:
        log.info("caching all-layers HarmBench activations + computing peak-layer AUC ...")
        extra_all = (f"{bundle.name}|dtype={bundle.model.cfg.dtype}|all_layers|"
                     f"resid_post|last_token|harmbench_n{args.n_harmbench}")
        key_hb_all = content_hash(harmbench_harmful, extra=extra_all + "|harmful")
        key_alpaca_all = content_hash(alpaca_harmless, extra=extra_all + "|harmless")
        H_hb_all = cached_activations(
            key_hb_all,
            lambda: cache_resid_all_layers(bundle, harmbench_harmful, show_progress=False)
        )
        L_alpaca_all = cached_activations(
            key_alpaca_all,
            lambda: cache_resid_all_layers(bundle, alpaca_harmless, show_progress=False)
        )
        # Per-layer: extract d_hat_harmbench at each layer, compute its AUC
        n_layers = H_hb_all.shape[1]
        per_layer_auc = []
        for ly in range(n_layers):
            d = unit(diff_of_means(H_hb_all[:, ly, :], L_alpaca_all[:, ly, :]))
            h_proj = project(H_hb_all[:, ly, :], d).numpy()
            l_proj = project(L_alpaca_all[:, ly, :], d).numpy()
            auc = float(roc_auc_score([1]*len(h_proj) + [0]*len(l_proj),
                                       np.concatenate([h_proj, l_proj])))
            per_layer_auc.append({"layer": ly, "auc": auc})
        peak_layer = max(per_layer_auc, key=lambda r: r["auc"])
        log.info("HarmBench d_hat peak layer: L%d (AUC=%.3f)",
                 peak_layer["layer"], peak_layer["auc"])

    # === Save ===
    record = {
        "step": "phase1_cross_extraction",
        "model": bundle.name,
        "extract_layer": args.extract_layer,
        "n_advbench": len(advbench_harmful),
        "n_harmbench": len(harmbench_harmful),
        "n_alpaca_harmless": len(alpaca_harmless),
        "cos_advbench_vs_harmbench": cos,
        "d_hat_advbench_stats": advbench_dhat_stats,
        "d_hat_harmbench_stats": harmbench_dhat_stats,
        "per_layer_auc_harmbench": per_layer_auc,
        "method_note": (
            "Harmless side held constant (Alpaca, the same set used in our "
            "AdvBench d_hat extraction). Only the harmful source varies "
            "(AdvBench vs HarmBench). Length-matching between HarmBench "
            "harmful and Alpaca harmless is not audited for this run."
        ),
    }
    write_json(run_dir / "result.json", record)
    log.info("run record -> %s", run_dir / "result.json")

    md_path = RESULTS / "phase1_cross_extraction.md"
    md_path.write_text(_render_summary(record))
    log.info("summary -> %s", md_path)

    print()
    print("=== phase1_cross_extraction ===")
    print(f"cos(d_hat_advbench, d_hat_harmbench) = {cos:.4f}")
    print()
    print("d_hat_advbench:")
    print(f"  AUC on AdvBench  vs Alpaca: {advbench_dhat_stats['auc_advbench_vs_alpaca']:.3f}")
    print(f"  AUC on HarmBench vs Alpaca: {advbench_dhat_stats['auc_harmbench_vs_alpaca']:.3f}")
    print("d_hat_harmbench:")
    print(f"  AUC on AdvBench  vs Alpaca: {harmbench_dhat_stats['auc_advbench_vs_alpaca']:.3f}")
    print(f"  AUC on HarmBench vs Alpaca: {harmbench_dhat_stats['auc_harmbench_vs_alpaca']:.3f}")
    print()
    if cos > 0.9:
        print(">>> Essentially the same direction. Refusal direction is invariant to harmful source.")
    elif cos > 0.5:
        print(">>> Related but distinct directions. Phase B behavioral test would disambiguate.")
    else:
        print(">>> DISTINCT directions. Phase B behavioral test needed.")
    return 0


def _render_summary(rec):
    cos = rec["cos_advbench_vs_harmbench"]
    a = rec["d_hat_advbench_stats"]; h = rec["d_hat_harmbench_stats"]
    md = [
        "# Phase 1 — Cross-extraction: is the refusal direction distribution-specific?",
        "",
        f"**Headline.** cos(d_hat extracted from AdvBench, d_hat extracted from HarmBench) "
        f"= **{cos:.4f}** at L{rec['extract_layer']}.",
        "",
        "Both directions extracted using diff-of-means, with the **same Alpaca harmless side** "
        "held constant. Only the harmful source varies (AdvBench vs HarmBench).",
        "",
        f"- Model: `{rec['model']}`",
        f"- Extract layer: L{rec['extract_layer']}",
        f"- AdvBench harmful prompts: {rec['n_advbench']}",
        f"- HarmBench harmful prompts: {rec['n_harmbench']} (standard subset)",
        f"- Alpaca harmless prompts: {rec['n_alpaca_harmless']} (held constant)",
        "",
        "## Cosine similarity interpretation",
        "",
    ]
    if cos > 0.9:
        md.append("**cos > 0.9 → essentially the same direction.** The refusal direction is "
                  "invariant to which harmful-prompt benchmark you extract it from. The "
                  "cross-extraction question collapses: there is one direction, not two.")
    elif cos > 0.5:
        md.append(f"**cos = {cos:.3f} → related but distinct directions.** The two harmful "
                  f"sources produce noticeably different diff-of-means directions. A behavioral "
                  f"cross-ablation test (Phase B) would disambiguate which (or both) is causal.")
    else:
        md.append(f"**cos = {cos:.3f} → DISTINCT directions.** The two harmful sources produce "
                  f"largely orthogonal diff-of-means directions. The 'refusal direction' is "
                  f"meaningfully distribution-specific. Behavioral cross-ablation test needed.")
    md.append("")
    md.append("## Per-direction AUC on each harmful source")
    md.append("")
    md.append("Each row shows how cleanly a d_hat separates a given harmful source from Alpaca harmless.")
    md.append("")
    md.append("| Direction | AUC vs AdvBench | AUC vs HarmBench |")
    md.append("|---|---:|---:|")
    md.append(f"| d_hat_advbench | {a['auc_advbench_vs_alpaca']:.3f} | {a['auc_harmbench_vs_alpaca']:.3f} |")
    md.append(f"| d_hat_harmbench | {h['auc_advbench_vs_alpaca']:.3f} | {h['auc_harmbench_vs_alpaca']:.3f} |")
    md.append("")
    md.append("## Natural-scale comparison")
    md.append("")
    md.append("Cluster-mean projection (harmful − harmless) along each d_hat:")
    md.append("")
    md.append("| Direction | scale on AdvBench | scale on HarmBench |")
    md.append("|---|---:|---:|")
    md.append(f"| d_hat_advbench | {a['scale_advbench']:.2f} | {a['scale_harmbench']:.2f} |")
    md.append(f"| d_hat_harmbench | {h['scale_advbench']:.2f} | {h['scale_harmbench']:.2f} |")
    md.append("")
    if rec["per_layer_auc_harmbench"]:
        md.append("## Per-layer AUC for d_hat_harmbench")
        md.append("")
        md.append("| Layer | AUC (HarmBench vs Alpaca, d_hat extracted at same layer) |")
        md.append("|---:|---:|")
        for r in rec["per_layer_auc_harmbench"]:
            md.append(f"| L{r['layer']} | {r['auc']:.3f} |")
        md.append("")
    md.append("## Method caveat")
    md.append("")
    md.append(rec["method_note"])
    return "\n".join(md) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
