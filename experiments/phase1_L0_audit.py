"""Phase 1 §3 supplementary — check what AUC looks like at L0 on Gemma.

Companion to the Qwen finding that AUC = 0.9996 at L0 (embedding output
on Qwen2.5-3B + code_contrastive). Does Gemma + AdvBench/Alpaca also show
this lexical-by-construction separation at L0, or is it Qwen-specific?

Uses the cached all-layers tensors from phase1_step2_layer_sweep — no
model load needed. Pure CPU math.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "/Users/anshulsinghle/safe_ai/mech-security")

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from experiments._runner import cached_activations, content_hash, get_model, load_jsonl_pairs
from src.activations import cache_resid_all_layers
from src.directions import diff_of_means, project, random_unit_vector, unit


def auc(scores_h, scores_l):
    s = np.concatenate([scores_h.numpy(), scores_l.numpy()])
    y = np.concatenate([np.ones(len(scores_h)), np.zeros(len(scores_l))])
    return float(roc_auc_score(y, s))


def main():
    print("=" * 70)
    print("Phase 1 — does Gemma's AdvBench+Alpaca contrastive set also show")
    print("AUC ≈ 1 at L0 (embedding output) the way Qwen does?")
    print("=" * 70)

    # Need Gemma loaded only to access cfg/dim — model itself unused
    bundle = get_model("gemma-2-2b-it")

    advbench_h, alpaca_l = load_jsonl_pairs(
        Path("/Users/anshulsinghle/safe_ai/mech-security/data/contrastive.jsonl")
    )
    extra_all = (f"{bundle.name}|dtype={bundle.model.cfg.dtype}|"
                 f"all_layers|resid_post|last_token")
    key_h = content_hash(advbench_h, extra=extra_all + "|harmful")
    key_l = content_hash(alpaca_l, extra=extra_all + "|harmless")
    print(f"\nLoading all-layers cache (key_h={key_h[:12]}...) ...")
    H_all = cached_activations(
        key_h, lambda: cache_resid_all_layers(bundle, advbench_h, show_progress=False)
    )  # [n_h, n_layers, d_model]
    L_all = cached_activations(
        key_l, lambda: cache_resid_all_layers(bundle, alpaca_l, show_progress=False)
    )
    print(f"  shapes: harmful {tuple(H_all.shape)}, harmless {tuple(L_all.shape)}")

    n_layers = H_all.shape[1]
    # Pick a few representative layers for the table
    layers_to_check = [0, 1, 2, 5, 8, 13, 20, 23, n_layers - 1]
    layers_to_check = [L for L in layers_to_check if L < n_layers]

    print(f"\nPer-layer real diff-of-means AUC + 5-seed random AUC distribution:")
    print(f"\n{'Layer':>6} | {'Real AUC':>10} | {'Random (5 seeds): mean ± std, range':>50}")
    print(f"-{'-' * 6}-+-{'-' * 10}-+-{'-' * 50}")

    for L in layers_to_check:
        H = H_all[:, L, :]
        Lh = L_all[:, L, :]
        d = unit(diff_of_means(H, Lh))
        real_auc = auc(project(H, d), project(Lh, d))
        rand_aucs = []
        for seed in range(5):
            r = random_unit_vector(bundle.d_model, seed=seed, device="cpu")
            rand_aucs.append(auc(project(H, r), project(Lh, r)))
        m, s = np.mean(rand_aucs), np.std(rand_aucs, ddof=1)
        print(f"{L:>6d} | {real_auc:>10.4f} | mean={m:.4f}, std={s:.4f}, range=[{min(rand_aucs):.4f}, {max(rand_aucs):.4f}]")

    print()
    print("=" * 70)
    print("Comparison with Qwen2.5-3B + code_contrastive (from phase2_step1 result):")
    print("  Qwen L0:  real AUC ≈ 0.9996  (linearly separable at embedding output)")
    print("  Qwen L14: real AUC = 1.0000  (peak)")
    print("  Qwen plateau spans EVERY layer 0–35 at AUC ≥ 0.994.")


if __name__ == "__main__":
    main()
