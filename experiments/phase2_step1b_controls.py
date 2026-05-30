"""
Phase 2 Step 1b — characterize the controls.

Reuse the cached residual-stream activations from phase2_step1 and run the
two LOO-CV controls under MULTIPLE seeds, so the single-seed flags from
phase2_step1 (random AUC=0.835, shuffled-LOO=0.416) can be put in proper
distributions.

Two questions this answers:
  1. What's the random-direction AUC distribution at L14 across 5 random
     unit-vector seeds? If the mean is ≈ 0.835 with small SD, the high
     floor is structural (vocabulary confound, geometric noise). If it
     varies wildly, the single-seed reading was unrepresentative.
  2. What's the shuffled-labels LOO-CV AUC distribution across 5 shuffle
     seeds? If mean ≈ 0.5 with SD ≈ 0.03, the single-seed 0.416 was just
     a sample. If mean stays ~0.42, there's a structural under-chance
     bias in the LOO-CV protocol on this dataset.

Run:
    cd ~/safe_ai/mech-security
    python -m experiments.phase2_step1b_controls
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from experiments._runner import (
    cached_activations,
    content_hash,
    get_logger,
    get_model,
    load_jsonl_pairs,
    new_run_dir,
    write_json,
)
from src.activations import cache_resid_all_layers
from src.directions import diff_of_means, project, random_unit_vector, unit
from src.model import format_prompt_for_bundle

log = get_logger("phase2_step1b")


def _auc_from_scores(scores_h, scores_l):
    s = np.concatenate([scores_h, scores_l])
    y = np.concatenate([np.ones(len(scores_h)), np.zeros(len(scores_l))])
    return float(roc_auc_score(y, s))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--data", type=str, default="data/code_contrastive.jsonl")
    ap.add_argument("--peak-layer", type=int, default=14,
                    help="Layer at which to characterize controls (Phase 2 step 1 found L14).")
    ap.add_argument("--n-shuffle-seeds", type=int, default=5)
    ap.add_argument("--n-random-seeds", type=int, default=5)
    args = ap.parse_args()

    run_dir = new_run_dir("phase2_step1b")
    log.info("run_dir: %s | model=%s peak=L%d", run_dir, args.model, args.peak_layer)

    bundle = get_model(args.model)
    pairs_path = Path(args.data)
    if not pairs_path.is_absolute():
        pairs_path = Path(__file__).resolve().parent.parent / pairs_path
    harmful, harmless = load_jsonl_pairs(pairs_path)

    fmt = lambda msg: format_prompt_for_bundle(bundle, msg)
    extra_all = (
        f"{bundle.name}|dtype={bundle.model.cfg.dtype}|all_layers|resid_post|last_token|phase2"
    )
    key_h = content_hash(harmful, extra=extra_all + "|harmful")
    key_l = content_hash(harmless, extra=extra_all + "|harmless")
    log.info("loading cached residuals (no recompute expected)...")
    harmful_acts = cached_activations(
        key_h, lambda: cache_resid_all_layers(bundle, harmful, show_progress=False, format_fn=fmt),
    )
    harmless_acts = cached_activations(
        key_l, lambda: cache_resid_all_layers(bundle, harmless, show_progress=False, format_fn=fmt),
    )

    L = args.peak_layer
    h_L = harmful_acts[:, L, :]  # [150, d]
    l_L = harmless_acts[:, L, :]  # [150, d]
    log.info("activations at L%d: harmful %s, harmless %s",
             L, tuple(h_L.shape), tuple(l_L.shape))

    n_h, n_l = len(harmful), len(harmless)
    n_total = n_h + n_l
    all_acts_L = torch.cat([h_L, l_L], dim=0)
    labels_real = np.array([1] * n_h + [0] * n_l, dtype=int)

    # === Real direction LOO-CV at L (re-derived to confirm) ===
    def loo_auc(acts_L, labels):
        scores = np.zeros(n_total, dtype=float)
        for i in range(n_total):
            mask = np.ones(n_total, dtype=bool)
            mask[i] = False
            train_acts = acts_L[mask]
            train_labels = labels[mask]
            h_train = train_acts[train_labels == 1]
            l_train = train_acts[train_labels == 0]
            if len(h_train) == 0 or len(l_train) == 0:
                scores[i] = float("nan")
                continue
            d_hat = unit(diff_of_means(h_train, l_train))
            scores[i] = float(acts_L[i] @ d_hat)
        valid = ~np.isnan(scores)
        return float(roc_auc_score(labels[valid], scores[valid]))

    auc_real = loo_auc(all_acts_L, labels_real)
    log.info("real direction LOO-AUC @ L%d: %.4f (confirms step 1)", L, auc_real)

    # === Random direction AUC distribution (5 seeds) ===
    random_aucs = []
    for seed in range(args.n_random_seeds):
        r = random_unit_vector(bundle.d_model, seed=seed, device="cpu")
        a = _auc_from_scores(project(h_L, r).numpy(), project(l_L, r).numpy())
        random_aucs.append(a)
        log.info("  random_seed=%d  AUC=%.4f", seed, a)
    rand_mean = float(np.mean(random_aucs))
    rand_std = float(np.std(random_aucs, ddof=1))
    log.info("random direction AUC: mean=%.4f, std=%.4f, range=[%.4f, %.4f]",
             rand_mean, rand_std, min(random_aucs), max(random_aucs))

    # === Shuffled-labels LOO-CV AUC distribution (5 seeds) ===
    shuffled_aucs = []
    for seed in range(args.n_shuffle_seeds):
        rng = np.random.default_rng(seed)
        labels_shuf = labels_real.copy()
        rng.shuffle(labels_shuf)
        a = loo_auc(all_acts_L, labels_shuf)
        shuffled_aucs.append(a)
        log.info("  shuffle_seed=%d  AUC=%.4f", seed, a)
    shuf_mean = float(np.mean(shuffled_aucs))
    shuf_std = float(np.std(shuffled_aucs, ddof=1))
    log.info("shuffled-labels LOO-AUC: mean=%.4f, std=%.4f, range=[%.4f, %.4f]",
             shuf_mean, shuf_std, min(shuffled_aucs), max(shuffled_aucs))

    # === Z-scores: where does the real direction sit? ===
    z_vs_random = (auc_real - rand_mean) / max(rand_std, 1e-6)
    z_vs_shuffled = (auc_real - shuf_mean) / max(shuf_std, 1e-6)
    log.info("real-direction z vs random null:    %+.2f σ", z_vs_random)
    log.info("real-direction z vs shuffled null:  %+.2f σ", z_vs_shuffled)

    record = {
        "step": "phase2_step1b_controls",
        "model": bundle.name,
        "data": str(pairs_path),
        "peak_layer": L,
        "n_random_seeds": args.n_random_seeds,
        "n_shuffle_seeds": args.n_shuffle_seeds,
        "real_direction_auc": auc_real,
        "random_aucs": random_aucs,
        "random_mean": rand_mean,
        "random_std": rand_std,
        "shuffled_aucs": shuffled_aucs,
        "shuffled_mean": shuf_mean,
        "shuffled_std": shuf_std,
        "z_real_vs_random": z_vs_random,
        "z_real_vs_shuffled": z_vs_shuffled,
        "controls_pass": {
            "shuffled_near_chance_within_2sigma":
                bool(abs(shuf_mean - 0.5) < 2 * max(shuf_std, 0.02)),
            "real_well_above_random":
                bool(z_vs_random > 3.0),
        },
    }
    write_json(run_dir / "result.json", record)
    log.info("result -> %s", run_dir / "result.json")

    print(f"\n=== phase2_step1b: control characterization at L{L} ===")
    print(f"  real-direction LOO-AUC:       {auc_real:.4f}")
    print(f"  random direction (n={args.n_random_seeds}):     "
          f"mean={rand_mean:.4f}, std={rand_std:.4f}, range=[{min(random_aucs):.4f}, {max(random_aucs):.4f}]")
    print(f"  shuffled labels (n={args.n_shuffle_seeds}):       "
          f"mean={shuf_mean:.4f}, std={shuf_std:.4f}, range=[{min(shuffled_aucs):.4f}, {max(shuffled_aucs):.4f}]")
    print(f"  z(real vs random null):  {z_vs_random:+.2f} σ")
    print(f"  z(real vs shuffled null): {z_vs_shuffled:+.2f} σ")
    print(f"  controls_pass: {record['controls_pass']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
