"""
Phase 1 §3 supplementary — characterize random-direction AUC as a
DISTRIBUTION over 5 seeds at L13 on both AdvBench and HarmBench.

Why this exists: the published §3 cross-distribution AUC table reports
"random unit vector (floor)" as a single number (0.611 on AdvBench,
0.533 on HarmBench). Phase 2 work on Qwen revealed that the random-
direction AUC distribution is bimodal and nearly spans [0, 1] when the
contrastive set has a vocabulary/topic mean-shift. A re-run with 5 seeds
on Gemma's AdvBench/HarmBench setup turns the single sample into a
characterized distribution and closes a hole the Qwen step opened in the
Gemma writeup.

Reads cached activations only — no model forward passes. ~10 seconds.

Run:
    cd ~/safe_ai/mech-security
    python -m experiments.phase1_random_baseline_5seed
"""

from __future__ import annotations

import sys
from pathlib import Path


import numpy as np
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
from mech_security.activations import cache_resid
from mech_security.directions import project, random_unit_vector

log = get_logger("phase1_random_baseline_5seed")


def _auc(scores_h, scores_l):
    s = np.concatenate([scores_h.numpy(), scores_l.numpy()])
    y = np.concatenate([np.ones(len(scores_h)), np.zeros(len(scores_l))])
    return float(roc_auc_score(y, s))


def main() -> int:
    run_dir = new_run_dir("phase1_random_baseline_5seed")
    log.info("run_dir: %s", run_dir)
    L = 13  # peak layer from Phase 1 §3
    n_seeds = 5

    bundle = get_model()  # Gemma-2-2b-it, fp16 on MPS (Phase 1 default)
    log.info("model: %s | d_model=%d", bundle.name, bundle.d_model)

    # === AdvBench (Phase 1 frozen contrastive set) ===
    advbench_harmful, alpaca_harmless = load_jsonl_pairs(
        Path(__file__).resolve().parent.parent / "data/contrastive.jsonl"
    )
    extra_a = (f"{bundle.name}|dtype={bundle.model.cfg.dtype}|L{L}|"
               f"resid_post|last_token|advbench_full")
    key_ah = content_hash(advbench_harmful, extra=extra_a + "|harmful")
    key_al = content_hash(alpaca_harmless, extra=extra_a + "|harmless")
    log.info("loading AdvBench cached acts (should hit cache)...")
    H_adv = cached_activations(key_ah, lambda: cache_resid(bundle, advbench_harmful, layer=L, show_progress=False))
    L_alpaca = cached_activations(key_al, lambda: cache_resid(bundle, alpaca_harmless, layer=L, show_progress=False))
    log.info("AdvBench acts: harmful %s, harmless %s", tuple(H_adv.shape), tuple(L_alpaca.shape))

    # === HarmBench (the published cross-distribution comparator) ===
    # Pulled from data/harmbench_behaviors_text_all.csv at the same N=200 used
    # in phase1_cross_extraction.py.
    from experiments.phase1_cross_extraction import load_harmbench
    hb_rows = load_harmbench(n_prompts=200, seed=0)
    hb_harmful = [r["prompt"] for r in hb_rows]
    extra_hb = (f"{bundle.name}|dtype={bundle.model.cfg.dtype}|L{L}|"
                f"resid_post|last_token|harmbench_n200")
    key_hb_h = content_hash(hb_harmful, extra=extra_hb + "|harmful")
    log.info("loading HarmBench cached acts (should hit cache)...")
    H_hb = cached_activations(key_hb_h, lambda: cache_resid(bundle, hb_harmful, layer=L, show_progress=False))
    log.info("HarmBench acts: harmful %s", tuple(H_hb.shape))

    # === 5-seed random direction AUC distribution ===
    advbench_aucs = []
    harmbench_aucs = []
    for seed in range(n_seeds):
        r = random_unit_vector(bundle.d_model, seed=seed, device="cpu")
        a_adv = _auc(project(H_adv, r), project(L_alpaca, r))
        a_hb = _auc(project(H_hb, r), project(L_alpaca, r))
        advbench_aucs.append(a_adv)
        harmbench_aucs.append(a_hb)
        log.info("  seed=%d  AdvBench AUC=%.4f  HarmBench AUC=%.4f", seed, a_adv, a_hb)

    def stats(xs):
        return {
            "mean": float(np.mean(xs)),
            "std": float(np.std(xs, ddof=1)),
            "min": float(min(xs)),
            "max": float(max(xs)),
            "values": [float(x) for x in xs],
        }

    s_adv = stats(advbench_aucs)
    s_hb = stats(harmbench_aucs)

    log.info("=== Random unit vector AUC distribution at L%d (n=%d seeds) ===", L, n_seeds)
    log.info("  AdvBench:  mean=%.4f std=%.4f  range=[%.4f, %.4f]",
             s_adv["mean"], s_adv["std"], s_adv["min"], s_adv["max"])
    log.info("  HarmBench: mean=%.4f std=%.4f  range=[%.4f, %.4f]",
             s_hb["mean"], s_hb["std"], s_hb["min"], s_hb["max"])

    record = {
        "step": "phase1_random_baseline_5seed",
        "model": bundle.name,
        "layer": L,
        "n_seeds": n_seeds,
        "n_advbench_harmful": int(H_adv.shape[0]),
        "n_harmbench_harmful": int(H_hb.shape[0]),
        "n_alpaca_harmless": int(L_alpaca.shape[0]),
        "advbench": s_adv,
        "harmbench": s_hb,
    }
    write_json(run_dir / "result.json", record)
    log.info("result -> %s", run_dir / "result.json")

    print(f"\n=== Random unit vector AUC (5 seeds) at L{L} on {bundle.name} ===")
    print(f"  AdvBench:  mean={s_adv['mean']:.4f}  std={s_adv['std']:.4f}  range=[{s_adv['min']:.4f}, {s_adv['max']:.4f}]")
    print(f"  HarmBench: mean={s_hb['mean']:.4f}  std={s_hb['std']:.4f}  range=[{s_hb['min']:.4f}, {s_hb['max']:.4f}]")
    print("  (previously §3 reported single-seed: AdvBench 0.611, HarmBench 0.533)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
