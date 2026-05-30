"""
Phase 1 — Step 2: layer sweep with leave-one-out cross-validation.

Question: at which layer is refusal best represented in the residual stream?

Method: at each layer, leave-one-out CV. For each held-out prompt, compute
diff-of-means d_hat from the OTHER 29 prompts; record the held-out prompt's
projection onto that d_hat. Aggregate projections across folds → AUC.

This is the small-N fix for the first pass of this script, which evaluated
in-sample (computed d_hat from prompts X and then scored X). With n=30 and
d=2304, in-sample AUC is heavily inflated — the shuffled-labels control
caught that: shuffled AUC was ~0.87 at the in-sample peak (should be ~0.5).

Two control curves under the SAME LOO-CV protocol:
- Random direction (fixed seed, data-independent): identical between in-
  sample and held-out evaluation, kept as a reference.
- Shuffled labels: each fold trains d_hat on shuffled labels, evaluates on
  the held-out prompt's true label. Now must sit at ~0.5 across all layers.
  If it doesn't, something else is leaking.

Peak layer is the working hypothesis L for Step 3 (steering); NOT a finding
yet — separation is correlational. Step 3 tests causality by ablation.

Run:
    cd ~/safe_ai/mech-security
    python -m experiments.phase1_step2_layer_sweep
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

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
from mech_security.activations import cache_resid_all_layers
from mech_security.directions import diff_of_means, project, random_unit_vector, unit

log = get_logger("phase1_step2")


def _auc(scores_h: torch.Tensor, scores_l: torch.Tensor) -> float:
    s = torch.cat([scores_h, scores_l]).numpy()
    y = [1] * len(scores_h) + [0] * len(scores_l)
    return float(roc_auc_score(y, s))


def _check_finite(name: str, t: torch.Tensor) -> None:
    if torch.isnan(t).any() or torch.isinf(t).any():
        raise RuntimeError(f"non-finite values in {name}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Layer sweep with LOO-CV.")
    ap.add_argument(
        "--data", type=str, default="data/contrastive.jsonl",
        help="Path to contrastive jsonl (default: data/contrastive.jsonl)."
    )
    ap.add_argument(
        "--tag", type=str, default=None,
        help="Optional run tag appended to summary/figure filenames."
    )
    args = ap.parse_args()

    run_dir = new_run_dir("phase1_step2")
    log.info("run_dir: %s | data=%s", run_dir, args.data)

    bundle = get_model()
    log.info("model: %s | n_layers=%d d_model=%d device=%s",
             bundle.name, bundle.n_layers, bundle.d_model, bundle.device)

    pairs_path = Path(args.data)
    if not pairs_path.is_absolute():
        pairs_path = Path(__file__).resolve().parent.parent / pairs_path
    harmful, harmless = load_jsonl_pairs(pairs_path)
    log.info("pairs: %d harmful, %d harmless from %s",
             len(harmful), len(harmless), pairs_path.name)

    # One forward pass per prompt, all layers cached. Disk-cached on a single
    # key so re-running this script is instant.
    extra_all = (
        f"{bundle.name}|dtype={bundle.model.cfg.dtype}|all_layers|resid_post|last_token"
    )
    key_h = content_hash(harmful, extra=extra_all + "|harmful")
    key_l = content_hash(harmless, extra=extra_all + "|harmless")
    log.info("caching residuals across all %d layers...", bundle.n_layers)
    harmful_acts = cached_activations(
        key_h, lambda: cache_resid_all_layers(bundle, harmful, show_progress=False)
    )
    harmless_acts = cached_activations(
        key_l, lambda: cache_resid_all_layers(bundle, harmless, show_progress=False)
    )
    _check_finite("harmful_acts", harmful_acts)
    _check_finite("harmless_acts", harmless_acts)
    log.info("shapes: harmful %s, harmless %s",
             tuple(harmful_acts.shape), tuple(harmless_acts.shape))

    n_layers = bundle.n_layers
    n_h = len(harmful)
    n_l = len(harmless)
    n_total = n_h + n_l

    # Stack into a single [n_total, n_layers, d_model] tensor with a parallel
    # labels vector so LOO-CV loops are uniform across real and shuffled runs.
    all_acts = torch.cat([harmful_acts, harmless_acts], dim=0)  # [n_total, L, d]
    labels_real = np.array([1] * n_h + [0] * n_l, dtype=int)

    # Shuffled-labels control: permute labels under a fixed seed.
    rng = np.random.default_rng(0)
    labels_shuf = labels_real.copy()
    rng.shuffle(labels_shuf)

    # Random direction (data-independent; LOO-CV doesn't change it).
    rand_dir = random_unit_vector(bundle.d_model, seed=0, device="cpu")

    def loo_auc(layer: int, labels: np.ndarray) -> float:
        """Leave-one-out AUC at a given layer for a given labelling.

        For each prompt i: compute diff-of-means direction from the other
        n_total - 1 prompts under `labels`, project prompt i's activation,
        record the scalar. After all folds, compute AUC of (scalars) vs
        (true `labels`).

        With n_total ≈ 30 this loop is cheap.
        """
        acts_L = all_acts[:, layer, :]  # [n_total, d]
        scores = np.zeros(n_total, dtype=float)
        for i in range(n_total):
            mask = np.ones(n_total, dtype=bool)
            mask[i] = False
            train_acts = acts_L[mask]
            train_labels = labels[mask]
            h_train = train_acts[train_labels == 1]
            l_train = train_acts[train_labels == 0]
            if len(h_train) == 0 or len(l_train) == 0:
                # Degenerate fold (can happen if labels are unbalanced after
                # holding one out under shuffled labels). Skip with NaN.
                scores[i] = float("nan")
                continue
            d_hat = unit(diff_of_means(h_train, l_train))
            scores[i] = float(acts_L[i] @ d_hat)

        # Drop NaN folds for AUC; mask labels accordingly.
        valid = ~np.isnan(scores)
        if valid.sum() < 4:
            return float("nan")
        return float(roc_auc_score(labels[valid], scores[valid]))

    auc_real = np.zeros(n_layers)
    auc_rand = np.zeros(n_layers)
    auc_shuf = np.zeros(n_layers)

    log.info("running LOO-CV across %d layers (%d folds each)...", n_layers, n_total)
    for L in range(n_layers):
        auc_real[L] = loo_auc(L, labels_real)
        auc_shuf[L] = loo_auc(L, labels_shuf)

        # Random direction is data-independent — no CV needed; this matches
        # the prior in-sample run for direct comparison.
        h = all_acts[labels_real == 1, L, :]
        l = all_acts[labels_real == 0, L, :]
        auc_rand[L] = _auc(project(h, rand_dir), project(l, rand_dir))

    peak_layer = int(np.argmax(auc_real))
    peak_auc = float(auc_real[peak_layer])
    log.info("peak: L%d, LOO-AUC=%.3f", peak_layer, peak_auc)
    log.info("at peak: random=%.3f, shuffled-LOO=%.3f", auc_rand[peak_layer], auc_shuf[peak_layer])
    log.info("shuffled-LOO mean across layers: %.3f (should be ≈ 0.5)", float(auc_shuf.mean()))

    # Figure
    fig_name = "phase1_step2_layer_sweep.png" if not args.tag else f"phase1_step2_layer_sweep_{args.tag}.png"
    fig_path = ARTIFACTS_FIGURES / fig_name
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    layers = np.arange(n_layers)
    ax.plot(layers, auc_real, "o-", label="real direction (LOO-CV)", linewidth=2)
    ax.plot(layers, auc_rand, "s--", label="random direction (in-sample)", alpha=0.6)
    ax.plot(layers, auc_shuf, "^--", label="shuffled labels (LOO-CV)", alpha=0.6)
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=1, label="chance")
    ax.axvline(peak_layer, color="red", linestyle=":", linewidth=1, alpha=0.5)
    ax.annotate(f"peak L{peak_layer}\nAUC={peak_auc:.3f}",
                xy=(peak_layer, peak_auc), xytext=(peak_layer + 1, peak_auc - 0.1),
                fontsize=9, color="red")
    ax.set_xlabel("layer")
    ax.set_ylabel("AUC of harmful-vs-harmless projection")
    ax.set_title("Phase 1 Step 2 — refusal-direction separation vs layer (LOO-CV)")
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=140)
    plt.close(fig)
    log.info("figure -> %s", fig_path)

    # Persist
    run_record = {
        "step": "phase1_step2",
        "model": bundle.name,
        "device": bundle.device,
        "n_layers": n_layers,
        "n_harmful": n_h,
        "n_harmless": n_l,
        "peak_layer": peak_layer,
        "peak_auc": peak_auc,
        "auc_real": auc_real.tolist(),
        "auc_random_direction": auc_rand.tolist(),
        "auc_shuffled_labels": auc_shuf.tolist(),
        "figure": str(fig_path.relative_to(fig_path.parent.parent.parent)),
        "controls_pass": {
            "real_beats_random_at_peak": bool(auc_real[peak_layer] - auc_rand[peak_layer] > 0.2),
            "real_beats_shuffled_at_peak": bool(auc_real[peak_layer] - auc_shuf[peak_layer] > 0.2),
            "shuffled_near_chance_overall": bool(abs(auc_shuf.mean() - 0.5) < 0.15),
            "random_near_chance_overall": bool(abs(auc_rand.mean() - 0.5) < 0.15),
        },
    }
    write_json(run_dir / "result.json", run_record)
    log.info("run record -> %s", run_dir / "result.json")

    # Summary
    summary_name = "phase1_step2.md" if not args.tag else f"phase1_step2_{args.tag}.md"
    summary_path = RESULTS / summary_name
    summary_path.write_text(_render_summary(run_record))
    log.info("summary -> %s", summary_path)

    print(
        f"\nphase1_step2 | peak L{peak_layer} AUC={peak_auc:.3f} | "
        f"random={auc_rand[peak_layer]:.3f} | shuffled={auc_shuf[peak_layer]:.3f}"
    )
    return 0


def _render_summary(rec: dict) -> str:
    real = rec["auc_real"]
    rand = rec["auc_random_direction"]
    shuf = rec["auc_shuffled_labels"]
    cp = rec["controls_pass"]

    # Find the contiguous high-AUC band: layers within 0.02 of the peak.
    peak = rec["peak_auc"]
    band = [i for i, a in enumerate(real) if a >= peak - 0.02]
    band_str = f"L{min(band)}–L{max(band)}" if len(band) > 1 else f"L{band[0]}"

    lines = [
        "# Phase 1 Step 2 — layer sweep",
        "",
        f"- Model: `{rec['model']}` on `{rec['device']}` | n_layers={rec['n_layers']}",
        f"- n harmful = {rec['n_harmful']}, n harmless = {rec['n_harmless']}",
        "- Data: `data/phase0-pairs.jsonl` (the Phase-0 hand-written set; replace per Step 1 (H) before publishing)",
        "",
        "## Headline",
        "",
        f"- **Peak layer: L{rec['peak_layer']}** (AUC = {peak:.3f})",
        f"- Plateau (within 0.02 of peak): **{band_str}**",
        f"- AUC band layers: {band}",
        "",
        "## Controls at the peak layer",
        "",
        f"- Real direction:     **{real[rec['peak_layer']]:.3f}**",
        f"- Random direction:   {rand[rec['peak_layer']]:.3f}",
        f"- Shuffled labels:    {shuf[rec['peak_layer']]:.3f}",
        "",
        f"- Real beats random by ≥ 0.20 at peak? **{cp['real_beats_random_at_peak']}**",
        f"- Real beats shuffled by ≥ 0.20 at peak? **{cp['real_beats_shuffled_at_peak']}**",
        f"- Shuffled-labels curve near chance overall (|mean − 0.5| < 0.15)? **{cp['shuffled_near_chance_overall']}** (actual mean = {sum(shuf)/len(shuf):.3f})",
        f"- Random-direction curve near chance overall? **{cp['random_near_chance_overall']}** (actual mean = {sum(rand)/len(rand):.3f})",
        "",
        f"Figure: `{rec['figure']}`",
        "",
        "## How to read the figure",
        "",
        "- The blue line should climb above 0.5 (chance) somewhere mid-network and plateau or peak. That's where refusal information is concentrated in the residual stream.",
        "- The orange (random-direction) and green (shuffled-labels) lines should hover near 0.5 at every layer. If either tracks the real curve, the result is not trustworthy.",
        "- The peak is **a working hypothesis layer**, not a finding. Separation is correlational. Step 3 (steering) tests causality: ablate at the peak layer, does refusal actually drop?",
        "",
        "## What to do next",
        "",
        f"- Record L = {rec['peak_layer']} in `tasks.md` as the working hypothesis layer.",
        f"- Run Step 3 — steering: ablate `d_hat` at L{rec['peak_layer']} on the harmful set; measure refusal-rate drop. Add `d_hat` at the same layer on the harmless set; measure over-refusal. The four mandatory controls (random direction, coherence, generalization split, both-directions-agree) gate that step.",
        "- After Step 1 (H) — when the contrastive set is replaced with AdvBench + Alpaca — re-run this script. The peak layer may move; that's information, not failure.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
