"""
Phase 2 — Step 1: layer sweep on a new model + code-refusal contrastive set.

Mirrors phase1_step2_layer_sweep.py but parameterized for any TL-supported
instruct model. Defaults to Qwen2.5-3B-Instruct + data/code_contrastive.jsonl.

Pre-step (runs at the top): discover per-model first-token sets via the
`get_or_discover_token_sets` API (src.causal_metric). Caches the result to
artifacts/cache/token_sets/<model_sanitized>.json so later phase2 runners
read it from disk.

Question: at which layer is code-refusal best represented in the residual
stream of Qwen2.5-3B-Instruct?

Method: LOO-CV at each layer (same protocol as Phase 1 Step 2) plus the
two controls — random direction and shuffled labels.

Run:
    cd ~/safe_ai/mech-security
    python -m experiments.phase2_step1_layer_sweep
    # or with a different model/data:
    python -m experiments.phase2_step1_layer_sweep \
        --model Qwen/Qwen2.5-3B-Instruct --data data/code_contrastive.jsonl
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
from src.activations import cache_resid_all_layers
from src.causal_metric import get_or_discover_token_sets
from src.directions import diff_of_means, project, random_unit_vector, unit
from src.model import format_prompt_for_bundle

log = get_logger("phase2_step1")

ARTIFACTS_CACHE = Path(__file__).resolve().parent.parent / "artifacts" / "cache"
TOKEN_SET_CACHE_DIR = ARTIFACTS_CACHE / "token_sets"


def _auc(scores_h: torch.Tensor, scores_l: torch.Tensor) -> float:
    s = torch.cat([scores_h, scores_l]).numpy()
    y = [1] * len(scores_h) + [0] * len(scores_l)
    return float(roc_auc_score(y, s))


def _check_finite(name: str, t: torch.Tensor) -> None:
    if torch.isnan(t).any() or torch.isinf(t).any():
        raise RuntimeError(f"non-finite values in {name}")


def _sanitize_model_name(name: str) -> str:
    """Turn 'Qwen/Qwen2.5-3B-Instruct' into 'qwen-qwen2.5-3b-instruct'
    (filesystem-safe, lowercase, no slashes)."""
    return name.lower().replace("/", "-").replace(":", "-")


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 2 Step 1 layer sweep.")
    ap.add_argument(
        "--model", type=str, default="Qwen/Qwen2.5-3B-Instruct",
        help="HF model name (must be in TransformerLens OFFICIAL_MODEL_NAMES).",
    )
    ap.add_argument(
        "--data", type=str, default="data/code_contrastive.jsonl",
        help="Path to contrastive jsonl.",
    )
    ap.add_argument(
        "--tag", type=str, default=None,
        help="Optional run tag appended to summary/figure filenames.",
    )
    args = ap.parse_args()

    run_dir = new_run_dir("phase2_step1")
    log.info("run_dir: %s | model=%s | data=%s", run_dir, args.model, args.data)

    bundle = get_model(args.model)
    log.info("model: %s | n_layers=%d d_model=%d device=%s",
             bundle.name, bundle.n_layers, bundle.d_model, bundle.device)

    pairs_path = Path(args.data)
    if not pairs_path.is_absolute():
        pairs_path = Path(__file__).resolve().parent.parent / pairs_path
    harmful, harmless = load_jsonl_pairs(pairs_path)
    log.info("pairs: %d harmful, %d harmless from %s",
             len(harmful), len(harmless), pairs_path.name)

    # === Per-model chat template via the new generic helper ===
    fmt = lambda msg: format_prompt_for_bundle(bundle, msg)

    # === First-token set discovery (runs once per model, cached) ===
    sanitized = _sanitize_model_name(bundle.name)
    token_cache_path = TOKEN_SET_CACHE_DIR / f"{sanitized}.json"
    log.info("getting/discovering first-token sets (cache: %s)...", token_cache_path)
    templated_harmful = [fmt(p) for p in harmful]
    templated_harmless = [fmt(p) for p in harmless]
    token_discovery = get_or_discover_token_sets(
        bundle,
        cache_path=token_cache_path,
        templated_harmful=templated_harmful,
        templated_harmless=templated_harmless,
    )
    log.info("refusal_ids=%s coverage=%.2f", token_discovery.refusal_ids[:5], token_discovery.refusal_coverage)
    log.info("compliance_ids=%s coverage=%.2f", token_discovery.compliance_ids[:5], token_discovery.compliance_coverage)

    # === Cache residuals across all layers ===
    extra_all = (
        f"{bundle.name}|dtype={bundle.model.cfg.dtype}|all_layers|resid_post|last_token|phase2"
    )
    key_h = content_hash(harmful, extra=extra_all + "|harmful")
    key_l = content_hash(harmless, extra=extra_all + "|harmless")
    log.info("caching residuals across all %d layers...", bundle.n_layers)
    harmful_acts = cached_activations(
        key_h,
        lambda: cache_resid_all_layers(bundle, harmful, show_progress=False, format_fn=fmt),
    )
    harmless_acts = cached_activations(
        key_l,
        lambda: cache_resid_all_layers(bundle, harmless, show_progress=False, format_fn=fmt),
    )
    _check_finite("harmful_acts", harmful_acts)
    _check_finite("harmless_acts", harmless_acts)
    log.info("shapes: harmful %s, harmless %s",
             tuple(harmful_acts.shape), tuple(harmless_acts.shape))

    n_layers = bundle.n_layers
    n_h = len(harmful)
    n_l = len(harmless)
    n_total = n_h + n_l

    all_acts = torch.cat([harmful_acts, harmless_acts], dim=0)  # [n_total, L, d]
    labels_real = np.array([1] * n_h + [0] * n_l, dtype=int)
    rng = np.random.default_rng(0)
    labels_shuf = labels_real.copy()
    rng.shuffle(labels_shuf)
    rand_dir = random_unit_vector(bundle.d_model, seed=0, device="cpu")

    def loo_auc(layer: int, labels: np.ndarray) -> float:
        acts_L = all_acts[:, layer, :]
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
        h = all_acts[labels_real == 1, L, :]
        l = all_acts[labels_real == 0, L, :]
        auc_rand[L] = _auc(project(h, rand_dir), project(l, rand_dir))

    peak_layer = int(np.argmax(auc_real))
    peak_auc = float(auc_real[peak_layer])
    log.info("peak: L%d, LOO-AUC=%.3f", peak_layer, peak_auc)
    log.info("at peak: random=%.3f, shuffled-LOO=%.3f", auc_rand[peak_layer], auc_shuf[peak_layer])
    log.info("shuffled-LOO mean across layers: %.3f (should be ≈ 0.5)", float(auc_shuf.mean()))

    # Figure
    fig_suffix = f"_{args.tag}" if args.tag else ""
    fig_path = ARTIFACTS_FIGURES / f"phase2_step1_layer_sweep_{sanitized}{fig_suffix}.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    layers = np.arange(n_layers)
    ax.plot(layers, auc_real, "o-", label="real direction (LOO-CV)", linewidth=2)
    ax.plot(layers, auc_rand, "s--", label="random direction", alpha=0.6)
    ax.plot(layers, auc_shuf, "^--", label="shuffled labels (LOO-CV)", alpha=0.6)
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=1, label="chance")
    ax.axvline(peak_layer, color="red", linestyle=":", linewidth=1, alpha=0.5)
    ax.annotate(f"peak L{peak_layer}\nAUC={peak_auc:.3f}",
                xy=(peak_layer, peak_auc), xytext=(peak_layer + 1, peak_auc - 0.1),
                fontsize=9, color="red")
    ax.set_xlabel("layer")
    ax.set_ylabel("AUC of harmful-vs-harmless projection")
    ax.set_title(f"Phase 2 Step 1 — code-refusal layer sweep on {bundle.name}")
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=140)
    plt.close(fig)
    log.info("figure -> %s", fig_path)

    record = {
        "step": "phase2_step1",
        "model": bundle.name,
        "device": bundle.device,
        "data": str(pairs_path.relative_to(pairs_path.parent.parent)),
        "n_layers": n_layers,
        "n_harmful": n_h,
        "n_harmless": n_l,
        "peak_layer": peak_layer,
        "peak_auc": peak_auc,
        "auc_real": auc_real.tolist(),
        "auc_random_direction": auc_rand.tolist(),
        "auc_shuffled_labels": auc_shuf.tolist(),
        "figure": str(fig_path.relative_to(fig_path.parent.parent.parent)),
        "token_discovery": {
            "refusal_ids": token_discovery.refusal_ids,
            "compliance_ids": token_discovery.compliance_ids,
            "refusal_coverage": token_discovery.refusal_coverage,
            "compliance_coverage": token_discovery.compliance_coverage,
            "refusal_top_decoded": token_discovery.refusal_top_decoded,
            "compliance_top_decoded": token_discovery.compliance_top_decoded,
        },
        "controls_pass": {
            "shuffled_near_chance": bool(abs(float(auc_shuf.mean()) - 0.5) < 0.1),
            "random_below_real_at_peak": bool(auc_rand[peak_layer] < peak_auc),
        },
    }
    write_json(run_dir / "result.json", record)
    log.info("result -> %s", run_dir / "result.json")

    md_path = RESULTS / f"phase2_step1_{sanitized}{fig_suffix}.md"
    md_path.write_text(_render_summary(record))
    log.info("summary -> %s", md_path)

    print(f"\n=== phase2_step1: {bundle.name} ===")
    print(f"  Peak layer: L{peak_layer}  AUC={peak_auc:.3f}")
    print(f"  Random @ peak: {auc_rand[peak_layer]:.3f}")
    print(f"  Shuffled-LOO mean: {float(auc_shuf.mean()):.3f}")
    print(f"  Refusal token set:  {token_discovery.refusal_ids[:5]}  cov={token_discovery.refusal_coverage:.2f}")
    print(f"  Compliance token set: {token_discovery.compliance_ids[:5]}  cov={token_discovery.compliance_coverage:.2f}")
    return 0


def _render_summary(rec):
    md = [
        f"# Phase 2 Step 1 — layer sweep ({rec['model']})",
        "",
        f"- Data: `{rec['data']}` ({rec['n_harmful']} harmful + {rec['n_harmless']} harmless)",
        f"- Device: {rec['device']}",
        f"- **Peak: L{rec['peak_layer']}, LOO-AUC = {rec['peak_auc']:.3f}**",
        "",
        "## Controls",
        f"- Shuffled-labels LOO-CV mean across layers: "
        f"{sum(rec['auc_shuffled_labels'])/len(rec['auc_shuffled_labels']):.3f} "
        f"(should be ≈ 0.5)",
        f"- Random direction AUC at peak layer: {rec['auc_random_direction'][rec['peak_layer']]:.3f}",
        f"- Controls pass: {rec['controls_pass']}",
        "",
        "## First-token sets (discovered)",
        f"- Refusal openers (cov {rec['token_discovery']['refusal_coverage']:.2f}):",
    ]
    for item in rec['token_discovery']['refusal_top_decoded'][:8]:
        md.append(f"  - `{item['decoded']!r}` (token_id={item['token_id']}): {item['count']} / {rec['n_harmful']} = {item['fraction']:.2f}")
    md.append(f"- Compliance openers (cov {rec['token_discovery']['compliance_coverage']:.2f}):")
    for item in rec['token_discovery']['compliance_top_decoded'][:8]:
        md.append(f"  - `{item['decoded']!r}` (token_id={item['token_id']}): {item['count']} / {rec['n_harmless']} = {item['fraction']:.2f}")
    md.append("")
    md.append(f"Figure: `{rec['figure']}`")
    return "\n".join(md) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
