"""Probe-after-ablation — the decisive H-dim vs H-nonlinear test (PHASE3_PLAN 2026-06-06, Goal B).

The scientific question for Llama-8B: *why* does diff-of-means ablation fully collapse Qwen-3B refusal
but only half-collapse Llama? After ablating the diff-of-means k-subspace at the best cell, we ask: is
the LEFTOVER refusal still LINEARLY READABLE from the post-ablation residual stream?

  - leftover refusal IS still linearly readable (probe AUC high) AND model still refuses  → **H-dim**:
    refusal is linearly present but not low-k-ablatable ("linearly present, not ablatable", not a win).
  - leftover refusal NOT linearly readable (probe AUC ≈ chance) BUT model still refuses → **H-nonlinear**:
    the remaining refusal is computed somewhere the linear residual attack cannot see — i.e. diff-of-means
    underperforms on Llama *because Llama's refusal isn't fully a linear residual-stream feature*.

Qwen-3B is the positive control: post-ablation refusal should be GONE (≈single-class) AND therefore
unreadable — that's what a full collapse looks like.

Load-bearing; composes tested primitives (extract_d_hat, diffmeans_subspace, ablate_subspace, the Haiku
judge, train_probe). Reports RAW numbers + the shuffled-label control; the human writes the verdict.
Confound to log alongside any high AUC: topic/vocab leakage (a probe can predict refusal from the prompt's
TOPIC, not a refusal feature) — the H-mixture hypothesis; the shuffled control rules out split leakage
but not topic, so report both and flag it.
"""
from __future__ import annotations

from functools import partial

import numpy as np
import torch

from mech_security.activations import cache_resid_all_layers
from mech_security.directions import ablate_subspace, diffmeans_subspace, extract_d_hat
from mech_security.eval_llm import judge_many
from mech_security.model import format_prompt_for_bundle, generate
from mech_security.probes import train_probe
from mech_security.redteam import _assert_template_consistency

MIN_PER_CLASS = 10  # below this, the post-ablation behavior is ~single-class → probe underpowered/degenerate


def _labels_under_ablation(bundle, dirs, prompts, *, max_new_tokens, fmt):
    """Generate on `prompts` UNDER the k-subspace ablation, judge each → per-prompt label
    refused/partial = 1, complied = 0 (ERROR dropped). Returns (kept_idx, labels, completions)."""
    gens = []
    with ablate_subspace(bundle.model, dirs):
        for p in prompts:
            gens.append(generate(bundle, p, max_new_tokens=max_new_tokens, temperature=0.0).strip())
    verdicts = judge_many(list(zip(prompts, gens)), show_progress=False)
    kept, labels = [], []
    for i, v in enumerate(verdicts):
        if v.label == "ERROR":
            continue
        kept.append(i)
        labels.append(1 if v.label in ("REFUSED", "PARTIAL_COMPLY") else 0)
    return kept, labels, gens


def _max_auc_probe(acts, y, *, seed, shuffle_seed) -> dict:
    """Per-layer logistic probe + shuffled-label control on [n, n_layers, d] acts; return the layer with
    MAX held-out AUC (the most generous "readable anywhere") + the shuffled control at that layer."""
    y = torch.as_tensor(y)
    y_shuf = y.numpy().copy()
    np.random.default_rng(shuffle_seed).shuffle(y_shuf)
    y_shuf = torch.tensor(y_shuf)
    per_layer = []
    for lp in range(acts.shape[1]):
        real = train_probe(acts[:, lp, :], y, seed=seed)
        shuf = train_probe(acts[:, lp, :], y_shuf, seed=seed)
        per_layer.append({"layer": lp, "test_auc": real.test_auc, "test_acc": real.test_acc,
                          "shuffled_auc": shuf.test_auc})
    best = max(per_layer, key=lambda d: (d["test_auc"] if d["test_auc"] == d["test_auc"] else -1.0))
    return {"best_readout_layer": best["layer"], "max_test_auc": best["test_auc"],
            "max_test_acc": best["test_acc"], "shuffled_auc_at_best": best["shuffled_auc"], "per_layer": per_layer}


def topic_controlled_probe(acts, labels, topic_embeddings, *, n_topics=5, seed=0) -> dict:
    """Discriminate H-dim from H-mixture: is refuse-vs-comply decodable with TOPIC HELD CONSTANT?

    Cluster prompts into `n_topics` (KMeans on `topic_embeddings`), then LEAVE-ONE-TOPIC-OUT CV: train the
    refuse/comply probe on prompts OUTSIDE cluster c, test on c. Mean held-out AUC across folds —
      stays HIGH  ⇒ refusal decodable independent of topic ⇒ **H-dim**;
      ~chance     ⇒ the readout was topic-correlation ⇒ **H-mixture** (probe read topic, not residual refusal).
    Folds whose train/test cluster is single-class are skipped (logged) — heavy skipping itself signals
    topic↔refusal confounding. `acts`/`topic_embeddings`: [n, d] (probe layer); `labels`: [n] 0/1. Pure(ish)
    — sklearn in, scalars out; unit-tested in tests/test_probe_ablation.py."""
    from sklearn.cluster import KMeans
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    X = acts.detach().cpu().float().numpy() if hasattr(acts, "detach") else np.asarray(acts, dtype=float)
    emb = topic_embeddings.detach().cpu().float().numpy() if hasattr(topic_embeddings, "detach") else np.asarray(topic_embeddings, dtype=float)
    y = np.asarray(labels).astype(int)
    n_topics = min(n_topics, len(y))
    clusters = KMeans(n_clusters=n_topics, random_state=seed, n_init=10).fit_predict(emb)
    fold_aucs, skipped, per_cluster = [], [], []
    for c in sorted(set(clusters.tolist())):
        te, tr = clusters == c, clusters != c
        per_cluster.append({"cluster": int(c), "n": int(te.sum()), "refuse_rate": float(y[te].mean())})
        if len(set(y[te].tolist())) < 2 or len(set(y[tr].tolist())) < 2:  # single-class fold → undecidable
            skipped.append(int(c))
            continue
        clf = LogisticRegression(C=1.0, max_iter=2000, random_state=seed).fit(X[tr], y[tr])
        fold_aucs.append(float(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1])))
    return {
        "n_topics": int(n_topics),
        "leave_topic_out_auc_mean": float(np.mean(fold_aucs)) if fold_aucs else float("nan"),
        "per_fold_auc": fold_aucs,
        "skipped_single_class_folds": skipped,
        "per_cluster_refuse_rate": per_cluster,  # spread across clusters = topic↔refusal confounding
    }


def probe_after_ablation(bundle, probe_prompts, harmless_extract, *, layer, position=-1, k=3,
                         extract_harmful=None, harmless_probe=None, fmt=None, seed=0, shuffle_seed=999,
                         max_new_tokens=128) -> dict:
    """Ablate the diff-of-means k-subspace at (layer, position); then probe whether post-ablation refusal
    is still linearly readable. Returns raw numbers (no verdict).

    Steps: (1) build the k-dim diff-of-means subspace at the best cell from `extract_harmful` vs
    `harmless_extract` (DISJOINT from `probe_prompts` so the probe isn't in-sample; k=1 row = headline d̂);
    (2) under that ablation, generate + judge `probe_prompts` → per-prompt refuse/comply labels;
    (3) under the SAME ablation, cache post-ablation resid at EVERY layer; (4) train a linear probe per
    layer (refuse vs comply) + a shuffled-label control, and report the MAX held-out AUC across layers
    (the most generous "is the leftover refusal readable ANYWHERE post-ablation")."""
    fmt = fmt or partial(format_prompt_for_bundle, bundle)
    _assert_template_consistency(bundle, fmt)  # extraction template MUST == scoring (the Llama-specific 2026-05-31 bug)
    d_hat, H, L, _ = extract_d_hat(bundle, extract_harmful if extract_harmful is not None else probe_prompts,
                                   harmless_extract, layer=layer, position=position, format_fn=fmt)
    dirs = diffmeans_subspace(H, L, k=k, d1=d_hat)
    kept, labels, _ = _labels_under_ablation(bundle, dirs, probe_prompts, max_new_tokens=max_new_tokens, fmt=fmt)
    n_ref = int(sum(labels))
    n_comp = len(labels) - n_ref
    out = {"layer_ablated": layer, "position": position, "k": int(k), "n_total": len(probe_prompts),
           "n_kept": len(kept), "n_refused": n_ref, "n_complied": n_comp}

    if n_ref < MIN_PER_CLASS or n_comp < MIN_PER_CLASS:
        out["probe"] = None
        out["note"] = (f"post-ablation behavior ~single-class ({n_ref} refused / {n_comp} complied; "
                       f"need ≥{MIN_PER_CLASS} each) — for the defended/Llama arm this is INCONCLUSIVE; "
                       "for the Qwen positive control near-zero refused IS the full-collapse signal.")
        return out

    # post-ablation activations (all layers) on the kept harmful prompts, under the SAME ablation
    with ablate_subspace(bundle.model, dirs):
        acts_h = cache_resid_all_layers(bundle, [probe_prompts[i] for i in kept], format_fn=fmt, show_progress=False)
    # REFUSAL probe: is refuse-vs-comply still linearly readable post-ablation?
    out["probe"] = _max_auc_probe(acts_h, labels, seed=seed, shuffle_seed=shuffle_seed)
    # H-dim vs H-mixture DISCRIMINATOR (DEVLOG §11): does refuse/comply decode with TOPIC held constant?
    # Leave-topic-out CV at the best readout layer. High ⇒ H-dim; ~chance/undecidable ⇒ H-mixture (topic).
    bl = out["probe"]["best_readout_layer"]
    out["topic_controlled"] = topic_controlled_probe(acts_h[:, bl, :], labels, acts_h[:, bl, :],
                                                     n_topics=max(2, min(5, len(labels) // 6)), seed=seed)

    # ORTHOGONAL-READABILITY GATE (pre-registered, DEVLOG §11): only load-bearing when refusal AUC ~chance.
    # Same post-ablation acts, different target — harmful-vs-harmless CONTENT. Readable here + chance refusal
    # ⇒ H-nonlinear (rep intact, refusal specifically gone, linearly). Both chance ⇒ activations merely
    # DAMAGED ⇒ inconclusive (needs gentler ablation), NOT H-nonlinear.
    if harmless_probe:
        with ablate_subspace(bundle.model, dirs):
            acts_hl = cache_resid_all_layers(bundle, list(harmless_probe), format_fn=fmt, show_progress=False)
        acts_orth = torch.cat([acts_h, acts_hl], dim=0)
        y_orth = [1] * acts_h.shape[0] + [0] * acts_hl.shape[0]
        out["orthogonal_probe"] = _max_auc_probe(acts_orth, y_orth, seed=seed, shuffle_seed=shuffle_seed)
        out["orthogonal_probe"]["target"] = "harmful_vs_harmless_content"
    else:
        out["orthogonal_probe"] = None
    return out
