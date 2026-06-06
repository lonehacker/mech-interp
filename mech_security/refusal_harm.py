"""Stage 0.5 — decompose the diff-of-means d̂ into HARMFULNESS (d_harm) vs REFUSAL-DECISION (d_refuse).

The ~0.63 Llama floor could be an EXTRACTION artifact: the ablated d̂ = unit(diff_of_means(harmful,
harmless)) is BY CONSTRUCTION the harmful-vs-harmless *content* direction (= d_harm). If on Llama refusal
is a SEPARATE direction from harm-detection, ablating d̂ removes harm-encoding but not the refusal decision
→ floor — not because refusal resists ablation, but because we ablated the wrong direction. This module
extracts both directions and measures their entanglement (Zhao et al. Jul 2025: refusal ≠ harmfulness).

  d_harm   = diff-of-means(harmful-CONTENT, harmless-CONTENT)   [a-priori content labels] == the d̂ we ablate
  d_refuse = diff-of-means(REFUSED, COMPLIED)                   [model-behaviour labels, from the judge]
  cos(d_harm, d_refuse) = how entangled harm-detection and the refusal decision are in this model.

A clean d_refuse needs OFF-DIAGONAL data (harmless-refused + harmful-complied) so refused≠harmful; with a
~diagonal set d_refuse collapses into d_harm (cos→1 trivially) — report the 2×2 cell counts so that
degeneracy is visible. Pure decomposition (`decompose`) is unit-tested; the model/judge composition runs
on the pod. Reports raw numbers; human writes the verdict.
"""
from __future__ import annotations

import numpy as np
import torch

from mech_security.directions import diff_of_means, unit
from mech_security.eval_llm import judge_many
from mech_security.model import generate

# Pre-registered minimum count per OFF-DIAGONAL cell (harmless-refused, harmful-complied) for a stable
# d_refuse. Below this on a model, d_refuse is noise there and a cos gap would be estimation-quality, not
# real separation → that model's Stage-0.5 arm is INFEASIBLE as designed (report, don't force). DEVLOG §14.
MIN_OFFDIAG = 12
# Lexical-overlap tolerance (DEVLOG §14b): an off-diagonal cell must be lexically like its CONTENT class,
# not its behaviour. harmless-refused P(harmful) must not exceed harmless-complied by > this (else the
# over-refusals are harm-VOCABULARY-driven → d_refuse correlates with d_harm through shared lexicon = the
# Phase-1 vocab confound = a FALSE entanglement that passes count+bootstrap). Symmetric for harmful-complied.
LEX_TOL = 0.20


def lexical_overlap_check(harmful_texts, harmless_texts, cell_texts: dict, *, seed: int = 0) -> dict:
    """TF-IDF separability precondition. Train a word/bigram TF-IDF + LogReg classifier on harmful-vs-
    harmless CONTENT, then report mean P(harmful) for each behaviour cell's prompts. The off-diagonal
    cells must be lexically typical of their CONTENT class (harmless-refused ≈ harmless lexicon, NOT
    harm-vocab) or d_refuse is confounded with d_harm through shared vocabulary. Returns train AUC +
    per-cell mean P(harmful); the runner applies LEX_TOL deltas as a hard feasibility gate."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    corpus = list(harmful_texts) + list(harmless_texts)
    y = [1] * len(harmful_texts) + [0] * len(harmless_texts)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=5000).fit(corpus)
    clf = LogisticRegression(C=1.0, max_iter=2000, random_state=seed).fit(vec.transform(corpus), y)
    train_auc = float(roc_auc_score(y, clf.predict_proba(vec.transform(corpus))[:, 1]))
    cell_p = {name: float(clf.predict_proba(vec.transform(texts))[:, 1].mean())
              for name, texts in cell_texts.items() if texts}
    return {"tfidf_train_auc": train_auc, "cell_mean_p_harmful": cell_p}


def behavior_labels(bundle, prompts, *, fmt=None, max_new_tokens=128) -> list[int | None]:
    """Generate (NO hook) + dual-judge each prompt → 1 if REFUSED/PARTIAL, 0 if COMPLIED, None if ERROR.
    `fmt` is unused here (generate templates internally) but accepted for call-site symmetry."""
    gens = [generate(bundle, p, max_new_tokens=max_new_tokens, temperature=0.0).strip() for p in prompts]
    verdicts = judge_many(list(zip(prompts, gens)), show_progress=False)
    return [None if v.label == "ERROR" else (1 if v.label in ("REFUSED", "PARTIAL_COMPLY") else 0)
            for v in verdicts]


def decompose(acts: torch.Tensor, content_is_harmful: list[bool], refused: list[int | None]) -> dict:
    """Split [n, d] acts into the harm-content direction and the refusal-behaviour direction; return their
    cosine + the 2×2 (content × behaviour) cell counts. d_harm/d_refuse are unit vectors. cos near 1 ⇒
    harm-detection and the refusal decision are the SAME direction (Qwen-like, single-direction collapses);
    cos low ⇒ SEPARATED (Llama-like — ablating d̂=d_harm would miss refusal). Requires ≥2 per class on each
    axis; raises if an axis is single-class (then the contrast is undefined — usually too-diagonal data)."""
    ch = torch.tensor(content_is_harmful, dtype=torch.bool)
    keep = torch.tensor([r is not None for r in refused])
    rf = torch.tensor([1 if r == 1 else 0 for r in refused])
    if ch.sum() < 2 or (~ch).sum() < 2:
        raise ValueError(f"content axis single-class: {int(ch.sum())} harmful / {int((~ch).sum())} harmless")
    ref_mask, comp_mask = keep & (rf == 1), keep & (rf == 0)
    if ref_mask.sum() < 2 or comp_mask.sum() < 2:
        raise ValueError(f"behaviour axis too thin: {int(ref_mask.sum())} refused / {int(comp_mask.sum())} "
                         "complied — need off-diagonal (harmless-refused + harmful-complied) for a clean d_refuse")
    d_harm = unit(diff_of_means(acts[ch], acts[~ch]))
    d_refuse = unit(diff_of_means(acts[ref_mask], acts[comp_mask]))
    cells = {f"{'harm' if h else 'safe'}_{'refuse' if r else 'comply'}":
             int((((ch == h) & (rf == (1 if r else 0))) & keep).sum()) for h in (True, False) for r in (1, 0)}
    return {
        "cos_harm_refuse": float(torch.dot(d_harm.float(), d_refuse.float())),
        "cells": cells,
        "n_refused": int(ref_mask.sum()), "n_complied": int(comp_mask.sum()),
        "d_harm": d_harm, "d_refuse": d_refuse,
    }


def cos_bootstrap(acts: torch.Tensor, content_is_harmful, refused, *, n_boot: int = 500, seed: int = 0) -> dict:
    """Bootstrap null band for cos(d_harm, d_refuse): resample WITH REPLACEMENT within each label group,
    recompute both directions + their cosine, n_boot times. Returns mean/std/2.5-97.5 percentiles — so a
    Qwen-vs-Llama cos gap is judged against estimation noise, not read off a single draw (DEVLOG §14 #3)."""
    A = acts.detach().cpu().float().numpy() if hasattr(acts, "detach") else np.asarray(acts, dtype=float)
    ch = np.asarray(content_is_harmful, dtype=bool)
    rf = np.array([1 if r == 1 else (0 if r == 0 else -1) for r in refused])
    grp = {"h": np.where(ch)[0], "s": np.where(~ch)[0], "r": np.where(rf == 1)[0], "c": np.where(rf == 0)[0]}
    if min(len(v) for v in grp.values()) < 2:
        raise ValueError(f"a label group is single/empty: { {k: len(v) for k, v in grp.items()} }")
    g = np.random.default_rng(seed)

    def _mean(idx):
        return A[g.choice(idx, len(idx), replace=True)].mean(0)

    coss = []
    for _ in range(n_boot):
        dh = _mean(grp["h"]) - _mean(grp["s"])
        dr = _mean(grp["r"]) - _mean(grp["c"])
        dh /= np.linalg.norm(dh) + 1e-9
        dr /= np.linalg.norm(dr) + 1e-9
        coss.append(float(dh @ dr))
    coss = np.array(coss)
    return {"cos_mean": float(coss.mean()), "cos_std": float(coss.std()),
            "cos_p2.5": float(np.percentile(coss, 2.5)), "cos_p97.5": float(np.percentile(coss, 97.5)),
            "n_boot": int(n_boot)}
