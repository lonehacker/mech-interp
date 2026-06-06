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

import torch

from mech_security.directions import diff_of_means, unit
from mech_security.eval_llm import judge_many
from mech_security.model import generate


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
