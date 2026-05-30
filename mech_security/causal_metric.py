"""Continuous causal-effect metric: first-token refusal-vs-compliance logit shift.

See HOW_IT_WORKS.md §"continuous causal metric" for the rationale and math.
TL;DR: under an ablation hook, read `logits[:, -1, :]` (first-response-token
distribution) and compute mean(logit[refusal_tokens]) − mean(logit[compliance_tokens]).
The contrast cancels uniform damping — only directional shifts toward
compliance survive.

Token sets are model-specific. `VALIDATED_TOKEN_SETS` has the Gemma reference
(pre-registered, validated against 200-prompt HarmBench generations).
`discover_first_token_sets` runs the same procedure on any new model.
"""
from __future__ import annotations

import json
from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from mech_security.model import ModelBundle, format_prompt_for_bundle, tokenize_prompt

# ============================================================================
# Token-set discovery: which tokens count as "refusal openers" vs "compliance"?
# ============================================================================

@dataclass(frozen=True)
class TokenSetDiscovery:
    """Result of first-response-token argmax discovery on a model.

    Token IDs are tokenizer-specific — don't copy across model families.
    """
    refusal_ids: list[int]
    compliance_ids: list[int]
    refusal_coverage: float       # fraction of harmful baselines whose top-1 opener is in refusal_ids
    compliance_coverage: float    # same for harmless / compliance_ids
    refusal_top_decoded: list[dict] = field(default_factory=list)
    compliance_top_decoded: list[dict] = field(default_factory=list)
    discovery_n_harmful: int = 0
    discovery_n_harmless: int = 0
    model_name: str = ""
    coverage_threshold: float = 0.90
    max_tokens_per_set: int = 8


# Gemma reference — pre-registered, validated on 200 HarmBench baseline + ablated
# generations. Refusal: 198/200 baseline refusals open with token id 235285 ("I").
# Compliance: from cell-B (d̂-ablated) generations; 192/200 covered by 6 tokens.
# "##" (1620) appears in both (83× compliance, 2× refusal) but the contrast
# metric structurally cancels shared mass — placed in compliance set.
VALIDATED_TOKEN_SETS: dict[str, TokenSetDiscovery] = {
    "gemma-2-2b-it": TokenSetDiscovery(
        refusal_ids=[235285],  # "I"
        compliance_ids=[
            1620,    # "##"   (83)
            4858,    # "Here" (76)
            1917,    # "```"  (20)
            651,     # "The"  (6)
            235281,  # "\""   (4)
            6750,    # "Hey"  (3)
        ],
        refusal_coverage=0.99,
        compliance_coverage=0.96,
        refusal_top_decoded=[
            {"token_id": 235285, "decoded": "I", "count": 198, "fraction": 0.99},
        ],
        compliance_top_decoded=[
            {"token_id": 1620, "decoded": "##", "count": 83, "fraction": 0.415},
            {"token_id": 4858, "decoded": "Here", "count": 76, "fraction": 0.380},
            {"token_id": 1917, "decoded": "```", "count": 20, "fraction": 0.100},
            {"token_id": 651, "decoded": "The", "count": 6, "fraction": 0.030},
            {"token_id": 235281, "decoded": "\"", "count": 4, "fraction": 0.020},
            {"token_id": 6750, "decoded": "Hey", "count": 3, "fraction": 0.015},
        ],
        discovery_n_harmful=200,
        discovery_n_harmless=200,
        model_name="gemma-2-2b-it",
    ),
}

# Backward-compat aliases. New code should use get_or_discover_token_sets().
REFUSAL_FIRST_TOKEN_IDS_GEMMA2 = VALIDATED_TOKEN_SETS["gemma-2-2b-it"].refusal_ids
COMPLIANCE_FIRST_TOKEN_IDS_GEMMA2 = VALIDATED_TOKEN_SETS["gemma-2-2b-it"].compliance_ids


@torch.no_grad()
def _argmax_first_response_token_ids(bundle: ModelBundle, templated: list[str]) -> list[int]:
    """For each templated prompt: argmax of logits[0, -1, :] under no hook."""
    device = bundle.model.cfg.device
    out = []
    for text in templated:
        ids = bundle.model.to_tokens(text, prepend_bos=False).to(device)
        logits = bundle.model(ids, return_type="logits")
        out.append(int(logits[0, -1, :].argmax().item()))
    return out


def _greedy_cover(
    ids: list[int], threshold: float, max_tokens: int
) -> tuple[list[int], list[tuple[int, int]], float]:
    """Top-k by frequency until cumulative coverage ≥ threshold (or max_tokens).

    Returns (selected, sorted_freqs, achieved_coverage).
    """
    n = len(ids)
    if n == 0:
        return [], [], 0.0
    freqs = Counter(ids).most_common()
    selected: list[int] = []
    cum = 0
    for tid, cnt in freqs[:max_tokens]:
        selected.append(tid)
        cum += cnt
        if cum / n >= threshold:
            break
    return selected, freqs, cum / n


def _decode_top(bundle: ModelBundle, freqs: list[tuple[int, int]], n: int, k: int) -> list[dict]:
    return [
        {"token_id": int(tid), "decoded": bundle.model.tokenizer.decode([tid]),
         "count": int(cnt), "fraction": cnt / n}
        for tid, cnt in freqs[:k]
    ]


@torch.no_grad()
def discover_first_token_sets(
    bundle: ModelBundle,
    templated_harmful: list[str],
    templated_harmless: list[str],
    *,
    coverage_threshold: float = 0.90,
    max_tokens_per_set: int = 8,
) -> TokenSetDiscovery:
    """Argmax-based discovery of refusal + compliance first-token sets.

    Procedure:
      1. Argmax logits[:, -1, :] on harmful prompts (no hook) → refusal openers.
      2. Same on harmless → compliance openers.
      3. Greedy top-k from each side until coverage ≥ threshold.
      4. Disjointness: shared tokens go to the side with higher raw count.

    Caller templates prompts before passing in (keeps this module independent
    of chat-template choice). Harmless-baseline-as-compliance-proxy is a v1
    approximation; gold standard is `refine_compliance_set_from_ablation` once
    a d̂ exists.
    """
    h_ids = _argmax_first_response_token_ids(bundle, templated_harmful)
    l_ids = _argmax_first_response_token_ids(bundle, templated_harmless)
    r_sel, r_freqs, r_cov = _greedy_cover(h_ids, coverage_threshold, max_tokens_per_set)
    c_sel, c_freqs, c_cov = _greedy_cover(l_ids, coverage_threshold, max_tokens_per_set)

    # Disjointness: place shared tokens on the heavier side
    r_count, c_count = dict(r_freqs), dict(c_freqs)
    for tok in set(r_sel) & set(c_sel):
        if r_count.get(tok, 0) >= c_count.get(tok, 0):
            c_sel = [t for t in c_sel if t != tok]
        else:
            r_sel = [t for t in r_sel if t != tok]

    return TokenSetDiscovery(
        refusal_ids=r_sel, compliance_ids=c_sel,
        refusal_coverage=r_cov, compliance_coverage=c_cov,
        refusal_top_decoded=_decode_top(bundle, r_freqs, len(h_ids), max_tokens_per_set),
        compliance_top_decoded=_decode_top(bundle, c_freqs, len(l_ids), max_tokens_per_set),
        discovery_n_harmful=len(templated_harmful),
        discovery_n_harmless=len(templated_harmless),
        model_name=bundle.name,
        coverage_threshold=coverage_threshold,
        max_tokens_per_set=max_tokens_per_set,
    )


_TOKEN_SET_FIELDS = (
    "refusal_ids", "compliance_ids", "refusal_coverage", "compliance_coverage",
    "refusal_top_decoded", "compliance_top_decoded",
    "discovery_n_harmful", "discovery_n_harmless",
    "model_name", "coverage_threshold", "max_tokens_per_set",
)


def save_token_sets(discovery: TokenSetDiscovery, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(discovery), indent=2))


def load_token_sets(path: Path) -> TokenSetDiscovery:
    """Field-by-field reconstruction so cache files with extra fields don't break."""
    data = json.loads(path.read_text())
    return TokenSetDiscovery(**{k: data[k] for k in _TOKEN_SET_FIELDS if k in data})


def get_or_discover_token_sets(
    bundle: ModelBundle,
    *,
    cache_path: Path | None = None,
    templated_harmful: list[str] | None = None,
    templated_harmless: list[str] | None = None,
    coverage_threshold: float = 0.90,
    max_tokens_per_set: int = 8,
) -> TokenSetDiscovery:
    """Validated → cache → discover. Raises if none available.

    Priority:
      1. VALIDATED_TOKEN_SETS[bundle.name] if present (frozen reference).
      2. cache_path if it exists.
      3. discover_first_token_sets(bundle, templated_harmful, templated_harmless)
         and cache to cache_path if provided.
    """
    if bundle.name in VALIDATED_TOKEN_SETS:
        return VALIDATED_TOKEN_SETS[bundle.name]
    if cache_path is not None and cache_path.exists():
        return load_token_sets(cache_path)
    if templated_harmful is None or templated_harmless is None:
        raise ValueError(
            f"No validated entry for {bundle.name!r} and no cache at {cache_path!r}. "
            "Pass templated_harmful + templated_harmless to trigger discovery."
        )
    discovery = discover_first_token_sets(
        bundle, templated_harmful, templated_harmless,
        coverage_threshold=coverage_threshold, max_tokens_per_set=max_tokens_per_set,
    )
    if cache_path is not None:
        save_token_sets(discovery, cache_path)
    return discovery


@torch.no_grad()
def refine_compliance_set_from_ablation(
    bundle: ModelBundle,
    templated_harmful_prompts: list[str],
    ablation_hook_ctx,
    *,
    max_tokens_per_set: int = 8,
) -> dict:
    """Gold-standard compliance distribution: argmax under the actual ablation hook.

    `ablation_hook_ctx` is e.g. `ablate_dir(model, d_hat)`. Returns a frequency
    summary the caller uses to validate/refine the discovery-derived compliance set.
    """
    device = bundle.model.cfg.device
    out: list[int] = []
    with ablation_hook_ctx:
        for text in templated_harmful_prompts:
            ids = bundle.model.to_tokens(text, prepend_bos=False).to(device)
            logits = bundle.model(ids, return_type="logits")
            out.append(int(logits[0, -1, :].argmax().item()))
    freqs = Counter(out).most_common()
    n = len(out)
    return {
        "n": n,
        "top_decoded": _decode_top(bundle, freqs, n, max_tokens_per_set),
        "raw_freqs": {int(tid): int(cnt) for tid, cnt in freqs},
    }


# ============================================================================
# Continuous causal-effect metric
# ============================================================================

@dataclass(frozen=True)
class CausalEffect:
    """Per-prompt continuous causal-effect measurement.

    effect_signed (primary):
        (refusal_logit_ablated − compliance_logit_ablated)
      − (refusal_logit_baseline − compliance_logit_baseline)
      Negative = causal (mass moved from refusal toward compliance).

    refusal_delta / compliance_delta (diagnostic):
        Raw per-side shifts. Confounded by norm perturbations; use for
        eyeball debugging only, not for the headline.
    """
    effect_signed: float
    refusal_delta: float
    compliance_delta: float
    n_prompts: int
    per_prompt: torch.Tensor | None = None
    refusal_per_prompt: torch.Tensor | None = None
    compliance_per_prompt: torch.Tensor | None = None


def verify_template_boundary(bundle: ModelBundle, sample_prompt: str = "Hello") -> dict:
    """Eyeball-check: last 5 tokens of templated prompt + top-1 prediction.

    Each model's chat template has its own suffix (Gemma: `<start_of_turn>model\\n`,
    Qwen ChatML: `<|im_start|>assistant\\n`). Caller verifies last_5_decoded looks
    like the expected suffix; if not, logits[:, -1, :] reads the wrong thing.
    """
    templated = format_prompt_for_bundle(bundle, sample_prompt)
    ids = tokenize_prompt(bundle, templated)
    last_5 = ids[0, -5:].tolist()
    with torch.no_grad():
        logits = bundle.model(ids, return_type="logits")
    top1 = int(logits[0, -1, :].argmax().item())
    return {
        "last_5_token_ids": last_5,
        "last_5_decoded": [bundle.model.tokenizer.decode([t]) for t in last_5],
        "top1_first_response_token_id": top1,
        "top1_first_response_token_decoded": bundle.model.tokenizer.decode([top1]),
    }


@torch.no_grad()
def causal_effect_under_hook(
    bundle: ModelBundle,
    prompts: list[str],
    refusal_token_ids: list[int],
    compliance_token_ids: list[int],
    hook_ctx=None,
    return_per_prompt: bool = True,
) -> dict:
    """First-token refusal/compliance logit means under an active hook (or none).

    Same code path for baseline, causal d̂, inert directions, AND random vectors —
    no path differences, so the null band and causal cells are on the same scale.

    Returns: {refusal_logit_mean, compliance_logit_mean, contrast_mean, n_prompts,
              [refusal_per_prompt, compliance_per_prompt]}
    """
    device = bundle.model.cfg.device
    r_idx = torch.tensor(refusal_token_ids, dtype=torch.long, device=device)
    c_idx = torch.tensor(compliance_token_ids, dtype=torch.long, device=device)
    r_per, c_per = [], []
    with (hook_ctx if hook_ctx is not None else nullcontext()):
        for raw in prompts:
            text = format_prompt_for_bundle(bundle, raw)
            ids = tokenize_prompt(bundle, text).to(device)
            logits = bundle.model(ids, return_type="logits")    # [1, seq, vocab]
            last = logits[0, -1, :]                              # [vocab]
            r_per.append(last[r_idx].mean().detach().float().cpu())
            c_per.append(last[c_idx].mean().detach().float().cpu())
    r_t, c_t = torch.stack(r_per), torch.stack(c_per)
    out = {
        "refusal_logit_mean": float(r_t.mean()),
        "compliance_logit_mean": float(c_t.mean()),
        "contrast_mean": float((r_t - c_t).mean()),
        "n_prompts": len(prompts),
    }
    if return_per_prompt:
        out["refusal_per_prompt"] = r_t
        out["compliance_per_prompt"] = c_t
    return out


def compute_causal_effect(
    bundle: ModelBundle,
    prompts: list[str],
    direction: torch.Tensor | None,
    refusal_token_ids: list[int],
    compliance_token_ids: list[int],
    baseline: dict | None = None,
) -> CausalEffect:
    """direction=None → baseline (no hook, effect_signed=0 by definition).

    Same call used for the causal d̂, inert cells, L3 d̂, and random vectors —
    pass a precomputed baseline dict to avoid recomputing it per cell.
    """
    if baseline is None:
        baseline = causal_effect_under_hook(
            bundle, prompts, refusal_token_ids, compliance_token_ids,
            hook_ctx=None, return_per_prompt=True,
        )
    if direction is None:
        return CausalEffect(
            effect_signed=0.0, refusal_delta=0.0, compliance_delta=0.0,
            n_prompts=baseline["n_prompts"],
            per_prompt=torch.zeros(baseline["n_prompts"]),
            refusal_per_prompt=baseline["refusal_per_prompt"],
            compliance_per_prompt=baseline["compliance_per_prompt"],
        )

    from mech_security.directions import ablate_dir
    abl = causal_effect_under_hook(
        bundle, prompts, refusal_token_ids, compliance_token_ids,
        hook_ctx=ablate_dir(bundle.model, direction), return_per_prompt=True,
    )
    r_per = abl["refusal_per_prompt"] - baseline["refusal_per_prompt"]
    c_per = abl["compliance_per_prompt"] - baseline["compliance_per_prompt"]
    effect_per = r_per - c_per   # negative = causal
    return CausalEffect(
        effect_signed=float(effect_per.mean()),
        refusal_delta=float(r_per.mean()),
        compliance_delta=float(c_per.mean()),
        n_prompts=abl["n_prompts"],
        per_prompt=effect_per,
        refusal_per_prompt=r_per,
        compliance_per_prompt=c_per,
    )
