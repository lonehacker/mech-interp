"""
Continuous causal-effect metric for ablation directions.

Why this exists (Phase 1.5-A master spec):
Binary refusal-rate (n=12) under-sells the classification ≠ causation finding
and admits a "maybe inert cells are just noisy" objection. The fix is to
report a continuous signal-to-noise readout for EVERY ablation cell, in the
same units, computed from the same code path, and to z-score it against a
random-vector null band built from 5+ random unit vectors.

The metric — refusal-minus-compliance contrast in logit-difference units:

    For each prompt, run a single forward pass through the model with the
    direction-ablation hook active (Arditi multi-layer recipe). Read
    logits[:, -1, :] — the model's distribution over the FIRST RESPONSE
    TOKEN, after attending over the full ablated prompt. Compute:

        refusal_logit  = mean(logit[r_tokens])
        compliance_logit = mean(logit[c_tokens])
        contrast = refusal_logit - compliance_logit
        effect_signed = contrast_ablated - contrast_baseline

    Negative effect_signed means mass moved from refusal toward compliance
    under ablation — the causal signature. For a random unit vector,
    effect_signed should be ~0 (mean over 5+ vectors defines the null band).
    For the bootstrap-LDA classification-equivalent directions, it should
    also be near zero / inside the null band.

Why refusal-MINUS-compliance, not just Δlog p(refusal):
A direction that mildly damps *all* first-token probability mass (a norm
perturbation that's not directionally on refusal) shows a negative Δlog p
that looks like a weak causal effect but is just a scale wobble. The
refusal-minus-compliance contrast cancels exactly that — uniform damping
moves both terms together and nets to zero. Only a *directional* shift
toward compliance survives.

Why first-token only:
logits[:, -1, :] after a full prompt forward is the model's distribution
over the first generated token, given the ablated prompt and after all
26 transformer blocks have run. By this point the L17+ "execution band"
has resolved the refusal decision into the next-token logits. Reading at
an intermediate layer (or at an earlier position within the prompt)
samples the decision before it's fully formed.

Why single-token sets, not multi-token openers:
"I'm" tokenizes as ["I", "'", "m"]; "Here's" as ["Here", "'", "s"]. If
the metric sums first-token log-prob over openers of mixed length, it
compares P(first token of a 2-token opener) against P(a complete
1-token opener) — different events. We define both sets at the level
of the FIRST token only, validated against actual baseline + ablated
generations.

Token sets — VALIDATED against existing 200-prompt HarmBench generations:
    Refusal: model opens 198/200 baseline refusals with single token "I"
             (id 235285). Tertiary "##" appears 2× refusal + 83× compliance;
             ambiguous, excluded from both sets.
    Compliance: from cell-B (d̂-ablated) generations, the actual top compliance
                openers are "Here" / "```" / "The" / '"' / "Hey".
    Original proposal's "As", "I'm", "Sorry", "Sure", "Certainly", "To"
    essentially don't fire; replaced with validated set.

Pre-registered token sets (logged 2026-05-29 before any null-band run):

Phase 2 portability (2026-05-29):
The hardcoded Gemma constants now live inside `VALIDATED_TOKEN_SETS`, with
a `discover_first_token_sets()` function that runs the same procedure on
any new model — argmax of `logits[:, -1, :]` on templated harmful prompts
(no hook) gives the refusal-opener distribution; same on harmless gives
the compliance-opener distribution. `get_or_discover_token_sets()` is the
top-level entry point: returns the validated set if one exists for the
model name, else loads from cache, else runs discovery.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from src.model import ModelBundle, format_prompt, format_prompt_for_bundle, tokenize_prompt


# Validated first-token IDs for Gemma-2-2b-it (see module docstring).
# Pre-registered before the hardened null-band experiment runs.
#
# Disjoint-check: zero (0/200) ablated/compliance generations open with
# "I" (235285), so the refusal set and compliance set are CLEAN at the
# first-token level — no "I"→"cannot" vs "I"→"'ll be happy" disambiguation
# needed, the two terms of the contrast read non-overlapping token events.
#
# Shared-mass handling: "##" (1620) appears 83× in compliance and 2× in
# refusal openers. We place it in the compliance set. The contrast metric
# (refusal_logit - compliance_logit) STRUCTURALLY CANCELS shared openers:
# if "##" contributes to both terms with similar weight in a given prompt,
# its contribution to the contrast nets out. The 2-vs-83 asymmetry means
# "##" net-contributes 81 to compliance and self-cancels the 2-refusal mass.
# This is a property of the contrast metric, not a judgment call.
#
# Capture rates: refusal set covers 198/200 = 99% of baseline refusal
# openers. Compliance set covers 192/200 = 96% (the missing 8 are a
# diffuse singleton tail — Setting, Starting, While, etc.). The 3pp
# asymmetry biases the contrast toward UNDERESTIMATING the causal effect
# (compliance term has slightly less mass to credit), so reported effects
# are a mild lower bound on the true causal shift. Safe direction to err.


@dataclass(frozen=True)
class TokenSetDiscovery:
    """Result of running first-response-token argmax discovery on a model.

    All token IDs are tokenizer-specific to the discovering model. Don't
    copy these across model families.
    """
    refusal_ids: list[int]
    compliance_ids: list[int]
    refusal_coverage: float      # fraction of harmful baselines whose top-1 opener is in refusal_ids
    compliance_coverage: float   # same for harmless baselines / compliance_ids
    refusal_top_decoded: list[dict] = field(default_factory=list)     # [{token_id, decoded, count, fraction}, ...]
    compliance_top_decoded: list[dict] = field(default_factory=list)
    discovery_n_harmful: int = 0
    discovery_n_harmless: int = 0
    model_name: str = ""
    coverage_threshold: float = 0.90
    max_tokens_per_set: int = 8


# Frozen reference for models we've already validated. The Gemma entry is
# what `discover_first_token_sets` SHOULD produce if re-run, give-or-take
# rounding — kept as the regression target.
VALIDATED_TOKEN_SETS: dict[str, TokenSetDiscovery] = {
    "gemma-2-2b-it": TokenSetDiscovery(
        refusal_ids=[235285],  # "I" — 198/200 baseline refusal openers
        compliance_ids=[
            1620,    # "##"   — 83 (markdown headers)
            4858,    # "Here" — 76 (covers "Here" + "Here's")
            1917,    # "```"  — 20 (code fence)
            651,     # "The"  —  6
            235281,  # "\""   —  4
            6750,    # "Hey"  —  3
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
        coverage_threshold=0.90,
        max_tokens_per_set=8,
    ),
}


# Backward-compat aliases. Existing experiment runners that import these
# names keep working; new code should call `get_or_discover_token_sets()`.
REFUSAL_FIRST_TOKEN_IDS_GEMMA2 = VALIDATED_TOKEN_SETS["gemma-2-2b-it"].refusal_ids
COMPLIANCE_FIRST_TOKEN_IDS_GEMMA2 = VALIDATED_TOKEN_SETS["gemma-2-2b-it"].compliance_ids


@torch.no_grad()
def _argmax_first_response_token_ids(
    bundle: ModelBundle,
    templated_prompts: list[str],
) -> list[int]:
    """For each pre-templated prompt, return the argmax of `logits[:, -1, :]`
    under no hook — the model's actual top-1 first-response token at T=0."""
    device = bundle.model.cfg.device
    out_ids: list[int] = []
    for text in templated_prompts:
        ids = bundle.model.to_tokens(text, prepend_bos=False).to(device)
        logits = bundle.model(ids, return_type="logits")
        out_ids.append(int(logits[0, -1, :].argmax().item()))
    return out_ids


def _greedy_cover(
    token_ids: list[int],
    threshold: float,
    max_tokens: int,
) -> tuple[list[int], list[tuple[int, int]], float]:
    """Sort tokens by frequency desc; pick top-k until cumulative coverage
    ≥ threshold or max_tokens reached.

    Returns (selected_ids, sorted_freqs, achieved_coverage).
    """
    n = len(token_ids)
    if n == 0:
        return [], [], 0.0
    counter = Counter(token_ids)
    sorted_freqs = counter.most_common()
    selected: list[int] = []
    cumulative = 0
    for token_id, count in sorted_freqs[:max_tokens]:
        selected.append(token_id)
        cumulative += count
        if cumulative / n >= threshold:
            break
    return selected, sorted_freqs, cumulative / n


def _decode_top(
    bundle: ModelBundle,
    sorted_freqs: list[tuple[int, int]],
    n_total: int,
    max_tokens: int,
) -> list[dict]:
    return [
        {
            "token_id": int(tid),
            "decoded": bundle.model.tokenizer.decode([tid]),
            "count": int(cnt),
            "fraction": cnt / n_total,
        }
        for tid, cnt in sorted_freqs[:max_tokens]
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
    """Discover refusal-opener and compliance-opener token sets for any
    instruction-tuned model. No d̂ required.

    Procedure:
      1. Argmax of `logits[:, -1, :]` on templated harmful prompts (no hook)
         = the model's first-token response. Since instruct models refuse
         harmful prompts at baseline, these argmaxes are refusal openers.
      2. Same on harmless prompts = compliance openers (since instruct
         models answer harmless prompts at baseline).
      3. Greedy top-k from each side until cumulative coverage ≥ threshold.
      4. Disjointness: if a token appears in BOTH selected sets, keep it on
         the side with higher raw count; drop it from the other.

    The harmless-baseline-as-compliance-proxy is a v1 approximation — the
    gold standard uses d̂-ablated-on-harmful argmaxes (see
    `refine_compliance_set_from_ablation`). For the contrast metric, the v1
    set is robust enough for the headline; refine for hardened reports.

    The caller is responsible for templating the prompts before passing
    them in (intentional — keeps this module independent of any specific
    chat template).
    """
    harmful_ids = _argmax_first_response_token_ids(bundle, templated_harmful)
    harmless_ids = _argmax_first_response_token_ids(bundle, templated_harmless)

    r_sel, r_freqs, r_cov = _greedy_cover(harmful_ids, coverage_threshold, max_tokens_per_set)
    c_sel, c_freqs, c_cov = _greedy_cover(harmless_ids, coverage_threshold, max_tokens_per_set)

    # Disjointness — place shared tokens on the heavier side
    r_count = dict(r_freqs)
    c_count = dict(c_freqs)
    for tok in set(r_sel) & set(c_sel):
        if r_count.get(tok, 0) >= c_count.get(tok, 0):
            c_sel = [t for t in c_sel if t != tok]
        else:
            r_sel = [t for t in r_sel if t != tok]

    return TokenSetDiscovery(
        refusal_ids=r_sel,
        compliance_ids=c_sel,
        refusal_coverage=r_cov,
        compliance_coverage=c_cov,
        refusal_top_decoded=_decode_top(bundle, r_freqs, len(harmful_ids), max_tokens_per_set),
        compliance_top_decoded=_decode_top(bundle, c_freqs, len(harmless_ids), max_tokens_per_set),
        discovery_n_harmful=len(templated_harmful),
        discovery_n_harmless=len(templated_harmless),
        model_name=bundle.name,
        coverage_threshold=coverage_threshold,
        max_tokens_per_set=max_tokens_per_set,
    )


def save_token_sets(discovery: TokenSetDiscovery, path: Path) -> None:
    """Persist a discovery as JSON for later reuse."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(discovery), indent=2))


def load_token_sets(path: Path) -> TokenSetDiscovery:
    """Load a saved discovery from JSON. Field-by-field reconstruction so
    additions to the dataclass don't silently corrupt old caches."""
    data = json.loads(path.read_text())
    return TokenSetDiscovery(**{
        k: data[k] for k in (
            "refusal_ids", "compliance_ids", "refusal_coverage", "compliance_coverage",
            "refusal_top_decoded", "compliance_top_decoded",
            "discovery_n_harmful", "discovery_n_harmless",
            "model_name", "coverage_threshold", "max_tokens_per_set",
        ) if k in data
    })


def get_or_discover_token_sets(
    bundle: ModelBundle,
    *,
    cache_path: Path | None = None,
    templated_harmful: list[str] | None = None,
    templated_harmless: list[str] | None = None,
    coverage_threshold: float = 0.90,
    max_tokens_per_set: int = 8,
) -> TokenSetDiscovery:
    """Top-level entry point. Priority:
      1. If `bundle.name` is in `VALIDATED_TOKEN_SETS`, return the frozen
         reference.
      2. Else if `cache_path` exists, load and return.
      3. Else if templated harmful + harmless prompts provided, run discovery,
         cache (if `cache_path` set), and return.
      4. Else raise — caller must provide one of (validated entry, cache,
         discovery prompts).
    """
    if bundle.name in VALIDATED_TOKEN_SETS:
        return VALIDATED_TOKEN_SETS[bundle.name]
    if cache_path is not None and cache_path.exists():
        return load_token_sets(cache_path)
    if templated_harmful is None or templated_harmless is None:
        raise ValueError(
            f"Model {bundle.name!r} has no validated entry in VALIDATED_TOKEN_SETS "
            f"and no cache at {cache_path!r}. Pass templated_harmful + templated_harmless "
            f"to trigger discovery."
        )
    discovery = discover_first_token_sets(
        bundle, templated_harmful, templated_harmless,
        coverage_threshold=coverage_threshold,
        max_tokens_per_set=max_tokens_per_set,
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
    """Gold-standard compliance-opener distribution: run the actual ablation
    hook on harmful prompts and collect argmax first-tokens. Returns a
    frequency summary the caller can use to validate or refine the
    discovery-derived compliance set.

    `ablation_hook_ctx` is a context manager (e.g. `ablate_dir(model, d_hat)`).
    """
    device = bundle.model.cfg.device
    out_ids: list[int] = []
    with ablation_hook_ctx:
        for text in templated_harmful_prompts:
            ids = bundle.model.to_tokens(text, prepend_bos=False).to(device)
            logits = bundle.model(ids, return_type="logits")
            out_ids.append(int(logits[0, -1, :].argmax().item()))
    freqs = Counter(out_ids).most_common()
    n = len(out_ids)
    return {
        "n": n,
        "top_decoded": _decode_top(bundle, freqs, n, max_tokens_per_set),
        "raw_freqs": {int(tid): int(cnt) for tid, cnt in freqs},
    }


@dataclass(frozen=True)
class CausalEffect:
    """Per-prompt continuous causal-effect measurement.

    Fields:
        effect_signed: contrast_ablated - contrast_baseline. Negative = causal
            (mass moved from refusal toward compliance under ablation).
        refusal_delta: refusal_logit_ablated - refusal_logit_baseline (raw,
            diagnostic-only — confounded by norm perturbations).
        compliance_delta: compliance_logit_ablated - compliance_logit_baseline.
        per_prompt: optional [n_prompts] tensor of per-prompt effect_signed.
    """
    effect_signed: float          # primary metric — mean across prompts
    refusal_delta: float          # secondary / diagnostic
    compliance_delta: float       # secondary / diagnostic
    n_prompts: int
    per_prompt: torch.Tensor | None = None  # [n_prompts] effect_signed
    refusal_per_prompt: torch.Tensor | None = None
    compliance_per_prompt: torch.Tensor | None = None


def verify_template_boundary(bundle: ModelBundle, sample_prompt: str = "Hello") -> dict:
    """Sanity check: confirm the templated prompt's last token is at the
    boundary where the model's next prediction IS the first response token.

    Each model's chat template has its own end-of-template suffix — Gemma
    is `<start_of_turn>model\\n`, Qwen ChatML is
    `<|im_start|>assistant\\n`. The last tokenized position should be one
    of the tokens of whichever suffix the bundle's tokenizer emits. If
    the template trails into something else (e.g., a stray newline),
    `logits[:, -1, :]` predicts the wrong thing.

    Returns a diagnostic dict — the caller eyeballs `last_5_decoded` to
    confirm the suffix looks right for the model.
    """
    templated = format_prompt_for_bundle(bundle, sample_prompt)
    ids = tokenize_prompt(bundle, templated)
    last_5 = ids[0, -5:].tolist()
    decoded = [bundle.model.tokenizer.decode([t]) for t in last_5]

    # Run a tiny baseline forward to confirm the top-1 first-token prediction
    # is something content-shaped (not a special token), which signals the
    # template boundary is in the right place.
    with torch.no_grad():
        logits = bundle.model(ids, return_type="logits")
    top1_id = int(logits[0, -1, :].argmax().item())
    top1_decoded = bundle.model.tokenizer.decode([top1_id])

    return {
        "last_5_token_ids": last_5,
        "last_5_decoded": decoded,
        "top1_first_response_token_id": top1_id,
        "top1_first_response_token_decoded": top1_decoded,
    }


@torch.no_grad()
def causal_effect_under_hook(
    bundle: ModelBundle,
    prompts: list[str],
    refusal_token_ids: list[int],
    compliance_token_ids: list[int],
    hook_ctx=None,  # context manager (e.g., ablate_dir(model, d_hat)) or None for baseline
    return_per_prompt: bool = True,
) -> dict:
    """Compute first-response-token logit-difference metrics under an active
    ablation hook (or no hook for baseline).

    Same code path is used for baseline, causal direction, inert directions,
    AND random vectors — no path difference between cells. Critical so the
    null band and the causal cells are on the same scale.

    Returns dict with refusal_logit_mean, compliance_logit_mean, and
    optionally per-prompt tensors.
    """
    device = bundle.model.cfg.device
    refusal_idx = torch.tensor(refusal_token_ids, dtype=torch.long, device=device)
    compliance_idx = torch.tensor(compliance_token_ids, dtype=torch.long, device=device)

    refusal_per_prompt = []
    compliance_per_prompt = []

    if hook_ctx is None:
        # No-hook baseline — use nullcontext()
        from contextlib import nullcontext
        hook_ctx = nullcontext()

    with hook_ctx:
        for raw_prompt in prompts:
            templated = format_prompt_for_bundle(bundle, raw_prompt)
            ids = tokenize_prompt(bundle, templated).to(device)
            logits = bundle.model(ids, return_type="logits")  # [1, seq, vocab]
            last_logits = logits[0, -1, :]  # [vocab] — first-response-token distribution
            # logit-difference units (refinement 1): mean of selected logits.
            r_logit = last_logits[refusal_idx].mean().detach().float().cpu()
            c_logit = last_logits[compliance_idx].mean().detach().float().cpu()
            refusal_per_prompt.append(r_logit)
            compliance_per_prompt.append(c_logit)

    r_t = torch.stack(refusal_per_prompt)
    c_t = torch.stack(compliance_per_prompt)
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
    """Top-level function: returns CausalEffect summarizing direction's
    causal signature against an optional precomputed baseline.

    Pass direction=None for the baseline (no hook).
    Pass an existing baseline dict (from a previous baseline call) to reuse;
    otherwise computes it inline (which is wasteful for batched calls).

    The same `compute_causal_effect` call is used for: the causal d̂,
    bootstrap-LDA inert cells, L3 d̂, the LDA top-5 subspace, AND each
    random unit vector. No path differences.
    """
    if baseline is None:
        baseline = causal_effect_under_hook(
            bundle, prompts, refusal_token_ids, compliance_token_ids,
            hook_ctx=None, return_per_prompt=True,
        )

    if direction is None:
        # This call is just the baseline; effect = 0 by definition.
        return CausalEffect(
            effect_signed=0.0,
            refusal_delta=0.0,
            compliance_delta=0.0,
            n_prompts=baseline["n_prompts"],
            per_prompt=torch.zeros(baseline["n_prompts"]),
            refusal_per_prompt=baseline["refusal_per_prompt"],
            compliance_per_prompt=baseline["compliance_per_prompt"],
        )

    from src.directions import ablate_dir
    hook = ablate_dir(bundle.model, direction)
    abl = causal_effect_under_hook(
        bundle, prompts, refusal_token_ids, compliance_token_ids,
        hook_ctx=hook, return_per_prompt=True,
    )

    r_per_prompt = abl["refusal_per_prompt"] - baseline["refusal_per_prompt"]
    c_per_prompt = abl["compliance_per_prompt"] - baseline["compliance_per_prompt"]
    effect_per_prompt = r_per_prompt - c_per_prompt  # signed, negative = causal

    return CausalEffect(
        effect_signed=float(effect_per_prompt.mean()),
        refusal_delta=float(r_per_prompt.mean()),
        compliance_delta=float(c_per_prompt.mean()),
        n_prompts=abl["n_prompts"],
        per_prompt=effect_per_prompt,
        refusal_per_prompt=r_per_prompt,
        compliance_per_prompt=c_per_prompt,
    )
