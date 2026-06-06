"""Cache residual-stream activations at the LAST token of each templated prompt.

See HOW_IT_WORKS.md §3 for why we read at `hook_resid_post` and at the last
token position. Both choices are fixed by the runbook: changing either
introduces a free hyperparameter the rest of the project would have to
confound-audit.

Returns are CPU fp32 tensors — Phase 1 caches are small (n × 2304 floats
≈ MB) and downstream math (sklearn, plotting) lives on CPU.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from functools import partial

import torch
from tqdm.auto import tqdm

from .model import ModelBundle, format_prompt_for_bundle, tokenize_prompt


def _resid_hook_name(layer: int) -> str:
    return f"blocks.{layer}.hook_resid_post"


@torch.no_grad()
def _cache_at_hooks(
    bundle: ModelBundle,
    prompts: Iterable[str],
    hook_names: list[str],
    *,
    apply_template: bool,
    show_progress: bool,
    format_fn: Callable[[str], str] | None,
    desc: str,
    position: int = -1,
) -> list[torch.Tensor]:
    """Run a forward pass per prompt; return the activation at `position` per hook.

    position is the (negative-indexed) sequence position to read; default -1
    is the last token. Phase 2 bypass-gap-layer-selection sweeps positions
    -4..-1 to cover the Qwen ChatML end-of-instruction tokens.

    Returns a list of length n_prompts, each entry [n_hooks, d_model] on CPU fp32.
    """
    if format_fn is None:
        # Model-correct chat template (Gemma→Gemma, Qwen→ChatML, Llama-3→Llama-3), resolved from the
        # bundle — the SAME path scoring (model.generate→format_prompt_for_bundle) uses. NEVER hardcode
        # Gemma here: that silently mis-templated every non-Gemma extraction (2026-05-31 Phase-3 bug).
        format_fn = partial(format_prompt_for_bundle, bundle)
    iterator = tqdm(list(prompts), desc=desc) if show_progress else prompts
    out: list[torch.Tensor] = []
    for raw in iterator:
        if apply_template:
            text = format_fn(raw)
            ids = tokenize_prompt(bundle, text)
        else:
            ids = bundle.model.to_tokens(raw)
        _, cache = bundle.model.run_with_cache(ids, names_filter=hook_names, return_type=None)
        per_hook = torch.stack(
            [cache[h][0, position].detach().to("cpu").float() for h in hook_names], dim=0
        )  # [n_hooks, d_model]
        out.append(per_hook)
    return out


@torch.no_grad()
def cache_resid(
    bundle: ModelBundle,
    prompts: Iterable[str],
    layer: int,
    apply_template: bool = True,
    show_progress: bool = True,
    format_fn: Callable[[str], str] | None = None,
    position: int = -1,
) -> torch.Tensor:
    """Cache resid_post activations at ONE layer, at one sequence position.

    Returns [n_prompts, d_model] on CPU. position defaults to -1 (last token)
    — set to e.g. -4 for the Phase 2 bypass-gap-layer-selection position sweep.

    format_fn defaults to Gemma `format_prompt`. For Qwen / Phase 2 / any
    non-Gemma model, pass `lambda m: format_prompt_for_bundle(bundle, m)`.
    """
    if layer < 0 or layer >= bundle.n_layers:
        raise ValueError(f"layer {layer} out of range [0, {bundle.n_layers})")
    desc = f"cache L{layer}" if position == -1 else f"cache L{layer} pos{position}"
    per_hook_list = _cache_at_hooks(
        bundle, prompts, [_resid_hook_name(layer)],
        apply_template=apply_template, show_progress=show_progress,
        format_fn=format_fn, desc=desc, position=position,
    )
    # squeeze hook dim — only 1 hook
    return torch.stack([p[0] for p in per_hook_list], dim=0)


@torch.no_grad()
def cache_resid_all_layers(
    bundle: ModelBundle,
    prompts: Iterable[str],
    apply_template: bool = True,
    show_progress: bool = True,
    format_fn: Callable[[str], str] | None = None,
) -> torch.Tensor:
    """Same as cache_resid but caches every layer in one forward pass.

    Returns [n_prompts, n_layers, d_model] on CPU. Used by the layer-sweep
    experiment so each prompt is forwarded once instead of n_layers times.
    """
    hook_names = [_resid_hook_name(L) for L in range(bundle.n_layers)]
    per_hook_list = _cache_at_hooks(
        bundle, prompts, hook_names,
        apply_template=apply_template, show_progress=show_progress,
        format_fn=format_fn, desc="cache all layers",
    )
    return torch.stack(per_hook_list, dim=0)  # [n, n_layers, d_model]
