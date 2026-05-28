"""
Residual-stream caching at the last token position.

The runbook fixes one choice: cache `hook_resid_post` at the LAST token of each
templated prompt. Position is fixed because (a) refusal-direction work in the
literature operates on the final token's residual and (b) varying the position
silently introduces an extra dimension we'd then have to confound-audit.

Function shape: activations in, tensors out. No analysis here.
"""

from __future__ import annotations

from typing import Iterable

import torch
from tqdm.auto import tqdm

from .model import ModelBundle, format_prompt, tokenize_prompt


def _resid_hook_name(layer: int) -> str:
    return f"blocks.{layer}.hook_resid_post"


@torch.no_grad()
def cache_resid(
    bundle: ModelBundle,
    prompts: Iterable[str],
    layer: int,
    apply_template: bool = True,
    show_progress: bool = True,
) -> torch.Tensor:
    """Cache residual-stream activations at the last token of each prompt.

    Parameters
    ----------
    bundle: loaded model + metadata.
    prompts: raw user messages; templated inside this function unless
        apply_template=False (only useful for the Phase 0 sanity check).
    layer: block index; activation read from blocks.{layer}.hook_resid_post.

    Returns
    -------
    Tensor of shape [n_prompts, d_model] on CPU. CPU because Phase 1 caches
    are small and downstream ops (sklearn, plotting) live on CPU.
    """
    if layer < 0 or layer >= bundle.n_layers:
        raise ValueError(f"layer {layer} out of range [0, {bundle.n_layers})")

    hook_name = _resid_hook_name(layer)
    out: list[torch.Tensor] = []

    iterator = tqdm(list(prompts), desc=f"cache L{layer}") if show_progress else prompts

    for raw in iterator:
        text = format_prompt(raw) if apply_template else raw
        ids = tokenize_prompt(bundle, text) if apply_template else bundle.model.to_tokens(text)
        _, cache = bundle.model.run_with_cache(
            ids,
            names_filter=[hook_name],
            return_type=None,
        )
        # cache[hook_name] is [1, seq_len, d_model]; last token is the final
        # position of the templated prompt.
        last = cache[hook_name][0, -1].detach().to("cpu").float()
        out.append(last)

    return torch.stack(out, dim=0)


@torch.no_grad()
def cache_resid_all_layers(
    bundle: ModelBundle,
    prompts: Iterable[str],
    apply_template: bool = True,
    show_progress: bool = True,
) -> torch.Tensor:
    """Same as cache_resid but stacked across every layer in one forward pass.

    Returns
    -------
    Tensor of shape [n_prompts, n_layers, d_model] on CPU. Used for the layer
    sweep (Step 2) so we only forward each prompt once.
    """
    n_layers = bundle.n_layers
    hook_names = [_resid_hook_name(L) for L in range(n_layers)]
    out: list[torch.Tensor] = []

    iterator = tqdm(list(prompts), desc="cache all layers") if show_progress else prompts

    for raw in iterator:
        text = format_prompt(raw) if apply_template else raw
        ids = tokenize_prompt(bundle, text) if apply_template else bundle.model.to_tokens(text)
        _, cache = bundle.model.run_with_cache(
            ids,
            names_filter=hook_names,
            return_type=None,
        )
        per_layer = torch.stack(
            [cache[h][0, -1].detach().to("cpu").float() for h in hook_names],
            dim=0,
        )  # [n_layers, d_model]
        out.append(per_layer)

    return torch.stack(out, dim=0)  # [n, n_layers, d_model]
