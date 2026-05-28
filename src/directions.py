"""
Refusal-direction extraction + ablation/addition hooks.

Method: difference-in-means refusal direction (Arditi et al., "Refusal in
Language Models Is Mediated by a Single Direction"). One vector, one layer.

Two interventions, both as context managers (so a `with` block guarantees the
hooks are removed even on exception):

- ablate_dir: project the residual stream onto the orthogonal complement of
  d_hat at EVERY residual hook (incl. attn_out, mlp_out for the faithful
  variant). Expected effect on harmful prompts: refusal rate drops.
- add_dir: add coeff * d_hat at the residual stream of a specific layer.
  Expected effect on harmless prompts: over-refusal appears.

NEVER summarize an intervention's effect as positive/negative here. This
module produces tensors. Interpretation happens in notebooks under human eyes.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint


# Residual-affecting hook points we ablate at, for the "faithful" Arditi-style
# variant. Subtracting from hook_resid_post alone leaves the direction
# reachable via the per-component reads — ablating at attn_out and mlp_out as
# well closes that loop.
_FAITHFUL_HOOK_SUFFIXES = ("hook_resid_pre", "hook_resid_mid", "hook_resid_post",
                            "hook_attn_out", "hook_mlp_out")


def diff_of_means(
    harmful_acts: torch.Tensor,
    harmless_acts: torch.Tensor,
) -> torch.Tensor:
    """Compute the (un-normalized) diff-of-means direction.

    Inputs are [n, d_model]; output is [d_model]. Caller normalizes if they
    want d_hat — kept separate so both the raw and the unit vector are
    available for diagnostics.
    """
    if harmful_acts.shape[1] != harmless_acts.shape[1]:
        raise ValueError(
            f"d_model mismatch: harmful {harmful_acts.shape} vs "
            f"harmless {harmless_acts.shape}"
        )
    return harmful_acts.mean(dim=0) - harmless_acts.mean(dim=0)


def unit(vec: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Return vec / ||vec||, with eps guard for the degenerate all-zero case."""
    norm = vec.norm()
    if norm.item() < eps:
        raise ValueError("Cannot normalize a near-zero vector.")
    return vec / norm


def project(acts: torch.Tensor, d_hat: torch.Tensor) -> torch.Tensor:
    """Scalar projection of each activation onto d_hat. Returns [n] tensor."""
    return acts @ d_hat


def _is_residual_hook(name: str) -> bool:
    return any(name.endswith(s) for s in _FAITHFUL_HOOK_SUFFIXES)


@contextmanager
def ablate_dir(
    model: HookedTransformer,
    d_hat: torch.Tensor,
    layers: list[int] | None = None,
) -> Iterator[None]:
    """Context manager that ablates the d_hat component from every residual-
    affecting hook for the duration of the block.

    Parameters
    ----------
    layers: if None, ablate at every layer (the standard Arditi recipe). If a
        list, ablate only at those layer indices — used in Step 5 for the
        layer-restricted localization sweep.

    Semantics: at each hook, x ← x - (x @ d_hat) * d_hat. d_hat MUST be a unit
    vector; we assert this rather than re-normalize so an unnormalized input
    is loud, not silent.
    """
    norm = d_hat.norm().item()
    if abs(norm - 1.0) > 1e-3:
        raise ValueError(f"ablate_dir expects a unit vector, got ||d_hat||={norm:.4f}")

    d_hat_dev = d_hat.to(model.cfg.device, dtype=model.W_E.dtype)

    layer_set = set(layers) if layers is not None else None

    def hook_fn(x: torch.Tensor, hook: HookPoint) -> torch.Tensor:
        # x: [..., d_model]. Last dim is d_model regardless of which residual
        # hook we're at.
        coeff = (x * d_hat_dev).sum(dim=-1, keepdim=True)
        return x - coeff * d_hat_dev

    hooks: list[tuple[str, callable]] = []
    for L in range(model.cfg.n_layers):
        if layer_set is not None and L not in layer_set:
            continue
        for suffix in _FAITHFUL_HOOK_SUFFIXES:
            hooks.append((f"blocks.{L}.{suffix}", hook_fn))

    with model.hooks(fwd_hooks=hooks):
        yield


@contextmanager
def add_dir(
    model: HookedTransformer,
    d_hat: torch.Tensor,
    coeff: float,
    layer: int,
    hook_suffix: str = "hook_resid_post",
) -> Iterator[None]:
    """Context manager that adds coeff * d_hat at one layer's residual hook.

    Used to test the second leg of the causal claim: if d_hat really mediates
    refusal, then INJECTING it into the residual on harmless prompts should
    induce refusals.

    coeff units are in residual-stream norm; typical sweeps cover ~0 to 10×
    the harmful-mean projection.
    """
    norm = d_hat.norm().item()
    if abs(norm - 1.0) > 1e-3:
        raise ValueError(f"add_dir expects a unit vector, got ||d_hat||={norm:.4f}")
    if layer < 0 or layer >= model.cfg.n_layers:
        raise ValueError(f"layer {layer} out of range [0, {model.cfg.n_layers})")

    d_hat_dev = d_hat.to(model.cfg.device, dtype=model.W_E.dtype)
    add_vec = coeff * d_hat_dev

    def hook_fn(x: torch.Tensor, hook: HookPoint) -> torch.Tensor:
        return x + add_vec

    hook_name = f"blocks.{layer}.{hook_suffix}"
    with model.hooks(fwd_hooks=[(hook_name, hook_fn)]):
        yield


def random_unit_vector(d_model: int, seed: int, device: str = "cpu") -> torch.Tensor:
    """Generate a random unit vector matching the d_hat shape, for the random-
    direction control (Step 3, Control 1). The matched norm is critical — a
    random vector of *different* norm would test "any perturbation of this
    size", not "any direction at this norm".
    """
    g = torch.Generator(device=device).manual_seed(seed)
    v = torch.randn(d_model, generator=g, device=device)
    return v / v.norm()
