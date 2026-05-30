"""Refusal-direction extraction + ablation/addition hooks.

See HOW_IT_WORKS.md §4 (diff-of-means math) and §6-7 (ablate/add hooks).

Public API:
  diff_of_means(H, L)             -> Tensor[d_model]    H - L cluster centroids
  unit(v)                          -> Tensor[d_model]    v / ||v||, asserts non-zero
  project(acts, d_hat)             -> Tensor[n]          acts @ d_hat
  random_unit_vector(d, seed)      -> Tensor[d]          for specificity controls
  ablate_dir(model, d_hat)         -> contextmanager     all-layer projection-out
  add_dir(model, d_hat, coeff, L)  -> contextmanager     single-layer addition

Function shape: tensors in, tensors / contexts out. NEVER interpret an
intervention's effect as positive/negative here — that belongs in
experiments/ runners under human eyes.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint

# Residual-affecting hook points. The faithful Arditi-style ablation hits
# all five per layer: subtracting at resid_post alone leaves the direction
# reachable through the next block's attention reads (which read resid_pre).
_FAITHFUL_HOOK_SUFFIXES = (
    "hook_resid_pre", "hook_resid_mid", "hook_resid_post",
    "hook_attn_out", "hook_mlp_out",
)


def diff_of_means(harmful_acts: torch.Tensor, harmless_acts: torch.Tensor) -> torch.Tensor:
    """mean(harmful) - mean(harmless). Inputs [n, d_model], output [d_model]."""
    if harmful_acts.shape[1] != harmless_acts.shape[1]:
        raise ValueError(
            f"d_model mismatch: harmful {harmful_acts.shape} vs harmless {harmless_acts.shape}"
        )
    return harmful_acts.mean(dim=0) - harmless_acts.mean(dim=0)


def unit(vec: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """v / ||v||. Raises rather than silently returning zero on a near-zero input."""
    norm = vec.norm()
    if norm.item() < eps:
        raise ValueError("Cannot normalize a near-zero vector.")
    return vec / norm


def project(acts: torch.Tensor, d_hat: torch.Tensor) -> torch.Tensor:
    """Scalar projection: acts @ d_hat. Inputs [n, d], [d]; output [n]."""
    return acts @ d_hat


def random_unit_vector(d_model: int, seed: int, device: str = "cpu") -> torch.Tensor:
    """Seeded random unit vector for specificity controls.

    Matched norm is the point: a random vector of *different* norm would test
    "any perturbation of this magnitude" rather than "any direction".
    """
    g = torch.Generator(device=device).manual_seed(seed)
    v = torch.randn(d_model, generator=g, device=device)
    return v / v.norm()


def _is_residual_hook(name: str) -> bool:
    return any(name.endswith(s) for s in _FAITHFUL_HOOK_SUFFIXES)


def _validate_unit_and_prep(d_hat: torch.Tensor, model: HookedTransformer) -> torch.Tensor:
    """Assert d_hat is unit-norm; move to model device/dtype."""
    norm = d_hat.norm().item()
    if abs(norm - 1.0) > 1e-3:
        raise ValueError(f"expected unit vector, got ||d_hat||={norm:.4f}")
    return d_hat.to(model.cfg.device, dtype=model.W_E.dtype)


@contextmanager
def ablate_dir(
    model: HookedTransformer,
    d_hat: torch.Tensor,
    layers: list[int] | None = None,
) -> Iterator[None]:
    """Project d_hat out of every residual-affecting hook for the `with` block.

    At each hook:  x ← x - (x · d_hat) * d_hat   (orthogonal projection).

    layers=None → ablate at every layer (standard Arditi recipe).
    layers=[L]  → single-layer ablation (used by the localization sweep).

    d_hat MUST be a unit vector; we raise rather than re-normalize so a
    miscalibrated caller is loud, not silent.
    """
    d_hat_dev = _validate_unit_and_prep(d_hat, model)
    layer_set = set(layers) if layers is not None else None

    def hook_fn(x: torch.Tensor, hook: HookPoint) -> torch.Tensor:
        # x: [..., d_model] — last dim is d_model at every residual hook
        coeff = (x * d_hat_dev).sum(dim=-1, keepdim=True)   # [..., 1]
        return x - coeff * d_hat_dev                         # [..., d_model]

    hooks = [
        (f"blocks.{L}.{suffix}", hook_fn)
        for L in range(model.cfg.n_layers)
        if layer_set is None or L in layer_set
        for suffix in _FAITHFUL_HOOK_SUFFIXES
    ]
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
    """Add coeff * d_hat at one layer's residual hook for the `with` block.

    At hook:  x ← x + coeff * d_hat.

    Used to test the bidirectional causal claim: if d_hat mediates refusal,
    INJECTING it into the residual on harmless prompts should induce refusals.

    coeff is in residual-stream-norm units — typical sweeps cover ~0–10×
    the harmful-mean projection at the EXTRACTION layer (NOT the injection
    layer; see HOW_IT_WORKS.md §7).
    """
    if layer < 0 or layer >= model.cfg.n_layers:
        raise ValueError(f"layer {layer} out of range [0, {model.cfg.n_layers})")
    d_hat_dev = _validate_unit_and_prep(d_hat, model)
    add_vec = coeff * d_hat_dev

    def hook_fn(x: torch.Tensor, hook: HookPoint) -> torch.Tensor:
        return x + add_vec

    with model.hooks(fwd_hooks=[(f"blocks.{layer}.{hook_suffix}", hook_fn)]):
        yield
