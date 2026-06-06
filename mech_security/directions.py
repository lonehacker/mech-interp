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

import numpy as np
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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


def project_out_subspace(x: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    """Remove the component of `x` in span(D): x − (x Dᵀ) D. `D` is [k, d] and MUST be orthonormal —
    then this is the exact orthogonal-complement projection, equal to sequentially projecting out each
    row but in ONE matmul (O(1) in k, not a Python loop). Inputs [..., d] and [k, d]; output [..., d]."""
    return x - (x @ D.transpose(-1, -2)) @ D


def random_unit_vector(d_model: int, seed: int, device: str = "cpu") -> torch.Tensor:
    """Seeded random unit vector for specificity controls.

    Matched norm is the point: a random vector of *different* norm would test
    "any perturbation of this magnitude" rather than "any direction".
    """
    g = torch.Generator(device=device).manual_seed(seed)
    v = torch.randn(d_model, generator=g, device=device)
    return v / v.norm()


def random_orthonormal(d_model: int, k: int, seed: int, device: str = "cpu") -> torch.Tensor:
    """k random ORTHONORMAL vectors [k, d_model] (seeded) — the matched-k random-SUBSPACE control.

    Peer to `random_unit_vector` generalized to k directions: ablating k random orthogonal directions
    tests whether removing *any* k residual dims dents refusal, isolating the effect of the refusal
    subspace specifically. QR of a seeded Gaussian gives orthonormal columns; we return them as rows.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    A = torch.randn(d_model, k, generator=g, device=device)
    Q, _ = torch.linalg.qr(A)            # [d_model, k] with orthonormal columns
    return Q.T.contiguous()              # [k, d_model], each row a unit vector ⟂ the others


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


def lda_directions(
    H: torch.Tensor,
    L: torch.Tensor,
    k: int = 1,
    bootstrap_seed: int | None = None,
) -> torch.Tensor:
    """Top-k orthogonal Linear-Discriminant directions over (H, L).

    Iteratively fits LDA on (H, L), orthogonalizes each new direction against
    the previous ones, and deflates the data matrix along that direction
    before refitting. Returns a [k, d_model] tensor of unit vectors. Peer to
    `diff_of_means(H, L)` as an extraction method.

    `bootstrap_seed` (optional): resample BOTH H and L with replacement before
    fitting — used for bootstrap-stability tests (see Phase 1 hardened-subspace
    runner).
    """
    if bootstrap_seed is not None:
        rng = np.random.default_rng(bootstrap_seed)
        idx_h = rng.choice(len(H), size=len(H), replace=True)
        idx_l = rng.choice(len(L), size=len(L), replace=True)
        H = H[idx_h]
        L = L[idx_l]

    X = torch.cat([H, L], dim=0).numpy().astype(np.float64)
    y = np.array([1] * len(H) + [0] * len(L))
    X_curr = X.copy()
    directions: list[np.ndarray] = []
    for _ in range(k):
        lda = LinearDiscriminantAnalysis(solver="svd").fit(X_curr, y)
        d = lda.coef_[0]
        n = float(np.linalg.norm(d))
        if n < 1e-10:
            break
        d = d / n
        for d_prev in directions:
            d = d - float(np.dot(d, d_prev)) * d_prev
        d = d / (np.linalg.norm(d) + 1e-12)
        directions.append(d)
        X_curr = X_curr - np.outer(X_curr @ d, d)
    return torch.tensor(np.array(directions), dtype=torch.float32)


def diffmeans_subspace(
    H: torch.Tensor,
    L: torch.Tensor,
    k: int = 1,
    d1: torch.Tensor | None = None,
    seed: int = 0,
) -> torch.Tensor:
    """Diff-of-means-ANCHORED orthonormal k-subspace — ONE construction method across the whole sweep.

    Row 0 = the full-set diff-of-means headline d̂ (so k=1 reproduces the single-direction result exactly).
    Rows 1..k-1 = diff-of-means on BOOTSTRAP RESAMPLES of (H, L), each Gram-Schmidt-orthogonalized against
    the kept rows. Every row is therefore itself a diff-of-means direction — deliberately NOT PCA (which
    would be variance, not refusal) and NOT LDA/Fisher (a *different* instrument: Track-3 showed the
    extended-refusal defense resists LDA but falls to diff-of-means, so an LDA sweep must be run + labelled
    SEPARATELY, never folded in here).

    Interpretation of the extra rows: if refusal is ~1-D, the resampled d̂'s are near-parallel to d̂ and
    their orthogonal residuals are sampling noise (≈ random dirs → ablating them adds nothing over the
    matched-k random-subspace control). If refusal is genuinely multi-D, the resamples carry a CONSISTENT
    off-d̂ component and the extra rows capture it. (Above k≈3 this is unmeasurable — capability-damage and
    refusal-dimensionality share the same k-range; see PHASE3_DEVLOG. Use low k only.)

    Seeded (bootstrap) → deterministic. Returns an orthonormal [k', d] tensor. Pass `d1` to pin row 0 to
    the cached headline d̂ — guaranteeing k=1 == the single-direction vector.
    """
    d1 = unit(d1 if d1 is not None else diff_of_means(H, L)).float()
    dirs = [d1]
    if k <= 1:
        return torch.stack(dirs)
    Hf, Lf = H.float(), L.float()
    g = torch.Generator().manual_seed(seed)
    attempts = 0
    while len(dirs) < k and attempts < 50 * k:           # backstop against a degenerate (rank-deficient) set
        attempts += 1
        ih = torch.randint(len(Hf), (len(Hf),), generator=g)   # bootstrap resample (with replacement)
        il = torch.randint(len(Lf), (len(Lf),), generator=g)
        d = diff_of_means(Hf[ih], Lf[il])
        for dp in dirs:                                  # Gram-Schmidt against kept rows
            d = d - (d @ dp) * dp
        n = d.norm()
        if n.item() < 1e-6:
            continue
        dirs.append(d / n)
    return torch.stack(dirs)


@contextmanager
def ablate_subspace(
    model: HookedTransformer,
    dirs: torch.Tensor,
) -> Iterator[None]:
    """Multi-direction Arditi ablation: project the SPAN of `dirs` out of every residual hook.

    `dirs` is [k, d_model] and MUST be ORTHONORMAL (rows unit + mutually orthogonal) — every caller
    builds them that way (diffmeans_subspace / lda_directions / random_orthonormal). Under that
    assumption the orthogonal-complement projection is a single matmul per hook (`project_out_subspace`):
    x ← x − (x Dᵀ) D, which is O(1) in k. The earlier per-direction Python loop made high-k generation
    pathologically slow (k× the per-token hook overhead at every layer). Peer to `ablate_dir`.
    """
    D = torch.stack([_validate_unit_and_prep(dirs[i], model) for i in range(dirs.shape[0])])  # [k, d]
    gram = D @ D.T  # the projection identity holds only for orthonormal dirs — fail loud if not
    if not torch.allclose(gram, torch.eye(D.shape[0], device=D.device, dtype=D.dtype), atol=1e-2):
        raise ValueError("ablate_subspace requires ORTHONORMAL dirs (rows unit + mutually orthogonal)")

    def hook_fn(x: torch.Tensor, hook: HookPoint) -> torch.Tensor:
        return project_out_subspace(x, D)

    hooks = [
        (f"blocks.{L}.{suffix}", hook_fn)
        for L in range(model.cfg.n_layers)
        for suffix in _FAITHFUL_HOOK_SUFFIXES
    ]
    with model.hooks(fwd_hooks=hooks):
        yield


def extract_d_hat(
    bundle,
    harmful: list[str],
    harmless: list[str],
    *,
    layer: int,
    position: int = -1,
    format_fn=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """Canonical diff-of-means extraction at one (layer, position) cell.

    Composes the four operations the per-cell sweep does at every cell:
      1. cache_resid on harmful set at (layer, position) → H [n_h, d_model]
      2. cache_resid on harmless set at (layer, position) → L [n_l, d_model]
      3. d_hat = unit(diff_of_means(H, L))
      4. natural_scale = mean projection of H onto d_hat (the alpha for addition)

    Returns (d_hat, H, L, meta) where meta = {"natural_scale",
    "harmless_mean", "midpoint"}. No caching — runs the forward passes fresh
    on each call. For cached invocations (Phase 2 step3 family that reuses
    activations across runs), see `experiments._runner.cached_extract_d_hat`.

    For an example of how this composes with `bypass_gap` for a per-cell
    sweep, see HOW_IT_WORKS.md §"How the code does the per-cell sweep" or
    `experiments/phase2_part2_dim_bypass_gap_sweep.py`.
    """
    from mech_security.activations import cache_resid
    H = cache_resid(bundle, harmful, layer=layer, position=position,
                    format_fn=format_fn, show_progress=False)
    L = cache_resid(bundle, harmless, layer=layer, position=position,
                    format_fn=format_fn, show_progress=False)
    d_hat = unit(diff_of_means(H, L))
    h_mean = float((H @ d_hat).mean().item())
    l_mean = float((L @ d_hat).mean().item())
    return d_hat, H, L, {
        "natural_scale": h_mean,
        "harmless_mean": l_mean,
        "midpoint": 0.5 * (h_mean + l_mean),
    }


def bypass_gap(
    bundle,
    direction: torch.Tensor,
    prompts: list[str],
    *,
    baseline_completions: list[str] | None = None,
    max_new_tokens: int = 160,
    scorer=None,
) -> dict:
    """Causal-effect-on-refusal measurement: the bypass gap.

    Generates twice (baseline + ablated) over `prompts`; scores both with
    `scorer`; returns the gap.

    The "bypass gap" of a direction = (baseline refusal rate) − (ablated
    refusal rate) on a held-out set. Used by Phase 2 bypass-gap-layer-
    selection: sweep layers, pick the one with the largest gap. Peer to
    `diff_of_means` as an experiment primitive — diff-of-means PRODUCES a
    direction; bypass_gap MEASURES whether that direction is causal.

    Pass `baseline_completions` to avoid the baseline regeneration when
    sweeping many directions over the same test set.

    scorer: callable `list[str] -> float` (refusal rate). Defaults to
    substring `is_refusal`. Pass a Haiku-4.5-judge wrapper for dual-judge.

    Returns:
        {"baseline_refusal": float, "ablated_refusal": float, "gap": float,
         "baseline_completions": list[str], "ablated_completions": list[str],
         "mean_chars_baseline": float, "mean_chars_ablated": float}
    """
    from mech_security.eval import is_refusal
    from mech_security.model import generate

    if scorer is None:
        scorer = lambda gens: sum(is_refusal(g) for g in gens) / len(gens)  # noqa: E731

    if baseline_completions is None:
        baseline_completions = [
            generate(bundle, p, max_new_tokens=max_new_tokens, temperature=0.0).strip()
            for p in prompts
        ]
    baseline_refusal = scorer(baseline_completions)

    with ablate_dir(bundle.model, direction):
        ablated_completions = [
            generate(bundle, p, max_new_tokens=max_new_tokens, temperature=0.0).strip()
            for p in prompts
        ]
    ablated_refusal = scorer(ablated_completions)

    mean_chars = lambda gs: sum(len(g) for g in gs) / max(1, len(gs))  # noqa: E731
    return {
        "baseline_refusal": baseline_refusal,
        "ablated_refusal": ablated_refusal,
        "gap": baseline_refusal - ablated_refusal,
        "baseline_completions": baseline_completions,
        "ablated_completions": ablated_completions,
        "mean_chars_baseline": mean_chars(baseline_completions),
        "mean_chars_ablated": mean_chars(ablated_completions),
    }
