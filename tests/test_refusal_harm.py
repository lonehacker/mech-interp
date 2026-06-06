"""Unit tests for the refusal-vs-harmfulness decomposition (Stage 0.5).

cos(d_harm, d_refuse) must track how entangled harm-detection and the refusal decision are: ~0 when the
two are separate axes (Llama-worry), ~1 when refusal is just harm-detection (Qwen-like).
"""
import numpy as np
import pytest
import torch

from mech_security.refusal_harm import cos_bootstrap, decompose


def _acts(n=40, d=8, seed=0):
    return np.random.default_rng(seed).standard_normal((n, d)) * 0.2, np.random.default_rng(seed + 99)


class TestDecompose:
    def test_separated_axes_low_cos(self):
        X, g = _acts(seed=0)
        harm, ref = g.standard_normal(len(X)), g.standard_normal(len(X))  # independent signals
        X[:, 0] += harm * 3   # harm on dim 0
        X[:, 1] += ref * 3    # refusal on dim 1, INDEPENDENT of harm
        out = decompose(torch.tensor(X), [h > 0 for h in harm], [1 if r > 0 else 0 for r in ref])
        assert abs(out["cos_harm_refuse"]) < 0.3          # near-orthogonal ⇒ separated

    def test_refusal_equals_harm_high_cos(self):
        X, g = _acts(seed=1)
        harm = g.standard_normal(len(X))
        X[:, 0] += harm * 3
        out = decompose(torch.tensor(X), [h > 0 for h in harm], [1 if h > 0 else 0 for h in harm])  # refuse iff harmful
        assert out["cos_harm_refuse"] > 0.9               # same direction ⇒ entangled

    def test_single_class_behaviour_raises(self):
        X, _ = _acts(seed=2)
        with pytest.raises(ValueError):
            decompose(torch.tensor(X), [i % 2 == 0 for i in range(len(X))], [1] * len(X))  # all refused


class TestCosBootstrap:
    def test_band_brackets_separated_near_zero(self):
        X, g = _acts(n=80, seed=3)
        harm, ref = g.standard_normal(len(X)), g.standard_normal(len(X))
        X[:, 0] += harm * 3
        X[:, 1] += ref * 3
        b = cos_bootstrap(torch.tensor(X), [h > 0 for h in harm], [1 if r > 0 else 0 for r in ref],
                          n_boot=200, seed=0)
        assert b["cos_p2.5"] < 0.4 and b["cos_p97.5"] > b["cos_p2.5"]   # band near 0, has width

    def test_band_high_when_aligned(self):
        X, g = _acts(n=80, seed=4)
        harm = g.standard_normal(len(X))
        X[:, 0] += harm * 3
        b = cos_bootstrap(torch.tensor(X), [h > 0 for h in harm], [1 if h > 0 else 0 for h in harm], n_boot=200)
        assert b["cos_mean"] > 0.85
