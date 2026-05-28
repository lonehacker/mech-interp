"""Unit tests for src.directions math.

Pure synthetic data with closed-form expected answers. If these fail, the
diff-of-means / projection / normalization arithmetic is broken and every
result above it is suspect.

Run:
    python -m pytest tests/test_directions.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch

from src.directions import (
    diff_of_means,
    project,
    random_unit_vector,
    unit,
)


class TestDiffOfMeans:
    def test_simple_orthogonal(self):
        """Harmful all (+1, 0); harmless all (-1, 0). Diff = (2, 0)."""
        h = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
        l = torch.tensor([[-1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]])
        d = diff_of_means(h, l)
        assert torch.allclose(d, torch.tensor([2.0, 0.0]))

    def test_averaging_works(self):
        """Means are computed across rows, not collapsed."""
        h = torch.tensor([[2.0, 0.0], [0.0, 0.0]])  # mean = (1, 0)
        l = torch.tensor([[0.0, 2.0], [0.0, 0.0]])  # mean = (0, 1)
        d = diff_of_means(h, l)
        assert torch.allclose(d, torch.tensor([1.0, -1.0]))

    def test_dim_mismatch_raises(self):
        h = torch.zeros(3, 4)
        l = torch.zeros(3, 5)
        with pytest.raises(ValueError):
            diff_of_means(h, l)

    def test_d_model_preserved(self):
        h = torch.randn(7, 16)
        l = torch.randn(11, 16)
        d = diff_of_means(h, l)
        assert d.shape == (16,)


class TestUnit:
    def test_normalizes_to_one(self):
        v = torch.tensor([3.0, 4.0])
        u = unit(v)
        assert abs(u.norm().item() - 1.0) < 1e-6
        # direction preserved
        assert torch.allclose(u, torch.tensor([0.6, 0.8]))

    def test_zero_vector_raises(self):
        with pytest.raises(ValueError):
            unit(torch.zeros(5))


class TestProject:
    def test_aligned_projection(self):
        """Project (3, 0) onto unit (1, 0) = scalar 3."""
        acts = torch.tensor([[3.0, 0.0]])
        d_hat = torch.tensor([1.0, 0.0])
        out = project(acts, d_hat)
        assert torch.allclose(out, torch.tensor([3.0]))

    def test_orthogonal_projection_is_zero(self):
        acts = torch.tensor([[0.0, 5.0]])
        d_hat = torch.tensor([1.0, 0.0])
        out = project(acts, d_hat)
        assert torch.allclose(out, torch.tensor([0.0]))

    def test_batch_projection(self):
        acts = torch.tensor([[3.0, 0.0], [0.0, 7.0], [1.0, 1.0]])
        d_hat = torch.tensor([1.0, 0.0])
        out = project(acts, d_hat)
        assert torch.allclose(out, torch.tensor([3.0, 0.0, 1.0]))


class TestRandomUnitVector:
    def test_norm_is_one(self):
        v = random_unit_vector(1024, seed=0)
        assert abs(v.norm().item() - 1.0) < 1e-5

    def test_reproducible(self):
        v1 = random_unit_vector(64, seed=42)
        v2 = random_unit_vector(64, seed=42)
        assert torch.allclose(v1, v2)

    def test_different_seeds_differ(self):
        v1 = random_unit_vector(64, seed=0)
        v2 = random_unit_vector(64, seed=1)
        assert not torch.allclose(v1, v2)


class TestRoundtrip:
    """End-to-end pipeline: synthetic clusters → diff_of_means → unit →
    project. The cluster separation should be recovered cleanly."""

    def test_two_cluster_recovery(self):
        torch.manual_seed(0)
        # Two clusters separated along axis 0, noise on other dims
        n = 50
        d = 10
        h = torch.randn(n, d) * 0.1
        h[:, 0] += 5.0  # harmful cluster centered at x=5
        l = torch.randn(n, d) * 0.1
        l[:, 0] -= 5.0  # harmless cluster centered at x=-5

        d_hat = unit(diff_of_means(h, l))
        proj_h = project(h, d_hat)
        proj_l = project(l, d_hat)

        # All harmful projections > all harmless projections
        assert proj_h.min().item() > proj_l.max().item()
