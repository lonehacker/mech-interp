"""Unit tests for mech_security.directions math.

Pure synthetic data with closed-form expected answers. If these fail, the
diff-of-means / projection / normalization arithmetic is broken and every
result above it is suspect.

Run:
    python -m pytest tests/test_directions.py -v
"""

import sys
from pathlib import Path


import pytest
import torch

from mech_security.directions import (
    bypass_gap,
    diff_of_means,
    diffmeans_subspace,
    lda_directions,
    project,
    project_out_subspace,
    random_orthonormal,
    random_unit_vector,
    unit,
)


def _two_clusters(n=60, d=16, seed=0, scale=0.3, sep=4.0):
    torch.manual_seed(seed)
    h = torch.randn(n, d) * scale
    h[:, 0] += sep
    l = torch.randn(n, d) * scale
    l[:, 0] -= sep
    return h, l


class TestDiffmeansSubspace:
    """The corrected k-sweep instrument: row 0 is EXACTLY diff-of-means (so k=1 == single-direction
    headline), rows 1..k-1 are orthogonal additions. Distinct from lda_directions, whose k=1 differs."""

    def test_k1_is_exactly_diff_of_means(self):
        h, l = _two_clusters()
        dirs = diffmeans_subspace(h, l, k=1)
        assert dirs.shape == (1, h.shape[1])
        assert torch.allclose(dirs[0], unit(diff_of_means(h, l)).float(), atol=1e-6)

    def test_d1_override_pins_row0_exactly(self):
        # passing the cached headline d̂ must reproduce it byte-for-byte as row 0 (the run_attack guarantee)
        h, l = _two_clusters()
        d_hat = unit(diff_of_means(h, l)).float()
        dirs = diffmeans_subspace(h, l, k=4, d1=d_hat)
        assert torch.allclose(dirs[0], d_hat, atol=1e-6)

    def test_orthonormal_rows_for_k_gt_1(self):
        h, l = _two_clusters(d=20, seed=1)
        dirs = diffmeans_subspace(h, l, k=5)
        assert dirs.shape == (5, 20)
        gram = dirs @ dirs.T
        assert torch.allclose(gram, torch.eye(5), atol=1e-4)  # unit rows, mutually orthogonal

    def test_extra_dims_orthogonal_to_diff_of_means(self):
        h, l = _two_clusters(d=20, seed=2)
        dirs = diffmeans_subspace(h, l, k=4)
        for i in range(1, 4):
            assert abs(float(torch.dot(dirs[0], dirs[i]))) < 1e-4

    def test_seeded_deterministic(self):
        # bootstrap extra rows are seeded → reproducible (no Date.now/Random surprises across runs)
        h, l = _two_clusters(d=20, seed=3)
        assert torch.allclose(diffmeans_subspace(h, l, k=3, seed=7), diffmeans_subspace(h, l, k=3, seed=7))

    def test_extra_rows_are_diffmeans_not_pca(self):
        # the extra rows come from bootstrap diff-of-means, so different seeds give different extra rows
        # (a PCA/SVD construction would be seed-invariant). Row 0 (the headline) stays identical.
        h, l = _two_clusters(d=20, seed=4)
        a, b = diffmeans_subspace(h, l, k=3, seed=1), diffmeans_subspace(h, l, k=3, seed=2)
        assert torch.allclose(a[0], b[0], atol=1e-6)            # headline identical
        assert not torch.allclose(a[1], b[1])                   # bootstrap extra rows differ by seed


class TestProjectOutSubspace:
    """The vectorized subspace ablation must equal the old per-direction loop for orthonormal dirs
    (that equivalence is why the O(1)-in-k matmul is a safe speedup, not a behavior change)."""

    def test_matches_sequential_loop_and_removes_span(self):
        torch.manual_seed(0)
        D = random_orthonormal(16, k=3, seed=1)
        x = torch.randn(5, 16)
        out = project_out_subspace(x, D)
        assert torch.allclose(out @ D.T, torch.zeros(5, 3), atol=1e-5)   # no component left in span(D)
        seq = x.clone()                                                  # the old sequential ablation
        for i in range(3):
            d = D[i]
            seq = seq - (seq * d).sum(-1, keepdim=True) * d
        assert torch.allclose(out, seq, atol=1e-5)

    def test_leaves_orthogonal_component_untouched(self):
        D = random_orthonormal(8, k=2, seed=2)
        x = torch.randn(8)
        x = x - (x @ D.T) @ D                       # x already ⟂ span(D)
        assert torch.allclose(project_out_subspace(x, D), x, atol=1e-5)


class TestRandomOrthonormal:
    """Matched-k random-subspace control: k orthonormal vectors, seeded + reproducible."""

    def test_shape_and_orthonormal(self):
        dirs = random_orthonormal(64, k=3, seed=0)
        assert dirs.shape == (3, 64)
        assert torch.allclose(dirs @ dirs.T, torch.eye(3), atol=1e-5)

    def test_reproducible_and_seed_sensitive(self):
        assert torch.allclose(random_orthonormal(32, 2, seed=7), random_orthonormal(32, 2, seed=7))
        assert not torch.allclose(random_orthonormal(32, 2, seed=7), random_orthonormal(32, 2, seed=8))


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


class TestLdaDirections:
    """lda_directions is a peer to diff_of_means as an extraction primitive."""

    def test_k1_unit_norm(self):
        torch.manual_seed(0)
        n, d = 60, 16
        h = torch.randn(n, d) * 0.3
        h[:, 0] += 4.0
        l = torch.randn(n, d) * 0.3
        l[:, 0] -= 4.0
        dirs = lda_directions(h, l, k=1)
        assert dirs.shape == (1, d)
        assert abs(dirs[0].norm().item() - 1.0) < 1e-5

    def test_k1_recovers_separation_axis(self):
        """Top LDA direction on a two-cluster setup should align with axis 0."""
        torch.manual_seed(0)
        n, d = 80, 16
        h = torch.randn(n, d) * 0.2
        h[:, 0] += 5.0
        l = torch.randn(n, d) * 0.2
        l[:, 0] -= 5.0
        dirs = lda_directions(h, l, k=1)
        # Either +x or -x; magnitude of x-component dominates
        assert abs(dirs[0, 0].item()) > 0.9

    def test_orthogonal_when_k_gt_1(self):
        torch.manual_seed(1)
        n, d = 80, 20
        h = torch.randn(n, d) * 0.5
        h[:, 0] += 4.0
        h[:, 1] += 2.0
        l = torch.randn(n, d) * 0.5
        l[:, 0] -= 4.0
        l[:, 1] -= 2.0
        dirs = lda_directions(h, l, k=3)
        assert dirs.shape == (3, d)
        # Each pair orthogonal
        for i in range(3):
            for j in range(i + 1, 3):
                cos = float(torch.dot(dirs[i], dirs[j]))
                assert abs(cos) < 1e-4, f"dirs[{i}] · dirs[{j}] = {cos}"

    def test_bootstrap_seed_changes_direction(self):
        """Bootstrap resampling should perturb the recovered direction."""
        torch.manual_seed(0)
        n, d = 40, 8
        h = torch.randn(n, d) * 0.3
        h[:, 0] += 3.0
        l = torch.randn(n, d) * 0.3
        l[:, 0] -= 3.0
        d_plain = lda_directions(h, l, k=1)
        d_boot = lda_directions(h, l, k=1, bootstrap_seed=42)
        # Different bootstrap → different direction (but both still close to axis 0)
        assert not torch.allclose(d_plain, d_boot)


class TestBypassGap:
    """bypass_gap is the causal-effect measurement that drives Phase 2
    bypass-gap-layer-selection. These tests use stub objects to verify the
    contract (counts, gap arithmetic, scorer override, baseline reuse) without
    a real model load."""

    class _StubModel:
        def hooks(self, fwd_hooks=None):
            from contextlib import contextmanager

            @contextmanager
            def _noop():
                yield
            return _noop()

        @property
        def cfg(self):
            class _Cfg:
                n_layers = 12
                d_model = 8
                device = "cpu"
                dtype = torch.float32
            return _Cfg()

    class _StubBundle:
        def __init__(self):
            self.model = TestBypassGap._StubModel()

    def test_arithmetic_and_scorer_override(self, monkeypatch):
        prompts = ["p1", "p2", "p3", "p4"]
        # Baseline scorer says 1.0; ablated says 0.25 → gap = 0.75
        baselines = ["REFUSED", "REFUSED", "REFUSED", "REFUSED"]
        ablations = ["sure", "here", "REFUSED", "ok"]
        # Patch generate so it returns predictable strings
        calls = {"n": 0}
        def fake_generate(bundle, prompt, max_new_tokens=160, temperature=0.0):
            calls["n"] += 1
            return baselines[calls["n"] - 1] if calls["n"] <= 4 else ablations[calls["n"] - 5]
        monkeypatch.setattr("mech_security.directions.generate", fake_generate, raising=False)
        # Also patch ablate_dir to be a no-op (we don't have a real model)
        from contextlib import contextmanager
        @contextmanager
        def noop_ablate(*a, **k):
            yield
        monkeypatch.setattr("mech_security.directions.ablate_dir", noop_ablate, raising=False)

        # NOTE: with monkeypatch on mech_security.directions.generate, calling from inside
        # bypass_gap (which imports generate lazily inside the function) won't
        # pick up the patch. The function does `from mech_security.model import generate`
        # locally — we patch that instead.
        import mech_security.model
        monkeypatch.setattr(mech_security.model, "generate", fake_generate)

        bundle = self._StubBundle()
        d = torch.zeros(8)

        def scorer(gens):
            return sum(g == "REFUSED" for g in gens) / len(gens)

        out = bypass_gap(bundle, d, prompts, max_new_tokens=8, scorer=scorer)
        assert out["baseline_refusal"] == 1.0
        assert out["ablated_refusal"] == 0.25
        assert out["gap"] == 0.75
        assert out["baseline_completions"] == baselines
        assert out["ablated_completions"] == ablations

    def test_baseline_reuse_skips_regeneration(self, monkeypatch):
        prompts = ["p1", "p2"]
        # Track calls — if baseline_completions is provided we should call gen
        # only n_prompts times (for the ablation), not 2*n.
        calls = {"n": 0}
        def fake_generate(bundle, prompt, max_new_tokens=160, temperature=0.0):
            calls["n"] += 1
            return "compliant"
        import mech_security.model
        monkeypatch.setattr(mech_security.model, "generate", fake_generate)
        from contextlib import contextmanager
        @contextmanager
        def noop_ablate(*a, **k):
            yield
        monkeypatch.setattr("mech_security.directions.ablate_dir", noop_ablate, raising=False)

        bundle = self._StubBundle()
        d = torch.zeros(8)
        precomputed = ["REFUSED", "REFUSED"]
        out = bypass_gap(bundle, d, prompts, baseline_completions=precomputed,
                         scorer=lambda gs: sum(g == "REFUSED" for g in gs) / len(gs))
        assert calls["n"] == 2  # only ablation gens, baseline reused
        assert out["baseline_completions"] is precomputed
