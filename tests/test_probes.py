"""Unit tests for src.probes.

Pure synthetic data with planted-separable structure at a known layer. If
these fail, the per-layer probe / shuffled-control machinery is broken and
every Step 4 number above it is suspect.

Run:
    python -m pytest tests/test_probes.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
import torch

from src.probes import (
    ProbeResult,
    probe_layer_sweep,
    shuffled_control_sweep,
    train_probe,
)


@pytest.fixture
def planted_separable():
    """[n=100, n_layers=4, d_model=16] with layer 2 linearly separable.

    Returns (acts, labels). Layers 0/1/3 are pure noise; layer 2 has a 3-sigma
    cluster shift on the harmful (label=1) half. A logistic probe should
    return ~1.0 at L2 and ~0.5 elsewhere.
    """
    torch.manual_seed(0)
    acts = torch.randn(100, 4, 16)
    acts[:50, 2, :] += 3.0
    labels = torch.tensor([1] * 50 + [0] * 50, dtype=torch.long)
    return acts, labels


class TestTrainProbe:
    def test_separable_signal_gives_high_acc(self, planted_separable):
        acts, labels = planted_separable
        r = train_probe(acts[:, 2, :], labels, seed=0)
        assert r.test_acc >= 0.9
        assert r.train_acc >= 0.9

    def test_pure_noise_sits_near_chance(self, planted_separable):
        acts, labels = planted_separable
        r = train_probe(acts[:, 0, :], labels, seed=0)
        # n_test = 25, so 0.5 ± 0.25 is the realistic chance band.
        assert 0.25 <= r.test_acc <= 0.75

    def test_deterministic_under_same_seed(self, planted_separable):
        acts, labels = planted_separable
        r1 = train_probe(acts[:, 2, :], labels, seed=42)
        r2 = train_probe(acts[:, 2, :], labels, seed=42)
        assert r1.test_acc == r2.test_acc
        assert r1.train_acc == r2.train_acc

    def test_test_size_partitions_correctly(self, planted_separable):
        acts, labels = planted_separable
        r = train_probe(acts[:, 2, :], labels, seed=0, test_size=0.25)
        assert r.n_train + r.n_test == 100
        # Stratified 25/75 on n=100 with balanced labels.
        assert r.n_test == 25
        assert r.n_train == 75

    def test_shape_mismatch_raises(self):
        acts = torch.randn(10, 16)
        labels = torch.tensor([1, 0])  # mismatched
        with pytest.raises(ValueError, match="n mismatch"):
            train_probe(acts, labels, seed=0)


class TestProbeLayerSweep:
    def test_finds_separable_layer(self, planted_separable):
        acts, labels = planted_separable
        results = probe_layer_sweep(acts, labels, seed=0)
        assert len(results) == 4
        # Layer index annotations match positional layer index.
        assert [r.layer for r in results] == [0, 1, 2, 3]
        # Layer 2 wins by a wide margin.
        peak = max(range(4), key=lambda L: results[L].test_acc)
        assert peak == 2
        assert results[2].test_acc - max(results[L].test_acc for L in [0, 1, 3]) >= 0.30

    def test_rejects_wrong_rank(self):
        acts_2d = torch.randn(10, 16)
        labels = torch.tensor([1, 0] * 5)
        with pytest.raises(ValueError, match="expected"):
            probe_layer_sweep(acts_2d, labels, seed=0)


class TestShuffledControl:
    def test_shuffled_sits_at_chance(self, planted_separable):
        acts, labels = planted_separable
        results = shuffled_control_sweep(acts, labels, seed=0, shuffle_seed=42)
        # Even at the separable layer, shuffled labels should not be learnable.
        for r in results:
            assert 0.30 <= r.test_acc <= 0.70, (
                f"L{r.layer}: shuffled test_acc={r.test_acc:.3f} not at chance"
            )

    def test_shuffle_seed_changes_outcome(self, planted_separable):
        acts, labels = planted_separable
        r1 = shuffled_control_sweep(acts, labels, seed=0, shuffle_seed=1)
        r2 = shuffled_control_sweep(acts, labels, seed=0, shuffle_seed=2)
        # Different shuffle → different per-layer numbers (at least one layer differs).
        assert any(a.test_acc != b.test_acc for a, b in zip(r1, r2))

    def test_real_far_above_shuffled_at_separable_layer(self, planted_separable):
        acts, labels = planted_separable
        real = probe_layer_sweep(acts, labels, seed=0)
        shuf = shuffled_control_sweep(acts, labels, seed=0, shuffle_seed=42)
        # Step 4 gate: real beats shuffled by ≥ 0.30 at the peak layer.
        assert real[2].test_acc - shuf[2].test_acc >= 0.30
