"""Unit tests for the topic-controlled probe — the H-dim vs H-mixture discriminator.

If this is wrong, we can't tell "residual refusal decodable independent of topic" (H-dim) from "probe was
reading topic that correlates with refusal" (H-mixture) — the load-bearing Part-2 distinction.
"""
import numpy as np
import torch

from mech_security.probe_ablation import topic_controlled_probe


def _four_topics(n_per=30, d=16, seed=0):
    g = np.random.default_rng(seed)
    centers = np.array([[5, 5], [5, -5], [-5, 5], [-5, -5]], dtype=float)
    cl = np.repeat(np.arange(4), n_per)
    X = g.standard_normal((n_per * 4, d)) * 0.3
    X[:, :2] += centers[cl]          # topic structure lives in dims 0-1 (what KMeans will cluster on)
    return X, cl, g


class TestTopicControlledProbe:
    def test_topic_independent_signal_stays_high(self):
        # refusal signal on dim 8, INDEPENDENT of topic → decodable across held-out topics → H-dim
        X, cl, g = _four_topics(seed=0)
        sig = g.standard_normal(len(cl))
        X[:, 8] += sig * 3.0
        y = (sig > 0).astype(int)
        r = topic_controlled_probe(torch.tensor(X), y, torch.tensor(X), n_topics=4, seed=0)
        assert r["leave_topic_out_auc_mean"] > 0.9
        assert len(r["skipped_single_class_folds"]) == 0

    def test_topic_confounded_is_undecidable(self):
        # label == topic membership → every held-out topic is single-class → folds skip → cannot decode
        X, cl, g = _four_topics(seed=1)
        y = (cl < 2).astype(int)
        r = topic_controlled_probe(torch.tensor(X), y, torch.tensor(X), n_topics=4, seed=0)
        assert len(r["skipped_single_class_folds"]) >= 3
        assert r["leave_topic_out_auc_mean"] != r["leave_topic_out_auc_mean"]  # nan ⇒ undecidable
