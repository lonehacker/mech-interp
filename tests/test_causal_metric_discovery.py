"""Unit tests for src.causal_metric token-set discovery helpers.

No model load — these test the pure-Python helpers and the dataclass
roundtrip. The end-to-end discovery against an actual model is implicit
in any phase-2 experiment that calls `get_or_discover_token_sets`.

Run:
    python -m pytest tests/test_causal_metric_discovery.py -v
"""

import json
import sys
from pathlib import Path


import pytest

from mech_security.causal_metric import (
    COMPLIANCE_FIRST_TOKEN_IDS_GEMMA2,
    REFUSAL_FIRST_TOKEN_IDS_GEMMA2,
    VALIDATED_TOKEN_SETS,
    TokenSetDiscovery,
    _greedy_cover,
    load_token_sets,
    save_token_sets,
)


class TestGreedyCover:
    def test_single_dominant_token_covers_threshold(self):
        # 8 'A' + 2 'B' out of 10; threshold 0.7 → A covers 0.8, single pick
        ids = ["A"] * 8 + ["B"] * 2
        sel, freqs, cov = _greedy_cover(ids, threshold=0.7, max_tokens=8)
        assert sel == ["A"]
        assert cov == pytest.approx(0.8)
        assert freqs[0] == ("A", 8)

    def test_picks_multiple_until_threshold_met(self):
        # Three tokens: 5 A, 4 B, 1 C; threshold 0.85
        # A alone = 0.5 (below); A+B = 0.9 (meets); stop with [A, B]
        ids = ["A"] * 5 + ["B"] * 4 + ["C"] * 1
        sel, _, cov = _greedy_cover(ids, threshold=0.85, max_tokens=8)
        assert sel == ["A", "B"]
        assert cov == pytest.approx(0.9)

    def test_max_tokens_cap_stops_early(self):
        # 10 unique tokens, max_tokens=3, threshold high enough to require >3
        ids = [f"T{i}" for i in range(10)]
        sel, _, cov = _greedy_cover(ids, threshold=0.99, max_tokens=3)
        assert len(sel) == 3
        assert cov == pytest.approx(0.3)  # 3/10 even though threshold unmet

    def test_empty_input_returns_zero(self):
        sel, freqs, cov = _greedy_cover([], threshold=0.9, max_tokens=8)
        assert sel == []
        assert freqs == []
        assert cov == 0.0


class TestValidatedSetIntegrity:
    """The Gemma entry is the regression anchor — what discovery should
    output if re-run. If these change, the writeup numbers are stale."""

    def test_gemma_refusal_set_matches_writeup(self):
        g = VALIDATED_TOKEN_SETS["gemma-2-2b-it"]
        assert g.refusal_ids == [235285]  # "I"
        assert g.refusal_coverage == 0.99

    def test_gemma_compliance_set_matches_writeup(self):
        g = VALIDATED_TOKEN_SETS["gemma-2-2b-it"]
        assert g.compliance_ids == [1620, 4858, 1917, 651, 235281, 6750]
        assert g.compliance_coverage == 0.96

    def test_gemma_refusal_compliance_disjoint(self):
        g = VALIDATED_TOKEN_SETS["gemma-2-2b-it"]
        overlap = set(g.refusal_ids) & set(g.compliance_ids)
        assert not overlap, f"Gemma refusal/compliance sets overlap: {overlap}"

    def test_backward_compat_aliases_match(self):
        # Old import sites continue to work; values match the validated entry.
        g = VALIDATED_TOKEN_SETS["gemma-2-2b-it"]
        assert g.refusal_ids == REFUSAL_FIRST_TOKEN_IDS_GEMMA2
        assert g.compliance_ids == COMPLIANCE_FIRST_TOKEN_IDS_GEMMA2


class TestJsonRoundtrip:
    def test_save_and_load_recovers_all_fields(self, tmp_path):
        original = TokenSetDiscovery(
            refusal_ids=[1, 2, 3],
            compliance_ids=[4, 5],
            refusal_coverage=0.95,
            compliance_coverage=0.88,
            refusal_top_decoded=[{"token_id": 1, "decoded": "X", "count": 90, "fraction": 0.9}],
            compliance_top_decoded=[{"token_id": 4, "decoded": "Y", "count": 80, "fraction": 0.8}],
            discovery_n_harmful=100,
            discovery_n_harmless=100,
            model_name="test-model",
            coverage_threshold=0.9,
            max_tokens_per_set=8,
        )
        path = tmp_path / "td.json"
        save_token_sets(original, path)
        loaded = load_token_sets(path)
        assert loaded.refusal_ids == original.refusal_ids
        assert loaded.compliance_ids == original.compliance_ids
        assert loaded.refusal_coverage == original.refusal_coverage
        assert loaded.compliance_coverage == original.compliance_coverage
        assert loaded.refusal_top_decoded == original.refusal_top_decoded
        assert loaded.model_name == original.model_name

    def test_load_tolerates_extra_unknown_fields(self, tmp_path):
        # Forward-compat: a future field on disk shouldn't blow up the loader.
        d = {
            "refusal_ids": [1],
            "compliance_ids": [2],
            "refusal_coverage": 0.5,
            "compliance_coverage": 0.5,
            "refusal_top_decoded": [],
            "compliance_top_decoded": [],
            "discovery_n_harmful": 1,
            "discovery_n_harmless": 1,
            "model_name": "x",
            "coverage_threshold": 0.9,
            "max_tokens_per_set": 8,
            "future_field_we_dont_know_about": "ignored",
        }
        path = tmp_path / "td.json"
        path.write_text(json.dumps(d))
        loaded = load_token_sets(path)
        assert loaded.refusal_ids == [1]
        assert loaded.compliance_ids == [2]
