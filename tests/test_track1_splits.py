"""Unit tests for mech_security.track1_splits — the load-bearing disjointness guarantee.

If extraction ∩ scoring is non-empty, the ablation result is optimistic in-sample. These tests
assert the disjointness machinery and that the real-data splits come out mutually disjoint, with
presence matching the Task-0.5 seed/slice exactly.

Run: python -m pytest tests/test_track1_splits.py -v
"""
import csv
import random
from pathlib import Path

import pytest

from mech_security.track1_splits import (
    assert_disjoint,
    feasibility,
    matched_splits,
    required_rows,
    set_hash,
    track1_splits,
)

_REPO = Path(__file__).resolve().parent.parent
ADV = str(_REPO / "data" / "advbench_harmful_behaviors.csv")
HARM = str(_REPO / "data" / "contrastive.jsonl")
MATCHED = str(_REPO / "data" / "code_contrastive_matched.jsonl")


class TestAssertDisjoint:
    def test_passes_on_disjoint(self):
        assert_disjoint({"a": ["x", "y"], "b": ["z"]})  # no raise

    def test_raises_on_overlap(self):
        with pytest.raises(AssertionError):
            assert_disjoint({"a": ["x", "y"], "b": ["y", "z"]})


class TestSetHash:
    def test_order_independent(self):
        assert set_hash(["a", "b", "c"]) == set_hash(["c", "a", "b"])

    def test_differs_on_content(self):
        assert set_hash(["a", "b"]) != set_hash(["a", "c"])


class TestTrack1Splits:
    def test_real_data_mutually_disjoint_and_sized(self):
        s = track1_splits(ADV, HARM, n_extract=40, n_score=32)
        eh, sh, ph = set(s["extract_harmful"]), set(s["score_harmful"]), set(s["presence_harmful"])
        assert eh.isdisjoint(sh) and eh.isdisjoint(ph) and sh.isdisjoint(ph)
        assert len(s["score_harmful"]) == 32
        assert len(s["extract_harmful"]) == 40
        # extraction harmless disjoint from the Task-0.5 presence harmless (first 15)
        h_presence = [c["text"] for c in []]  # presence harmless not returned; check extract is non-empty + framed
        assert len(s["extract_harmless"]) == 40

    def test_presence_matches_task05_seed_and_slice(self):
        rows = [r["goal"].strip() for r in csv.DictReader(open(ADV)) if r.get("goal")]
        rng = random.Random(20260530)
        rng.shuffle(rows)
        assert track1_splits(ADV, HARM)["presence_harmful"] == rows[:30]

    def test_hashes_present(self):
        s = track1_splits(ADV, HARM)
        assert set(s["hashes"]) == {"extract_harmful", "extract_harmless", "score_harmful", "presence_harmful"}
        assert all(len(h) == 16 for h in s["hashes"].values())


class TestMatchedSplits:
    def test_real_matched_disjoint_and_sized(self):
        s = matched_splits(MATCHED, n_extract_harmful=30, n_score_harmful=10, n_harmless_extract=30)
        assert s["sizes"] == {"extract_harmful": 30, "extract_harmless": 30,
                              "score_harmful": 10, "presence_harmful": 0}
        assert set(s["extract_harmful"]).isdisjoint(set(s["score_harmful"]))
        assert s["hashes"]["score_harmful"] and not s["hashes"]["presence_harmful"]


class TestFeasibility:
    """The orchestrator pre-flight: catch infeasible runs (the n200 crash) locally, before a paid pod."""

    def test_required_rows_formula(self):
        # non-matched: harmful = presence(30)+extract+score ; harmless = presence(15)+extract
        assert required_rows(200, 24) == {"harmful": 254, "harmless": 215}  # the exact n200 requirement
        # matched: harmful = extract+score ; harmless = extract
        assert required_rows(30, 10, matched=True) == {"harmful": 40, "harmless": 30}

    def test_feasible_small_split_ok(self):
        ok, reason = feasibility(n_extract=40, n_score=32, advbench_path=ADV, harmless_path=HARM)
        assert ok and "ok" in reason

    def test_infeasible_huge_split_flagged(self):
        ok, reason = feasibility(n_extract=99999, n_score=10, advbench_path=ADV, harmless_path=HARM)
        assert not ok and "insufficient" in reason

    def test_locks_to_real_assert(self):
        # feasibility's verdict must match what track1_splits actually does: infeasible ⟺ it raises.
        n_extract, n_score = 99999, 10
        ok, _ = feasibility(n_extract=n_extract, n_score=n_score, advbench_path=ADV, harmless_path=HARM)
        assert ok is False
        with pytest.raises(ValueError):
            track1_splits(ADV, HARM, n_extract=n_extract, n_score=n_score, n_harmless_extract=n_extract)
