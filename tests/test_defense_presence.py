"""Unit tests for mech_security.defense_presence (Task 0.5) pre-registered verdict logic.

Pure-logic tests — no model load. `presence_verdict` encodes the PHASE3 Task-0.5 thresholds; if it
is wrong we either build a headline on a checkpoint that lacks its defense, or wrongly shelve a real one.

Run: python -m pytest tests/test_defense_presence.py -v
"""

import pytest

from mech_security.defense_presence import (
    MIN_HARMLESS_COMPLIED,
    MIN_UPLIFT,
    harmless_complied_rate,
    presence_verdict,
)


class TestHarmlessCompliedRate:
    def test_basic(self):
        assert harmless_complied_rate({"n": 15, "complied": 12, "refused": 3, "partial_comply": 0}) == pytest.approx(0.8)

    def test_empty_safe(self):
        assert harmless_complied_rate({"n": 0, "complied": 0}) == 0.0


class TestPresenceVerdict:
    def _v(self, **kw):
        base = dict(harmful_refusal_vanilla=0.30, harmful_refusal_spine=0.95, harmless_complied_spine=0.93)
        base.update(kw)
        return presence_verdict(**base)

    def test_defense_present(self):
        v = self._v()
        assert v.label == "defense_present"
        assert v.uplift == pytest.approx(0.65)

    def test_not_detectable_when_uplift_small(self):
        # spine ≈ vanilla on harmful → not detectable at baseline (report, not auto-fail)
        assert self._v(harmful_refusal_spine=0.38).label == "not_detectable_at_baseline"

    def test_uplift_exactly_at_threshold_is_present(self):
        # uplift exactly == MIN_UPLIFT, constructed float-clean (spine - 0.0)
        assert presence_verdict(harmful_refusal_vanilla=0.0, harmful_refusal_spine=MIN_UPLIFT,
                                harmless_complied_spine=0.93).label == "defense_present"

    def test_over_refusing_guard_takes_precedence(self):
        # huge harmful uplift BUT it refuses harmless prompts too → over_refusing, not "present"
        v = self._v(harmful_refusal_spine=0.99, harmless_complied_spine=0.40)
        assert v.label == "over_refusing"

    def test_over_refusing_boundary(self):
        assert self._v(harmless_complied_spine=MIN_HARMLESS_COMPLIED).label == "defense_present"
        assert self._v(harmless_complied_spine=MIN_HARMLESS_COMPLIED - 0.01).label == "over_refusing"
