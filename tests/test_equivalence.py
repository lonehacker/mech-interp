"""Unit tests for mech_security.equivalence gate logic (no model load).

The pre-registered pass/fail thresholds (PHASE3_PLAN.md Amendment A2) and the spine definition
are load-bearing: if `gate_verdict` is wrong, the gate silently admits a mis-keyed checkpoint
(corrupting every downstream number) or drops a good one. These are pure-logic tests.

Run:
    python -m pytest tests/test_equivalence.py -v
"""

import pytest

from mech_security.equivalence import (
    SPINE_CHECKPOINTS,
    calibrate_ceiling,
    format_chat,
    gate_verdict,
    is_spine,
)


def _row(top1=1.0, cen=0.01, shape_ok=True, abl=True):
    return {
        "logits": {"top1_agreement": top1, "centered_max_abs_diff": cen,
                   "raw_max_abs_diff": cen, "mean_abs_diff": 0.0, "n_prompts": 16},
        "resid": {"resid_shape_ok": shape_ok, "hooks_bite": abl,
                  "resid_shape": [1, 5, 8], "resid_layer": 6},
    }


class TestCalibrateCeiling:
    def test_adds_margin(self):
        assert calibrate_ceiling({"centered_max_abs_diff": 0.05}, 0.15) == pytest.approx(0.20)


class TestGateVerdict:
    def test_pass_clean(self):
        assert gate_verdict(_row(), ceiling=0.2) is True

    def test_pass_at_exact_ceiling(self):
        assert gate_verdict(_row(cen=0.2), 0.2) is True

    def test_fail_on_top1_disagreement(self):
        # even a single prompt disagreeing (15/16) is a FAIL — the decisive mis-key signal
        assert gate_verdict(_row(top1=0.9375), 0.2) is False

    def test_fail_on_centered_over_ceiling(self):
        assert gate_verdict(_row(cen=0.5), 0.2) is False

    def test_fail_on_bad_resid_shape(self):
        assert gate_verdict(_row(shape_ok=False), 0.2) is False

    def test_fail_when_ablation_has_no_effect(self):
        # hooks not biting the right tensor -> generation unchanged -> FAIL
        assert gate_verdict(_row(abl=False), 0.2) is False


class TestIsSpine:
    def test_lat_and_deeprefusal_are_spine(self):
        assert is_spine("LLM-LAT/robust-llama3-8b-instruct")
        assert is_spine("skysys00/Meta-Llama-3-8B-Instruct-DeepRefusal")
        assert len(SPINE_CHECKPOINTS) == 2

    def test_confirmatory_checkpoints_not_spine(self):
        assert not is_spine("samuelsimko/Meta-Llama-3-8B-Instruct-ReFAT")
        assert not is_spine("GraySwanAI/Llama-3-8B-Instruct-RR")
        assert not is_spine("lapisrocks/Llama-3-8B-Instruct-TAR-Refusal")


class TestFormatChat:
    def test_no_chat_template_returns_raw(self):
        class Tok:
            chat_template = None
        assert format_chat(Tok(), "hello") == "hello"

    def test_uses_chat_template_when_present(self):
        class Tok:
            chat_template = "exists"

            def apply_chat_template(self, msgs, tokenize, add_generation_prompt):
                assert tokenize is False and add_generation_prompt is True
                return "TEMPLATED:" + msgs[0]["content"]

        assert format_chat(Tok(), "hi") == "TEMPLATED:hi"
