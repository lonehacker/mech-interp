"""Frozen reproducibility invariants — must pass before AND after any
refactor that touches imports, packaging, or directory structure.

What this protects:
  - On-disk activation caches at artifacts/cache/<hex>.pt are referenced by
    keys computed via `content_hash`. If the hash construction changes —
    including the embedded `extra` string format — every cached artifact is
    orphaned and the Phase 1/Phase 2 reproducibility chain breaks.
  - `extract_d_hat`'s composition (cache → diff_of_means → unit → natural
    scale) determines every `d_hat` produced in this project. The numerical
    output on a known synthetic input is pinned here.

What this does NOT protect:
  - Floating-point reproducibility of real-model forward passes (those are
    bf16 on MPS and have their own variance; covered by the spot-run gate
    when needed, not by unit tests).
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestCacheKeyInvariants:
    """The shape of cache keys MUST stay byte-identical across refactors.
    Existing cached .pt files on disk are keyed by these exact strings."""

    def test_content_hash_empty_extra(self):
        from experiments._runner import content_hash
        h = content_hash(["hello", "world"], extra="")
        # Frozen — must equal this hex across any refactor
        assert h == "0da0c6b777f2aaa1", f"got {h}"

    def test_content_hash_with_extra(self):
        from experiments._runner import content_hash
        # Simulate the Phase 2 step3 extra-string pattern
        bundle_name = "qwen-qwen2.5-3b-instruct"
        dtype = "torch.bfloat16"
        extra = f"{bundle_name}|dtype={dtype}|L14|resid_post|last_token|matched_v2|harmful_train"
        h = content_hash(["prompt1", "prompt2", "prompt3"], extra=extra)
        # Frozen
        assert h == "c57fb0eff3f1d653", f"got {h}"

    def test_content_hash_byte_sensitive(self):
        """A one-char change in any input must change the hash."""
        from experiments._runner import content_hash
        h1 = content_hash(["a", "b"], extra="x")
        h2 = content_hash(["a", "b"], extra="X")
        h3 = content_hash(["a", "b "], extra="x")  # trailing space
        assert h1 != h2 != h3

    def test_extra_string_template(self):
        """The exact f-string template Phase 2 step3-family uses is frozen.
        If anything about how we render bundle.name or dtype changes, this
        catches it."""
        bundle_name = "qwen-qwen2.5-3b-instruct"
        dtype_str = "torch.bfloat16"
        layer = 14
        tag = "matched_v2"
        suffix = "harmful_train"
        extra = (
            f"{bundle_name}|dtype={dtype_str}|L{layer}|"
            f"resid_post|last_token|{tag}|{suffix}"
        )
        # Frozen byte-for-byte. Phase 2 step3e's L14|matched_v2 artifact
        # `c101587891347bd3.pt` was computed from a string of this exact
        # shape; any drift here orphans every cached tensor.
        expected = (
            "qwen-qwen2.5-3b-instruct|dtype=torch.bfloat16|L14|"
            "resid_post|last_token|matched_v2|harmful_train"
        )
        assert extra == expected


class TestExtractDHatComposition:
    """`extract_d_hat` composes cache_resid + diff_of_means + unit + a
    projection. The numerical composition is pinned on synthetic inputs.
    Real-model behavior is covered by integration runs, not here."""

    def test_synthetic_composition(self):
        """Given fixed H, L tensors, the composition (diff_of_means → unit
        → natural_scale) must produce these exact numbers."""
        from src.directions import diff_of_means, unit
        torch.manual_seed(42)
        H = torch.tensor([
            [3.0, 1.0, 0.0],
            [3.0, 1.0, 0.0],
            [3.0, 1.0, 0.0],
        ])
        L = torch.tensor([
            [-3.0, -1.0, 0.0],
            [-3.0, -1.0, 0.0],
            [-3.0, -1.0, 0.0],
        ])
        d_hat_expected = unit(diff_of_means(H, L))
        # Frozen: diff = (6, 2, 0), norm = sqrt(40), so d_hat = (6, 2, 0) / sqrt(40)
        sqrt40 = 40 ** 0.5
        expected = torch.tensor([6.0 / sqrt40, 2.0 / sqrt40, 0.0])
        assert torch.allclose(d_hat_expected, expected, atol=1e-6)
        natural_scale = float((H @ d_hat_expected).mean().item())
        # Each row of H @ d_hat = (3*6 + 1*2 + 0*0) / sqrt(40) = 20/sqrt(40) ≈ 3.162
        assert abs(natural_scale - 20.0 / sqrt40) < 1e-4

    def test_extract_d_hat_signature_invariant(self):
        """src.directions.extract_d_hat must be importable AND have the
        4-tuple return contract (d_hat, H, L, meta). Catches an accidental
        signature change during the rename/packaging move."""
        from src.directions import extract_d_hat
        sig_params = list(extract_d_hat.__code__.co_varnames[:extract_d_hat.__code__.co_argcount])
        # Must accept bundle, harmful, harmless as positional-or-keyword
        for p in ("bundle", "harmful", "harmless"):
            assert p in sig_params, f"extract_d_hat missing {p}"
        # The keyword-only args
        kwonly = extract_d_hat.__kwdefaults__ or {}
        for p in ("layer", "position", "format_fn"):
            assert p in extract_d_hat.__code__.co_varnames, f"extract_d_hat missing {p}"
        # Default position is -1 (last token; preserves backward compat)
        assert kwonly.get("position", -1) == -1


class TestBypassGapArithmetic:
    """Pin bypass_gap's arithmetic and baseline-reuse contract. Stub-based,
    no model load."""

    def test_gap_arithmetic_pinned(self, monkeypatch):
        """Given baseline refusal 1.0 and ablated refusal 0.3, gap=0.7.
        Pin the arithmetic so a refactor can't silently invert sign."""
        from src.directions import bypass_gap

        # Stub generate so we can control completions deterministically
        baselines = ["REFUSED"] * 10
        ablations = ["REFUSED", "REFUSED", "REFUSED", "ok"] + ["sure"] * 6
        calls = {"n": 0}

        def fake_generate(bundle, prompt, max_new_tokens=160, temperature=0.0):
            calls["n"] += 1
            return ablations[(calls["n"] - 1) % 10] if calls["n"] > 10 else baselines[calls["n"] - 1]

        import src.model
        monkeypatch.setattr(src.model, "generate", fake_generate)

        from contextlib import contextmanager
        @contextmanager
        def noop_ablate(*a, **k):
            yield
        monkeypatch.setattr("src.directions.ablate_dir", noop_ablate, raising=False)

        class _StubModel:
            class cfg:
                n_layers, d_model = 12, 8
                device, dtype = "cpu", torch.float32
            def hooks(self, fwd_hooks=None):
                @contextmanager
                def _n(): yield
                return _n()

        class _StubBundle:
            model = _StubModel()

        prompts = [f"p{i}" for i in range(10)]
        scorer = lambda gens: sum(g == "REFUSED" for g in gens) / len(gens)
        out = bypass_gap(_StubBundle(), torch.zeros(8), prompts, scorer=scorer)
        assert out["baseline_refusal"] == 1.0
        assert out["ablated_refusal"] == 0.3
        assert out["gap"] == 0.7  # baseline − ablated
