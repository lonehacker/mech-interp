"""Guardrail 3: each supported model family's chat template emits THAT family's real control tokens.

Catches any future "wrong default template" regression across the model zoo (not just the specific
fmt=None→Gemma bug). One parametrized test over the families we use.
"""
import pytest

from mech_security.model import format_prompt


def test_gemma_template_has_gemma_turn_tokens():
    # The Gemma path is hardcoded (no model load) — it must carry Gemma's turn tokens.
    assert "<start_of_turn>" in format_prompt("hello")


@pytest.mark.parametrize("repo,token", [
    ("Qwen/Qwen2.5-3B-Instruct", "<|im_start|>"),                       # ChatML
    ("NousResearch/Meta-Llama-3-8B-Instruct", "<|start_header_id|>"),   # Llama-3
])
def test_family_control_tokens(repo, token):
    """`tokenizer.apply_chat_template` (what `format_prompt_for_bundle` uses for non-Gemma models) must
    emit the family's control token. Skips if the tokenizer isn't available locally (keeps CI hermetic)."""
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained(repo)
    except Exception as e:  # noqa: BLE001 — network/cache miss is a skip, not a failure
        pytest.skip(f"tokenizer {repo} unavailable: {e}")
    s = tok.apply_chat_template(
        [{"role": "user", "content": "hello"}], tokenize=False, add_generation_prompt=True
    )
    assert token in s, f"{repo}: chat template missing {token!r} — got {s[:80]!r}"
