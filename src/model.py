"""Model loader + chat-template helpers.

See HOW_IT_WORKS.md §2 for the BOS-handling rationale and §1 for the
device/dtype dispatch. Two gotchas this module makes hard to hit:
  - BOS double-add (template emits BOS + to_tokens prepends BOS).
  - Gemma-2 attention/logit softcapping requires center_unembed=False
    handled by TransformerLens; coherence_check verifies.

Public API:
  load_model(name)                       -> ModelBundle
  format_prompt(user_msg)                -> str  (Gemma hardcoded fast-path)
  format_prompt_for_bundle(bundle, msg)  -> str  (any model; dispatches on tokenizer)
  tokenize_prompt(bundle, text)          -> Tensor[1, seq_len]  (BOS-safe)
  generate(bundle, user_msg, ...)        -> str
  coherence_check(bundle)                -> str  (smoke test after load)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformer_lens import HookedTransformer

DEFAULT_MODEL = "gemma-2-2b-it"

# Gemma instruct chat template, reproduced from the upstream tokenizer config.
# Kept hardcoded so (a) we don't depend on tokenizer.apply_chat_template at
# hot paths and (b) the exact templated string is visible at the call site.
GEMMA_CHAT_TEMPLATE = (
    "<bos><start_of_turn>user\n{user_msg}<end_of_turn>\n"
    "<start_of_turn>model\n"
)


@dataclass(frozen=True)
class ModelBundle:
    model: HookedTransformer
    name: str
    n_layers: int
    d_model: int
    device: str


def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _auto_dtype(device: str, model_name: str | None = None) -> torch.dtype:
    """device → dtype, with one model-name exception.

    Gemma on MPS uses fp16 for byte-identical Phase 1 reproducibility
    (the on-disk activation cache key includes dtype). Everything else on
    MPS uses bf16. CUDA → bf16; CPU → fp32.
    """
    if device == "cuda":
        return torch.bfloat16
    if device == "mps":
        if model_name and "gemma" in model_name.lower():
            return torch.float16
        return torch.bfloat16
    return torch.float32


def load_model(
    name: str = DEFAULT_MODEL,
    device: str | None = None,
    dtype: torch.dtype | None = None,
) -> ModelBundle:
    """Wrap HookedTransformer.from_pretrained in a frozen bundle."""
    if device is None:
        device = _auto_device()
    if dtype is None:
        dtype = _auto_dtype(device, model_name=name)
    model = HookedTransformer.from_pretrained(name, dtype=dtype)
    model.to(device)
    model.eval()
    return ModelBundle(
        model=model, name=name,
        n_layers=model.cfg.n_layers,
        d_model=model.cfg.d_model,
        device=device,
    )


def format_prompt(user_msg: str) -> str:
    """Gemma-only chat template. New code should prefer format_prompt_for_bundle."""
    return GEMMA_CHAT_TEMPLATE.format(user_msg=user_msg)


def format_prompt_for_bundle(bundle: ModelBundle, user_msg: str) -> str:
    """Model-agnostic chat template.

    Gemma → hardcoded GEMMA_CHAT_TEMPLATE (byte-identical Phase 1 path).
    Others → tokenizer.apply_chat_template (Qwen ChatML, Llama-3, Phi-3, …).
    Tokenizer lacks chat_template → falls back to Gemma template.
    """
    if "gemma" in bundle.name.lower():
        return GEMMA_CHAT_TEMPLATE.format(user_msg=user_msg)
    tok = bundle.model.tokenizer
    if getattr(tok, "chat_template", None) is None:
        return GEMMA_CHAT_TEMPLATE.format(user_msg=user_msg)
    return tok.apply_chat_template(
        [{"role": "user", "content": user_msg}],
        tokenize=False, add_generation_prompt=True,
    )


def _templated_text_emits_bos(bundle: ModelBundle, text: str) -> bool:
    """True iff `text` starts with the model's bos_token string."""
    bos = bundle.model.tokenizer.bos_token
    return bool(bos and text.startswith(bos))


def tokenize_prompt(bundle: ModelBundle, text: str) -> torch.Tensor:
    """Tokenize a templated prompt; assert exactly 1 BOS iff template emits one.

    The assertion catches the double-BOS bug (template + prepend_bos both
    firing) at the earliest detectable moment. Skipped when template doesn't
    start with bos_token (e.g. Qwen ChatML), since some tokenizers report a
    bos_token_id that doesn't correspond to a single-BOS-at-start semantic.

    Returns: (1, seq_len) LongTensor on model device.
    """
    ids = bundle.model.to_tokens(text, prepend_bos=False)
    bos_id = bundle.model.tokenizer.bos_token_id
    if bos_id is None or not _templated_text_emits_bos(bundle, text):
        return ids
    n_bos = int((ids == bos_id).sum().item())
    assert n_bos == 1, (
        f"Templated text starts with {bundle.model.tokenizer.bos_token!r} but "
        f"tokenized ids contain {n_bos} BOS (expected 1). Likely double-BOS."
    )
    return ids


def generate(
    bundle: ModelBundle,
    user_msg: str,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    apply_template: bool = True,
) -> str:
    """Greedy generation (temperature=0 → deterministic). Returns the model's
    continuation only (prompt prefix stripped).

    apply_template=False is for the Phase 0 templated-vs-raw sanity check;
    everything else should leave it True.
    """
    text = format_prompt_for_bundle(bundle, user_msg) if apply_template else user_msg
    ids = tokenize_prompt(bundle, text) if apply_template else bundle.model.to_tokens(text)
    with torch.no_grad():
        out = bundle.model.generate(
            ids, max_new_tokens=max_new_tokens, temperature=temperature,
            do_sample=temperature > 0.0, verbose=False,
        )
    full = bundle.model.tokenizer.decode(out[0], skip_special_tokens=False)
    return full.removeprefix(text)


def coherence_check(bundle: ModelBundle, probe_prompt: str = "What is 2 + 2?") -> str:
    """Smoke generation. Run after load_model; gibberish here means
    softcapping/dtype/config is wrong and no activation is trustworthy."""
    return generate(bundle, probe_prompt, max_new_tokens=32, temperature=0.0)
