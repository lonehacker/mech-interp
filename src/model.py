"""
Model loader + chat-template helper for gemma-2-2b-it under transformer_lens.

The two gotchas this module exists to make impossible:

1. BOS double-add. The Gemma chat template emits <bos>. transformer_lens'
   to_tokens prepends BOS by default. If both fire, every activation in the
   experiment is computed at the wrong token positions.
2. Gemma-2 attention/logit softcapping. transformer_lens handles it, but a
   misconfig surfaces as incoherent generations. Always sanity-check by
   generating from a plain prompt before trusting any activation.

Per CLAUDE.md: never feed raw strings to an instruct model. Use format_prompt.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformer_lens import HookedTransformer


DEFAULT_MODEL = "gemma-2-2b-it"

# Gemma instruct chat template. Matches the upstream tokenizer config; reproduced
# here so we don't depend on the tokenizer's apply_chat_template at hot paths,
# and so the exact format is visible at the call site.
#
# Reference: https://huggingface.co/google/gemma-2-2b-it
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
    """Pick the best available device. Order: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _auto_dtype(device: str, model_name: str | None = None) -> torch.dtype:
    """Pick a dtype that's actually well-supported on the chosen device.

    - cuda: bf16 (native on Ampere+, the released gemma-2-2b precision)
    - mps:  fp16 for Gemma family (matches Phase 1 reproducibility — every
            cached activation in artifacts/cache/ was extracted at fp16);
            bf16 otherwise (modern Apple Silicon supports bf16 fine, and
            Qwen2.5 / Llama-3 / Phi-3 are released at bf16).
    - cpu:  fp32 (bf16/fp16 are emulated on CPU and ~10x slower than fp32)

    The model-name dispatch on MPS exists because changing dtype across the
    Phase 1 boundary would invalidate every disk-cached activation (the
    cache key includes dtype). Gemma stays at fp16 forever; new models
    pick bf16 by default and can override via explicit `dtype=` arg to
    `load_model`.
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
    """Load gemma-2-2b-it under transformer_lens with sane defaults.

    Returns a frozen bundle so downstream code can pass model + metadata around
    without re-querying cfg fields.

    Notes
    -----
    - device auto-picks cuda > mps > cpu. Apple Silicon ('mps') is fully
      usable for Phase 0 / Phase 1 on gemma-2-2b-it (~5 GB at fp16). Phase 2
      (~14B) needs a real GPU.
    - dtype auto-picks per device when None — see `_auto_dtype`. Pass an
      explicit dtype only if you know what you're overriding and why.
    """
    if device is None:
        device = _auto_device()
    if dtype is None:
        dtype = _auto_dtype(device, model_name=name)

    model = HookedTransformer.from_pretrained(name, dtype=dtype)
    model.to(device)
    model.eval()

    return ModelBundle(
        model=model,
        name=name,
        n_layers=model.cfg.n_layers,
        d_model=model.cfg.d_model,
        device=device,
    )


def format_prompt(user_msg: str) -> str:
    """Apply the Gemma instruct chat template to a user message.

    The returned string already contains <bos> via the template, so when
    tokenizing pass prepend_bos=False (see tokenize_prompt).

    Gemma-only. For Phase 2 / Qwen / any non-Gemma model, use
    `format_prompt_for_bundle(bundle, user_msg)` which dispatches on the
    tokenizer's own chat template.
    """
    return GEMMA_CHAT_TEMPLATE.format(user_msg=user_msg)


def format_prompt_for_bundle(bundle: ModelBundle, user_msg: str) -> str:
    """Generic chat-template application:

      - Gemma family: keep the hardcoded `GEMMA_CHAT_TEMPLATE` for byte-
        identical Phase 1 reproducibility. apply_chat_template on the
        Gemma tokenizer can differ in subtle whitespace; we don't want to
        invalidate every cached activation.
      - Any other model: use the tokenizer's own `apply_chat_template`
        (Qwen ChatML, Llama-3, Phi-3, …).
      - No chat_template at all: fall back to the Gemma template (rare;
        most instruct tokenizers ship a chat_template).

    Use this for any new code; Phase 2 runners should call this instead of
    `format_prompt`.
    """
    if "gemma" in bundle.name.lower():
        return GEMMA_CHAT_TEMPLATE.format(user_msg=user_msg)
    tok = bundle.model.tokenizer
    chat_template = getattr(tok, "chat_template", None)
    if chat_template is None:
        return GEMMA_CHAT_TEMPLATE.format(user_msg=user_msg)
    return tok.apply_chat_template(
        [{"role": "user", "content": user_msg}],
        tokenize=False,
        add_generation_prompt=True,
    )


def _templated_text_emits_bos(bundle: ModelBundle, text: str) -> bool:
    """True iff the templated `text` already contains the model's BOS
    token at the front. Gemma's template emits `<bos>`; Qwen's ChatML
    template does NOT. The BOS-count assertion in tokenize_prompt has to
    know which regime applies."""
    bos_token = bundle.model.tokenizer.bos_token
    if not bos_token:
        return False
    return text.startswith(bos_token)


def tokenize_prompt(
    bundle: ModelBundle,
    text: str,
) -> torch.Tensor:
    """Tokenize a *templated* prompt without double-prepending BOS.

    transformer_lens' to_tokens default is prepend_bos=True. We force
    prepend_bos=False and then, IFF the template starts with the
    tokenizer's bos_token, assert exactly one BOS in the resulting ids
    (the Gemma case). For models whose template doesn't start with
    bos_token (Qwen ChatML, where the tokenizer's bos_token_id often
    aliases <|im_end|> or another non-BOS marker that legitimately
    appears multiple times), the count is uninterpretable and the check
    is skipped.

    The check exists to catch the Gemma double-BOS bug at its earliest
    detectable moment — silently poisoning all activations otherwise.

    Returns a (1, seq_len) LongTensor on the model's device.
    """
    ids = bundle.model.to_tokens(text, prepend_bos=False)
    bos_id = bundle.model.tokenizer.bos_token_id
    if bos_id is None:
        return ids

    if not _templated_text_emits_bos(bundle, text):
        # Template doesn't start with bos_token; the tokenizer may report a
        # bos_token_id that doesn't correspond to a single-BOS-at-start
        # semantic (Qwen's bos_token is <|im_end|>, which appears legitimately
        # in mid-text). Nothing meaningful to assert.
        return ids

    n_bos = int((ids == bos_id).sum().item())
    assert n_bos == 1, (
        f"Templated text starts with {bundle.model.tokenizer.bos_token!r} but "
        f"tokenized ids contain {n_bos} BOS tokens (expected 1). Either the "
        "chat template was skipped and prepend_bos=False removed the only BOS, "
        "or prepend_bos got flipped back to True (double-BOS bug)."
    )
    return ids


def generate(
    bundle: ModelBundle,
    user_msg: str,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    apply_template: bool = True,
) -> str:
    """Generate from the model. apply_template=False is provided ONLY for the
    Phase 0 sanity check (templated vs raw must produce visibly different
    behavior). All real experiments must run on templated input.

    Uses `format_prompt_for_bundle` so the same call works for Gemma, Qwen,
    or any other tokenizer with a registered chat_template.
    """
    text = format_prompt_for_bundle(bundle, user_msg) if apply_template else user_msg
    if apply_template:
        ids = tokenize_prompt(bundle, text)
    else:
        # Raw path: let transformer_lens decide BOS handling; this is *not*
        # the distribution we study.
        ids = bundle.model.to_tokens(text)

    with torch.no_grad():
        out = bundle.model.generate(
            ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0.0,
            verbose=False,
        )

    # Strip the prompt prefix from the decoded text so callers see only the
    # continuation.
    full = bundle.model.tokenizer.decode(out[0], skip_special_tokens=False)
    if full.startswith(text):
        full = full[len(text):]
    return full


def coherence_check(bundle: ModelBundle, probe_prompt: str = "What is 2 + 2?") -> str:
    """Cheap sanity generation. Run after load_model on a fresh environment;
    if the output is gibberish the softcapping/dtype/config is wrong and no
    activation from this bundle is trustworthy.
    """
    return generate(bundle, probe_prompt, max_new_tokens=32, temperature=0.0)
