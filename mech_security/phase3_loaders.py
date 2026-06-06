"""Phase 3 — load a fine-tuned HF checkpoint into TransformerLens on a supported base arch.

The Phase-3 defended checkpoints (LAT, DeepRefusal, ReFAT, Circuit-Breakers, TAR) are full
fine-tunes of meta-llama/Meta-Llama-3-8B-Instruct. TransformerLens keys off the *base* model
name for architecture/config, so the fine-tuned weights are injected via `hf_model=`.

A load that succeeds — and even generates coherent text — can STILL serve wrong activations on
a silent state-dict key mismatch. Phase-3 Task 0 (`experiments/phase3_tl_equivalence_gate.py`)
gates every checkpoint against raw-HF logits before any scoring. This module only LOADS; it
makes no positive/negative claim about a checkpoint (per CLAUDE.md).

Settings note: we pass NO fold_ln / center_* overrides, so TransformerLens defaults apply
(fold_ln=True, center_writing_weights=True, center_unembed=True) — identical to
`mech_security.model.load_model`, so Phase-3 activations stay comparable to Phase-1/2.

Public API:
  load_hf_reference(checkpoint_id, dtype, device) -> (hf_model, tokenizer)
  load_defended_model(checkpoint_id, base=..., hf_model=None, tokenizer=None) -> ModelBundle
"""
from __future__ import annotations

import torch
from transformer_lens import HookedTransformer

from mech_security.model import ModelBundle, _auto_device, _auto_dtype

DEFAULT_BASE = "meta-llama/Meta-Llama-3-8B-Instruct"

# Config-identical ungated mirror, usable as `base` while the gated Meta license is pending.
# The architecture (n_layers/d_model/etc.) is what TransformerLens reads from `base`; the
# actual weights always come from `hf_model`. Switch back to DEFAULT_BASE once the license
# clears so the provenance trail points at the official repo.
UNGATED_BASE_MIRROR = "NousResearch/Meta-Llama-3-8B-Instruct"


def load_hf_reference(checkpoint_id: str, dtype: torch.dtype, device: str):
    """Raw HF AutoModelForCausalLM + tokenizer — the equivalence-gate reference."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # low_cpu_mem_usage avoids HF's default double-allocation (random-init model + loaded state
    # dict ≈ 2x weights in host RAM), which otherwise peaks ~32GB and trips the container's RAM
    # cgroup limit during construction.
    hf = AutoModelForCausalLM.from_pretrained(
        checkpoint_id, torch_dtype=dtype, low_cpu_mem_usage=True
    )
    hf.to(device).eval()
    tok = AutoTokenizer.from_pretrained(checkpoint_id)
    return hf, tok


def load_defended_model(
    checkpoint_id: str,
    base: str = DEFAULT_BASE,
    *,
    device: str | None = None,
    dtype: torch.dtype | None = None,
    hf_model=None,
    tokenizer=None,
    no_processing: bool = False,
) -> ModelBundle:
    """Load `checkpoint_id`'s fine-tuned weights into a HookedTransformer with `base`'s arch.

    Pass a preloaded `hf_model` + `tokenizer` (e.g. the equivalence-gate reference) to avoid
    re-downloading the ~16 GB weights. `base` is the TransformerLens-supported architecture key
    (Meta-Llama-3-8B-Instruct for the Phase-3 models, or `UNGATED_BASE_MIRROR` while the gated
    license is pending — the arch is what matters, the weights come from `hf_model`).

    Returns a `ModelBundle` whose `.name` is the checkpoint id (so artifacts trace to the
    actual defended model, not the base).
    """
    device = device or _auto_device()
    dtype = dtype or _auto_dtype(device, model_name=base)
    if hf_model is None or tokenizer is None:
        loaded_hf, loaded_tok = load_hf_reference(checkpoint_id, dtype, device)
        hf_model = hf_model or loaded_hf
        tokenizer = tokenizer or loaded_tok

    # Reduced-precision note (see PHASE3_PLAN.md + Phase-1/2 Gemma fp16/softcap fixes): in bf16
    # TransformerLens recommends `from_pretrained_no_processing` (skips LN-folding / centering,
    # which accumulate error in low precision) for HF-faithful activations. We default to
    # `from_pretrained` to match the existing harness; the equivalence gate's vanilla calibration
    # is the empirical decider — if vanilla fails top1=1.0, flip `no_processing=True`.
    loader = (
        HookedTransformer.from_pretrained_no_processing if no_processing
        else HookedTransformer.from_pretrained
    )
    model = loader(base, hf_model=hf_model, tokenizer=tokenizer, dtype=dtype)
    model.to(device)
    model.eval()
    return ModelBundle(
        model=model,
        name=checkpoint_id,
        n_layers=model.cfg.n_layers,
        d_model=model.cfg.d_model,
        device=device,
    )
