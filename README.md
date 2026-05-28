# mech-security

Mechanistic security analysis of deployed open-weight models. Phase 1 target:
`gemma-2-2b-it` under `transformer_lens`. Method lineage: difference-in-means
refusal direction (Arditi et al.), linear probing, activation patching.

## The project in one sentence

Take a genuinely-deployed open-weight model that people self-host with little
red-teaming, and show — causally, mechanistically, with evaluation rigor — how
and where its safety behavior is computed, how brittle that mechanism is, and
what that implies for anyone deploying it.

## Why this lane

- **Deeper than behavioral evals.** Causal and internal — locating and
  intervening on a mechanism, not "I prompted it and it misbehaved."
- **More real-world than Gemma-only interp.** Phase 2 targets a currently
  widely-deployed Qwen/Llama-class model under-scrutinized at the mechanistic
  level.
- **Original contribution is rigor in evaluation.** Precision/recall,
  distribution shift, failure taxonomy of the safety mechanism.

## The claim the finished artifact should make

> Open-weight model X has its refusal/safety behavior governed by a
> low-dimensional internal representation. It is localized to layers L, can be
> causally suppressed by a single-vector intervention that requires only
> weight access, and that mechanism is this robust / this brittle across these
> distributions. Implication for deployers: Z.

## Method ladder (tier 2, in order)

1. **Linear probing** — train probes on residual activations to detect refusal.
2. **Representation engineering / steering** — diff-of-means refusal direction;
   ablate and amplify it; quantify behavioral change.
3. **Activation patching** — localize where the decision is computed.

Explicitly **out of scope for v1:** training SAEs, standing up a cluster,
serving runtimes.

## Repo layout

```
mech-security/
  README.md
  CLAUDE.md            # agent working agreement
  requirements.txt
  data/
    contrastive.jsonl  # frozen, seeded contrastive pairs (built in Step 1)
  src/
    model.py           # load model + chat-template helper
    activations.py     # cache residual stream by layer/position
    directions.py      # diff-of-means; ablation/addition hooks
    probes.py          # per-layer linear probes
    eval.py            # refusal scoring + coherence check
    controls.py        # random-direction baseline, audits
  notebooks/
    00_trigger.ipynb
    01_layer_sweep.ipynb
    02_steering.ipynb
    03_probes.ipynb
    04_patching.ipynb
  results/
  tests/
  tasks.md             # working checklist; (H) flags human-only items
```

## Reproducing Phase 0

1. Accept `gemma-2-2b-it` license on Hugging Face; set `HF_TOKEN`.
2. `pip install -r requirements.txt`
3. Open `notebooks/00_trigger.ipynb` on a hosted GPU (Colab/Kaggle). Run all.
4. Phase 0 "done" = harmful vs harmless visibly separate along the
   diff-of-means direction.

## Responsible scope

Defensive characterization, not a jailbreak cookbook. We report mechanism,
localization, and robustness. We do not publish operational, copy-paste
harmful-content recipes or optimized attack artifacts. See spec §11.
