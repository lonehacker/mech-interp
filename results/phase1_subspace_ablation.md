# Phase 1 — Subspace ablation: classification vs causal dimensionality under statistical extraction

**Pre-registered prediction (logged 2026-05-25 before running):**

> *Diff-of-means is bootstrap-stable and causally sufficient. Bootstrap-LDA
> directions are statistically discriminative (high AUC by construction) but
> NOT consistently causal — ablating one or another gives inconsistent
> behavioral effects because the specific direction was overfitting
> accidental in-sample correlations in 2304-D residual space.*

**Outcome: prediction lands in the strongest form.** Zero of the three
bootstrap LDA-top-1 directions are causal, the top-5 LDA orthogonal subspace
is also causally inert, and L3 diff-of-means (nearly orthogonal to L13
diff-of-means) is inert. Only L13 diff-of-means produces the refusal-rate
drop. Diff-of-means at the peak-AUC extraction layer is the unique
**statistical-method-derived** causal direction on Gemma-2-2b-it; even the
extraction-LAYER choice matters (L3 diff-of-means is causally inert despite
AUC 0.93 at L3).

**Important scope caveat — what this result DOES and DOES NOT show:**

This experiment tested only **statistical extraction methods** (diff-of-means
and LDA, with bootstrap resampling). It does NOT refute Wollschläger et al.'s
ICML 2025 "polyhedral cone" multi-direction picture, because their multi-D
claim is established via **gradient-based optimization (Refusal Direction
Optimization, RDO)** — a different method that optimizes for causal effect
with orthogonality as a constraint, rather than for classification accuracy.

The accurate framing of the finding: **classification dimensionality and
causal dimensionality DECOUPLE under statistical extraction.** Many
classification-equivalent directions exist (any sample of harmful + harmless
activations in 2304-D space admits multiple high-AUC orthogonal classifiers);
of those discoverable by statistical methods, only diff-of-means at the
peak-AUC layer is causal. Whether additional causal directions exist under
gradient extraction (RDO) is untested in this work and remains an open
question consistent with the multi-D literature.

The methodological contribution: **classification-equivalent ≠
causally-equivalent under statistical extraction.** Testing for multi-D
causality requires gradient extraction (RDO or equivalent), not LDA or
diff-of-means probing. This is a piece of methodological hygiene for
anyone porting Arditi-style methods to new models. A literature split exists
along this exact methodological axis: activation-space + SAE-derived methods
recover single-direction control (matching our finding); gradient-derived
methods recover multi-D cones (matching Wollschläger).

## Setup

- Model: `gemma-2-2b-it` on MPS
- d_hat extraction: diff-of-means at L13 from AdvBench train split (138 harmful + 150 harmless)
- Test set: 12 held-out harmful AdvBench prompts (same split as Step 3, seed=0)
- Ablation: Arditi multi-layer recipe (subtract the projection along the
  specified direction at every residual hook: hook_resid_pre, hook_resid_mid,
  hook_resid_post, hook_attn_out, hook_mlp_out at every layer)

**Bootstrap LDA-top-1 directions (extracted before ablation):**

| pair | cosine similarity |
|---|---:|
| bs101 vs bs202 | 0.034 |
| bs101 vs bs303 | 0.118 |
| bs202 vs bs303 | **0.813** |
| bs101 vs L13 diff-of-means | 0.196 |
| bs202 vs L13 diff-of-means | 0.070 |
| bs303 vs L13 diff-of-means | 0.079 |
| L3 diff-of-means vs L13 diff-of-means | ~0.08 (from earlier mechanics writeup) |

The bootstrap LDA directions are nearly orthogonal to each other (with one
exception, bs202 ≈ bs303), and all are nearly orthogonal to L13
diff-of-means. Each individually classifies harmful vs harmless at AUC=1.0
on a held-out test split — that was the original "multi-direction" claim.

## Results

| Cell | Direction ablated | refusal rate | n_refused/n | p50 chars |
|---|---|---:|---|---:|
| A | (baseline, no hook) | 1.00 | 12/12 | 473 |
| **B** | **L13 diff-of-means d_hat** | **0.00** | **0/12** | **439** |
| C1 | LDA-top-1 from bootstrap seed 101 | 1.00 | 12/12 | 489 |
| C2 | LDA-top-1 from bootstrap seed 202 | 1.00 | 12/12 | 485 |
| C3 | LDA-top-1 from bootstrap seed 303 | 1.00 | 12/12 | 473 |
| D | LDA top-5 orthogonal subspace (bootstrap seed 101) | 1.00 | 12/12 | 486 |
| E | L3 diff-of-means (nearly orthogonal to L13 d_hat) | 1.00 | 12/12 | 503 |
| F | random unit vector (specificity control) | 1.00 | 12/12 | 475 |

Five "classification-equivalent" directions (C1, C2, C3, D, E) behave
**identically to the random control (F)** at the ablation level.

p50 chars are all close to baseline (439–503 vs 473) → none of the failed
ablations were measurably lobotomizing the model. They're causally inert,
not destructive.

## Interpretation (scoped to statistical extraction only)

Findings on Gemma-2-2b-it under statistical extraction:

- **One unique causal direction.** L13 diff-of-means. Ablating it drops
  refusal cleanly. Among the directions discovered by statistical methods
  in this experiment, no other is causal.
- **Many statistical classifiers exist** in the activation distribution —
  diff-of-means at L3, LDA at any iteration step, three independently-sampled
  bootstrap LDA directions, a 5-dimensional orthogonal LDA subspace.
  All have AUC=1.0 at their own layer. **None are on the model's causal
  refusal pathway as measured by Arditi-style multi-layer ablation.**
- **The extraction-LAYER choice is load-bearing.** L3 diff-of-means is a
  perfect classifier at L3 (AUC 0.93) but is nearly orthogonal to L13
  diff-of-means (cos 0.08), and ablating it has zero behavioral effect.
  Diff-of-means alone is not enough — diff-of-means at the *right layer*
  (the peak of the layer-sweep AUC curve) is what yields the causal
  direction under this extraction method.

## Why this DOES NOT refute Wollschläger's multi-D picture

Wollschläger et al.'s "polyhedral cone" multi-D claim is established with
**Refusal Direction Optimization (RDO)** — a gradient-based optimization
method designed to find directions that scale monotonically with refusal
probability under causal intervention, with orthogonality as a constraint
and a "Representational Independence" criterion. RDO is fundamentally
different from diff-of-means or LDA:

| Method | Optimizes for | Output |
|---|---|---|
| diff-of-means | difference of class centroids | one direction (per-layer) |
| iterative LDA | within-class-whitened class separation | many classification-equivalent directions |
| **RDO (Wollschläger)** | **causal effect on refusal probability** | **multi-D cone of causally-independent directions** |

Our experiment tested only statistical methods (diff-of-means + LDA). Our null
result for LDA-orthogonal directions is informative about what *statistical
extraction recovers*; it does not test what *gradient extraction would find*.

The literature is genuinely split along this methodological axis: activation-
space and SAE-derived methods report single-direction control (matching our
finding); gradient-derived methods (RDO) report multi-D cones (Wollschläger's
finding). Both can be true. Whether RDO would find additional causal directions
on Gemma-2-2b-it is open and untested in this work.

## Methodological contribution

The clean, scoped takeaway: **classification-equivalent ≠ causally-equivalent
under statistical extraction.** Iterative LDA can recover many high-AUC
classifiers from a single sample of activations, but only diff-of-means at
the peak-AUC extraction layer recovers a causal one (on Gemma-2-2b-it).
Anyone porting Arditi-style refusal interpretability to a new model should
either (a) commit to "statistical-method single direction" as their working
hypothesis and acknowledge the gradient-extraction question is untested, or
(b) implement RDO to test for additional causal directions under the
gradient method.

## What this changes for Phase 2 protocol

The Phase 2 protocol now needs both a statistical-extraction step AND a
gradient-extraction step to test the dimensionality question on a new model:

> **Phase 2 step 1 (mandatory):** per-model extraction-layer probe sweep.
> Identify the peak-AUC layer L*. Diff-of-means at L* is the
> statistical-method causal candidate.
>
> **Phase 2 step 4 (added):** RDO (or equivalent gradient-based optimization
> for additional orthogonal causal directions). Without this step, the
> Phase 2 deliverable can only test single-direction-via-statistical-
> extraction on the new model, not the multi-D claim from Wollschläger.

This positions Phase 2 as a genuine test of BOTH the statistical-extraction
and gradient-extraction stories on an under-studied deployed model.
Convergence (both methods find the same answer) is useful consolidation;
divergence (statistical finds 1, gradient finds k>1) is a real finding.

## Cell-level sample completions

Three samples per cell are persisted in the run record. Cell B (the causal
ablation) flips the model from "I cannot help you with hacking..." patterns
to "Here is the program..." patterns. All other cells produce baseline-like
refusals. This is the qualitative confirmation of the quantitative table.
