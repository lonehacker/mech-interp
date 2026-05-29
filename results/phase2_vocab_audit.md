# Phase 2 vocabulary audit — `data/code_contrastive.jsonl`

Companion to the L0 = 0.9996 AUC finding from `phase2_step1_layer_sweep`. The planner observed that AUC at L0 (embedding output) = direct evidence of lexical-by-construction separation, since "L0" means "the residual stream after embedding lookup, before any transformer block has computed anything." If the embedding alone separates harmful from harmless, the separability is in the *tokens*, not in any computation the model performs.

This audit quantifies the lexical confound on `code_contrastive.jsonl` and compares to Gemma's `contrastive.jsonl`.

## Per-layer real-direction AUC at L0 — Gemma vs Qwen

| | Gemma + AdvBench/Alpaca | Qwen + code_contrastive |
|---|---:|---:|
| Real AUC at L0 (embedding output) | **0.84** | **0.9996** |
| Real AUC at peak layer | 0.999 (L23) | 1.000 (L14) |
| Layers needed to gain ≥ 0.15 AUC | ~5 layers | 0 |

Gemma's contrastive set is **partially** separable at L0 — the embeddings know enough to do 84% AUC, but the harmful/harmless boundary gets *sharpened* by computation in layers 1–13. Qwen's contrastive set is **essentially fully** separable at L0 — the embeddings alone already do ~100% AUC. There's no room for the model's computation to add separation; the signal is already saturated at the embedding layer.

## Vocabulary-only classifier (no model, no activations)

Trained an L2-regularized logistic regression on bag-of-words features over the same 300 prompts, 70/30 train/test split, three vectorizers:

| Vectorizer | Vocab size | Train AUC | **Test AUC** |
|---|---:|---:|---:|
| Count (unigram) | 275 | 1.000 | **0.9965** |
| TF-IDF (unigram) | 275 | 1.000 | **0.9946** |
| TF-IDF (uni+bigram) | 543 | 1.000 | **0.9995** |

**The harmful side is identifiable from word-frequency alone at AUC ≈ 1.0.** This is the lexical confound made explicit — no activations, no transformer, no refusal mechanism involved. Any direction in the activation space that even partially aligns with the vocabulary-discrimination subspace will read as a "harmful-vs-harmless classifier" at near-perfect AUC.

## Most discriminative tokens (log-lift between sides)

**Harmful-side discriminative** (lift > +10, count ≥ 8 on harmful side):

| Token | Harmful count | Harmless count | log lift |
|---|---:|---:|---:|
| hack | 34 | 0 | +12.75 |
| instructions | 21 | 0 | +12.27 |
| step | 20 | 0 | +12.22 |
| give | 16 | 0 | +11.99 |
| secure | 16 | 0 | +11.99 |
| computer | 16 | 0 | +11.99 |
| sensitive | 16 | 0 | +11.99 |
| government | 13 | 0 | +11.79 |
| tutorial | 12 | 0 | +11.71 |
| provide | 11 | 0 | +11.62 |
| personal | 11 | 0 | +11.62 |
| malware | 11 | 0 | +11.62 |
| detailed | 11 | 0 | +11.62 |
| hacking | 10 | 0 | +11.52 |
| system | 9 | 0 | +11.42 |
| attack | 9 | 0 | +11.42 |
| steal/steals | 9 | 0 | +11.42 |
| virus | 8 | 0 | +11.30 |

**Harmless-side discriminative:**

| Token | Harmful count | Harmless count | log lift |
|---|---:|---:|---:|
| python | 0 | 21 | −12.24 |
| numbers | 0 | 18 | −12.09 |
| javascript | 0 | 18 | −12.09 |
| name | 0 | 18 | −12.09 |
| class | 0 | 12 | −11.68 |
| string | 0 | 12 | −11.68 |
| html | 0 | 12 | −11.68 |
| find | 0 | 11 | −11.60 |
| array | 0 | 11 | −11.60 |

**Zero overlap on the top-20 discriminative tokens of each side.** The two sides occupy nearly disjoint vocabulary spaces. Harmful-side words are imperative + cybercrime-targeting ("hack the X to do Y"); harmless-side words are language/data-structure terms ("python", "array", "javascript") and computer-science generics. A first-year ML model can tell these apart from words alone.

## What this means for the Phase 2 sweep interpretation

- If the operating-band addition sweep finds **no causal cell** → expected. d̂_diff-of-means is largely a vocabulary direction; ablating or adding it shouldn't reach the refusal mechanism because the mechanism doesn't live on the vocabulary axis. Leaf (A) of the pre-registration locks in.
- If the sweep **finds a coherent low-coefficient causal cell** → notable. It would mean d̂'s small non-vocabulary component is causally connected to refusal despite the overwhelming vocabulary signal. The lexical confound puts a *high* bar on the specificity control: random unit vectors here can also project onto the vocabulary subspace and look "directional", so the d̂ vs random comparison at the same coefficient is the necessary cleanness check.

## Implication for §9 and Phase 2 framing

This contrastive set was chosen to preserve the master-spec code-refusal angle without needing Qwen-Coder. The resulting vocabulary asymmetry was a known design risk (harmful prompts are HarmBench cybercrime + AdvBench code-filtered; harmless are CodeAlpaca) but the audit makes it concrete: **on this specific contrastive set, vocabulary IS the dominant signal at every residual depth**, and the diff-of-means recipe should NOT be expected to recover a refusal-specific direction with the same clean causal properties Gemma's d̂ had.

A more topic-aligned Phase 2 contrastive set — e.g., harmful side from HarmBench cybercrime, harmless side from sanitized/defensive equivalents of the same prompts — would let the diff-of-means recipe find a less vocabulary-confounded direction. That's the planner's option (a) rebuild, justified IFF the sweep result confirms inert (branch A) for the right reason.

Scripts: `experiments/phase1_L0_audit.py` (cross-model L0 AUC comparison) and `experiments/code_contrastive_vocab_audit.py` (vocabulary classifier baseline).
