# Vocabulary audit — both contrastive sets, unified narrative

**Run 2026-05-30 covers all three contrastive sets in the project at once. The unified result: both Phase 1 and Phase 2 contrastive sets are heavily lexically separable (TF-IDF unigram test AUC 0.99 each), and a topic-matched defensive-equivalent draft de-confounds to AUC 0.67. This reframes §3 across the writeup: the diff-of-means direction d̂ is a *mixture* of refusal + vocabulary + topic that happens to be collinear in the contrastive set used to extract it.**

## Headline TF-IDF unigram logistic-regression test AUC, 70/30 split

| Contrastive set | n / side | Test AUC | Interpretation |
|---|---:|---:|---|
| Phase 1 (Gemma) — `data/contrastive.jsonl` (AdvBench harmful + Alpaca harmless) | 150+150 | **0.9877** | Heavily confounded — vocabulary alone classifies at 99%. |
| Phase 2 (Qwen) — `data/code_contrastive.jsonl` (HB cyber + AdvBench-code harmful + CodeAlpaca harmless) | 150+150 | **0.9946** | Heavily confounded — even more so than Phase 1. |
| Phase 2 matched DRAFT — `/tmp/code_contrastive_matched_draft.jsonl` (HB cyber harmful + defensive equivalents) | 40+40 | **0.6736** | De-confounded — vocabulary alone barely above chance (0.5). |

## What this unified result establishes

1. **The confound is in the data, not the model.** Both Phase 1's AdvBench-vs-Alpaca and Phase 2's HB-cyber-vs-CodeAlpaca have nearly identical lexical separability (~99% AUC) under the same bag-of-words classifier. AdvBench and Alpaca differ enormously in vocabulary (AdvBench harmful imperatives vs Alpaca diverse instructions); HB cyber + AdvBench code differs from CodeAlpaca for the same reason on a different topic. Both are confounded; neither was chosen carefully to avoid lexical confounding (Phase 1 used length matching, Phase 2 used CodeAlpaca length-filtered, but neither matched topic + vocabulary intentionally).

2. **Gemma's d̂ is a mixture that includes the causal refusal component.** The mixture decomposes as (vocabulary direction) + (causal refusal direction) + (other topic/sentiment components that happen to be collinear with harmful-vs-harmless). When you ablate d̂, you remove all of them — and the causal refusal component being in the mixture is what makes ablation collapse refusal (99% → 8% on HarmBench N=200, intervention-verified). The Phase 1 causal claim survives the confound completely because *intervention* is what tests causality, and the intervention on Gemma's d̂ works.

3. **Qwen's d̂ on `code_contrastive` is a mixture without (or with negligible) refusal component.** Ablating d̂ on Qwen removes the vocabulary signal but does not collapse refusal (Step 3: 0.97 → 0.97 judge; Step 3d addition sweep: 0.00 induced refusal at 32 (layer × coeff ≥ 1×) cells with no degeneration). This is the empirical proof that **a pure or near-pure vocabulary direction, ablated, does NOT collapse refusal**. The Qwen null is the control that lets you see what a confounded-but-causally-empty d̂ looks like — and it doesn't look like Gemma.

4. **The §3 "multiple real classifiers, only one causal" reframe is now mandatory.** The LDA bootstrap directions at L13 on Gemma are *also* extracted from a vocabulary-confounded contrast and are *also* mixtures — they likely pick up the vocabulary subspace too, just along directions near-orthogonal to d̂. The honest framing: the residual stream contains *multiple linearly-separable harmful-vs-harmless directions*, almost all of which are vocabulary/topic separability directions, and *exactly one* of which (diff-of-means d̂) ALSO contains the causal refusal component such that ablation collapses behavior. Classification ≠ causation, but the sharper version: separability ≠ refusal, and intervention is the test that picks out which separability direction (if any) is also the refusal mechanism.

5. **The matched-set DRAFT (TF-IDF 0.67) is the right experimental fix.** Dropping from 0.99 → 0.67 shows the defensive-equivalent rewrites substantially remove the lexical confound. The Phase 2 matched-set causal sweep becomes a **prediction**: if d̂ extracted from the matched set is causal under ablation/addition on Qwen, the Qwen null was contrastive-set-driven and (A-i) locks. If d̂ on the matched set is still inert (with adequate N), then Qwen's refusal genuinely isn't in the diff-of-means direction even when the confound is removed, and (A-ii) earns confirmation.

---



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
