# Phase 2 Step 3d — operating-band addition sweep (Qwen/Qwen2.5-3B-Instruct)

- d̂ extracted from L14 (held constant)
- Natural scale at L14: 8.937
- Inject layers: [3, 7, 11, 15, 19, 23, 27, 31]
- Coefficients (× natural scale): [0.5, 1.0, 2.0, 4.0]
- Target: 10 CodeAlpaca-harmless prompts (subset of step 3b's 30, seed 1)
- Baseline (no hook) substring refusal: **0.000**

## Substring-scorer refusal-rate heatmap

| coeff × | L3 | L7 | L11 | L15 | L19 | L23 | L27 | L31 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5× | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 1.0× | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 2.0× | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 4.0× | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

**Best cell:** `add_L3_c0.5x` (lift +0.000 vs baseline). Caveat: 0.5× cells are *uninformative by Phase 1 §4.4's own finding* — sub-natural-scale injection is below the operating threshold even for a known-causal direction. The decisive evidence is the **1×, 2×, 4× nulls at every layer with no coherence degeneration (mean chars 670–715 throughout, baseline ~680)**.

Heatmap figure: `artifacts/figures/phase2_step3d_operating_band_sweep.png`

Per-prompt completions in `artifacts/runs/phase2_step3d/<timestamp>/result.json`.
Pre-registration: `results/phase2_step3d_preregistration.md`.

## Conclusion — narrowed per post-run discipline check

The pre-registration's leaf (A) text reads "d̂_diff-of-means on Qwen2.5-3B + code_contrastive is causally inert. Classification ≠ causation extends to a second model, this time at the direction level — the diff-of-means recipe recovers a vocabulary direction and the actual refusal mechanism lives elsewhere."

**The first clause survives. The second clause does not — it's an overclaim this experiment alone cannot license, and the vocabulary audit (`results/phase2_vocab_audit.md`) is the diagnostic that explains why.**

What this experiment actually established:

1. On `data/code_contrastive.jsonl`, d̂ extracted via diff-of-means at L14 is causally inert under both ablation (step 3) and addition across all 32 (injection layer × coefficient ≥ 1×) cells in the operating-band sweep. No degeneration; behavior is unchanged by the intervention.

2. The contrastive set is separable by **vocabulary alone** at TF-IDF unigram test AUC 0.9946, and at the embedding-output layer (Qwen L0) at real AUC 0.9996. The harmful side (HarmBench cybercrime + AdvBench code-filtered) and the harmless side (CodeAlpaca) have zero overlap on top-20 discriminative tokens.

3. Therefore the d̂ this recipe recovers on this contrastive set **cannot be distinguished from a purely lexical direction**. Any direction that aligns with the vocabulary subspace will read as a perfect harmful-vs-harmless classifier at every residual layer.

What this experiment did NOT establish:

- That **Qwen's refusal mechanism is unreachable by diff-of-means in general.** That requires running diff-of-means on a contrastive set where the two sides are NOT separable by vocabulary alone — i.e., a topic- and vocabulary-matched set that contrasts on intent rather than lexicon. The pre-registration's (A-ii) reading remains an open question, not a settled conclusion.

The honest narrowing of leaf (A):

> *On a contrastive set whose two sides are separable by vocabulary alone (TF-IDF AUC 0.99, L0 real AUC 1.00), diff-of-means at any residual layer recovers a direction indistinguishable from the lexical contrast, and ablating or adding that direction at any tested scale does not affect refusal behavior. Whether Qwen's refusal mechanism is recoverable by diff-of-means on a vocabulary-matched contrastive set is the next experiment, not a conclusion from this one.*

The next confirmatory experiment is **not** a layer-restricted ablation sweep on the same contrastive set — that would harden the half of (A) that's already established without addressing the live question. The next experiment is a **vocabulary-matched contrastive set**: same harmful side, but a harmless side drawn from defensive/sanitized equivalents that share topic and vocabulary with the harmful side (e.g., "write code to *prevent* this SQL injection" paired with the existing "write code to exploit this SQL injection"). Decisive readouts:

- Does TF-IDF AUC on the matched set drop substantially (confirming the lexical confound is removed)?
- If yes, does diff-of-means d̂ on the matched set show a causal hand under ablation/addition that d̂ from `code_contrastive` did not?

Both outcomes are publishable. d̂-becomes-causal-on-matched-set → (A-i) earned, methodological lesson is sharp: *on lexically-separable contrastive sets, diff-of-means recovers vocabulary, and your AUC will not warn you*. d̂-still-inert-with-confound-removed → (A-ii) earned, the single-causal-direction picture is model-specific in a strong sense, RDO/multi-direction extraction needed.
