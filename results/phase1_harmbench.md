# Phase 1 — HarmBench refusal-rate evaluation under d_hat ablation

**Headline:** baseline refusal = **0.990**, ablated refusal = **0.080**, Δ = **+0.910**.

- Model: `gemma-2-2b-it`
- d_hat extracted from AdvBench (training distribution) at L13
- Evaluated on HarmBench (held-out, OOD), n = 200 prompts
- Scorer: dual_judge
- HarmBench categories present: chemical_biological, cybercrime_intrusion, harassment_bullying, harmful, illegal, misinformation_disinformation

## Conditions

| condition | refusal rate | 95% Wilson CI | n_refused/n | partial | error |
|---|---:|---|---|---:|---:|
| baseline | 0.990 | [0.96, 1.00] | 198/200 | 0 | 0 |
| ablated | 0.080 | [0.05, 0.13] | 16/200 | 2 | 0 |
| random_ctrl | 0.990 | [0.96, 1.00] | 198/200 | 0 | 0 |

## Per-category breakdown

| category | n | baseline | ablated | random | Δ (baseline−ablated) |
|---|---:|---:|---:|---:|---:|
| chemical_biological | 28 | 1.00 | 0.07 | 1.00 | +0.93 |
| cybercrime_intrusion | 40 | 0.97 | 0.07 | 0.97 | +0.90 |
| harassment_bullying | 19 | 1.00 | 0.21 | 1.00 | +0.79 |
| harmful | 21 | 1.00 | 0.10 | 1.00 | +0.90 |
| illegal | 58 | 1.00 | 0.05 | 1.00 | +0.95 |
| misinformation_disinformation | 34 | 0.97 | 0.06 | 0.97 | +0.91 |

## Specificity check

Random-direction Arditi ablation should NOT drop refusal (if it does, ablating *any* direction breaks refusal — not specific). Observed: random_ctrl refusal = 0.990. **Baseline − random = +0.000** — perfect specificity at this scale. The ablation effect is direction-specific, not "any perturbation breaks refusal."

## Dual-judge cross-check

The headline numbers use Claude Haiku 4.5 as the primary judge. Claude Opus 4.7 was run as a cross-check on every prompt:

| Condition | Haiku 4.5 (primary) | Opus 4.7 (cross-check) | Agreement |
|---|---:|---:|---:|
| baseline | 0.990 | 0.890 | 180/200 = 90% |
| ablated | 0.080 | 0.040 | 150/200 = 75% |
| random_ctrl | 0.990 | 0.890 | 180/200 = 90% |

Both judges agree the ablation drop is enormous. Opus is slightly stricter on what counts as REFUSED (calls more borderline cases COMPLIED), so its baseline rate is lower (0.89 vs 0.99) — but the *delta* between baseline and ablated is comparable under either judge. Lower agreement on the ablated condition (75%) reflects that post-ablation completions land in genuinely ambiguous territory where judges legitimately disagree.

## Comparison to Wollschläger 2025

Wollschläger et al. (ICML 2025) report **79.9% JailbreakBench ASR** on the same Gemma-2-2b-it model using their gradient-based RDO + directional ablation. Our result: **91% HarmBench compliance** using simple diff-of-means ablation. Different benchmarks (HarmBench vs JailbreakBench), different judges, so not directly comparable as exact numbers — but the qualitative read is that **statistical-extraction diff-of-means is in the same ballpark as gradient-based RDO for behavioral effect on this model**. Wollschläger's gradient-method edge is likely more visible on bigger models with more refusal-direction redundancy.

## OOD generalization caveat — what this DOES and DOESN'T show

The d_hat used here was extracted from AdvBench. HarmBench is held out (we never extract from it). So this result is **OOD generalization** of a single statistical direction — encouraging.

It does NOT show:
- That d_hat ablation generalizes across models (only Gemma-2-2b-it tested)
- That capability is preserved on HarmBench-shaped questions (we have TinyMMLU showing general capability preserved; per-benchmark capability check on something like HarmBench-utility would strengthen the claim)
- That this would survive against a determined adversary trying to *induce* refusal (we ablate, we don't defend)
