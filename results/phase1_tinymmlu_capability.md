# Phase 1 — TinyMMLU capability check under d_hat ablation

**Question:** is the d_hat ablation specific to refusal, or does it also degrade general capability?

- Model: `gemma-2-2b-it`
- d_hat from AdvBench at L13
- TinyMMLU: 100 questions (statistically representative of full MMLU within ~2pp)

## Results

| condition | accuracy | Wilson 95% CI | correct / n | no_answer |
|---|---:|---|---|---:|
| baseline | 0.540 | [0.44, 0.63] | 54/100 | 0 |
| ablated  | 0.570 | [0.47, 0.66] | 57/100 | 0 |

**Δ accuracy = +0.030** (specific to refusal)

## Interpretation

- |Δ| < 0.05: ablation is specific to refusal — supports the causal claim.
- Δ < −0.10: ablation degraded general capability — refusal-rate claim has a confound.
- This run: |Δ| = 0.030.
