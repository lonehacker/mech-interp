# Phase 1.5-A — hardened classification ≠ causation

**Headline.** Continuous causal metric (refusal-minus-compliance logit-difference shift at first response token) on N = 200 held-out HarmBench prompts. Z-scored against a random-vector null band (5 random unit vectors: mean |effect| = 0.136, σ = 0.160, 2σ upper bound = 0.455).

- Model: `gemma-2-2b-it`
- Extract layer: L13
- Refusal first-token set: [235285]  (validated: 99% coverage of baseline refusal openers)
- Compliance first-token set: [1620, 4858, 1917, 651, 235281, 6750]  (96% coverage; structurally cancels shared mass via contrast)
- Baseline contrast (no hook): refusal_logit - compliance_logit = 12.972

## Results — all cells

| Category | Name | effect_signed | \|effect\| | z-score | refusal_Δ | compliance_Δ |
|---|---|---:|---:|---:|---:|---:|
| causal_d_hat | d_hat_seed_0 | -14.876 | 14.876 | +92.21 | -10.452 | +4.424 |
| causal_d_hat | d_hat_seed_1 | -14.607 | 14.607 | +90.53 | -10.210 | +4.397 |
| causal_d_hat | d_hat_seed_2 | -14.980 | 14.980 | +92.87 | -10.348 | +4.632 |
| causal_d_hat | d_hat_seed_3 | -14.421 | 14.421 | +89.37 | -9.884 | +4.537 |
| causal_d_hat | d_hat_seed_4 | -15.081 | 15.081 | +93.50 | -10.858 | +4.223 |
| inert_lda | LDA_top1_bs101 | -0.106 | 0.106 | -0.18 | -0.020 | +0.086 |
| inert_lda | LDA_top1_bs202 | +0.063 | 0.063 | -0.46 | +0.081 | +0.018 |
| inert_lda | LDA_top1_bs303 | -0.090 | 0.090 | -0.29 | -0.015 | +0.075 |
| inert_lda | LDA_top1_bs404 | -0.009 | 0.009 | -0.79 | +0.035 | +0.045 |
| inert_lda | LDA_top1_bs505 | -0.113 | 0.113 | -0.14 | +0.054 | +0.167 |
| inert_other_layer | d_hat_L3 | -1.033 | 1.033 | +5.62 | -0.439 | +0.595 |
| random_null | random_seed_7 | -0.015 | 0.015 | -0.76 | -0.053 | -0.038 |
| random_null | random_seed_17 | -0.015 | 0.015 | -0.76 | +0.044 | +0.059 |
| random_null | random_seed_27 | -0.141 | 0.141 | +0.03 | -0.003 | +0.138 |
| random_null | random_seed_37 | +0.104 | 0.104 | -0.20 | +0.190 | +0.086 |
| random_null | random_seed_47 | -0.404 | 0.404 | +1.68 | -0.094 | +0.310 |

## Sign convention

- `effect_signed` = (refusal_logit − compliance_logit)_ablated − (refusal_logit − compliance_logit)_baseline.
- Negative `effect_signed` = mass moved from refusal toward compliance under ablation = causal.
- `|effect|` is reported because the figure plots magnitude vs the null band; causal direction sits *above* the band, inert overlaps it.
- `z_score = (|effect| − null_mean) / null_std`. Inert ≈ 0 (inside band); causal ≫ 2.

## Diagnostic check (raw refusal_Δ alongside contrast)

If a direction shows a big `|effect|` but a small raw `refusal_Δ`, the contrast is being driven by `compliance_Δ` (mass moving INTO compliance without leaving refusal) — which IS a valid causal signature but worth flagging. If `refusal_Δ` is large and `compliance_Δ` is similarly large in the opposite direction, the contrast is doing exactly what it's designed to do: tracking *directional* shift, not norm wobble.
