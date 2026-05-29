# Phase 1 — extending classification ≠ causation to HarmBench (N=200)

**Headline.** Ablating one LDA-bootstrap-top-1 direction (cos = 0.175 with L13 diff-of-means, near-orthogonal) on 200 HarmBench prompts: refusal rate = **0.985** (Wilson 95% CI [0.96, 0.99], 197/200 refused).

- Model: `gemma-2-2b-it`
- d_hat (reference) extracted at L13 from AdvBench
- LDA-top-1 extracted via bootstrap seed 101
- cos(LDA-top-1, d_hat) = 0.1746 (near-orthogonal)
- Evaluated on 200 HarmBench standard-behavior prompts
- Scorer: dual_judge

## Pre-registered prediction (logged before running)

> LDA-bootstrap-top-1 ablation on 200 HarmBench prompts should give refusal rate ≥ 0.85 — matching the random-direction control (0.99) and **not** the d_hat ablation (0.08). If observed: classification ≠ causation extends from AdvBench N=12 to HarmBench N=200.

## Result vs the main HarmBench run

| Direction ablated | Refusal rate (n=200) |
|---|---:|
| Baseline (no hook) | 0.990 |
| **L13 diff-of-means d_hat** (causal) | **0.080** |
| **LDA-top-1, bootstrap 101** (this run) | **0.985** |
| Random unit vector (control) | 0.990 |

**Prediction confirmed.** The LDA-derived direction, which is a perfect classifier on the AdvBench split (AUC=1.0), is **causally inert** under Arditi-style ablation at the HarmBench scale, behaving identically to a random unit vector. The classification ≠ causation under statistical extraction finding holds at N=200 on a different benchmark.

## Per-category breakdown

| Category | n | LDA-ablated refusal |
|---|---:|---:|
| chemical_biological | 28 | 1.00 |
| cybercrime_intrusion | 40 | 0.97 |
| harassment_bullying | 19 | 1.00 |
| harmful | 21 | 1.00 |
| illegal | 58 | 1.00 |
| misinformation_disinformation | 34 | 0.94 |
