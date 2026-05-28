# Phase 0 — diff-of-means projection

- Model: `gemma-2-2b-it` on `mps`  
- Layer: 13 of 26
- n harmful = 15, n harmless = 15

## The most important number: does Gemma actually refuse?

- Harmful refusal rate: **1.00** (15/15)
- Harmless refusal rate: **0.07** (1/15)
- Spread (harmful − harmless): **+0.93**

> ✅ Gemma is refusing harmful prompts reliably. The diff-of-means direction below has a chance of being a real refusal direction.

## Three AUC numbers, not one (real direction vs controls)

- AUC at L13 (chosen mid layer, d_hat): **0.982**
- AUC at L2 (shallow control, d_hat): **0.800**
- AUC of random direction at L13: **0.444**

- Beats shallow control by ≥ 0.05? **True**
- Beats random direction by ≥ 0.05? **True**

## Projection numbers

- ||d|| = 92.8198
- proj harmful  mean ± std = 83.816 ± 23.151
- proj harmless mean ± std = -9.004 ± 28.435

Figure: `artifacts/figures/phase0_projection.png`

## Sanity gotchas
- Coherence probe: `2 + 2 = 4 
<end_of_turn><eos>`
- Templated vs raw differ? **True**
- Tokenized templated prompt shape: [1, 10]
- BOS assertion fires on force-fed double-BOS? **True**
- Activations finite (no NaN/Inf)? **True** (checked on harmful_acts, harmless_acts, d, harmful_shallow, harmless_shallow)

## What to do next
- If refusal rates AND the AUC controls look right, move to the layer sweep (`experiments/phase1_step2_layer_sweep.py` — not yet written).
- Look at 3–5 sample completions in `result.json` `sanity.refusal.harmful_samples` to see what Gemma actually said. The substring scorer can over- or under-count refusals; eyeballing is the source of truth.
- If the refusal split is not strong, do NOT proceed to Phase 1 without reframing. The mechanism we'd be characterizing wouldn't be refusal.
