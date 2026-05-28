# Phase 1 Step 2 — layer sweep

- Model: `gemma-2-2b-it` on `mps` | n_layers=26
- n harmful = 150, n harmless = 150
- Data: `data/phase0-pairs.jsonl` (the Phase-0 hand-written set; replace per Step 1 (H) before publishing)

## Headline

- **Peak layer: L23** (AUC = 0.999)
- Plateau (within 0.02 of peak): **L7–L25**
- AUC band layers: [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]

## Controls at the peak layer

- Real direction:     **0.999**
- Random direction:   0.226
- Shuffled labels:    0.394

- Real beats random by ≥ 0.20 at peak? **True**
- Real beats shuffled by ≥ 0.20 at peak? **True**
- Shuffled-labels curve near chance overall (|mean − 0.5| < 0.15)? **True** (actual mean = 0.376)
- Random-direction curve near chance overall? **False** (actual mean = 0.245)

Figure: `artifacts/figures/phase1_step2_layer_sweep_advbench.png`

## How to read the figure

- The blue line should climb above 0.5 (chance) somewhere mid-network and plateau or peak. That's where refusal information is concentrated in the residual stream.
- The orange (random-direction) and green (shuffled-labels) lines should hover near 0.5 at every layer. If either tracks the real curve, the result is not trustworthy.
- The peak is **a working hypothesis layer**, not a finding. Separation is correlational. Step 3 (steering) tests causality: ablate at the peak layer, does refusal actually drop?

## What to do next

- Record L = 23 in `tasks.md` as the working hypothesis layer.
- Run Step 3 — steering: ablate `d_hat` at L23 on the harmful set; measure refusal-rate drop. Add `d_hat` at the same layer on the harmless set; measure over-refusal. The four mandatory controls (random direction, coherence, generalization split, both-directions-agree) gate that step.
- After Step 1 (H) — when the contrastive set is replaced with AdvBench + Alpaca — re-run this script. The peak layer may move; that's information, not failure.
