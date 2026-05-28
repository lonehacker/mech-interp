# Phase 1 Step 4 — per-layer probing

**Independent line of evidence for the refusal representation.** A linear probe at each layer answers a different question than steering: not *does intervening change behavior?* but *is the label linearly readable?* Agreement between probe and steering bands is strong; tension is research.

- Model: `gemma-2-2b-it` on `mps` | n_layers=26
- n harmful = 150, n harmless = 150
- Split: stratified, test_size=0.25, seed=0
- Logistic regression: C=1.0, solver=lbfgs
- Shuffled-label control: shuffle_seed=42

## Headline

- **Probe peak: L5** (test_acc = 1.000)
- **Probe band (within 0.02 of peak): L1–L25** (24 layers)
- **Steering band (from Step 2): L7–L25** (19 layers)
- **Band overlap: 19 layers** — [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
- At steering-injection L13: probe test_acc = 1.000, shuffled = 0.387

## Controls (gates, not extras)

| Control | Numbers | Pass |
|---|---|---|
| Shuffled at chance overall (\|mean − 0.5\| < 0.10) | mean shuffled test acc = 0.473 | ✅ |
| Shuffled at chance at probe peak (\|acc − 0.5\| < 0.15) | shuffled at L5 = 0.520 | ✅ |
| Real beats shuffled at peak by ≥ 0.30 | real = 1.000, shuffled = 0.520 | ✅ |

## Per-layer table

| layer | real test acc | real train acc | shuffled test acc | in steering band? |
|---:|---:|---:|---:|:---:|
| L0 | 0.960 | 0.996 | 0.453 |  |
| L1 | 0.987 | 1.000 | 0.493 |  |
| L2 | 0.987 | 1.000 | 0.440 |  |
| L3 | 0.947 | 1.000 | 0.427 |  |
| L4 | 0.987 | 1.000 | 0.493 |  |
| L5 | 1.000 | 1.000 | 0.520 |  |
| L6 | 1.000 | 1.000 | 0.507 |  |
| L7 | 1.000 | 1.000 | 0.520 | ✓ |
| L8 | 1.000 | 1.000 | 0.493 | ✓ |
| L9 | 1.000 | 1.000 | 0.467 | ✓ |
| L10 | 0.987 | 1.000 | 0.440 | ✓ |
| L11 | 1.000 | 1.000 | 0.387 | ✓ |
| L12 | 1.000 | 1.000 | 0.427 | ✓ |
| L13 | 1.000 | 1.000 | 0.387 | ✓ |
| L14 | 1.000 | 1.000 | 0.440 | ✓ |
| L15 | 1.000 | 1.000 | 0.467 | ✓ |
| L16 | 1.000 | 1.000 | 0.453 | ✓ |
| L17 | 1.000 | 1.000 | 0.507 | ✓ |
| L18 | 1.000 | 1.000 | 0.520 | ✓ |
| L19 | 1.000 | 1.000 | 0.480 | ✓ |
| L20 | 1.000 | 1.000 | 0.453 | ✓ |
| L21 | 1.000 | 1.000 | 0.507 | ✓ |
| L22 | 1.000 | 1.000 | 0.507 | ✓ |
| L23 | 1.000 | 1.000 | 0.520 | ✓ |
| L24 | 1.000 | 1.000 | 0.507 | ✓ |
| L25 | 1.000 | 1.000 | 0.493 | ✓ |

Figure: `/Users/anshulsinghle/safe_ai/mech-security/artifacts/figures/phase1_step4_probe_by_layer.png`

## (H) Interpretation — to be filled in by the human

The runbook §7 is explicit: if the probe is strong where steering is weak (or vice versa), that tension IS the research — do not paper over it. Read the per-layer table and answer:

1. Does the probe band overlap the steering band? (19 / 19 layers do.)
2. Layers in steering band but NOT probe band: []
3. Layers in probe band but NOT steering band: [1, 2, 4, 5, 6]
4. Is the steering-injection layer L13 a strong probe layer? (test_acc = 1.000)

Write the answer here, in raw-numbers form, before claiming convergence:

> _(H) finding goes here_
