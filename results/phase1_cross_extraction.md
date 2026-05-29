# Phase 1 — Cross-extraction: is the refusal direction distribution-specific?

**Headline.** cos(d_hat extracted from AdvBench, d_hat extracted from HarmBench) = **0.9508** at L13.

Both directions extracted using diff-of-means, with the **same Alpaca harmless side** held constant. Only the harmful source varies (AdvBench vs HarmBench).

- Model: `gemma-2-2b-it`
- Extract layer: L13
- AdvBench harmful prompts: 150
- HarmBench harmful prompts: 200 (standard subset)
- Alpaca harmless prompts: 150 (held constant)

## Cosine similarity interpretation

**cos > 0.9 → essentially the same direction.** The refusal direction is invariant to which harmful-prompt benchmark you extract it from. The cross-extraction question collapses: there is one direction, not two.

## Per-direction AUC on each harmful source

Each row shows how cleanly a d_hat separates a given harmful source from Alpaca harmless.

| Direction | AUC vs AdvBench | AUC vs HarmBench |
|---|---:|---:|
| d_hat_advbench | 0.998 | 0.997 |
| d_hat_harmbench | 0.999 | 1.000 |

## Natural-scale comparison

Cluster-mean projection (harmful − harmless) along each d_hat:

| Direction | scale on AdvBench | scale on HarmBench |
|---|---:|---:|
| d_hat_advbench | 105.17 | 84.96 |
| d_hat_harmbench | 100.00 | 89.36 |

## Per-layer AUC for d_hat_harmbench

| Layer | AUC (HarmBench vs Alpaca, d_hat extracted at same layer) |
|---:|---:|
| L0 | 0.932 |
| L1 | 0.982 |
| L2 | 0.953 |
| L3 | 0.929 |
| L4 | 0.913 |
| L5 | 0.977 |
| L6 | 0.946 |
| L7 | 0.982 |
| L8 | 0.986 |
| L9 | 0.997 |
| L10 | 0.993 |
| L11 | 1.000 |
| L12 | 1.000 |
| L13 | 1.000 |
| L14 | 0.999 |
| L15 | 0.998 |
| L16 | 0.998 |
| L17 | 0.998 |
| L18 | 0.997 |
| L19 | 0.997 |
| L20 | 0.997 |
| L21 | 0.997 |
| L22 | 0.998 |
| L23 | 0.999 |
| L24 | 1.000 |
| L25 | 0.999 |

## Method caveat

Harmless side held constant (Alpaca, the same set used in our AdvBench d_hat extraction). Only the harmful source varies (AdvBench vs HarmBench). Length-matching between HarmBench harmful and Alpaca harmless is not audited for this run.
