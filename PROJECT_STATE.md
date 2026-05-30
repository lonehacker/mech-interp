# PROJECT_STATE — Phase 2 current state + terminology

This is the canonical source for how to talk about refusal-direction work in
this repo. The terminology section below is binding for new docs/code.

## Terminology / experiment axes

Refusal-direction experiments have **two orthogonal axes**. Conflating them
corrupts result statements and trips a methods-literate reviewer.

### Axis 1 — Layer-selection criterion

Choosing WHICH layer (and position) to extract or intervene at. Two methods:

- **AUC-layer-selection** — rank layers by linear-probe AUC of the harmful-vs-
  harmless classifier. Picks the most-separable layer. **Broken on Qwen2.5-3B:**
  separability is saturated (AUC ≥ 0.994 at every layer including L0 = 0.9996),
  so AUC cannot discriminate layers. The L14 we used was effectively arbitrary
  (first layer to hit float-1.000).

- **bypass-gap-layer-selection** — rank layers by the drop in refusal-token
  probability when a candidate direction at that (layer, pos) is ablated.
  Causal-effect criterion, not separability.

Layer selection PRODUCES a layer. It is not an extraction method.

### Axis 2 — Direction extraction method

Given a layer, compute the direction vector. Two methods:

- **diff-of-means** → the vector we call **`d̂`**. Statistical: difference of
  class centroids at the chosen layer/position.
- **RDO** (Refusal Direction Optimization, Wollschläger 2025) → the vector we
  call **`d_rdo`**. Gradient: optimize a direction at a FIXED layer such that
  its ablation maximizes refusal drop.

**`d̂` and `d_rdo` are different vectors even at the same layer.**
`d̂` is by definition the diff-of-means direction — **RDO does not produce `d̂`.**
The backwards-decomposition step computes `cos(d_rdo, d̂)` precisely to compare
the two extractors; calling RDO's output `d̂` destroys the comparison.

### Why "bypass" shows up in both axes

RDO's training objective is itself "maximize bypass" — same word as the
layer-selector's ranking metric. They are different uses:
- The **selector** uses bypass to RANK LAYERS.
- **RDO** uses bypass to OPTIMIZE A DIRECTION at one fixed layer.

### The 2×2 of (layer-selection criterion × extraction method)

|  | diff-of-means (`d̂`) | RDO (`d_rdo`) |
|---|---|---|
| **AUC-layer-selection** | Phase 2 Step 3/3e: INERT at L14 (layer arbitrary; AUC saturated) | Not interesting (AUC layer is wrong on Qwen) |
| **bypass-gap-layer-selection** | ← **Part 2: pending** (cheap, may resolve A-i) | Wollschläger end-to-end = **bypass-gap-layer-selection then RDO extraction** (the planned RDO run) |

The bottom-left cell — diff-of-means with bypass-gap layer selection — was
untested before Part 2 because L14 was AUC-selected and we never tried
diff-of-means at a causally-selected layer. The DIM-matched run (see
artifacts) already flagged L20-L24 around pos -4/-1 as the candidate region
under bypass-gap-style scoring in Wollschläger's harness; Part 2 verifies
this in our harness with dual-judge + coherence check.

### Correct phrasings (use these)

- **"the RDO run"** (or "Wollschläger end-to-end") refers to the TWO-step
  sequence: bypass-gap-layer-selection THEN RDO extraction. Spell out both
  steps if it matters which one is doing the work.
- **"`d_rdo` that passed causal verification"** — only after ablation drops
  refusal AND a random-direction specificity control fails AND completions
  are coherent. RDO produces a *candidate* direction; verification gates
  whether it's reference-grade.
- **"diff-of-means at the bypass-gap-selected layer"** — `d̂_L` at the layer
  chosen by bypass-gap criterion, distinct from `d̂` at the AUC-selected
  L14.

### Wrong phrasings (rewrite these)

- ❌ "RDO finds the (causal) layer" — RDO takes the layer as INPUT.
  bypass-gap-layer-selection finds it.
- ❌ "RDO found `d̂`" / "RDO's `d̂`" — `d̂` is the diff-of-means output.
  RDO produces `d_rdo`.
- ❌ "RDO found a causal direction" as a bare act — imprecise. Use
  "`d_rdo` that passed causal verification."
- ❌ "Wollschläger pipeline" / "Path A" as one blob — split into the two
  steps it actually performs.

## Current state — Phase 2 (Qwen2.5-3B-Instruct)

### Done
- **Step 1**: per-layer linear-probe AUC sweep. Verdict: saturated; AUC cannot
  discriminate layers. → `results/phase2_step1_qwen-qwen2.5-3b-instruct.md`
- **Step 3** (AUC-selected L14, diff-of-means, code_contrastive): ablation of
  `d̂` drops refusal 1.00 → 0.97. INERT. → `results/phase2_step3_causal.md`
- **Step 3d** (operating-band ADDITION sweep on `d̂` at L14): natural-scale
  addition shows weak but present effect, peak at L15 × 4× coefficient.
  → `results/phase2_step3d_operating_band_sweep.md`
- **Step 3e** (matched-set causal sweep on `d̂_matched` at L14):
  INCONCLUSIVE — n=10 underpowered; near-orthogonal to `d̂_old` (cos=−0.087),
  4× smaller natural scale (2.05 vs 8.94). → `results/phase2_step3e_matched_set_sweep.md`

### Resolved
- **Step 4a** — geometry-of-refusal loadability for Qwen2.5-3B: confirmed.
  QwenModel + rdo.py + bypass-gap-layer-selector all work on Qwen2.x paths.
- **Step 4b** — diff-of-means × bypass-gap-layer-selection in Wollschläger's
  harness on matched set: 180 (pos × layer) cells swept. 18 cells with
  refusal_score < −5 AND steering > 0 at non-pruned layers (L0-L27).
  Strongest: pos -4 around L20-L24 (refusal -10 to -11, steering +4.3 to +4.6,
  KL 1.5-3). All cells failed the paper's strict kl_threshold=0.1 — could
  be miscalibration for Qwen OR could indicate the bypasses come with real
  distributional disruption. **Part 2 in our harness adjudicates.**

### In progress
- **Part 1** — terminology audit (this file is the artifact).
- **Part 2** — diff-of-means + bypass-gap layer sweep in mech-security
  harness, dual-judge, targeted at L19-L25 × positions {−1, −4} (prior from
  Step 4b). Coherence check load-bearing at best cell.

### Pending (conditional on Part 2)
- **RDO extraction run** + d_rdo causal verification (if Part 2 shows
  layer-selection-artifact confirmed).
- **Backwards decomposition** — cos(d_rdo, d̂_at_that_layer), cos(d_rdo,
  d̂_old), cos(d_rdo, TF-IDF/L0-separability subspace).

### Headline candidate (if Part 2 confirms)

*"On saturated-separability models, AUC-based layer-selection fails where
bypass-gap-based layer-selection succeeds — plain diff-of-means recovers the
causal direction at the causally-selected layer."* RDO extraction would then
be CONFIRMATION (cos(d_rdo, d̂_at_layer) high), not the headline.
