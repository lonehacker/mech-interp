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

Given a layer, compute the direction vector. The two primaries in Phase 2:

- **diff-of-means** → the vector we call **`d̂`**. Statistical: difference of
  class centroids at the chosen layer/position. Implementation:
  `src.directions.diff_of_means`.
- **RDO** (Refusal Direction Optimization, Wollschläger 2025) → the vector we
  call **`d_rdo`**. Gradient: optimize a direction at a FIXED layer such that
  its ablation maximizes refusal drop. Implementation: external,
  `~/safe_ai/geometry-of-refusal/rdo.py`.

**`d̂` and `d_rdo` are different vectors even at the same layer.**
`d̂` is by definition the diff-of-means direction — **RDO does not produce `d̂`.**
The backwards-decomposition step computes `cos(d_rdo, d̂)` precisely to compare
the two extractors; calling RDO's output `d̂` destroys the comparison.

A third peer extraction method, **LDA** (`src.directions.lda_directions`),
returned top-k orthogonal Fisher-LDA directions; Phase 1 used it for the
bootstrap-stability check (HarmBench `phase1_harmbench_lda_extension`). LDA
is not the focus of Phase 2's 2×2 but lives in the same `src/` layer as
diff-of-means and `ablate_subspace`.

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
- **Part 1** — terminology audit. This file + HOW_IT_WORKS §"Two axes"
  are the artifact.
- **Step 4a** — geometry-of-refusal loadability for Qwen2.5-3B: confirmed.
  QwenModel + rdo.py + bypass-gap-layer-selector all work on Qwen2.x paths.
- **Step 4b** — diff-of-means × bypass-gap-layer-selection in Wollschläger's
  harness on matched set: 180 (pos × layer) cells swept. 18 cells with
  refusal_score < −5 AND steering > 0 at non-pruned layers (L0-L27).
  Strongest: pos −4 around L20-L24 (refusal −10 to −11, steering +4.3 to +4.6,
  KL 1.5–3). Paper's strict `kl_threshold=0.1` rejected all candidates;
  Part 2 in our harness adjudicated whether that's miscalibration or
  real disruption.
- **Part 2** — diff-of-means + bypass-gap layer sweep in mech-security
  harness, dual-judge, L19-L25 × positions {−1, −4} (pre-committed to
  Step 4b's prior region). **Result: Reading A locked.** At L22 (both
  positions) + L23 + L25, ablating `d̂` drops refusal 9-10/10 → 0/10
  cleanly (judge: 0 refused / 0 partial / 10 complied), with mean_chars
  matching baseline (770-787 vs 737, within ±10%). The d̂-ablated
  completions are detailed on-target answers to each specific harmful
  request, in fluent Qwen voice — not on-topic rambling, not degraded
  output. Random-direction control at the *same* L22 cells (3 seeds
  × 2 positions = 6 cells) reproduces baseline refusal: judge 9/10
  refused, completions are textbook "I'm sorry, but I can't assist…"
  refusals. d̂-specificity holds across seeds; layer is not fragile.
  - See `results/phase2_part2_dim_bypass_gap_sweep.md` (forthcoming
    writeup) and `artifacts/runs/phase2_part2/20260530-134914/result.json`
    + `artifacts/runs/phase2_part2_random_at_prior/20260530-153438/result.json`
    for the raw numbers + per-prompt completions.
  - **Caveat (pre-registered, survives the clean result):** the matched
    contrast controls vocabulary (TF-IDF AUC ≈ 0.50) but cannot control
    attacker-vs-defender *stance*. The honest headline sentence is
    "diff-of-means at the bypass-gap-selected layer recovers a causal
    direction whose ablation collapses refusal on this matched contrast,"
    with the standing note that the direction isolates *refusal-or-stance*,
    not refusal cleanly. Orthogonal to the layer-selection finding.

### Pending — optional (no longer load-bearing)
- **RDO extraction at L22** + d_rdo causal verification — was the fallback
  for "diff-of-means can't find it." Diff-of-means did find it; RDO is now
  confirmation at best (does cos(d_rdo, d̂_L22) ≈ 1?), not a required next
  experiment. Treat as possible appendix.
- **Backwards decomposition** cosines — same: appendix material, not
  spine.

### Headline (earned)

**"AUC-based layer-selection fails on saturated-separability models;
bypass-gap-based layer-selection recovers the causal refusal direction
that plain diff-of-means then extracts."**

This is the practical-teeth version: the fix is a layer-selection
criterion anyone can apply, not a gradient pipeline. The Phase 2 L14
inertness was the wrong *layer/position* via the wrong *selection
criterion*; the direction-extraction method was always fine.

### Unified writeup spine (across Phase 1 + 1.5 + 2)

Classification ≠ Causation at three levels:

1. **Gemma direction-level** (Phase 1.5-A) — LDA-bootstrap directions
   classify harmful/harmless at AUC = 1.0, sit inside the random null
   band (z ≈ 0) under continuous causal-effect ablation. `d̂` at z ≈ +90.
2. **Gemma prompt-level** (Phase 1.5-B) — fictional-framing prompts
   project onto the refusal direction the same as direct harmful prompts,
   but the model complies anyway. Same activations, different behavior.
3. **Qwen model-level** (Phase 2) — separability saturates at every
   layer including L0; the causal direction localizes to one specific
   layer/position region recoverable only by bypass-gap-based selection.

Unifying claim: *separability is cheap and uninformative about mechanism
location; intervention localizes it; the gap between them is real and
grows from Gemma to Qwen.* The Qwen-specific methodological lesson
(AUC vs bypass-gap layer-selection) is the bridge — a portable fix for
the next saturated-separability model.
