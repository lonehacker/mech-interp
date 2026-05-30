# Phase 2 Step 3d — operating-band sweep pre-registration

**Logged 2026-05-29 BEFORE running the sweep, per planner protocol. The A/B/C decision below is the framing the Phase 2 result resolves into; do not relitigate it after seeing the numbers.**

## Context

- Phase 2 Step 1 found AUC = 1.000 at L14 on Qwen2.5-3B + code_contrastive.jsonl.
- Per-layer AUC inspection (post-hoc): every layer 0-35 sits at AUC ≥ 0.994 including the embedding output L0 (0.9996). The "peak" at L14 is meaningless — the harmful-vs-harmless signal is linearly separable everywhere from token-embedding onward, which is direct evidence that the separability is partly lexical-by-construction (vocabulary, not refusal mechanism).
- Phase 2 Step 3 ablated d̂_L14 multi-layer on 30 held-out code-harmful prompts: Δ refusal = −0.033 substring, 0.000 judge. Random ablation: 0.000.
- Phase 2 Step 1b 5-seed random control: random unit vectors hit AUC up to 0.98 on Qwen (mean 0.40, std 0.47, range [0.05, 0.98]) — the contrastive set has a strong vocabulary mean shift any direction picks up partially.
- Normalization gate (STEP 0): hook correspondence PASSED, both d̂ unit-normalized, natural-scale gap (Gemma 110.8 / Qwen 8.9 = 12.4×) decomposes as ~5× architectural (residual norm) + ~2.5× real (d̂ explains less variance on Qwen).

## Pre-registered three-way decision

Exactly one of (A), (B), (C) is the Phase 2 finding. Bind to the evidence below; do not pre-commit.

### (A) Genuinely inert
**Definition:** No addition cell in the operating-band sweep induces refusal on CodeAlpaca-harmless test prompts at any (injection layer × coefficient ≥ 1×) combination tested, AND single-layer ablation at any causally-live layer (if none, then any tested layer) fails to drop refusal on held-out code-harmful prompts.

**Interpretation (narrowed post-result):** On *this* contrastive set, d̂_diff-of-means at L14 is causally inert. The contrastive set turns out to be separable by vocabulary alone (TF-IDF AUC 0.99, L0 real AUC 1.00 — see `results/phase2_vocab_audit.md`), which means the d̂ this recipe recovers cannot be distinguished from a purely lexical direction. **The honest claim is "diff-of-means on a lexically-separable contrastive set recovers the lexical direction, not refusal."** Whether Qwen's refusal mechanism is reachable by diff-of-means *on a vocabulary-matched contrastive set* is an open question — answering it requires a topic-matched harmless side (defensive/sanitized equivalents of the harmful prompts). Do NOT claim "the refusal mechanism lives elsewhere on Qwen" from (A) alone; that's the (A-ii) sub-reading and requires the matched-set experiment to license.

### (B) Causal, wrong layer/scale
**Definition:** Addition sweep induces refusal at some (injection layer × coefficient) cell — refusal lift > 0.30 over baseline-harmless with specificity (random direction at same cell doesn't induce) — OR single-layer ablation drops refusal at some layer, even though all-layer ablation at L14 didn't.

**Interpretation:** d̂ IS causally connected to refusal on Qwen. L14 was wrong because AUC-peak ≠ causal layer, and/or 1.0× of L14 natural scale was too weak. Phase 2 lesson: select layer by operating-band intervention sweep + scale by per-layer natural scale, not by AUC peak. Report Qwen's causal layer depth vs Gemma's L13.

### (C) Replicated, different depth
**Definition:** Operating band identified (probably L3-LX or similar contiguous range), ablation at a layer within that band drops refusal on the 30 held-out code-harmful prompts to a Gemma-comparable level (Δ ≥ 0.5 vs baseline).

**Interpretation:** Phase 2 replicates Gemma's single-direction causal-refusal finding at a different residual depth. Report the band difference. This is the "boring but useful" outcome that confirms the protocol is portable.

## Sweep design (locked before run)

### Addition sweep — primary diagnostic
- **Direction**: d̂ extracted from L14 (the existing one). Holding extraction layer constant for this sweep mirrors Phase 1 §4.4.
- **Injection layers**: L3, L7, L11, L15, L19, L23, L27, L31 — 8 layers spread evenly across depth.
- **Coefficients**: 0.5×, 1×, 2×, 4× of **natural scale at extraction layer** (= 8.94). Per Phase 1 §4.4: per-extraction-layer scale is the portable unit, NOT per-injection-layer scale.
- **Target prompts**: 10 of the 30 held-out CodeAlpaca harmless prompts (fixed subset, seed-stable).
- **Scoring**: substring (`src.eval.is_refusal`) primary, Haiku 4.5 judge on any cell with substring lift > 0.10.
- **Specificity control**: random unit vector at the cell with highest induced refusal (if any), same (injection layer × coefficient).
- **Coherence check**: median completion length per cell (lobotomization sentinel).

### Ablation sweep — confirmatory
- Triggered ONLY if addition sweep identifies a causally-live region.
- Single-layer ablation at the live layer(s), on the same 30 held-out code-harmful prompts as Step 3. Dual-judge.

### Cost
- ~33 cells × 10 prompts = ~330 generations × ~13.4 s ≈ 74 min on MPS.
- Judge calls only on cells with substring lift > 0.10.
- Ablation sweep follow-up only if (B) or (C) triggers.

## Constraints

- Do NOT update writeup framing until the tree resolves.
- Do NOT point RDO at any specific layer until this sweep gives a causally-validated one.
- Keep the L14 null-ablation result as the motivating evidence, not a discard.
- The L0 = 0.9996 observation is reported for the writeup regardless of which leaf resolves: separability before any computation = direct evidence of vocabulary-component-in-separability, independent of the causal question.
