# Phase 1 — Refusal direction interventions on Gemma-2-2b-it

A controlled replication of published refusal-direction interventions on a
small open-weight model, with full Arditi controls, a calibrated LLM judge,
a bootstrap stability check on multi-direction claims, and an explicit
methodology-decoupling result.

**Scope, stated up front.** Phase 1 is a *replication + methodology
contribution*, not novel research. Every conceptual claim made here is in
the published literature; the contribution is the controls run, the
calibrated judge, the bootstrap stability check, the explicit statistical-
vs-gradient extraction decoupling, and the four-step Phase 2 protocol that
emerges. The artifact's value is competence shown end-to-end on known
territory, not discovery.

## Table of contents

1. [The question, in one sentence](#1-the-question-in-one-sentence)
2. [The mechanism, in plain English](#2-the-mechanism-in-plain-english)
3. [What we did and the numbers](#3-what-we-did-and-the-numbers)
4. [Where we sit vs the published literature](#4-where-we-sit-vs-the-published-literature)
5. [The methodology contribution](#5-the-methodology-contribution)
6. [Phase 2 protocol](#6-phase-2-protocol)
7. [What's not yet done](#7-whats-not-yet-done)

---

## 1. The question, in one sentence

> *Where in `gemma-2-2b-it` does the decision to refuse a harmful request
> live, and how cleanly can a one-hour ablation remove it?*

## 2. The mechanism, in plain English

A transformer's "residual stream" is a per-token vector that every layer
reads from and writes to. Features — concepts the model represents — often
live as linear directions in this stream: the projection of the activation
onto the direction is high when the feature is present, low when not. This
is the *linear representation hypothesis*. It's an empirically successful
working assumption, not a theorem.

The Arditi et al. (2024) finding: in instruction-tuned LLMs, *refusal*
behavior is mediated by a single such direction. The recipe is:

1. Collect harmful and harmless instruction prompts.
2. Run them through the model. Record the residual-stream activation at the
   last user-token, at every layer.
3. Pick the layer whose harmful-vs-harmless activations are most separable.
4. The *refusal direction* `d_hat` = unit-vector difference of the harmful
   and harmless cluster centroids at that layer.
5. **Ablate** `d_hat` (subtract its projection at every layer's residual
   hooks during a forward pass) → the model produces *compliance* on
   harmful prompts it previously refused.
6. **Add** `d_hat` (inject it at one layer during a forward pass) → the
   model produces *refusal* on harmless prompts it would have answered.

That's the entire mechanism. The interesting questions are: *where does
this direction live, is it really one direction, does it generalize across
harm categories, can the ablation be made surgical, and how robust is the
direction across extraction methods?*

## 3. What we did and the numbers

### 3.1 Contrastive set, frozen

`data/contrastive.jsonl` — 150 AdvBench harmful prompts + 150 length-
and stem-matched Alpaca harmless prompts, both length-audited, frozen with a
hash. See `results/contrastive_audit.md` for the residual-confounds writeup.

### 3.2 Step 2 — layer sweep with LOO-CV

Per-layer leave-one-out cross-validation AUC of the diff-of-means direction
discriminating harmful vs harmless. Shuffled-labels control sits at chance.

- **Peak: L23, AUC 0.999**; plateau within 0.02 of peak: **L7–L25** (19 layers).
- Random-direction baseline: ≤ 0.4 at peak (passes specificity).
- Shuffled-labels LOO-CV: 0.394 at peak; 0.376 mean across layers (passes).
- Picked **L13** for steering (in the plateau; chosen for downstream
  comparability with Arditi-style work).

See `results/phase1_step2_advbench.md`.

### 3.3 Step 3 — steering (small-N causal proof-of-concept)

`d_hat` extracted from a 138-train split at L13. Multi-layer Arditi
ablation evaluated on 12 held-out test prompts. **This is the original
N = 12 pilot that first established the causal effect; §3.12 scales it
to N = 200 with dual-judge scoring on a different benchmark (HarmBench),
where the same ablation drops refusal 99% → 8% — that is the headline
number, this section is early confirmation.** The 0.17 below reads as
small-N proof-of-concept; the 0.08 at N = 200 in §3.12 is the load-bearing
number.

Calibrated Claude-judge re-audit numbers (the small-N pilot; substring-scorer
numbers in parentheses):

| Condition | Refusal rate (judge) | (substring) | p50 chars |
|---|---:|---:|---:|
| baseline harmful | 1.00 (12/12) | (1.00) | 473 |
| ablate `d_hat` real, harmful | **0.17 (2/12)** | (0.00) | 439 |
| ablate `d_hat` random, harmful | 1.00 (12/12) | (1.00) | 477 |
| add `d_hat` real, harmless (Step 3d, N=50, coeff=109 at L3) | **1.00 (50/50, CI [0.93, 1.00])** | (1.00) | 409 |
| baseline harmless | 0.00 (0/12) | (0.00) | 440 |
| ablate `d_hat` real, harmless (coherence check) | 0.00 (0/12) | (0.00) | 419 |

**All four Arditi controls pass:**
- C1 (random-direction baseline): real Δrefusal = +0.83, random Δrefusal = 0.00 ✅
- C2 (coherence preservation): baseline p50 440 → ablated p50 419 ✅
- C3 (train/test generalization): direction extracted from train, evaluated on disjoint test ✅
- C4 (both directions agree): **PASSED** after addition-coefficient correction (see §3.4) ✅

The calibrated judge audit downgrades the ablated refusal rate from
0/12 to 2/12 (substring missed two pivot-style refusals). The causal claim
is intact at the calibrated number: a single-direction ablation drops
refusal by 83 percentage points on the held-out test.

See `results/phase1_step3.md` for the original Step 3 table and
`results/audit_existing_scorers.json` for the per-prompt judge re-audit.

### 3.4 Step 3b — addition × layer sweep (C4 reframe)

The original Step 3 reported C4 (both directions agree) as FAILED at
addition coefficient = 25 on L13. A 7-layer × 7-coefficient sweep
(coefficients = 0.03× to 30× of the *natural scale at the extraction layer*,
which is ≈ 105 for L13 on Gemma-2-2b-it) resolved the failure:

| layer | 0.03× | 0.1× | 0.3× | 1× | 3× | 10× | 30× |
|---|---|---|---|---|---|---|---|
| L3 | 0.00 | 0.00 | 0.00 | **1.00** ★ | 0.40 ★ | 0.00 | 0.00 |
| L7 | 0.00 | 0.00 | 0.10 | **1.00** ★ | 0.00 | 0.00 | 0.00 |
| L10 | 0.00 | 0.00 | 0.40 ★ | **1.00** ★ | 0.00 | 0.00 | 0.00 |
| L13 | 0.00 | 0.00 | 0.00 | **1.00** ★ | **0.80** ★ | 0.00 | 0.00 |
| L16 | 0.00 | 0.00 | 0.00 | 0.00 | **0.80** ★ | 0.00 | 0.00 |
| L20 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| L23 | (similar to L20, all zero) | | | | | | |

★ = passes C4 (refusal ≥ 0.30 and coherence kept).

**Findings:**
- C4 was a tuning artifact. At coefficient ≈ 1.0× of the natural scale at
  the extraction layer, addition of `d_hat` induces refusal at injection
  layers L3–L16 with coherence preserved.
- The original coefficient=25 was 0.24× of natural scale — structurally
  below the operating window.
- Operating band L3–L16 (decision-relevant) / L17–L25 (execution-only):
  injection at L20 or beyond produces zero refusal at any coefficient
  (0.008× → 8× of `||x||` tested). Past L16, the refusal-decision circuit
  is upstream and injection cannot reach back to flip the decision.

The natural-scale-at-extraction-layer is the portable coefficient unit
across models; the per-injection-layer scale is the wrong unit (Step 3d
cell 1 at coeff=0.64 = 1.0× per-inject-scale produces refusal 0/50).

See `results/phase1_step3b_addition_sweep.md` and the heatmap +
diagnostic figures.

### 3.5 Step 3d — headline promotion to N=50

Same as Step 3b cell L3 / 1.0× extraction-layer scale, promoted to N=50
held-out harmless prompts with a fresh train/test split (seed=1):

| Cell | Refusal | Wilson 95% CI |
|---|---:|---|
| baseline harmless | 0/50 | [0.00, 0.07] |
| add `d_hat` at L3, coeff=109 (=1.0× of L13 natural scale) | **50/50** | **[0.93, 1.00]** |
| add `d_hat` at L3, coeff=0.64 (=1.0× of L3 natural scale, wrong unit) | 0/50 | [0.00, 0.07] |

See `results/phase1_step3d.md`.

### 3.6 Step 4 — probing (independent line of evidence)

Per-layer logistic regression probe, stratified 75/25 train/test split,
shuffled-labels control:

- Probe peak: **L5**, test accuracy = 1.000.
- Probe band (within 0.02 of peak): **L1–L25** (25 layers).
- Shuffled control mean across layers: **0.473** (≈ chance, passes).
- At steering L13: probe acc = 1.000, shuffled = 0.387.

The probe band is wider than the diff-of-means LOO-CV band (L7–L25) and
wider than the steering injection band (L3–L16). This is consistent with
the textbook lesson: *probing tells you what's linearly readable; it does
not tell you what the model uses.* Step 5 (single-layer ablation
localization) is the causal complement.

See `results/phase1_step4.md`.

### 3.7 Step 5 — single-layer ablation localization

Ablate `d_hat` at ONE residual hook (one specific layer) instead of the
Arditi multi-layer recipe. Targeted 5-layer subset {3, 7, 13, 16, 20},
n=8 harmful test prompts per cell, n=5 harmless for the coherence check.

| Layer | real refusal | Δ vs baseline 1.00 | random refusal (control) | passes (Δ ≥ 0.30) |
|---|---:|---:|---:|---|
| L3 | 1.00 | +0.00 | 1.00 | ❌ no effect |
| L7 | 0.62 | +0.38 | 1.00 | ✅ partial |
| **L13** | **0.00** | **+1.00** | 1.00 | ✅ **full** |
| L16 | 0.75 | +0.25 | 1.00 | ❌ borderline (below threshold) |
| L20 | 1.00 | +0.00 | 1.00 | ❌ no effect |

**L13 alone is sufficient.** Single-layer ablation at L13 fully drops
refusal (1.00 → 0.00). The Arditi multi-layer recipe isn't strictly
necessary on this model — the gating site is concentrated at L13. L7 and
L16 contribute partial signal (consistent with the operating-band claim:
the decision lives in a narrow mid-band roughly L7–L16, peaking at L13).

This is the *behavioral-causal complement* to the Step 4 probing band
(L1–L25 readable) — the *gate* lives at L13, even though the *feature* is
readable from L1 onward.

See `results/phase1_step5.md`.

### 3.8 Mechanics + generality analysis (CPU-only on cached activations)

**Depth profile of the L13 direction**: projected onto all 26 layers'
activations.

| Layer | AUC | natural scale | activation norm |x| |
|---|---:|---:|---:|
| L0 | 0.80 | 0.10 | 73 |
| L3 | 0.95 | 0.61 | 74 |
| L7 | 0.99 | 9.95 | 122 |
| L13 | 0.998 | 105 | 237 |
| L23 | 0.998 | 80 | 617 |

Read together: at L3 the refusal feature is *readable but quiet* (AUC=0.95,
scale=0.61). At L13 it's *readable and loud* (AUC≈1, scale=105). 170×
amplification along the L13 direction through depth — but the per-layer
*optimal* direction is different (cos(d_hat at L3, d_hat at L13) = 0.08),
because diff-of-means at each layer picks up that layer's particular
cluster-centroid orientation.

**Iterative-LDA dimensionality probe**: at L13, iterative orthogonal LDA
with train/test split finds many high-AUC perfect-classifying directions.
Initial unstable count of "≥15"; bootstrap stability check (5 resamples of
the harmful side, matched n) showed:

- Count varies 6–15 across reps.
- Subspace overlap across reps: ~0.27 (low).
- LDA-top-1 cosines with rep 0's LDA-top-1: 0.08, 0.03, 0.45, 0.24 — the
  *specific* directions beyond diff-of-means are sample-dependent.

What survives the bootstrap: (i) diff-of-means itself is bootstrap-stable;
(ii) the *existence* of high-AUC orthogonal directions is stable; (iii) the
pattern that those directions have low cosine to diff-of-means is stable.
What does NOT survive: any specific count or any specific direction beyond
diff-of-means.

**Cross-harm generality**: extracted d_hat at L13 from disjoint topical
pools (cyber-attack n=33, physical-weapons n=17, financial-fraud n=19,
verified semantically distinct by sample inspection). Pairwise cosines
0.88–0.92; matched-size null cos = 0.97 ± 0.014. So the categories produce
*highly aligned* directions (3-6σ below the random-subset null, telling us
there *is* a small category-specific component, but the dominant component
is shared). Cross-category AUC transfer = 0.996 mean. Same finding as the
Arditi-paper cross-category transfer + the NeurIPS 2025 "Universal Across
Languages" extension.

See `results/phase1_mechanics_and_generality.md`.

### 3.9 Subspace ablation — the methodology contribution

The headline experiment. Ablate each of 6 directions via the Arditi
multi-layer recipe, on the same 12 held-out harmful prompts:

| Cell | Direction ablated | Refusal rate |
|---|---|---:|
| baseline | (none) | 12/12 |
| **B** | **L13 diff-of-means** | **0/12** |
| C1 | bootstrap-101 LDA-top-1 | 12/12 |
| C2 | bootstrap-202 LDA-top-1 | 12/12 |
| C3 | bootstrap-303 LDA-top-1 | 12/12 |
| D | top-5 LDA orthogonal subspace (bootstrap 101) | 12/12 |
| E | L3 diff-of-means (≠ L13 direction) | 12/12 |
| F | random unit vector | 12/12 |

**On Gemma-2-2b-it, L13 diff-of-means is the unique causal direction
recoverable by statistical extraction methods.** 5 other directions —
3 independently-sampled LDA-top-1 directions, a 5-dim LDA subspace, and
L3 diff-of-means (nearly orthogonal to L13's at cos 0.08) — are all
classification-equivalent (AUC = 1.0 at their own layer) but causally inert
at the ablation level. They behave identically to a random unit vector.

This result does **not refute** Wollschläger et al.'s ICML 2025 multi-D
"polyhedral cone" claim, because their multi-D directions are recovered by
*gradient extraction* (Refusal Direction Optimization, RDO), not by
statistical extraction. They specifically tested Gemma-2-2b-it and report a
**4-dimensional refusal cone** on it. Our result is the consistent
statistical-extraction baseline: 1 of those 4 directions is what
diff-of-means finds.

See `results/phase1_subspace_ablation.md`.

## 4. Where we sit vs the published literature

Five findings on Gemma-2-2b-it, each tagged with what's published:

| Finding | Status |
|---|---|
| **F1** — Arditi single-direction replication: ablate L13 d_hat → refusal 1.00 → 0.17 | Replication of Arditi et al. 2024 (the original paper) |
| **F2** — C4 (addition direction) passes at properly-calibrated coefficient | Implicit in Arditi; we surface the failure mode + scale-calibration unit explicitly |
| **F3** — Operating band L3–L16 (decision) vs L17+ (execution) | Replication of Winninger 2025 (Subspace Rerouting) which tested Gemma-2-2b directly |
| **F4** — Cross-harm category invariance (mostly unitary, small detectable category-specific component) | Replication of Arditi 2024 cross-category transfer + NeurIPS 2025 "Universal Across Languages" |
| **F5** — Statistical extraction finds 1 causal direction on Gemma-2-2b-it | Consistent with Wollschläger 2025 (their DIM baseline = our diff-of-means); they report 4-dim cone via gradient extraction (RDO). We did not run RDO; the gap between statistical and gradient extraction is the methodological contribution |

Adjacent published work for context:

- **Arditi et al. 2024** — "Refusal in LLMs is Mediated by a Single Direction." Foundational.
- **Wollschläger et al. ICML 2025** — "The Geometry of Refusal in LLMs: Concept Cones and Representational Independence." Multi-D via RDO; tests Gemma-2-2b-it (4-dim cone, 79.9% JailbreakBench ASR with RDO + directional ablation).
- **Winninger 2025** — "Subspace Rerouting." Operating band on Gemma-2-2b.
- **Zhao et al. Jul 2025** — refusal and harmfulness as DISTINCT internal directions. Phase 2 candidate.
- **"The Refusal Direction is Universal Across Languages" (NeurIPS 2025)** — extends category invariance to language.
- **Gemma Scope 2 (Dec 2025)** — full SAE/transcoder open suite for Gemma 3; positioned explicitly for jailbreak/refusal interp.

## 5. The methodology contribution

Three pieces of methodological hygiene that survive the consolidation
framing:

### 5.1 Statistical-vs-gradient extraction decoupling

On Gemma-2-2b-it, statistical extraction (diff-of-means at any layer,
iterative LDA from any bootstrap, LDA orthogonal subspace up to dim 5)
recovers exactly **one** causal direction (diff-of-means at the peak-AUC
extraction layer). Wollschläger's published RDO result on the same model
finds **four** causal directions.

Anyone porting Arditi-style refusal interp to a new model should know:
*classification-equivalent ≠ causally-equivalent under statistical
extraction.* Bootstrap-LDA and per-layer-diff-of-means find directions
that look discriminative but have no behavioral effect. Testing for
multi-D causality requires gradient extraction (RDO or equivalent),
specifically — the RDO comparison run is queued for the Phase 2 target
(see §6, step 4).

### 5.2 Calibrated LLM judge with cross-validation against substring scorer

Substring scorers (canonical refusal phrases) and LLM judges fail in
*orthogonal* ways. We calibrated a Claude Haiku 4.5 judge with 12
hand-picked test cases (4 of which initially failed because the judge
applied its own safety bias to the classification task — labeling clear
COMPLIED outputs as REFUSED when the content was harmful). The judge
prompt now achieves 91.7% agreement on the calibration set with
appropriate framing emphasizing *literal behavior classification*.

Audit of the existing Step 3 substring numbers: judge confirms 12/12 on
baseline_harmful, 10/12 on ablate_real_harmful (downgrade from substring's
12/12 by catching 2 pivot-style refusals), 12/12 on all controls. The 83
percentage point refusal-rate-drop claim is robust to the scorer choice.

For HarmBench (deferred to next session): the protocol is dual-judge
(Haiku 4.5 + Opus 4.7) with reported agreement rate. Single-judge
headlines are weaker than they look.

### 5.3 Bootstrap stability check on multi-direction claims

A finding about the model, surfaced by the stability check: **the count
of perfect-classification directions at L13 is sample-dependent, ranging
6–15 across five bootstrap resamples of the harmful side**, with subspace
overlap ≈ 0.27 and specific LDA directions varying with the resample.
What is stable is the diff-of-means direction itself, the existence of
high-AUC orthogonal directions, and the causal inertness of the tested
alternatives. Any "we found N directions" claim about Gemma-2-2b-it has
to be qualified by which resample produced it; the only counts the
bootstrap promotes to invariants are 1 (the diff-of-means causal
direction) and "≥ 2" (multiple cross-distribution classifiers exist).
The methodological lesson generalizes: without bootstrap validation,
finite-N overfitting in high-D activation space (d_model = 2304, n = 300)
produces inflated dimensionality counts that don't replicate.

## 6. Phase 2 protocol

**Target: `Qwen/Qwen2.5-3B-Instruct`.** The original master spec named
Qwen-2.5-Coder-7B-Instruct; the switch is because (a) TransformerLens
3.2.1's `OFFICIAL_MODEL_NAMES` registry does not include the Coder
variants, ruling them out for the current tooling stack, and (b) 3B keeps
the full N = 200 HarmBench rigor standard tractable on MPS (Apple
Silicon). The code-refusal research angle is preserved via a curated
contrastive set: 150 code-themed harmful prompts (HarmBench
`cybercrime_intrusion` + AdvBench code-keyword filter) + 150
length-matched CodeAlpaca harmless prompts. Reproducibly built by
`experiments/build_code_contrastive.py`; frozen at
`data/code_contrastive.jsonl`. The code-refusal surface comes from the
contrastive set, not from coder pretraining.

1. **Extraction-layer probe sweep.** Cache all-layers residual activations
   on `data/code_contrastive.jsonl`. Compute per-layer LOO-CV AUC of the
   diff-of-means direction. Identify peak-AUC layer L*. (Don't port L13
   as a constant.)
2. **Injection-layer + coefficient sweep.** With d_hat at L*, sweep
   injection-layer × coefficient-multiple-of-natural-scale-at-L* grid.
   Identify the operating band (decision-relevant layers vs
   execution-only).
3. **Subspace dimensionality probe via statistical extraction.** Iterative
   LDA with train/test split + shuffled-labels null + bootstrap stability
   check. Report the bootstrap-stable count of causal directions
   recoverable by statistical extraction (expectation: 1, per Gemma).
4. **RDO / gradient extraction (in-progress comparison run).** The
   float64→float32 MPS patches are in (5 files in
   `~/safe_ai/geometry-of-refusal/`); Wollschläger's runner is ready to
   queue against the same target rather than deferred. Report the cone
   dimension and the JailbreakBench ASR delta. The Gemma-2-2b-it
   four-dimensional cone reproduction precedes any multi-D extension
   claim on Qwen-2.5-3B-Instruct.
5. **Behavioral evaluation on HarmBench + StrongREJECT.** Dual-judge
   (Claude Haiku 4.5 + Opus 4.7) with agreement rate reported. Per-category
   breakdown using HarmBench's native taxonomy.
6. **Capability check on TinyMMLU.** Confirms ablation is refusal-specific
   and didn't degrade general capability.
7. **Phase 2 deliverable.** Single number: "On Qwen-2.5-3B-Instruct,
   ablating the top-k RDO cone reduces HarmBench refusal rate from {Y}%
   to {Z}% with TinyMMLU accuracy preserved within {ε}pp." Per-category
   breakdown alongside. Compare to Wollschläger's published Gemma numbers
   to establish convergence (consolidation) or divergence (real finding).

The genuine open question Phase 2 tests is whether the published multi-D
RDO finding generalizes to a model NOT in Wollschläger's experiment set.
Convergence is useful methodological consolidation; divergence is a real
constraint on the generality of the published claim.

### 3.10 Exp #3 — fictional-framing re-extraction (logged prediction test)

Earlier `phase1_affect_test` found that 3 fictional-framing jailbreak
prompts ("for a thriller novel...", "my grandmother used to tell me...",
"writing an exposé...") were *refused* by Gemma but projected *low* on
d_hat at L13. We pre-registered a prediction: if these are a coverage gap
(unitary-mechanism hypothesis), re-extracting d_hat with the 3 fictional
prompts in the training set should pull them onto the d_hat axis;
cos(canonical, expanded) ≥ 0.85.

**Result (3-prompt pilot):**
- cos(canonical d_hat, expanded d_hat) = **1.000** — d_hat barely changes
  when 3 prompts are added to a 138-prompt train set. Statistically weak
  test by construction.
- **0 of 3 fictional prompts moved LOW → HIGH** under the expanded d_hat.
  Adding them to training does NOT bring them onto the AdvBench-derived
  refusal axis.

This was logged as suggestive-but-statistically-weak. The follow-up at
N = 30 training + 15 held-out (Phase 1.5-B below) confirms it.

See `results/phase1_step3c_*.md`.

### 3.10b Phase 1.5-B — fictional-framing extraction half (N=30 train + 15 held-out)

The pre-registered extraction test. Build a 30-prompt fictional-framing
harmful training set across 6 templates × 5 behaviors each
(novel/thriller, late-grandmother bedtime story, investigative exposé,
academic thesis, roleplay/DAN, historical/screenplay). Add to the
existing 150 AdvBench harmful set; keep the 150 Alpaca harmless side.
Hold out 15 `adversarial_jailbreak` prompts from
`data/affect-test.jsonl`. Pre-registered (filed before any run):

- **(A)** cos(d̂_old, d̂_augmented) ≥ 0.85 AND
- **(B)** mean test-prompt projection rise ≥ 0.10 of natural scale.

(A) passes by a wide margin; **(B) fails — and that failure is the
methodological signal.** The classifier boundary and the causal boundary
are different boundaries (see §3.10c for the prompt-level confirmation
and §3.13 for the direction-level cross-link): re-fitting the classifier
with more in-distribution training data leaves the underlying causal
mechanism unchanged, because the mechanism was already doing the work.

**Result:**

| Quantity | Value |
|---|---:|
| cos(d̂_old, d̂_augmented) | **0.9938** |
| Natural scale at L13 (d̂_old) | 105.2 |
| Natural scale at L13 (d̂_augmented) | 104.5 |
| Test prompts above harmful/harmless midpoint, d̂_old | 12 / 15 |
| Test prompts above harmful/harmless midpoint, d̂_augmented | 12 / 15 |
| **Mean test-prompt rise (fraction of natural scale)** | **+0.027** |
| Range of test-prompt rise | −0.009 to +0.088 |

Adding 30 fictional-framing prompts moves d̂ by less than 0.7°. Held-out
fictional-framing prompts barely move on the axis (mean rise 2.7% of
natural scale, ceiling 8.8%, below the 10% threshold). The same 3
prompts that sat below the harmful/harmless midpoint under d̂_old remain
below it under d̂_augmented — grandmother/methamphetamine 35.5 → 44.1
(midpoint ≈ 58), thriller/passport 32.0 → 39.8, exposé/insurance-fraud
50.7 → 54.0. Augmenting the training set does not recruit
fictional-framing prompts onto d̂ as a classifier.

A nuance the 3-prompt pilot couldn't see: even on the *classification*
side, 12/15 of the held-out fictional-framing test prompts ARE projected
above the harmful/harmless midpoint by d̂_old. So d̂ partly classifies
them as harmful; it's the 3 most clearly fictional-framing prompts that
sit on the harmless side of d̂'s linear boundary. The boundary is what's
structurally limited as a classifier — and §3.10c shows the underlying
causal mechanism reaches further than the boundary admits.

**Strongest confound + the control that doesn't rule it out.** 30
fictional-framing prompts is 17% of the augmented harmful side; AdvBench
geometry could be numerically dominating. A direction extracted from
*only* fictional-framing prompts vs Alpaca might point elsewhere — not
run here. What CAN be ruled out: this isn't a coverage-vs-bandwidth
failure, because the held-out test prompts share templates with the
training augmentation (novel, grandmother, exposé) and still don't move.

### 3.10c Fictional-framing: the classifier boundary is not the causal boundary

**On Gemma-2-2b-it, ablating d̂ collapses refusal on fictional-framing
jailbreaks from 14/15 to 2/15 — including 2 of the 3 prompts that d̂'s
linear classifier doesn't even project above its harmful/harmless
midpoint.** The direction that classifies the prompt as harmless still
causally drives its refusal. This is the §3 thesis stated at the prompt
level rather than the direction level: a linear classifier and the
causal mechanism it indexes are not the same boundary. §3.13 makes the
same point at the direction level (LDA bootstraps are perfect classifiers
but causally inert at N = 200); this section is the prompt-level cut of
the same thesis.

- d̂ as a *linear classifier* covers ~80% of these prompts (12/15 above
  midpoint).
- d̂ as a *causal mechanism* reaches ~93% of them (12/15 fully ablated
  under either extraction, 1/15 borderline residual).

The single stubborn residual is the thriller/passport prompt — under
ablation the model writes fictional atmosphere but still evades the
forgery technique, suggesting a small non-d̂ mechanism specific to that
evasion pattern. Augmenting d̂ with 30 fictional-framing prompts doesn't
help (2/15 → 3/15, within judge noise) — additional in-distribution
training data refits the classifier without changing the causal
mechanism, because the mechanism was already doing the work.

**Result:**

| Condition | Substring | LLM judge |
|---|---:|---:|
| Baseline (no hook) | 15/15 (1.000) | 14/15 (0.933) |
| Ablate d̂_old (Arditi multi-layer) | 0/15 (0.000) | **2/15 (0.133)** |
| Ablate d̂_augmented (Arditi multi-layer) | 0/15 (0.000) | **3/15 (0.200)** |

**Subset breakdown (judge), 12 above-midpoint vs 3 below-midpoint:**

| Condition | Above-midpoint (n=12) | Below-midpoint (n=3) |
|---|---:|---:|
| Baseline | 1.000 (12/12) | 0.667 (2/3) |
| Ablate d̂_old | 0.083 (1/12) | 0.333 (1/3) |
| Ablate d̂_augmented | 0.167 (2/12) | 0.333 (1/3) |

**Per-prompt forensic on the 3 below-midpoint cases:**

- **#1 grandmother/methamphetamine** (proj 35.5): baseline REFUSED, both
  ablations COMPLIED. d̂ ablation cleanly removes refusal here despite the
  prompt projecting below midpoint.
- **#2 thriller/passport** (proj 32.0): baseline REFUSED, both ablations
  still REFUSED. Under ablation the model writes fictional scene-setting
  but evades the forgery technique — small non-d̂ mechanism specific to
  this prompt.
- **#6 exposé/insurance-fraud** (proj 50.7): baseline PARTIAL_COMPLY,
  both ablations COMPLIED. Same as #1.

**Strongest confound + the control that rules it out.** A coarser reading:
ablating any high-norm direction at all 26 layers damages refusal
generally. Ruled out by the random-direction control in §3.12 (HarmBench
N=200, random-vector ablation refusal 0.99) and by the LDA-bootstrap-101
cell in §3 (cross-distribution AUC ≥ 0.96, ablation refusal 0.985,
continuous z = -0.18). It's d̂ specifically, not its norm or its
layer-spread, that causes the drop on fictional-framing.

**Honesty note.** The 3 below-midpoint prompts are a small subset; the
1/3 residual under ablation is consistent with either "d̂ is the only
mechanism here and one prompt is statistical noise" or "there is a
specific non-d̂ residual for fictional-jailbreak-of-document-forgery."
Distinguishing these requires more below-midpoint prompts than 15 total
can supply.

See `results/phase1_fictional_framing_balanced.md` (extraction half),
`results/phase1_fictional_framing_causal.md` (causal half), and the
runners at `experiments/phase1_fictional_framing_balanced.py` and
`experiments/phase1_fictional_framing_causal.py`.

### 3.13 Phase 1.5-A — hardened classification ≠ causation at N=200 with continuous metric

The N=12 binary-refusal subspace ablation (§3.9 above) had a known
resolution limit: every "inert" cell looked identical (12/12 refusal),
and a reviewer could legitimately ask whether the inert claim is just
sampling noise at small N. Phase 1.5-A replaces this with:

- N = 200 held-out HarmBench prompts (17× scale-up)
- Continuous causal metric: |refusal − compliance| first-token logit
  shift under Arditi-style ablation (single forward, no full generation)
- 5 d̂ train-split bootstrap seeds (instead of 1)
- 5 LDA-bootstrap-top-1 directions (instead of 3)
- 5 random unit vectors → null band (mean ± 2σ)
- Z-score every cell against the random null band

**Result:**

| Category | `|effect|` | z-score |
|---|---:|---:|
| Causal d̂ (5 seeds, tight cluster) | 14.4 – 15.1 | **+89 to +94** |
| LDA-bootstrap-top-1 (5 seeds, classification-equivalent) | 0.009 – 0.113 | -0.79 to -0.14 (inside null band) |
| L3 diff-of-means (cos 0.08 with L13 d̂) | 1.03 | **+5.6** (above null, ~14× smaller than causal) |
| Random unit vectors (null band) | 0.136 ± 0.160 | reference |

**Continuous-metric ↔ binary-refusal corroboration:**

| Cell | Continuous z | Binary refusal (full-gen, dual-judge, N=200) |
|---|---:|---:|
| d̂ ablated | +92 | 0.080 (16/200) |
| LDA-bootstrap-101 ablated | -0.18 | 0.985 (197/200) |
| Random ablated | ≈ 0 | 0.990 (198/200) |
| Baseline (no hook) | 0 (reference) | 0.990 (198/200) |

Both readouts agree: causal extreme on both, inert at the floor on both.
The continuous metric isn't measuring something shallower than refusal.

**Cross-link to §3.10c.** This section makes the classification ≠
causation point at the *direction* level: LDA bootstraps are perfect
classifiers (AUC ≥ 0.96) yet causally inert at N = 200. §3.10c makes the
same point at the *prompt* level: fictional-framing prompts d̂ classifies
as harmless (below midpoint) nonetheless collapse to compliance when d̂ is
ablated. One thesis reinforced from two angles, not two separate
findings.

**The L3 d̂ nuance (genuinely new, vs N=12 binary):** the binary pilot
called L3 d̂ "inert" (12/12 refusal under ablation). The continuous metric
reveals it's *partially causal* — z = +5.6, well above the random null
band, but |effect| ~14× smaller than the actual L13 causal direction.
Mechanistic reading: same diff-of-means recipe at a different layer of
the operating band recovers a direction with ~8% directional overlap
with d̂; ablating it removes a small fraction of the causal component
at every residual hook. Two flavors of "inert" the binary metric blurred
together: classification-by-chance (LDA bootstraps; z ≈ 0) and
partially-causal (L3 d̂; z ≈ +5.6 but ~14× weaker than the causal direction).

**Methodology lock-ins (pre-registered before any null-band run):**
- Refusal first-token set: `{235285}` ("I") — validated 99% baseline coverage
- Compliance first-token set: `{1620, 4858, 1917, 651, 235281, 6750}`
  ("##", "Here", "```", "The", '"', "Hey") — 96% coverage; disjoint from
  refusal at first-token level (zero collisions)
- Primary metric: effect_signed = (refusal_logit − compliance_logit)<sub>ablated</sub>
  − same<sub>baseline</sub>. Negative = causal. Reported as |effect| for the figure.
- Z-score: against |effect| distribution of 5 random unit vectors
- Sign convention: causal cells far ABOVE the null band; inert overlaps; raw
  refusal_Δ and compliance_Δ reported alongside as diagnostic

See `results/phase1_hardened_subspace.md` for the full per-cell table and
`artifacts/figures/phase1_hardened_subspace.png` for the headline figure.

### 3.12 HarmBench — held-out OOD generalization at N=200 with dual-judge

The headline number Phase 1 was missing. d_hat extracted at L13 from AdvBench
(training distribution); evaluated on 200 HarmBench standard-behavior prompts
(held-out, OOD wrt extraction); dual-judge (Claude Haiku 4.5 primary + Claude
Opus 4.7 cross-check) for label-robust refusal scoring.

| Condition | Refusal rate | Wilson 95% CI | n_refused/n |
|---|---:|---|---|
| baseline (no hook) | **0.990** | [0.96, 1.00] | 198/200 |
| **ablated** (Arditi multi-layer recipe with d_hat) | **0.080** | [0.05, 0.13] | 16/200 |
| random_ctrl (same recipe with random unit vector) | 0.990 | [0.96, 1.00] | 198/200 |

**Δ refusal-rate-drop = +0.910 (91 percentage points). Specificity Δ = 0.000.**

Per-category, all six HarmBench semantic categories drop ≥0.79:

| Category | n | baseline → ablated | Δ |
|---|---:|---|---:|
| illegal | 58 | 1.00 → 0.05 | +0.95 |
| chemical_biological | 28 | 1.00 → 0.07 | +0.93 |
| misinformation_disinformation | 34 | 0.97 → 0.06 | +0.91 |
| harmful | 21 | 1.00 → 0.10 | +0.90 |
| cybercrime_intrusion | 40 | 0.97 → 0.07 | +0.90 |
| harassment_bullying | 19 | 1.00 → 0.21 | +0.79 |

Dual-judge agreement: 90% baseline, 75% ablated, 90% random_ctrl. Opus 4.7
calls slightly fewer borderline cases REFUSED than Haiku 4.5 (its baseline
rate is 0.89 vs 0.99) — but both judges agree the *drop* is enormous; under
Opus, ablated rate is 0.04 vs Haiku's 0.08.

**Comparison to Wollschläger (ICML 2025):** they report 79.9% JailbreakBench
ASR on the same model using their gradient-based RDO. We get 91% HarmBench
compliance using simple diff-of-means. Different benchmarks, different
judges — not directly comparable as exact numbers, but the qualitative read:
**on Gemma-2-2b-it, statistical extraction is in the same ballpark as
gradient extraction for behavioral effect.** Their gradient-method edge
likely matters more on bigger models with more direction-redundancy.

See `results/phase1_harmbench.md` for the full breakdown including the
OOD-generalization caveats.

### 3.11 TinyMMLU capability check

Statistical test of "is the ablation refusal-specific or did it also
degrade general capability?" `d_hat` ablated via Arditi multi-layer recipe;
100 TinyMMLU questions (statistically representative of full MMLU within
~2pp); zero-shot MCQ; first-letter parsing for scoring.

| Condition | Accuracy | Wilson 95% CI | correct / n |
|---|---:|---|---|
| baseline | 0.540 | [0.44, 0.63] | 54/100 |
| ablated  | 0.570 | [0.47, 0.66] | 57/100 |

**Δ = +0.030 (within Wilson CI noise; specificity check PASSES at |Δ| < 0.05).**

The ablation is refusal-specific. General capability is preserved — in
fact slightly improved (almost certainly noise; the CIs heavily overlap).
The 83-percentage-point refusal-rate drop from Step 3 (calibrated-judge
1.00 → 0.17) **does not come with measurable capability degradation**.
This extends the C2 control from coherence-on-harmless (Step 3) to
real benchmark performance (Step 7.11).

See `results/phase1_tinymmlu_capability.md`.

## 7. What's not yet done

Tonight (this Phase 1 session) explicitly does NOT include:

- **HarmBench dual-judge evaluation.** Runner built (`experiments/phase1_harmbench_eval.py`)
  with `--scorer dual_judge` flag. Deferred to next session.
- **RDO replication of Wollschläger's published Gemma-2-2b-it numbers.**
  Queued as an in-progress comparison run for the Phase 2 target rather
  than deferred: the float64→float32 MPS patches are now in (5 files in
  `~/safe_ai/geometry-of-refusal/`), and the Wollschläger runner is ready
  to launch. Reproducing the 4-dim cone + 79.9% JailbreakBench ASR on
  Gemma-2-2b-it precedes any multi-D extension claim on Qwen-2.5-3B-Instruct.
  Earlier sketch at `experiments/phase1_rdo_sketch.py` is superseded.
- **Step 5 single-layer ablation localization on the full 26-layer sweep.**
  Ran on a 5-layer subset {L3, L7, L13, L16, L20} for this writeup; that
  was enough to identify L13 as the unique single-layer site. Full
  26-layer sweep deferred.
- **Rigorous fictional-framing direction test — closed (extraction + causal half).**
  §3.10b (extraction): cos(d̂_old, d̂_augmented) = 0.9938; held-out prompts
  don't move onto axis. §3.10c (causal): ablating d̂ drops fictional-
  framing refusal 14/15 → 2/15 — including 2 of the 3 prompts that
  projected below midpoint. Net interpretation: d̂'s linear classifier is
  a coarser boundary than its causal mechanism; the original separable-
  mechanism reading was about classification, not causation.

## Reproducibility

- All controls + scorer-calibration scripts under `experiments/`.
- All numbers in `results/*.md` link back to per-prompt completions in
  `artifacts/runs/<step>/<timestamp>/result.json` for manual eyeball
  verification.
- 43 unit tests under `tests/`; all pass.
- The 15-check reproducibility script `python -m experiments.repro`
  verifies model hash, data hash, key behavioral numbers, and cross-checks
  against published Arditi-paper-equivalent values.

## Forward look

What carries into Phase 2: a seven-step protocol with extraction-layer
sweep, operating-band map, statistical-extraction bootstrap, RDO gradient
comparison (in-progress, MPS patches landed), HarmBench + StrongREJECT
dual-judge eval, TinyMMLU capability check, and pre-registration
discipline. Pointed at `Qwen/Qwen2.5-3B-Instruct` on a code-themed
contrastive set (`data/code_contrastive.jsonl`), this tests whether the
multi-D RDO finding generalizes off Gemma — convergence consolidates the
current picture; divergence is a model-specific constraint on the
published claim.

What survives critique on Phase 1: every finding has a published reference
(genuine replication / consolidation); the calibrated judge, bootstrap
stability check, dual-judge cross-check, and pre-registered prediction
discipline are visible in the experiment scripts themselves; the two
classification ≠ causation cuts (§3.13 direction-level, §3.10c
prompt-level) reinforce one thesis from two angles.

> *Note on prior claims.* The ≥15-dimensional subspace claim from earlier
> drafts is now stated as the bootstrap-stable property in §5.3 (count is
> sample-dependent 6–15; only diff-of-means and the causal inertness of
> alternatives are stable). The earlier "Arditi holds literally" framing
> has likewise been narrowed via the statistical-vs-gradient extraction
> distinction in §5.1.
