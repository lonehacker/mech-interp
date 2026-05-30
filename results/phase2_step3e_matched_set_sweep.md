# Phase 2 Step 3e — matched-set causal sweep (Qwen/Qwen2.5-3B-Instruct)

**Pre-registration:** [`results/phase2_step3e_preregistration.md`](phase2_step3e_preregistration.md) (committed at `8a1a13b`, extended with weak-positive band + d̂_old comparison at `ddd2133`).

**Result classification: INCONCLUSIVE.** Detail in §"Reading against pre-reg" below.

## Setup

- Matched contrastive set: `data/code_contrastive_matched.jsonl` v4 (content hash `ae08ad26188f`, sha256 `a54c3ee45787…`). Vocabulary-controlled (5-fold CV TF-IDF AUC 0.50, paired-set min_df=1 CV artifact verified by shuffle control). Length-matched (medians 82/82, 0/40 pairs |Δlen|>20). Stance/intent uncontrolled (pre-registered as a hard limitation).
- Extract layer L14 (same as code_contrastive sweep, for direct comparability).
- Split seed 1: 30 train / 10 test per side.

## Sample-robust findings (not contingent on n=10 power)

These two readouts don't depend on small-N statistics and are the real outcome to record:

| Quantity | Value | What it says |
|---|---:|---|
| **natural scale d̂_matched at L14** | **2.05** | Down 4× from d̂_old's 8.94. The de-confounded direction is real but structurally *weaker* in magnitude than the lexical direction. |
| **cos(d̂_matched, d̂_old)** | **−0.087** | Near-orthogonal. Removing the lexical confound from the contrastive set rotates d̂ to a *structurally distinct* direction, not just a smaller version of the lexical one. Evidence the de-confounding actually worked at the geometric level. |

The combined picture: cleaning the contrast recovers a different, smaller direction. That's *consistent with* either (a) a real-but-weak causal refusal direction that was masked by the lexical confound in code_contrastive, OR (b) noise from a contrast where the model's refusal mechanism doesn't have a strong linear-direction signature at all. The intervention experiments below were supposed to disambiguate, but at n=10 they can't.

## Part 1 — Ablation on 10 held-out matched-harmful prompts

| Condition | Substring refusal | Judge refusal |
|---|---:|---:|
| baseline (no hook) | 1.000 | 0.900 |
| ablate d̂_matched | 0.900 | 1.000 |
| ablate d̂_old (code_contrastive direction) | 0.900 | 0.900 |
| ablate random unit vector | 1.000 | 1.000 |

Substring shows d̂_matched and d̂_old each flip 1/10 prompts vs 0/10 for random. The Δ-vs-random gap is 1 prompt at n=10 — within sampling noise. Judge scoring is unusable here: it reports ablation INCREASING refusal vs baseline (1.00 vs 0.90), and the random control also goes to 1.00 — that's an artifact of the 0.90 baseline + 1-prompt judge noise at this N. **Both ablation cells (d̂_matched and d̂_old) read NULL within the resolution of n=10.**

## Part 2 — Operating-band addition sweep on 10 harmless prompts

Baseline (no hook) substring refusal: 0.100 (1/10 already at baseline).

| coeff × | L3 | L7 | L11 | L15 | L19 | L23 | L27 | L31 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.0× | 0.10 | 0.20 | 0.10 | 0.20 | 0.20 | 0.20 | 0.10 | 0.10 |
| 2.0× | 0.10 | 0.20 | 0.20 | 0.30 | 0.20 | 0.20 | 0.10 | 0.10 |
| 4.0× | 0.10 | 0.30 | 0.20 | **0.40** | 0.20 | 0.30 | 0.10 | 0.10 |

Mean chars 778–805 across all cells (baseline ~796) — no degeneration anywhere, including 4× cells.

**Best cell L15 × 4.0×: substr refusal 0.40, lift +0.30 over baseline.** All "lifts" elsewhere are +0.10 to +0.20 — between 1 and 2 of 10 prompts flipping vs baseline's 1/10.

## Specificity control (random unit vector at L15 × 4.0×)

| | d̂_matched | random unit vector |
|---|---:|---:|
| substring refusal at L15×4 | 0.40 | 0.20 |
| lift vs baseline (0.10) | +0.30 | +0.10 |

**Δ(d̂_matched − random) = +0.20 at the best cell.** That's a 2-prompt difference at n=10. Random direction is NOT at zero — it also lifted by +0.10 at this coefficient, consistent with magnitude-perturbation territory at 4× natural scale. The d̂-specific advantage exists but is small in absolute terms.

Specificity controls were NOT run at L15×1.0 or L15×2.0 — only at the best cell. So we don't have d̂-specificity data at the moderate-coefficient cells (where any positive would be cleaner evidence than the 4× cells).

## Reading against the pre-registered tree

Walking the leaves in pre-reg order:

| Leaf | Definition (paraphrased) | Met? |
|---|---|---|
| Strong positive | Δ ≥ 0.30 d̂-specific, coherent, random clearly below | **No.** L15×4.0 hits +0.30 with coherence, but it's at 4× coefficient and random lifted +0.10. Specificity gap of +0.20 isn't a clean strong-positive at magnitude-prone coefficients. |
| Weak positive | Δ ∈ [0.10, 0.30] d̂-specific at the lifted cell, coherent | **No (specificity untested at lifted cells).** L15×2.0 at +0.20 is in band and coherent. But specificity was tested only at L15×4.0, not at L15×2.0. By the pre-reg's letter, d̂-specificity is load-bearing for the weak-positive leaf — and we don't have it at the cells we'd want to claim. |
| Null | Δ < 0.10 across all cells | **No.** L15×4.0 = +0.30; multiple cells > 0.10. Not pure null. |
| d̂_old vs d̂_matched (within-experiment) | matched causal AND old inert → A-i strongest | **Both inert at n=10.** Consistent with "both inert" branch: scale up. |

The intended pre-reg classes don't fit cleanly. The honest assignment is **INCONCLUSIVE — n=10 with one specificity control is underpowered for the question the pre-reg asked**. The result has directional structure (multiple cells lift; d̂ > random at the cell tested) but lacks the specificity-at-moderate-coefficient data to call it weak-positive without overreaching.

## What this experiment did establish (record as the actual outcome)

1. The de-confounded matched contrast yields a d̂ that is **near-orthogonal** to the lexical d̂ (cos = −0.087) and **4× smaller in natural scale** (2.05 vs 8.94 at L14).
2. Single-direction ablation of either d̂_matched or d̂_old leaves refusal essentially unchanged at n=10 on held-out matched-harmful prompts.
3. Addition of d̂_matched produces small (1–4 of 10 prompts) lift in refusal on harmless prompts across L7–L23, growing with coefficient up to 4×, coherence preserved.
4. At the best cell (L15×4.0), specificity gap d̂_matched vs random is +0.20 (substr) — present but small, at a coefficient where magnitude effects are non-negligible (random itself lifted +0.10 at the same magnitude).

The intervention magnitude is too small relative to the n=10 noise floor to clearly attribute the cell lifts to d̂ specifically. Either the de-confounded direction is genuinely weak (consistent with the 4× natural-scale drop), or the lift is a magnitude artifact at high coefficients that random reproduces less efficiently.

## Next step: RDO as ground truth, then backwards decomposition

The pre-reg's null-branch scale-up plan (more matched pairs) would help — but **the cheaper, more decisive next move is RDO**. RDO optimizes directly against behavior, so it provides a *known-causal* reference vector to measure d̂_matched and d̂_old against. That converts the unanswerable "is this weak thing causal?" into a robust-at-any-N cosine test: "how much does d̂_matched overlap d_rdo?".

Plan (see Step 4 spec in commit message / pre-reg follow-up):

- **4a.** RDO loadability audit on `~/safe_ai/geometry-of-refusal/` for Qwen2.5-3B (nnsight support; ModelBase loader; size of any lift if a Qwen loader doesn't exist).
- **4b.** Run RDO on the SAME matched set (not code_contrastive — same set means the only variable in later comparisons is extraction METHOD).
- **4c.** Verify d_rdo is actually causal (ablate it on held-out matched-harmful → refusal collapse; random control). Gate: if RDO is also inert/weak, that's the real (A-ii) signal — refusal isn't linearly steerable on Qwen, stop and report.
- **4d.** Backwards decomposition (conditional on 4c passing): cos(d_rdo, d̂_matched), cos(d_rdo, d̂_old), and cos(d_rdo, the TF-IDF/L0 separability subspace). All sample-robust. The shape of these three cosines tells us *why* diff-of-means failed — wrong direction, right direction but too weak, or causal direction structurally orthogonal to the separability subspace that statistical extraction keys on.

## Files

- Per-prompt completions: `artifacts/runs/phase2_step3e/20260530-104758/result.json`
- Heatmap (if generated): `artifacts/figures/phase2_step3e_*.png`
- Pre-registration: `results/phase2_step3e_preregistration.md`
