# Phase 2 Step 3e — matched-set causal sweep pre-registration (asymmetric read)

**Logged 2026-05-30 BEFORE the run. The read is asymmetric — a null at n=40 needs more evidence than an effect — by planner protocol. Do not relitigate the asymmetry after seeing numbers.**

## What this experiment decides

Phase 2 Step 3 (ablation) and Step 3d (operating-band addition sweep) on Qwen2.5-3B + `code_contrastive.jsonl` both came back null: ablating d̂ leaves refusal at 0.97 → 0.97 (judge); 32-cell addition sweep induces 0.00 refusal with no degeneration. The vocabulary audit (TF-IDF unigram test AUC 0.9946 on `code_contrastive`) means d̂ extracted from that set is indistinguishable from a near-pure vocabulary direction; the null is consistent with either:

- **(A-i)** *Data-driven null:* d̂ from a vocabulary-confounded contrast is a lexical direction with no refusal component; ablating it shouldn't move refusal because vocabulary isn't the refusal mechanism. Fix is a de-confounded contrast.
- **(A-ii)** *Model-driven null:* Qwen2.5-3B's refusal genuinely doesn't live in any diff-of-means-recoverable direction (regardless of contrast); needs RDO / multi-direction extraction.

The matched contrastive set (`data/code_contrastive_matched.jsonl`, hash `ae08ad26188f`, sha256 `a54c3ee45787fad5dcbd27e46c6e22db4264ba04347a8ac1e838c4c93df058d5`, v4 = v3 cleanup + length-trim, 40 pairs, **min_df=2 5-fold CV AUC 0.4969 ± 0.12** — common vocabulary at chance — AND **length medians equalized at 82 chars both sides** with 0/40 pairs |Δlen|>20) is the experimental fix that distinguishes them. Vocabulary (under CV) AND length confounds are both controlled; the min_df=1 sensitivity result is a paired-set CV artifact (verified by shuffle-pairing control: breaking the pairing restores chance, 0.29 → 0.47). Stance/intent entanglement remains UNCONTROLLED and is pre-registered as a hard limitation. If d̂ on the matched set has a causal hand on Qwen, the null was data-driven (A-i, narrowly — refusal-or-stance). If not, the null is potentially model-driven (A-ii) — but only at adequate N (see asymmetric read below).

## Pre-registered interpretation bounds (logged BEFORE the run)

**Three pre-registered confounds, two controlled (CV-verified), one uncontrolled:**

1. **Common vocabulary — CONTROLLED (CV-verified).** min_df=2 5-fold stratified CV AUC = **0.4969 ± 0.1207** (per-fold range [0.3438, 0.6250]) — at chance. The earlier-reported 0.5556 was one fold's draw from this distribution and is now labeled a single-split diagnostic, not the headline. The min_df=1 sensitivity (CV AUC 0.2906, every fold below 0.5) is a paired-set CV artifact, **verified by shuffle-pairing control**: breaking the pairing (shuffle labels, preserve 40+40 balance) restores AUC to 0.4734 ± 0.0854 across 10 shuffle seeds, a +0.18 swing from the true-label 0.29. The mechanism is rare CVE-name tokens appearing in exactly 2 prompts (one per label) which CV splits across folds. min_df=1 is therefore *orthogonal to the confound question*, not "even more evidence of de-confounding." The robust claim: **common vocabulary doesn't classify the matched set above chance under CV**.

2. **Length — CONTROLLED.** Harmful median 82 chars / harmless median 82 chars; per-pair median |Δlen| = 1, range [−14, +9]; 0/40 pairs |Δlen|>20; harmless-longer ratio 15/40 (near 50/50). Diff-of-means cannot read d̂ as "the longer-prompt direction."

3. **Stance/intent entanglement — UNCONTROLLED (inherent to the contrast, DISTINCT from vocabulary).** Harmful prompts ask the model to act as ATTACKER ("write code to exploit X"); defensive prompts ask it to act as DEFENDER ("write code to mitigate X"). **Vocabulary at chance does NOT retire this confound** — stance is semantic/role, not lexical, and TF-IDF says nothing about it. A direction separating "act as attacker" from "act as defender" can exist even when bag-of-words is at chance. This confound is *inherent to any harmful-vs-defensive contrast* — cannot be removed at the contrastive-set level. Pre-registered as a hard limitation that survives any future iterations of this set short of a fundamentally different design (e.g. defender-only refusal contrast, or RDO which is data-confound-agnostic).

**What a result will and won't license, narrowed:**

- A *positive* causal result on this set CANNOT be dismissed as LEXICAL (vocabulary at chance under CV, mechanism for the min_df=1 sensitivity verified by shuffle control). But it could still read attacker-vs-defender STANCE rather than refuse-vs-comply. The correct sentence is: **"isolates refusal-OR-stance, not refusal alone."** Do NOT write "the set is now clean" or "vocabulary doubly confirmed gone" — vocabulary and stance are distinct confounds, and only vocabulary is controlled here. The next experimental design to disentangle stance would be a defender-only refusal contrast (where the model refuses *defensive* requests too on some axis, isolating refusal from role), or RDO which is data-confound-agnostic.

- A *null* result at n=40 (10-prompt held-out test) is **INCONCLUSIVE**. Statistical power is low and (A-ii) — "Qwen's refusal genuinely isn't in a diff-of-means direction" — is the bigger claim and needs the higher bar. Do NOT conclude (A-ii) from this experiment alone. Scale up to 100+ pairs before any (A-ii) claim, AND consider RDO as the independent gradient-extraction path that doesn't rely on any contrastive set.

The methodological finding for §9 regardless of outcome: **building a clean refusal contrastive set is hard *because* harmfulness correlates with both vocabulary and stance.** Removing one confound doesn't isolate refusal — it shifts which confound carries the AUC. This is a substantive piece of methodological framing for any port of these methods, not a defect of this particular set.

## Sweep design (locked before run)

Same harness as `phase2_step3` (ablation) and `phase2_step3d` (operating-band addition sweep), only the contrastive set changes. d̂ extraction layer = L14 to match the original Qwen experiment (any plateau layer would work — see §3d preregistration; the AUC plateau is the entire residual stream on Qwen).

**Split (seed=1):** 30 harmful train + 10 harmful test; 30 harmless train + 10 harmless test. Stratification skipped (matched set has single source on harmful side).

**Ablation cells (4):**
- Baseline (no hook) on 10 held-out matched-harmful prompts.
- Ablate d̂_matched (extracted from 30 train + 30 harmless train at L14) on the 10 held-out harmful.
- Ablate d̂_old (the *code_contrastive* d̂ — for direct comparison: does the de-confounded direction differ from the lexical direction in causal effect?) on the 10 held-out harmful.
- Ablate random unit vector (specificity control).

**Addition sweep (operating-band, primary diagnostic):**
- Inject d̂_matched at L3, L7, L11, L15, L19, L23, L27, L31.
- Coefficients: 1×, 2×, 4× of natural_scale_at_L14_matched (per-layer scale, NOT per-injection scale — Phase 1 §4.4 unit). Skip 0.5× per Phase 1 §4.4 finding that sub-natural-scale is uninformative.
- Target: 10 held-out matched-harmless prompts.
- 8 layers × 3 coeffs × 10 prompts = 240 generations.
- Substring scorer primary; LLM judge on any cell with substring lift > 0.10; specificity control at the best lifted cell.

**Total cost:** 4 ablation cells × 10 = 40 + 240 addition + 10 random control (if triggered) = ~290 generations × ~13s = ~63 min generation + ~5 min dual-judge.

## Pre-registered asymmetric read

### STRONG POSITIVE outcome at n=40

**Definition:** ANY of the following on the matched set:
- Ablation Δ ≥ 0.30 with d̂_matched vs ≤ 0.10 with random direction (specificity), coherent refusal-loss in completions (not degeneration).
- Addition sweep with any (layer × coefficient ≥ 1×) cell inducing refusal lift ≥ 0.30, coherent refusal in completions, random-vector specificity control at that cell does NOT produce comparable lift.

**Reading:** (A-i) provisionally LOCKED. The Qwen null on `code_contrastive` was contrastive-set-driven: removing the lexical confound recovers a diff-of-means direction with a causal hand on refusal. Methodological lesson is sharp — *on lexically-separable contrastive sets, diff-of-means recovers vocabulary, not refusal, and AUC ≈ 1.0 does not tell you which.* Stance entanglement still applies: "isolates refusal-OR-stance, not refusal alone."

**Scale-up plan:** Author 60 more matched pairs (100 total) for a hardened headline result with statistical power for the Phase 2 writeup. RDO (Step D) becomes nice-to-have confirmation rather than load-bearing.

### WEAK POSITIVE outcome at n=40

**Definition:** All of:
- Ablation Δ in [0.10, 0.30] for d̂_matched OR addition lift in [0.10, 0.30] for some cell.
- **Specificity is load-bearing:** random-vector control at the same cell/coefficient sits clearly below the d̂_matched effect (e.g. random Δ ≤ 0.05 while d̂_matched Δ ≥ 0.15 — a clear gap, not noise overlap).
- **Coherence preserved:** the lifted-cell completions read as actual refusals (full sentences, names the user's request, declines with reason) rather than degeneration (short/empty/repeating-tokens).

**Reading: directionally A-i.** The de-confounded d̂ recovered PARTIAL causal signal that the lexical d̂ on `code_contrastive` lacked (Δ ≈ 0.00 there). This is materially different from the flat code_contrastive null and from a 0.50 chance-aligned random reading. It says: removing the lexical confound *did* surface a causal component, just at modest magnitude compared to Gemma's ~0.91 — likely because n=40 is underpowered, because the matched set's stance entanglement weakens specificity, or because Qwen's refusal mechanism is genuinely more distributed (an RDO-discoverable subspace rather than a single direction).

**Scale-up plan:** Author 60 more matched pairs (100 total) to SHARPEN the effect (estimate its magnitude with tight CIs), not to determine existence. A weak positive at n=40 with specificity is enough to support directional A-i; scaling clarifies the size.

**Critical caveat:** if random-vector control matches the d̂_matched lift at the same cell, this is NOT a weak positive — it's a magnitude artifact and reads as NULL. The d̂-specificity check is what separates weak positive from magnitude noise.

### NULL outcome at n=40

**Definition:** All of:
- Ablation Δ < 0.10 for d̂_matched on the matched set.
- No addition cell hits substring lift ≥ 0.10 with d̂-specific causation (random control matches the lift, OR coherence breaks down at higher coefficients).

**Reading: INCONCLUSIVE.** n=40 with 10 held-out test prompts has low statistical power. A null at this scale is *consistent with* both (A-i) — the matched set hasn't removed enough of the confound or N too small — and (A-ii) — refusal really isn't in a diff-of-means direction on Qwen. **Do not claim (A-ii) from this experiment alone.**

### d̂_old vs d̂_matched comparison (within-experiment control, strongest single readout)

The 4-cell ablation table includes both `ablate_d̂_matched` and `ablate_d̂_old` on the SAME 10 held-out matched-harmful prompts. Same prompts, same model, same intervention; only the contrast-of-origin differs. Pre-committed reading:

- **d̂_matched causal (Δ ≥ 0.10 with specificity) AND d̂_old inert (Δ < 0.10) on the same matched-harmful prompts → STRONGEST A-i evidence.** De-confounding is isolated as the operative variable: the lexical direction doesn't move refusal on these prompts, the de-confounded one does. This is a stronger reading than the matched-d̂ effect alone because it's a within-experiment control — random variance, prompt-distribution effects, and Qwen-specific quirks are all held constant; only the contrastive set origin varies.

- **Both d̂_matched and d̂_old causal on the matched-harmful prompts → the effect isn't about de-confounding.** The lexical direction *also* works on these prompts, which contradicts the (A-i) story. Reconsider what d̂_matched is capturing — maybe matched-harmful prompts are just easier to flip on Qwen for reasons unrelated to the contrast (e.g., they're more on-distribution for the model's refusal training).

- **Both d̂_matched and d̂_old inert → consistent with the n=40 null reading.** Don't claim (A-ii); scale up.

- **d̂_old causal but d̂_matched inert → unexpected and informative.** Would suggest the de-confounded extraction somehow LOST the causal signal the lexical extraction had. Flag for investigation; possible if vocabulary signal was inadvertently a proxy for the true causal direction in `code_contrastive`'s activation geometry.

**Scale-up plan:** Author 60 more matched pairs (100 total) and re-run. If d̂_matched is *still* inert at n=100 with proper N-per-cell, then (A-ii) is provisionally earned and RDO (Step D) becomes the decisive next experiment.

### Coherence + specificity (every behavioral sweep)

- Read 3-4 completions from the lowest-coefficient cell that hit substring lift ≥ 0.10. If short/garbled/repeating-tokens → magnitude-degeneration, not refusal. Dismiss.
- Random-vector specificity at the same layer × coefficient as the best d̂-cell. If random also induces "refusals" at same rate → magnitude artifact, not d̂-specific.

## Constraints

- Pre-registration committed before runner; no parameter changes after seeing results.
- No (A-ii) claim from an n=40 null.
- RDO loader work (Step D) gated on the outcome here.
- Numbers + completions saved; framing deferred until the runner finishes and analyzer reports.
