# Phase 2 Step 3e — matched-set causal sweep pre-registration (asymmetric read)

**Logged 2026-05-30 BEFORE the run. The read is asymmetric — a null at n=40 needs more evidence than an effect — by planner protocol. Do not relitigate the asymmetry after seeing numbers.**

## What this experiment decides

Phase 2 Step 3 (ablation) and Step 3d (operating-band addition sweep) on Qwen2.5-3B + `code_contrastive.jsonl` both came back null: ablating d̂ leaves refusal at 0.97 → 0.97 (judge); 32-cell addition sweep induces 0.00 refusal with no degeneration. The vocabulary audit (TF-IDF unigram test AUC 0.9946 on `code_contrastive`) means d̂ extracted from that set is indistinguishable from a near-pure vocabulary direction; the null is consistent with either:

- **(A-i)** *Data-driven null:* d̂ from a vocabulary-confounded contrast is a lexical direction with no refusal component; ablating it shouldn't move refusal because vocabulary isn't the refusal mechanism. Fix is a de-confounded contrast.
- **(A-ii)** *Model-driven null:* Qwen2.5-3B's refusal genuinely doesn't live in any diff-of-means-recoverable direction (regardless of contrast); needs RDO / multi-direction extraction.

The matched contrastive set (`data/code_contrastive_matched.jsonl`, hash `1e6df4923256`, 40 pairs, **TF-IDF AUC 0.5556** — vocabulary confound essentially removed at the bag-of-words level after v3 cleanup) is the experimental fix that distinguishes them. If d̂ on the matched set has a causal hand on Qwen, the null was data-driven (A-i, narrowly). If not, the null is potentially model-driven (A-ii) — but only at adequate N (see asymmetric read below).

## Pre-registered interpretation bounds (logged BEFORE the run)

**Two limitations of this contrastive set that bound what a result can license:**

1. **Residual lexical confound.** The matched set reduces but does not eliminate lexical separability. Residual TF-IDF AUC 0.56 (chance 0.50) concentrates in security-intent verbs spread across audit / scan / monitor / harden / mitigate / profile / quarantine.

2. **Stance/intent entanglement (the deeper caveat).** Harmful prompts ask the model to act as ATTACKER ("write code to exploit X"); defensive prompts ask it to act as DEFENDER ("write code to mitigate X"). A diff-of-means direction extracted from this contrast may capture an **attacker-vs-defender stance direction** rather than a refuse-vs-comply direction. Stance correlates with refusal because the model refuses attacker-role requests. This set confounds security-intent AND agentive stance; it does not isolate refusal from either.

**What a result will and won't license, narrowed:**

- A *positive* causal result on this set supports the **narrow claim**: "diff-of-means recovers a causal direction on a less-lexically-confounded contrast (and on a contrast that flips agentive stance from attacker to defender)" — NOT "refusal isolated from vocabulary." State the stance/intent entanglement explicitly alongside any positive result.

- A *null* result at n=40 (10-prompt held-out test) is **INCONCLUSIVE**. Statistical power is low and (A-ii) — "Qwen's refusal genuinely isn't in a diff-of-means direction" — is the bigger claim and needs the higher bar. Do NOT conclude (A-ii) from this experiment alone. Scale up to 100+ pairs before any (A-ii) claim.

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

### POSITIVE outcome at n=40

**Definition:** ANY of the following on the matched set:
- Ablation Δ ≥ 0.30 with d̂_matched vs ≤ 0.10 with random direction (specificity), coherent refusal-loss in completions (not degeneration).
- Addition sweep with any (layer × coefficient ≥ 1×) cell inducing refusal lift ≥ 0.30, coherent refusal in completions, random-vector specificity control at that cell does NOT produce comparable lift.

**Reading:** (A-i) provisionally LOCKED. The Qwen null on `code_contrastive` was contrastive-set-driven: removing the lexical confound recovers a diff-of-means direction with a causal hand on refusal. Methodological lesson is sharp — *on lexically-separable contrastive sets, diff-of-means recovers vocabulary, not refusal, and AUC ≈ 1.0 does not tell you which.*

**Scale-up plan:** Author 60 more matched pairs (100 total) for a hardened headline result with statistical power for the Phase 2 writeup. RDO (Step D) becomes nice-to-have confirmation rather than load-bearing.

### NULL outcome at n=40

**Definition:** Ablation Δ < 0.10 on the matched set AND no addition cell hits substring lift ≥ 0.30 with d̂-specific causation (specificity control matches the lift, OR coherence breaks down).

**Reading: INCONCLUSIVE.** n=40 with 10 held-out test prompts has low statistical power. A null at this scale is *consistent with* both (A-i) — the matched set hasn't removed enough of the confound or N too small — and (A-ii) — refusal really isn't in a diff-of-means direction on Qwen. **Do not claim (A-ii) from this experiment alone.**

**Scale-up plan:** Author 60 more matched pairs (100 total) and re-run. If d̂_matched is *still* inert at n=100 with proper N-per-cell, then (A-ii) is provisionally earned and RDO (Step D) becomes the decisive next experiment.

### Coherence + specificity (every behavioral sweep)

- Read 3-4 completions from the lowest-coefficient cell that hit substring lift ≥ 0.10. If short/garbled/repeating-tokens → magnitude-degeneration, not refusal. Dismiss.
- Random-vector specificity at the same layer × coefficient as the best d̂-cell. If random also induces "refusals" at same rate → magnitude artifact, not d̂-specific.

## Constraints

- Pre-registration committed before runner; no parameter changes after seeing results.
- No (A-ii) claim from an n=40 null.
- RDO loader work (Step D) gated on the outcome here.
- Numbers + completions saved; framing deferred until the runner finishes and analyzer reports.
