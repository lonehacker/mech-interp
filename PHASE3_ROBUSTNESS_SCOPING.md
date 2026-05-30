# Phase 3 scoping — can refusal be made robust to weight-level ablation?

**Status:** scoping / not started. This is context for the Phase 3 work, to be read
alongside `PROJECT_STATE.md`. Phases 1–2 *removed* refusal (the attack); Phase 3 asks
whether refusal can be *defended* against that same removal — and crucially, tests any
proposed defense with the existing attack apparatus.

---

## The pivot in one line

Phases 1–2 were red-team: extract a refusal direction by diff-of-means, ablate it, watch
refusal collapse (Gemma 99%→8%; Qwen the same once the layer is selected by bypass-gap).
Phase 3 adds the **defender** side: actually fine-tune an open-weight model to make its
refusal robust to that ablation, then run the Phase 1/2 attack against the hardened model
and measure honestly whether it survived, was reduced, or just relocated to a new direction.

The interesting and motivating part is the **two-track loop**: I'm not only attacking
existing checkpoints, I want to *build* a defense and red-team my own defense with the same
rigor I used to break undefended models. Both outcomes are publishable:
- "I trained refusal to survive ablation and it held (attack success X%→Y%)" — a defense result.
- "I trained the defense and the attack still wins / just finds a new direction" — an
  honest negative, equally useful, and the more likely outcome given the literature.

---

## Why this is a real open problem (not an engineering oversight)

My own Phase 1–2 finding constrains the solution space. Refusal is mediated by one (or a few)
linear direction(s) that ablation removes cleanly, *without* degrading capability (I checked:
mean-chars stable, coherent on-target output under ablation — refusal is **separable** from
capability on current models). So robustness requires changing one of those two facts:

- **Family A — distribute the mechanism.** Make refusal *not* low-dimensional, so there's no
  single direction (or small subspace) to project out. If refusal is genuinely high-dimensional
  or redundant, diff-of-means + ablation has nothing clean to remove.
- **Family B — entangle refusal with capability.** Make refusal inseparable from the useful
  behavior, so ablating it *breaks the thing the attacker wants*. My ablation-preserves-capability
  result is evidence that current models are in the *separable* regime; robustness here means
  pushing them out of it.

The deep tension, and why this isn't solved: **shallow refusal is cheap, robust refusal is
expensive and may cost capability.** Instruction-tuning/RLHF add refusal as a late, shallow
layer precisely because that's the cheap way. Distributing it or entangling it cuts against
the training economics. That tension is the whole problem.

Open question I specifically want my tooling to answer: **adversarial-training-against-ablation
— does it converge to robustness, or is it a cat-and-mouse treadmill?** (Ablate the current
direction → fine-tune to restore refusal → the model routes refusal through a *new* direction
→ re-extract → repeat. Does refusal eventually become un-ablatable, or does diff-of-means just
find the new direction every round?) My harness automates exactly this loop, so I can answer it
empirically rather than speculate.

---

## The attack apparatus that already exists (the defender has to beat this)

From Phases 1–2, all in `~/safe_ai/mech-security/` (TransformerLens, MPS):
- **diff-of-means + LDA extraction** of a candidate direction at any (layer, position).
- **ablation hooks** — project a direction out of the residual stream during generation;
  single-layer/position or all-layer.
- **bypass-gap layer selection** — rank layers by refusal-token-logit drop under ablation
  (the criterion that works when AUC-based selection saturates, as on Qwen).
- **random-direction specificity controls** — the load-bearing control distinguishing
  "this direction is causal" from "this layer is fragile to any perturbation."
- **calibrated dual-judge** (Haiku-4.5) refusal scorer: refused/partial/complied, plus a
  **continuous causal metric** (refusal-vs-compliance logit shift) z-scored against a random
  null band.
- frozen, vocabulary+length-controlled matched contrastive set (the stance caveat applies:
  isolates refusal-OR-stance, not refusal alone — carry this into Phase 3 too).

A defense "works" only if, *after* applying it, this pipeline can no longer find a causal
direction (or the bypass-gap selector can't find a working layer, or refusal-drop is
substantially reduced and does **not** simply reappear along a new direction).

---

## Candidate defenses to survey (maturity-flag each; all are unsettled)

Lead a literature survey here — search current work, don't rely on memory; this area moves
fast. For each: what it claims, what's *demonstrated* vs hoped, whether anyone tested it
against the diff-of-means ablation attack specifically, and whether follow-up attacks defeat
it. Map each to Family A (distribute) or B (entangle).

- **Latent Adversarial Training (LAT)** — perturb activations during training, train for
  robustness to a ball of activation perturbations (ablation is one such perturbation). The
  most serious academic attempt at Family A. Term to search.
- **Tamper-Resistant safeguards / TAR** — train so *fine-tuning* can't easily remove safety.
  The work most directly aimed at my exact threat model (recipient has weights, wants to strip
  safety). Early; partial results.
- **Adversarial training against ablation specifically** — the treadmill question above. The
  one I most want to run because my tooling automates the loop.
- **Circuit-breakers / Representation Engineering (RepE)** — interrupt harmful representations
  rather than relying on output refusal.
- Also scan: robust unlearning, distributed/redundant safety reps, anything entangling refusal
  with capability (Family B).

---

## What a Phase 3 experiment should look like (cheap, runnable)

Hard constraints: Apple Silicon (MPS) + a single modest consumer/cloud GPU for short runs —
**not** a cluster, not days of training. 2B–3B models (Qwen-2.5-3B, Gemma-2-2b-it). Must be
**TransformerLens-loadable** for the ablation attack. Favor, in order: download an
already-defended checkpoint > apply a defense via light LoRA-style fine-tuning > heavy
retraining (avoid). The defender-track experiments mean I *will* be fine-tuning, so LoRA on a
2–3B model on a single GPU is the realistic envelope.

Each experiment needs **pre-registered success/failure criteria decided before running** —
same discipline that made Phases 1–2 credible. The distinction that's easy to rationalize
post-hoc and must be pinned down in advance:
- **Defense holds:** attack success drops substantially AND the drop isn't recovered by
  re-running extraction (no new causal direction appears).
- **Attack still wins:** diff-of-means at the bypass-gap layer still collapses refusal.
- **Defense merely relocates:** refusal survives ablation of the *old* direction, but
  re-extraction finds a *new* causal direction that ablates it just as cleanly. (This is the
  treadmill signature, and the most likely outcome — pre-commit to calling it as such.)

Measure, for each defended model: does diff-of-means still find a causal direction? does the
bypass-gap selector still locate a working layer? is refusal-drop reduced / eliminated /
relocated? Report mean-chars + coherence so "defense broke the model" can't masquerade as
"defense worked." Always include the random-direction control at the tested cell.

---

## Likely shape of the two-track plan (to refine in the scoping chat)

1. **Baseline (free):** confirm the undefended attack on the target model (re-run Phase 1/2)
   so the defended comparison has a clean reference.
2. **Defender track — build:** apply the cheapest credible defense (download a defended
   checkpoint if one exists; else LoRA-fine-tune one defense, probably adversarial-against-
   ablation since the loop is mine to automate).
3. **Red-team track — break:** run the full attack pipeline against the defended model with
   pre-registered criteria; classify holds / wins / relocates.
4. **The treadmill experiment (the interesting one):** iterate ablate→restore→re-extract for
   N rounds on the adversarially-trained model; measure whether the extracted direction gets
   harder to find (higher-dimensional, lower cosine round-to-round, smaller refusal-drop) or
   whether diff-of-means tracks it every round. Pre-register what "converging" vs "treadmill"
   looks like numerically.

---

## Standing discipline carried from Phases 1–2

- Pre-register interpretation before seeing numbers; a null needs more evidence than an effect.
- Random-direction control is non-negotiable — without it, "ablation collapsed refusal" is
  consistent with "this layer is fragile."
- Coherence/mean-chars sentinel so disruption can't masquerade as a result (either direction:
  "attack worked" that's really garbage output, or "defense worked" that's really a broken model).
- Keep the stance caveat (refusal-OR-stance) on any matched-set result.
- Strategic framing for the transition: this is **defense evaluation / red-teaming a safety
  method**, reported honestly — a more directly hireable safety skill than "I can remove
  safety training." Frame it as such; don't let it drift into a pure better-attack project.
- Honest scoping: if the answer is "no cheap defense robustly survives this, here's why,"
  that's a fine and publishable answer. Don't manufacture a defense that works.
