# Reading guide — understanding this project deeply

A sequence to read the project from scratch and actually understand it. If
you only want the headline, read the
[published writeup](https://lonehacker.github.io/mech-interp/).

Total ~4-5 hours, split into five passes. Each has a goal and a "what you
should be able to answer" check at the end.

---

## Pass 0 — Mechanics, not journey (45 min)

**Goal:** understand the tensor operations that produce every number in
this project. After this you can read any experiment runner without
googling TransformerLens.

| # | Read | Why |
|---|---|---|
| 0.1 | `HOW_IT_WORKS.md` (top to bottom) | End-to-end Phase 1 walkthrough: model load → chat template → tokenize (BOS handling) → cache residual at layer 13 → diff-of-means → ablate_dir hook → generate. Every tensor index is explained. |
| 0.2 | `src/model.py` | The plumbing this codebase makes safe — load_model, format_prompt, tokenize_prompt (BOS-safe). 168 lines. |
| 0.3 | `src/activations.py` | cache_resid / cache_resid_all_layers. Run forward, capture `hook_resid_post` at the last token. 105 lines. |
| 0.4 | `src/directions.py` | diff_of_means, unit, project, ablate_dir, add_dir. The core math + the two hook context managers. 144 lines. |
| 0.5 | `src/causal_metric.py` (first half, through `get_or_discover_token_sets`) | Per-model first-token discovery for the continuous causal metric. Token IDs aren't portable across model families. ~200 lines. |
| 0.6 | `src/causal_metric.py` (second half) | The continuous metric itself: `compute_causal_effect` under an ablation hook. Same code path for d̂, inert directions, random vectors — no path differences. ~200 lines. |

**After Pass 0** you can explain: what does `logits[0, -1, :]` mean and why
that specific index; what `hook_resid_post` is and why we choose it; what
`x ← x - (x · d̂) * d̂` does to the residual stream and how the hook applies
it at every layer.

---

## Pass 1 — Phase 1 in one figure each (45 min)

**Goal:** flip through the figures with `results/phase1_writeup.md` open.
Build visual intuition for the headline result.

| # | Figure | Section | What to look for |
|---|---|---|---|
| 1.1 | `artifacts/figures/phase1_step2_layer_sweep_advbench.png` | §3.2 layer sweep | Real (blue) climbs to ~1.0 from L7+. Random + shuffled hug 0.4. Peak L23; chose L13 (in plateau, conventional). |
| 1.2 | `artifacts/figures/phase1_step3b_addition_sweep.png` | §3.4 addition sweep | Refusal-rate × coherence heatmaps. L3-L16 at 1× natural-scale light up green; coherence collapses past 3×. |
| 1.3 | `artifacts/figures/phase1_step3b_diagnostic.png` | §3.4 | THE diagnostic. Refusal rises BEFORE coherence collapses → addition works at the right coefficient. |
| 1.4 | `artifacts/figures/phase1_depth_profile.png` | §3.8 mechanics | AUC reaches 0.95 by L3, but scale only reaches 105 by L13. Readable early, "loud" by mid. |
| 1.5 | `artifacts/figures/phase1_dhat_per_layer_cossim.png` | §3.8 mechanics | Per-layer diff-of-means picks DIFFERENT directions. cos(L3, L13) ≈ 0.08 — nearly orthogonal. |
| 1.6 | `artifacts/figures/phase1_step4_probe_by_layer.png` | §3.6 probing | Probe accuracy hits 1.0 by L1. Probe band is WIDER than diff-of-means band. |
| 1.7 | `artifacts/figures/phase1_step5_localization.png` | §3.7 single-layer ablation | Only L13 alone gives Δ refusal = 1.0. L7 partial; L3/L20 inert. |
| 1.8 | `artifacts/figures/phase1_hardened_subspace.png` | §3.13 (Phase 1.5-A) | Continuous causal metric at N=200. Causal d̂ at z ≈ +90 vs random null band; LDA bootstraps at z ≈ 0. |

**After Pass 1** you can answer: what's the difference between "readable"
and "causally used"? Why was L13 the right choice (not L23)?

---

## Pass 2 — Phase 1 headline + Phase 1.5 hardening (1 hour)

**Goal:** the published finding, with all the controls and the post-hardening
reframe.

| # | Read | Why |
|---|---|---|
| 2.1 | `results/phase1_writeup.md` §1–§2 | The question + the mechanism in plain English |
| 2.2 | `results/phase1_writeup.md` §3.2, §3.3, §3.4 | Layer sweep + steering (causal test) + addition sweep (C4 reframe) |
| 2.3 | `results/phase1_writeup.md` §3.6, §3.7 | Probing + single-layer localization (independent lines of evidence) |
| 2.4 | `results/phase1_writeup.md` §3.9 + §3.13 | Subspace ablation: the methodology-contribution finding. §3.13 is the N=200 hardened version with continuous causal metric. |
| 2.5 | `results/phase1_writeup.md` §3.10c | Fictional-framing causal half: classification ≠ causation at the *prompt* level. Strongest single piece of evidence. |
| 2.6 | `results/phase1_writeup.md` §3.12 | HarmBench held-out OOD: 0.99 → 0.08 refusal under d̂ ablation, dual-judge. |
| 2.7 | `results/phase1_writeup.md` §5.4 + §5.5 | Methodology contributions: random-direction distributions (not floors), vocabulary-confound audit on the contrastive sets themselves. |

**After Pass 2** you can answer: why does diff-of-means d̂ "win" for
intervention even though LDA finds many classification-equivalent
directions? What does the continuous metric add over binary refusal rate?
Why doesn't our subspace null result refute Wollschläger's published 4-D
RDO finding?

---

## Pass 3 — The experiment runners + persisted results (1 hour)

**Goal:** see the headline numbers in the code that produced them. Read
the docstring of each runner (which states the question + predicted
outcome), then skim the `main()`.

### Phase 1

| # | Runner | What it answers |
|---|---|---|
| 3.1 | `experiments/phase0_trigger.py` | Does the model load? Chat template + BOS work? |
| 3.2 | `experiments/phase1_step2_layer_sweep.py` | Per-layer LOO-CV AUC of diff-of-means d̂ |
| 3.3 | `experiments/phase1_step3_steering.py` | The causal test — ablate d̂ at L13, all 4 Arditi controls |
| 3.4 | `experiments/phase1_step3b_addition_sweep.py` | 7 layers × 7 coefficients, resolved C4 from "failed" to "tuning artifact" |
| 3.5 | `experiments/phase1_step4_probing.py` | Per-layer probes + shuffled-label control |
| 3.6 | `experiments/phase1_step5_localization.py` | Single-layer ablation (L13 alone suffices) |
| 3.7 | `experiments/phase1_subspace_ablation.py` | Bootstrap LDA cells, pre-registered prediction logged in docstring |
| 3.8 | `experiments/phase1_tinymmlu_capability.py` | Capability specificity check (ablation doesn't lobotomize) |
| 3.9 | `experiments/phase1_harmbench_eval.py` | HarmBench N=200 held-out OOD generalization, dual-judge |
| 3.10 | `experiments/phase1_hardened_subspace.py` | Phase 1.5-A — continuous causal metric at N=200, 16 cells incl. 5-vector random null band |
| 3.11 | `experiments/phase1_fictional_framing_balanced.py` + `_causal.py` | Phase 1.5-B — extraction + causal halves of the fictional-framing test |

### Phase 2 (in flight)

| # | Runner | What it tests |
|---|---|---|
| 3.12 | `experiments/build_code_contrastive.py` | Reproducible curator of `data/code_contrastive.jsonl` (HB cyber + AdvBench code + CodeAlpaca, n=300) |
| 3.13 | `experiments/phase2_step1_layer_sweep.py` | Layer sweep on Qwen2.5-3B + code_contrastive (peak L14 AUC 1.0; full plateau across all 36 layers) |
| 3.14 | `experiments/phase2_step1b_controls.py` | 5-seed random + 5-seed shuffled controls; flagged bimodal random distribution that exposed the vocabulary confound |
| 3.15 | `experiments/phase2_step3_causal.py` | Ablate L14 d̂ on 30 held-out code-harmful prompts. Result: Δ ≈ 0.00 judge — null on the lexically-confounded set. |
| 3.16 | `experiments/phase2_step3d_operating_band_sweep.py` | Operating-band addition sweep across 8 layers × 4 coefficients. Result: 0/32 cells induced refusal — leaf (A) on `code_contrastive`. |
| 3.17 | `experiments/code_contrastive_vocab_audit.py` + `vocab_audit_compare.py` | TF-IDF unigram LR on each contrastive set. Phase 1 0.99, Phase 2 v1 0.99, matched v4 0.50 (CV). |
| 3.18 | `experiments/matched_dual_audit.py` + `matched_shuffle_control.py` | CV-primary + length audit + verified paired-set min_df=1 CV artifact on `data/code_contrastive_matched.jsonl` |
| 3.19 | `experiments/phase2_step3e_matched_set_sweep.py` | Matched-set causal sweep (the experiment that distinguishes A-i from A-ii). Pre-registered at `results/phase2_step3e_preregistration.md`. |

**After Pass 3** you can answer: how the per-prompt completions get from
the model to the substring/judge scorers; what each cache key in
`extra=` strings encodes (model name, dtype, layer, prompt-set identity);
why `_runner.py`'s `cached_activations` makes re-runs idempotent.

---

## Pass 4 — Methodology + retractions (45 min)

**Goal:** the artifact's value is here, not in the headline numbers.

| # | Read | Why |
|---|---|---|
| 4.1 | `experiments/calibrate_llm_judge.py` (read `CALIBRATION_CASES`) + `results/llm_judge_calibration.json` | 12 hand-picked cases including "harmful content → COMPLIED" cases that broke the first judge. The fix in `JUDGE_SYSTEM`: "audit literal behavior, not adjudicate ethics." |
| 4.2 | `experiments/audit_existing_scorers.py` + `results/audit_existing_scorers.json` | Re-judge of Step 3's 5 cells with the calibrated judge. Ablation cell: substring 12/12 COMPLIED → judge 10/12 (2 pivot-style refusals caught). |
| 4.3 | `results/phase1_writeup.md` §3.13 hardened result | Continuous causal metric flips the §3.9 binary-pilot framing. Causal d̂ at z ≈ +90, LDA bootstraps inside the null band, L3 d̂ as a partially-causal cell the binary metric blurred together with classification-by-chance. |
| 4.4 | `results/phase1_writeup.md` §3.10c | Fictional-framing causal half — 14/15 → 2/15 under ablation, including 2/3 prompts d̂ classified below midpoint. Classification ≠ causation at the prompt level. |
| 4.5 | `results/phase1_writeup.md` §5.3 + §5.4 + §5.5 | Three methodological-hygiene contributions: bootstrap stability check (sample-dependent direction count), random-direction baselines as distributions (not single-seed floors), vocabulary-confound audit on the contrastive sets themselves. |
| 4.6 | `results/phase2_step3e_preregistration.md` | The Phase 2 matched-set sweep pre-registration: asymmetric tree (strong / weak / null leaves), d̂_old vs d̂_matched within-experiment control, stance-entanglement caveat kept distinct from vocabulary. |

**After Pass 4** you can answer: what would have happened if we'd shipped
the writeup without judge calibration? What does "report distributions,
not point estimates" mean concretely, with two examples from this project?
Why is the stance/intent confound on `code_contrastive_matched` orthogonal
to the vocabulary confound and not retired by TF-IDF at chance?

---

## Pass 5 — Where this sits + what's open (30 min)

**Goal:** literature context + Phase 2 protocol.

| # | Read | Why |
|---|---|---|
| 5.1 | Arditi et al. 2024 abstract | Foundational single-direction finding |
| 5.2 | [Wollschläger et al. ICML 2025](https://arxiv.org/abs/2502.17420) abstract + the Gemma-2-2b-it section | Multi-D cone via gradient RDO. Tests this exact model. Reports 4-dim cone, 79.9% JailbreakBench ASR. Our subspace-ablation null is the *statistical-extraction* baseline they implicitly use. |
| 5.3 | Winninger 2025 — "Subspace Rerouting" abstract | Operating-band on Gemma-2-2b — replicated by our §3.4 addition-sweep result |
| 5.4 | Zhao et al. (Jul 2025) abstract — refusal vs harmfulness as distinct directions | Convergent with our §3.10c fictional-framing finding |
| 5.5 | `results/phase1_writeup.md` §4 | Each Phase 1 finding tagged with what's already published. **There is no novel finding in Phase 1.** The contribution is methodology + consolidation. |
| 5.6 | `results/phase1_writeup.md` §6 | Phase 2 protocol — target Qwen2.5-3B-Instruct + matched contrastive set + RDO as mandatory step before any multi-D claim |
| 5.7 | `data/README.md` matched-set section | The current Phase 2 contrastive set audit: CV-primary AUC at chance, length-matched, stance uncontrolled (pre-registered). |
| 5.8 | `~/.claude/projects/-Users-anshulsinghle-safe-ai/memory/project-phase2-kickoff-state.md` | Persistent project state — what's running, what's pending, what RDO needs to launch |

**After Pass 5** you can answer: why doesn't our subspace null result
refute Wollschläger? What would settle the multi-D question on this model?
What's the next experiment after the in-flight matched-set sweep, and how
does the result decide whether RDO is "nice to have" or "load-bearing"?

---

## Optional follow-ons

### Raw model completions

To see what the model actually said before vs after ablation:
- `artifacts/runs/phase1_step3/<latest>/result.json` — Step 3 N=12 baseline + ablation
- `artifacts/runs/phase1_step3d/<latest>/result.json` — Step 3d N=50 headline
- `artifacts/runs/phase1_harmbench/<latest>/result.json` — N=200 HarmBench + dual-judge
- `artifacts/runs/phase1_fictional_causal/<latest>/result.json` — 14/15 → 2/15 fictional-framing collapse
- `artifacts/runs/phase2_step3/<latest>/result.json` — Qwen L14 ablation null
- `artifacts/runs/phase2_step3e/<latest>/result.json` — matched-set sweep (when it lands)

### Run things yourself

```bash
cd ~/safe_ai/mech-security
python -m pytest tests/ -v                    # 53 unit tests
python -m experiments.repro                    # reproducibility checks
python -m experiments.phase1_step2_layer_sweep
python -m experiments.phase1_step3_steering
python -m experiments.matched_dual_audit       # vocab + length audit on matched set
python -m experiments.matched_shuffle_control  # paired-set min_df=1 CV artifact verification
```

---

## How to know you've read deeply enough

Answer without looking. If any feels uncertain, go back to the cited pass.

1. **Mechanics:** In two sentences, what does `ablate_dir(model, d_hat)` install as a hook, and why all five residual hook points per layer?
2. **Layer choice:** Why L13 and not L23 on Gemma? Why does the AUC peak NOT pick the causal layer on Qwen?
3. **The C4 reframe:** What was the original C4 "failure"; what did the addition sweep show; what's the portable coefficient unit?
4. **Single direction vs subspace:** Why does diff-of-means "win" for intervention even though LDA finds many classification-equivalent directions? Why doesn't the null result refute Wollschläger's 4-D RDO cone?
5. **Continuous metric:** Why refusal-MINUS-compliance and not just Δlog p(refusal)? What does the contrast cancel?
6. **The fictional-framing prompt-level result:** Why is §3.10c immune to the vocabulary confound that §3.13 carries?
7. **Vocabulary audit:** Phase 1 / Phase 2 v1 / matched v4 TF-IDF unigram CV AUCs are ~0.99 / ~0.99 / ~0.50. What does each number mean about what d̂ can be?
8. **Stance entanglement:** Why does "vocabulary at chance" NOT retire the attacker-vs-defender stance confound on the matched set?
9. **Pre-registration:** What's the strong / weak / null leaf structure of the Phase 2 step 3e pre-reg, and what does d̂_old-vs-d̂_matched comparison license that neither cell alone does?
10. **What we did NOT prove:** Why does the writeup leave the multi-D question open? What experiment settles it on Gemma? On Qwen?

If you can answer all ten, you understand this project deeply enough to
write the Phase 2 sweep interpretation and defend the Phase 1 artifact in
a research-meeting setting.
