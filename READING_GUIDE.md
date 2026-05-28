# Reading Guide — Understanding this project deeply

You built this project over many evening sessions. This is the sequence to
re-enter it as a human reader and actually understand what's there — not
just verify that it ran, but look at the code, the graphs, and the
controls until you can explain the result yourself.

The total reading is ~4-6 hours, split into five "passes." Each pass has a
purpose and a specific output (what you should be able to do after
finishing it). You don't have to do it in one sitting.

---

## Pass 1 — The mental model (45 min)

**Goal:** be able to explain in one paragraph what the project does and
why, without looking at anything.

| # | Read | Time | Why now |
|---|---|---:|---|
| 1.1 | `README.md` | 5 min | Project intro, top-level layout |
| 1.2 | `mech-interp-security-spec.md` §1–§4 (the WHY + WHAT) | 10 min | The Whatnot-eval-skill → safety-interp framing; the deliverable shape; what's in and out of scope |
| 1.3 | `phase-01-build-runbook.md` §0 (the delegation boundary) and §6 (controls) | 10 min | This is the spine. "Delegate the plumbing. Never delegate the science or the controls." |
| 1.4 | `results/phase1_writeup.md` §1 (the question) and §2 (the mechanism, plain English) | 10 min | The whole project in two pages |
| 1.5 | `artifacts/figures/phase0_projection.png` | 5 min | The "oh this works" moment — 15 harmful + 15 harmless prompts, visually separated along a single direction. Stop here before reading anything else and make sure you understand WHAT the axis is and WHY the clusters separate. |

**After Pass 1, you can answer:** What does diff-of-means compute? What
does "the refusal direction" mean? Why is this Tier 2 interp (not
behavioral, not full mechanistic)?

---

## Pass 2 — The graphs (the result you can SEE) (45 min)

**Goal:** flip through every figure with the writeup at hand, build
visual intuition for what was found.

Open `results/phase1_writeup.md` and have it side-by-side. For each
figure below, find the corresponding section in the writeup and read
back-and-forth.

| # | Figure | Section | What to look for |
|---|---|---|---|
| 2.1 | `artifacts/figures/phase1_step2_layer_sweep_advbench.png` | §3.2 layer sweep | Three curves. Real (blue) climbs to ~1.0 from L7+. Random + shuffled hug 0.4. The peak is L23; we chose L13 (in the plateau) for steering. Why L13, not L23? Because L13 is conventional in the lit and in the plateau either way. |
| 2.2 | `artifacts/figures/phase1_step3b_addition_sweep.png` | §3.4 addition sweep | Two heatmaps: refusal rate (left), coherence (right). Look at the diagonal/banded structure. L3-L16 column 1.0× has cells lighting up green; coherence collapses past 3.0×. The "★" marks are the C4-passing cells. |
| 2.3 | `artifacts/figures/phase1_step3b_diagnostic.png` | §3.4 addition sweep | THE diagnostic. At L13, refusal rate (red) rises BEFORE coherence (blue) collapses → addition works at the right coefficient. This is what killed the "asymmetric mechanism" claim. |
| 2.4 | `artifacts/figures/phase1_depth_profile.png` | §3.8 mechanics | AUC, scale, and \|\|x\|\| at each layer using the L13 d_hat. Notice: AUC reaches 0.95 by L3, but scale only reaches 105 by L13. Readable early, "loud" by mid-network. |
| 2.5 | `artifacts/figures/phase1_dhat_per_layer_cossim.png` | §3.8 mechanics | The 26×26 cos-sim matrix. Not uniformly bright. Per-layer diff-of-means picks DIFFERENT directions. cos(d_hat@L3, d_hat@L13) ≈ 0.08 — nearly orthogonal. |
| 2.6 | `artifacts/figures/phase1_dhat_transfer_auc.png` | §3.8 mechanics | The L13 d_hat (row 13) works at every layer. Other rows (e.g., L3) work only nearby. The L13 direction is privileged. |
| 2.7 | `artifacts/figures/phase1_cross_harm_cossim.png` | §3.8 cross-harm | Per-category d_hat cosine matrix. Off-diagonal ≈ 0.92 (close to random-subset null = 0.97). Categories produce almost the same direction. |
| 2.8 | `artifacts/figures/phase1_cross_harm_auc.png` | §3.8 cross-harm | Cross-category transfer AUC. d_hat from cyber works at AUC ≈ 0.996 on weapons, fraud, etc. |
| 2.9 | `artifacts/figures/phase1_per_category_depth_profile.png` | §3.8 cross-harm | All four categories trace the same depth shape. Same mechanism. |
| 2.10 | `artifacts/figures/phase1_step4_probe_by_layer.png` | §3.6 probing | Probe accuracy by layer. Reaches 1.0 by L1. Shuffled-control hugs 0.5. (Compare to figure 2.1 — probe band is WIDER than diff-of-means band.) |
| 2.11 | `artifacts/figures/phase1_step5_localization.png` | §3.7 single-layer ablation | The behavioral-causal complement to probe band. Only L13 alone gives Δ refusal = 1.0. L7 is partial; L3 and L20 are inert. |

**After Pass 2, you can answer:** What's the difference between
"readable" and "causally used"? Why does the addition sweep diagnostic
matter? What would the per-layer d_hat similarity matrix look like if
refusal were one direction at every layer (vs what it actually looks
like)?

---

## Pass 3 — The code, in dependency order (1.5 hours)

**Goal:** read every module that produces the numbers in the writeup.
You'll learn more from this pass than from the writeup itself.

### 3a. The plumbing

| # | File | Time | What to notice |
|---|---|---:|---|
| 3.1 | `src/model.py` | 15 min | `load_model()` — fp16 + MPS settings. `format_prompt()` — Gemma chat template. `tokenize_prompt()` — `prepend_bos=False` because the template already includes BOS. **The BOS double-add gotcha is in the docstring.** |
| 3.2 | `src/activations.py` | 10 min | `cache_resid()` runs forward with a hook capturing `hook_resid_post` at the LAST token. `cache_resid_all_layers()` does all 26 in one pass. Output is on CPU because Phase 1 caches are small. |
| 3.3 | `src/directions.py` | 20 min | The core. `diff_of_means()` — five lines. `ablate_dir()` — context manager subtracting `(x · d_hat) * d_hat` at every residual hook. `add_dir()` — context manager adding `coeff * d_hat` at one hook. **`_FAITHFUL_HOOK_SUFFIXES` is the multi-layer Arditi recipe.** |
| 3.4 | `src/eval.py` | 10 min | Substring scorer. Look at `REFUSAL_PHRASES` — the patched list. Note the failure modes called out in the comments. |
| 3.5 | `src/eval_llm.py` | 15 min | **Read `JUDGE_SYSTEM` carefully — twice.** It is iterated; the first version had 8/12 calibration agreement, the current version has 11/12. The diff is what fixed the safety-bias-on-classification problem. |
| 3.6 | `src/probes.py` | 10 min | Logistic-regression probe per layer + `shuffled_control_sweep` — the leakage check that must sit at chance. |

### 3b. The experiment scripts (in order built)

Read the **docstring** of each one — they explain the question being asked
and the predicted outcome before any code. Then skim the `main()` for
the structure. Don't re-read the helpers shared across scripts (they
mostly recycle from §3a above).

| # | File | What it answers |
|---|---|---|
| 3.7 | `experiments/phase0_trigger.py` | Can the model load? Does the chat template produce coherent output? Is BOS handled right? |
| 3.8 | `experiments/phase1_step1_build_dataset.py` | AdvBench harmful + Alpaca harmless → frozen `contrastive.jsonl` with hashes |
| 3.9 | `experiments/phase1_step2_layer_sweep.py` | Which layer is the refusal direction most separable at? Uses LOO-CV. |
| 3.10 | `experiments/phase1_step3_steering.py` | Causal test. Ablate d_hat at L13, all 4 controls. |
| 3.11 | `experiments/phase1_step3b_addition_sweep.py` | The big sweep that resolved C4 from "failed" to "tuning artifact." 7 layers × 7 coefficients. |
| 3.12 | `experiments/phase1_step3d_addition_headline.py` | Promote the headline cell to N=50 for tight Wilson CI. |
| 3.13 | `experiments/phase1_step4_probing.py` | Per-layer probes + label-shuffled control. |
| 3.14 | `experiments/phase1_step5_localization.py` | Single-layer ablation sweep. Tells you WHERE the gate is. |
| 3.15 | `experiments/phase1_mechanics_and_generality.py` | CPU-only mechanics + cross-harm analysis on cached activations. |
| 3.16 | `experiments/phase1_subspace_ablation.py` | The methodology contribution. Bootstrap LDA cells. Pre-registered prediction in the docstring. |
| 3.17 | `experiments/phase1_tinymmlu_capability.py` | Specificity check — is ablation refusal-specific or capability-degrading? |

**After Pass 3, you can answer:** Why is the BOS prepend setting important?
What exactly does `ablate_dir(d_hat).__enter__()` install as a hook? How
is the shuffled-labels control implemented in `probes.py`? Why does
`cache_resid_all_layers` exist instead of just calling `cache_resid` 26
times?

---

## Pass 4 — The methodology + retractions (1 hour)

**Goal:** understand the discipline that makes the result defensible. The
artifact's value is mostly here, not in the headline numbers. Discovery
isn't the bar; this is.

### 4a. The judge calibration story

| # | Read | Why |
|---|---|---|
| 4.1 | `experiments/calibrate_llm_judge.py` (look at `CALIBRATION_CASES`) | 12 hand-picked test cases with known expected labels, including the "harmful content → COMPLIED" cases that broke the first judge prompt |
| 4.2 | `results/llm_judge_calibration.json` | The actual labels the judge returned for each case |
| 4.3 | Diff `src/eval_llm.py` git history (or look at the current `JUDGE_SYSTEM` and try to imagine what the v1 looked like) | The fix was emphasizing "audit literal behavior, not adjudicate ethics" with explicit examples of "produced malware code → COMPLIED even though the content is dangerous" |

### 4b. The Step 3 audit (does the headline number survive a different scorer?)

| # | Read | Why |
|---|---|---|
| 4.4 | `experiments/audit_existing_scorers.py` | Re-judges all 5 Step 3 cells with the calibrated Claude judge |
| 4.5 | `results/audit_existing_scorers.json` | Per-prompt new labels. Cell `ablate_real_harmful` is the interesting one: substring said 12/12 COMPLIED, judge says 10/12 (2 pivot-style refusals caught). |
| 4.6 | The corrected headline in `phase1_writeup.md` §3.3 | Refusal drops from 1.00 → **0.17** (not 0.00). 83 percentage points is the honest number. |

### 4c. The bootstrap stability check that retracted the "≥15-dim" claim

| # | Read | Why |
|---|---|---|
| 4.7 | `phase1_writeup.md` §3.8 second half ("Iterative-LDA dimensionality probe... Bootstrap stability check") | The retraction in the writeup |
| 4.8 | `experiments/phase1_subspace_ablation.py` — read the BIG docstring + the `extract_lda_top_k_orthogonal` helper | The pre-registered prediction (logged BEFORE running) and the actual experiment |
| 4.9 | `results/phase1_subspace_ablation.md` | The result table — only diff-of-means is causal; 6 other directions all behave like random |

### 4d. The RDO sketch — the delegation-boundary lesson

| # | Read | Why |
|---|---|---|
| 4.10 | `experiments/phase1_rdo_sketch.py` — read ONLY the warning header at the top | This file is parked, never run. It's a record of an agent-driven scope escalation that was caught and reversed. The lesson: "let's try RDO" was the user's authorization; reimplementing Wollschläger from prose was an agent-driven implementation choice the user did NOT authorize. The line between those is the delegation boundary. |

**After Pass 4, you can answer:** What would have happened if we'd
shipped the writeup without the judge calibration? What's the difference
between sample-stable and sample-dependent findings, and which one is the
"≥15 directions" claim? Why is single-judge headlines weaker than they
look?

---

## Pass 5 — Literature context + Phase 2 (45 min)

**Goal:** understand where this work sits and what's open.

### 5a. The published work this replicates

Read the abstracts and the relevant Gemma-2-2b-it sections of these:

| # | Paper | Why |
|---|---|---|
| 5.1 | Arditi et al. 2024 — "Refusal in LLMs is Mediated by a Single Direction" | The foundational paper. We replicate it on Gemma-2-2b-it. |
| 5.2 | Wollschläger et al. 2025 — "The Geometry of Refusal" ([arXiv:2502.17420](https://arxiv.org/abs/2502.17420)) | Multi-direction cones via gradient-based RDO. **Tests Gemma-2-2b-it specifically. Reports 4-dim cone + 79.9% JailbreakBench ASR.** Our subspace-ablation result is the *statistical-extraction* baseline that they implicitly include as their DIM single-direction floor. |
| 5.3 | Winninger 2025 — "Subspace Rerouting" | Operating-band on Gemma-2-2b. We replicate the band claim. |
| 5.4 | Zhao et al. (Jul 2025) — refusal vs harmfulness | Distinct internal directions. Convergent with our exp #3 finding that AdvBench d_hat doesn't capture fictional-framing refusal. |

### 5b. Where we sit, honestly

| # | Read | Why |
|---|---|---|
| 5.5 | `phase1_writeup.md` §4 ("Where we sit vs the published literature") | Each of our 5 findings tagged with what's already published. **There is no novel finding in Phase 1.** The artifact value is the methodology + the consolidation + the explicit retractions. |
| 5.6 | `phase1_writeup.md` §5 (methodology contribution) | The actual contribution: the statistical-vs-gradient extraction decoupling, calibrated judge protocol, bootstrap stability check |

### 5c. What ships to Phase 2

| # | Read | Why |
|---|---|---|
| 5.7 | `phase1_writeup.md` §6 (Phase 2 protocol) | Six-step protocol with RDO mandatory before any multi-D claim |
| 5.8 | `phase1_writeup.md` §7 (what's not yet done) | Explicit list of deferrals — HarmBench, RDO replication, fictional-framing rigorous test, full Step 5 sweep |
| 5.9 | `~/.claude/projects/-Users-anshulsinghle-safe-ai/memory/project-phase1-subspace-finding.md` | The persistent project memory with the corrected framing |
| 5.10 | `~/.claude/projects/-Users-anshulsinghle-safe-ai/memory/feedback-novelty-and-methodology-audit.md` | The persistent lesson for future projects: lit audit, bootstrap stability, judge cross-check before claiming findings |

---

## Optional follow-on passes

### Pass 6 — Raw completions (1-2 hours)

If you want to see what the model *actually said* before vs after
ablation, read the per-condition completions stored in:

- `artifacts/runs/phase1_step3/20260525-152559/result.json` — Step 3 (the
  causal test). Look at `conditions.baseline_harmful.samples[*]` vs
  `conditions.ablate_real_harmful.samples[*]`. Same 12 prompts. Before:
  "I cannot provide you with a program that hacks..." After:
  "```python\nimport random\nimport time\nclass NetworkHacker..."
- `artifacts/runs/phase1_step3b/20260525-164037/result.json` — the sweep
  cells with sample completions per cell.
- `artifacts/runs/phase1_step3d/20260525-183252/result.json` — N=50
  headline cell. All 50 prompt+completion pairs are persisted.
- `artifacts/runs/phase1_subspace_ablation/<latest>/result.json` — the
  ablation cells with sample completions.

These are the qualitative confirmation of the quantitative claims. If you
want to do the (H) "eyeball the completions" task the runbook calls for,
this is the data.

### Pass 7 — Run things yourself

```bash
cd ~/safe_ai/mech-security

# Unit tests
python -m pytest tests/ -v

# Reproducibility check (15 verifiable claims about model + data)
python -m experiments.repro

# Re-run any experiment end-to-end
zsh -c 'source ~/.zshrc && python -m experiments.calibrate_llm_judge'
python -m experiments.phase1_step2_layer_sweep
python -m experiments.phase1_step3_steering --layer 13 --n-test 12
python -m experiments.phase1_subspace_ablation
python -m experiments.phase1_tinymmlu_capability

# Use the graphify codebase graph (instead of grepping)
graphify query "where is d_hat extracted"
graphify explain "ablate_dir"
graphify path "directions.py" "eval.py"
```

---

## How to know you've read deeply enough

Try to answer these without looking. If any feels uncertain, go back to
the cited section.

1. **Mechanism (Pass 1+2):** In two sentences, what does Arditi-style
   refusal ablation do, and why does it work?

2. **Layer choice (Pass 2.1):** Why L13 and not L23? What would change
   if we'd picked L7?

3. **The C4 reframe (Pass 2.3 + 3.11):** What was the original C4
   failure, what did the sweep show, and what's the portable coefficient
   unit?

4. **Single direction vs subspace (Pass 4.7-4.9):** Why does diff-of-means
   "win" for intervention even though LDA finds many classification-
   equivalent directions? What does Wollschläger's published result on
   Gemma-2-2b say, and why doesn't our null result refute it?

5. **The judge (Pass 4.1-4.3):** Name two failure modes of the substring
   scorer that the LLM judge catches, and one failure mode of the LLM
   judge that the substring scorer catches.

6. **Capability preservation (Pass 2 + 3.17):** What's the relationship
   between the coherence-preservation control (C2) in Step 3 and the
   TinyMMLU result?

7. **What we did NOT prove (Pass 5.2 + 5.7):** Why does the writeup
   leave the multi-D question open? What experiment would settle it on
   this model?

If you can answer all seven, you understand this project deeply enough to
write the Phase 2 plan and defend the Phase 1 artifact in a research-
hiring conversation.
