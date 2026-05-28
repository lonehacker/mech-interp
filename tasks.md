# tasks

`(H)` = human-only, do not delegate. Per CLAUDE.md / runbook §0.

## Phase 0 — trigger session ✅ DONE

- [x] (H) HF license accepted, token in keychain.
- [x] Model loads on MPS (transformer_lens, fp16, ~30s load).
- [x] Chat-template gotcha verified: templated vs raw differ.
- [x] BOS = 1 assertion satisfied; assertion verified live on a force-fed bad input.
- [x] Gemma-2 softcap sanity: `2 + 2 = 4` clean.
- [x] 15+15 hand-written set frozen at `data/phase0-pairs.jsonl`.
- [x] Mid-layer residuals cached, diff-of-means projection plotted.
- [x] `artifacts/figures/phase0_projection.png` exists.
- [x] Phase 0 reproduced on AdvBench/Alpaca data (`results/phase0_advbench.md`).

## Phase 1

### Step 1 — contrastive set (highest-leverage hour, (H)) ✅ DONE
- [x] AdvBench harmful + length+stem-matched Alpaca harmless → `data/contrastive.jsonl` (150+150).
- [x] `controls.audit_contrastive` run; written to `results/contrastive_audit.md`.
- [x] (H) Residual-confounds paragraph drafted in `results/contrastive_audit.md`. **(H) → you should still re-read and edit.**

### Step 2 — layer sweep ✅ DONE (LOO-CV)
- [x] Per-layer LOO-CV diff-of-means AUC.
- [x] Plotted to `artifacts/figures/phase1_step2_layer_sweep_advbench.png`.
- [x] Random-direction + shuffled-labels controls included.
- Working hypothesis: **L = 13** (chosen for steering; plateau extends L7–L25).

### Step 3 — steering (causal claim; controls mandatory) ✅ DONE
- [x] Ablate `d_hat` on harmful set, measure refusal-rate drop.
- [x] Add `d_hat` on harmless set, measure over-refusal.
- [x] C1: random-direction baseline included.
- [x] C2: coherence preservation measured (p50 chars on harmless after ablation).
- [x] C3: train/test split — d_hat from train only.
- [x] C4: both directions tested (ablate ↓ AND add ↑).
- [x] Raw numbers in `results/phase1_step3.md`.
- (H) Read the four controls' pass/fail and the per-condition completions yourself before claiming causal mediation.

### Step 4 — probing ⏳ RUNNER BUILT, RUN PENDING
- [x] Per-layer logistic-regression probe with proper train/test split (runner: `experiments/phase1_step4_probing.py`).
- [x] Label-shuffled control probe (must sit at chance) — wired via `shuffled_control_sweep`.
- [x] Plot to `artifacts/figures/phase1_step4_probe_by_layer.png` overlaid with steering band L7–L25 and Step 3 injection L13.
- [x] Unit tests in `tests/test_probes.py` (10/10 pass).
- [ ] Execute the runner (MPS-bound; queued after Step 3b sweep).
- [ ] (H) Reconcile any tension between probe and steering layer bands.

### Step 5 — localization ⏳ RUNNER BUILT, RUN PENDING (stretch)
- [x] Layer-restricted single-layer ablation sweep (runner: `experiments/phase1_step5_localization.py`); per-cell random-direction control + coherence check.
- [ ] (Optional) Activation patching from harmless → harmful run.
- [ ] Execute the runner (MPS-bound; queued after Step 4).
- [ ] (H) Report convergence or divergence honestly.

### Step 3b — addition × layer sweep ✅ DONE 2026-05-25 18:31
- C4 reframed: **tuning artifact, not asymmetric mechanism.** Addition passes C4 at L3, L7, L10, L13, L16 with coeff = 1.0–3.0× of natural scale (≈ 105). Original coeff=25 was 0.24× natural scale, structurally below the operating window.
- Operating band identified: **L3–L16 induces refusal; L20/L23 do not at any coefficient** (refusal flat 0 across 1000× range). Refusal-decision circuit lives in L3–L16; layers past L16 are downstream and can't be back-injected.
- Results in `results/phase1_step3b_addition_sweep.md`; heatmap + diagnostic figures in `artifacts/figures/phase1_step3b_*.png`.

### Affect-decoupled transfer test ✅ DONE (not in original runbook)
- [x] `data/affect-test.jsonl`: 15 soft_harmful + 15 loaded_harmless + 15 adversarial_jailbreak.
- [x] `experiments/phase1_affect_test.py` cross-tabulates by category.
- [x] Substring scorer patched to catch empathetic refusals.
- [ ] **(H)** Re-run with `--scorer llm` after setting `ANTHROPIC_API_KEY` for cleaner numbers.

### Reproducibility infrastructure ✅ DONE (not in original runbook)
- [x] `experiments/repro.py` — 15-check reproducibility script with literature cross-reference.
- [x] `tests/test_eval.py` — 20 unit tests for the refusal scorer.
- [x] `tests/test_directions.py` — 13 unit tests for diff-of-means / projection / random-direction math.

## Phase 1 done

- [ ] Confound-audited frozen contrastive set in `data/`.
- [ ] Diff-of-means direction localized by layer sweep.
- [ ] Steering result passes **all four** controls.
- [ ] Probe corroborates implicated layers.
- [ ] Every result stated as raw numbers with control numbers beside it.
