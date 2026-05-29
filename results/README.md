# results/ — index

Every experiment writes a markdown summary here. The headline is in
[`phase1_writeup.md`](phase1_writeup.md); the others are step-by-step
breakdowns with raw numbers.

> If you only read one file: [`phase1_writeup.md`](phase1_writeup.md) (or
> the rendered version at <https://lonehacker.github.io/mech-interp/>).

## The writeup

- [**`phase1_writeup.md`**](phase1_writeup.md) — single consolidated
  writeup with the headline result, controls, methodology contribution,
  literature context, and Phase 2 protocol. ~350 lines.

## Step-by-step results (in the order they were produced)

| File | What it reports |
|---|---|
| [`REPRO.md`](REPRO.md) | 15-check reproducibility script — model hash, data hash, key behavioral numbers. 14/15 pass (only the TransformerLens version check fails, cosmetic). |
| [`phase0.md`](phase0.md) | Phase 0 trigger session: model loads, BOS gotcha verified, harmful/harmless visibly separate along diff-of-means. |
| [`phase0_advbench.md`](phase0_advbench.md) | Phase 0 re-run on AdvBench-derived data (after the frozen contrastive set is built). |
| [`contrastive_audit.md`](contrastive_audit.md) | Confound audit of the 150+150 contrastive set: length, sentiment, topic. Includes the (H) residual-confounds paragraph. |
| [`phase1_step2.md`](phase1_step2.md) | Per-layer LOO-CV AUC layer sweep (initial 15+15 set). |
| [`phase1_step2_advbench.md`](phase1_step2_advbench.md) | Layer sweep on the 150+150 AdvBench contrastive set. Peak L23 AUC 0.999; plateau L7–L25. |
| [`phase1_step3.md`](phase1_step3.md) | The causal test. Ablate d_hat at L13 → refusal 1.00 → 0.00 on 12/12 (substring) or 0.17 (calibrated judge). C4 row reframed by Step 3b. |
| [`phase1_step3b_addition_sweep.md`](phase1_step3b_addition_sweep.md) | Coefficient × injection-layer sweep that resolved C4 from "failed" to "tuning artifact." Operating band L3–L16. |
| [`phase1_step3c_expanded_contrastive.md`](phase1_step3c_expanded_contrastive.md) | Fictional-framing re-extraction test. Pre-registered prediction; result suggests AdvBench d_hat doesn't capture fictional-framing refusal (convergent with Zhao et al.). |
| [`phase1_step3d.md`](phase1_step3d.md) | N=50 promotion of the headline addition cell; Wilson 95% CI [0.93, 1.00] at L3 with extraction-layer-scale coefficient. |
| [`phase1_step4.md`](phase1_step4.md) | Per-layer probing — independent line of evidence for readability. Probe band L1–L25 (broader than steering band). |
| [`phase1_step5.md`](phase1_step5.md) | Single-layer ablation localization. **L13 alone is sufficient** (Δ=+1.00); L7 partial; L3/L20 inert. |
| [`phase1_mechanics_and_generality.md`](phase1_mechanics_and_generality.md) | CPU-only depth profile + cross-harm generality. **Note:** this file's "≥15-dim subspace" framing is superseded by the bootstrap-corrected analysis in `phase1_subspace_ablation.md`; see the supersession note at the top. |
| [`phase1_subspace_ablation.md`](phase1_subspace_ablation.md) | **The N=12 pilot.** Bootstrap LDA cells + L3 d_hat ablation test on 12 held-out AdvBench prompts. Motivated the scaled run below. |
| [`phase1_hardened_subspace.md`](phase1_hardened_subspace.md) | **The hardened version of the classification ≠ causation finding** at N=200 with a continuous causal metric (refusal-vs-compliance first-token logit shift) z-scored against a random-vector null band. Causal d̂ at z ≈ +90; LDA-bootstrap directions inside the null band; L3 d̂ as a "partially causal" cell at z ≈ +5.6 the binary metric couldn't resolve. |
| [`phase1_tinymmlu_capability.md`](phase1_tinymmlu_capability.md) | Capability check. Ablated accuracy 0.570 vs baseline 0.540 — within noise, ablation is refusal-specific. |
| [`phase1_affect_test.md`](phase1_affect_test.md) | Affect-decoupled probe (45 prompts: soft_harmful + loaded_harmless + adversarial_jailbreak). Supplementary; not headline. |
| [`phase1_affect_test_rejudge.md`](phase1_affect_test_rejudge.md) | Re-judge of affect test with patched substring scorer. |

## Supporting JSONs

| File | Use |
|---|---|
| [`audit_existing_scorers.json`](audit_existing_scorers.json) | Per-prompt calibrated-judge re-audit of the Step 3 substring labels (12+12+12+12+12 = 60 verdicts). Confirmed Step 3's headline drop survives the scorer change (12/12 → 2/12 instead of 0/12). |
| [`llm_judge_calibration.json`](llm_judge_calibration.json) | The 12 hand-picked calibration cases used to iterate the Claude-judge prompt. Final agreement: 11/12 (91.7%). |

## Where to find the raw model completions

`results/` has the **summaries**. The **per-prompt prompt + completion
pairs** are persisted under `artifacts/runs/<step>/<timestamp>/result.json`
in the project root. For any result here, the matching `artifacts/runs/`
directory has the underlying data you can eyeball yourself.

For example:
- Step 3 baseline vs ablation completions: `artifacts/runs/phase1_step3/20260525-152559/result.json`
- Step 3d N=50 headline: `artifacts/runs/phase1_step3d/20260525-183252/result.json`
- Subspace ablation per-cell completions: `artifacts/runs/phase1_subspace_ablation/<latest>/result.json`
