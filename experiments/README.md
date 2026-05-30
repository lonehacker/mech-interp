# `experiments/` inventory

Each `phase*` / `matched*` / pipeline script is a small `main()` invoked as
`python -m experiments.<name>`. Helpers live in `_runner.py` (paths, model
load, activation cache, dataset split, d̂ extraction, batch generation) and
`_rejudge.py` (post-hoc dual-judge re-scoring).

Status legend
- `active` — current pipeline, results live or reproducible.
- `superseded by X` — kept for the audit trail; do not run for headline results.
- `journey artifact` — exploratory / one-shot; not on the pipeline.

| Runner | Experiment | Output(s) under `results/` | Status |
|---|---|---|---|
| `_runner.py` | Shared run-context (paths, model, cache, splits, d̂). | — | active (library) |
| `_rejudge.py` | Re-run dual judge over an existing `partial.json`. | written into the same run dir | active (utility) |
| `audit_existing_scorers.py` | Audit substring vs LLM judges on Phase 0 generations. | `audit_existing_scorers.json` | active |
| `build_code_contrastive.py` | Build `data/code_contrastive.jsonl` for Phase 2. | — (writes `data/`) | active |
| `calibrate_llm_judge.py` | Calibrate Haiku 4.5 vs Opus 4.7 judges. | `llm_judge_calibration.json` | active |
| `code_contrastive_vocab_audit.py` | TF-IDF vocabulary audit of `code_contrastive.jsonl`. | `phase2_vocab_audit.md` | active |
| `matched_dual_audit.py` | Vocab (5-fold CV) + length audit on the matched contrastive set. | console / journal | active |
| `matched_shuffle_control.py` | Shuffle-pairing control on `min_df=1` to test the paired-CV-artifact mechanism. | console / journal | active |
| `matched_draft_audit.py` | Verb-vs-vocabulary decomposition of the matched draft. | console | superseded by `matched_dual_audit.py` (CV-primary) |
| `phase0_trigger.py` | Phase 0 trigger study (vocabulary-vs-intent). | `phase0.md` (+ tag variants) | active |
| `phase1_step1_build_dataset.py` | Build `data/contrastive.jsonl` + audit. | `contrastive_audit.md` | active (Phase 1 frozen) |
| `phase1_step2_layer_sweep.py` | LOO-CV AUC by layer; identifies L13 as peak. | `phase1_step2.md` (+ tag variants) | active (Phase 1 frozen) |
| `phase1_step3_steering.py` | L13 d̂ ablation — the causal headline (1.00 → 0.00 / 0.17). | `phase1_step3.md` | active (Phase 1 frozen) |
| `phase1_step3b_addition_sweep.py` | Layer × coefficient addition sweep. | `phase1_step3b_addition_sweep.md` | active (Phase 1 frozen) |
| `phase1_step3d_addition_headline.py` | Per-natural-scale headline addition result. | `phase1_step3d.md` | active (Phase 1 frozen) |
| `phase1_step4_probing.py` | Linear probe specificity check. | `phase1_step4.md` | active (Phase 1 frozen) |
| `phase1_step5_localization.py` | Layer-by-layer ablation localization. | `phase1_step5.md` | active (Phase 1 frozen) |
| `phase1_subspace_ablation.py` | 1-D vs higher-d subspace ablation comparison. | `phase1_subspace_ablation.md` | active (Phase 1 frozen) |
| `phase1_hardened_subspace.py` | Hardened (multi-d) subspace ablation. | `phase1_hardened_subspace.md` | active (Phase 1 frozen) |
| `phase1_cross_extraction.py` | Cross-extraction (LDA vs diff-of-means). | `phase1_cross_extraction.md` | active (Phase 1 frozen) |
| `phase1_harmbench_eval.py` | HarmBench held-out evaluation under ablation. | `phase1_harmbench.md` | active (Phase 1 frozen) |
| `phase1_harmbench_lda_extension.py` | LDA-direction extension of the HarmBench eval. | `phase1_harmbench_lda_extension.md` | active (Phase 1 frozen) |
| `phase1_tinymmlu_capability.py` | TinyMMLU capability check under ablation. | `phase1_tinymmlu_capability.md` | active (Phase 1 frozen) |
| `phase1_mechanics_and_generality.py` | Phase 1 wrap-up writeup runner. | `phase1_mechanics_and_generality.md` | active (Phase 1 frozen) |
| `phase1_fictional_framing_balanced.py` | Balanced fictional-framing extraction (Phase 1.5-B). | `phase1_fictional_framing_balanced.md` | active (Phase 1.5 frozen) |
| `phase1_fictional_framing_causal.py` | Causal test on the rebalanced direction (Phase 1.5-B). | `phase1_fictional_framing_causal.md` | active (Phase 1.5 frozen) |
| `phase1_random_baseline_5seed.py` | Post-hoc 5-seed random-direction baseline. | `phase1_random_baseline_5seed*.md` (run-dir) | active (post-hoc, no model load) |
| `phase1_L0_audit.py` | Post-hoc L0 sanity audit. | console / run-dir | active (post-hoc, no model load) |
| `phase1_affect_test.py` | Phase 0 affect-test set evaluation. | `phase1_affect_test.md` | superseded by `phase1_step3c_expanded_contrastive.py`, then `phase1_fictional_framing_*` (Phase 1.5-B) |
| `phase1_step3c_expanded_contrastive.py` | Expanded contrastive with fictional-framing prompts. | `phase1_step3c_expanded_contrastive.md` | superseded by `phase1_fictional_framing_balanced.py` and `phase1_fictional_framing_causal.py` (Phase 1.5-B) |
| `phase1_rdo_sketch.py` | Homegrown RDO sketch; not Wollschläger RDO. | n/a | superseded — parked, never run; scope-escalation example |
| `phase2_step1_layer_sweep.py` | Phase 2 Qwen layer-sweep AUC. | `phase2_step1_qwen-qwen2.5-3b-instruct.md` | active |
| `phase2_step1b_controls.py` | Phase 2 control baselines (random / shuffled). | run-dir | active |
| `phase2_step3_causal.py` | Phase 2 L14 ablation causal test. | `phase2_step3_causal.md` | active |
| `phase2_step3b_addition.py` | Phase 2 single-cell addition diagnostic. | `phase2_step3b_addition.md` | active |
| `phase2_step3c_layer_ablation_sweep.py` | Phase 2 extraction-layer ablation sweep. | `phase2_step3c_layer_ablation_sweep.md` | active (built; not yet run) |
| `phase2_step3d_operating_band_sweep.py` | Phase 2 operating-band addition sweep (layer × coeff). | `phase2_step3d_operating_band_sweep.md` | active |
| `phase2_step3d_analyze.py` | Post-hoc analyzer over a 3d run's `result.json`. | console | active |
| `phase2_step3e_matched_set_sweep.py` | Phase 2 matched-set causal sweep. | `phase2_step3e_matched_set_sweep.md` | active |
| `vocab_audit_compare.py` | Compare vocabulary audits across contrastive sets. | console | journey artifact |
| `repro.py` | End-to-end reproducibility check + REPRO.md generator. | `REPRO.md` | active |
