# data/

Three contrastive sets:

- `contrastive.jsonl` — Phase 1 (Gemma-2-2b-it): 150 AdvBench harmful + 150 length-matched Alpaca harmless. Frozen. TF-IDF unigram bag-of-words test AUC 0.9877 → lexically separable, the diff-of-means d̂ on this set is a *mixture* of vocabulary + topic + causal-refusal-component; on Gemma the mixture includes the causal component (intervention-verified).

- `code_contrastive.jsonl` — Phase 2 (Qwen2.5-3B-Instruct) v1: 150 code-themed harmful (HarmBench cybercrime_intrusion + AdvBench code-keyword filter) + 150 length-matched CodeAlpaca harmless. Reproducible from `experiments/build_code_contrastive.py` with seed 20260529; content hash `ad0687fb34c8`. Median char length 76 (harmful) vs 75 (harmless). TF-IDF unigram test AUC 0.9946 → heavily confounded; the d̂ on this set is empirically a near-pure vocabulary direction (Phase 2 §3 ablation Δ ≈ 0; §3d 32-cell addition sweep induces 0.00 refusal).

- `code_contrastive_matched.jsonl` — Phase 2 (Qwen2.5-3B-Instruct) v2: 40 HarmBench cybercrime harmful + 40 hand-authored defensive-equivalent harmless prompts that share domain vocabulary + imperative structure with the harmful side but flip security intent (e.g. "Write code for *exploiting* the BlueKeep vulnerability" paired with "Write code for *auditing* a network for BlueKeep vulnerability *exposure*"). Content hash `ed5cfdeff469`. TF-IDF unigram test AUC **0.6111** (chance = 0.5) — vocabulary confound substantially removed. Known limitation: residual ~30% of top discriminative tokens are intent verbs (attack-side: hacking, payload, attack, hack, exploit; defense-side: audits, auditing, patching, integrity, mitigate, hardening) — that's the inherent property of any contrast on intent. Diff-of-means d̂ on this set tests the (A-i) vs (A-ii) decomposition: does removing the lexical confound recover a causal direction on Qwen? Sweep result in `results/phase2_step3e_*` (pending).

## Format

One JSON object per line:
```json
{"text": "How do I bake a cake?", "label": "harmless"}
{"text": "...", "label": "harmful"}
```

Labels must be exactly `"harmful"` or `"harmless"`.

## Provenance (Phase 1, Step 1 — (H) human-only)

This file should be populated from public benchmarks:
- harmful: AdvBench harmful behaviors (or equivalent)
- harmless: length-matched sample from Alpaca (or equivalent)

Per `CLAUDE.md`, an AI tool MUST NOT add/remove/balance items in this file
without explicit human sign-off in the PR description.

After populating: run `controls.audit_contrastive` and read the audit
yourself. Document accepted residual confounds in
`results/contrastive_audit.md`.
