# data/

Two contrastive sets, one per phase:

- `contrastive.jsonl` — Phase 1 (Gemma-2-2b-it): 150 AdvBench harmful + 150 length-matched Alpaca harmless. Frozen.
- `code_contrastive.jsonl` — Phase 2 (Qwen2.5-3B-Instruct): 150 code-themed harmful (HarmBench cybercrime_intrusion + AdvBench code-keyword filter) + 150 length-matched CodeAlpaca harmless. Reproducible from `experiments/build_code_contrastive.py` with seed 20260529; content hash logged at build time. Median char length 76 (harmful) vs 75 (harmless).

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
