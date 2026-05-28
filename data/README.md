# data/

`contrastive.jsonl` — the frozen contrastive set used by every Phase 1 step.

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
