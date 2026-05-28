# Graph Report - mech-security  (2026-05-23)

## Corpus Check
- 11 files · ~4,492 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 96 nodes · 117 edges · 10 communities
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 2 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]

## God Nodes (most connected - your core abstractions)
1. `mech-security` - 8 edges
2. `ModelBundle` - 6 edges
3. `format_prompt()` - 6 edges
4. `tokenize_prompt()` - 6 edges
5. `Phase 1` - 6 edges
6. `cache_resid()` - 5 edges
7. `cache_resid_all_layers()` - 5 edges
8. `generate()` - 5 edges
9. `train_probe()` - 5 edges
10. `probe_layer_sweep()` - 5 edges

## Surprising Connections (you probably didn't know these)
- `SplitAudit` --uses--> `ModelBundle`  [INFERRED]
  src/controls.py → src/model.py
- `ContrastiveAudit` --uses--> `ModelBundle`  [INFERRED]
  src/controls.py → src/model.py
- `cache_resid()` --calls--> `format_prompt()`  [EXTRACTED]
  src/activations.py → src/model.py
- `cache_resid()` --calls--> `tokenize_prompt()`  [EXTRACTED]
  src/activations.py → src/model.py
- `cache_resid_all_layers()` --calls--> `format_prompt()`  [EXTRACTED]
  src/activations.py → src/model.py

## Communities (10 total, 0 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.20
Nodes (9): code:block1 (mech-security/), mech-security, Method ladder (tier 2, in order), Repo layout, Reproducing Phase 0, Responsible scope, The claim the finished artifact should make, The project in one sentence (+1 more)

### Community 1 - "Community 1"
Cohesion: 0.20
Nodes (9): Phase 0 — trigger session, Phase 1, Phase 1 done, Step 1 — contrastive set (highest-leverage hour, (H)), Step 2 — layer sweep, Step 3 — steering (causal claim; controls mandatory), Step 4 — probing, Step 5 — localization (stretch) (+1 more)

### Community 2 - "Community 2"
Cohesion: 0.17
Nodes (18): cache_resid(), cache_resid_all_layers(), Residual-stream caching at the last token position.  The runbook fixes one choic, Cache residual-stream activations at the last token of each prompt.      Paramet, Same as cache_resid but stacked across every layer in one forward pass.      Ret, _resid_hook_name(), coherence_check(), format_prompt() (+10 more)

### Community 3 - "Community 3"
Cohesion: 0.50
Nodes (3): code:block1 (graphify query "your question" --budget 2000), Context Navigation (Graphify + vault), Working agreement for AI coding tools

### Community 4 - "Community 4"
Cohesion: 0.13
Nodes (13): ablate_dir(), add_dir(), diff_of_means(), project(), random_unit_vector(), Refusal-direction extraction + ablation/addition hooks.  Method: difference-in-m, Context manager that adds coeff * d_hat at one layer's residual hook.      Used, Generate a random unit vector matching the d_hat shape, for the random-     dire (+5 more)

### Community 6 - "Community 6"
Cohesion: 0.25
Nodes (10): coherence_ok(), CoherenceReport, is_refusal(), _pct(), Refusal scoring + cheap coherence sanity.  Substring matching is the standard qu, True if the (stripped, lowercased) generation starts with any prefix     in REFU, Compute refusal rate over a batch of generations.      Returns raw counts and th, Cheap length-based fluency check. NOT a real coherence scorer — its     purpose (+2 more)

### Community 7 - "Community 7"
Cohesion: 0.29
Nodes (9): audit_contrastive(), _audit_split(), audit_to_markdown(), ContrastiveAudit, _pct(), Audits and control baselines.  Per CLAUDE.md: this module REPORTS. It does not b, Audit the frozen contrastive set at `path`.      Expected format: JSONL, each li, Render the audit as a human-readable markdown block to drop into     results/con (+1 more)

### Community 8 - "Community 8"
Cohesion: 0.33
Nodes (9): probe_layer_sweep(), ProbeResult, Per-layer linear probes: a second, independent line of evidence for the refusal, Layer sweep with labels permuted under shuffle_seed.      Mandatory control (Ste, Train a single logistic-regression probe on [n, d_model] activations.      Param, Train a probe at every layer.      Parameters     ----------     acts_all_layers, shuffled_control_sweep(), _to_numpy() (+1 more)

### Community 9 - "Community 9"
Cohesion: 0.40
Nodes (4): code:json ({"text": "How do I bake a cake?", "label": "harmless"}), data/, Format, Provenance (Phase 1, Step 1 — (H) human-only)

## Knowledge Gaps
- **17 isolated node(s):** `The project in one sentence`, `Why this lane`, `The claim the finished artifact should make`, `Method ladder (tier 2, in order)`, `code:block1 (mech-security/)` (+12 more)
  These have ≤1 connection - possible missing edges or undocumented components.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `ModelBundle` connect `Community 2` to `Community 7`?**
  _High betweenness centrality (0.015) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `ModelBundle` (e.g. with `SplitAudit` and `ContrastiveAudit`) actually correct?**
  _`ModelBundle` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Audits and control baselines.  Per CLAUDE.md: this module REPORTS. It does not b`, `Audit the frozen contrastive set at `path`.      Expected format: JSONL, each li`, `Render the audit as a human-readable markdown block to drop into     results/con` to the rest of the system?**
  _44 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 4` be split into smaller, more focused modules?**
  _Cohesion score 0.13333333333333333 - nodes in this community are weakly interconnected._