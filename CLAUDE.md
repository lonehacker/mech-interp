# Working agreement for AI coding tools

This is an interpretability research repo. Correctness of *experimental logic*
matters more than speed.

ALWAYS:
- Keep functions pure and testable; activations in, tensors/metrics out.
- Make randomness explicit: every function that samples takes a `seed`.
- When formatting prompts, use the Gemma chat template; never feed raw strings
  to an instruct model.

NEVER without explicit human sign-off in the PR description:
- Add, remove, or "balance" items in `data/contrastive.jsonl`.
- Change which control baselines run, or relax a control's threshold.
- Summarize a result as positive/negative — report raw numbers only.

When asked for an experiment, also output the strongest confound that could
explain the result, and the control that rules it out.

## Context Navigation (Graphify + vault)

This repo has a knowledge graph at `graphify-out/graph.json`. Before reading
source files to answer a structural question ("where is X defined?", "what
calls Y?", "what does module Z connect to?"), query the graph first:

```
graphify query "your question" --budget 2000
graphify explain "ablate_dir"
graphify path "directions.py" "eval.py"
graphify affected "format_prompt"
```

Persistent project memory lives in `~/vault/mech-security/`:
- `MOC.md` — map of content, current phase status
- `architecture/decisions.md` — design decisions, append-only
- `logs/YYYY-MM-DD-*.md` — session logs created by `/save`

Session commands per `~/vault/CLAUDE.md`: `/resume` reads recent logs and
decisions; `/save` writes a new session log.
