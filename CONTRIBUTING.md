# Contributing — code conventions

Short doc, opinionated, ML-research-sized. Conventions benchmarked against
peer mech-interp repos (SAELens, ARENA, EleutherAI/sae, Wollschläger's
geometry-of-refusal, Arditi's refusal_direction, Karpathy's nanoGPT).

## Layout

```
src/                  → core library — produces tensors / contexts; never interprets
  model.py            → load model, format prompts, tokenize, generate
  activations.py      → cache residual stream at the last token
  directions.py       → diff-of-means + ablate / add hooks
  causal_metric.py    → first-token logit-shift continuous metric
  eval.py             → substring refusal scorer
  eval_llm.py         → calibrated Claude judge
  probes.py           → per-layer logistic-regression probing
experiments/          → one runner per experiment; orchestration only
  _runner.py          → shared run-context (paths, run-dir, JSON, cache)
results/              → markdown summary per experiment, the writeup
data/                 → frozen contrastive sets + benchmark CSVs
artifacts/            → run outputs (per-prompt JSONs), figures, cache
tests/                → pure-Python unit tests (no model load)
```

Don't put interpretation in `src/`. Don't put core math in `experiments/`.
Don't put long-lived state outside `data/` or `artifacts/`.

## Style

### Names

- **`harmful` / `harmless`** for the two contrastive classes — never `X` / `y`. The whole refusal-direction literature uses these (Arditi, Wollschläger, Zhao); stick with the field convention.
- **`d_hat`** for the unit-norm refusal direction, **`d`** or **`d_raw`** for the unnormalized diff. Always Greek `d̂` in markdown/docstrings, ASCII `d_hat` in code.
- **`H`, `L`** for harmful / harmless activation matrices `[n, d_model]`. `h`, `l` for single rows. Both are ignored by `E741` (ambiguous-name) in `ruff.toml`.
- **`L`** also means *layer index* by context. Don't fight it; readers parse from surrounding shape.
- **`acts`** for activation tensors, never `embeddings` (that's the embedding-matrix output specifically).

### Shapes

Tensor shapes are the load-bearing convention in this kind of code. Two options, both fine:

- **Inline shape comments**, Karpathy style:
  ```python
  H = cache_resid(bundle, harmful, layer=13)    # [n_h, d_model]
  L = cache_resid(bundle, harmless, layer=13)   # [n_l, d_model]
  d_hat = unit(diff_of_means(H, L))             # [d_model]
  proj = H @ d_hat                              # [n_h]
  ```
- **`jaxtyping` annotations** on public-facing signatures in `src/`:
  ```python
  def diff_of_means(
      H: Float[Tensor, "n_h d_model"],
      L: Float[Tensor, "n_l d_model"],
  ) -> Float[Tensor, "d_model"]: ...
  ```
  Not yet adopted but the ruff config (`F722` ignored) is ready for it.

Pick one per function. Don't mix.

### Device + dtype

- **Set device once** at module/script load via `_auto_device()` in `src/model.py`. Pass explicitly to every tensor creation; **never use `.cuda()`** (deprecated convention, breaks MPS).
- **Set dtype once** via `_auto_dtype(device, model_name)`. Gemma on MPS = fp16 (Phase 1 cache compat). Everything else on MPS = bf16. CUDA = bf16. CPU = fp32.
- Use **`torch.from_numpy(arr)`** when converting from numpy — `torch.tensor(arr)` makes a copy and re-guesses dtype.
- Annotate every `torch.zeros / ones / empty / randn` call with `device=...` and `dtype=...`. No `.to(device)` afterthoughts.

### Seeds

- One **`SEED`** constant per runner near the top.
- Use `transformers.set_seed(SEED)` at script entry — it seeds Python, NumPy, and torch in one call. Borrow from Wollschläger ([`rdo.py`](https://github.com/wollschlager/geometry-of-refusal/blob/main/rdo.py)).
- Pass `torch.Generator(device=device).manual_seed(seed)` explicitly to ops that take a `generator=` argument (e.g. `torch.randn`). Don't rely on the default generator.
- Per-experiment splits: pass `seed=...` to `np.random.default_rng(seed)`, not `np.random.seed(...)`.

### Hooks

- TransformerLens hooks **must be lifecycle-managed**. Always:
  ```python
  with ablate_dir(model, d_hat):
      # everything that should see the ablation
  ```
  Never bare `model.add_hook(...)` without a paired `remove_hooks()`.
- `@torch.no_grad()` decorates every inference function in `src/causal_metric.py` and `src/activations.py`. If you write a new inference function, add it.

### Imports

- Modern type syntax: `list[int]`, `dict[K, V]`, `X | None`. Legacy `typing.Optional/Union/List/Dict/Tuple` are banned via `ruff.toml`'s `flake8-tidy-imports.banned-api` — change them with `ruff check --fix`.
- Use `from __future__ import annotations` at the top of every src/ file. Lets annotations stay forward-referenced without runtime cost.
- Imports sorted by `ruff format` (isort behind the scenes). Don't hand-order.

## Tools

```bash
make test       # 53 unit tests, no model load (~4s)
make lint       # ruff check src/ experiments/ tests/
make lint-fix   # ruff check --fix (auto-fix safe rules only)
make format     # ruff format
make check      # lint + test
make audit      # re-run vocab + length audit on data/code_contrastive_matched.jsonl
```

Linter is **ruff 0.14+**. Config in `ruff.toml` — modeled on SAELens
(`UP/TID/I/F/E/ARG/SIM/RET`) with research-code escape hatches for
unicode-in-math (`d̂/Δ/≈/×`), scientific-shorthand vars (`l/h/I/x`),
and `print()` in runners.

## Anti-patterns (concrete things to avoid)

Drawn from observation of mech-interp codebases that have decayed:

- **Deeply nested config dicts** like `cfg["model"]["optimizer"]["beta1"]`. Use `@dataclass(frozen=True)` instead. If you have ≥5 hyperparameters, that's the threshold; below that, just argparse.
- **Magic constants in functions.** Move to module-level: `SEED = 0`, `DEFAULT_LAYER = 13`. Especially seeds — every reviewer asks where they came from.
- **Copy-pasted `model = AutoModelForCausalLM.from_pretrained(...)`** in N runners with subtle device/dtype differences. Use `experiments._runner.get_model(name)` — it's memoized per-process.
- **Hooks that silently mutate captured state.** Always wrap in `with` blocks. `try/finally` if you must construct the hook outside the block.
- **Untracked seeds** — `np.random.seed()` mutates global state. Use `np.random.default_rng(seed)` and pass the generator explicitly.
- **Long-lived state outside `data/` / `artifacts/`.** Don't `pickle.dump(...)` into `experiments/`.

## What we deliberately don't enforce

If you've worked in production Python, these omissions might feel wrong.
They're deliberate — ML research code has different tradeoffs.

- **Per-function docstrings** (`D*`). The result is documented in
  markdown + the writeup. Function bodies are short; readers parse
  signatures directly.
- **Function-length / cyclomatic-complexity limits** (`PLR0915`,
  `PLR0912`, etc.). Experiment runners' `main()` is legitimately
  100-200 lines of orchestration. Extracting helpers just to make
  the metric look better creates worse indirection.
- **Bandit security suite** (`S*`). No production threat model. `eval`,
  `pickle.load`, `torch.load(weights_only=False)` on local files are
  fine.
- **Strict typing** (`ANN*`, mypy --strict). Annotate at API boundaries
  (`src/` public functions). Leave `experiments/*.py` runners untyped
  if it's friction.
- **No `print` statements** (`T20`). `print()` is a primary research
  output. Logging matters in production; `print` matters in research.

When in doubt, look at what SAELens or ARENA does. If neither enforces
the rule, we probably shouldn't either.

## References

- `HOW_IT_WORKS.md` — end-to-end Phase 1 walkthrough with every tensor index explained.
- `READING_GUIDE.md` — 5-pass reading sequence to understand the project deeply.
- `results/phase1_writeup.md` — the headline finding + methodology + retractions.
- `data/README.md` — contrastive-set provenance + the vocab/length audit results.

Peer repos worth referencing when in doubt:

- **[SAELens](https://github.com/jbloomAus/SAELens)** — closest stylistic match. Uses ruff, bans legacy typing, has pre-commit hooks. Most disciplined small mech-interp library.
- **[Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT)** — best example of inline shape comments + minimal-tooling research code that's still readable.
- **[ARENA 3.0](https://github.com/callummcdougall/ARENA_3.0)** — heavy jaxtyping use; ignores `F722, E741, E731, E402`. Reference for jaxtyping adoption.
- **[Karpathy's coding-style guidelines](https://github.com/forrestchang/andrej-karpathy-skills/blob/main/skills/karpathy-guidelines/SKILL.md)** — short and worth reading.
