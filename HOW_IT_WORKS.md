# How the refusal-direction intervention works, end to end

An ML-engineer-level walkthrough of what every line of `mech_security` is
doing and why. By the end you should be able to read the code without
looking up tensor shapes, why a specific index was chosen, or why we cache
intermediates one way and not another.

Audience: ML engineer who hasn't done mech-interp before. Familiar with
PyTorch and the residual-stream idea, but not with the conventions of
TransformerLens or the index choices that refusal-direction work has to
make right.

The four files to read alongside this doc:

- `mech_security/model.py` — load the model, format prompts, tokenize, generate.
- `mech_security/activations.py` — cache the residual stream at the layer (and position) you care about.
- `mech_security/directions.py` — extract refusal directions, ablate or add them during a forward pass, measure their causal effect.
- `mech_security/causal_metric.py` — continuous first-token-logit-shift metric (Phase 1.5).

The runnable starting point: `experiments/phase1_step3_steering.py` — does
Steps 1–6 below end-to-end on Gemma-2-2b-it and reports the 12/12 → 0/12
refusal-rate collapse on 12 held-out AdvBench prompts.

---

## Mental model first: two orthogonal choices

Every refusal-direction experiment makes **two independent decisions** that
this codebase keeps deliberately separate. Conflating them produces wrong
conclusions; the cleanest example is the Phase 2 Qwen result.

### Axis 1 — Layer-selection criterion

Choosing WHICH layer (and token position) to extract or intervene at:

- **AUC-layer-selection** — train a linear probe on harmful-vs-harmless
  activations at each layer; pick the peak. Used in Phase 1 to land on L13
  for Gemma.
- **bypass-gap-layer-selection** — ablate a candidate direction at each
  layer × position cell; pick the one with the largest refusal-token logit
  drop. Causal-effect criterion. Used by Wollschläger's `select_direction`
  step.

Layer-selection PRODUCES a layer. It is not an extraction method.

### Axis 2 — Extraction method

Given a layer, compute the direction vector:

- **`diff_of_means`** → the vector we call **`d̂`**. Statistical:
  difference of class centroids at the chosen layer. (`mech_security.directions.diff_of_means`)
- **`lda_directions`** → top-k orthogonal Fisher-LDA directions. Also
  statistical. Phase 1 hardened-subspace ablation tests these as
  classification-equivalent-but-causally-inert. (`mech_security.directions.lda_directions`)
- **RDO** (Refusal Direction Optimization, Wollschläger 2025) → the vector
  we call **`d_rdo`**. Gradient: optimize a direction at a FIXED layer so
  that ablating it maximally drops refusal. External code at
  `~/safe_ai/geometry-of-refusal/rdo.py`.

`d̂` and `d_rdo` are different vectors even at the same layer — they're not
synonyms; `d̂` is by definition the diff-of-means output.

### The 2×2 (and why you need to think about it)

|  | diff-of-means (`d̂`) | RDO (`d_rdo`) |
|---|---|---|
| **AUC-layer-selection** | Phase 1 headline (Gemma L13: refusal 1.0→0.0). Phase 2 at L14 on Qwen: **inert** — AUC saturated, layer arbitrary. | Not interesting — AUC-picked layer is wrong on Qwen. |
| **bypass-gap-layer-selection** | **Phase 2 Part 2 (Qwen L22): refusal 1.0→0.0.** Diff-of-means works fine once the layer is causally selected. | Wollschläger end-to-end = bypass-gap-layer-selection → RDO extraction. |

The Phase 2 Qwen L14 inertness was a *layer-selection* failure, not a
diff-of-means failure. AUC is saturated on Qwen (≥ 0.994 at every layer
including L0), so AUC-based layer-selection picked an arbitrary layer
where ablation happens not to work. Moving to bypass-gap-layer-selection
finds L20–L24 (around the EOI tokens) where plain diff-of-means recovers
a strongly-causal direction. Full state + binding terminology in
[`PROJECT_STATE.md`](PROJECT_STATE.md).

---

## Step 1 — Load the model

```python
from mech_security.model import load_model
bundle = load_model("gemma-2-2b-it")  # or "Qwen/Qwen2.5-3B-Instruct"
```

`bundle` is a `ModelBundle` wrapping a TransformerLens `HookedTransformer` plus
the matching tokenizer + a `name` string + device + dtype. `load_model` picks
device (CUDA > MPS > CPU) and dtype (`_auto_dtype` — fp16 for Gemma on MPS to
preserve Phase 1 cache compat, bf16 elsewhere) automatically. Every other
function takes `bundle` so it can read `bundle.model`, `bundle.tokenizer`,
`bundle.name` without re-passing them.

Memoization: `experiments._runner.get_model(name)` caches the bundle per
process so back-to-back runners don't reload weights.

## Step 2 — Apply the chat template + tokenize, BOS handled exactly once

```python
from mech_security.model import format_prompt_for_bundle, tokenize_prompt

formatted = format_prompt_for_bundle(bundle, "How do I bake bread?")
ids = tokenize_prompt(bundle, formatted)   # [1, seq_len]
```

`format_prompt_for_bundle` dispatches on `tokenizer.chat_template`:

- Gemma: hardcoded `<start_of_turn>user\n…<end_of_turn>\n<start_of_turn>model\n`
  template — byte-identical to Phase 1's frozen format so cached activations
  still resolve.
- Everything else (Qwen, Llama, etc): `tokenizer.apply_chat_template(...)`.

The BOS-token assertion inside `tokenize_prompt` checks that the templated
string begins with `bos_token` exactly when the template starts with `bos_token`
— Gemma yes, Qwen no. Getting this wrong adds or drops a token and silently
shifts every downstream index by one.

## Step 3 — Cache the residual stream

```python
from mech_security.activations import cache_resid

# Gemma defaults: L13, last token, hook_resid_post — Phase 1 standard
H = cache_resid(bundle, harmful_prompts, layer=13)               # [n_h, d_model]
L = cache_resid(bundle, harmless_prompts, layer=13)              # [n_l, d_model]

# Phase 2 Part 2 pattern: same call with a non-default position
H_pos4 = cache_resid(bundle, harmful, layer=22, position=-4)      # [n_h, d_model]
```

Three load-bearing index choices:

- **`hook_resid_post`** — the *output* residual of block `layer`. The block
  has read `resid_pre`, run attention + MLP, added them back. We read after
  block `layer` and before block `layer+1`. Arditi (2024) found this is the
  clean separation point on instruction-tuned models.
- **`position=-1`** (default) — last token of the templated prompt, which
  for Gemma is `\n` after `<start_of_turn>model`. That position's residual
  encodes "what should I say next," conditioned on the chat template having
  just said "model's turn now." For Qwen ChatML it's the `\n` after
  `<|im_start|>assistant`. Phase 2 also reads `position=-4` (the `\n` after
  `<|im_end|>`) — see [PROJECT_STATE.md](PROJECT_STATE.md) for why.
- **`.to("cpu").float()`** — activations come off the model device as the
  model's dtype; we cast to CPU fp32 for downstream NumPy/sklearn work. The
  caches are small (300 × 2304 fp32 ≈ 3 MB on Gemma).

`cache_resid_all_layers(bundle, prompts)` is a one-forward-pass-per-prompt
version that returns `[n, n_layers, d_model]` — used by the layer-sweep
experiment so each prompt is forwarded once instead of `n_layers` times.

## Step 4 — Compute the refusal direction

The two-line version:

```python
from mech_security.directions import diff_of_means, unit, project

d_raw = diff_of_means(H, L)        # [d_model]  = mean(H) - mean(L)
d_hat = unit(d_raw)                # [d_model]  = d_raw / ||d_raw||
proj_h = project(H, d_hat)         # [n_h]      = H @ d_hat (each prompt's score)
```

The whole pipeline composed:

```python
from mech_security.directions import extract_d_hat

d_hat, H, L, meta = extract_d_hat(
    bundle, harmful_prompts, harmless_prompts,
    layer=13, position=-1,
)
# meta = {"natural_scale", "harmless_mean", "midpoint"}
# natural_scale ≈ 105 on Gemma L13 — used as α for addition (Step 7)
```

`d̂` is the **refusal direction**: the linear axis in activation space along
which "this prompt is harmful" is most readable. Why diff-of-means works at
all is the *linear representation hypothesis* — high-level features in the
late-mid layers of instruction-tuned models tend to live in linear
directions. The estimator picks up not only refusal but everything
correlated with the harmful/harmless labelling (vocabulary, topic,
sentiment), so `d̂` is a mixture; Step 6 is what tells you whether the
refusal component dominates causally.

**LDA as a peer extractor.** `lda_directions(H, L, k=3)` returns the top-3
orthogonal Fisher-LDA directions. Used in the hardened-subspace experiment
to check the classification-vs-causation gap: each LDA direction is a
near-perfect harmful/harmless classifier yet ablating it leaves refusal
intact (z ≈ 0 against random null). `ablate_subspace(model, dirs)` ablates
a `[k, d_model]` stack at once.

**Targeted tests to read** (no model load needed; ~1 sec each):

- `tests/test_directions.py::TestDiffOfMeans` — what `diff_of_means`
  produces on toy 2-D clusters.
- `tests/test_directions.py::TestLdaDirections` — unit norm, axis
  recovery, k>1 orthogonality, bootstrap perturbation.
- `tests/test_directions.py::TestRoundtrip::test_two_cluster_recovery` —
  the four primitives composed end-to-end on synthetic clusters. Read this
  one first; it's the shortest spec of how the pipeline fits together.
- `tests/test_reproducibility.py::TestExtractDHatComposition` — pins
  `extract_d_hat`'s numerical output on synthetic input so refactors can't
  silently change the math.

## Step 5 — Generate normally; establish the baseline refusal rate

```python
from mech_security.model import generate

test_prompts = harmful[-12:]   # held out, never used in d̂ extraction
baseline = [generate(bundle, p, max_new_tokens=160) for p in test_prompts]
```

`generate(bundle, prompt, max_new_tokens=160, temperature=0.0)` returns the
completion only (the formatted prompt prefix is stripped). T=0 is greedy
generation; `do_sample = temperature > 0.0` is False, so output is
deterministic.

Substring scoring is the cheap first-pass:

```python
from mech_security.eval import is_refusal, refusal_rate
baseline_rate = refusal_rate(baseline)  # fraction starting with "I cannot…" etc.
```

The dual-judge (`mech_security.eval_llm.judge_many`) catches what substring
misses — pivot-style refusals like "Sure, here's… actually I can't" — and
returns one of three labels per completion: REFUSED / PARTIAL_COMPLY /
COMPLIED. Used as the headline scorer for any result we report.

## Step 6 — Ablate `d̂` during generation; observe the refusal collapse

```python
from mech_security.directions import ablate_dir

with ablate_dir(bundle.model, d_hat):
    ablated = [generate(bundle, p, max_new_tokens=160) for p in test_prompts]
```

Inside `ablate_dir`, the hook function at every residual point is:

```python
def hook_fn(x, hook):
    # x: [..., d_model]
    coeff = (x * d_hat).sum(dim=-1, keepdim=True)
    return x - coeff * d_hat
```

`x · d̂` gives the scalar projection; subtracting that component leaves only
the orthogonal complement. After the hook, `x_new · d̂ == 0` at every
position.

It runs at every layer × every residual-affecting hook — the *faithful
Arditi recipe* — to close the loop where attention in block `L+1` would
otherwise read an un-modified `resid_pre`:

- `blocks.{L}.hook_resid_pre`, `_mid`, `_post`
- `blocks.{L}.hook_attn_out`, `_mlp_out`

5 hooks × n_layers per ablation. Lifecycle-managed by a context manager so
hooks are guaranteed to be removed on `__exit__` even if generation raises.

**Headline result on Gemma-2-2b-it + AdvBench: refusal drops 12/12 → 0/12
on substring, 12/12 → 2/12 on the judge** (the two missed are pivot-style
refusals the judge catches). That's the canonical "ablation collapses
refusal" result, intervention-verified.

### `bypass_gap` — the measurement primitive

The "refusal-rate drop under ablation" pattern is named:

```python
from mech_security.directions import bypass_gap

gap = bypass_gap(bundle, d_hat, test_prompts, baseline_completions=baseline)
# {"baseline_refusal", "ablated_refusal", "gap", "baseline_completions",
#  "ablated_completions", "mean_chars_baseline", "mean_chars_ablated"}
```

This is what Phase 2 Part 2 uses to rank candidate (layer, position) cells
in bypass-gap-layer-selection: extract `d̂_(L,pos)` cheaply at every cell
via Stage 1 below; then call `bypass_gap` on each candidate to rank by
causal effect.

Test pointer: `tests/test_directions.py::TestBypassGap` — gap arithmetic
+ baseline-reuse contract on stub objects, no model load.

## How the code does the per-cell sweep

Phase 2 Part 2 selects the bypass-gap-best layer × position by composing
the primitives above. Two stages:

**Stage 1 — extract a candidate direction at every cell. Cheap.** One
forward pass per prompt at each cell; centroid subtraction:

```python
candidates = {}
for L in [19, 20, 21, 22, 23, 24, 25]:        # the prior region from Step 4b
    for pos in [-1, -4]:                       # last-token + the EOI newline
        H = cache_resid(bundle, harmful_train, layer=L, position=pos)
        Lact = cache_resid(bundle, harmless_train, layer=L, position=pos)
        candidates[(L, pos)] = unit(diff_of_means(H, Lact))
```

After this loop we have 14 candidate directions — one per cell. There's no
chicken-and-egg ("how do we ablate before we have the direction?"): every
candidate exists before any ablation runs.

**Stage 2 — rank candidates by causal effect.** Expensive: full generation
per cell.

```python
baseline = generate_batch(bundle, harmful_test, max_new_tokens=160)
for (L, pos), d_hat_cell in candidates.items():
    result = bypass_gap(
        bundle, d_hat_cell, harmful_test,
        baseline_completions=baseline,         # reuse — don't regenerate per cell
    )
    print(f"L{L} pos{pos}: gap = {result['gap']:.3f}")
```

Same shape works for any (extractor × selector) combination: swap
`diff_of_means` for `lda_directions(..., k=1)[0]` in Stage 1, or swap
`bypass_gap` for an AUC-probe scorer in Stage 2. The PROJECT_STATE.md 2×2
is exactly this code structure.

Runner that does both stages end-to-end:
`experiments/phase2_part2_dim_bypass_gap_sweep.py`.

## Step 7 (optional, bidirectional check) — add `d̂` to harmless prompts

If `d̂` mediates refusal, *adding* it to harmless prompts at the right scale
should induce refusals:

```python
from mech_security.directions import add_dir

# Inject at the EXTRACTION layer; α from natural scale of H on d_hat (~105 Gemma L13)
alpha = meta["natural_scale"]
with add_dir(bundle.model, d_hat, coeff=alpha, layer=13):
    induced = [generate(bundle, p, max_new_tokens=160) for p in harmless[-50:]]
```

`add_dir(model, d_hat, coeff, layer)` is a single-hook context manager at
`blocks.{layer}.hook_resid_post`: `x ← x + coeff · d̂`.

Operating band: the addition effect is layer-localized and scale-sensitive.
The Phase 1 step3b layer × coefficient sweep finds the strongest
elicitation cell at L15 × 4× natural_scale; the Phase 2 step3d operating-
band sweep replicates this pattern on Qwen with a similar shape.

## How the continuous causal metric (Phase 1.5) works

Beyond binary refusal rates, the continuous *first-token logit-shift*
metric measures causal effect at a finer resolution:

```python
from mech_security.causal_metric import (
    compute_causal_effect,
    REFUSAL_FIRST_TOKEN_IDS_GEMMA2,
    COMPLIANCE_FIRST_TOKEN_IDS_GEMMA2,
)
effect = compute_causal_effect(
    bundle, prompts, direction=d_hat,
    refusal_token_ids=REFUSAL_FIRST_TOKEN_IDS_GEMMA2,
    compliance_token_ids=COMPLIANCE_FIRST_TOKEN_IDS_GEMMA2,
)
print(effect.effect_signed)   # negative = mass moved from refusal toward compliance
```

For each prompt, the function runs a single forward pass under the ablation
hook (no full generation — much cheaper) and reads `logits[:, -1, :]`:

```
refusal_logit    = mean over r ∈ refusal_token_ids of logits[0, -1, r]
compliance_logit = mean over c ∈ compliance_token_ids of logits[0, -1, c]
contrast         = refusal_logit - compliance_logit
effect_signed    = contrast_under_ablation - contrast_at_baseline
```

The contrast term cancels uniform damping: a direction that mildly reduces
*all* first-token mass (a magnitude artifact) would show a negative
`Δlog p(refusal)` that *looks* causal — but the matching compliance shift
subtracts it out and the contrast nets to zero. Only a *directional* shift
toward compliance survives.

z-scored against a 5-vector random null band, the actual causal direction
on Gemma sits at **z ≈ +90**; 5 LDA-bootstrap directions (each AUC=1.0 as
classifiers, near-orthogonal to `d̂`) sit **inside the null band at z ≈ 0**.
This is the hardened classification ≠ causation result.

Test pointer: `tests/test_causal_metric_discovery.py` covers the token-set
discovery procedure that runs once per model.

## The reproducibility harness

`tests/test_reproducibility.py` freezes the invariants that an aggressive
refactor (package rename, import restructure, etc.) could silently break:

- **Cache-key extra-string template** — the exact f-string format used to
  hash an activation cache. Renames of `bundle.name` rendering, dtype
  rendering, or layer formatting all get caught.
- **`content_hash` pinned outputs** — two reference inputs with their
  expected SHA256 hex.
- **Real-on-disk artifact resolution** — re-derives the cache key for the
  Phase 2 step3e matched-set L14 d̂ extraction and asserts it equals
  `c101587891347bd3` (an actual file on disk). If the rename touched any
  byte in the path that gets hashed, this test catches it.
- **`extract_d_hat` numerical composition** — synthetic H, L → expected
  `d̂` + `natural_scale` to atol 1e-6.

`make verify` runs just this file (~3 sec). Use it as a green-light gate
before committing any refactor.

## Where each file fits

| File | What | Reading order |
|---|---|---|
| `mech_security/model.py` | Load model, format prompts, tokenize (BOS), generate | 1 |
| `mech_security/activations.py` | Cache residual stream at one layer + one position | 2 |
| `mech_security/directions.py` | `diff_of_means`, `lda_directions`, `unit`, `project`, `ablate_dir`, `ablate_subspace`, `add_dir`, `extract_d_hat`, `bypass_gap` | 3 |
| `mech_security/causal_metric.py` | Continuous first-token logit-shift metric + per-model token-set discovery | 4 |
| `mech_security/eval.py` | Substring refusal scorer | 5 |
| `mech_security/eval_llm.py` | Calibrated Claude judge (Haiku 4.5 primary, Opus 4.7 cross-check) | 6 (optional) |
| `mech_security/probes.py` | Per-layer logistic-regression probing | 7 (optional) |
| `experiments/phase1_step3_steering.py` | End-to-end runnable example (Phase 1 headline) | run after reading |
| `experiments/phase2_part2_dim_bypass_gap_sweep.py` | The bypass-gap-layer-selection sweep (Phase 2 headline) | run after reading the 2×2 |

`mech_security/` modules produce tensors and contexts; `experiments/`
runners orchestrate; `results/` reports numbers. No interpretation happens
in `mech_security/` — that's the point of "function shape: activations in,
tensors/metrics out."

## Reading order for the rest of the codebase

`READING_GUIDE.md` has a 5-pass guide for ~4–6 hours of structured reading:

1. **HOW_IT_WORKS.md** (this doc) — understand the mechanics
2. `mech_security/` modules in the order above
3. `results/phase1_writeup.md` §3–5 — the Phase 1 headline + replication
4. `results/phase1_writeup.md` §3.13 + §3.10c — hardened classification ≠ causation
5. `PROJECT_STATE.md` + `results/phase2_*.md` — Phase 2: what the layer-selection lesson is and the matched-set audit
