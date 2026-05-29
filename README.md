# mech-interp — refusal-direction interventions on Gemma-2-2b-it

A controlled replication of Arditi-style refusal-direction interventions on
`gemma-2-2b-it`, with full Arditi controls, a calibrated LLM judge, a
bootstrap stability check, and an explicit statistical-vs-gradient
extraction decoupling.

**The writeup (read this first):** [Classification ≠ Causation in Refusal
Directions — A Replication on Gemma-2-2b-it](https://lonehacker.github.io/mech-interp/)

**The headline finding, one sentence:** on Gemma-2-2b-it, ablating the L13
diff-of-means direction drops refusal **from 99% to 8% on HarmBench (n=200,
dual-judge cross-check) — a 91-percentage-point drop** with perfect
specificity (random direction = 99% refusal). Five
classification-equivalent alternatives extracted by other statistical
methods (3 bootstrap LDA directions, a 5-D LDA subspace, diff-of-means
from a different layer) are perfect classifiers but causally inert under
the same ablation. TinyMMLU accuracy is preserved (Δ = +0.03, within
noise) — the ablation is refusal-specific, not lobotomization.

**Scope:** replication + methodology contribution. Not a novel finding.
Every conceptual claim is in the published literature (Arditi 2024,
Wollschläger ICML 2025, Winninger 2025). The contribution is the controls,
the calibrated judge, the bootstrap stability check, and the explicit
statistical-vs-gradient decoupling result.

## Where to start

| If you want to… | Read |
|---|---|
| **Understand the finding** in 10 minutes | [The writeup](https://lonehacker.github.io/mech-interp/) (also at `docs/index.html`) |
| **Deep-read the project** in a structured 4–6 hours | [`READING_GUIDE.md`](READING_GUIDE.md) — five-pass reading sequence |
| **Verify the numbers** | [`VERIFY.md`](VERIFY.md) — file-by-file verification checklist |
| **Browse the actual results** | [`results/`](results/) — every experiment has a markdown file with raw numbers |
| **Read the code** | [`src/`](src/) for the libraries, [`experiments/`](experiments/) for the runners |
| **See raw model completions** | `artifacts/runs/<step>/<timestamp>/result.json` — per-prompt prompt+completion pairs |
| **See the figures** | [`artifacts/figures/`](artifacts/figures/) — 14 PNGs |

## Headline result table

| Cell | Direction ablated (Arditi multi-layer recipe) | Refusal rate (12 held-out harmful) |
|---|---|---:|
| A | (baseline, no intervention) | 12 / 12 |
| **B** | **L13 diff-of-means d̂** | **0 / 12** |
| C1 | Bootstrap-101 LDA-top-1 | 12 / 12 |
| C2 | Bootstrap-202 LDA-top-1 | 12 / 12 |
| C3 | Bootstrap-303 LDA-top-1 | 12 / 12 |
| D | Top-5 LDA orthogonal subspace | 12 / 12 |
| E | L3 diff-of-means (cos 0.08 with L13) | 12 / 12 |
| F | Random unit vector (control) | 12 / 12 |

Only L13 diff-of-means is causal. Five other classification-equivalent
directions behave identically to a random direction under ablation.
Details: [`results/phase1_subspace_ablation.md`](results/phase1_subspace_ablation.md).

This result does **not refute** Wollschläger et al.'s ICML 2025
multi-dimensional refusal-cone claim on the same model — their multi-D
directions are recovered by **gradient-based** Refusal Direction
Optimization (RDO), not by statistical extraction. The methodological
contribution is making the statistical-vs-gradient extraction decoupling
explicit and showing what statistical methods do vs do not recover on this
model.

## Reproducing the numbers

```bash
# Set up
git clone https://github.com/lonehacker/mech-interp.git
cd mech-interp
pip install -r requirements.txt

# Accept gemma-2-2b-it license on Hugging Face; export HF_TOKEN
# Optionally set ANTHROPIC_API_KEY for the LLM-judge re-audit (substring
# scorer works without it but undercounts pivot-style refusals)

# Verify (43 unit tests + reproducibility script with 15 cross-checks)
python -m pytest tests/ -v
python -m experiments.repro

# Re-run any experiment end-to-end (each ~5-40 min on MPS, much faster on A100)
python -m experiments.phase1_step2_layer_sweep        # layer sweep
python -m experiments.phase1_step3_steering            # the causal test
python -m experiments.phase1_step3b_addition_sweep     # the C4 reframe
python -m experiments.phase1_subspace_ablation         # the headline result
python -m experiments.phase1_step5_localization        # single-layer ablation
python -m experiments.phase1_mechanics_and_generality  # CPU-only depth + cross-harm
python -m experiments.phase1_tinymmlu_capability       # capability check
```

## Phase 2 — what comes next

The Phase 2 protocol (writeup §9) targets an under-studied open-weight
model (Qwen-2.5-Coder, DeepSeek-Coder-V2, Phi-3 variants) and includes
**RDO gradient extraction** as a mandatory step — without it, the
multi-direction claim from Wollschläger cannot be tested. Setup notes for
adapting Wollschläger's published RDO code (`github.com/wollschlager/geometry-of-refusal`)
to MPS are at `../geometry-of-refusal/RDO_REPLICATION_SETUP_NOTES.md`
(not in this repo; the Wollschläger code is cloned alongside).

## Repository layout

```
.
├── docs/index.html              # The published writeup (GitHub Pages)
├── README.md                    # This file
├── READING_GUIDE.md             # 5-pass guide for deep reading
├── VERIFY.md                    # Verification checklist
├── data/
│   ├── contrastive.jsonl        # 150 AdvBench harmful + 150 length-matched Alpaca harmless (frozen)
│   ├── affect-test.jsonl        # Affect-decoupling probe set
│   └── harmbench_behaviors_text_all.csv  # HarmBench (downloaded from canonical GitHub CSV)
├── src/                         # Core libraries
│   ├── model.py                 # gemma-2-2b-it loader + chat-template helpers
│   ├── activations.py           # residual-stream caching
│   ├── directions.py            # diff-of-means + ablate_dir + add_dir
│   ├── probes.py                # logistic-regression probes + shuffled control
│   ├── eval.py                  # substring refusal scorer
│   ├── eval_llm.py              # calibrated Claude judge (Haiku 4.5)
│   └── controls.py              # contrastive-set audit
├── experiments/                 # Runners — one file per experiment
├── results/                     # Markdown summaries (see results/README.md for the index)
├── artifacts/
│   ├── figures/                 # 13 PNGs referenced in the writeup
│   └── runs/                    # per-run JSON results with prompt+completion pairs
└── tests/                       # 43 unit tests (pytest)
```

The activation-cache directory (`artifacts/cache/`) is gitignored —
it's a 100 MB binary cache reproducible by running the experiments.

## Method ladder (tier 2)

1. **Diff-of-means refusal direction** — Arditi et al. (2024)
2. **Per-layer linear probing** — second line of evidence for the
   feature's readability
3. **Ablation / addition steering** — the causal test
4. **Layer-restricted ablation** — localization
5. **Iterative LDA + bootstrap stability** — methodological test for
   multi-direction claims
6. **Calibrated LLM judging** — controls for substring-scorer failure modes

Explicitly **out of scope for v1**: SAE training, training a model from
scratch, multimodal extensions. **Phase 2 will add**: RDO gradient
extraction, HarmBench + StrongREJECT benchmarks, port to an under-studied
open-weight model.

## Responsible scope

Defensive characterization, not an attack cookbook. The artifact reports
mechanism, localization, robustness, and per-category refusal-rate
deltas — measurements, not capability uplift. Per-prompt completions are
persisted under `artifacts/runs/` for verification; harmful content
is from the public AdvBench and HarmBench datasets and is not
expanded beyond what those benchmarks already publish.

## Citation

If you use this code or the methodology described in the writeup, please
also cite the foundational papers it replicates:

- Arditi et al. (2024) — Refusal in LLMs is Mediated by a Single Direction
- Wollschläger et al. (ICML 2025) — The Geometry of Refusal in LLMs:
  Concept Cones and Representational Independence ([arXiv:2502.17420](https://arxiv.org/abs/2502.17420))
- Winninger (2025) — Subspace Rerouting

## License

This project's code: see `LICENSE` (if present). Replicated methodology
follows the published papers' methods and is reproduced for safety
research purposes.
