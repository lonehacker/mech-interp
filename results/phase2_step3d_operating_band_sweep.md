# Phase 2 Step 3d — operating-band addition sweep (Qwen/Qwen2.5-3B-Instruct)

- d̂ extracted from L14 (held constant)
- Natural scale at L14: 8.937
- Inject layers: [3, 7, 11, 15, 19, 23, 27, 31]
- Coefficients (× natural scale): [0.5, 1.0, 2.0, 4.0]
- Target: 10 CodeAlpaca-harmless prompts (subset of step 3b's 30, seed 1)
- Baseline (no hook) substring refusal: **0.000**

## Substring-scorer refusal-rate heatmap

| coeff × | L3 | L7 | L11 | L15 | L19 | L23 | L27 | L31 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5× | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 1.0× | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 2.0× | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 4.0× | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

**Best cell:** `add_L3_c0.5x` (lift +0.000 vs baseline)

Heatmap figure: `artifacts/figures/phase2_step3d_operating_band_sweep.png`

Per-prompt completions in `artifacts/runs/phase2_step3d/<timestamp>/result.json`.
Pre-registration: `results/phase2_step3d_preregistration.md`.
