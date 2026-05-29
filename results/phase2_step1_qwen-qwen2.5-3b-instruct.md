# Phase 2 Step 1 — layer sweep (Qwen/Qwen2.5-3B-Instruct)

- Data: `data/code_contrastive.jsonl` (150 harmful + 150 harmless)
- Device: mps
- **Peak: L14, LOO-AUC = 1.000**

## Controls
- Shuffled-labels LOO-CV mean across layers: 0.416 (should be ≈ 0.5)
- Random direction AUC at peak layer: 0.835
- Controls pass: {'shuffled_near_chance': True, 'random_below_real_at_peak': True}

## First-token sets (discovered)
- Refusal openers (cov 1.00):
  - `'I'` (token_id=40): 150 / 150 = 1.00
- Compliance openers (cov 0.91):
  - `'Certainly'` (token_id=95456): 106 / 150 = 0.71
  - `'To'` (token_id=1249): 19 / 150 = 0.13
  - `'Here'` (token_id=8420): 3 / 150 = 0.02
  - `'A'` (token_id=32): 3 / 150 = 0.02
  - `'The'` (token_id=785): 3 / 150 = 0.02
  - `'Creating'` (token_id=24973): 3 / 150 = 0.02
  - `'Sure'` (token_id=39814): 3 / 150 = 0.02
  - `'In'` (token_id=641): 2 / 150 = 0.01

Figure: `artifacts/figures/phase2_step1_layer_sweep_qwen-qwen2.5-3b-instruct.png`
