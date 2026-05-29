"""
Post-sweep analyzer for phase2_step3d_operating_band_sweep results.

Implements the planner's read-order:
  1. Identify the LOWEST-coefficient cell with substring lift > threshold.
     (Not max lift — lowest coeff that worked. A 1× cell beats a 4× cell
     at the same rate because 4× is more likely to be magnitude-degeneration.)
  2. Dump 4 completions from that cell.
  3. Dump baseline completions on the same prompts for comparison.
  4. If specificity_control exists, dump 4 of its completions too.
  5. Coherence sentinel: mean_chars at each cell vs baseline.
  6. Assign A/B/C leaf with the evidence chain.

Run:
    python -m experiments.phase2_step3d_analyze [path/to/result.json]
    # default = newest result under artifacts/runs/phase2_step3d/
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


REPO = Path("/Users/anshulsinghle/safe_ai/mech-security")
RUNS = REPO / "artifacts/runs/phase2_step3d"


def _latest_result() -> Path:
    candidates = sorted(RUNS.glob("*/result.json"))
    if not candidates:
        raise SystemExit("No phase2_step3d results yet.")
    return candidates[-1]


def main(path: str | None = None):
    p = Path(path) if path else _latest_result()
    print(f"Analyzing: {p}")
    rec = json.loads(p.read_text())

    base_rate = rec["baseline_substr_refusal_rate"]
    cells = rec["cell_meta"]
    inject_layers = rec["inject_layers"]
    coeff_mults = rec["coeff_mults"]
    nat_scale = rec["natural_scale"]
    LIFT_THRESHOLD = 0.30  # planner's "induced refusal" threshold

    # === Coherence + rate per cell — full heatmap ===
    print(f"\nBaseline refusal: {base_rate:.2f}  | mean baseline chars: "
          f"{sum(len(c) for c in rec['baseline_gen']) / len(rec['baseline_gen']):.0f}")
    print(f"\n{'cell':<25} {'rate':>6} {'lift':>7} {'mean_chars':>11}  flag")
    print("-" * 65)
    for cmult in coeff_mults:
        for L in inject_layers:
            name = f"add_L{L}_c{cmult}x"
            m = cells[name]
            flag = ""
            if m["lift_vs_baseline"] >= LIFT_THRESHOLD:
                flag = "  ★ LIFTED"
            if m["mean_chars"] < 50:
                flag += "  ⚠️ DEGENERATE (short)"
            elif m["mean_chars"] < 200 and m["lift_vs_baseline"] > 0:
                flag += "  ⚠️ likely-degenerate (short + lifted)"
            print(f"  {name:<23} {m['refusal_rate_substr']:>5.2f}  "
                  f"{m['lift_vs_baseline']:>+5.2f}   {m['mean_chars']:>9.0f}{flag}")

    # === Identify lowest-coeff lifted cell (planner's primary read) ===
    print("\n" + "=" * 70)
    print("LOWEST-COEFFICIENT lifted cells (the cells that matter)")
    print("=" * 70)
    lifted_low_first = []
    for cmult in sorted(coeff_mults):  # ascending: smallest first
        for L in inject_layers:
            name = f"add_L{L}_c{cmult}x"
            m = cells[name]
            if m["lift_vs_baseline"] >= LIFT_THRESHOLD:
                lifted_low_first.append((cmult, L, name, m))
        if lifted_low_first:
            break  # only show the smallest-coeff group

    if not lifted_low_first:
        print("  NONE — no cell hit substring lift ≥ %.2f at any (layer × coefficient)." % LIFT_THRESHOLD)
        print("\n  >>> Leaf candidate: (A) GENUINELY INERT <<<")
        print("\n  Sanity check: print 3 baseline completions and 3 completions from")
        print("  the highest-rate cell (likely 4× cell — magnitude artifact territory):")
        max_cell_name = max(cells, key=lambda k: cells[k]["refusal_rate_substr"])
        max_cell = cells[max_cell_name]
        print(f"\n  Highest-rate cell: {max_cell_name}  rate={max_cell['refusal_rate_substr']:.2f}  "
              f"mean_chars={max_cell['mean_chars']:.0f}")
        _dump_completions(rec, "baseline", rec["baseline_gen"][:3], indent="  ")
        _dump_completions(rec, max_cell_name, rec["completions"][max_cell_name][:3], indent="  ")
    else:
        cmult, L, name, m = lifted_low_first[0]
        print(f"  Lowest lift coefficient: {cmult}×  ({len(lifted_low_first)} cells at this multiplier)")
        for cmult2, L2, name2, m2 in lifted_low_first:
            print(f"    {name2}  rate={m2['refusal_rate_substr']:.2f}  "
                  f"lift={m2['lift_vs_baseline']:+.2f}  mean_chars={m2['mean_chars']:.0f}")
        print(f"\n  PRIMARY CELL TO EYEBALL: {name} (lowest layer at smallest lifted coeff)")
        print(f"\n  Coherence check: mean baseline chars = "
              f"{sum(len(c) for c in rec['baseline_gen']) / len(rec['baseline_gen']):.0f}, "
              f"this cell's mean_chars = {m['mean_chars']:.0f}")
        if m["mean_chars"] < 200:
            print(f"  ⚠️ mean_chars dropped substantially — possible degeneration.")
        else:
            print(f"  ✓ mean_chars in normal range — coherence likely preserved.")
        _dump_completions(rec, "baseline", rec["baseline_gen"][:4], indent="  ")
        _dump_completions(rec, name, rec["completions"][name][:4], indent="  ")

        if rec.get("specificity_control"):
            sc = rec["specificity_control"]
            print(f"\n  Specificity control at best cell ({sc['cell']}):")
            print(f"    rate={sc['refusal_rate_substr']:.2f}  lift={sc['lift_vs_baseline_substr']:+.2f}")
            if "completions" in sc:
                _dump_completions(rec, sc["cell"] + " (random vector)", sc["completions"][:4], indent="  ")

    # === Leaf assignment ===
    print("\n" + "=" * 70)
    print("A/B/C LEAF ASSIGNMENT")
    print("=" * 70)
    if not lifted_low_first:
        print("  (A) Genuinely inert — no addition cell induced refusal at any layer/scale.")
        print("  Next step: this is the strongest §3 thesis version; d̂ is causally inert on Qwen.")
        print("  Confirmatory check: optionally run step 3c ablation at multiple extraction")
        print("  layers; if all are also null, (A) is locked.")
    else:
        cmult, L, name, m = lifted_low_first[0]
        if cmult <= 1.0:
            print(f"  Candidate: (B) Causal — d̂ induces refusal at {cmult}× of natural scale at L{L}.")
            print(f"  → Eyeball completions above. If coherent refusal AND specificity holds,")
            print(f"    fire step 3c ablation at L{L} (or its neighborhood) to confirm the")
            print(f"    operating layer. If ablation there also collapses refusal → branch (C).")
            print(f"    If ablation still null at L{L} → asymmetric Qwen (causal additively,")
            print(f"    robust to ablation) — a more interesting finding than A or B.")
        else:
            print(f"  Inconclusive — lift only at coeff ≥ {cmult}× which is at the magnitude-")
            print(f"  artifact threshold. Eyeball completions AND specificity control rigorously")
            print(f"  before assigning (B). If random vector at same coeff also induces 'refusal'-")
            print(f"  shaped output, the lift is the magnitude artifact and the leaf is (A).")


def _dump_completions(rec, label, gens, indent=""):
    print(f"\n{indent}--- {label} completions ---")
    prompts = rec["harmless_subset"]
    for i, g in enumerate(gens):
        p = prompts[i][:60]
        clip = g[:200].replace("\n", " ")
        print(f"{indent}  [{i}] {p}...")
        print(f"{indent}      → {clip!r}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
