"""
Phase 1 — Step 3d: addition-half HEADLINE promotion + matched random control.

What the Step 3b sweep established with n=10: there exists a (layer, coeff)
operating point where addition of d_hat to harmless prompts induces refusal
without breaking coherence. The peak cell at L=3 with coeff ≈ 1.0× the
natural scale of x·d_hat is a candidate.

What Step 3b did NOT establish:
  1. A random unit vector at the same operating point produces Δ refusal = 0.
     (At the prior L13/coeff=25 cell, random Δ=0 was uninformative because the
     real Δ was also 0. With the new operating point, random Δ=0 IS the
     headline control.)
  2. The result holds at AdvBench-derived N. 10/10 has Wilson 95% CI ≈
     0.72–1.00 — too wide to support "addition induces refusal" as a clean
     claim. Promote N to 50.

This script:
- Re-splits contrastive.jsonl under a new seed (100 train + 50 test per side)
  so the held-out N=50 is disjoint from d_hat extraction.
- Extracts d_hat at the extraction layer (default L=13).
- Caches activations at the INJECTION layer to compute natural scale there
  (the portable knob — see writeup notes). Logs natural scale per injection
  layer for cross-model comparison in Phase 2.
- Generates a SEED-MATCHED random unit vector.
- Runs FOUR cells at the headline injection layer:
    real_at_inj_scale:   add (1.0× natural_scale_at_inj) × d_hat at L_inj
    rand_at_inj_scale:   add (1.0× natural_scale_at_inj) × r_hat at L_inj  ← matched abs coeff
    real_at_extract_scale: add (1.0× natural_scale_at_extract) × d_hat at L_inj  ← what Step 3b actually did
    no_hook baseline:     baseline harmless refusal floor
- Wilson 95% CI on every refusal rate.
- Saves every completion to result.json for (H) eyeball.

Run:
    cd ~/safe_ai/mech-security
    python -m experiments.phase1_step3d_addition_headline
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from experiments._runner import (
    RESULTS,
    cached_activations,
    content_hash,
    get_logger,
    get_model,
    load_jsonl_pairs,
    new_run_dir,
    write_json,
)
from src.activations import cache_resid
from src.directions import add_dir, diff_of_means, project, random_unit_vector, unit
from src.eval import coherence_ok, refusal_rate
from src.model import generate

log = get_logger("phase1_step3d")


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% CI for a proportion k/n. Closed-form, no statsmodels."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def _gen(bundle, prompts, max_new_tokens=96):
    return [generate(bundle, p, max_new_tokens=max_new_tokens, temperature=0.0)
            for p in prompts]


def _natural_scale(harmful_acts: torch.Tensor, harmless_acts: torch.Tensor,
                   d_hat: torch.Tensor) -> dict:
    """Per-layer natural scale of x · d_hat: harmful-mean − harmless-mean.

    The class-separation scale at a single layer. For Phase 2 cross-model
    portability the layer-pinned scale is the portable knob — absolute
    coefficients are not.
    """
    h_proj = project(harmful_acts, d_hat)
    l_proj = project(harmless_acts, d_hat)
    return {
        "harmful_proj_mean": float(h_proj.mean()),
        "harmless_proj_mean": float(l_proj.mean()),
        "harmful_proj_std": float(h_proj.std()),
        "harmless_proj_std": float(l_proj.std()),
        "natural_scale": float(h_proj.mean() - l_proj.mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract-layer", type=int, default=13)
    ap.add_argument("--inject-layer", type=int, default=3,
                    help="Headline injection layer (default 3 — the Step 3b peak).")
    ap.add_argument("--data", type=str, default="data/contrastive.jsonl")
    ap.add_argument("--n-test", type=int, default=50,
                    help="Held-out harmless prompts (target Wilson CI ≥ 0.93 at 1.00).")
    ap.add_argument("--n-train", type=int, default=100,
                    help="Training prompts per side for d_hat extraction.")
    ap.add_argument("--split-seed", type=int, default=1,
                    help="Distinct from Step 3's seed=0 so the test set is fresh.")
    ap.add_argument("--rand-seed", type=int, default=7,
                    help="Random unit vector seed.")
    args = ap.parse_args()

    run_dir = new_run_dir("phase1_step3d")
    log.info("run_dir: %s | extract=L%d inject=L%d n_train=%d n_test=%d",
             run_dir, args.extract_layer, args.inject_layer, args.n_train, args.n_test)

    bundle = get_model()
    log.info("model: %s | device=%s", bundle.name, bundle.device)

    # Re-split. Seed=1 (Step 3 used seed=0) so the held-out set is genuinely new.
    pairs_path = Path(args.data)
    if not pairs_path.is_absolute():
        pairs_path = Path(__file__).resolve().parent.parent / pairs_path
    harmful_all, harmless_all = load_jsonl_pairs(pairs_path)
    if args.n_train + args.n_test > len(harmful_all):
        raise SystemExit(
            f"n_train + n_test = {args.n_train + args.n_test} > "
            f"available harmful ({len(harmful_all)})"
        )

    rng = np.random.default_rng(args.split_seed)
    h_perm = rng.permutation(len(harmful_all))
    l_perm = rng.permutation(len(harmless_all))
    train_h = [harmful_all[i] for i in h_perm[:args.n_train]]
    test_h = [harmful_all[i] for i in h_perm[args.n_train:args.n_train + args.n_test]]
    train_l = [harmless_all[i] for i in l_perm[:args.n_train]]
    test_l = [harmless_all[i] for i in l_perm[args.n_train:args.n_train + args.n_test]]
    log.info("split: %d/%d train, %d/%d test (seed=%d)",
             len(train_h), len(train_l), len(test_h), len(test_l), args.split_seed)

    # d_hat from train at extraction layer.
    extra = (
        f"{bundle.name}|dtype={bundle.model.cfg.dtype}|L{args.extract_layer}|"
        f"resid_post|last_token|split_seed{args.split_seed}|n_train{args.n_train}"
    )
    key_h_ext = content_hash(train_h, extra=extra + "|harmful")
    key_l_ext = content_hash(train_l, extra=extra + "|harmless")
    log.info("caching residuals at extract L%d ...", args.extract_layer)
    train_h_ext = cached_activations(
        key_h_ext, lambda: cache_resid(bundle, train_h, layer=args.extract_layer, show_progress=False)
    )
    train_l_ext = cached_activations(
        key_l_ext, lambda: cache_resid(bundle, train_l, layer=args.extract_layer, show_progress=False)
    )
    d_hat = unit(diff_of_means(train_h_ext, train_l_ext))
    d_hat_dev = d_hat.to(bundle.model.cfg.device).to(bundle.model.W_E.dtype)

    scale_at_extract = _natural_scale(train_h_ext, train_l_ext, d_hat)
    log.info("natural scale at extract L%d: %.2f (h_mean=%.2f, l_mean=%.2f)",
             args.extract_layer, scale_at_extract["natural_scale"],
             scale_at_extract["harmful_proj_mean"], scale_at_extract["harmless_proj_mean"])

    # Cache injection-layer activations to compute natural scale THERE.
    # The portable knob. The Step 3b sweep used scale-at-extract as the unit,
    # which is fine for d_hat extracted-and-injected at the same layer; at a
    # different injection layer it's the wrong yardstick.
    extra_inj = (
        f"{bundle.name}|dtype={bundle.model.cfg.dtype}|L{args.inject_layer}|"
        f"resid_post|last_token|split_seed{args.split_seed}|n_train{args.n_train}"
    )
    key_h_inj = content_hash(train_h, extra=extra_inj + "|harmful")
    key_l_inj = content_hash(train_l, extra=extra_inj + "|harmless")
    log.info("caching residuals at inject L%d ...", args.inject_layer)
    train_h_inj = cached_activations(
        key_h_inj, lambda: cache_resid(bundle, train_h, layer=args.inject_layer, show_progress=False)
    )
    train_l_inj = cached_activations(
        key_l_inj, lambda: cache_resid(bundle, train_l, layer=args.inject_layer, show_progress=False)
    )
    scale_at_inject = _natural_scale(train_h_inj, train_l_inj, d_hat)
    log.info("natural scale at inject L%d: %.2f (h_mean=%.2f, l_mean=%.2f)",
             args.inject_layer, scale_at_inject["natural_scale"],
             scale_at_inject["harmful_proj_mean"], scale_at_inject["harmless_proj_mean"])

    # Random unit vector — seeded once, reused at every cell.
    r_hat = random_unit_vector(bundle.d_model, seed=args.rand_seed)
    r_hat_dev = r_hat.to(bundle.model.cfg.device).to(bundle.model.W_E.dtype)
    rand_scale_at_inject = _natural_scale(train_h_inj, train_l_inj, r_hat)
    log.info("random-direction class separation at inject L%d: %.3f (expected ≈ 0)",
             args.inject_layer, rand_scale_at_inject["natural_scale"])

    # Baseline (no hook) on held-out harmless test set.
    log.info("baseline_harmless (no hook) on n=%d ...", len(test_l))
    base_gens = _gen(bundle, test_l)
    base_r = refusal_rate(base_gens)
    base_c = coherence_ok(base_gens)
    base_ci = wilson_ci(base_r.n_refused, base_r.n)
    log.info("baseline_harmless: refusal=%.2f [%.2f, %.2f], p50_chars=%.0f",
             base_r.rate, base_ci[0], base_ci[1], base_c.p50_chars)

    coeff_inj = scale_at_inject["natural_scale"]
    coeff_ext = scale_at_extract["natural_scale"]

    cells = []

    def _run_cell(name: str, vector: torch.Tensor, coeff: float, vector_label: str,
                  scale_label: str):
        log.info("cell %s: vector=%s coeff=%.2f (=%s) layer=L%d ...",
                 name, vector_label, coeff, scale_label, args.inject_layer)
        with add_dir(bundle.model, vector, coeff=coeff, layer=args.inject_layer):
            gens = _gen(bundle, test_l)
        r = refusal_rate(gens)
        c = coherence_ok(gens)
        ci = wilson_ci(r.n_refused, r.n)
        delta = r.rate - base_r.rate
        coherence_kept = c.p50_chars >= max(80, 0.6 * base_c.p50_chars)
        cell = {
            "name": name,
            "vector": vector_label,
            "coeff_absolute": coeff,
            "coeff_label": scale_label,
            "inject_layer": args.inject_layer,
            "refusal_rate": r.rate,
            "n_refused": r.n_refused,
            "n": r.n,
            "wilson_95_ci": list(ci),
            "delta_vs_baseline": delta,
            "p50_chars": c.p50_chars,
            "coherence_kept": coherence_kept,
            "completions": [{"prompt": p, "completion": g}
                            for p, g in zip(test_l, gens)],
        }
        cells.append(cell)
        log.info("    refusal=%.2f [%.2f, %.2f] (%d/%d), Δ=%+.2f, p50=%.0f, kept=%s",
                 r.rate, ci[0], ci[1], r.n_refused, r.n, delta, c.p50_chars, coherence_kept)

    # Cell 1: REAL d_hat at injection layer, coeff = scale-at-injection-layer.
    # This is the cleanest portable headline.
    _run_cell(
        "real_at_inject_scale", d_hat_dev, coeff_inj, "d_hat",
        f"1.0× natural_scale[L{args.inject_layer}]",
    )

    # Cell 2: RANDOM unit vector at injection layer, MATCHED ABSOLUTE coeff.
    # Tests "is the L_inj site special to d_hat?". Expected Δ ≈ 0.
    _run_cell(
        "rand_at_inject_scale_matched_abs", r_hat_dev, coeff_inj, "r_hat",
        f"matched absolute coeff = 1.0× natural_scale[d_hat, L{args.inject_layer}]",
    )

    # Cell 3 (sanity / Step 3b parity): REAL d_hat at injection layer with the
    # OLD scale (scale-at-extraction). This is what Step 3b actually swept.
    # Useful only if it differs materially from cell 1.
    if abs(coeff_ext - coeff_inj) > 0.1:
        _run_cell(
            "real_at_extract_scale", d_hat_dev, coeff_ext, "d_hat",
            f"1.0× natural_scale[L{args.extract_layer}] (Step 3b convention)",
        )

    record = {
        "step": "phase1_step3d",
        "model": bundle.name,
        "device": bundle.device,
        "extract_layer": args.extract_layer,
        "inject_layer": args.inject_layer,
        "split_seed": args.split_seed,
        "rand_seed": args.rand_seed,
        "n_train": args.n_train,
        "n_test": args.n_test,
        "natural_scale_at_extract": scale_at_extract,
        "natural_scale_at_inject": scale_at_inject,
        "random_class_separation_at_inject": rand_scale_at_inject,
        "baseline_harmless": {
            "refusal_rate": base_r.rate,
            "n_refused": base_r.n_refused,
            "n": base_r.n,
            "wilson_95_ci": list(base_ci),
            "p50_chars": base_c.p50_chars,
        },
        "cells": cells,
    }
    write_json(run_dir / "result.json", record)
    log.info("run record -> %s", run_dir / "result.json")

    summary_path = RESULTS / "phase1_step3d.md"
    summary_path.write_text(_render_summary(record))
    log.info("summary -> %s", summary_path)

    # Headline
    real_cell = next(c for c in cells if c["name"] == "real_at_inject_scale")
    rand_cell = next(c for c in cells if c["name"] == "rand_at_inject_scale_matched_abs")
    print(
        f"\nphase1_step3d | inject L{args.inject_layer} | "
        f"real: refusal={real_cell['refusal_rate']:.2f} {real_cell['wilson_95_ci']} | "
        f"rand: refusal={rand_cell['refusal_rate']:.2f} {rand_cell['wilson_95_ci']}"
    )
    return 0


def _render_summary(rec: dict) -> str:
    cells = rec["cells"]
    base = rec["baseline_harmless"]
    s_ext = rec["natural_scale_at_extract"]
    s_inj = rec["natural_scale_at_inject"]
    s_rnd = rec["random_class_separation_at_inject"]
    inj = rec["inject_layer"]
    ext = rec["extract_layer"]

    lines = [
        "# Phase 1 Step 3d — addition-half HEADLINE promotion + random control",
        "",
        f"**Purpose:** promote the Step 3b peak cell (L{inj}, 1.0× natural scale) "
        f"to a Wilson-CI-survivable N={rec['n_test']} and add the matched-coefficient "
        "random-direction control. This is the addition half of the four-control "
        "claim. Until both pass at this N, the claim 'addition of d_hat at L"
        f"{inj} causes refusal' is not promoted.",
        "",
        f"- Model: `{rec['model']}` on `{rec['device']}`",
        f"- d_hat extracted at L{ext} from {rec['n_train']} harmful + "
        f"{rec['n_train']} harmless train prompts (split_seed={rec['split_seed']})",
        f"- Held-out test: {rec['n_test']} harmless prompts disjoint from d_hat training",
        f"- Injection layer: L{inj}",
        "",
        "## Natural scales (per-layer, the portable Phase-2 knob)",
        "",
        f"- **Natural scale at extract L{ext}:** {s_ext['natural_scale']:.2f} "
        f"(harmful_proj_mean={s_ext['harmful_proj_mean']:.2f}, harmless_proj_mean={s_ext['harmless_proj_mean']:.2f})",
        f"- **Natural scale at inject L{inj}:** {s_inj['natural_scale']:.2f} "
        f"(harmful_proj_mean={s_inj['harmful_proj_mean']:.2f}, harmless_proj_mean={s_inj['harmless_proj_mean']:.2f})",
        f"- Random direction class-separation at L{inj}: {s_rnd['natural_scale']:.3f} "
        f"(should be ≈ 0 — random unit vector does not separate classes)",
        "",
        f"_Phase 2 will compute these per-layer scales on the target model and use "
        f"them as the coefficient unit. Absolute coeffs are not portable; "
        f"natural-scale multiples are._",
        "",
        "## Baseline",
        "",
        f"- Baseline harmless refusal (no hook): **{base['refusal_rate']:.2f}** "
        f"({base['n_refused']}/{base['n']}, Wilson 95% CI = "
        f"[{base['wilson_95_ci'][0]:.2f}, {base['wilson_95_ci'][1]:.2f}])",
        f"- Baseline harmless p50 chars: {base['p50_chars']:.0f}",
        "",
        "## Headline cells",
        "",
        "| Cell | Vector | Coeff | Coeff label | Refusal rate | Wilson 95% CI | Δ vs base | p50 chars | Coherence kept |",
        "|---|---|---:|---|---:|---|---:|---:|:---:|",
    ]
    for c in cells:
        kept = "✅" if c["coherence_kept"] else "❌"
        lines.append(
            f"| {c['name']} | {c['vector']} | {c['coeff_absolute']:.2f} | "
            f"{c['coeff_label']} | {c['refusal_rate']:.2f} | "
            f"[{c['wilson_95_ci'][0]:.2f}, {c['wilson_95_ci'][1]:.2f}] | "
            f"{'+' if c['delta_vs_baseline'] >= 0 else ''}{c['delta_vs_baseline']:.2f} | "
            f"{c['p50_chars']:.0f} | {kept} |"
        )

    real = next(c for c in cells if c["name"] == "real_at_inject_scale")
    rand = next(c for c in cells if c["name"] == "rand_at_inject_scale_matched_abs")

    lines += [
        "",
        "## The two-cell verdict (the part that matters for promotion)",
        "",
        f"- **real (d_hat at L{inj}, 1.0× natural scale at inject):** "
        f"refusal = {real['refusal_rate']:.2f} "
        f"[CI {real['wilson_95_ci'][0]:.2f}, {real['wilson_95_ci'][1]:.2f}], "
        f"p50 = {real['p50_chars']:.0f}",
        f"- **rand (r_hat at L{inj}, matched absolute coeff):** "
        f"refusal = {rand['refusal_rate']:.2f} "
        f"[CI {rand['wilson_95_ci'][0]:.2f}, {rand['wilson_95_ci'][1]:.2f}], "
        f"p50 = {rand['p50_chars']:.0f}",
        "",
        "**Promotion criterion:** real Δrefusal ≥ 0.30 with coherence kept "
        "AND random Δrefusal < 0.10. With N=50 the Wilson CIs are tight "
        "enough that this two-cell comparison carries the claim on its own — "
        "no longer a 10/10 number that could be a 7-out-of-10 chance event.",
        "",
        "## (H) Eyeball task — five-minute read",
        "",
        f"The substring scorer says {real['n_refused']}/{real['n']} 'refused' at the real "
        "cell. The substring scorer cannot distinguish:",
        "",
        "- Structural refusals (\"I can't help with that because…\") — these support the claim.",
        "- Templated artifact refusals (repetitive, off-topic, weirdly formatted) — these UNDERMINE the claim.",
        "",
        "Read all "
        f"{real['n']} completions in the real cell from "
        "`artifacts/runs/phase1_step3d/<latest>/result.json` (field: `cells[0].completions`). "
        "Decide for each: 'refusal' vs 'artifact'. Report the corrected rate.",
        "",
        f"> _(H) corrected rate goes here. (Substring rate: {real['refusal_rate']:.2f})_",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
