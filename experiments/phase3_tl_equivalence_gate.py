"""Phase 3 Task 0 — TL-equivalence gate RUNNER (thin CLI/IO over mech_security.equivalence).

All load-bearing logic — comparison metrics, residual sanity, the fixed prompt set, the spine
definition, and the pre-registered pass/fail thresholds — lives in `mech_security/equivalence.py`
(logic in the package; this file only parses args, loops checkpoints, renders the report, and
applies the spine-halt exit). See PHASE3_PLAN.md Amendment A2.

Usage (8B, via the ungated Llama-3 mirror so no Meta-license wait):
    python experiments/phase3_tl_equivalence_gate.py \
        --base NousResearch/Meta-Llama-3-8B-Instruct \
        --checkpoints LLM-LAT/robust-llama3-8b-instruct skysys00/Meta-Llama-3-8B-Instruct-DeepRefusal ...

Calibrates the tolerance on vanilla `--base` first, then gates each checkpoint. Only PASS
checkpoints proceed to scoring; a spine FAIL exits non-zero to halt for human review.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mech_security import equivalence as eq
from mech_security.model import _auto_device
from mech_security.phase3_loaders import DEFAULT_BASE

DEFAULT_CHECKPOINTS = [
    "LLM-LAT/robust-llama3-8b-instruct",
    "skysys00/Meta-Llama-3-8B-Instruct-DeepRefusal",
    "samuelsimko/Meta-Llama-3-8B-Instruct-ReFAT",
    "GraySwanAI/Llama-3-8B-Instruct-RR",
    "lapisrocks/Llama-3-8B-Instruct-TAR-Refusal",
]


def render_md(rows: list[dict], ceiling: float, vanilla: str) -> str:
    lines = [
        "# Phase 3 Task 0 — TL-equivalence gate",
        "",
        f"Vanilla calibrator: `{vanilla}` · pass ceiling (centered_max_abs_diff) = **{ceiling:.4f}** "
        "· hard gate also requires top1_agreement = 1.0 and resid sanity (Amendment A2).",
        "",
        "| checkpoint | top1_agree | centered_max | raw_max | mean_abs | resid_ok | hooks_bite | **GATE** |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lg, rs = r["logits"], r["resid"]
        g = "✅ PASS" if r["pass"] else "❌ FAIL"
        err = f" — {r['error']}" if r.get("error") else ""
        lines.append(
            f"| `{r['checkpoint']}` | {lg['top1_agreement']:.3f} | {lg['centered_max_abs_diff']:.4f} | "
            f"{lg['raw_max_abs_diff']:.4f} | {lg['mean_abs_diff']:.5f} | {rs['resid_shape_ok']} | "
            f"{rs.get('hooks_bite')} | {g}{err} |"
        )
    lines += ["", "## Coherence samples (eyeball; necessary not sufficient)", ""]
    for r in rows:
        lines.append(f"**`{r['checkpoint']}`** ({'PASS' if r['pass'] else 'FAIL'}):")
        for c in r.get("coherence_samples", []):
            lines.append(f"  - {c[:200]}")
        lines.append("")
    return "\n".join(lines)


def _fail_row(checkpoint_id: str, base: str, err: str) -> dict:
    return {
        "checkpoint": checkpoint_id, "base": base, "error": err,
        "logits": {"top1_agreement": 0.0, "centered_max_abs_diff": float("inf"),
                   "raw_max_abs_diff": float("inf"), "mean_abs_diff": float("inf"), "n_prompts": 0},
        "resid": {"resid_shape_ok": False, "hooks_bite": False, "resid_shape": [], "resid_layer": -1},
        "coherence_samples": [], "pass": False,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=DEFAULT_BASE,
                    help="TransformerLens arch key (Llama-3-8B, the ungated NousResearch mirror, or a Qwen base for dev)")
    ap.add_argument("--vanilla", default=None, help="model to calibrate tolerance on; defaults to --base")
    ap.add_argument("--checkpoints", nargs="*", default=DEFAULT_CHECKPOINTS)
    ap.add_argument("--device", default=None)
    ap.add_argument("--tol-margin", type=float, default=0.15)
    ap.add_argument("--no-processing", action="store_true",
                    help="load via from_pretrained_no_processing (HF-faithful in reduced precision); "
                         "flip this if vanilla calibration fails top1=1.0 with processing on")
    ap.add_argument("--out", default="results/phase3_tl_equivalence.md")
    args = ap.parse_args()

    device = args.device or _auto_device()
    vanilla = args.vanilla or args.base
    print(f"[gate] device={device} base={args.base} vanilla={vanilla}")

    print(f"[gate] calibrating on vanilla {vanilla} ... (no_processing={args.no_processing})")
    van = eq.evaluate_checkpoint(vanilla, args.base, device, no_processing=args.no_processing)
    ceiling = eq.calibrate_ceiling(van["logits"], args.tol_margin)
    van["pass"] = eq.gate_verdict(van, ceiling)
    if van["logits"]["top1_agreement"] != 1.0:
        print(f"[gate] !! vanilla top1_agreement={van['logits']['top1_agreement']} != 1.0 — "
              "harness/load is broken; results below are not trustworthy.")
    print(f"[gate] vanilla centered_max={van['logits']['centered_max_abs_diff']:.4f} -> ceiling={ceiling:.4f} "
          f"| top1={van['logits']['top1_agreement']:.3f} pass={van['pass']}")

    rows = [van]
    spine_failures = []
    for c in args.checkpoints:
        if c == vanilla:
            continue
        print(f"[gate] evaluating {c} ...")
        try:
            r = eq.evaluate_checkpoint(c, args.base, device, no_processing=args.no_processing)
            r["pass"] = eq.gate_verdict(r, ceiling)
        except Exception as e:  # a load that errors is a FAIL, not a crash of the whole gate
            r = _fail_row(c, args.base, f"{type(e).__name__}: {str(e)[:300]}")
        rows.append(r)
        lg = r["logits"]
        print(f"[gate] {c}: top1={lg['top1_agreement']:.3f} centered_max={lg['centered_max_abs_diff']:.4f} "
              f"-> {'PASS' if r['pass'] else 'FAIL'}" + (f" ({r.get('error')})" if r.get("error") else ""))
        if eq.is_spine(c) and not r["pass"]:
            spine_failures.append(c)

    out_md = Path(args.out)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render_md(rows, ceiling, vanilla))
    out_md.with_suffix(".json").write_text(
        json.dumps({"base": args.base, "vanilla": vanilla, "device": device, "ceiling": ceiling, "rows": rows}, indent=2)
    )
    print(f"[gate] wrote {out_md} and {out_md.with_suffix('.json')}")
    print("[gate] PASS:", [r["checkpoint"] for r in rows if r["pass"]])
    print("[gate] FAIL:", [r["checkpoint"] for r in rows if not r["pass"]])

    if spine_failures:
        print(f"[gate] !!! SPINE CHECKPOINT FAILED THE GATE: {spine_failures} — HALT and report (Amendment A1).")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
