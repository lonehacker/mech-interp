"""Offline re-judge of a saved phase1_affect_test result.json using the
current src.eval_llm.JUDGE_SYSTEM. No GPU required — only uses the
completions already on disk.

Run:
    python -m experiments._rejudge \
        --run artifacts/runs/phase1_affect_test/20260525-160435
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments._runner import RESULTS, get_logger, new_run_dir, write_json
from src.eval_llm import judge_many

log = get_logger("rejudge")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=str)
    args = ap.parse_args()

    run_path = Path(args.run)
    if not run_path.is_absolute():
        run_path = Path(__file__).resolve().parent.parent / run_path
    rec = json.load((run_path / "result.json").open())
    samples = rec["samples"]
    log.info("loaded %d samples from %s (original scorer=%s)",
             len(samples), run_path, rec.get("scorer"))

    pairs = [(s["text"], s["completion"]) for s in samples]
    log.info("re-judging %d (prompt, completion) pairs with current JUDGE_SYSTEM...", len(pairs))
    verdicts = judge_many(pairs, show_progress=False)

    out_dir = new_run_dir("phase1_affect_test_rejudge")
    enriched = []
    for s, v in zip(samples, verdicts):
        new_label = v.label
        old_label = s.get("llm_judge_label")
        flipped = old_label is not None and old_label != new_label
        enriched.append({
            **s,
            "llm_judge_label_new": new_label,
            "llm_judge_raw_new": v.raw,
            "llm_judge_label_old": old_label,
            "label_flipped": flipped,
        })

    by_cat: dict[str, dict] = {}
    for s in enriched:
        c = s["category"]
        d = by_cat.setdefault(c, {"n": 0, "refused": 0, "partial": 0, "complied": 0,
                                  "error": 0, "old_refused": 0, "flipped": 0,
                                  "proj_high": 0, "agree_new": 0})
        d["n"] += 1
        if s["llm_judge_label_new"] == "REFUSED":
            d["refused"] += 1
            if s["predicted_harmful_by_dhat"]:
                d["agree_new"] += 1
        elif s["llm_judge_label_new"] == "PARTIAL_COMPLY":
            d["partial"] += 1
        elif s["llm_judge_label_new"] == "COMPLIED":
            d["complied"] += 1
            if not s["predicted_harmful_by_dhat"]:
                d["agree_new"] += 1
        else:
            d["error"] += 1
        if s["llm_judge_label_old"] == "REFUSED":
            d["old_refused"] += 1
        if s["label_flipped"]:
            d["flipped"] += 1
        if s["predicted_harmful_by_dhat"]:
            d["proj_high"] += 1

    for c, d in by_cat.items():
        d["refused_rate"] = d["refused"] / max(1, d["n"])
        d["partial_rate"] = d["partial"] / max(1, d["n"])
        d["complied_rate"] = d["complied"] / max(1, d["n"])
        d["old_refused_rate"] = d["old_refused"] / max(1, d["n"])
        d["agreement_rate_new"] = d["agree_new"] / max(1, d["n"])
        d["proj_high_rate"] = d["proj_high"] / max(1, d["n"])

    out_record = {
        "step": "phase1_affect_test_rejudge",
        "source_run": str(run_path),
        "n_samples": len(samples),
        "n_flipped_from_old_judge": sum(1 for s in enriched if s["label_flipped"]),
        "by_category": by_cat,
        "samples": enriched,
    }
    write_json(out_dir / "result.json", out_record)
    summary_path = RESULTS / "phase1_affect_test_rejudge.md"
    summary_path.write_text(_render(out_record))
    log.info("rejudge -> %s", summary_path)

    print(f"\nrejudge | {out_record['n_flipped_from_old_judge']}/{len(samples)} labels flipped from old judge")
    for c, d in by_cat.items():
        print(f"  {c}: refuse={d['refused_rate']:.2f} (was {d['old_refused_rate']:.2f}) | "
              f"partial={d['partial_rate']:.2f} | comply={d['complied_rate']:.2f} | "
              f"agree={d['agreement_rate_new']:.2f}")
    return 0


def _render(rec):
    lines = [
        "# Phase 1 affect-test rejudge (corrected LLM judge prompt)",
        "",
        f"Source run: `{rec['source_run']}`",
        "",
        f"**{rec['n_flipped_from_old_judge']} of {rec['n_samples']} labels flipped from the old (broken) judge.**",
        "",
        "## By category",
        "",
        "| category | n | refuse (new) | refuse (old) | partial | comply | d_hat proj_high | agreement |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for c, d in rec["by_category"].items():
        lines.append(
            f"| {c} | {d['n']} | {d['refused_rate']:.2f} | {d['old_refused_rate']:.2f} | "
            f"{d['partial_rate']:.2f} | {d['complied_rate']:.2f} | {d['proj_high_rate']:.2f} | "
            f"**{d['agreement_rate_new']:.2f}** |"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- *new* = label under the corrected JUDGE_SYSTEM in `src/eval_llm.py` "
                 "(response-shaped: does the response refuse / hedge / comply, regardless of "
                 "whether the request was harmful or benign).")
    lines.append("- *old* = label from the previous run with the misspecified judge that "
                 "asked about 'compliance with harmful intent', which defaulted to REFUSED "
                 "on genuinely benign requests.")
    lines.append("- Agreement here = the new-judge call (refused vs complied) matches d_hat's "
                 "prediction (proj_high vs proj_low at L13 midpoint).")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
