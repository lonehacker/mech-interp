"""
Phase 1 — TinyMMLU capability check under d_hat ablation.

The point: the ablation must be SPECIFIC to refusal. If it also drops MMLU
accuracy materially, the result is "we lobotomized the model and it stopped
refusing as a side-effect" rather than "we removed the refusal mechanism
cleanly." This is the C2 control extended from coherence-on-harmless to
real capability measured on a standard benchmark.

Loads `tinyBenchmarks/tinyMMLU` (100 questions, statistically representative
of full MMLU within ~2pp). Two conditions:
  baseline   — no hook
  ablated    — Arditi multi-layer ablation of d_hat at L13

Scoring: parse first A/B/C/D character of the generation. If neither, score
as "no_answer" (separate from incorrect).

Headline numbers: baseline accuracy, ablated accuracy, delta. If |delta| < 5pp,
ablation is specific. If ablated < baseline by ≥10pp, ablation degraded general
capability — the refusal-rate claim has a confound.

Run:
    cd ~/safe_ai/mech-security
    python -m experiments.phase1_tinymmlu_capability
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

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
from src.directions import ablate_dir, diff_of_means, unit
from src.model import generate

log = get_logger("phase1_tinymmlu")


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0: return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def load_tiny_mmlu(n: int | None = None, seed: int = 0) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("tinyBenchmarks/tinyMMLU", split="test")
    rows = []
    for r in ds:
        question = r["question"]
        choices = r["choices"]  # list of 4 strings
        answer = r["answer"]    # int 0-3
        if len(choices) != 4:
            continue
        rows.append({
            "question": question,
            "choices": choices,
            "answer_idx": int(answer),
            "answer_letter": "ABCD"[int(answer)],
            "subject": r.get("subject", "unknown"),
        })
    if n is not None and n < len(rows):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(rows), size=n, replace=False)
        rows = [rows[i] for i in sorted(idx)]
    return rows


def format_mcq(row: dict) -> str:
    """Format an MMLU question as a multiple-choice prompt. Style matches
    common MMLU few-shot conventions; we use zero-shot."""
    q = row["question"]
    c = row["choices"]
    return (
        f"Answer the following multiple choice question. Output ONLY the "
        f"letter (A, B, C, or D) of the correct answer.\n\n"
        f"Question: {q}\n\n"
        f"A. {c[0]}\nB. {c[1]}\nC. {c[2]}\nD. {c[3]}\n\n"
        f"Answer:"
    )


def parse_answer(completion: str) -> str | None:
    """Extract the first A/B/C/D from the completion. Returns the letter or None."""
    if not completion:
        return None
    m = re.search(r"[A-D]", completion.strip())
    if m:
        return m.group(0)
    return None


def score_run(rows: list[dict], completions: list[str]) -> dict:
    correct = 0
    no_answer = 0
    per_subject = {}
    for row, gen in zip(rows, completions):
        pred = parse_answer(gen)
        truth = row["answer_letter"]
        subj = row["subject"]
        per_subject.setdefault(subj, {"n": 0, "correct": 0, "no_answer": 0})
        per_subject[subj]["n"] += 1
        if pred is None:
            no_answer += 1
            per_subject[subj]["no_answer"] += 1
        elif pred == truth:
            correct += 1
            per_subject[subj]["correct"] += 1
    n = len(rows)
    return {
        "n": n,
        "correct": correct,
        "no_answer": no_answer,
        "accuracy": correct / max(1, n),
        "wilson_95_ci": list(wilson_ci(correct, n)),
        "no_answer_rate": no_answer / max(1, n),
        "per_subject": per_subject,
    }


def _gen_mcq(bundle, prompts, max_new_tokens=10):
    """Generate short responses for MCQ scoring."""
    return [generate(bundle, p, max_new_tokens=max_new_tokens, temperature=0.0)
            for p in prompts]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None,
                    help="Subsample to this many questions (default: all 100).")
    ap.add_argument("--extract-layer", type=int, default=13)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run_dir = new_run_dir("phase1_tinymmlu")
    log.info("run_dir: %s | extract=L%d", run_dir, args.extract_layer)

    bundle = get_model()
    log.info("model: %s | device=%s", bundle.name, bundle.device)

    # === d_hat from AdvBench ===
    pairs_path = Path(__file__).resolve().parent.parent / "data/contrastive.jsonl"
    harmful, harmless = load_jsonl_pairs(pairs_path)
    extra = (f"{bundle.name}|dtype={bundle.model.cfg.dtype}|L{args.extract_layer}|"
             f"resid_post|last_token|advbench_full")
    key_h = content_hash(harmful, extra=extra + "|harmful")
    key_l = content_hash(harmless, extra=extra + "|harmless")
    log.info("extracting d_hat at L%d from AdvBench ...", args.extract_layer)
    H = cached_activations(key_h, lambda: cache_resid(bundle, harmful, layer=args.extract_layer, show_progress=False))
    L = cached_activations(key_l, lambda: cache_resid(bundle, harmless, layer=args.extract_layer, show_progress=False))
    d_hat = unit(diff_of_means(H, L))

    log.info("loading TinyMMLU ...")
    rows = load_tiny_mmlu(n=args.n, seed=args.seed)
    log.info("TinyMMLU rows: %d", len(rows))
    prompts = [format_mcq(r) for r in rows]

    log.info("[1/2] baseline generations ...")
    base_gens = _gen_mcq(bundle, prompts)

    log.info("[2/2] ablated generations ...")
    with ablate_dir(bundle.model, d_hat):
        abl_gens = _gen_mcq(bundle, prompts)

    base_score = score_run(rows, base_gens)
    abl_score = score_run(rows, abl_gens)

    delta = abl_score["accuracy"] - base_score["accuracy"]
    record = {
        "step": "phase1_tinymmlu_capability",
        "model": bundle.name,
        "extract_layer": args.extract_layer,
        "n_questions": len(rows),
        "baseline": base_score,
        "ablated": abl_score,
        "accuracy_delta": delta,
        "specificity_pass": abs(delta) < 0.05,
        "per_question": [
            {"question": r["question"][:120], "subject": r["subject"],
             "truth": r["answer_letter"],
             "baseline_pred": parse_answer(bg),
             "ablated_pred": parse_answer(ag),
             "baseline_correct": parse_answer(bg) == r["answer_letter"],
             "ablated_correct": parse_answer(ag) == r["answer_letter"]}
            for r, bg, ag in zip(rows, base_gens, abl_gens)
        ],
    }
    write_json(run_dir / "result.json", record)
    log.info("run record -> %s", run_dir / "result.json")

    md_path = RESULTS / "phase1_tinymmlu_capability.md"
    md_path.write_text(_render_summary(record))
    log.info("summary -> %s", md_path)

    print(f"\nphase1_tinymmlu | n={len(rows)}")
    print(f"  baseline accuracy: {base_score['accuracy']:.3f} "
          f"[{base_score['wilson_95_ci'][0]:.2f}, {base_score['wilson_95_ci'][1]:.2f}] "
          f"({base_score['correct']}/{base_score['n']}, no_answer={base_score['no_answer']})")
    print(f"  ablated accuracy:  {abl_score['accuracy']:.3f} "
          f"[{abl_score['wilson_95_ci'][0]:.2f}, {abl_score['wilson_95_ci'][1]:.2f}] "
          f"({abl_score['correct']}/{abl_score['n']}, no_answer={abl_score['no_answer']})")
    print(f"  Δ accuracy:        {delta:+.3f}")
    print(f"  specificity (|Δ| < 0.05): {'PASS' if record['specificity_pass'] else 'FAIL'}")
    return 0


def _render_summary(rec):
    base = rec["baseline"]; abl = rec["ablated"]; delta = rec["accuracy_delta"]
    md = [
        "# Phase 1 — TinyMMLU capability check under d_hat ablation",
        "",
        "**Question:** is the d_hat ablation specific to refusal, or does it "
        "also degrade general capability?",
        "",
        f"- Model: `{rec['model']}`",
        f"- d_hat from AdvBench at L{rec['extract_layer']}",
        f"- TinyMMLU: {rec['n_questions']} questions (statistically representative of full MMLU within ~2pp)",
        "",
        "## Results",
        "",
        "| condition | accuracy | Wilson 95% CI | correct / n | no_answer |",
        "|---|---:|---|---|---:|",
        f"| baseline | {base['accuracy']:.3f} | [{base['wilson_95_ci'][0]:.2f}, {base['wilson_95_ci'][1]:.2f}] | "
        f"{base['correct']}/{base['n']} | {base['no_answer']} |",
        f"| ablated  | {abl['accuracy']:.3f} | [{abl['wilson_95_ci'][0]:.2f}, {abl['wilson_95_ci'][1]:.2f}] | "
        f"{abl['correct']}/{abl['n']} | {abl['no_answer']} |",
        "",
        f"**Δ accuracy = {delta:+.3f}** "
        f"({'specific to refusal' if abs(delta) < 0.05 else 'POSSIBLE GENERAL CAPABILITY DEGRADATION'})",
        "",
        "## Interpretation",
        "",
        "- |Δ| < 0.05: ablation is specific to refusal — supports the causal claim.",
        "- Δ < −0.10: ablation degraded general capability — refusal-rate claim has a confound.",
        f"- This run: |Δ| = {abs(delta):.3f}.",
    ]
    return "\n".join(md) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
