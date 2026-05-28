"""
Phase 1 — Step 1 (H): build the frozen contrastive set.

Source:
- harmful:  AdvBench `harmful_behaviors.csv` (520 prompts, GitHub raw)
- harmless: tatsu-lab/alpaca (52k examples, HuggingFace datasets)

Goal: a length- and stem-stratified harmless sample that controls the
syntactic + length confounds the planner flagged. The remaining residual
confounds (sentiment, topic) are surfaced by the audit step for the human
to interpret.

Per CLAUDE.md this is an (H) step — the human explicitly authorized this
build. The script is reproducible (fixed seed, all sampling deterministic);
the writer of data/contrastive.jsonl is the human's intent expressed through
this script, not an autonomous data edit.

Run:
    cd ~/safe_ai/mech-security
    python -m experiments.phase1_step1_build_dataset

Output:
    data/contrastive.jsonl         frozen set, jsonl of {text, label, source, stem}
    results/contrastive_audit.md   audit + paragraph TEMPLATE for human to fill
"""

from __future__ import annotations

import csv
import io
import json
import re
import sys
import urllib.request
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments._runner import DATA, RESULTS, get_logger, new_run_dir, write_json
from src.controls import audit_contrastive, audit_to_markdown

log = get_logger("phase1_step1")

ADVBENCH_URL = (
    "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/"
    "data/advbench/harmful_behaviors.csv"
)

# Target sizes
N_PER_SPLIT = 150
SEED = 0

# Stem extraction: the first content word of the prompt, normalized.
_STEM_RE = re.compile(r"^\s*([A-Za-z']+)")


def _stem(text: str) -> str:
    m = _STEM_RE.match(text)
    return m.group(1).lower() if m else "<unknown>"


def fetch_advbench() -> list[str]:
    """Download AdvBench harmful behaviors. Returns the `goal` column."""
    log.info("downloading AdvBench from %s", ADVBENCH_URL)
    with urllib.request.urlopen(ADVBENCH_URL, timeout=30) as resp:
        text = resp.read().decode("utf-8")
    reader = csv.DictReader(io.StringIO(text))
    goals = [row["goal"].strip() for row in reader if row.get("goal", "").strip()]
    log.info("AdvBench: %d harmful behaviors loaded", len(goals))
    return goals


def fetch_alpaca() -> list[str]:
    """Load Alpaca instructions. Filters:
    - no input field (pure-instruction asks only, matches AdvBench shape)
    - drop empty / unusually short / non-imperative records
    """
    from datasets import load_dataset
    log.info("loading tatsu-lab/alpaca from HuggingFace...")
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    log.info("alpaca raw: %d records", len(ds))

    candidates: list[str] = []
    for rec in ds:
        instr = (rec.get("instruction") or "").strip()
        inp = (rec.get("input") or "").strip()
        if inp:
            continue  # skip records that depend on an input — not pure instructions
        if not instr or len(instr) < 10:
            continue
        # Strip newlines / collapse whitespace
        instr = re.sub(r"\s+", " ", instr)
        candidates.append(instr)
    log.info("alpaca after filter (no input, len ≥ 10): %d records", len(candidates))
    return candidates


def stem_stratified_length_matched_sample(
    harmless_pool: list[str],
    harmful: list[str],
    n_target: int,
    seed: int,
) -> list[str]:
    """Sample `n_target` harmless prompts whose joint (stem, char-length-bucket)
    distribution matches the harmful split as closely as the pool allows.

    Strategy:
    1. Compute (stem, length-bucket) counts on harmful.
    2. For each cell with k samples in harmful, draw k samples from the harmless
       pool restricted to the same cell. If the cell is under-supplied in the
       pool, draw what's available and account for the deficit at the end.
    3. Fill any deficit with closest-stem, closest-length-bucket harmless
       prompts so we still hit n_target.

    Length-bucket width is chosen so harmful spreads across ~6 buckets.
    """
    import numpy as np
    rng = np.random.default_rng(seed)

    def char_bucket(s: str, width: int = 25) -> int:
        return len(s) // width

    # Build (stem, bucket) -> count on harmful
    harmful_cells: Counter = Counter((_stem(s), char_bucket(s)) for s in harmful)

    # Index harmless pool by cell
    pool_by_cell: dict[tuple[str, int], list[str]] = {}
    for s in harmless_pool:
        cell = (_stem(s), char_bucket(s))
        pool_by_cell.setdefault(cell, []).append(s)

    chosen: list[str] = []
    deficit = 0

    for cell, need in harmful_cells.items():
        # Scale down to n_target — we want the OUTPUT distribution to match
        # harmful's, not duplicate its exact counts (harmful has more cells
        # than the pool can usually fill).
        proportional_need = max(1, round(need * (n_target / len(harmful))))
        available = pool_by_cell.get(cell, [])
        if not available:
            deficit += proportional_need
            continue
        take = min(proportional_need, len(available))
        idx = rng.choice(len(available), size=take, replace=False)
        chosen.extend(available[i] for i in idx)
        if take < proportional_need:
            deficit += proportional_need - take

    # Fill deficit with closest-stem matches (same stem, any bucket).
    log.info("after stratified pass: %d chosen, %d deficit", len(chosen), deficit)
    if deficit > 0 or len(chosen) < n_target:
        remaining = n_target - len(chosen)
        already = set(chosen)
        pool_remaining = [s for s in harmless_pool if s not in already]
        # Sort by similarity to harmful char-length distribution
        harmful_median = sorted(len(s) for s in harmful)[len(harmful) // 2]
        pool_remaining.sort(key=lambda s: abs(len(s) - harmful_median))
        chosen.extend(pool_remaining[:remaining])

    # Trim if over-target
    if len(chosen) > n_target:
        # Random downsample for fairness
        idx = rng.choice(len(chosen), size=n_target, replace=False)
        chosen = [chosen[i] for i in idx]

    return chosen


def downsample_harmful(harmful: list[str], n_target: int, seed: int) -> list[str]:
    import numpy as np
    rng = np.random.default_rng(seed)
    if len(harmful) <= n_target:
        return list(harmful)
    idx = rng.choice(len(harmful), size=n_target, replace=False)
    return [harmful[i] for i in idx]


def main() -> int:
    run_dir = new_run_dir("phase1_step1")
    log.info("run_dir: %s", run_dir)

    harmful_all = fetch_advbench()
    harmless_pool = fetch_alpaca()

    harmful = downsample_harmful(harmful_all, N_PER_SPLIT, SEED)
    log.info("harmful sample: %d", len(harmful))

    harmless = stem_stratified_length_matched_sample(
        harmless_pool, harmful, N_PER_SPLIT, SEED
    )
    log.info("harmless sample: %d", len(harmless))

    # Write the frozen set.
    out_path = DATA / "contrastive.jsonl"
    records = []
    for text in harmful:
        records.append({"text": text, "label": "harmful", "source": "advbench", "stem": _stem(text)})
    for text in harmless:
        records.append({"text": text, "label": "harmless", "source": "alpaca", "stem": _stem(text)})

    with out_path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    log.info("frozen set -> %s (%d records)", out_path, len(records))

    # Provenance manifest
    manifest = {
        "step": "phase1_step1",
        "seed": SEED,
        "n_per_split": N_PER_SPLIT,
        "harmful_source": "https://github.com/llm-attacks/llm-attacks/blob/main/data/advbench/harmful_behaviors.csv",
        "harmless_source": "huggingface://tatsu-lab/alpaca",
        "harmful_pool_size": len(harmful_all),
        "harmless_pool_filter": "no input field, len ≥ 10",
        "stratification": "stem (first word, lowercased) × char-length bucket (width 25)",
        "n_harmful": len(harmful),
        "n_harmless": len(harmless),
        "output": str(out_path),
    }
    write_json(run_dir / "manifest.json", manifest)
    log.info("manifest -> %s", run_dir / "manifest.json")

    # Run the audit
    log.info("auditing the frozen set...")
    audit = audit_contrastive(out_path, bundle=None)  # tokenizer-accurate audit happens in Phase 0 re-run
    audit_md = audit_to_markdown(audit)

    audit_path = RESULTS / "contrastive_audit.md"
    audit_path.write_text(_render_audit(audit_md, manifest, harmful, harmless))
    log.info("audit -> %s", audit_path)

    print(f"\nphase1_step1 | wrote {len(records)} records to {out_path}")
    print(f"phase1_step1 | audit at {audit_path}")
    print("phase1_step1 | READ THE AUDIT before running Phase 0 / Step 2 on this data.")
    return 0


def _render_audit(audit_md: str, manifest: dict, harmful: list[str], harmless: list[str]) -> str:
    """Wrap the raw audit markdown with manifest + space for the human's
    residual-confounds paragraph."""
    h_stems = Counter(_stem(s) for s in harmful).most_common(10)
    l_stems = Counter(_stem(s) for s in harmless).most_common(10)
    h_stem_str = ", ".join(f"{w}({c})" for w, c in h_stems)
    l_stem_str = ", ".join(f"{w}({c})" for w, c in l_stems)

    lines = [
        "# Contrastive set audit — `data/contrastive.jsonl`",
        "",
        "## Provenance",
        f"- Seed: {manifest['seed']}",
        f"- n per split: {manifest['n_per_split']}",
        f"- Harmful source: AdvBench harmful_behaviors.csv ({manifest['harmful_pool_size']} pool size)",
        f"- Harmless source: tatsu-lab/alpaca (filtered: {manifest['harmless_pool_filter']})",
        f"- Stratification: {manifest['stratification']}",
        "",
        "## Stem (first-word) distribution",
        f"- harmful  top-10: {h_stem_str}",
        f"- harmless top-10: {l_stem_str}",
        "",
        audit_md,
        "",
        "## (H) — Required: residual-confounds paragraph",
        "",
        "> Write 3–5 sentences here documenting what confounds you have knowingly",
        "> accepted and why. Examples of things to address:",
        ">",
        "> - Are the top n-grams across splits dominated by different content words?",
        ">   (Almost certainly yes — that's the topic confound. Acceptable iff the",
        ">   refusal-direction analysis is interpreted as 'direction in the model's",
        ">   representation space that tracks harmful-vs-harmless content', not",
        ">   'pure harmful-intent direction'.)",
        "> - Is the sentiment proxy hit count systematically different? If yes,",
        ">   any refusal direction may partly be a sentiment direction.",
        "> - Are length quantiles aligned? If harmful is systematically longer,",
        ">   the direction will partly be a length direction.",
        "> - Are stems aligned? If 'write' dominates one split but 'explain' the",
        ">   other, the direction will partly be an instruction-syntax direction.",
        ">",
        "> Replace this block with your paragraph. The paragraph is a research",
        "> artifact: it will appear verbatim in the Phase 3 write-up's limitations",
        "> section.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
