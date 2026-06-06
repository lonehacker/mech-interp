"""Phase 3 Track 1 — frozen, mutually-disjoint prompt splits (load-bearing; package).

Three sets that MUST be mutually disjoint, or the attack result is compromised:
  - extraction  : prompts used to extract d̂ (diff-of-means harmful_train vs harmless_train)
  - scoring     : held-out prompts the attack is evaluated on
  - presence    : the Task-0.5 defense-presence prompts (already consumed)

If extraction ∩ scoring is non-empty you get an optimistic *in-sample* ablation result — the first
thing a reviewer probes. So `track1_splits` builds the splits from one seeded shuffle and ASSERTS
disjointness (set-level) + records sha256 hashes per set (frozen-set provenance, like the Phase-1
contrastive set). REPORTS data only; no interpretation. Unit-tested in tests/test_track1_splits.py.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random


def _advbench_goals(path: str) -> list[str]:
    return [r["goal"].strip() for r in csv.DictReader(open(path)) if r.get("goal")]


def _harmless(path: str) -> list[str]:
    out = []
    for line in open(path):
        line = line.strip()
        if line and json.loads(line).get("label") == "harmless":
            out.append(json.loads(line)["text"].strip())
    return out


def set_hash(prompts: list[str]) -> str:
    """sha256 over the sorted, newline-joined prompts — order-independent freeze id."""
    return hashlib.sha256("\n".join(sorted(prompts)).encode()).hexdigest()[:16]


def assert_disjoint(named: dict[str, list[str]]) -> None:
    """Raise loudly if any two named sets share a prompt (the load-bearing guarantee)."""
    items = list(named.items())
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            (na, a), (nb, b) = items[i], items[j]
            overlap = set(a) & set(b)
            if overlap:
                raise AssertionError(
                    f"Track-1 split overlap: {na} ∩ {nb} = {len(overlap)} prompts "
                    f"(e.g. {next(iter(overlap))[:70]!r}). Extraction/scoring/presence must be disjoint."
                )


def _count_label(path: str, label: str) -> int:
    return sum(1 for line in open(path) if line.strip() and json.loads(line).get("label") == label)


def required_rows(n_extract: int, n_score: int, *, matched: bool = False,
                  n_presence: int = 30, n_harmless_presence: int = 15) -> dict:
    """Minimum harmful/harmless row counts a split of this size needs — MIRRORS the slicing in
    `track1_splits`/`matched_splits`. Kept in lockstep (tests/test_track1_splits.py locks them) so the
    orchestrator pre-flight check can never silently drift from the real ValueError-raising asserts."""
    if matched:
        return {"harmful": n_extract + n_score, "harmless": n_extract}
    return {"harmful": n_presence + n_extract + n_score, "harmless": n_harmless_presence + n_extract}


def feasibility(*, n_extract: int, n_score: int, advbench_path: str | None = None,
                harmless_path: str | None = None, matched_path: str | None = None) -> tuple[bool, str]:
    """(ok, reason): can a split of this size be built from these files? The orchestrator pre-flight
    calls this BEFORE provisioning, so an infeasible run fails locally in seconds — not after a paid
    pod boots (the n200 lesson: alpaca_harmless had 160 rows, the run needed 215). Pass `matched_path`
    for the matched set; otherwise advbench (harmful) + harmless paths."""
    if matched_path:
        have = {"harmful": _count_label(matched_path, "harmful"),
                "harmless": _count_label(matched_path, "harmless")}
        need = required_rows(n_extract, n_score, matched=True)
    else:
        have = {"harmful": len(_advbench_goals(advbench_path)), "harmless": len(_harmless(harmless_path))}
        need = required_rows(n_extract, n_score, matched=False)
    short = {key: (need[key], have[key]) for key in need if have[key] < need[key]}
    if short:
        return False, "insufficient data — " + ", ".join(
            f"{key}: need {nd} have {hv}" for key, (nd, hv) in short.items())
    return True, f"ok — harmful {have['harmful']}≥{need['harmful']}, harmless {have['harmless']}≥{need['harmless']}"


def track1_splits(
    advbench_path: str,
    harmless_path: str,
    *,
    seed: int = 20260530,            # SAME seed as Task 0.5 -> [:30] is exactly the presence set
    n_presence: int = 30,
    n_extract: int = 40,
    n_score: int = 32,
    n_harmless_presence: int = 15,   # SAME as Task 0.5 -> harmless[:15] is the presence harmless
    n_harmless_extract: int = 40,
) -> dict:
    """Frozen, mutually-disjoint extraction / scoring / presence splits + hash provenance.

    Harmful: one seeded AdvBench shuffle, sliced presence | extract | score (disjoint by construction,
    re-asserted). Harmless: contrastive.jsonl, presence[:15] then a disjoint extraction slice.
    """
    goals = _advbench_goals(advbench_path)
    rng = random.Random(seed)
    rng.shuffle(goals)
    presence = goals[:n_presence]
    extract = goals[n_presence:n_presence + n_extract]
    score = goals[n_presence + n_extract:n_presence + n_extract + n_score]
    if len(score) < n_score:
        raise ValueError(f"AdvBench has {len(goals)} goals; not enough for {n_presence}+{n_extract}+{n_score}")

    harmless = _harmless(harmless_path)
    h_presence = harmless[:n_harmless_presence]
    h_extract = harmless[n_harmless_presence:n_harmless_presence + n_harmless_extract]
    if len(h_extract) < n_harmless_extract:
        raise ValueError(f"only {len(harmless)} harmless; need {n_harmless_presence}+{n_harmless_extract}")

    assert_disjoint({"presence_harmful": presence, "extract_harmful": extract, "score_harmful": score})
    assert_disjoint({"presence_harmless": h_presence, "extract_harmless": h_extract})

    return {
        "extract_harmful": extract, "extract_harmless": h_extract, "score_harmful": score,
        "presence_harmful": presence,
        "seed": seed,
        "hashes": {
            "extract_harmful": set_hash(extract), "extract_harmless": set_hash(h_extract),
            "score_harmful": set_hash(score), "presence_harmful": set_hash(presence),
        },
        "sizes": {"extract_harmful": len(extract), "extract_harmless": len(h_extract),
                  "score_harmful": len(score), "presence_harmful": len(presence)},
    }


def matched_splits(
    matched_path: str,
    *,
    seed: int = 20260530,
    n_extract_harmful: int = 30,
    n_score_harmful: int = 10,
    n_harmless_extract: int = 30,
) -> dict:
    """Splits for the Phase-2 *matched* contrastive set (one jsonl with both labels) — the
    apples-to-apples code-matched comparison under the CURRENT harness (Amendment A9 read). Harmful is
    seed-shuffled then sliced into extract | held-out score; harmless → extract. No presence set
    (Track-1 presence was AdvBench-based). Same hash/disjointness provenance as `track1_splits`."""
    def _texts(label: str) -> list[str]:
        return [json.loads(line)["text"].strip() for line in open(matched_path)
                if line.strip() and json.loads(line).get("label") == label]

    harmful, harmless = _texts("harmful"), _texts("harmless")
    rng = random.Random(seed)
    rng.shuffle(harmful)
    extract = harmful[:n_extract_harmful]
    score = harmful[n_extract_harmful:n_extract_harmful + n_score_harmful]
    h_extract = harmless[:n_harmless_extract]
    if len(score) < n_score_harmful or len(h_extract) < n_harmless_extract:
        raise ValueError(f"matched set too small ({len(harmful)} harmful / {len(harmless)} harmless) for "
                         f"{n_extract_harmful}+{n_score_harmful} harmful + {n_harmless_extract} harmless")
    assert_disjoint({"extract_harmful": extract, "score_harmful": score})
    return {
        "extract_harmful": extract, "extract_harmless": h_extract, "score_harmful": score,
        "presence_harmful": [], "seed": seed,
        "hashes": {"extract_harmful": set_hash(extract), "extract_harmless": set_hash(h_extract),
                   "score_harmful": set_hash(score), "presence_harmful": ""},
        "sizes": {"extract_harmful": len(extract), "extract_harmless": len(h_extract),
                  "score_harmful": len(score), "presence_harmful": 0},
    }
