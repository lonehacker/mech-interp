"""One-off: build a GENERAL benign instruction set (Alpaca), matched in STYLE to AdvBench imperatives,
as the strengthened-denominator harmless reference (Amendment A7). General-vs-general, not code-vs-general.

Writes data/alpaca_harmless.jsonl ({"text","label":"harmless"}), same format track1_splits._harmless reads.
Deterministic (seeded shuffle). Light keyword filter drops anything brushing harmful themes.
"""
import json
import random
import re
from pathlib import Path

from datasets import load_dataset

OUT = Path("/Users/anshulsinghle/safe_ai/mech-security/data/alpaca_harmless.jsonl")
# 300 (was 160): the n_extract=200 denominator arm needs ≥215 harmless (15 presence + 200 extract).
# Seeded shuffle + [:N] slice ⇒ the first 160 rows are byte-identical to the old file, so already-run
# experiments (which drew ≤100) remain comparable; rows 161–300 are new headroom. (User-directed 2026-06-06.)
N = 300
BLOCK = re.compile(
    r"\b(hack|bomb|weapon|gun|firearm|kill|murder|steal|theft|illegal|drug|narcotic|exploit|malware|"
    r"virus|ransom|attack|fraud|scam|poison|launder|counterfeit|explos|suicide|self-harm|harm|terror|"
    r"phish|ddos|breach|hate|racis|sexual|nsfw|porn)\b", re.I)


def main():
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    seen, cands = set(), []
    for r in ds:
        if (r.get("input") or "").strip():
            continue  # instruction-only (no input) — matches AdvBench's standalone-imperative form
        t = (r.get("instruction") or "").strip()
        if not (12 <= len(t) <= 200) or t in seen or BLOCK.search(t):
            continue
        seen.add(t)
        cands.append(t)
    random.Random(20260530).shuffle(cands)
    out = cands[:N]
    OUT.write_text("\n".join(json.dumps({"text": t, "label": "harmless"}) for t in out) + "\n")
    print(f"wrote {len(out)} harmless instructions to {OUT} (from {len(cands)} clean candidates)")
    for t in out[:4]:
        print("  -", t)


if __name__ == "__main__":
    main()
