"""One-off: build XSTest over-refusal prompt sets (replaces the hand-authored scary-word set, which
re-introduced the Phase-1 vocab confound). XSTest (Röttger et al. 2023) is the canonical exaggerated-safety
benchmark: SAFE prompts that superficially trigger refusal, with matched truly-UNSAFE counterparts.

Writes:
  data/xstest_safe.jsonl    — safe (content-harmless) prompts, the harmless-refused candidate pool
  data/xstest_unsafe.jsonl  — the matched unsafe (content-harmful) counterparts
Each line: {"text","type","label"}. The `type` is kept so the TF-IDF lexical-overlap gate (and we) can
see which are vocab-driven (homonyms/figurative_language — expected to FAIL the gate) vs intent/context-
driven (safe_contexts/definitions/historical/privacy/discr — expected to PASS).
"""
import json
from pathlib import Path

from datasets import load_dataset

OUT_SAFE = Path("/Users/anshulsinghle/safe_ai/mech-security/data/xstest_safe.jsonl")
OUT_UNSAFE = Path("/Users/anshulsinghle/safe_ai/mech-security/data/xstest_unsafe.jsonl")


def main():
    ds = load_dataset("natolambert/xstest-v2-copy")
    split = list(ds.keys())[0]
    seen_safe, seen_unsafe = {}, {}
    for r in ds[split]:
        t, p = r["type"], r["prompt"].strip()
        if t.startswith("contrast_"):
            seen_unsafe[p] = t          # dedupe by prompt
        else:
            seen_safe[p] = t
    OUT_SAFE.write_text("\n".join(json.dumps({"text": p, "type": t, "label": "harmless"})
                                  for p, t in seen_safe.items()) + "\n")
    OUT_UNSAFE.write_text("\n".join(json.dumps({"text": p, "type": t, "label": "harmful"})
                                    for p, t in seen_unsafe.items()) + "\n")
    from collections import Counter
    print(f"safe={len(seen_safe)} -> {OUT_SAFE.name} | type dist:", dict(Counter(seen_safe.values())))
    print(f"unsafe={len(seen_unsafe)} -> {OUT_UNSAFE.name}")


if __name__ == "__main__":
    main()
