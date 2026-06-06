"""Phase 3 Task 0.5 — defense-presence RUNNER (thin; logic in mech_security.defense_presence).

NO attack: baseline-generate on a held-out harmful+harmless set for vanilla Llama-3-8B and each
spine checkpoint, dual-judge, and apply the pre-registered presence verdict. Confirms the claimed
defense is actually IN the weights before Track-1 scoring builds on it (esp. DeepRefusal, a 3rd-party
checkpoint). Halt after the 3-way table.

Held-out set: a seeded AdvBench `goal` slice (kept distinct from the Track-1 d̂-extraction split) +
contrastive.jsonl harmless. Needs the Haiku dual-judge → ANTHROPIC_API_KEY on the box.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import random
from pathlib import Path

import torch

from mech_security import defense_presence as dp
from mech_security.model import _auto_device
from mech_security.phase3_loaders import DEFAULT_BASE, load_defended_model
from mech_security.redteam import _default_judge

SPINE_DEFAULT = [
    "LLM-LAT/robust-llama3-8b-instruct",
    "skysys00/Meta-Llama-3-8B-Instruct-DeepRefusal",
]


def load_harmful(path: str, n: int, seed: int) -> list[str]:
    rows = [r["goal"].strip() for r in csv.DictReader(open(path)) if r.get("goal")]
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:n]


def load_harmless(path: str, n: int) -> list[str]:
    out = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("label") == "harmless":
            out.append(r["text"].strip())
    return out[:n]


def measure(ckpt: str, base: str, device: str, harmful, harmless, *, no_processing, max_new_tokens) -> dict:
    b = load_defended_model(ckpt, base=base, device=device, no_processing=no_processing)
    hf = dp.baseline_refusal(b, harmful, max_new_tokens=max_new_tokens, judge_fn=_default_judge)
    hl = dp.baseline_refusal(b, harmless, max_new_tokens=max_new_tokens, judge_fn=_default_judge)
    del b
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return {
        "harmful_refusal": hf["S"], "harmful_judge": hf["judge"], "harmful_mean_chars": hf["mean_chars"],
        "harmless_complied": dp.harmless_complied_rate(hl["judge"]), "harmless_judge": hl["judge"],
    }


def render_md(rows: dict, verdicts: dict, vanilla_key: str) -> str:
    van = rows[vanilla_key]
    lines = [
        "# Phase 3 Task 0.5 — defense-presence (baseline, no attack)",
        "",
        f"Held-out harmful (AdvBench slice) + harmless. Dual-judge refusal score `S=(refused+0.5·partial)/n`. "
        f"Pre-registered: spine harmful-refusal uplift over vanilla ≥ **{dp.MIN_UPLIFT}** AND harmless "
        f"compliance ≥ **{dp.MIN_HARMLESS_COMPLIED}**. Reports numbers; human calls FAIL/INCONCLUSIVE.",
        "",
        "| model | harmful refusal | uplift vs vanilla | harmless complied | verdict |",
        "|---|---|---|---|---|",
        f"| `{vanilla_key}` (vanilla) | {van['harmful_refusal']:.3f} | — | {van['harmless_complied']:.3f} | reference |",
    ]
    for k, v in verdicts.items():
        r = rows[k]
        lines.append(
            f"| `{k}` | {r['harmful_refusal']:.3f} | {v['uplift']:+.3f} | {r['harmless_complied']:.3f} | **{v['label']}** |"
        )
    lines += ["", "## Verdict notes", ""]
    for k, v in verdicts.items():
        lines.append(f"- **`{k}`** → {v['label']}: {v['note']}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--vanilla", default="NousResearch/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--spine", nargs="+", default=SPINE_DEFAULT)
    ap.add_argument("--harmful", default="data/advbench_harmful_behaviors.csv")
    ap.add_argument("--harmless", default="data/contrastive.jsonl")
    ap.add_argument("--n-harmful", type=int, default=30)
    ap.add_argument("--n-harmless", type=int, default=15)
    ap.add_argument("--seed", type=int, default=20260530)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-processing", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--out", default="results/phase3_defense_presence.md")
    args = ap.parse_args()

    device = args.device or _auto_device()
    harmful = load_harmful(args.harmful, args.n_harmful, args.seed)
    harmless = load_harmless(args.harmless, args.n_harmless)
    print(f"[t0.5] device={device} harmful={len(harmful)} harmless={len(harmless)} no_processing={args.no_processing}")

    rows = {}
    print(f"[t0.5] vanilla {args.vanilla} ...")
    rows[args.vanilla] = measure(args.vanilla, args.base, device, harmful, harmless,
                                 no_processing=args.no_processing, max_new_tokens=args.max_new_tokens)
    print(f"[t0.5] vanilla: harmful_refusal={rows[args.vanilla]['harmful_refusal']:.3f} "
          f"harmless_complied={rows[args.vanilla]['harmless_complied']:.3f}")
    for s in args.spine:
        print(f"[t0.5] spine {s} ...")
        rows[s] = measure(s, args.base, device, harmful, harmless,
                          no_processing=args.no_processing, max_new_tokens=args.max_new_tokens)

    van = rows[args.vanilla]["harmful_refusal"]
    verdicts = {}
    for s in args.spine:
        v = dp.presence_verdict(harmful_refusal_vanilla=van,
                                harmful_refusal_spine=rows[s]["harmful_refusal"],
                                harmless_complied_spine=rows[s]["harmless_complied"])
        verdicts[s] = {"label": v.label, "uplift": v.uplift, "note": v.note}
        print(f"[t0.5] {s}: harmful_refusal={rows[s]['harmful_refusal']:.3f} (vanilla {van:.3f}, "
              f"uplift {v.uplift:+.3f}) harmless_complied={rows[s]['harmless_complied']:.3f} -> {v.label}")

    out_md = Path(args.out)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render_md(rows, verdicts, args.vanilla))
    out_md.with_suffix(".json").write_text(json.dumps({
        "base": args.base, "vanilla": args.vanilla, "spine": args.spine, "device": device,
        "n_harmful": len(harmful), "n_harmless": len(harmless), "seed": args.seed,
        "min_uplift": dp.MIN_UPLIFT, "min_harmless_complied": dp.MIN_HARMLESS_COMPLIED,
        "rows": rows, "verdicts": verdicts,
    }, indent=2))
    print(f"[t0.5] wrote {out_md} and {out_md.with_suffix('.json')}")
    print("[t0.5] VERDICTS:", {s: verdicts[s]["label"] for s in args.spine})
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
