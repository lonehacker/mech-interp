"""Re-judge existing Step 3 + affect-test completions with the calibrated
Claude judge. Audit whether the substring scorer's headline numbers are robust.

Cost: ~50 API calls × ~0.001 = $0.05.

Run:
    zsh -c 'source ~/.zshrc && python -m experiments.audit_existing_scorers'
"""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.eval_llm import _client, judge_one


def main() -> int:
    client = _client()

    # ===== Step 3 audit =====
    print("=" * 72)
    print("Audit 1: Step 3 — full re-judge of all 12+12 baseline_harmful & ablate_real_harmful")
    print("=" * 72)
    step3 = json.loads(Path("artifacts/runs/phase1_step3/20260525-152559/result.json").read_text())
    audits = []

    for cell_name, expected_dominant in [
        ("baseline_harmful", "REFUSED"),
        ("ablate_real_harmful", "COMPLIED"),
        ("ablate_random_harmful", "REFUSED"),
        ("add_real_harmless", "REFUSED"),
        ("baseline_harmless", "COMPLIED"),
    ]:
        cell = step3["conditions"][cell_name]
        samples = cell["samples"]
        substring_refused = sum(s["refused"] for s in samples)
        substring_rate = substring_refused / len(samples)
        print(f"\n[{cell_name}]  substring rate: {substring_rate:.2f} ({substring_refused}/{len(samples)}) — expected dominant label: {expected_dominant}")
        new_labels = []
        for i, s in enumerate(samples):
            v = judge_one(s["prompt"], s["completion"], client=client)
            new_labels.append(v.label)
            old = "REFUSED" if s["refused"] else "COMPLIED"
            mark = "OK" if v.label == old else f"DISAGREE (old={old})"
            print(f"  [{i+1:>2}] new={v.label:<14} {mark}")
        new_refused = sum(1 for l in new_labels if l == "REFUSED")
        new_partial = sum(1 for l in new_labels if l == "PARTIAL_COMPLY")
        new_complied = sum(1 for l in new_labels if l == "COMPLIED")
        agree = sum(1 for s, l in zip(samples, new_labels)
                    if (s["refused"] and l == "REFUSED") or (not s["refused"] and l == "COMPLIED"))
        audits.append({
            "cell": cell_name,
            "substring_refused": substring_refused,
            "substring_rate": substring_rate,
            "new_refused": new_refused,
            "new_partial": new_partial,
            "new_complied": new_complied,
            "n": len(samples),
            "agree_with_substring": agree,
        })
        print(f"   → new judge: REFUSED={new_refused}, PARTIAL={new_partial}, COMPLIED={new_complied}; "
              f"agreement with substring: {agree}/{len(samples)} = {agree/len(samples):.0%}")

    # ===== Affect-test audit =====
    print("\n" + "=" * 72)
    print("Audit 2: Affect-test — re-judge 30 prompts (full set = 45)")
    print("=" * 72)
    affect = json.loads(Path("artifacts/runs/phase1_affect_test/20260525-160435/result.json").read_text())
    print("Top keys of affect-test:", list(affect.keys())[:20])
    # Try to find per-prompt records
    items = None
    if "results" in affect:
        items = affect["results"]
    elif "items" in affect:
        items = affect["items"]
    elif "samples" in affect:
        items = affect["samples"]
    else:
        # Maybe per-category structure?
        for k, v in affect.items():
            if isinstance(v, dict) and "samples" in v:
                print(f"  found samples under affect['{k}']")
    print(f"items type: {type(items)}, len: {len(items) if items else '?'}")
    if items and isinstance(items, list):
        for i, s in enumerate(items[:30]):
            prompt = s.get("prompt", s.get("input", ""))
            gen = s.get("completion", s.get("response", s.get("generation", "")))
            old = s.get("refused", s.get("scorer_label", "?"))
            old_label = ("REFUSED" if (old is True or str(old).upper() == "REFUSED")
                          else "COMPLIED" if (old is False or str(old).upper() == "COMPLIED")
                          else f"OLD={old}")
            v = judge_one(prompt, gen, client=client)
            mark = "OK" if v.label == old_label else "DISAGREE"
            print(f"  [{i+1:>2}] {mark} new={v.label:<14} old={old_label}: {prompt[:50]}")

    # ===== Save =====
    out_path = Path("results/audit_existing_scorers.json")
    out_path.write_text(json.dumps(audits, indent=2))
    print(f"\nSummary -> {out_path}")
    print("\n=== Bottom line ===")
    for a in audits:
        print(f"  {a['cell']:<30} substring={a['substring_rate']:.2f}  new={a['new_refused']}/{a['n']} refused; agreement={a['agree_with_substring']}/{a['n']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
