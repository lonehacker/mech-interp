"""STEP B — audit the matched draft before promoting.

Two checks per planner:
  1. Decompose the residual TF-IDF 0.67: are the top discriminative
     coefficients spread across vocabulary, or concentrated in
     attack/defense intent verbs? If concentrated in verbs, the
     de-confounding is weaker than the headline suggests.
  2. Confirm request-form matching: both sides should be the same
     imperative form ("Write code that does X") differing ONLY in
     security intent. Spot-check 10 pairs by eye.

Decision: if separability is evenly spread AND request-form matches →
promote to data/. If verb-driven or form-mismatched → either rebalance
or document the limitation explicitly.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

PATH = Path("/tmp/code_contrastive_matched_draft.jsonl")

recs = [json.loads(l) for l in PATH.open()]
harmful = [r["text"] for r in recs if r["label"] == "harmful"]
harmless = [r["text"] for r in recs if r["label"] == "harmless"]
print(f"=== STEP B audit — {PATH.name} ===")
print(f"  {len(harmful)} harmful (HarmBench cybercrime) + {len(harmless)} defensive matched")

# =================================================================
# CHECK 1 — Decompose the residual TF-IDF 0.67
# =================================================================
print("\n" + "=" * 70)
print("CHECK 1 — Top discriminative TF-IDF coefficients on the matched draft")
print("=" * 70)
texts = harmful + harmless
labels = np.array([1] * len(harmful) + [0] * len(harmless))
# Fit on ALL data — we want to see what the classifier locks onto;
# not test-set generalization (already established at 0.67).
vec = TfidfVectorizer(min_df=2)
X = vec.fit_transform(texts)
clf = LogisticRegression(max_iter=2000, C=1.0, solver="liblinear")
clf.fit(X, labels)
vocab = vec.get_feature_names_out()
coefs = clf.coef_[0]
order = np.argsort(coefs)[::-1]

# Categorize: attack/defense intent verbs vs other vocabulary
attack_words = {
    "hacking", "hack", "hacks", "exploit", "exploiting", "exploits",
    "attack", "attacks", "attacking", "deface", "defacing",
    "bypass", "bypassing", "evade", "evading", "evasion",
    "steal", "stealing", "exfiltrate", "exfiltration", "exfiltrates",
    "inject", "injecting", "injection", "payload", "payloads",
    "ransomware", "keylogger", "worm", "virus", "malware", "rootkit",
    "ddos", "flood", "drop", "dropping", "delete", "deleting", "corrupt", "corrupts",
    "obfuscate", "obfuscating", "scrape", "scraping", "spy", "spying",
    "make", "money", "mine", "hijack", "hijacks", "unauthorized",
}
defense_words = {
    "secure", "securing", "security", "harden", "hardening",
    "patch", "patching", "patches", "remediated", "remediation",
    "detect", "detecting", "detection", "detector",
    "prevent", "preventing", "prevention", "mitigation", "mitigate", "mitigates",
    "monitor", "monitoring", "monitors", "audit", "audits", "auditing",
    "verify", "verifying", "verifies", "verification", "integrity",
    "protect", "protecting", "protection",
    "sanitize", "sanitizing", "validate", "validation", "validating",
    "alert", "alerting", "alerts", "flag", "flags", "flagging",
    "deobfuscate", "block", "blocks", "blocking", "salted", "recovery",
}

def category(w):
    if w in attack_words: return "attack-verb"
    if w in defense_words: return "defense-verb"
    return "other"

# Top 25 harmful-side and top 25 harmless-side
print("\n  Top 25 harmful-side coefficients (positive = harmful):")
print(f"  {'token':>20s}  {'coef':>7s}  category")
print(f"  {'-' * 20}-+-{'-' * 7}-+-{'-' * 18}")
for i in order[:25]:
    print(f"  {vocab[i]:>20s}  {coefs[i]:>+7.3f}  {category(vocab[i])}")

print("\n  Top 25 harmless-side coefficients (negative = harmless):")
print(f"  {'token':>20s}  {'coef':>7s}  category")
print(f"  {'-' * 20}-+-{'-' * 7}-+-{'-' * 18}")
for i in order[::-1][:25]:
    print(f"  {vocab[i]:>20s}  {coefs[i]:>+7.3f}  {category(vocab[i])}")

# Aggregate: what fraction of the top-N coefficients are intent verbs?
print("\n  Verb-concentration audit:")
for N in [20, 40, 60]:
    top_h = [vocab[i] for i in order[:N]]
    top_l = [vocab[i] for i in order[::-1][:N]]
    n_attack_h = sum(1 for w in top_h if w in attack_words)
    n_defense_l = sum(1 for w in top_l if w in defense_words)
    n_other_h = N - n_attack_h - sum(1 for w in top_h if w in defense_words)
    n_other_l = N - n_defense_l - sum(1 for w in top_l if w in attack_words)
    print(f"    Top {N}: harmful side  {n_attack_h}/{N} attack verbs  ({n_other_h}/{N} other)")
    print(f"            harmless side {n_defense_l}/{N} defense verbs ({n_other_l}/{N} other)")

# =================================================================
# CHECK 2 — Spot-check 10 pairs for form match
# =================================================================
print("\n" + "=" * 70)
print("CHECK 2 — Form match: harmful and matched harmless side-by-side")
print("=" * 70)
# pairs alternate in the file: even idx = harmful, odd idx = harmless
pairs = []
for i in range(0, len(recs), 2):
    if i + 1 < len(recs) and recs[i]["label"] == "harmful" and recs[i + 1]["label"] == "harmless":
        pairs.append((recs[i]["text"], recs[i + 1]["text"]))

import random

random.seed(0)
sample = random.sample(pairs, 10)
for i, (h, l) in enumerate(sample):
    h_len, l_len = len(h), len(l)
    h_stem = h.split()[0].lower()
    l_stem = l.split()[0].lower()
    stem_match = "✓" if h_stem == l_stem else "✗"
    len_ratio = abs(h_len - l_len) / max(h_len, l_len)
    len_match = "✓" if len_ratio < 0.30 else "⚠️"
    print(f"\n  [pair {i+1}] stem={stem_match}({h_stem}/{l_stem})  len-ratio={len_match}({h_len}/{l_len})")
    print(f"    H: {h}")
    print(f"    L: {l}")

# =================================================================
# DECISION GUIDANCE
# =================================================================
print("\n" + "=" * 70)
print("DECISION")
print("=" * 70)
print("""
Read the table + the 10 pairs above. Decide:

  - If top coefficients are spread across diverse vocabulary AND the
    10 pairs read as same imperative form differing only in intent
    → PROMOTE to data/code_contrastive_matched.jsonl with seed +
       content hash. Run the matched-set causal sweep next.

  - If top coefficients concentrate heavily in attack/defense intent
    verbs (e.g., top 10 harmful-side are mostly "exploit/hack/attack",
    top 10 harmless-side are mostly "detect/prevent/patch") → the
    de-confounding is weaker than the 0.67 headline suggests, because
    a model could be reading the intent verbs without engaging with
    the harm/refusal semantics. Either rebalance the prompts to
    diversify the verbs, or DOCUMENT the limitation explicitly so a
    causal result on this set can't be misread as "the model has
    learned 'detect/prevent' = harmless" rather than "the model
    refuses requests for harmful actions."

  - If any pair fails the form/length match by a lot → flag it for
    rewrite before promotion.

Recommendation comes from the human reading the output above.
""")
