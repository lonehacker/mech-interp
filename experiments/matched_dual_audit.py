"""Dual audit (vocabulary + length) for matched contrastive set.

Both must pass simultaneously for promotion. Vocabulary check is 5-fold
stratified CV (PRIMARY) plus single-split as a labeled diagnostic — the
single-split alone underreports variance on n=80 paired sets (SD 0.12
across folds; a single draw can land anywhere in [0.34, 0.62]). Length
check is per-pair |Δlen| distribution.

min_df ∈ {1, 2} sensitivity is reported because:
  (a) the activation space applies no rare-token filter, and
  (b) paired sets have a known min_df=1 CV artifact — rare tokens (e.g.
      CVE names) appear in exactly 2 prompts each, one per label, and
      stratified-shuffle CV systematically splits the pair across folds,
      producing anti-prediction (AUC well below 0.5) that is NOT
      evidence of residual lexical separability.

Run:
    python /path/to/code_contrastive_matched.jsonl
"""
import hashlib
import json
import statistics as s
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

path = Path(sys.argv[1] if len(sys.argv) > 1 else
            "/Users/anshulsinghle/safe_ai/mech-security/data/code_contrastive_matched.jsonl")
sha = hashlib.sha256(path.read_bytes()).hexdigest()
print(f"=== Dual audit — {path.name} ===")
print(f"sha256: {sha}\n")

recs = [json.loads(l) for l in path.open()]
harmful = [r["text"] for r in recs if r["label"] == "harmful"]
harmless = [r["text"] for r in recs if r["label"] == "harmless"]
texts = harmful + harmless
labels = np.array([1] * len(harmful) + [0] * len(harmless))
print(f"records: {len(harmful)} harmful + {len(harmless)} harmless\n")

# ===================================================================
# CHECK 1 — TF-IDF: 5-fold CV PRIMARY, single-split DIAGNOSTIC
# ===================================================================
print("=" * 70)
print("CHECK 1 — Vocabulary classifier (TF-IDF unigram LR)")
print("=" * 70)
print("PRIMARY:    5-fold StratifiedKFold(shuffle, random_state=0); vectorizer fit on train only")
print("DIAGNOSTIC: single 70/30 stratified split (random_state=0)")
print(f"LR: max_iter=2000, C=1.0, solver='liblinear'  Vec: ngram_range=(1,1), lowercase=True\n")

primary_min_df = 2
results = {}
for min_df in [1, 2]:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    aucs = []
    for tr, te in skf.split(texts, labels):
        Xtr = [texts[i] for i in tr]; Xte = [texts[i] for i in te]
        ytr, yte = labels[tr], labels[te]
        vec = TfidfVectorizer(min_df=min_df, ngram_range=(1, 1), lowercase=True)
        Xt = vec.fit_transform(Xtr); Xv = vec.transform(Xte)
        clf = LogisticRegression(max_iter=2000, C=1.0, solver="liblinear").fit(Xt, ytr)
        aucs.append(roc_auc_score(yte, clf.decision_function(Xv)))
    cv_m, cv_s = float(np.mean(aucs)), float(np.std(aucs, ddof=1))
    cv_lo, cv_hi = float(min(aucs)), float(max(aucs))

    Xtr, Xte, ytr, yte = train_test_split(texts, labels, test_size=0.3,
                                            random_state=0, stratify=labels)
    vec = TfidfVectorizer(min_df=min_df, ngram_range=(1, 1), lowercase=True)
    Xt = vec.fit_transform(Xtr); Xv = vec.transform(Xte)
    clf = LogisticRegression(max_iter=2000, C=1.0, solver="liblinear").fit(Xt, ytr)
    single_auc = roc_auc_score(yte, clf.decision_function(Xv))

    label = "PRIMARY" if min_df == primary_min_df else "sensitivity"
    print(f"min_df={min_df}  ({label}):")
    print(f"  5-fold CV:      mean={cv_m:.4f}  SD={cv_s:.4f}  range=[{cv_lo:.4f}, {cv_hi:.4f}]")
    print(f"  per-fold:       {['%.4f' % a for a in aucs]}")
    print(f"  single 70/30:   {single_auc:.4f}  (diagnostic — one draw from the CV distribution)\n")
    results[min_df] = {"cv_mean": cv_m, "cv_sd": cv_s, "cv_range": [cv_lo, cv_hi],
                        "per_fold": aucs, "single_split": single_auc}

print("Reading:")
m1 = results[1]["cv_mean"]; m2 = results[2]["cv_mean"]
print(f"  min_df=2 (PRIMARY): common-vocabulary CV AUC = {m2:.4f}")
print(f"      Gate (CV mean within 0.5 ± 0.15): {'PASS' if abs(m2 - 0.5) < 0.15 else 'FAIL'}")
print(f"  min_df=1 sensitivity: includes singleton tokens, CV AUC = {m1:.4f}")
if m1 < 0.4:
    print(f"      Below chance: paired-set CV ARTIFACT (rare tokens like CVE names appear in")
    print(f"      exactly 2 prompts — one per label — and CV splits pair members across folds,")
    print(f"      so the LR is always wrong on rare-token test prompts). NOT evidence of")
    print(f"      residual separability.")
elif m1 > 0.6:
    print(f"      Above chance — singleton tokens carry residual signal. Narrows the claim:")
    print(f"      common vocabulary at chance, rare tokens still separate.")
else:
    print(f"      Also near chance — vocabulary doesn't classify at either filter setting.")

# ===================================================================
# CHECK 2 — Length distribution
# ===================================================================
print("\n" + "=" * 70)
print("CHECK 2 — Length distribution (per-pair |Δlen|)")
print("=" * 70)
h_lens = [len(t) for t in harmful]; l_lens = [len(t) for t in harmless]
deltas = [l_lens[i] - h_lens[i] for i in range(len(harmful))]
print(f"  harmful  median={s.median(h_lens):.0f}  mean={s.mean(h_lens):.1f}  range=[{min(h_lens)}, {max(h_lens)}]")
print(f"  harmless median={s.median(l_lens):.0f}  mean={s.mean(l_lens):.1f}  range=[{min(l_lens)}, {max(l_lens)}]")
print(f"  delta (L-H): median={s.median(deltas):+.0f}  mean={s.mean(deltas):+.1f}  range=[{min(deltas):+d}, {max(deltas):+d}]")
print(f"  #pairs |Δ|>20: {sum(1 for d in deltas if abs(d)>20)}/{len(deltas)} (target: few/none)")
print(f"  #pairs |Δ|>10: {sum(1 for d in deltas if abs(d)>10)}/{len(deltas)} (target: most below)")
print(f"  harmless-longer ratio: {sum(1 for d in deltas if d>0)}/{len(deltas)} (target: ~50%)")

# ===================================================================
# Promotion gate (both must pass)
# ===================================================================
print("\n" + "=" * 70)
print("PROMOTION GATE — both must pass")
print("=" * 70)
vocab_ok = abs(results[2]["cv_mean"] - 0.5) < 0.15
median_diff = abs(s.median(l_lens) - s.median(h_lens))
length_ok = (median_diff < 8 and
              sum(1 for d in deltas if abs(d) > 20) < 5 and
              abs(sum(1 for d in deltas if d > 0) - len(deltas) / 2) < 8)
print(f"  Vocabulary (min_df=2 CV mean within 0.5 ± 0.15):  {'PASS' if vocab_ok else 'FAIL'}")
print(f"  Length (medians equal, few outliers, ratio ~50%):  {'PASS' if length_ok else 'FAIL'}")
print(f"  PROMOTE: {'YES — both audits pass' if vocab_ok and length_ok else 'NO — iterate or flag for RDO'}")

print("\n" + "=" * 70)
print("STANCE CAVEAT (uncontrolled, distinct from vocabulary)")
print("=" * 70)
print("Vocabulary at chance does NOT retire the attacker-vs-defender STANCE confound.")
print("Stance is semantic/role, not lexical — TF-IDF says nothing about it. A direction")
print("separating 'act as attacker' from 'act as defender' can exist even when bag-of-words")
print("is at chance. Any positive causal result on this set licenses the claim 'isolates")
print("refusal-OR-stance, not refusal alone.' Do NOT write 'the set is now clean.'")
