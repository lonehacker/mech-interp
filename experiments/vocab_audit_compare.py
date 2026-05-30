"""TF-IDF lexical-confound audit on ANY contrastive jsonl.

Generalized version of code_contrastive_vocab_audit.py. Takes a list of
paths and reports the headline AUC for each: if a contrastive set's two
sides can be classified by word counts alone, ANY direction the diff-of-
means recipe finds is confounded with the lexical contrast.

Run:
    python /tmp/vocab_audit_compare.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def audit(path: Path, name: str = None):
    name = name or path.name
    recs = [json.loads(l) for l in path.open()]
    harmful = [r["text"] for r in recs if r["label"] == "harmful"]
    harmless = [r["text"] for r in recs if r["label"] == "harmless"]
    texts = harmful + harmless
    labels = np.array([1] * len(harmful) + [0] * len(harmless))
    print(f"\n=== {name} | {len(harmful)} harmful + {len(harmless)} harmless ===")
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.3, random_state=0, stratify=labels
    )
    out = {}
    for vec_name, vec in [
        ("count (unigram)", CountVectorizer(min_df=2)),
        ("tfidf (unigram)", TfidfVectorizer(min_df=2)),
        ("tfidf (uni+bigram)", TfidfVectorizer(min_df=2, ngram_range=(1, 2))),
    ]:
        Xt = vec.fit_transform(X_train)
        Xv = vec.transform(X_test)
        clf = LogisticRegression(max_iter=2000, C=1.0, solver="liblinear")
        clf.fit(Xt, y_train)
        tr = roc_auc_score(y_train, clf.decision_function(Xt))
        te = roc_auc_score(y_test, clf.decision_function(Xv))
        print(f"  {vec_name:<25s}  train AUC={tr:.4f}  test AUC={te:.4f}  (vocab={Xt.shape[1]})")
        out[vec_name] = (tr, te)
    return out


sets = [
    ("Phase 1 / Gemma — data/contrastive.jsonl (AdvBench + Alpaca)",
     "/Users/anshulsinghle/safe_ai/mech-security/data/contrastive.jsonl"),
    ("Phase 2 / Qwen — data/code_contrastive.jsonl (HB+AdvBench-code + CodeAlpaca)",
     "/Users/anshulsinghle/safe_ai/mech-security/data/code_contrastive.jsonl"),
    ("Phase 2 matched DRAFT — /tmp/code_contrastive_matched_draft.jsonl (HB-cyber + defensive equivalents)",
     "/tmp/code_contrastive_matched_draft.jsonl"),
]
print("=" * 80)
print("Lexical confound audit — TF-IDF bag-of-words logistic regression, 70/30 split")
print("=" * 80)
results = {}
for name, path in sets:
    p = Path(path)
    if not p.exists():
        print(f"\n[skip] {name}: file not found at {path}")
        continue
    results[name] = audit(p, name=name)

print("\n" + "=" * 80)
print("Summary — test AUC (tfidf unigram) by set")
print("=" * 80)
for name, r in results.items():
    print(f"  {r['tfidf (unigram)'][1]:.4f}  {name}")
