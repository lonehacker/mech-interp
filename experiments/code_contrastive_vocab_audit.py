"""Vocabulary audit of data/code_contrastive.jsonl.

Quantifies the lexical mean shift between the harmful and harmless sides.
The planner's observation: AUC = 0.9996 at Qwen L0 (embedding output)
implies separability is present BEFORE any computation — i.e., partly
lexical by construction. This script puts numbers on that.

Outputs:
  1. Token-frequency overlap on each side (top-K words).
  2. Discriminative words: TF-IDF score by side.
  3. A simple "vocabulary classifier" baseline: train an L2-regularized
     logistic regression on bag-of-words features, hold out 30%, report
     test AUC. If vocabulary alone hits AUC ≥ 0.95, the embedding-layer
     separation is overwhelmingly lexical.

No torch / no model load. Pure sklearn + counts.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/Users/anshulsinghle/safe_ai/mech-security")

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def tokenize(s):
    """Simple word-tokenizer; lowercases, drops punctuation."""
    return re.findall(r"[a-z]+", s.lower())


def main():
    print("=" * 70)
    print("Vocabulary audit — data/code_contrastive.jsonl")
    print("=" * 70)

    recs = [json.loads(l) for l in open(
        "/Users/anshulsinghle/safe_ai/mech-security/data/code_contrastive.jsonl"
    )]
    harmful = [r["text"] for r in recs if r["label"] == "harmful"]
    harmless = [r["text"] for r in recs if r["label"] == "harmless"]
    print(f"\n  harmful: {len(harmful)} | harmless: {len(harmless)}")

    # === 1. Token-frequency comparison ===
    print("\n" + "=" * 70)
    print("CHECK 1 — Most frequent words on each side (after stop-word naivety)")
    print("=" * 70)
    h_tokens = Counter()
    l_tokens = Counter()
    for s in harmful: h_tokens.update(tokenize(s))
    for s in harmless: l_tokens.update(tokenize(s))

    stop = {"a", "the", "to", "of", "for", "in", "on", "and", "or", "is", "that",
            "with", "by", "as", "an", "be", "have", "from", "it", "this", "are"}

    print(f"\n  Top 15 harmful-side words (ex stopwords):")
    for w, c in h_tokens.most_common():
        if w in stop: continue
        print(f"    {w:>20s}  {c:>4d}  (harmless: {l_tokens.get(w, 0)})")
        if not [w for w, c in h_tokens.most_common() if w not in stop][:15].index(w) < 14:
            break
    print(f"\n  Top 15 harmless-side words (ex stopwords):")
    seen = 0
    for w, c in l_tokens.most_common():
        if w in stop or seen >= 15: continue
        print(f"    {w:>20s}  {c:>4d}  (harmful: {h_tokens.get(w, 0)})")
        seen += 1

    # === 2. Most discriminative words (raw lift) ===
    print("\n" + "=" * 70)
    print("CHECK 2 — Most discriminative tokens by relative-frequency lift")
    print("=" * 70)
    h_total = sum(h_tokens.values())
    l_total = sum(l_tokens.values())
    vocab = set(h_tokens) | set(l_tokens)
    # Lift = log( (P(w|harmful) + ε) / (P(w|harmless) + ε) )
    eps = 1e-4
    lifts = []
    for w in vocab:
        ph = (h_tokens.get(w, 0) + eps) / (h_total + eps)
        pl = (l_tokens.get(w, 0) + eps) / (l_total + eps)
        lift = np.log(ph / pl)
        n = h_tokens.get(w, 0) + l_tokens.get(w, 0)
        if n >= 3:  # ignore singletons
            lifts.append((w, lift, h_tokens.get(w, 0), l_tokens.get(w, 0)))
    lifts.sort(key=lambda x: x[1], reverse=True)
    print("\n  Most harmful-discriminative (log P(w|H) / P(w|L)):")
    for w, l, hc, lc in lifts[:20]:
        print(f"    {w:>20s}  lift={l:+.2f}  H={hc:>3d}  L={lc:>3d}")
    print("\n  Most harmless-discriminative:")
    for w, l, hc, lc in lifts[-15:][::-1]:
        print(f"    {w:>20s}  lift={l:+.2f}  H={hc:>3d}  L={lc:>3d}")

    # === 3. Vocabulary classifier baseline ===
    print("\n" + "=" * 70)
    print("CHECK 3 — Vocabulary-only classifier (LR on TF-IDF bag of words)")
    print("=" * 70)
    texts = harmful + harmless
    labels = np.array([1] * len(harmful) + [0] * len(harmless))
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.3, random_state=0, stratify=labels
    )

    for vec_name, vec in [
        ("count (unigram)", CountVectorizer(min_df=2)),
        ("tfidf (unigram)", TfidfVectorizer(min_df=2)),
        ("tfidf (uni+bigram)", TfidfVectorizer(min_df=2, ngram_range=(1, 2))),
    ]:
        Xt = vec.fit_transform(X_train)
        Xv = vec.transform(X_test)
        clf = LogisticRegression(max_iter=2000, C=1.0, solver="liblinear")
        clf.fit(Xt, y_train)
        train_auc = roc_auc_score(y_train, clf.decision_function(Xt))
        test_auc = roc_auc_score(y_test, clf.decision_function(Xv))
        print(f"  {vec_name:<25s}  train AUC={train_auc:.4f}  test AUC={test_auc:.4f}  "
              f"(vocab size={Xt.shape[1]})")

    print("\n  Interpretation: if 'tfidf unigram' test AUC ≈ 1.0, then the harmful")
    print("  side is essentially perfectly identifiable from the BAG OF WORDS alone.")
    print("  In that regime, ANY direction in the activation space that aligns with")
    print("  the vocabulary-discrimination subspace will look like a 'harmful-vs-")
    print("  harmless classifier' at near-perfect AUC — including the diff-of-means")
    print("  direction and most random unit vectors that happen to project onto it.")


if __name__ == "__main__":
    main()
