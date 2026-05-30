"""Shuffle-pairing control on min_df=1.

Mechanism asserted: anti-prediction at min_df=1 on the matched set is a
paired-set CV artifact. Rare tokens (CVE names) appear in exactly 2
prompts each — one harmful, one defensive — and stratified-shuffle CV
splits the pair across folds. The classifier learns a token's label
from training and gets the test occurrence's OPPOSITE label wrong.

Prediction: if we BREAK the pairing by shuffling labels (preserving 40
H + 40 L class balance but breaking the rare-token-pair correspondence),
the anti-prediction should vanish and min_df=1 CV AUC should return to
chance (~0.5).

If shuffled labels → chance: mechanism confirmed.
If shuffled labels → still ~0.29: explanation wrong, something else.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

path = Path("/Users/anshulsinghle/safe_ai/mech-security/data/code_contrastive_matched.jsonl")
print(f"file:   {path}")
print(f"sha256: {hashlib.sha256(path.read_bytes()).hexdigest()}\n")
recs = [json.loads(l) for l in path.open()]
harmful = [r["text"] for r in recs if r["label"] == "harmful"]
harmless = [r["text"] for r in recs if r["label"] == "harmless"]
texts = harmful + harmless
true_labels = np.array([1] * len(harmful) + [0] * len(harmless))


def run_cv(texts, labels, min_df, seed):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in skf.split(texts, labels):
        Xtr = [texts[i] for i in tr]; Xte = [texts[i] for i in te]
        ytr, yte = labels[tr], labels[te]
        vec = TfidfVectorizer(min_df=min_df, ngram_range=(1, 1), lowercase=True)
        Xt = vec.fit_transform(Xtr); Xv = vec.transform(Xte)
        clf = LogisticRegression(max_iter=2000, C=1.0, solver="liblinear").fit(Xt, ytr)
        aucs.append(roc_auc_score(yte, clf.decision_function(Xv)))
    return aucs


# 1. Baseline: true labels (reproduce 0.29)
print("=" * 70)
print("Baseline (true paired labels, min_df=1):")
print("=" * 70)
baseline = run_cv(texts, true_labels, min_df=1, seed=0)
print(f"  per-fold: {['%.4f' % a for a in baseline]}")
print(f"  mean={np.mean(baseline):.4f}  SD={np.std(baseline, ddof=1):.4f}  "
      f"range=[{min(baseline):.4f}, {max(baseline):.4f}]")
print("  → asserted: anti-prediction (every fold below 0.5)\n")

# 2. Shuffled labels: 10 independent seeds for the shuffle, 5-fold CV each
# Each shuffle preserves 40-40 balance but breaks pair-label correspondence
print("=" * 70)
print("Shuffled labels (preserves 40 H + 40 L balance, breaks pairing), min_df=1:")
print("=" * 70)
shuffle_seeds = list(range(10))
all_shuffle_means = []
for shuf_seed in shuffle_seeds:
    rng = np.random.default_rng(shuf_seed)
    shuf = true_labels.copy()
    rng.shuffle(shuf)
    aucs = run_cv(texts, shuf, min_df=1, seed=0)
    m = float(np.mean(aucs))
    all_shuffle_means.append(m)
    print(f"  shuf_seed={shuf_seed}: per-fold {['%.4f' % a for a in aucs]}  mean={m:.4f}")

shuf_mean = float(np.mean(all_shuffle_means))
shuf_sd = float(np.std(all_shuffle_means, ddof=1))
print(f"\n  Across {len(shuffle_seeds)} shuffle seeds:")
print(f"  mean of fold-means = {shuf_mean:.4f}  SD = {shuf_sd:.4f}")
print(f"  range of fold-means = [{min(all_shuffle_means):.4f}, {max(all_shuffle_means):.4f}]")

# 3. Verdict
print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)
baseline_mean = float(np.mean(baseline))
print(f"  True-label baseline mean:   {baseline_mean:.4f}  (anti-prediction)")
print(f"  Shuffled-label mean:        {shuf_mean:.4f}  (chance prediction is ≈ 0.5)")
print(f"  Δ (shuffled - baseline):    {shuf_mean - baseline_mean:+.4f}")
if abs(shuf_mean - 0.5) < 0.05 and shuf_mean - baseline_mean > 0.1:
    print("  → MECHANISM CONFIRMED: breaking the pairing restored chance.")
    print("     min_df=1 anti-prediction on the matched set is a paired-construction")
    print("     CV artifact, not residual lexical separability.")
else:
    print("  → MECHANISM NOT CONFIRMED: shuffled labels didn't restore chance.")
    print("     The explanation is incomplete; something else drives the 0.29.")
