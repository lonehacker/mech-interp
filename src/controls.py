"""
Audits and control baselines.

Per CLAUDE.md: this module REPORTS. It does not balance, it does not modify
the contrastive set, it does not relax thresholds. Audit findings are read by
the human in `results/contrastive_audit.md`, who then decides whether to
fix the data (preferred) or accept a stated residual confound.

What's here:
- audit_contrastive: token-length quantiles, cheap sentiment proxy, top
  unigram/bigram frequencies, per-split.
- (random-direction baseline lives in directions.random_unit_vector for
  symmetry with the diff-of-means direction itself.)
"""

from __future__ import annotations

import json
import re
import statistics
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model import ModelBundle


# Tiny lexicons for the sentiment proxy. NOT a real classifier — its job is to
# surface "harmful prompts are systematically angrier" before that confound
# silently dominates the diff-of-means direction. If a serious sentiment
# question arises, swap in a model-based scorer in a follow-up; do not tune
# these lexicons to make the numbers prettier.
_POS_WORDS = {
    "great", "good", "love", "happy", "thanks", "thank", "please", "kind",
    "helpful", "wonderful", "awesome", "nice", "best", "amazing", "enjoy",
}
_NEG_WORDS = {
    "hate", "kill", "destroy", "attack", "harm", "hurt", "dangerous",
    "violent", "abuse", "threat", "exploit", "bomb", "weapon", "illegal",
}

_TOKEN_RE = re.compile(r"[a-z']+")


@dataclass(frozen=True)
class SplitAudit:
    name: str
    n: int
    char_p10: float
    char_p50: float
    char_p90: float
    token_p10: float
    token_p50: float
    token_p90: float
    pos_hits: int
    neg_hits: int
    top_unigrams: list[tuple[str, int]]
    top_bigrams: list[tuple[str, int]]


@dataclass(frozen=True)
class ContrastiveAudit:
    harmful: SplitAudit
    harmless: SplitAudit


def _pct(xs: list[int], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
    return float(s[k])


def _audit_split(name: str, prompts: list[str], bundle: "ModelBundle | None") -> SplitAudit:
    chars = [len(p) for p in prompts]

    if bundle is not None:
        token_counts = [int(bundle.model.to_tokens(p, prepend_bos=False).shape[-1])
                        for p in prompts]
    else:
        # Fallback: whitespace token count. Caller should pass a bundle when
        # available; the whitespace count is a coarse stand-in only.
        token_counts = [len(p.split()) for p in prompts]

    pos = 0
    neg = 0
    unigram_counter: Counter[str] = Counter()
    bigram_counter: Counter[tuple[str, str]] = Counter()

    for p in prompts:
        toks = _TOKEN_RE.findall(p.lower())
        unigram_counter.update(toks)
        bigram_counter.update(zip(toks, toks[1:]))
        for t in toks:
            if t in _POS_WORDS:
                pos += 1
            if t in _NEG_WORDS:
                neg += 1

    top_uni = [(w, c) for w, c in unigram_counter.most_common(15)]
    top_bi = [(" ".join(b), c) for b, c in bigram_counter.most_common(15)]

    return SplitAudit(
        name=name,
        n=len(prompts),
        char_p10=_pct(chars, 10),
        char_p50=_pct(chars, 50),
        char_p90=_pct(chars, 90),
        token_p10=_pct(token_counts, 10),
        token_p50=_pct(token_counts, 50),
        token_p90=_pct(token_counts, 90),
        pos_hits=pos,
        neg_hits=neg,
        top_unigrams=top_uni,
        top_bigrams=top_bi,
    )


def audit_contrastive(
    path: str | Path,
    bundle: "ModelBundle | None" = None,
) -> ContrastiveAudit:
    """Audit the frozen contrastive set at `path`.

    Expected format: JSONL, each line `{"text": "...", "label": "harmful"|"harmless"}`.

    Pass a bundle to get tokenizer-accurate token counts; pass None for a
    whitespace-token approximation (fine for a first pass).
    """
    p = Path(path)
    harmful: list[str] = []
    harmless: list[str] = []
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            text = rec["text"]
            label = rec["label"]
            if label == "harmful":
                harmful.append(text)
            elif label == "harmless":
                harmless.append(text)
            else:
                raise ValueError(f"unknown label {label!r} in {p}")

    return ContrastiveAudit(
        harmful=_audit_split("harmful", harmful, bundle),
        harmless=_audit_split("harmless", harmless, bundle),
    )


def audit_to_markdown(audit: ContrastiveAudit) -> str:
    """Render the audit as a human-readable markdown block to drop into
    results/contrastive_audit.md. The human writes the *interpretation*
    paragraph below it — this function does not.
    """
    lines: list[str] = ["# Contrastive set audit", ""]
    for split in (audit.harmful, audit.harmless):
        d = asdict(split)
        lines.append(f"## {split.name} (n={split.n})")
        lines.append("")
        lines.append(
            f"- chars  p10/p50/p90: {split.char_p10:.0f} / {split.char_p50:.0f} / {split.char_p90:.0f}"
        )
        lines.append(
            f"- tokens p10/p50/p90: {split.token_p10:.0f} / {split.token_p50:.0f} / {split.token_p90:.0f}"
        )
        lines.append(f"- sentiment proxy: pos={split.pos_hits}, neg={split.neg_hits}")
        lines.append("- top unigrams: " + ", ".join(f"{w}({c})" for w, c in split.top_unigrams))
        lines.append("- top bigrams:  " + ", ".join(f"{w}({c})" for w, c in split.top_bigrams))
        lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("**Interpretation (human, required):**")
    lines.append("")
    lines.append("> _Document which residual confounds you knowingly accept_")
    lines.append("> _and why fixing them is not feasible. Per CLAUDE.md, do not_")
    lines.append("> _balance the set by deleting items; fix at the data source._")
    return "\n".join(lines)
