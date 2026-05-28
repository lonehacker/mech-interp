# Contrastive set audit — `data/contrastive.jsonl`

## Provenance
- Seed: 0
- n per split: 150
- Harmful source: AdvBench harmful_behaviors.csv (520 pool size)
- Harmless source: tatsu-lab/alpaca (filtered: no input field, len ≥ 10)
- Stratification: stem (first word, lowercased) × char-length bucket (width 25)

## Stem (first-word) distribution
- harmful  top-10: create(31), write(26), develop(23), provide(16), teach(7), give(7), instruct(7), describe(5), generate(5), outline(4)
- harmless top-10: create(32), write(30), develop(23), provide(17), generate(11), describe(8), give(7), explain(3), design(3), teach(2)

# Contrastive set audit

## harmful (n=150)

- chars  p10/p50/p90: 55 / 74 / 97
- tokens p10/p50/p90: 9 / 12 / 16
- sentiment proxy: pos=0, neg=25
- top unigrams: a(187), to(102), and(65), that(65), how(58), on(41), create(33), write(26), for(25), develop(23), commit(21), information(21), of(20), or(20), can(18)
- top bigrams:  how to(58), create a(31), on how(31), write a(26), develop a(22), to commit(21), that can(18), a program(14), program that(14), hack into(12), into a(12), tutorial on(11), a tutorial(10), to hack(10), a fake(10)

## harmless (n=150)

- chars  p10/p50/p90: 55 / 74 / 89
- tokens p10/p50/p90: 9 / 12 / 15
- sentiment proxy: pos=1, neg=1
- top unigrams: a(184), the(108), of(70), to(54), for(43), an(34), create(32), write(31), and(23), in(23), develop(23), about(19), that(18), provide(17), between(16)
- top bigrams:  create a(31), write a(22), develop a(20), for a(18), of a(14), a new(10), the following(10), generate a(10), of the(8), a list(8), in a(8), from the(8), describe the(7), list of(7), for an(7)

---

**Interpretation (human, required):**

> _Document which residual confounds you knowingly accept_
> _and why fixing them is not feasible. Per CLAUDE.md, do not_
> _balance the set by deleting items; fix at the data source._

## (H) — Residual-confounds paragraph

The contrastive set is length- and stem-matched (token p10/p50/p90 of 9/12/15
on both splits; top-5 instruction verbs overlap 4-of-5 by rank). Two residual
confounds are knowingly accepted. First, a sentiment / negative-affect
imbalance: harmful prompts contain 25 negative-affect word hits (hack, attack,
exploit, illegal, etc.) versus 1 in harmless, which is unavoidable given that
the harmful split is by definition about harm. Second, a topic imbalance
visible in the dominant bigrams ("how to commit", "hack into" vs "the
following", "a list"). Together these mean the diff-of-means direction
extracted from this set should be interpreted as a "harmful-vs-harmless
content" direction, not a pure "harmful-intent" direction. Disentangling
intent from affect on Gemma-2-2b-it would require either an affect-matched
harmful set (likely impossible at this dataset size on AdvBench alone) or a
probe-based decomposition. We address this with an **affect-decoupled
transfer test** (`data/affect-test.jsonl`): the working-hypothesis direction
extracted on the main set is evaluated on prompts where harmful-intent and
harmful-vocabulary are deliberately decorrelated (soft-harmful = harmful
intent in neutral vocabulary; loaded-harmless = harm-loaded vocabulary in
benign context). Disagreement between the projection and the model's actual
refusal behavior on those prompts would imply the direction is a sentiment
detector, not a refusal mechanism.
