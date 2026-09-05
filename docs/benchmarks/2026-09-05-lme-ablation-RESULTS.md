# F5 results — ablating the SQLite ranking stack on LongMemEval-S

Pre-registration: `2026-09-04-lme-ablation-PREREGISTRATION.md` (commit
`b26a5a6`, written before any result existed). Post-hoc addendum:
`2026-09-05-lme-ablation-ADDENDUM-facts.md` (commit `7af9217`, written while
the battery ran and before any of its results were read). **Every mask in both
documents was run and every one is reported below**, including the ones that
made retrieval worse and the ones that did nothing measurable.

Artifacts: `docs/benchmarks/ablation/lme-<mask>.ndjson` — one row per
question per run, carrying the served ranking itself, so every number here
can be recomputed without re-running anything.

---

## 1. The answer, in one paragraph

**The six post-`bm25()` modifiers do not explain the loss.** Twelve masked
configurations of the ranking stack — including one that switches off
*everything* — all remain significantly worse than the zero-dep BM25 floor on
`recall_any@5` (p ≤ 0.0072) and MRR (p ≤ 0.0076). Three of the six modifiers
move **exactly zero questions**. The architecture seat's recency hypothesis is
refuted with a measured mechanism, not merely unsupported. What *does* account
for the gap sits one layer earlier, at ingest: the indexer mints a FACT
sub-block per extracted card, so a 53-session haystack becomes a **373**-document
`bm25()` statistics surface. Switching that layer off *and* flattening the
ranking stack produces a run that is statistically **indistinguishable from the
floor on all three metrics** (p = 1.0000 / 0.2430 / 0.6353). That is a finding
about where to look next, not a licence to change a default.

---

## 2. Positive control — the harness is not what is being measured

`control` was run with **no `--mask` argument at all**, so
`benchmarks/ablation_mask` was never imported and no product symbol was
rebound.

| | recall_any@5 | recall_all@5 | MRR |
|---|---|---|---|
| committed `mind_mem` rep1 (2026-09-03) | 0.940426 | 0.817021 | 0.877622 |
| `control` (this battery, fresh) | 0.940426 | 0.817021 | 0.877622 |
| discordant questions | **0** | **0** | **0** |

The shipped default is unchanged when no mask is given — measured, not
asserted. It also re-confirms that this pipeline is deterministic run to run,
so every delta below is signal, not noise.

## 3. Power — could this battery have seen a difference?

The same tests, over the same 470 pairs, applied to a pair known to differ:

| comparison | discordant (cand/base) | p | verdict |
|---|---|---|---|
| `control` vs floor, `recall_any@5` | 22 (4/18) | 0.0043 | floor better |
| `control` vs floor, `recall_all@5` | 50 (22/28) | 0.4799 | not significant |
| `control` vs floor, MRR | 77 (24/53) | 0.0013 | floor better |

`min_discordant_for_significance = 6` on every metric: **any mask that moved
six or more questions in one direction would have been detected.** A zero in
the table below is therefore a measurement, not a battery that could not
report anything else. `recall_all@5` is the exception and was already the
exception in the run that started this — the two arms have never been
distinguishable on it, so a null there is weak evidence in both directions.

Independent re-derivation of the audit's diagnosis, from the committed
artifacts: complete misses at k=10 are **11** (ours) vs **9** (floor), with
`recall_any@10` 0.9766 vs 0.9809. Candidate generation is near-equal; the loss
is ordering inside the top ten.

---

## 4. Pre-registered ablation table

All 470 eligible questions, `turns=all`, `k=10`, config `b90090007f728b47`,
vector leg off. Each row is one full run. Δ is against `control`.

| mask | any@5 | all@5 | MRR | discordant vs control (any/all/MRR) | what it did |
|---|---|---|---|---|---|
| **`control`** | 0.9404 | 0.8170 | 0.8776 | — | reference |
| *floor (`bm25_baseline`)* | *0.9702* | *0.8298* | *0.9081* | *22/50/77* | *the thing to beat* |
| `no_recency` | 0.9404 | 0.8170 | 0.8777 | **0 / 0 / 1** | nothing |
| `no_status` | 0.9404 | 0.8170 | 0.8774 | **0 / 0 / 2** | nothing |
| `no_priority` | 0.9404 | 0.8170 | 0.8776 | **0 / 0 / 0** | nothing |
| `no_calibration` | 0.9404 | 0.8170 | 0.8776 | **0 / 0 / 0** | nothing |
| `no_date_boost` | 0.9426 | 0.8149 | 0.8780 | 1 / 3 / 11 | nothing measurable |
| `no_multipliers` | 0.9426 | 0.8149 | 0.8781 | 1 / 3 / 12 | nothing measurable |
| `no_rerank` | 0.9362 | 0.8064 | 0.8784 | 2 / 9 / 20 | nothing measurable |
| `bm25f_only` | 0.9362 | 0.8085 | 0.8779 | 2 / 10 / 29 | nothing measurable |
| `flat_columns` | 0.9340 | 0.8043 | 0.8678 | 3 / 6 / 39 | **HURTS** |
| `flat_bm25` | 0.9340 | 0.8064 | 0.8663 | 5 / 15 / 53 | **HURTS** (MRR) |
| `no_expansion` | 0.9426 | 0.8234 | 0.8853 | 11 / 17 / 36 | **HELPS** (MRR) |
| `plain_bm25` | 0.9404 | 0.8106 | 0.8810 | 14 / 25 / 67 | nothing measurable |

Exact paired statistics for the rows that moved anything (McNemar exact on the
binary metrics, exact paired sign test on MRR, both two-sided; bootstrap CI
seed `20260903`, 10,000 resamples):

| mask | metric | cand-only | base-only | p | 95% CI on mean paired Δ |
|---|---|---|---|---|---|
| `flat_columns` | all@5 | 0 | 6 | **0.0312** | [−0.0234, −0.0043] |
| `flat_columns` | MRR | 11 | 28 | **0.0095** | [−0.0166, −0.0039] |
| `flat_bm25` | MRR | 17 | 36 | **0.0127** | [−0.0218, −0.0009] |
| `no_expansion` | MRR | 25 | 11 | **0.0288** | [−0.0028, +0.0184] |
| `no_rerank` | all@5 | 2 | 7 | 0.1797 | [−0.0234, +0.0021] |
| `plain_bm25` | MRR | 37 | 30 | 0.4638 | [−0.0113, +0.0179] |

### Which modifiers help, which hurt, which do nothing

* **Do nothing — measurably, and for a measured reason:** `recency`, `status`,
  `priority`, `calibration`. Zero discordant questions on `any@5` and `all@5`
  in every case. The mechanism was probed directly in the eval workspace:
  `date_score` returns **0.5 for all 373 indexed blocks** (no `Date` field
  survives ingest), every block is `active`, no block carries a `Priority`,
  and there are no calibration-feedback rows. Each of those four is therefore
  a *uniform positive multiplier*, and a uniform positive multiplier cannot
  reorder a ranking. This is the strongest kind of null: predicted from
  structure before the runs finished, then measured.
* **Do nothing measurable:** `date_boost` (the one non-uniform multiplier —
  exactly 5 of 373 blocks carry a date string, all of them FACT sub-blocks),
  and `rerank_hits` v7. The reranker's removal drifts `all@5` down 0.0106 and
  MRR up 0.0008; neither clears the test.
* **Hurt:** flattening the BM25F column weights. `flat_columns` is worse than
  `control` on `all@5` (p=0.0312, 0/6 discordant — every moved question moved
  against it) and on MRR (p=0.0095). The hand-tuned field weights are earning
  their keep even on a corpus where most text lands in `all_text`.
* **Help:** disabling query expansion. `no_expansion` is the only single stage
  that improves anything measurably: MRR 0.8776 → 0.8853, 25 questions better
  vs 11 worse, p=0.0288. It also lifts `all@5` to 0.8234 (not significant).

### Multiple comparisons — stated, not buried

The pre-registered battery is 12 masks × 3 metrics = **36 tests**; with the
post-hoc masks it is **42**. A Bonferroni threshold at α=0.05 is 0.00139 (36)
or 0.00119 (42). **None of the four "moved" rows above survives that
correction.** They are reported as directional evidence worth a follow-up,
never as established effects. Only the post-hoc composite in §6 clears it.

---

## 5. Per-type breakdown — where the hypothesis lived

The hypothesis was that the recency prior costs us on the two types where the
gold answer is old. Both types say no.

**`single-session-preference` (n = 30)** — recall_any@5 / recall_all@5 / MRR

| run | any@5 | all@5 | MRR |
|---|---|---|---|
| floor | 0.8667 | 0.8667 | 0.5522 |
| `control` | 0.6667 | 0.6667 | 0.4097 |
| `no_recency` | **0.6667** | **0.6667** | **0.4097** |
| `no_date_boost` | 0.6667 | 0.6667 | 0.4097 |
| `no_multipliers` | 0.6667 | 0.6667 | 0.4097 |
| `no_rerank` | 0.6667 | 0.6667 | 0.4128 |
| `flat_columns` | 0.6333 | 0.6333 | 0.3985 |
| `no_expansion` | 0.6333 | 0.6333 | 0.4505 |
| `plain_bm25` | 0.6667 | 0.6667 | 0.4729 |

Disabling recency on the type the recency hypothesis was built from changes
**nothing at all** — not one of the 30 questions moves, on any metric.

**`temporal-reasoning` (n = 127)**

| run | any@5 | all@5 | MRR |
|---|---|---|---|
| floor | 0.9606 | 0.7795 | 0.8826 |
| `control` | 0.9213 | 0.7402 | 0.8386 |
| `no_recency` | **0.9213** | **0.7402** | **0.8388** |
| `no_date_boost` | 0.9291 | 0.7244 | 0.8464 |
| `no_multipliers` | 0.9291 | 0.7244 | 0.8466 |
| `no_rerank` | 0.9134 | 0.7323 | 0.8415 |
| `flat_columns` | 0.9055 | 0.7244 | 0.8246 |
| `no_expansion` | 0.9291 | 0.7559 | 0.8479 |
| `plain_bm25` | 0.9213 | 0.7323 | 0.8436 |

`no_recency` moves `any@5` and `all@5` by zero and MRR by +0.0002. On the
`temporal` query type the recency weight is 0.6 — double the default — and it
still cannot reorder, because the input it scales is constant.

**The recency hypothesis is refuted, with its mechanism.** It was a reasonable
read of the per-type losses; the losses are real and the cause is not this.

---

## 6. Post-hoc: the FACT sub-block layer

Declared and committed *before* these two runs and before any battery result
was read (`7af9217`), from a direct probe of the eval workspace rather than
from a result. Reported separately, because it was not pre-registered.

The probe: 53 session documents seeded through the real adapter produce
**373** rows in `blocks_fts` — 53 session parents plus **320** FACT sub-blocks
minted by `extract_facts`, median length 65 characters against a parent mean
of 1355. `bm25()` computes IDF and the average document length over every row
in that table, corpus-wide, with no `WHERE` clause narrowing it. So our 53
session documents are scored against statistics drawn from a population that
is 86% short fact cards, while the floor scores them against the 53 alone.

| run | any@5 | all@5 | MRR |
|---|---|---|---|
| `control` | 0.9404 | 0.8170 | 0.8776 |
| `no_facts` | 0.9553 | 0.8319 | 0.8834 |
| `no_facts_plain` (facts + all 8 stages) | **0.9681** | **0.8468** | **0.9069** |
| floor | 0.9702 | 0.8298 | 0.9081 |

| comparison | any@5 | all@5 | MRR |
|---|---|---|---|
| `no_facts` vs `control` | 15 disc (11/4), p=0.1185 | 35 (21/14), p=0.3105 | 73 (41/32), p=0.3492 |
| `no_facts` vs **floor** | 21 (7/14), p=0.1892 | 47 (24/23), p=1.0000 | 89 (34/55), p=0.0334 |
| `no_facts_plain` vs `control` | 15 (14/1), **p=0.0010** | 42 (28/14), p=0.0436 | 77 (55/22), **p=0.0002** |
| `no_facts_plain` vs **floor** | 13 (6/7), **p=1.0000** | 36 (22/14), p=0.2430 | 71 (38/33), p=0.6353 |

Read carefully:

* `no_facts` alone is **not** significant against `control` on any metric — but
  it is enough to make `any@5` and `all@5` indistinguishable from the floor
  (p=0.19, p=1.00), and it weakens the MRR loss from p=0.0013 to p=0.0334.
* `no_facts_plain` is significantly better than `control` on all three metrics
  and **indistinguishable from the floor on all three** — the gap that started
  this is gone. Its `any@5` (p=0.0010) and MRR (p=0.0002) survive Bonferroni
  over all 42 tests; its `all@5` (p=0.0436) does not.
* This is an **interaction**: neither the fact layer nor the ranking stack is
  significant on its own against `control` (`plain_bm25` p = 1.0000 / 0.6900 /
  0.4638), and together they are. The composite says the two layers *jointly*
  account for the gap; it does not apportion it between them, and a
  single-factor attribution would be unsupported by this data.
* Per type, `no_facts_plain` is the only run that moves
  `single-session-preference` at all: 0.6667 → 0.7667 `any@5` (floor 0.8667),
  MRR 0.4097 → 0.5889.

---

## 7. What this does and does not license

* **No product default moves on this.** F5 is a measurement; the architecture
  seat ruled that any default move is a 5.1 change needing its own paired
  gate. Nothing in this document is a change to shipped behaviour.
* **The FACT layer is a capability, not a stray multiplier.** It powers
  small-to-big retrieval (`_aggregate_facts_to_parents`). "It costs ranking on
  this benchmark" is an argument for *understanding* it — a corpus-scoped
  statistics surface, a separate FTS table for cards, or an ingest-time
  policy — not for deleting it. Measuring that a thing costs something is not
  measuring that it is worthless.
* **Scope.** This is one synthetic session-document corpus with a specific
  shape: undated, uniform-status, unprioritised, fact-heavy. The four null
  modifiers are null *because of that shape*. On an operator's governed
  workspace, where dates, statuses, priorities and calibration feedback all
  vary, those same modifiers have inputs that are not constant, and this
  battery says nothing about what they do there.
* **The next probe.** The pre-registered null pushes the question off the
  modifier stack; the post-hoc composite points it at the `bm25()` statistics
  surface. The concrete follow-up is to measure the fact layer and the ranking
  stack as *separated* factors at a size that can resolve them (a 2×2 over the
  same 470 questions is already three-quarters run: `control`, `plain_bm25`,
  `no_facts`, `no_facts_plain` are exactly that design, and its interaction
  term is what §6 reports).

## 8. Reproducing this

```bash
# one run per mask, full eligible set (each ~10 min on an idle box)
python3.12 benchmarks/longmemeval_full_run.py --adapter mind_mem \
    --mask <name> --ndjson docs/benchmarks/ablation/lme-<name>.ndjson

# the control: no --mask at all, so the mask module is never imported
python3.12 benchmarks/longmemeval_full_run.py --adapter mind_mem \
    --ndjson docs/benchmarks/ablation/lme-control.ndjson

# every paired comparison in this document
python3.12 benchmarks/lme_ablation_report.py --dir docs/benchmarks/ablation
```

Mask definitions: `benchmarks/ablation_mask.py` (`MASKS`, `POST_HOC_MASKS`).
Statistics: `benchmarks/paired_scorecard.py`, unmodified.
