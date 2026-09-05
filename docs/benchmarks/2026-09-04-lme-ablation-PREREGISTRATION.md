# F5 — SQLite ranking-stack ablation on LongMemEval-S: pre-registration

**Committed before any ablation result was produced.** The git timestamp on
this file and on `benchmarks/ablation_mask.py` is the audit trail: the mask
list below is the complete set of configurations that will be run, and every
one of them will be reported — including the ones that make retrieval worse
and the ones that do nothing measurable. Running masks until one wins and
reporting that one is the failure this pre-registration exists to prevent.

## The measured problem this probes

On LongMemEval-S, 470 eligible questions, committed artifacts of 2026-09-03:

| arm | recall_any@5 | recall_all@5 | MRR |
|---|---|---|---|
| `mind_mem` (BM25F/SQLite, vector off, cfg `b90090007f728b47`) | 0.9404 | 0.8170 | 0.8776 |
| `bm25_baseline` (zero-dep, in-memory) | 0.9702 | 0.8298 | 0.9081 |

Paired: `any@5` p=0.0043 (baseline better), `all@5` p=0.4799 (indistinguishable),
MRR p=0.0013 (baseline better). An independent audit refuted the artefact
explanations and diagnosed **ordering, not candidate recall**.

The ordering is produced in `sqlite_index.query_index`, not in
`hybrid_recall.py`: `recall()` → `_load_backend == "sqlite"` → `query_index`.

## Hypothesis under test (the architecture seat's, restated as its own claim)

> The recency prior is doing the damage, because the two largest per-type
> losses are `single-session-preference` and `temporal-reasoning` — the types
> where the gold answer is OLD.

This is a hypothesis, not a finding. A null on it is a real result and will be
reported as one: it would move the next probe to the FTS5/BM25F scoring itself
or to `rerank_hits`.

## Stages

| stage | what it disables |
|---|---|
| `columns` | BM25F per-field weights → uniform 1.0 |
| `recency` | `recency_weight` → 0.0 for every query type |
| `date_boost` | `date_boost` → 1.0 for every query type |
| `status` | the inline ×1.2 active / ×1.1 todo\|doing multiplier |
| `priority` | the inline ×1.1 for `P0`/`P1` |
| `calibration` | the per-block calibration-feedback multiplier |
| `rerank` | `rerank_hits` v7 re-scoring of the top 200 |
| `expansion` | `expand_query` widening of the FTS5 `MATCH` token set |

`status` and `priority` are inline literals with no constant to rebind; they
are cancelled through the one per-row multiplicative seam that exists (the
calibration weight), carrying the exact reciprocal of the boost each row is
about to receive. That composes with the real calibration weight, so `status`
can be ablated without silently ablating `calibration` too.

Not ablated, and why: `graph_boost` (`_apply_graph_boost`) is reachable only
via `multi-hop`'s `graph_boost_override`, and the eval workspace seeds no
graph edges — whether it can fire at all is checked separately and reported.

## The runs (13, complete list, pre-committed)

| mask | stages disabled |
|---|---|
| `control` | — (shipped default, unmasked; positive control) |
| `no_recency` | recency |
| `no_date_boost` | date_boost |
| `no_status` | status |
| `no_priority` | priority |
| `no_calibration` | calibration |
| `no_rerank` | rerank |
| `flat_columns` | columns |
| `no_expansion` | expansion |
| `no_multipliers` | recency + date_boost + status + priority + calibration |
| `bm25f_only` | the above + rerank |
| `flat_bm25` | the above + columns |
| `plain_bm25` | the above + expansion |

Each is the FULL 470-question eligible set, `turns=all`, `k=10`, config
`b90090007f728b47`, vector leg off.

## Analysis, fixed in advance

* Every mask is compared to **two** references with
  `benchmarks/paired_scorecard.py` (exact McNemar for `recall_any@5` /
  `recall_all@5`, exact paired sign test for MRR, seeded bootstrap CI,
  discordant counts reported in both directions):
  1. the **`control`** run (what the modifier does), and
  2. the committed **`bm25_baseline` rep1** artifact (whether it closes the gap).
* Per-type breakdown for `single-session-preference` and `temporal-reasoning`,
  where the hypothesis lives.
* `n` is stated for every comparison, together with whether that comparison
  **could** have detected a difference — a battery too thin to show a delta
  reports "identical" having been unable to report anything else.

## What this is not

F5 is a measurement. **No product default moves here.** If a mask wins, that
is a finding and needs its own paired gate as a 5.1 change, per the
architecture seat's ruling.
