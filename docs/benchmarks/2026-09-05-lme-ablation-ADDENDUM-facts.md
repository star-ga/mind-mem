# F5 addendum — the FACT sub-block layer, added post-hoc, pre-committed

**Written and committed after the pre-registered battery started running and
BEFORE any of its results were read.** It is kept out of
`ablation_mask.MASKS`, in `ablation_mask.POST_HOC_MASKS`, and reported in its
own section of the results, so the pre-registered list stays exactly what was
pre-registered. Nothing here was chosen because a result pointed at it — the
trigger was a direct probe of the eval workspace, run while the battery ran.

## What the probe measured

One eligible question, seeded through the real adapter, index inspected
directly (`benchmarks/.cache/ablation/` battery config, `mind_mem` adapter,
`turns=all`):

| fact | measured |
|---|---|
| session documents handed to the adapter | 53 |
| rows in `blocks` / `blocks_fts` after `build_index` | **373** |
| of those, `D-<n>` session parents | 53 |
| of those, `D-<n>::F<i>` FACT sub-blocks | 320 |
| `blocks_fts` rows withheld | 0 |
| median FTS document length | 65 characters |
| mean `statement` length | 1355 characters |
| distinct `status` | `{active: 373}` |
| blocks carrying a `Priority` | 0 |
| `date_score` over all 373 blocks | `{0.5: 373}` |
| `blocks.date` non-empty | 5 (`last week`, `last month`, `last Thursday`, `weeks ago` ×2 — all FACT sub-blocks) |
| `xref_edges` rows | 0 |

`index_block` mints a FACT card per extracted fact from any statement longer
than 15 characters. Those cards are rows in `blocks_fts`, and `bm25()`
computes IDF and the average document length over every row in that table,
corpus-wide, with no `WHERE` clause narrowing it — the module's own admission
section says so. So the ranking of the 53 session documents is computed
against corpus statistics drawn from a 373-document population that is 86%
short fact cards, while the zero-dep floor computes over exactly the 53
session documents it was given.

This is an **ingest-time** difference, not one of the six post-`bm25()`
modifiers F5 was ruled to ablate, which is why it needs its own mask rather
than a place in the pre-registered list.

Three of the probe rows also predict, before any run finishes, that three
pre-registered masks must be nulls on this corpus: every block is `active`
(so the ×1.2 status boost is a uniform constant), no block carries a
`Priority` (so the ×1.1 is never taken), and `date_score` is 0.5 for all 373
(so `1 - rw + rw·0.5` is a uniform constant for any `rw`). A uniform positive
multiplier cannot reorder a ranking. The battery measures this rather than
resting on it — a prediction that the measurement then confirms is worth more
than either alone — but it means a zero there is a *structural* null, not a
power failure, and the report must say which.

## The two post-hoc masks

| mask | stages disabled |
|---|---|
| `no_facts` | `facts` — `extract_facts` mints nothing, so the FTS corpus is the 53 session documents and nothing else |
| `no_facts_plain` | `facts` + all eight ranking stages — the closest structural analogue of the zero-dep floor that the product's own index can produce |

Patch verified before the runs: with `no_facts` applied, the same workspace
indexes **53** blocks instead of 373.

Same discipline as the pre-registered battery: both are run over the same 470
questions, both are reported whatever they show, and no product default moves
on the strength of either.
