# `no_expansion` on the full set — the ordering gap closes, at no recall cost

The F5 ablation (2026-09-05) found disabling query expansion to be the only
single stage that measurably improved ordering: MRR 0.8776 → 0.8853, 25
questions better against 11 worse, p=0.0288. It did not survive Bonferroni over
that battery's 42 tests, so it was recorded as a lead, not an effect, and the
architecture ruling was that any default move needs **its own paired gate**.

This is that gate. Full 470 eligible questions, `k=5`, both arms on HEAD.

| run | recall_any@5 | recall_all@5 (official) | MRR |
|---|---|---|---|
| HEAD (shipped default) | 0.9745 | 0.8447 | 0.8954 |
| **`no_expansion`** | **0.9787** | **0.8447** | **0.9048** |
| floor (`bm25_baseline`) | 0.9702 | 0.8298 | 0.9081 |

| comparison | metric | cand-only | base-only | p |
|---|---|---|---|---|
| vs HEAD | **MRR** | **27** | **11** | **0.0139** |
| vs HEAD | any@5 | 5 | 3 | 0.7266 |
| vs HEAD | all@5 | 6 | 6 | 1.0000 |
| vs floor | MRR | 33 | 38 | 0.6353 |
| vs floor | all@5 | 21 | 14 | 0.3105 |

**It is free.** `recall_all@5` is *identical* — 0.8447 both, six discordant
questions in each direction. Nothing is traded away for the ordering gain.

**It closes the last gap against the floor.** HEAD trailed the floor on MRR
32/48 (p=0.0929, directional). `no_expansion` sits at 33/38 (p=0.6353) — not
merely indistinguishable but centred.

## The sample said otherwise, and the sample was wrong

A stride-4 sub-sample (118 questions, all six types) run first suggested this
change **cost** `recall_all@5`: 0.8814 → 0.8644. On the full set that drop does
not exist. It was sampling noise, and a default proposed off that sample would
have been justified by a number that evaporates at full power.

Recorded because the sub-sample is otherwise a good instrument — it is
type-representative and a quarter of the cost — and this is precisely its
limit: it can order candidates, it cannot price them.

## Two dead levers, so they are not retried

* **`recall.expand_query` is not the switch.** `sqlite_index` reads
  `expand_mode` from `_QUERY_TYPE_PARAMS`, a per-question-TYPE table, not from
  user config, so the key has no effect under `recall`. Measured: config
  expansion-off is byte-identical to HEAD, **zero** discordant questions on all
  three metrics.

  **The product was not at fault and said so.** `expand_query` is absent from
  `_VALID_RECALL_KEYS`, and `_recall_core` logs `unknown_recall_config_keys`
  on every call — verified by reproducing it. The warning went to stderr, which
  the benchmark invocation filtered, so *I* did not see it. Corrected here
  because the first draft of this file called the key "silently inert", which
  blamed the product for my filtered log.
* The working lever is `--mask no_expansion`, which rebinds the product symbol.
  Every row of this run carries `mask: no_expansion`, which is the proof it was
  applied — the config run's rows carry `mask: ""`.

## What this does and does not authorise

It authorises the **decision** to move the default; it is not itself that move.
Flipping `expand_query` off by default changes shipped behaviour for every
caller, so by the versioning rule it is a **minor**, not a patch, and it is the
architecture seat's call rather than a benchmark's.

Scope, restated: one synthetic session-document corpus. Query expansion exists
to help queries whose vocabulary does not match the corpus. LongMemEval-S
questions are drawn from the same generated text as the sessions, so the
mismatch expansion was built for is largely absent here. That is a reason to
expect this result to be corpus-specific, and a reason not to generalise it to
an operator's governed workspace without measuring there.
