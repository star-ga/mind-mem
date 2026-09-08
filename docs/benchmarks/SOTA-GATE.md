# The SOTA gate — what is required, and what is measured

A SOTA claim for mind-mem retrieval has never been permitted on a floor
comparison alone. This file tracks the three standing requirements and their
current measured state, so the claim can be checked rather than asserted.

Last updated 2026-09-07.

## Requirement 1 — beat, or match, the zero-dependency floor · **MET**

Full 470 eligible LongMemEval-S questions, `k=5`, both arms on the same box on
the same day. Artifacts: `docs/benchmarks/head-20260907/`.

| run | recall_any@5 | recall_all@5 (official) | MRR |
|---|---|---|---|
| floor (`bm25_baseline`) | 0.9702 | 0.8298 | 0.9081 |
| **HEAD (BM25F/SQLite, vector off)** | **0.9745** | **0.8447** | 0.8954 |

Paired: `recall_any@5` p=0.7905, `recall_all@5` p=0.3240, MRR p=0.0929 — all
three **not significant**, i.e. indistinguishable from the floor, where on
2026-09-03 the floor was significantly better on two of them. The null is not
underpowered: six discordant questions suffice for significance and the observed
counts are 14, 37 and 80.

**This alone is not SOTA.** A zero-dependency BM25 written by us is a FLOOR. It
establishes that we are not worse than the cheapest honest thing, and nothing
more.

## Requirement 2 — a vector-on hybrid run · **MET**

Measured on the full 470 eligible questions (`2026-09-07-hybrid-vector-on.md`,
commit `ae7ae2f`): the hybrid takes `recall_all@5` to **0.8660** against the
floor's 0.8298 — 35 questions won, 18 lost, **p=0.0270**. `recall_any@5` ties
the `no_expansion` arm at 0.9787 and MRR is 0.9006, neither significant.

So the dense leg buys *completeness*, not first-place ordering: it pulls the
remaining gold documents into the top five. **36 of 470 runs still got no dense
leg** and are disclosed in that report rather than dropped.

Every headline above runs the BM25F/SQLite leg with the dense leg OFF. A
product that ships a hybrid and reports a lexical number is reporting one leg.

Status: unblocked 2026-09-07 and running. It required a product fix first —
`VectorBackend.index()` called the raw sentence-transformers embedder instead of
the `_embed_for_provider` chain that five other call sites use, so a workspace
configured for ollama built an **empty** index (`vector_index_blocks: 0`,
`vector_leg_inert: True`). See commit `b7c3e26`.

That defect is the reason this requirement matters: without the adapter's
honest dense-leg probe, the run would have produced a "hybrid" number whose
vector leg contributed nothing — the same failure mode the harness docstring
already records once, when a `hybrid` config was silently answered by the
Markdown scan.

Config: `docs/benchmarks/head-20260907/hybrid-config.json`
(ollama `mxbai-embed-large`, 1024d, RRF k=60, equal BM25/vector weights).

## Requirement 3 — an external system under the same contract · **MET**

Chroma 1.5.9 ran the full set (`2026-09-07-external-chroma.md`, commit
`794fcf4`). On the **436 questions it completed** — the 34 it crashed on are
excluded rather than scored as wrong, which would have flattered us —
mind-mem's hybrid wins all three metrics: any@5 0.9794 vs 0.9312
(**p=0.000104**), all@5 0.8601 vs 0.8119 (**p=0.0257**), MRR 0.9014 vs 0.8472
(**p=0.000066**). Chroma also loses to our own zero-dependency floor on any@5
and MRR, which is the expected result on a corpus whose questions share
vocabulary with their gold documents — and is the reason the floor, not Chroma,
is the honest bar.

Until 2026-09-07 the adapter registry held exactly two entries and both were
ours. `chroma_baseline` (commit `4d1ac54`) is the first third-party retrieval
system to run the same corpus, the same questions and the same
`init`/`query`/`teardown` contract on the same box.

Verified at smoke: `chroma_hnsw_cosine`, 53 of 53 documents indexed, count
round-tripped through the store rather than taken from the writer's return,
chromadb 1.5.9.

**Disclosed trade:** both sides are given the same vectors from the same pinned
`mxbai-embed-large`. Chroma ships all-MiniLM-L6-v2; using its default would make
any difference a comparison of embedding MODELS wearing the label of a
comparison of SYSTEMS. This measures Chroma's retrieval, not Chroma as it ships,
and that limit belongs next to any number it produces.

## The remaining gap, and a candidate for it — MRR

After requirement 1, ordering is the only metric where the floor is still
ahead (0.9081 vs 0.8954 on the full set, p=0.0929 — directional, not
significant). The F5 ablation's one measurable improvement was disabling query
expansion, and it has now been re-measured on HEAD.

Stride-4 sample, 118 questions spanning all six question types:

| run | recall_any@5 | recall_all@5 | MRR |
|---|---|---|---|
| HEAD (shipped default) | 0.9746 | **0.8814** | 0.8894 |
| `no_expansion` | **0.9915** | 0.8644 | **0.9062** |
| floor | 0.9746 | 0.8390 | 0.9069 |

MRR reaches 0.9062 against the floor's 0.9069 — the ordering gap essentially
closes. **It is not free:** `recall_all@5`, the official strict metric, drops
0.8814 -> 0.8644. Nothing here is significant at n=118 (MRR p=0.2668), so a
full-set run is required before this is more than a direction, and a default
move needs its own paired gate on 470 regardless.

**Two failed approaches worth recording, so they are not retried.** The
`recall.expand_query` config key is NOT the lever: `sqlite_index` reads
`expand_mode` from `_QUERY_TYPE_PARAMS`, a per-question-TYPE table, so the key
has no effect under `recall`. The product flags it — `expand_query` is not in
`_VALID_RECALL_KEYS` and `_recall_core` logs `unknown_recall_config_keys` — but
that warning goes to stderr, which the benchmark invocation filtered. Measured: expansion-off via config is
byte-identical to HEAD, zero discordant questions on all three metrics. A knob
the code never reads produces a perfect null that looks exactly like "this
change does nothing". The working lever is `--mask no_expansion`, which rebinds
the product symbol, and the run's `mask` field proves it was applied.

## What may be said today

* The measured deficit against the zero-dependency floor is **closed**.
* mind-mem is **not** yet demonstrated SOTA, because requirements 2 and 3 have
  not both produced a committed full-set number.
* No comparison against a commercial memory product exists at all.

## What may never be said on this evidence

* That beating `bm25_baseline` is beating the field.
* That a BM25F-only run characterises the shipped hybrid.
* That a single corpus (one synthetic session-document set, undated,
  uniform-status, unprioritised) generalises to an operator's governed
  workspace, where the modifiers this corpus renders inert have live inputs.
