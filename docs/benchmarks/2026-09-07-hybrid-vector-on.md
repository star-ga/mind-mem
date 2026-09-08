# Vector-on hybrid on the full set — `all@5` beats the floor, and 36 runs did not get a dense leg

Every LongMemEval number this project has published ran the BM25F/SQLite leg
with the dense leg **off**. A product that ships a hybrid and reports one leg is
reporting one leg. This is the first full-set run with the vector leg live.

It needed a product fix first: `VectorBackend.index()` called the raw
sentence-transformers embedder instead of the `_embed_for_provider` chain that
five other call sites use, so an ollama-configured workspace built an **empty**
index (commit `b7c3e26`).

## Result — full 470 eligible questions, `k=5`, all arms on HEAD, same box

| run | recall_any@5 | recall_all@5 (official) | MRR |
|---|---|---|---|
| floor (`bm25_baseline`) | 0.9702 | 0.8298 | **0.9081** |
| HEAD BM25F, vector off | 0.9745 | 0.8447 | 0.8954 |
| HEAD + `no_expansion` | 0.9787 | 0.8447 | 0.9048 |
| **hybrid, vector on** | **0.9787** | **0.8660** | 0.9006 |

| comparison | metric | cand-only | base-only | p |
|---|---|---|---|---|
| hybrid vs **floor** | **recall_all@5** | **35** | **18** | **0.0270** |
| hybrid vs floor | recall_any@5 | 10 | 6 | 0.4545 |
| hybrid vs floor | MRR | 44 | 49 | 0.6785 |
| hybrid vs BM25F HEAD | recall_all@5 | 17 | 7 | 0.0639 |
| hybrid vs BM25F HEAD | MRR | 46 | 36 | 0.3203 |

**This is the first significant win against the floor on the official strict
metric.** `recall_all@5` — every gold document inside the top five — goes 0.8298
to 0.8660, p=0.0270. `any@5` and MRR remain indistinguishable from the floor.

## The caveat, which is not small: 36 runs had no dense leg

The run's own probe reports `vector_leg_inert: True` on **36 of 470 questions**
(7.7%), each with `vector_index_build_failed:OSError`. Those questions ran
BM25F-only while the row was labelled `hybrid`. Restricted to the 434 questions
whose dense leg was genuinely live:

| set | recall_any@5 | recall_all@5 |
|---|---|---|
| all 470 as reported | 0.9787 | 0.8660 |
| **434 with a live dense leg** | **0.9770** | **0.8618** |
| the 36 inert ones | 1.0000 | 0.9167 |

The inert questions were the *easier* ones, so they flatter the aggregate
slightly rather than depressing it. The pure-hybrid figures are the lower pair
and should be the ones quoted.

The failures are intermittent, not systematic — they spread across all six
question types and the whole range of session counts, with ollama healthy before
and after. The cause is the fallback chain: under sustained embedding load a
provider briefly declines, the chain falls through, and its last resort
(sentence-transformers) can never succeed for an ollama model tag.

## The error was also misdirecting, and that is fixed

The escaping message read `mxbai-embed-large is not a local folder and is not a
valid model identifier` — true, and pointed at HuggingFace, which is not where
the problem was. `_embed_for_provider` now raises `EmbeddingProvidersExhausted`
naming the provider that actually failed, when the configured model is a
provider tag rather than a repo id. Narrow by construction: a genuine
HuggingFace id keeps its own error, and a test asserts that so the rewrite
cannot swallow real repo problems.

## What this does and does not settle

It closes the standing requirement that a vector-on run exist, and it produces
the first metric on which this product is significantly ahead of the
zero-dependency floor.

It does **not** make a SOTA claim available. The remaining requirement is an
external system under the same adapter contract, and `chroma_baseline` (commit
`4d1ac54`) has landed but not yet been run over the full set. Beyond that, every
competing figure in this field is *answer accuracy* under an LLM judge, while
these are *retrieval* metrics; the two are not the same quantity and must not be
placed on one axis.

Before the next hybrid number is published, the 36 inert runs should be zero.
A 7.7% silent-degradation rate is small enough to miss and large enough to move
a p-value.
