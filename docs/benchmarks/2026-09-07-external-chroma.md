# An external system under the same contract — and what it actually shows

The standing gate has always required a third-party system run under the same
adapter contract, on the same box, over the same questions. Until now the
registry held two adapters and both were ours, so every published comparison
was against a floor we wrote.

`chroma_baseline` (commit `4d1ac54`) is the first outside system to run it.
Chroma 1.5.9, HNSW/cosine, **attempted** on all 470 eligible LongMemEval-S
questions and **completing 436**. Every figure below is conditional on that
436-question subset; the 34 crashes are itemised further down and are never
folded into a full-470 claim.

## Result — the 436 questions Chroma completed

| system | recall_any@5 | recall_all@5 | MRR |
|---|---|---|---|
| Chroma 1.5.9 (external) | 0.9312 | 0.8119 | 0.8472 |
| floor (`bm25_baseline`) | 0.9725 | 0.8257 | **0.9080** |
| **mind-mem hybrid** | **0.9794** | **0.8601** | 0.9014 |

| comparison | any@5 | all@5 | MRR |
|---|---|---|---|
| mind-mem vs Chroma | p=0.000104 | p=0.0257 | p=0.000066 |
| **Chroma vs the floor** | **p=0.0021 (floor wins)** | p=0.5943 | **p=0.0008 (floor wins)** |

mind-mem is ahead of Chroma on all three. The row that matters more is the
second one: **a purpose-built vector store loses to a forty-line BM25** on this
corpus, significantly, on two metrics of three.

That independently reproduces the one fixed-harness study in this field, which
found plain full-context retrieval above every memory system it tested. It is
the same result from a different direction, and it is a caution against the
category's published numbers rather than a triumph over Chroma.

## Why 34 questions are excluded, and why including them would be dishonest

The raw run is 470 rows, and the naive aggregate is `0.8638 / 0.7532 / 0.7859`.
That figure counts **34 crashes as wrong answers**. Each of those rows carries a
traceback from `adapter.init` and `n_retrieved: 0`.

The cause is shared with a defect on our own side: an intermittent embedding
failure under sustained load, the same one that left 36 of 470 hybrid runs with
an inert dense leg. The blast radius differs by architecture — mind-mem falls
back to its lexical leg and still answers, while a dense-only adapter has
nothing to fall back to and the question dies. Some of that is Chroma's shape
and some of it is my adapter being less defensive than the product it wraps.
Either way it is an availability difference, not a retrieval-quality one, and
scoring it as accuracy would be the exact unfairness this survey exists to call
out elsewhere.

Both figures are recorded here. The 436-question numbers are the ones to quote.

## What was held fixed, and what that costs

Both systems were given the **same vectors** from the same locally pinned
`mxbai-embed-large`. Chroma ships `all-MiniLM-L6-v2`; using its default would
have made any difference a comparison of embedding MODELS wearing the label of a
comparison of SYSTEMS.

So this measures Chroma's **retrieval**, not Chroma as it ships, and it compares
a hybrid against a dense-only configuration. A vector store with no lexical leg
is expected to lose on a corpus whose questions share vocabulary with their
sessions. Read as "hybrid beats dense-only here", which is a fair claim, rather
than "mind-mem beats Chroma", which is not one this run supports.

## The gate is met; the claim still is not available

All three standing requirements now have a committed full-set number: floor
parity, a vector-on run, and an external system under the same contract.

That does **not** make a SOTA claim available, for a reason no further run of
this kind will fix. Every competing figure published in this field is *answer
accuracy* under an LLM judge. These are *retrieval* metrics. They are different
quantities and must not share an axis — see
`docs/benchmarks/SOTA-GATE.md` and the landscape survey.

To claim SOTA against the field the missing piece is an answerer-and-judge arm,
with the judge and protocol stated in the figure, run against the systems that
field actually cites: Graphiti, Letta, Mem0, Hindsight, Cognee.
