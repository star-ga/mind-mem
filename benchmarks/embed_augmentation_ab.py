#!/usr/bin/env python3
"""M1 — does `_augment_for_embedding` help or hurt? Measure it.

`VectorBackend._augment_for_embedding` prepends `[Category] [Speaker] [Date]
[tags[:50]]` into the SAME string that gets embedded, so metadata and content
share one vector and there is no embed-vs-store split. That was a deliberate
disambiguation choice ("it cost $50" needs an anchor) and it may well be
net-positive -- the roadmap's complaint is that it had never been MEASURED
against the failure it can cause.

Two corpora, because they fail in opposite directions:

* SYMPTOM->RESOLUTION. Blocks are written in the vocabulary of the ANSWER;
  queries arrive in the vocabulary of the PROBLEM. If the vectors never meet,
  nothing errors and recall returns wrong-but-plausible blocks forever.
* BOILERPLATE. Near-constant records differing only in an id and a hash --
  the shape `512-mind/src/memory.mind` actually writes. Inter-block cosine
  variance collapses toward zero, so similarity ranking carries almost no
  signal and BM25 silently does all the work while the vector leg looks fine.

Reports rank-of-correct and the margin over the best distractor, augmented vs
raw. Run: `python benchmarks/embed_augmentation_ab.py`
Needs the `all-MiniLM-L6-v2` model; uses the local cache when present.
"""

from __future__ import annotations

import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mind_mem.recall_vector import VectorBackend  # noqa: E402

# --- corpus 1: written as resolutions, queried as problems -----------------
RESOLUTION_BLOCKS = [
    (
        {"Category": "decision", "Speaker": "ops", "Date": "2026-03-02", "Tags": "db,pool"},
        "Set pool_size to 32 and pool_recycle to 900 in the connection manager.",
    ),
    (
        {"Category": "decision", "Speaker": "ops", "Date": "2026-03-05", "Tags": "cache,redis"},
        "Enable the write-through cache and raise the eviction threshold to 4 GB.",
    ),
    (
        {"Category": "decision", "Speaker": "sec", "Date": "2026-03-09", "Tags": "auth,token"},
        "Rotate the signing key every 24 hours and shorten token TTL to 15 minutes.",
    ),
    (
        {"Category": "decision", "Speaker": "ops", "Date": "2026-03-11", "Tags": "index,sqlite"},
        "Rebuild the FTS index nightly and VACUUM after each compaction pass.",
    ),
    (
        {"Category": "decision", "Speaker": "app", "Date": "2026-03-14", "Tags": "retry,http"},
        "Add exponential backoff with jitter, capped at five attempts.",
    ),
]
PROBLEM_QUERIES = [
    (0, "the database runs out of connections under load"),
    (1, "memory keeps growing until the process is killed"),
    (2, "a stolen session stays valid for far too long"),
    (3, "search gets slower every day and the file keeps growing"),
    (4, "the service hammers a failing upstream and never backs off"),
]

# --- corpus 2: near-duplicate boilerplate (the 512-mind shape) -------------
BOILERPLATE = [
    (
        {"Category": "witness", "Speaker": "512", "Date": "2026-04-0%d" % (i + 1), "Tags": "witness,compliance"},
        f"WITNESS system=sys{i:02d} time=2026-04-0{i + 1}T00:00:00Z hash={'abcdef%02d' % i}{'0' * 50} result=COMPLIANT invariants=9/9",
    )
    for i in range(8)
]
BOILERPLATE_QUERIES = [(i, f"512:witness system=sys{i:02d}") for i in range(8)]


def _rank_and_margin(backend, blocks, queries, *, augment: bool):
    texts = [backend._augment_for_embedding(meta, body) if augment else body for meta, body in blocks]
    doc_vecs = backend.embed(texts)
    ranks, margins = [], []
    for correct, query in queries:
        qv = backend.embed([query])[0]
        scored = sorted(
            ((backend.cosine_similarity(qv, dv), i) for i, dv in enumerate(doc_vecs)),
            reverse=True,
        )
        order = [i for _, i in scored]
        ranks.append(order.index(correct) + 1)
        best_other = max(s for s, i in scored if i != correct)
        margins.append(dict(scored and [(i, s) for s, i in scored])[correct] - best_other)
    return ranks, margins


def _pairwise_spread(backend, blocks, *, augment: bool) -> float:
    texts = [backend._augment_for_embedding(meta, body) if augment else body for meta, body in blocks]
    vecs = backend.embed(texts)
    sims = [backend.cosine_similarity(vecs[i], vecs[j]) for i in range(len(vecs)) for j in range(i + 1, len(vecs))]
    return statistics.pstdev(sims) if len(sims) > 1 else 0.0


def _report(name, blocks, queries, backend) -> None:
    print(f"\n=== {name} ===")
    for label, aug in (("raw       ", False), ("augmented ", True)):
        ranks, margins = _rank_and_margin(backend, blocks, queries, augment=aug)
        top1 = sum(1 for r in ranks if r == 1) / len(ranks)
        print(
            f"  {label} top1={top1:.0%}  mean_rank={statistics.mean(ranks):.2f}  "
            f"mean_margin={statistics.mean(margins):+.4f}  min_margin={min(margins):+.4f}"
        )
    for label, aug in (("raw       ", False), ("augmented ", True)):
        print(f"  {label} inter-block cosine spread (sd) = {_pairwise_spread(backend, blocks, augment=aug):.4f}")


def main() -> int:
    backend = VectorBackend({"model": "all-MiniLM-L6-v2"})
    _report("corpus 1 - symptom query vs resolution text", RESOLUTION_BLOCKS, PROBLEM_QUERIES, backend)
    _report("corpus 2 - near-duplicate boilerplate (512-mind shape)", BOILERPLATE, BOILERPLATE_QUERIES, backend)
    print("\nHigher top1 / mean_margin is better. A spread near 0 in corpus 2")
    print("means similarity ranking carries almost no signal there.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
