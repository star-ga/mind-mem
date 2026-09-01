"""M1 — the embed-vs-store exposure, measured rather than assumed.

``VectorBackend._augment_for_embedding`` prepends ``[Category] [Speaker]
[Date] [tags[:50]]`` into the SAME string that gets embedded. Metadata and
content therefore share one vector and the API offers no embed-vs-store split.
The roadmap's objection was never that this is wrong -- it is that it had never
been measured against the failure it can cause.

Measured 2026-08-31 with all-MiniLM-L6-v2 (full A/B in
``benchmarks/embed_augmentation_ab.py``):

  symptom-query vs resolution-text   raw top1 80%  margin +0.0944  spread 0.1083
                                     aug top1 80%  margin +0.1156  spread 0.1068
  near-duplicate boilerplate         raw top1 12%  margin -0.0067  spread 0.0021
                                     aug top1 25%  margin -0.0056  spread 0.0012

Two conclusions, and the second is the one that matters:

1. Augmentation does NOT cause the vocabulary-mismatch collapse it was
   suspected of. It is mildly POSITIVE on ordinary content (+0.02 margin, top1
   unchanged). This closes as a negative finding WITH A NUMBER, which is the
   honest outcome and still better than the previous state of having none.
2. On boilerplate corpora the vector leg is useless. 12.5% top1 over 8 blocks
   is exactly random, the mean margin is NEGATIVE (the right block scores below
   the best distractor), and an inter-block cosine spread of 0.002 says the
   vectors are indistinguishable. Augmentation makes that spread WORSE, because
   it adds still more shared text to records that were already near-identical.
   This is the shape ``512-mind/src/memory.mind`` actually writes.

These tests pin both, so a future embedding change cannot silently regress the
ordinary case or quietly "fix" the boilerplate case without updating the claim.
"""

from __future__ import annotations

import statistics

import pytest

pytest.importorskip("sentence_transformers", reason="vector probe needs an embedder")

from mind_mem.recall_vector import VectorBackend  # noqa: E402

_RESOLUTIONS = [
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
_PROBLEMS = [
    (0, "the database runs out of connections under load"),
    (1, "memory keeps growing until the process is killed"),
    (2, "a stolen session stays valid for far too long"),
    (3, "search gets slower every day and the file keeps growing"),
    (4, "the service hammers a failing upstream and never backs off"),
]
_BOILERPLATE = [
    (
        {"Category": "witness", "Speaker": "512", "Date": f"2026-04-0{i + 1}", "Tags": "witness,compliance"},
        f"WITNESS system=sys{i:02d} time=2026-04-0{i + 1}T00:00:00Z hash={'abcdef%02d' % i}{'0' * 50} result=COMPLIANT invariants=9/9",
    )
    for i in range(8)
]


@pytest.fixture(scope="module")
def backend():
    try:
        b = VectorBackend({"model": "all-MiniLM-L6-v2"})
        b.embed(["warmup"])
    except Exception as exc:  # noqa: BLE001 - no cached model / no network
        pytest.skip(f"embedding model unavailable: {exc}")
    return b


def _texts(backend, blocks, *, augment: bool) -> list[str]:
    return [backend._augment_for_embedding(meta, body) if augment else body for meta, body in blocks]


def _top1(backend, blocks, queries, *, augment: bool) -> float:
    vecs = backend.embed(_texts(backend, blocks, augment=augment))
    hits = 0
    for correct, query in queries:
        qv = backend.embed([query])[0]
        best = max(range(len(vecs)), key=lambda i: backend.cosine_similarity(qv, vecs[i]))
        hits += best == correct
    return hits / len(queries)


def _spread(backend, blocks, *, augment: bool) -> float:
    vecs = backend.embed(_texts(backend, blocks, augment=augment))
    sims = [backend.cosine_similarity(vecs[i], vecs[j]) for i in range(len(vecs)) for j in range(i + 1, len(vecs))]
    return statistics.pstdev(sims)


class TestOrdinaryContent:
    """Augmentation must not collapse problem-phrasing -> resolution-text recall."""

    def test_symptom_queries_still_find_their_resolution(self, backend) -> None:
        assert _top1(backend, _RESOLUTIONS, _PROBLEMS, augment=True) >= 0.6

    def test_augmentation_is_not_worse_than_raw_here(self, backend) -> None:
        """The suspicion was that augmentation POISONS the vector. It does not."""
        aug = _top1(backend, _RESOLUTIONS, _PROBLEMS, augment=True)
        raw = _top1(backend, _RESOLUTIONS, _PROBLEMS, augment=False)
        assert aug >= raw, f"augmented {aug:.0%} < raw {raw:.0%}"

    def test_blocks_remain_distinguishable(self, backend) -> None:
        assert _spread(backend, _RESOLUTIONS, augment=True) > 0.02


class TestBoilerplateIsTheRealExposure:
    """Near-duplicate records: the vector leg carries almost no signal.

    Pinned as a KNOWN LIMITATION, not a passing feature. If a future change
    makes this pass properly, this test fails and the docs claim gets updated
    -- which is the point.
    """

    def test_inter_block_spread_is_effectively_zero(self, backend) -> None:
        assert _spread(backend, _BOILERPLATE, augment=True) < 0.01

    def test_augmentation_does_not_rescue_boilerplate(self, backend) -> None:
        """And in fact it narrows the spread further by adding shared text."""
        assert _spread(backend, _BOILERPLATE, augment=True) <= _spread(backend, _BOILERPLATE, augment=False) + 1e-4

    def test_ranking_here_is_near_random(self, backend) -> None:
        queries = [(i, f"512:witness system=sys{i:02d}") for i in range(8)]
        top1 = _top1(backend, _BOILERPLATE, queries, augment=True)
        assert top1 <= 0.5, (
            f"boilerplate top1 is {top1:.0%} -- better than the recorded finding. "
            "If this is a real improvement, update the M1 numbers in the docstring "
            "and the roadmap rather than loosening this bound."
        )
