# Copyright 2026 STARGA, Inc.
"""Edge confidence from cross-source corroboration (roadmap RM-0505).

The claim the roadmap makes is precise: *an edge seen in N independent
source blocks outranks a single-source edge*. Until now the graph
answered every claim with the confidence of whichever row a query
happened to reach, so three blocks agreeing and one block asserting were
the same number.

Two properties carry the weight and both are tested against their
inverse:

* **Corroboration must beat strength.** Two independent sources at 0.6
  outrank one source at 0.8. A combiner that returned the maximum would
  pass every other test in this file.
* **It must be derived, never stored.** The item says "sidecar only —
  never in the sealed audit-hash preimage"; this goes further and keeps
  no sidecar at all, so a score can never become ungoverned content that
  entered the graph without a receipt. The schema-and-rows test is what
  holds that line.
"""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path
from typing import Iterator

import pytest

from mind_mem.edge_grounded_answer import GAP_SINGLE_SOURCE_CLAIM, build_context
from mind_mem.knowledge_graph import Corroboration, KnowledgeGraph

BLOCK_A = "D-20260101-001"
BLOCK_B = "D-20260101-002"
BLOCK_C = "D-20260101-003"


@pytest.fixture()
def graph(admitted) -> Iterator[KnowledgeGraph]:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        kg = KnowledgeGraph(str(Path(td) / "kg.db"))
        try:
            yield kg
        finally:
            kg.close()


def _db_fingerprint(db_path: str) -> tuple:
    """Every table name and every edge row, as a comparable value."""
    conn = sqlite3.connect(db_path)
    try:
        tables = tuple(sorted(r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()))
        rows = tuple(
            conn.execute(
                "SELECT subject, predicate, object, source_block_id, confidence, "
                "valid_from, valid_until, metadata FROM edges ORDER BY 1,2,3,4"
            ).fetchall()
        )
        return tables, rows
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# The combiner
# ---------------------------------------------------------------------------


class TestCombiner:
    def test_two_weak_sources_outrank_one_strong_source(self) -> None:
        """The roadmap's claim, stated as an inequality."""
        pair = Corroboration("a", "depends_on", "b", (BLOCK_A, BLOCK_B), (0.6, 0.6))
        single = Corroboration("a", "depends_on", "c", (BLOCK_A,), (0.8,))
        assert pair.sources == 2
        assert single.sources == 1
        assert pair.corroborated_confidence > single.corroborated_confidence

    def test_bounded_and_monotone(self) -> None:
        one = Corroboration("a", "depends_on", "b", (BLOCK_A,), (0.9,))
        two = Corroboration("a", "depends_on", "b", (BLOCK_A, BLOCK_B), (0.9, 0.9))
        three = Corroboration("a", "depends_on", "b", (BLOCK_A, BLOCK_B, BLOCK_C), (0.9, 0.9, 0.9))
        assert one.corroborated_confidence < two.corroborated_confidence < three.corroborated_confidence
        assert three.corroborated_confidence <= 1.0

    def test_a_certain_source_saturates(self) -> None:
        certain = Corroboration("a", "depends_on", "b", (BLOCK_A,), (1.0,))
        assert certain.corroborated_confidence == 1.0

    def test_no_sources_scores_zero(self) -> None:
        empty = Corroboration("a", "depends_on", "b", (), ())
        assert empty.sources == 0
        assert empty.corroborated_confidence == 0.0

    def test_score_is_a_function_of_the_set_not_the_row_order(self) -> None:
        forward = Corroboration.from_edges(
            ("a", "depends_on", "b"),
            [_edge("a", "b", BLOCK_A, 0.3), _edge("a", "b", BLOCK_B, 0.7)],
        )
        backward = Corroboration.from_edges(
            ("a", "depends_on", "b"),
            [_edge("a", "b", BLOCK_B, 0.7), _edge("a", "b", BLOCK_A, 0.3)],
        )
        assert forward == backward
        assert forward.corroborated_confidence == backward.corroborated_confidence

    def test_one_block_asserting_twice_is_one_source(self) -> None:
        """Anti-gaming: a repeated document must not manufacture evidence."""
        doubled = Corroboration.from_edges(
            ("a", "depends_on", "b"),
            [_edge("a", "b", BLOCK_A, 0.6), _edge("a", "b", BLOCK_A, 0.6)],
        )
        assert doubled.sources == 1
        assert doubled.source_block_ids == (BLOCK_A,)
        # Positive control: two DIFFERENT blocks at the same confidence do
        # corroborate, so the assertion above is the dedup and not a
        # constructor that always collapses to one.
        genuine = Corroboration.from_edges(
            ("a", "depends_on", "b"),
            [_edge("a", "b", BLOCK_A, 0.6), _edge("a", "b", BLOCK_B, 0.6)],
        )
        assert genuine.sources == 2
        assert genuine.corroborated_confidence > doubled.corroborated_confidence

    def test_strongest_confidence_wins_within_one_block(self) -> None:
        c = Corroboration.from_edges(
            ("a", "depends_on", "b"),
            [_edge("a", "b", BLOCK_A, 0.2), _edge("a", "b", BLOCK_A, 0.9)],
        )
        assert c.confidences == (0.9,)


def _edge(subject: str, obj: str, block: str, confidence: float):
    from mind_mem.knowledge_graph import Edge, Predicate

    return Edge(
        subject=subject,
        predicate=Predicate.DEPENDS_ON,
        object=obj,
        source_block_id=block,
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# Over a real graph
# ---------------------------------------------------------------------------


class TestOverTheStore:
    def test_index_groups_a_claim_across_blocks(self, graph: KnowledgeGraph) -> None:
        graph.add_edge("starga", "depends_on", "mindc", source_block_id=BLOCK_A, confidence=0.6)
        graph.add_edge("starga", "depends_on", "mindc", source_block_id=BLOCK_B, confidence=0.6)
        graph.add_edge("starga", "depends_on", "llvm", source_block_id=BLOCK_C, confidence=0.8)
        index = graph.corroboration_index()
        assert index[("starga", "depends_on", "mindc")].sources == 2
        assert index[("starga", "depends_on", "llvm")].sources == 1
        assert (
            index[("starga", "depends_on", "mindc")].corroborated_confidence
            > index[("starga", "depends_on", "llvm")].corroborated_confidence
        )

    def test_corroboration_of_folds_aliases(self, graph: KnowledgeGraph) -> None:
        graph.add_edge("STARGA", "depends_on", "mindc", source_block_id=BLOCK_A, confidence=0.5)
        found = graph.corroboration_of("  starga ", "depends-on", "MINDC")
        assert found.sources == 1
        assert found.source_block_ids == (BLOCK_A,)

    def test_unknown_claim_scores_zero_without_minting(self, graph: KnowledgeGraph) -> None:
        before = graph.stats().entities
        found = graph.corroboration_of("nobody", "depends_on", "nothing")
        assert found.sources == 0
        assert found.corroborated_confidence == 0.0
        assert graph.stats().entities == before

    def test_single_source_claims_selects_exactly_the_singletons(self, graph: KnowledgeGraph) -> None:
        graph.add_edge("starga", "depends_on", "mindc", source_block_id=BLOCK_A, confidence=0.6)
        graph.add_edge("starga", "depends_on", "mindc", source_block_id=BLOCK_B, confidence=0.6)
        graph.add_edge("starga", "depends_on", "llvm", source_block_id=BLOCK_C, confidence=0.8)
        singles = graph.single_source_claims()
        assert [c.object for c in singles] == ["llvm"]
        # Positive control: the corroborated claim exists and is visible
        # to the same scan, so the exclusion above is a decision.
        assert ("starga", "depends_on", "mindc") in graph.corroboration_index()

    def test_computing_it_writes_nothing(self, graph: KnowledgeGraph) -> None:
        """Sidecar-only, taken to its end: there is no sidecar.

        A derived score that never lands on disk cannot enter the audit
        chain ungoverned, and cannot drift from the edges it summarises.
        """
        graph.add_edge("starga", "depends_on", "mindc", source_block_id=BLOCK_A, confidence=0.6)
        graph.add_edge("starga", "depends_on", "mindc", source_block_id=BLOCK_B, confidence=0.6)
        before = _db_fingerprint(graph._db_path)
        assert before[1], "positive control: the fingerprint sees the edge rows"

        graph.corroboration_index()
        graph.corroboration_of("starga", "depends_on", "mindc")
        graph.single_source_claims()

        after = _db_fingerprint(graph._db_path)
        assert after == before
        assert "edge_confidence" not in after[0]
        assert "corroboration" not in after[0]

    def test_repeat_computation_is_identical(self, graph: KnowledgeGraph) -> None:
        graph.add_edge("starga", "depends_on", "mindc", source_block_id=BLOCK_A, confidence=0.55)
        graph.add_edge("starga", "depends_on", "mindc", source_block_id=BLOCK_B, confidence=0.35)
        first = graph.corroboration_index()
        second = graph.corroboration_index()
        assert first == second
        assert list(first) == list(second)


# ---------------------------------------------------------------------------
# The consumer: an answer that weights its evidence
# ---------------------------------------------------------------------------


class TestGroundedAnswerWeighting:
    def test_triples_carry_the_corroboration(self, graph: KnowledgeGraph) -> None:
        graph.add_edge("starga", "depends_on", "mindc", source_block_id=BLOCK_A, confidence=0.6)
        graph.add_edge("starga", "depends_on", "mindc", source_block_id=BLOCK_B, confidence=0.6)
        context = build_context(graph, "starga", hops=1)
        assert {t.sources for t in context.triples} == {2}
        assert context.triples[0].corroborating_blocks == (BLOCK_A, BLOCK_B)
        assert context.triples[0].corroborated_confidence == pytest.approx(0.84)
        assert "sources=2" in context.triples[0].as_line()

    def test_ranked_order_prefers_corroboration_while_hops_keep_shape(self, graph: KnowledgeGraph) -> None:
        # hop 1: a single-source claim at 0.8. hop 2: a two-source claim.
        graph.add_edge("starga", "depends_on", "mindc", source_block_id=BLOCK_A, confidence=0.8)
        graph.add_edge("mindc", "depends_on", "llvm", source_block_id=BLOCK_B, confidence=0.6)
        graph.add_edge("mindc", "depends_on", "llvm", source_block_id=BLOCK_C, confidence=0.6)
        context = build_context(graph, "starga", hops=2)
        # Structure is still traversal order -- and the two-source claim
        # is two rows, because provenance is part of the edge key.
        assert [t.object for t in context.triples] == ["mindc", "llvm", "llvm"]
        assert [t.hop for t in context.triples] == [1, 2, 2]
        # ...and the weighting is a separate, total order: the
        # corroborated claim outranks the stronger single-source one.
        assert [t.object for t in context.ranked_triples] == ["llvm", "llvm", "mindc"]
        assert context.ranked_triples[0].sources == 2
        assert context.ranked_triples[-1].sources == 1

    def test_single_source_claims_are_flagged_as_a_gap(self, graph: KnowledgeGraph) -> None:
        graph.add_edge("starga", "depends_on", "mindc", source_block_id=BLOCK_A, confidence=0.9)
        context = build_context(graph, "starga", hops=1)
        assert GAP_SINGLE_SOURCE_CLAIM in [g.kind for g in context.gaps]

    def test_a_fully_corroborated_subgraph_raises_no_such_gap(self, graph: KnowledgeGraph) -> None:
        """Positive control: the gap is a finding, not a constant."""
        graph.add_edge("starga", "depends_on", "mindc", source_block_id=BLOCK_A, confidence=0.6)
        graph.add_edge("starga", "depends_on", "mindc", source_block_id=BLOCK_B, confidence=0.6)
        context = build_context(graph, "starga", hops=1)
        assert len(context.triples) == 2, "both rows of the claim are served"
        assert GAP_SINGLE_SOURCE_CLAIM not in [g.kind for g in context.gaps]

    def test_the_serialised_context_shows_the_weight_to_a_generator(self, graph: KnowledgeGraph) -> None:
        graph.add_edge("starga", "depends_on", "mindc", source_block_id=BLOCK_A, confidence=0.6)
        graph.add_edge("starga", "depends_on", "mindc", source_block_id=BLOCK_B, confidence=0.6)
        text = build_context(graph, "starga", hops=1).serialize()
        assert "sources=2" in text
        assert "conf=0.84" in text
