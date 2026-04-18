# Copyright 2026 STARGA, Inc.
"""Tests for the SQLite-backed knowledge graph (v2.2.0)."""

from __future__ import annotations

import tempfile
import threading
from pathlib import Path

import pytest

from mind_mem.knowledge_graph import (
    KnowledgeGraph,
    Predicate,
)


@pytest.fixture()
def graph():
    with tempfile.TemporaryDirectory() as td:
        yield KnowledgeGraph(str(Path(td) / "kg.db"))


# ---------------------------------------------------------------------------
# Predicate
# ---------------------------------------------------------------------------


class TestPredicate:
    def test_values_are_snake_case(self) -> None:
        assert Predicate.AUTHORED_BY.value == "authored_by"
        assert Predicate.DEPENDS_ON.value == "depends_on"

    def test_from_str_hyphen_tolerant(self) -> None:
        assert Predicate.from_str("depends-on") is Predicate.DEPENDS_ON

    def test_from_str_case_insensitive(self) -> None:
        assert Predicate.from_str("Authored_By") is Predicate.AUTHORED_BY

    def test_from_str_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown predicate"):
            Predicate.from_str("bogus")


# ---------------------------------------------------------------------------
# Entity registry
# ---------------------------------------------------------------------------


class TestEntityRegistry:
    def test_resolve_new_entity_returns_canonical(self, graph: KnowledgeGraph) -> None:
        eid = graph.entities.resolve("STARGA Inc")
        assert eid == "starga inc"

    def test_resolve_case_insensitive_same_id(self, graph: KnowledgeGraph) -> None:
        a = graph.entities.resolve("STARGA Inc")
        b = graph.entities.resolve("starga   inc")
        c = graph.entities.resolve("Starga Inc")
        assert a == b == c

    def test_empty_name_rejected(self, graph: KnowledgeGraph) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            graph.entities.resolve("   ")

    def test_add_alias_binds_to_existing(self, graph: KnowledgeGraph) -> None:
        eid = graph.entities.resolve("STARGA Inc")
        graph.entities.add_alias("STARGA", eid)
        # Lookup via alias should resolve to the same id.
        assert graph.entities.resolve("STARGA") == eid

    def test_add_alias_unknown_entity_raises(self, graph: KnowledgeGraph) -> None:
        with pytest.raises(ValueError, match="unknown entity_id"):
            graph.entities.add_alias("x", "no-such-entity")

    def test_aliases_for_returns_sorted(self, graph: KnowledgeGraph) -> None:
        eid = graph.entities.resolve("STARGA Inc")
        graph.entities.add_alias("STARGA", eid)
        graph.entities.add_alias("Starga", eid)
        aliases = graph.entities.aliases_for(eid)
        # Both aliases canonicalise to "starga" so only one distinct.
        assert "starga" in aliases
        assert "starga inc" in aliases


# ---------------------------------------------------------------------------
# Edge CRUD
# ---------------------------------------------------------------------------


class TestEdges:
    def test_add_edge_resolves_endpoints(self, graph: KnowledgeGraph) -> None:
        e = graph.add_edge(
            "Alice",
            Predicate.AUTHORED_BY,
            "Project mind-mem",
            source_block_id="D-001",
        )
        assert e.subject == "alice"
        assert e.object == "project mind-mem"
        assert e.predicate is Predicate.AUTHORED_BY
        assert e.source_block_id == "D-001"

    def test_add_edge_string_predicate(self, graph: KnowledgeGraph) -> None:
        e = graph.add_edge(
            "A",
            "depends_on",
            "B",
            source_block_id="D-001",
        )
        assert e.predicate is Predicate.DEPENDS_ON

    def test_add_edge_empty_source_rejected(self, graph: KnowledgeGraph) -> None:
        with pytest.raises(ValueError, match="source_block_id"):
            graph.add_edge("A", Predicate.DEPENDS_ON, "B", source_block_id="")

    def test_duplicate_edge_is_idempotent(self, graph: KnowledgeGraph) -> None:
        for _ in range(3):
            graph.add_edge("A", Predicate.DEPENDS_ON, "B", source_block_id="D-001")
        assert graph.stats().edges == 1

    def test_confidence_out_of_range_rejected(self, graph: KnowledgeGraph) -> None:
        with pytest.raises(ValueError, match="confidence"):
            graph.add_edge(
                "A",
                Predicate.DEPENDS_ON,
                "B",
                source_block_id="D-001",
                confidence=1.5,
            )


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------


class TestQueries:
    def test_edges_from_filters_by_subject(self, graph: KnowledgeGraph) -> None:
        graph.add_edge("A", Predicate.DEPENDS_ON, "B", source_block_id="D-001")
        graph.add_edge("A", Predicate.AUTHORED_BY, "Alice", source_block_id="D-002")
        graph.add_edge("C", Predicate.DEPENDS_ON, "B", source_block_id="D-003")
        out = graph.edges_from("A")
        assert len(out) == 2

    def test_edges_from_predicate_filter(self, graph: KnowledgeGraph) -> None:
        graph.add_edge("A", Predicate.DEPENDS_ON, "B", source_block_id="D-001")
        graph.add_edge("A", Predicate.AUTHORED_BY, "Alice", source_block_id="D-002")
        out = graph.edges_from("A", predicate=Predicate.DEPENDS_ON)
        assert len(out) == 1
        assert out[0].object == "b"

    def test_edges_to_filters_by_object(self, graph: KnowledgeGraph) -> None:
        graph.add_edge("A", Predicate.DEPENDS_ON, "B", source_block_id="D-001")
        graph.add_edge("C", Predicate.DEPENDS_ON, "B", source_block_id="D-002")
        graph.add_edge("D", Predicate.DEPENDS_ON, "X", source_block_id="D-003")
        out = graph.edges_to("B")
        assert len(out) == 2

    def test_expired_edges_hidden_by_default(self, graph: KnowledgeGraph) -> None:
        graph.add_edge(
            "A",
            Predicate.DEPENDS_ON,
            "B",
            source_block_id="D-001",
            valid_until="2020-01-01T00:00:00Z",
        )
        assert graph.edges_from("A") == []
        assert len(graph.edges_from("A", include_expired=True)) == 1

    def test_fractional_second_valid_until_honoured(self, graph: KnowledgeGraph) -> None:
        """Audit regression: ASCII string compare breaks on `.999Z` suffixes."""
        # Expires ~50 years in the future with fractional seconds.
        graph.add_edge(
            "A",
            Predicate.DEPENDS_ON,
            "B",
            source_block_id="D-001",
            valid_until="2076-01-01T00:00:00.999Z",
        )
        # Must still return the edge — naive string compare would
        # reject it because '.' < 'Z' in ASCII.
        assert len(graph.edges_from("A")) == 1

    def test_malformed_timestamp_rejected(self, graph: KnowledgeGraph) -> None:
        with pytest.raises(ValueError):
            graph.add_edge(
                "A",
                Predicate.DEPENDS_ON,
                "B",
                source_block_id="D-001",
                valid_until="not-a-date",
            )

    def test_valid_until_before_valid_from_rejected(self, graph: KnowledgeGraph) -> None:
        with pytest.raises(ValueError, match="valid_until"):
            graph.add_edge(
                "A",
                Predicate.DEPENDS_ON,
                "B",
                source_block_id="D-001",
                valid_from="2030-01-01T00:00:00Z",
                valid_until="2020-01-01T00:00:00Z",
            )

    def test_metadata_with_non_json_types_accepted(self, graph: KnowledgeGraph) -> None:
        """Audit regression: datetime / set must not crash json.dumps."""
        from datetime import datetime as _dt

        graph.add_edge(
            "A",
            Predicate.DEPENDS_ON,
            "B",
            source_block_id="D-001",
            metadata={"extracted_at": _dt(2026, 4, 13)},
        )
        edges = graph.edges_from("A")
        assert len(edges) == 1
        # Value is stringified by json's default=str — safe persistence.
        assert "extracted_at" in edges[0].metadata

    def test_context_manager_closes_connection(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = str(Path(td) / "kg.db")
            with KnowledgeGraph(path) as kg:
                kg.add_edge("A", Predicate.DEPENDS_ON, "B", source_block_id="D-001")
            # After exit, a fresh KG over the same file sees the edge.
            kg2 = KnowledgeGraph(path)
            try:
                assert kg2.stats().edges == 1
            finally:
                kg2.close()


# ---------------------------------------------------------------------------
# Traversal
# ---------------------------------------------------------------------------


class TestTraversal:
    def _chain(self, graph: KnowledgeGraph) -> None:
        graph.add_edge("A", Predicate.DEPENDS_ON, "B", source_block_id="D-001")
        graph.add_edge("B", Predicate.DEPENDS_ON, "C", source_block_id="D-002")
        graph.add_edge("C", Predicate.DEPENDS_ON, "D", source_block_id="D-003")

    def test_one_hop_outgoing(self, graph: KnowledgeGraph) -> None:
        self._chain(graph)
        out = graph.neighbors("A", depth=1)
        assert [x["entity"] for x in out] == ["b"]

    def test_three_hop_reaches_d(self, graph: KnowledgeGraph) -> None:
        self._chain(graph)
        out = graph.neighbors("A", depth=3)
        entities = {x["entity"] for x in out}
        assert entities == {"b", "c", "d"}

    def test_depth_zero_returns_empty(self, graph: KnowledgeGraph) -> None:
        self._chain(graph)
        assert graph.neighbors("A", depth=0) == []

    def test_depth_capped_at_eight(self, graph: KnowledgeGraph) -> None:
        # Depth argument is silently capped at 8 for DoS safety.
        # Chain only has 3 nodes; depth=1000 must not loop forever.
        self._chain(graph)
        out = graph.neighbors("A", depth=1000)
        assert len(out) == 3

    def test_predicate_filter(self, graph: KnowledgeGraph) -> None:
        graph.add_edge("A", Predicate.DEPENDS_ON, "B", source_block_id="D-001")
        graph.add_edge("A", Predicate.AUTHORED_BY, "Alice", source_block_id="D-002")
        out = graph.neighbors("A", depth=1, predicate=Predicate.AUTHORED_BY)
        assert [x["entity"] for x in out] == ["alice"]

    def test_incoming_direction(self, graph: KnowledgeGraph) -> None:
        graph.add_edge("A", Predicate.DEPENDS_ON, "B", source_block_id="D-001")
        out = graph.neighbors("B", depth=1, direction="incoming")
        assert [x["entity"] for x in out] == ["a"]

    def test_both_directions(self, graph: KnowledgeGraph) -> None:
        graph.add_edge("A", Predicate.DEPENDS_ON, "B", source_block_id="D-001")
        graph.add_edge("C", Predicate.DEPENDS_ON, "B", source_block_id="D-002")
        out = graph.neighbors("B", depth=1, direction="both")
        entities = {x["entity"] for x in out}
        assert entities == {"a", "c"}

    def test_invalid_direction_rejected(self, graph: KnowledgeGraph) -> None:
        with pytest.raises(ValueError, match="direction"):
            graph.neighbors("A", direction="sideways")

    def test_cycle_does_not_loop(self, graph: KnowledgeGraph) -> None:
        graph.add_edge("A", Predicate.DEPENDS_ON, "B", source_block_id="D-001")
        graph.add_edge("B", Predicate.DEPENDS_ON, "A", source_block_id="D-002")
        out = graph.neighbors("A", depth=5)
        assert len(out) == 1  # B


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


class TestStats:
    def test_empty_graph_zero_counts(self, graph: KnowledgeGraph) -> None:
        s = graph.stats()
        assert s.entities == 0
        assert s.edges == 0

    def test_stats_populated(self, graph: KnowledgeGraph) -> None:
        graph.add_edge("A", Predicate.DEPENDS_ON, "B", source_block_id="D-001")
        graph.add_edge("A", Predicate.AUTHORED_BY, "Alice", source_block_id="D-002")
        s = graph.stats()
        assert s.entities == 3  # A, B, Alice
        assert s.edges == 2
        assert s.predicates == {"depends_on": 1, "authored_by": 1}

    def test_orphan_count(self, graph: KnowledgeGraph) -> None:
        graph.entities.resolve("Lonely")
        graph.add_edge("A", Predicate.DEPENDS_ON, "B", source_block_id="D-001")
        s = graph.stats()
        assert s.orphan_entities == 1


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


class TestConcurrency:
    def test_concurrent_add_edge_no_corruption(self, graph: KnowledgeGraph) -> None:
        errors: list[BaseException] = []
        barrier = threading.Barrier(8)

        def worker(tid: int) -> None:
            try:
                barrier.wait()
                for i in range(50):
                    graph.add_edge(
                        f"subj-{tid}",
                        Predicate.DEPENDS_ON,
                        f"obj-{i}",
                        source_block_id=f"D-{tid}-{i}",
                    )
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        assert graph.stats().edges == 8 * 50
