"""Tests for the v3.11.0 typed block-lineage graph (Pattern 3)."""

from __future__ import annotations

import pytest

from mind_mem.block_lineage import (
    ALLOWED_KINDS,
    KIND_DECAY,
    LINEAGE_DEPTH_CAP,
    LINEAGE_NODE_CAP,
    LineageEdge,
    LineageResult,
    add_block_edge,
    block_lineage,
    ensure_lineage_schema,
    lineage_adjacency,
)


@pytest.fixture
def workspace(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    return str(ws)


@pytest.mark.unit
class TestSchema:
    def test_ensure_lineage_schema_idempotent(self, workspace) -> None:
        ensure_lineage_schema(workspace)
        ensure_lineage_schema(workspace)  # second call is a no-op

    def test_kind_column_added(self, workspace) -> None:
        from mind_mem.retrieval_graph import _connect

        ensure_lineage_schema(workspace)
        conn = _connect(workspace)
        try:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(co_retrieval)").fetchall()}
            assert "kind" in cols
        finally:
            conn.close()


@pytest.mark.unit
class TestAddBlockEdge:
    def test_basic_edge(self, workspace) -> None:
        add_block_edge(workspace, "A", "B", "cites")
        result = block_lineage(workspace, "A")
        assert any(e.block_id == "B" and e.kind == "cites" for e in result.edges)

    def test_invalid_kind_rejected(self, workspace) -> None:
        with pytest.raises(ValueError, match="kind must be"):
            add_block_edge(workspace, "A", "B", "INVALID")

    def test_self_loop_rejected(self, workspace) -> None:
        with pytest.raises(ValueError, match="must differ"):
            add_block_edge(workspace, "A", "A", "cites")

    def test_empty_id_rejected(self, workspace) -> None:
        with pytest.raises(ValueError):
            add_block_edge(workspace, "", "B", "cites")

    def test_dedup_keeps_specific_kind_over_cooccurrence(self, workspace) -> None:
        # Pre-existing co-occurrence edge gets upgraded to a typed kind.
        add_block_edge(workspace, "A", "B", "cooccurrence")
        add_block_edge(workspace, "A", "B", "implements")
        edges = block_lineage(workspace, "A").edges
        kinds = {e.kind for e in edges if e.block_id == "B"}
        assert "implements" in kinds
        assert "cooccurrence" not in kinds


@pytest.mark.unit
class TestBlockLineage:
    def test_one_hop(self, workspace) -> None:
        add_block_edge(workspace, "A", "B", "cites")
        add_block_edge(workspace, "A", "C", "implements")
        result = block_lineage(workspace, "A", max_depth=1)
        assert {e.block_id for e in result.edges} == {"B", "C"}
        assert all(e.distance == 1 for e in result.edges)

    def test_two_hops(self, workspace) -> None:
        add_block_edge(workspace, "A", "B", "cites")
        add_block_edge(workspace, "B", "C", "cites")
        result = block_lineage(workspace, "A", max_depth=2)
        ids = {e.block_id: e.distance for e in result.edges}
        assert ids.get("B") == 1
        assert ids.get("C") == 2

    def test_depth_clamped_to_cap(self, workspace) -> None:
        add_block_edge(workspace, "A", "B", "cites")
        add_block_edge(workspace, "B", "C", "cites")
        add_block_edge(workspace, "C", "D", "cites")
        add_block_edge(workspace, "D", "E", "cites")
        # request depth=10, expect clamp at LINEAGE_DEPTH_CAP=3
        result = block_lineage(workspace, "A", max_depth=10)
        assert result.max_depth == LINEAGE_DEPTH_CAP
        ids = {e.block_id: e.distance for e in result.edges}
        assert "E" not in ids  # 4 hops is over the cap

    def test_cycle_does_not_loop(self, workspace) -> None:
        add_block_edge(workspace, "A", "B", "cites")
        add_block_edge(workspace, "B", "C", "cites")
        add_block_edge(workspace, "C", "A", "cites")
        result = block_lineage(workspace, "A", max_depth=3)
        # Visited set bounds traversal; A is not in edges (it's the root).
        assert all(e.block_id != "A" for e in result.edges)
        assert {e.block_id for e in result.edges} == {"B", "C"}

    def test_node_cap_truncates(self, workspace) -> None:
        for i in range(20):
            add_block_edge(workspace, "A", f"B{i:02d}", "cites")
        result = block_lineage(workspace, "A", max_depth=1, node_cap=5)
        assert result.truncated is True
        assert len(result.edges) == 5

    def test_kind_filter(self, workspace) -> None:
        add_block_edge(workspace, "A", "B", "cites")
        add_block_edge(workspace, "A", "C", "implements")
        result = block_lineage(workspace, "A", kind_filter="cites")
        ids = {e.block_id for e in result.edges}
        assert "B" in ids
        assert "C" not in ids

    def test_unknown_kind_filter_rejected(self, workspace) -> None:
        with pytest.raises(ValueError):
            block_lineage(workspace, "A", kind_filter="bogus")

    def test_empty_block_id_rejected(self, workspace) -> None:
        with pytest.raises(ValueError):
            block_lineage(workspace, "")

    def test_no_edges_returns_empty(self, workspace) -> None:
        result = block_lineage(workspace, "ISOLATED")
        assert result.edges == []
        assert result.truncated is False


@pytest.mark.unit
class TestKindDecay:
    def test_all_kinds_have_decay(self) -> None:
        for k in ALLOWED_KINDS:
            assert k in KIND_DECAY
            assert 0.0 < KIND_DECAY[k] <= 1.0

    def test_contradicts_fastest(self) -> None:
        # Contradiction propagates fastest because invalidating
        # information has the highest implied confidence.
        assert KIND_DECAY["contradicts"] == 1.0
        assert KIND_DECAY["contradicts"] > KIND_DECAY["cites"]
        assert KIND_DECAY["cites"] > KIND_DECAY["implements"]
        assert KIND_DECAY["implements"] > KIND_DECAY["refines"]

    def test_one_hop_confidence_uses_kind_multiplier(self, workspace) -> None:
        add_block_edge(workspace, "A", "X", "contradicts")
        add_block_edge(workspace, "A", "Y", "refines")
        result = block_lineage(workspace, "A", max_depth=1)
        by_id = {e.block_id: e.confidence for e in result.edges}
        assert by_id["X"] == KIND_DECAY["contradicts"]
        assert by_id["Y"] == KIND_DECAY["refines"]
        assert by_id["X"] > by_id["Y"]


@pytest.mark.unit
class TestAdjacency:
    def test_lineage_adjacency_mirrors_edges(self, workspace) -> None:
        add_block_edge(workspace, "A", "B", "cites")
        add_block_edge(workspace, "B", "C", "implements")
        adj = lineage_adjacency(workspace)
        assert "B" in adj.get("A", [])
        assert "A" in adj.get("B", [])
        assert "C" in adj.get("B", [])

    def test_adjacency_kind_filter(self, workspace) -> None:
        add_block_edge(workspace, "A", "B", "cites")
        add_block_edge(workspace, "A", "C", "implements")
        adj = lineage_adjacency(workspace, kind_filter="cites")
        assert "B" in adj.get("A", [])
        assert "C" not in adj.get("A", [])

    def test_propagator_consumes_adjacency(self, workspace) -> None:
        from mind_mem.staleness import propagate_staleness

        add_block_edge(workspace, "SEED", "A", "contradicts")
        add_block_edge(workspace, "A", "B", "contradicts")
        adj = lineage_adjacency(workspace)
        plan = propagate_staleness(["SEED"], adj, max_hops=2)
        assert plan.scores["SEED"] == 1.0
        assert plan.scores["A"] == 0.9
        assert plan.scores["B"] == 0.5


@pytest.mark.unit
class TestResultShape:
    def test_lineage_edge_to_dict(self) -> None:
        e = LineageEdge(block_id="X", kind="cites", distance=1, confidence=0.8)
        d = e.to_dict()
        assert d["block_id"] == "X"
        assert d["kind"] == "cites"
        assert d["distance"] == 1
        assert d["confidence"] == 0.8

    def test_lineage_result_to_dict(self) -> None:
        e = LineageEdge(block_id="X", kind="cites", distance=1, confidence=0.8)
        r = LineageResult(root="A", edges=[e], truncated=False, max_depth=3)
        d = r.to_dict()
        assert d["root"] == "A"
        assert d["count"] == 1
        assert d["edges"][0]["block_id"] == "X"


@pytest.mark.unit
class TestConstants:
    def test_depth_cap_is_3(self) -> None:
        assert LINEAGE_DEPTH_CAP == 3

    def test_node_cap_is_1000(self) -> None:
        assert LINEAGE_NODE_CAP == 1000

    def test_allowed_kinds_match_decay(self) -> None:
        assert set(KIND_DECAY.keys()) == ALLOWED_KINDS
