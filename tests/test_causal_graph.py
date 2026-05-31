"""Tests for mind-mem temporal causal graph (causal_graph.py)."""

import pytest

from mind_mem.causal_graph import (
    EDGE_CONTRADICTS,
    EDGE_DEPENDS_ON,
    EDGE_EXTENDS,
    EDGE_INFORMS,
    EDGE_SUPERSEDES,
    CausalGraph,
)


@pytest.fixture
def workspace(tmp_path):
    return str(tmp_path)


@pytest.fixture
def graph(workspace):
    return CausalGraph(workspace)


class TestAddEdge:
    def test_basic_edge(self, graph):
        edge = graph.add_edge("D-002", "D-001", EDGE_DEPENDS_ON)
        assert edge.source_id == "D-002"
        assert edge.target_id == "D-001"
        assert edge.edge_type == EDGE_DEPENDS_ON

    def test_invalid_type_raises(self, graph):
        with pytest.raises(ValueError, match="Invalid edge type"):
            graph.add_edge("D-002", "D-001", "invalid_type")

    def test_self_loop_raises(self, graph):
        with pytest.raises(ValueError, match="Self-loops"):
            graph.add_edge("D-001", "D-001", EDGE_DEPENDS_ON)

    def test_all_edge_types(self, graph):
        for i, etype in enumerate([EDGE_DEPENDS_ON, EDGE_SUPERSEDES, EDGE_INFORMS, EDGE_CONTRADICTS, EDGE_EXTENDS]):
            edge = graph.add_edge(f"A-{i}", f"B-{i}", etype)
            assert edge.edge_type == etype

    def test_upsert_on_conflict(self, graph):
        graph.add_edge("D-002", "D-001", EDGE_DEPENDS_ON, weight=1.0)
        graph.add_edge("D-002", "D-001", EDGE_DEPENDS_ON, weight=2.0)
        edges = graph.all_edges()
        assert len(edges) == 1
        assert edges[0].weight == 2.0


class TestCycleDetection:
    def test_direct_cycle_blocked(self, graph):
        graph.add_edge("A", "B", EDGE_DEPENDS_ON)
        with pytest.raises(ValueError, match="cycle"):
            graph.add_edge("B", "A", EDGE_DEPENDS_ON)

    def test_indirect_cycle_blocked(self, graph):
        graph.add_edge("A", "B", EDGE_DEPENDS_ON)
        graph.add_edge("B", "C", EDGE_DEPENDS_ON)
        with pytest.raises(ValueError, match="cycle"):
            graph.add_edge("C", "A", EDGE_DEPENDS_ON)

    def test_no_false_positive(self, graph):
        graph.add_edge("A", "B", EDGE_DEPENDS_ON)
        graph.add_edge("C", "B", EDGE_DEPENDS_ON)
        # A→B and C→B, adding A→C should be fine (no cycle)
        edge = graph.add_edge("A", "C", EDGE_DEPENDS_ON)
        assert edge.source_id == "A"


class TestDependencies:
    def test_dependents(self, graph):
        graph.add_edge("D-002", "D-001", EDGE_DEPENDS_ON)
        graph.add_edge("D-003", "D-001", EDGE_DEPENDS_ON)

        deps = graph.dependents("D-001")
        assert len(deps) == 2
        dep_sources = {d.source_id for d in deps}
        assert dep_sources == {"D-002", "D-003"}

    def test_dependencies(self, graph):
        graph.add_edge("D-003", "D-001", EDGE_DEPENDS_ON)
        graph.add_edge("D-003", "D-002", EDGE_DEPENDS_ON)

        deps = graph.dependencies("D-003")
        assert len(deps) == 2
        dep_targets = {d.target_id for d in deps}
        assert dep_targets == {"D-001", "D-002"}


class TestCausalChain:
    def test_linear_chain(self, graph):
        graph.add_edge("C", "B", EDGE_DEPENDS_ON)
        graph.add_edge("B", "A", EDGE_DEPENDS_ON)

        chains = graph.causal_chain("C")
        assert len(chains) == 1
        assert chains[0] == ["C", "B", "A"]

    def test_branching_chain(self, graph):
        graph.add_edge("D", "B", EDGE_DEPENDS_ON)
        graph.add_edge("D", "C", EDGE_DEPENDS_ON)
        graph.add_edge("B", "A", EDGE_DEPENDS_ON)

        chains = graph.causal_chain("D")
        assert len(chains) == 2

    def test_root_node(self, graph):
        chains = graph.causal_chain("A")
        assert chains == [["A"]]

    def test_max_depth(self, graph):
        # Build a long chain: M→L→K→J→I→H→G→F→E→D→C→B→A
        prev = "A"
        for c in "BCDEFGHIJKLM":
            graph.add_edge(c, prev, EDGE_DEPENDS_ON)
            prev = c

        # Without depth limit (default max_depth=10), chain traverses deeply
        full_chains = graph.causal_chain("M", max_depth=20)
        assert len(full_chains) == 1
        assert len(full_chains[0]) == 13  # M + 12 ancestors to A

        # With depth limit, chain is truncated
        limited = graph.causal_chain("M", max_depth=3)
        assert len(limited) == 1
        assert len(limited[0]) < 13  # Truncated from full chain


class TestStaleness:
    def test_propagate_staleness(self, graph):
        graph.add_edge("B", "A", EDGE_DEPENDS_ON)
        graph.add_edge("C", "A", EDGE_DEPENDS_ON)
        graph.add_edge("D", "B", EDGE_DEPENDS_ON)

        stale = graph.propagate_staleness("A")
        assert set(stale) == {"B", "C", "D"}

    def test_get_stale_blocks(self, graph):
        graph.add_edge("B", "A", EDGE_DEPENDS_ON)
        graph.propagate_staleness("A", reason="A was updated")

        stale = graph.get_stale_blocks()
        assert len(stale) == 1
        assert stale[0]["block_id"] == "B"
        assert "A was updated" in stale[0]["reason"]

    def test_clear_staleness(self, graph):
        graph.add_edge("B", "A", EDGE_DEPENDS_ON)
        graph.propagate_staleness("A")

        assert graph.clear_staleness("B")
        assert graph.get_stale_blocks() == []

    def test_no_dependents_no_staleness(self, graph):
        stale = graph.propagate_staleness("isolated-block")
        assert stale == []


class TestRemoveEdge:
    def test_remove_existing(self, graph):
        graph.add_edge("A", "B", EDGE_DEPENDS_ON)
        assert graph.remove_edge("A", "B", EDGE_DEPENDS_ON)
        assert graph.all_edges() == []

    def test_remove_nonexistent(self, graph):
        assert not graph.remove_edge("X", "Y", EDGE_DEPENDS_ON)


class TestSummary:
    def test_empty_graph(self, graph):
        s = graph.summary()
        assert s["total_edges"] == 0
        assert s["unique_nodes"] == 0

    def test_populated_graph(self, graph):
        graph.add_edge("A", "B", EDGE_DEPENDS_ON)
        graph.add_edge("C", "B", EDGE_SUPERSEDES)
        graph.add_edge("D", "A", EDGE_INFORMS)

        s = graph.summary()
        assert s["total_edges"] == 3
        assert s["unique_nodes"] == 4
        assert s["edges_by_type"][EDGE_DEPENDS_ON] == 1
        assert s["edges_by_type"][EDGE_SUPERSEDES] == 1
