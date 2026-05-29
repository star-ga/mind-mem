"""Tests for Group H typed-edge additions: supports, derived_from, edge_aware_boost.

Covers:
- New edge kinds present in ALLOWED_KINDS and KIND_DECAY.
- Decay ordering invariants for the new kinds.
- Round-trip write/BFS for supports and derived_from.
- edge_aware_boost is zero when weight=0 (behavior-preserving default).
- edge_aware_boost returns correct additive values when weight is non-zero.
- knowledge_graph.Predicate has SUPPORTS and DERIVED_FROM members.
"""

from __future__ import annotations

import pytest

from mind_mem.block_lineage import (
    ALLOWED_KINDS,
    EDGE_BOOST_WEIGHT,
    KIND_DECAY,
    add_block_edge,
    block_lineage,
    edge_aware_boost,
)
from mind_mem.knowledge_graph import KnowledgeGraph, Predicate


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def workspace(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    return str(ws)


# ---------------------------------------------------------------------------
# New kinds in vocabulary
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNewKindsInVocabulary:
    def test_supports_in_allowed_kinds(self) -> None:
        assert "supports" in ALLOWED_KINDS

    def test_derived_from_in_allowed_kinds(self) -> None:
        assert "derived_from" in ALLOWED_KINDS

    def test_supports_in_kind_decay(self) -> None:
        assert "supports" in KIND_DECAY
        assert 0.0 < KIND_DECAY["supports"] <= 1.0

    def test_derived_from_in_kind_decay(self) -> None:
        assert "derived_from" in KIND_DECAY
        assert 0.0 < KIND_DECAY["derived_from"] <= 1.0

    def test_all_allowed_kinds_covered_in_decay(self) -> None:
        assert set(KIND_DECAY.keys()) == ALLOWED_KINDS

    def test_supports_in_edge_boost_weight(self) -> None:
        assert "supports" in EDGE_BOOST_WEIGHT
        assert EDGE_BOOST_WEIGHT["supports"] > 0.0

    def test_derived_from_in_edge_boost_weight(self) -> None:
        assert "derived_from" in EDGE_BOOST_WEIGHT
        assert EDGE_BOOST_WEIGHT["derived_from"] > 0.0

    def test_contradicts_boost_is_zero(self) -> None:
        assert EDGE_BOOST_WEIGHT["contradicts"] == 0.0


# ---------------------------------------------------------------------------
# Decay ordering for new kinds
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNewKindDecayOrdering:
    def test_supports_stronger_than_derived_from(self) -> None:
        assert KIND_DECAY["supports"] > KIND_DECAY["derived_from"]

    def test_supports_weaker_than_cites(self) -> None:
        # supports is corroborating evidence; cites is a direct reference.
        assert KIND_DECAY["supports"] < KIND_DECAY["cites"]

    def test_derived_from_not_weaker_than_refines(self) -> None:
        # A derivation is as informative as a refinement for staleness.
        assert KIND_DECAY["derived_from"] >= KIND_DECAY["refines"]


# ---------------------------------------------------------------------------
# Round-trip write / BFS for the new kinds
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSupportsEdgeRoundTrip:
    def test_add_supports_edge(self, workspace) -> None:
        add_block_edge(workspace, "A", "B", "supports")
        result = block_lineage(workspace, "A", max_depth=1)
        found = [e for e in result.edges if e.block_id == "B"]
        assert found, "expected B in lineage of A"
        assert found[0].kind == "supports"

    def test_supports_confidence_uses_kind_decay(self, workspace) -> None:
        add_block_edge(workspace, "A", "B", "supports")
        result = block_lineage(workspace, "A", max_depth=1)
        found = {e.block_id: e.confidence for e in result.edges}
        assert found["B"] == KIND_DECAY["supports"]

    def test_kind_filter_supports(self, workspace) -> None:
        add_block_edge(workspace, "A", "B", "supports")
        add_block_edge(workspace, "A", "C", "cites")
        result = block_lineage(workspace, "A", kind_filter="supports")
        ids = {e.block_id for e in result.edges}
        assert "B" in ids
        assert "C" not in ids


@pytest.mark.unit
class TestDerivedFromEdgeRoundTrip:
    def test_add_derived_from_edge(self, workspace) -> None:
        add_block_edge(workspace, "S", "O", "derived_from")
        result = block_lineage(workspace, "S", max_depth=1)
        found = [e for e in result.edges if e.block_id == "O"]
        assert found, "expected O in lineage of S"
        assert found[0].kind == "derived_from"

    def test_derived_from_confidence_uses_kind_decay(self, workspace) -> None:
        add_block_edge(workspace, "S", "O", "derived_from")
        result = block_lineage(workspace, "S", max_depth=1)
        found = {e.block_id: e.confidence for e in result.edges}
        assert found["O"] == KIND_DECAY["derived_from"]

    def test_kind_filter_derived_from(self, workspace) -> None:
        add_block_edge(workspace, "S", "O", "derived_from")
        add_block_edge(workspace, "S", "P", "supports")
        result = block_lineage(workspace, "S", kind_filter="derived_from")
        ids = {e.block_id for e in result.edges}
        assert "O" in ids
        assert "P" not in ids

    def test_derived_from_multi_hop(self, workspace) -> None:
        add_block_edge(workspace, "A", "B", "derived_from")
        add_block_edge(workspace, "B", "C", "derived_from")
        result = block_lineage(workspace, "A", max_depth=2)
        ids = {e.block_id: e.distance for e in result.edges}
        assert ids.get("B") == 1
        assert ids.get("C") == 2


# ---------------------------------------------------------------------------
# edge_aware_boost — behavior-preserving default (weight=0)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEdgeAwareBoostDefaultOff:
    def test_zero_weight_returns_empty(self, workspace) -> None:
        add_block_edge(workspace, "X", "Y", "supports")
        boosts = edge_aware_boost(workspace, ["Y"], weight=0.0)
        assert boosts == {}

    def test_empty_block_ids_returns_empty(self, workspace) -> None:
        add_block_edge(workspace, "X", "Y", "supports")
        boosts = edge_aware_boost(workspace, [], weight=0.1)
        assert boosts == {}

    def test_no_edges_zero_boost(self, workspace) -> None:
        boosts = edge_aware_boost(workspace, ["ISOLATED"], weight=0.1)
        assert boosts.get("ISOLATED", 0.0) == 0.0


# ---------------------------------------------------------------------------
# edge_aware_boost — non-zero weight behaviour
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEdgeAwareBoostOptIn:
    def test_supports_incoming_edge_produces_boost(self, workspace) -> None:
        add_block_edge(workspace, "A", "TARGET", "supports")
        boosts = edge_aware_boost(workspace, ["TARGET"], weight=1.0)
        assert boosts.get("TARGET", 0.0) > 0.0

    def test_derived_from_incoming_edge_produces_boost(self, workspace) -> None:
        add_block_edge(workspace, "A", "TARGET", "derived_from")
        boosts = edge_aware_boost(workspace, ["TARGET"], weight=1.0)
        assert boosts.get("TARGET", 0.0) > 0.0

    def test_contradicts_incoming_edge_zero_boost(self, workspace) -> None:
        add_block_edge(workspace, "A", "TARGET", "contradicts")
        boosts = edge_aware_boost(workspace, ["TARGET"], weight=1.0)
        # contradicts contributes 0 — should not inflate target score
        assert boosts.get("TARGET", 0.0) == 0.0

    def test_weight_scales_boost_linearly(self, workspace) -> None:
        add_block_edge(workspace, "A", "TARGET", "supports")
        b1 = edge_aware_boost(workspace, ["TARGET"], weight=0.1).get("TARGET", 0.0)
        b2 = edge_aware_boost(workspace, ["TARGET"], weight=0.2).get("TARGET", 0.0)
        assert abs(b2 - 2 * b1) < 1e-9

    def test_multiple_incoming_edges_accumulate(self, workspace) -> None:
        add_block_edge(workspace, "A", "TARGET", "supports")
        add_block_edge(workspace, "B", "TARGET", "supports")
        boosts_single_weight = edge_aware_boost(workspace, ["TARGET"], weight=1.0).get("TARGET", 0.0)
        assert boosts_single_weight > EDGE_BOOST_WEIGHT["supports"]  # > one edge

    def test_only_requested_ids_returned(self, workspace) -> None:
        add_block_edge(workspace, "A", "B", "supports")
        add_block_edge(workspace, "A", "C", "supports")
        boosts = edge_aware_boost(workspace, ["B"], weight=1.0)
        assert "B" in boosts
        assert "C" not in boosts

    def test_no_spurious_keys_for_source_blocks(self, workspace) -> None:
        add_block_edge(workspace, "SOURCE", "DEST", "supports")
        boosts = edge_aware_boost(workspace, ["SOURCE", "DEST"], weight=1.0)
        # SOURCE has outgoing edge, not incoming; should not receive a boost
        assert boosts.get("SOURCE", 0.0) == 0.0
        assert boosts.get("DEST", 0.0) > 0.0

    def test_supports_boost_greater_than_derived_from(self, workspace) -> None:
        add_block_edge(workspace, "A", "T1", "supports")
        add_block_edge(workspace, "B", "T2", "derived_from")
        b_sup = edge_aware_boost(workspace, ["T1"], weight=1.0).get("T1", 0.0)
        b_der = edge_aware_boost(workspace, ["T2"], weight=1.0).get("T2", 0.0)
        assert b_sup > b_der


# ---------------------------------------------------------------------------
# Predicate enum additions in knowledge_graph
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPredicateNewKinds:
    def test_supports_predicate_exists(self) -> None:
        assert Predicate.SUPPORTS.value == "supports"

    def test_derived_from_predicate_exists(self) -> None:
        assert Predicate.DERIVED_FROM.value == "derived_from"

    def test_from_str_supports(self) -> None:
        assert Predicate.from_str("supports") is Predicate.SUPPORTS

    def test_from_str_derived_from(self) -> None:
        assert Predicate.from_str("derived_from") is Predicate.DERIVED_FROM

    def test_from_str_hyphen_form_derived_from(self) -> None:
        assert Predicate.from_str("derived-from") is Predicate.DERIVED_FROM

    def test_knowledge_graph_add_edge_supports(self, tmp_path) -> None:
        with KnowledgeGraph(str(tmp_path / "kg.db")) as kg:
            edge = kg.add_edge(
                "block A",
                Predicate.SUPPORTS,
                "block B",
                source_block_id="src-001",
                confidence=0.9,
            )
            assert edge.predicate is Predicate.SUPPORTS
            edges = kg.edges_from("block A", predicate=Predicate.SUPPORTS)
            assert len(edges) == 1
            assert edges[0].confidence == 0.9

    def test_knowledge_graph_add_edge_derived_from(self, tmp_path) -> None:
        with KnowledgeGraph(str(tmp_path / "kg.db")) as kg:
            edge = kg.add_edge(
                "summary block",
                Predicate.DERIVED_FROM,
                "source block",
                source_block_id="src-002",
            )
            assert edge.predicate is Predicate.DERIVED_FROM
            edges = kg.edges_to("source block", predicate=Predicate.DERIVED_FROM)
            assert len(edges) == 1
