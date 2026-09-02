# Copyright 2026 STARGA, Inc.
"""Typed edge layer over the block store (roadmap §a).

Covers:
- The five first-class typed relations exist on the Predicate enum.
- Typed-predicate edge CRUD via the existing triple store.
- Legacy contradiction spellings subsume onto CONTRADICTS (back-compat).
- Wedge guardrail (Group H, load-bearing): a typed-edge write goes through
  a proposal — propose_edge stages, does NOT auto-commit; only approve_edge
  writes the source-of-truth graph.
- Deterministic proposal ids (no clock / rand) + idempotent restaging.
- Relationship-aware bucketing (relations_of) instead of a flat neighbour set.
"""

from __future__ import annotations

import pytest

from mind_mem.knowledge_graph import (
    PROPOSAL_APPLIED,
    PROPOSAL_REJECTED,
    PROPOSAL_STAGED,
    TYPED_RELATION_VALUES,
    TYPED_RELATIONS,
    EdgeProposal,
    KnowledgeGraph,
    Predicate,
    is_typed_relation,
)

FIVE = ("supports", "contradicts", "refines", "supersedes", "derived_from")


# Since 5.0.2 ``KnowledgeGraph.add_edge`` requires an open governance
# admission, exactly as ``BlockStore.write_block`` does. The ``admitted``
# fixture (tests/conftest.py) opens a REAL ``admit_proposal`` scope and
# writes a real chain entry -- there is no test-only bypass, and an
# invariant with one reserved for tests is not an invariant. The refusal
# itself is proven in tests/test_governed_signal_and_edge.py.
@pytest.fixture
def kg(tmp_path, admitted):
    with KnowledgeGraph(str(tmp_path / "kg.db")) as graph:
        yield graph


# ---------------------------------------------------------------------------
# The five typed relations
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTypedRelationSet:
    @pytest.mark.parametrize("value", FIVE)
    def test_predicate_member_exists(self, value) -> None:
        assert Predicate.from_str(value).value == value

    def test_refines_added(self) -> None:
        # The four others predate this work; refines is the new member.
        assert Predicate.REFINES.value == "refines"

    def test_typed_relations_are_exactly_the_five(self) -> None:
        assert {p.value for p in TYPED_RELATIONS} == set(FIVE)

    def test_typed_relation_values_lex_sorted(self) -> None:
        assert list(TYPED_RELATION_VALUES) == sorted(FIVE)

    @pytest.mark.parametrize("value", FIVE)
    def test_is_typed_relation_true(self, value) -> None:
        assert is_typed_relation(value)

    def test_is_typed_relation_false_for_non_typed(self) -> None:
        assert not is_typed_relation("related_to")
        assert not is_typed_relation("authored_by")

    def test_is_typed_relation_false_for_garbage(self) -> None:
        assert not is_typed_relation("not_a_predicate_at_all")


# ---------------------------------------------------------------------------
# Typed-predicate edge CRUD (existing triple store, not a parallel store)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTypedEdgeCRUD:
    @pytest.mark.parametrize("value", FIVE)
    def test_add_and_read_each_relation(self, kg, value) -> None:
        pred = Predicate.from_str(value)
        kg.add_edge("A", pred, "B", source_block_id="src-1", confidence=0.9)
        out = kg.edges_from("A", predicate=pred)
        assert len(out) == 1
        assert out[0].predicate.value == value
        assert out[0].confidence == 0.9

    def test_edges_to_reverse_lookup(self, kg) -> None:
        kg.add_edge("summary", Predicate.DERIVED_FROM, "source", source_block_id="s")
        incoming = kg.edges_to("source", predicate=Predicate.DERIVED_FROM)
        assert len(incoming) == 1
        assert incoming[0].subject == "summary"

    def test_refines_edge_roundtrip(self, kg) -> None:
        kg.add_edge("v2", Predicate.REFINES, "v1", source_block_id="blk")
        out = kg.edges_from("v2", predicate=Predicate.REFINES)
        assert len(out) == 1 and out[0].object == "v1"

    def test_duplicate_edge_idempotent(self, kg) -> None:
        kg.add_edge("A", Predicate.SUPPORTS, "B", source_block_id="s1")
        kg.add_edge("A", Predicate.SUPPORTS, "B", source_block_id="s1")
        assert len(kg.edges_from("A", predicate=Predicate.SUPPORTS)) == 1

    def test_predicate_filter_isolates_relation(self, kg) -> None:
        kg.add_edge("A", Predicate.SUPPORTS, "B", source_block_id="s1")
        kg.add_edge("A", Predicate.CONTRADICTS, "C", source_block_id="s2")
        sup = kg.edges_from("A", predicate=Predicate.SUPPORTS)
        assert {e.object for e in sup} == {"b"}


# ---------------------------------------------------------------------------
# Contradiction subsumption + back-compat
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestContradictionSubsumption:
    @pytest.mark.parametrize("legacy", ["contradiction", "contradicted_by", "contraindicates", "CONTRADICTION", "Contradicted-By"])
    def test_legacy_spellings_map_to_contradicts(self, legacy) -> None:
        assert Predicate.from_str(legacy) is Predicate.CONTRADICTS

    def test_legacy_write_lands_as_contradicts(self, kg) -> None:
        # A caller still using the old singular spelling writes one canonical edge.
        kg.add_edge("A", "contradiction", "B", source_block_id="s1")
        out = kg.edges_from("A", predicate=Predicate.CONTRADICTS)
        assert len(out) == 1
        assert out[0].predicate is Predicate.CONTRADICTS

    def test_new_and_legacy_collapse_to_one_edge(self, kg) -> None:
        kg.add_edge("A", Predicate.CONTRADICTS, "B", source_block_id="s1")
        kg.add_edge("A", "contradiction", "B", source_block_id="s1")
        assert len(kg.edges_from("A", predicate="contradicts")) == 1

    def test_stats_records_canonical_predicate_only(self, kg) -> None:
        kg.add_edge("A", "contradiction", "B", source_block_id="s1")
        stats = kg.stats()
        assert "contradicts" in stats.predicates
        assert "contradiction" not in stats.predicates


# ---------------------------------------------------------------------------
# Wedge guardrail — typed-edge writes are HITL proposals, never auto-commit
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestProposalGatedWrites:
    def test_propose_does_not_write_edge(self, kg) -> None:
        kg.propose_edge("A", Predicate.SUPPORTS, "B", source_block_id="s1")
        # Source-of-truth graph must be untouched by a proposal.
        assert kg.edges_from("A") == []
        assert kg.stats().edges == 0

    def test_propose_does_not_mint_entities(self, kg) -> None:
        kg.propose_edge("Brand New Entity", Predicate.SUPPORTS, "Another", source_block_id="s1")
        # Entity resolution is deferred to approval — staging touches nothing.
        assert kg.entities.lookup("Brand New Entity") is None
        assert kg.entities.lookup("Another") is None

    def test_propose_returns_staged(self, kg) -> None:
        prop = kg.propose_edge("A", Predicate.REFINES, "B", source_block_id="s1")
        assert isinstance(prop, EdgeProposal)
        assert prop.status == PROPOSAL_STAGED
        assert prop.predicate is Predicate.REFINES

    def test_approve_commits_edge(self, kg) -> None:
        prop = kg.propose_edge("A", Predicate.SUPPORTS, "B", source_block_id="s1", confidence=0.75)
        edge = kg.approve_edge(prop.proposal_id)
        assert edge.predicate is Predicate.SUPPORTS
        assert edge.confidence == 0.75
        out = kg.edges_from("A", predicate=Predicate.SUPPORTS)
        assert len(out) == 1
        assert kg.get_edge_proposal(prop.proposal_id).status == PROPOSAL_APPLIED

    def test_approve_resolves_entities_at_commit(self, kg) -> None:
        prop = kg.propose_edge("STARGA", Predicate.SUPPORTS, "MIND", source_block_id="s1")
        assert kg.entities.lookup("STARGA") is None  # not yet
        kg.approve_edge(prop.proposal_id)
        assert kg.entities.lookup("STARGA") is not None  # minted at approval

    def test_reject_never_writes(self, kg) -> None:
        prop = kg.propose_edge("A", Predicate.CONTRADICTS, "B", source_block_id="s1")
        rejected = kg.reject_edge(prop.proposal_id)
        assert rejected.status == PROPOSAL_REJECTED
        assert kg.stats().edges == 0

    def test_cannot_approve_rejected(self, kg) -> None:
        prop = kg.propose_edge("A", Predicate.CONTRADICTS, "B", source_block_id="s1")
        kg.reject_edge(prop.proposal_id)
        with pytest.raises(ValueError):
            kg.approve_edge(prop.proposal_id)

    def test_cannot_reject_applied(self, kg) -> None:
        prop = kg.propose_edge("A", Predicate.SUPPORTS, "B", source_block_id="s1")
        kg.approve_edge(prop.proposal_id)
        with pytest.raises(ValueError):
            kg.reject_edge(prop.proposal_id)

    def test_approve_is_idempotent(self, kg) -> None:
        prop = kg.propose_edge("A", Predicate.SUPPORTS, "B", source_block_id="s1")
        kg.approve_edge(prop.proposal_id)
        kg.approve_edge(prop.proposal_id)  # re-commit
        assert len(kg.edges_from("A", predicate=Predicate.SUPPORTS)) == 1

    def test_approve_unknown_raises(self, kg) -> None:
        with pytest.raises(KeyError):
            kg.approve_edge("EP-deadbeefdeadbeef")

    def test_contradiction_proposal_subsumes_legacy_spelling(self, kg) -> None:
        # The contradiction-edge work flows through the same gated path.
        prop = kg.propose_edge("A", "contradiction", "B", source_block_id="s1")
        assert prop.predicate is Predicate.CONTRADICTS
        edge = kg.approve_edge(prop.proposal_id)
        assert edge.predicate is Predicate.CONTRADICTS


@pytest.mark.unit
class TestProposalDeterminism:
    def test_proposal_id_deterministic(self, kg) -> None:
        p1 = kg.propose_edge("A", Predicate.SUPPORTS, "B", source_block_id="s1")
        p2 = kg.propose_edge("A", Predicate.SUPPORTS, "B", source_block_id="s1")
        assert p1.proposal_id == p2.proposal_id

    def test_restage_idempotent_single_row(self, kg) -> None:
        kg.propose_edge("A", Predicate.SUPPORTS, "B", source_block_id="s1")
        kg.propose_edge("A", Predicate.SUPPORTS, "B", source_block_id="s1")
        assert len(kg.list_edge_proposals()) == 1

    def test_id_stable_across_instances(self, tmp_path) -> None:
        path = str(tmp_path / "kg.db")
        with KnowledgeGraph(path) as g1:
            pid1 = g1.propose_edge("A", Predicate.SUPPORTS, "B", source_block_id="s1").proposal_id
        with KnowledgeGraph(path) as g2:
            pid2 = g2.propose_edge("A", Predicate.SUPPORTS, "B", source_block_id="s1").proposal_id
        assert pid1 == pid2

    def test_id_canonicalises_surfaces(self, kg) -> None:
        # Casing/whitespace differences that canonicalise identically share an id.
        a = kg.propose_edge("STARGA", Predicate.SUPPORTS, "MIND", source_block_id="s1")
        b = kg.propose_edge("  starga ", Predicate.SUPPORTS, "mind", source_block_id="s1")
        assert a.proposal_id == b.proposal_id

    def test_restage_does_not_reset_status(self, kg) -> None:
        prop = kg.propose_edge("A", Predicate.SUPPORTS, "B", source_block_id="s1")
        kg.approve_edge(prop.proposal_id)
        again = kg.propose_edge("A", Predicate.SUPPORTS, "B", source_block_id="s1")
        assert again.status == PROPOSAL_APPLIED  # INSERT OR IGNORE kept the applied row

    def test_list_filter_by_status(self, kg) -> None:
        p1 = kg.propose_edge("A", Predicate.SUPPORTS, "B", source_block_id="s1")
        p2 = kg.propose_edge("C", Predicate.CONTRADICTS, "D", source_block_id="s2")
        kg.approve_edge(p1.proposal_id)
        staged = kg.list_edge_proposals(status=PROPOSAL_STAGED)
        assert [p.proposal_id for p in staged] == [p2.proposal_id]


# ---------------------------------------------------------------------------
# Relationship-aware view (not flat fusion)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRelationsOf:
    def test_buckets_have_all_five_keys(self, kg) -> None:
        buckets = kg.relations_of("A")
        assert set(buckets.keys()) == set(FIVE)

    def test_edges_land_in_correct_bucket(self, kg) -> None:
        kg.add_edge("A", Predicate.SUPPORTS, "B", source_block_id="s1")
        kg.add_edge("A", Predicate.CONTRADICTS, "C", source_block_id="s2")
        kg.add_edge("A", Predicate.REFINES, "D", source_block_id="s3")
        buckets = kg.relations_of("A")
        assert {e.object for e in buckets["supports"]} == {"b"}
        assert {e.object for e in buckets["contradicts"]} == {"c"}
        assert {e.object for e in buckets["refines"]} == {"d"}
        assert buckets["supersedes"] == []

    def test_incoming_direction(self, kg) -> None:
        kg.add_edge("child", Predicate.DERIVED_FROM, "parent", source_block_id="s1")
        incoming = kg.relations_of("parent", direction="incoming")
        assert {e.subject for e in incoming["derived_from"]} == {"child"}
        outgoing = kg.relations_of("parent", direction="outgoing")
        assert outgoing["derived_from"] == []

    def test_bucket_ordering_deterministic(self, kg) -> None:
        kg.add_edge("A", Predicate.SUPPORTS, "B", source_block_id="s1", confidence=0.4)
        kg.add_edge("A", Predicate.SUPPORTS, "C", source_block_id="s2", confidence=0.9)
        buckets = kg.relations_of("A")
        confs = [e.confidence for e in buckets["supports"]]
        assert confs == sorted(confs, reverse=True)  # highest confidence first

    def test_invalid_direction_raises(self, kg) -> None:
        with pytest.raises(ValueError):
            kg.relations_of("A", direction="sideways")
