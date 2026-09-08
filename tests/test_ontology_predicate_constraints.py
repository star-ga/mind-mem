# Copyright 2026 STARGA, Inc.
"""What an edge may CONNECT, not just what a node may carry.

``EntityType`` constrained node properties. Nothing constrained edges, so the
typed triple store accepted ``DOG --works_at--> MICROSOFT`` exactly as readily
as a true statement: the predicate vocabulary said which verbs exist, never
which subjects and objects they are meaningful between.

Two decisions here are deliberate and each has a test that fails if they are
reversed:

* an **undeclared** predicate is unconstrained, so an ontology written before
  edge constraints existed does not suddenly start rejecting edges it used to
  accept; and
* an **untyped** entity against a **constrained** side is an ERROR, not a pass.
  Treating unknown as acceptable is how a constraint quietly stops
  constraining — every entity written before typing existed would slip through
  the very rule added to catch it.
"""

from __future__ import annotations

import json
import os

import pytest

from mind_mem.ontology import EntityType, Ontology, PredicateConstraint


@pytest.fixture()
def onto() -> Ontology:
    return Ontology(
        version="test-v1",
        types={
            "PERSON": EntityType(name="PERSON"),
            "ENGINEER": EntityType(name="ENGINEER", parent="PERSON"),
            "ORGANIZATION": EntityType(name="ORGANIZATION"),
            "ANIMAL": EntityType(name="ANIMAL"),
            "LANGUAGE": EntityType(name="LANGUAGE"),
        },
        predicates={
            "works_at": PredicateConstraint("works_at", frozenset({"PERSON"}), frozenset({"ORGANIZATION"})),
            # Range-only: a partially specified ontology must be useful at once.
            "prefers": PredicateConstraint("prefers", frozenset(), frozenset({"LANGUAGE"})),
        },
    )


class TestTheEdgeIsConstrained:
    def test_a_true_triple_is_allowed(self, onto):
        assert onto.validate_triple("PERSON", "works_at", "ORGANIZATION") == []

    def test_the_dog_cannot_work_at_microsoft(self, onto):
        errors = onto.validate_triple("ANIMAL", "works_at", "ORGANIZATION")
        assert errors, "the store accepted a domain-violating edge"
        assert "domain violation" in errors[0]
        assert "ANIMAL" in errors[0], "the error does not say what was actually supplied"

    def test_the_object_side_is_constrained_too(self, onto):
        errors = onto.validate_triple("PERSON", "works_at", "LANGUAGE")
        assert errors and "range violation" in errors[0]

    def test_a_subtype_satisfies_its_parent(self, onto):
        # Positive control for the rejection above: without subtype walking this
        # test fails and every rejection test still passes, so a validator that
        # refuses EVERYTHING would look correct.
        assert onto.validate_triple("ENGINEER", "works_at", "ORGANIZATION") == []

    def test_one_sided_constraints_still_bite(self, onto):
        assert onto.validate_triple("ANIMAL", "prefers", "LANGUAGE") == [], "an empty domain must mean unconstrained, not deny-all"
        assert onto.validate_triple("PERSON", "prefers", "ORGANIZATION"), "a declared range must be enforced even when the domain is open"


class TestTheTwoDeliberateChoices:
    def test_an_undeclared_predicate_is_unconstrained(self, onto):
        assert onto.validate_triple("ANIMAL", "mentioned_in", "LANGUAGE") == [], (
            "an ontology without this predicate started rejecting edges it used to accept"
        )

    def test_an_untyped_entity_fails_a_constrained_side(self, onto):
        errors = onto.validate_triple(None, "works_at", "ORGANIZATION")
        assert errors, (
            "an untyped subject passed a constrained domain — every entity written "
            "before typing existed would bypass the rule added to catch it"
        )
        assert "untyped" in errors[0]

    def test_an_untyped_entity_passes_an_unconstrained_side(self, onto):
        # The rule above must not become deny-all-untyped: `prefers` has an open
        # domain, so an untyped subject is fine there.
        assert onto.validate_triple(None, "prefers", "LANGUAGE") == []


class TestConstraintsSurviveStorage:
    def test_a_saved_ontology_keeps_its_constraints(self, onto):
        """A round trip that drops them makes enforcement vanish across a restart."""
        restored = Ontology.from_dict(onto.to_dict())
        assert restored.predicates, "predicates were lost in serialisation"
        assert restored.validate_triple("ANIMAL", "works_at", "ORGANIZATION"), (
            "the restored ontology no longer enforces what the original did"
        )

    def test_the_serialised_form_is_deterministic(self, onto):
        """It is hashed into provenance, and set iteration order is not stable."""
        once = json.dumps(onto.to_dict(), sort_keys=True)
        twice = json.dumps(Ontology.from_dict(onto.to_dict()).to_dict(), sort_keys=True)
        assert once == twice

    def test_an_ontology_with_no_predicates_still_loads(self):
        """Back-compat: files written before this feature carry no predicates key."""
        legacy = {"version": "old", "types": {"PERSON": {"name": "PERSON"}}}
        o = Ontology.from_dict(legacy)
        assert o.predicates == {}
        assert o.validate_triple("ANIMAL", "works_at", "ORGANIZATION") == []


class TestMalformedConstraintsAreRefused:
    def test_a_predicate_name_must_be_snake_case(self):
        with pytest.raises(ValueError, match="snake_case"):
            PredicateConstraint("WorksAt")


class TestTheGateRefusesTheEdgeEndToEnd:
    """Domain/range enforced on the real write path, not just in the validator.

    ``KnowledgeGraph.add_edge`` describes itself as "the governance choke point
    for the graph, and the only one" — approve_edge, graph_ingest and the admin
    tool all land here — so wiring the ontology check into it covers every door
    at once. These tests go through that door, with a real admission receipt,
    rather than calling ``validate_triple`` directly.

    The gate is OFF by default and these tests turn it on explicitly, because
    enabling it rejects edges a store previously accepted and an untyped entity
    fails a constrained side by design.
    """

    @staticmethod
    def _graph(tmp_path):
        from mind_mem.knowledge_graph import KnowledgeGraph

        return KnowledgeGraph(str(tmp_path / "g" / "graph.db"))

    @staticmethod
    def _typed(g, surface: str, type_name: str) -> str:
        eid = g.entities.resolve(surface)
        g.entities.set_entity_type(eid, type_name)
        return eid

    def _add(self, g, subj, pred, obj, block_id):
        """Add an edge through a real edge-admission receipt.

        ``admit_edge``, not ``admit_block``: EDGE_APPROVAL is bound to its own
        scope and the gate refuses it inside a block scope. The first draft of
        this helper used admit_block and was told so by name.
        """
        from mind_mem.governance_gate import get_gate
        from mind_mem.knowledge_graph import edge_id

        ws = os.path.dirname(os.path.dirname(g._db_path))
        name = pred.value if hasattr(pred, "value") else str(pred)
        eid = edge_id(subj, name, obj, block_id)
        with get_gate(ws).admit_edge(
            edge_id=eid,
            content=f"{subj} {name} {obj}",
            block_ids=[block_id],
            actor="agent:test",
        ):
            return g.add_edge(subj, pred, obj, source_block_id=block_id)

    def test_the_dog_is_refused_at_the_write_path(self, tmp_path, onto):
        from mind_mem.init_workspace import init
        from mind_mem.knowledge_graph import OntologyViolation, Predicate

        ws = str(tmp_path / "g")
        os.makedirs(ws, exist_ok=True)
        init(ws)
        g = self._graph(tmp_path)
        self._typed(g, "Dog", "ANIMAL")
        self._typed(g, "Ada", "PERSON")

        # Off by default: the pre-existing behaviour is unchanged.
        self._add(g, "Dog", Predicate.AUTHORED_BY, "Ada", "B-1")

        g.set_ontology(
            Ontology(
                version="e2e",
                types=onto.types,
                predicates={"authored_by": PredicateConstraint("authored_by", frozenset({"ORGANIZATION"}), frozenset({"PERSON"}))},
            ),
            mode="enforce",
        )
        with pytest.raises(OntologyViolation):
            self._add(g, "Dog", Predicate.AUTHORED_BY, "Ada", "B-2")

    def test_a_valid_edge_still_lands_with_the_gate_on(self, tmp_path, onto):
        """Positive control: the gate refuses violations, not everything."""
        from mind_mem.init_workspace import init
        from mind_mem.knowledge_graph import Predicate

        ws = str(tmp_path / "g")
        os.makedirs(ws, exist_ok=True)
        init(ws)
        g = self._graph(tmp_path)
        self._typed(g, "Acme", "ORGANIZATION")
        self._typed(g, "Ada", "PERSON")
        g.set_ontology(
            Ontology(
                version="e2e",
                types=onto.types,
                predicates={"authored_by": PredicateConstraint("authored_by", frozenset({"ORGANIZATION"}), frozenset({"PERSON"}))},
            ),
            mode="enforce",
        )
        edge = self._add(g, "Acme", Predicate.AUTHORED_BY, "Ada", "B-3")
        assert edge is not None
