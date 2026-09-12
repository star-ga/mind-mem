"""Auto-extract edges on the write path — as PROPOSALS, never commits.

ROADMAP: "wire lightweight entity/relation extraction (generalize the
block_parser `_ENTITY_ID_RE` beyond canonical IDs to named entities) into
propose_update, so writing a block *proposes* typed KG edges. **Extracted edges
land as proposals, never auto-committed** -- same approval gate as blocks,
honoring the Group H wedge guardrail (source-of-truth graph never self-modifies)."

Two halves already existed: `_ENTITY_ID_RE` finds canonical ids, and
`knowledge_graph.propose_edge` is the HITL-gated staging surface. Nothing joined
them, so writing a block proposed no edges at all.

THE GUARDRAIL IS THE FEATURE. This module returns candidate edges and writes
nothing. A version that called propose_edge itself would still be gated, but a
version that wrote the graph would violate the wedge -- and the distinction is
easy to lose in a refactor, so it is asserted here rather than trusted.

Predicates come from the EXISTING closed Predicate enum. An extractor free to
invent a predicate string would put untyped edges in a typed graph, which is the
same open-set failure M4 closed for slots.
"""

from __future__ import annotations

import pytest

from mind_mem.edge_extraction import (
    MAX_CANDIDATES,
    candidate_edges,
    extract_named_entities,
)


# --------------------------------------------------------------------------
# Generalising past canonical ids
# --------------------------------------------------------------------------

def test_canonical_ids_are_still_found():
    """POSITIVE CONTROL: the existing behaviour must not regress."""
    got = extract_named_entities("This refines D-20260101-001 and PRJ-mind-mem")
    assert "D-20260101-001" in got and "PRJ-mind-mem" in got, got


def test_named_entities_beyond_canonical_ids_are_found():
    """The generalisation the item asks for."""
    got = extract_named_entities("Ada Lovelace owns the Analytical Engine project")
    assert any("Ada Lovelace" == g for g in got), got


def test_ordinary_prose_does_not_become_an_entity():
    """POSITIVE CONTROL for the above: an extractor returning every capitalised
    word would satisfy that test while flooding the graph with noise."""
    got = extract_named_entities("The decision is final. We ship on Tuesday.")
    assert got == [], got


def test_extraction_is_deterministic():
    text = "Ada Lovelace refines D-20260101-001"
    assert extract_named_entities(text) == extract_named_entities(text)


# --------------------------------------------------------------------------
# Candidate edges: typed, closed predicate set, bounded
# --------------------------------------------------------------------------

def test_a_relation_phrase_yields_a_typed_candidate():
    block = {"_id": "D-NEW-001", "Statement": "This refines D-20260101-001"}
    cands = candidate_edges(block)
    assert cands, "no candidate edge extracted from an explicit relation phrase"
    c = cands[0]
    assert c["subject"] == "D-NEW-001"
    assert c["object"] == "D-20260101-001"
    assert c["predicate"] == "refines", c


def test_the_predicate_is_always_a_member_of_the_closed_enum():
    from mind_mem.knowledge_graph import Predicate

    legal = {p.value for p in Predicate}
    block = {"_id": "D-1", "Statement": "This supersedes D-20260101-001 and depends on PRJ-mind"}
    for c in candidate_edges(block):
        assert c["predicate"] in legal, c


def test_no_relation_phrase_means_no_candidate():
    """Mentioning an id is not a claim about a relationship."""
    block = {"_id": "D-1", "Statement": "Background reading: D-20260101-001"}
    assert candidate_edges(block) == []


def test_a_block_never_proposes_an_edge_to_itself():
    block = {"_id": "D-20260101-001", "Statement": "This refines D-20260101-001"}
    assert candidate_edges(block) == []


def test_the_candidate_list_is_bounded():
    """An unbounded extractor turns one pathological block into a review flood."""
    ids = " ".join(f"D-2026010{i % 9 + 1}-{i:03d}" for i in range(50))
    block = {"_id": "D-NEW", "Statement": f"This refines {ids}"}
    assert len(candidate_edges(block)) <= MAX_CANDIDATES


# --------------------------------------------------------------------------
# The wedge guardrail
# --------------------------------------------------------------------------

def test_the_module_never_writes_the_graph():
    """Group H: the source-of-truth graph never self-modifies.

    Asserted on the import graph rather than by trusting the docstring: this
    module must not be able to reach a write, even accidentally, in a refactor.
    """
    import ast

    import mind_mem.edge_extraction as m

    tree = ast.parse(open(m.__file__, encoding="utf-8").read())
    called = {
        n.func.attr for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    }
    forbidden = called & {"propose_edge", "add_edge", "approve_edge", "commit", "write"}
    assert not forbidden, f"edge_extraction must only RETURN candidates; it calls {forbidden}"


def test_candidates_are_plain_data_a_caller_can_review():
    block = {"_id": "D-1", "Statement": "This refines D-20260101-001"}
    for c in candidate_edges(block):
        assert set(c) == {"subject", "predicate", "object", "evidence"}
        for v in c.values():
            assert isinstance(v, str), c
