# Copyright 2026 STARGA, Inc.
"""``admit_edge`` — a knowledge-graph edge gets its own scope and its own tier.

Committing an edge needs an authorisation record. Until 5.0.2 it borrowed
one: all three edge doors (``mcp.graph_add_edge``, ``mcp.approve_edge``,
``graph_ingest.approve_relation_signals``) opened ``admit_proposal``, which
is the widest scope the gate has —

* its receipt is ``PROPOSAL``-kind, and
  :meth:`~mind_mem.admission.AdmissionReceipt.authorizes` answers ``True``
  for **every** id it is asked about; and
* it mints :attr:`~mind_mem.enums.IngestTier.PROPOSAL_APPLY`, the one tier
  whose :data:`~mind_mem.enums.INITIAL_STATUS` row is ``ACTIVE``.

So writing one edge carried, for the length of the scope, authority to
write any block in the corpus at a status recall serves. Nothing about an
edge needs that, and it left the ``ACTIVE``-minting scope with seven
openers when ``docs/GOVERNED_WRITES.md`` claims one.

:class:`TestTheReceiptIsNarrow` is the A/B: the same workspace, the same
call, one scope beside the other, so the narrowing is measured rather than
described. :class:`TestTheTierIsBoundToTheScope` covers the half that would
otherwise be a new hole — ``EDGE_APPROVAL``'s ``INITIAL_STATUS`` row is
``None`` (an edge has no ``Status`` field, so there is no honest value), and
a carrying row constrains no status at all, which is safe only because the
tier is unreachable from any scope but this one.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterator

import pytest

from mind_mem.admission import BATCH, BLOCK, PROPOSAL, UngatedRestoreError, UngatedWriteError
from mind_mem.enums import INITIAL_STATUS, IngestTier
from mind_mem.governance_gate import (
    EDGE,
    EDGE_ID_PREFIX,
    OPEN_SCOPE_TIERS,
    SCOPE_BOUND_TIERS,
    GovernanceBypassError,
    GovernanceGate,
    get_gate,
)
from mind_mem.init_workspace import init as init_workspace
from mind_mem.knowledge_graph import KnowledgeGraph, Predicate, default_db_path, edge_id
from mind_mem.storage import get_block_store

SUBJECT = "pineapple"
OBJECT = "protocol"
SOURCE_BLOCK = "D-20260101-001"


@pytest.fixture
def workspace(tmp_path: Path) -> Iterator[str]:
    ws = str(tmp_path / "ws")
    os.makedirs(ws)
    init_workspace(ws)
    yield ws


@pytest.fixture
def eid() -> str:
    return edge_id(SUBJECT, Predicate.DEPENDS_ON, OBJECT, SOURCE_BLOCK)


# ---------------------------------------------------------------------------
# The prefix copy the gate holds must be the prefix the graph mints
# ---------------------------------------------------------------------------


class TestThePrefixDoesNotDrift:
    @pytest.mark.unit
    def test_the_gates_prefix_is_the_one_edge_id_mints(self, eid: str) -> None:
        """``governance_gate`` copies the prefix rather than importing it.

        ``knowledge_graph`` imports ``admission``; making the gate import
        ``knowledge_graph`` to authorise an edge would put the graph on the
        authorisation path. The copy is the price, and this is the guard
        that keeps the copy honest.
        """
        assert eid.startswith(EDGE_ID_PREFIX), f"edge_id mints {eid!r}, which the gate's {EDGE_ID_PREFIX!r} does not match"

    @pytest.mark.unit
    def test_the_prefix_is_not_one_the_block_store_can_route(self) -> None:
        """The load-bearing consequence: an edge scope cannot reach a corpus file.

        ``covers`` already bounds the scope to the ids it names. This is the
        second, independent reason the same scope cannot land a block: the
        one id it covers has a prefix ``_resolve_block_file`` returns
        ``None`` for, so ``write_block`` refuses it before any file is
        opened.
        """
        from mind_mem.block_store import _BLOCK_PREFIX_MAP

        assert _BLOCK_PREFIX_MAP, "the routing table is empty; this guard would pass over nothing"
        assert EDGE_ID_PREFIX.rstrip("-") not in _BLOCK_PREFIX_MAP


# ---------------------------------------------------------------------------
# The receipt — measured beside the scope it replaced
# ---------------------------------------------------------------------------


class TestTheReceiptIsNarrow:
    @pytest.mark.unit
    def test_an_edge_receipt_covers_the_edge_and_nothing_else(self, workspace: str, eid: str) -> None:
        with get_gate(workspace).admit_edge(eid, "x") as receipt:
            assert receipt.kind == EDGE
            assert receipt.tier is IngestTier.EDGE_APPROVAL
            assert receipt.covers == frozenset({eid})
            assert receipt.authorizes(eid)
            assert not receipt.authorizes(SOURCE_BLOCK), "the edge scope authorises a corpus block it never named"

    @pytest.mark.unit
    def test_the_proposal_scope_it_replaced_authorises_everything(self, workspace: str) -> None:
        """The A/B half. Without it the assertion above is a claim about one scope.

        This is what the three edge doors were opening; it is why the move
        is a narrowing and not a rename.
        """
        with get_gate(workspace).admit_proposal("P-20260903-001", "x") as receipt:
            assert receipt.kind == PROPOSAL
            assert receipt.covers == frozenset()
            assert receipt.authorizes(SOURCE_BLOCK), "the reproduction changed; re-measure before editing this file"
            assert receipt.authorizes("anything-at-all")

    @pytest.mark.unit
    def test_an_edge_scope_refuses_a_block_it_does_not_cover(self, workspace: str, eid: str) -> None:
        """Refusal, with the positive control that the same write otherwise lands."""
        store = get_block_store(workspace)
        block = {"_id": "D-20260903-777", "Statement": "s", "Status": "active", "Type": "Decision"}

        with get_gate(workspace).admit_edge(eid, "x"):
            with pytest.raises(UngatedWriteError):
                store.write_block(dict(block))
        assert store.get_by_id("D-20260903-777") is None

        with get_gate(workspace).admit_proposal("P-20260903-002", "x"):
            store.write_block(dict(block))
        assert store.get_by_id("D-20260903-777") is not None, "the control write failed; the refusal above proved nothing"

    @pytest.mark.unit
    def test_a_named_block_rides_in_the_same_scope(self, workspace: str, eid: str) -> None:
        """One decision, one scope, and the ids it may touch are named in it.

        ``approve_relation_signals`` lands the edge and re-stamps the signal
        block that released it. Both are in ``covers``; nothing else is.
        """
        with get_gate(workspace).admit_edge(eid, "x", block_ids=("SIG-20260903-001",)) as receipt:
            assert receipt.covers == frozenset({eid, "SIG-20260903-001"})
            assert receipt.authorizes("SIG-20260903-001")
            assert not receipt.authorizes("SIG-20260903-002")

    @pytest.mark.unit
    def test_an_edge_receipt_does_not_authorise_a_restore(self, workspace: str, eid: str) -> None:
        """A restore needs a BATCH receipt; an EDGE one is not transferable.

        The same rule that stops a proposal receipt authorising one. Checked
        because ``admit_edge`` is a new receipt shape and a restore is the
        most destructive operation the product has.
        """
        store = get_block_store(workspace)
        with get_gate(workspace).admit_edge(eid, "x") as receipt:
            assert receipt.kind not in (BATCH,)
            with pytest.raises(UngatedRestoreError):
                store.restore(os.path.join(workspace, "no-such-snapshot"))


# ---------------------------------------------------------------------------
# The subject must be an edge
# ---------------------------------------------------------------------------


class TestTheSubjectMustBeAnEdge:
    @pytest.mark.unit
    def test_a_block_id_subject_is_refused(self, workspace: str) -> None:
        with pytest.raises(GovernanceBypassError, match="must be an edge id"):
            with get_gate(workspace).admit_edge(SOURCE_BLOCK, "x"):
                pass

    @pytest.mark.unit
    def test_the_refusal_mints_no_record(self, workspace: str, eid: str) -> None:
        """A scope that cannot open must leave no authorisation behind.

        Paired with a positive control on the same ledger, so "no rows" is
        shown to be a refusal rather than a ledger that never moves.
        """
        from _ledger_rows import chain_rows

        gate = get_gate(workspace)
        with gate.admit_edge(eid, "x"):
            pass
        before = len(chain_rows(workspace))
        assert before, "the ledger never moved at all; the assertion below would be vacuous"

        with pytest.raises(GovernanceBypassError):
            with gate.admit_edge("not-an-edge", "x"):
                pass
        assert len(chain_rows(workspace)) == before


# ---------------------------------------------------------------------------
# The tier is bound to the scope, in both directions
# ---------------------------------------------------------------------------


class TestTheTierIsBoundToTheScope:
    @pytest.mark.unit
    def test_the_edge_tier_carries_no_status_rule(self) -> None:
        """Stated where a reader will look for it, because it is the risk."""
        assert INITIAL_STATUS[IngestTier.EDGE_APPROVAL] is None

    @pytest.mark.unit
    def test_only_the_edge_scope_may_mint_the_edge_tier(self, workspace: str) -> None:
        """The direction that matters: a carrying tier out of an open scope.

        ``admit_block`` with this tier would be a receipt that constrains no
        status on whatever id it names — the exact reach ``admit_edge``
        exists to withdraw.
        """
        assert SCOPE_BOUND_TIERS[EDGE] is IngestTier.EDGE_APPROVAL
        assert IngestTier.EDGE_APPROVAL not in OPEN_SCOPE_TIERS

        gate = get_gate(workspace)
        with pytest.raises(GovernanceBypassError, match="bound to its own scope"):
            with gate.admit_block(action="WRITE", block_id="D-20260903-001", content="x", tier=IngestTier.EDGE_APPROVAL):
                pass
        with pytest.raises(GovernanceBypassError, match="bound to its own scope"):
            with gate.admit_batch(action="WRITE", batch_id="b1", block_ids=["D-20260903-001"], content="x", tier=IngestTier.EDGE_APPROVAL):
                pass

    @pytest.mark.unit
    def test_an_open_scope_tier_is_the_positive_control(self, workspace: str) -> None:
        """The same two calls succeed on a tier the open scopes DO admit."""
        gate = get_gate(workspace)
        with gate.admit_block(action="WRITE", block_id="D-20260903-001", content="x", tier=IngestTier.RESTAMP) as receipt:
            assert receipt.kind == BLOCK
        with gate.admit_batch(action="WRITE", batch_id="b1", block_ids=["D-20260903-001"], content="x", tier=IngestTier.RESTAMP):
            pass

    @pytest.mark.unit
    def test_the_edge_scope_may_mint_no_other_tier(self) -> None:
        """The other direction, checked at the classifier.

        ``admit_edge`` hardcodes its tier, so this is unreachable through
        the public surface — which is exactly why it is checked here: the
        rule must hold whatever a future scope passes.
        """
        for tier in IngestTier:
            if tier is IngestTier.EDGE_APPROVAL:
                GovernanceGate._check_tier(EDGE, "E-0000000000000000", tier)
                continue
            with pytest.raises(GovernanceBypassError, match="edge admission may only use"):
                GovernanceGate._check_tier(EDGE, "E-0000000000000000", tier)


# ---------------------------------------------------------------------------
# End to end — the doors still commit, and the ledgers still move
# ---------------------------------------------------------------------------


class TestTheDoorsStillCommit:
    @pytest.mark.unit
    def test_the_edge_lands_inside_the_scope_and_is_refused_outside_it(self, workspace: str) -> None:
        """The choke point still refuses, and the new scope still satisfies it."""
        with KnowledgeGraph(default_db_path(workspace)) as kg:
            with pytest.raises(UngatedWriteError):
                kg.add_edge(SUBJECT, Predicate.DEPENDS_ON, OBJECT, source_block_id=SOURCE_BLOCK)
            assert kg._query_edges() == []

            eid = edge_id(SUBJECT, Predicate.DEPENDS_ON, OBJECT, SOURCE_BLOCK)
            with get_gate(workspace).admit_edge(eid, "x", actor="pytest"):
                edge = kg.add_edge(SUBJECT, Predicate.DEPENDS_ON, OBJECT, source_block_id=SOURCE_BLOCK)
            assert edge.object == OBJECT
            assert len(kg._query_edges()) == 1

    @pytest.mark.unit
    def test_the_scope_records_the_edge_tier_on_the_chain(self, workspace: str, eid: str) -> None:
        """The record names the tier, so an auditor reads the scope off the row."""
        from _ledger_rows import authorisation_rows, evidence_rows

        with get_gate(workspace).admit_edge(eid, "x", metadata={"door": "test"}):
            pass
        rows = [r for r in authorisation_rows(evidence_rows(workspace)) if r.get("target_block_id") == eid]
        assert len(rows) == 1, f"expected one authorisation record for {eid}, got {len(rows)}"
        meta: dict[str, Any] = rows[0].get("metadata") or {}
        assert meta["ingest_tier"] == IngestTier.EDGE_APPROVAL.value
        assert meta["door"] == "test"
        assert meta["operation"] == "write"
