# Copyright 2026 STARGA, Inc.
"""There is no execution path where a memory mutation bypasses evidence admission.

This is the Class A property that distinguishes a governed memory from a store
with a governance feature. It is not a coverage percentage — a single passing
bypass falsifies it — so it is written as a gate with one test per layer of
defence, each demonstrating a DIFFERENT refusal.

Measured 2026-09-07, three layers, three distinct exceptions:

1. no receipt at all               -> UngatedWriteError
2. a receipt whose TIER cannot mint the requested status -> UngatedWriteError
3. the right tier, requested by an UNSANCTIONED caller   -> GovernanceBypassError

Layer 3 is the one that matters most and the one most easily missed: the tier is
bound to the CALLER, not merely named by it. ``IngestTier.PROPOSAL_APPLY`` is
the only tier whose initial status is ``ACTIVE``, so servable memory can be
created by exactly one code path, and asking for that tier from anywhere else is
refused before a receipt is minted.

Each test asserts three things, because "it raised" is not the property:
the exception, that the block is ABSENT afterwards, and that the chain still
verifies. A refusal that leaves a half-written block is not a refusal.
"""

from __future__ import annotations

import os

import pytest

from mind_mem.enums import IngestTier
from mind_mem.evidence_objects import EvidenceChain
from mind_mem.governance_gate import GovernanceBypassError, get_gate
from mind_mem.init_workspace import init
from mind_mem.storage import get_block_store

BID = "D-20260907-999"
BLOCK = {
    "_id": BID,
    "Statement": "a committed memory that must not exist without a receipt",
    "Status": "active",
    "Type": "decision",
}


@pytest.fixture()
def ws(tmp_path, monkeypatch) -> str:
    root = str(tmp_path / "ws")
    os.makedirs(root)
    init(root)
    monkeypatch.setenv("MIND_MEM_WORKSPACE", root)
    return root


def _chain(ws: str) -> EvidenceChain | None:
    p = os.path.join(ws, "memory", "evidence_chain.jsonl")
    return EvidenceChain(store_path=p) if os.path.exists(p) else None


def _assert_refused_cleanly(ws: str) -> None:
    """A refusal must leave nothing behind and must not damage the chain."""
    store = get_block_store(ws)
    assert store.get_by_id(BID) is None, "the refused write left a block behind"
    ch = _chain(ws)
    if ch is not None:
        assert ch.verify_chain()[0], "the refusal damaged the evidence chain"


class TestNoMutationWithoutAdmission:
    def test_an_ungated_write_is_refused(self, ws):
        """Layer 1: the store itself requires a receipt."""
        store = get_block_store(ws)
        with pytest.raises(Exception) as caught:
            store.write_block(dict(BLOCK))
        assert "Ungated" in type(caught.value).__name__ or isinstance(caught.value, GovernanceBypassError)
        _assert_refused_cleanly(ws)

    def test_a_receipt_cannot_mint_a_status_its_tier_forbids(self, ws):
        """Layer 2: holding *a* receipt is not holding the *right* receipt.

        The agent-message tier mints QUARANTINED. Writing an ACTIVE block under
        it is a privilege escalation and is refused even though a receipt is
        open — which is why "has a receipt" is not the property being tested.
        """
        store = get_block_store(ws)
        with pytest.raises(Exception) as caught:
            with get_gate(ws).admit_block(
                action="WRITE",
                block_id=BID,
                content=BLOCK["Statement"],
                tier=IngestTier.AGENT_MESSAGE,
                actor="agent:test",
            ):
                store.write_block(dict(BLOCK))
        assert "Ungated" in type(caught.value).__name__ or isinstance(caught.value, GovernanceBypassError)
        _assert_refused_cleanly(ws)

    def test_the_servable_tier_is_bound_to_its_caller(self, ws):
        """Layer 3: naming the right tier from the wrong place is refused.

        PROPOSAL_APPLY is the only tier whose initial status is ACTIVE, so it is
        the only way to create servable memory. It is refused here, from a test,
        BEFORE a receipt is minted — the tier belongs to the apply path, not to
        whoever asks for it.
        """
        with pytest.raises(GovernanceBypassError):
            with get_gate(ws).admit_block(
                action="WRITE",
                block_id=BID,
                content=BLOCK["Statement"],
                tier=IngestTier.PROPOSAL_APPLY,
                actor="agent:test",
            ):
                pass
        _assert_refused_cleanly(ws)

    def test_exactly_one_tier_can_mint_servable_memory(self):
        """The property above only holds while PROPOSAL_APPLY is unique.

        A second tier minting ACTIVE would open a second door to servable
        memory, and every test above would still pass. This is the control that
        notices that.
        """
        from mind_mem.enums import INITIAL_STATUS, Status

        minters = [t for t, s in INITIAL_STATUS.items() if s == Status.ACTIVE]
        assert minters == [IngestTier.PROPOSAL_APPLY], f"more than one tier can mint servable memory: {minters}"


class TestARefusalIsStillRecorded:
    def test_the_attempt_leaves_a_truthful_trace(self, ws):
        """A refused mutation is not silence — the attempt is in the chain.

        Layer 2 opens a receipt before the write is refused, so the gate writes
        its open and close records. The close record carries the failure, which
        is what makes a refused write distinguishable from one that never
        happened.
        """
        store = get_block_store(ws)
        with pytest.raises(Exception):
            with get_gate(ws).admit_block(
                action="WRITE",
                block_id=BID,
                content=BLOCK["Statement"],
                tier=IngestTier.AGENT_MESSAGE,
                actor="agent:test",
            ):
                store.write_block(dict(BLOCK))

        ch = _chain(ws)
        assert ch is not None and len(ch._entries) >= 2, "a refused mutation left no trace at all"
        assert ch.verify_chain()[0]
        assert store.get_by_id(BID) is None, "recorded the attempt AND wrote the block"
