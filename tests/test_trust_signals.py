# Copyright 2026 STARGA, Inc.
"""AUD-05 — ``load_rollback_history`` was reading the wrong ledger.

It counted field-audit sidecar rows whose ``operation`` was
``"rollback"``. No door has ever written that verb to the sidecar, so
the first element of its return value was ``{}`` on every workspace that
has ever existed — and an empty map reads as "nobody has been rolled
back", not as "this function is looking in the wrong place". A signal
that is zero by construction is worse than an absent one, because it
answers.

The function is public (re-exported from :mod:`mind_mem.trust_scores`),
so it was repointed rather than removed: withdrawals are recorded by the
governance gate in the evidence chain, under
:data:`~mind_mem.trust_signals.ROLLBACK_ACTION`.

The tests here pin the two facts a repoint can get wrong:

* the constants match what the **gate actually writes**, driven through
  the real gate rather than a fixture that could encode the same wrong
  guess as the code under test;
* the loader still creates nothing on a read path.
"""

from __future__ import annotations

import os

import pytest

from mind_mem.audit_chain import RETIRED_OPERATIONS
from mind_mem.trust_signals import (
    DELETE_PHASE_COUNTED,
    DELETE_PHASE_KEY,
    EVIDENCE_CHAIN_REL,
    ROLLBACK_ACTION,
    ROLLBACK_OPERATION,
    load_rollback_history,
)


def test_the_old_verb_is_a_retired_sidecar_operation() -> None:
    """Names the defect in code: what it counted was never appendable.

    If this ever fails, some door has started writing ``"rollback"`` to
    the sidecar and the two ledgers have to be reconciled again.
    """
    assert ROLLBACK_OPERATION in RETIRED_OPERATIONS


def test_falsy_workspace_returns_empty() -> None:
    assert load_rollback_history("") == ({}, {})


def test_a_read_creates_nothing(tmp_path) -> None:
    """``EvidenceChain.__init__`` makedirs its store's directory.

    So the ``isfile`` probe must come first, or asking "were there
    rollbacks?" is what creates ``memory/``.
    """
    before = sorted(os.listdir(tmp_path))
    assert load_rollback_history(str(tmp_path)) == ({}, {})
    assert sorted(os.listdir(tmp_path)) == before
    assert not (tmp_path / "memory").exists()


# ---------------------------------------------------------------------------
# Driven through the real gate
# ---------------------------------------------------------------------------


@pytest.fixture
def gated_workspace(tmp_path):
    from mind_mem.init_workspace import init

    ws = str(tmp_path / "ws")
    init(ws)
    return ws


def test_a_governed_delete_is_counted_once_by_the_real_gate(gated_workspace) -> None:
    """End-to-end: the gate writes the phases, the loader reads them.

    A fixture that hand-wrote ``delete_phase`` values would prove only
    that this module agrees with itself. Driving the real
    ``admit_delete`` scope is what makes
    :data:`DELETE_PHASE_COUNTED` a checked coupling instead of a guess —
    if the gate renames a phase, this goes red.
    """
    from mind_mem.governance_gate import get_gate

    ws = gated_workspace
    gate = get_gate(ws)
    with gate.admit_delete(
        "D-20260901-001",
        rationale="operator removed the block under test",
        actor="operator-1",
        target_file="decisions/DECISIONS.md",
    ) as receipt:
        receipt.record_removal("D-20260901-001", "the removed content")

    assert os.path.isfile(os.path.join(ws, EVIDENCE_CHAIN_REL)), (
        "positive control: the gate must have written an evidence chain, or the counts below are vacuous"
    )

    rollbacks, writes = load_rollback_history(ws)
    assert rollbacks == {"operator-1": 1}, (
        f"a single delete writes an 'admitted' and a 'removed' record; counting both charges it twice — got {rollbacks}"
    )
    assert writes == {"operator-1": 2}, writes


def test_both_delete_phases_are_really_on_disk(gated_workspace) -> None:
    """Positive control for the test above.

    ``rollbacks == 1`` would also hold if the gate had written only one
    record. This proves two exist and that exactly one of them carries
    :data:`DELETE_PHASE_COUNTED`, so the de-duplication is doing work.
    """
    import json

    from mind_mem.governance_gate import get_gate

    ws = gated_workspace
    with get_gate(ws).admit_delete(
        "D-20260901-002",
        rationale="operator removed the block under test",
        actor="operator-2",
        target_file="decisions/DECISIONS.md",
    ) as receipt:
        receipt.record_removal("D-20260901-002", "the removed content")

    with open(os.path.join(ws, EVIDENCE_CHAIN_REL), encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]

    mine = [r for r in rows if r.get("actor") == "operator-2"]
    assert len(mine) == 2, f"expected an admitted and a removed record, got {len(mine)}"
    assert all(r["action"] == ROLLBACK_ACTION for r in mine), [r["action"] for r in mine]
    phases = sorted((r.get("metadata") or {}).get(DELETE_PHASE_KEY) for r in mine)
    assert phases == ["admitted", DELETE_PHASE_COUNTED], phases


def test_an_unattributed_action_is_charged_to_nobody(gated_workspace) -> None:
    """A blank actor cannot be debited, and must not become a bucket."""
    from mind_mem.evidence_objects import EvidenceAction, EvidenceChain

    ws = gated_workspace
    chain = EvidenceChain(store_path=os.path.join(ws, EVIDENCE_CHAIN_REL))
    chain.create(
        action=EvidenceAction.ROLLBACK,
        actor="",
        target_block_id="D-1",
        target_file="decisions/DECISIONS.md",
        payload="x",
    )
    rollbacks, writes = load_rollback_history(ws)
    assert "" not in rollbacks
    assert "" not in writes
