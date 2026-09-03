# Copyright 2026 STARGA, Inc.
"""AUD-02 — the recall attestation's ``index_anchor`` must move when the store does.

The anchor is the only field in the attestation that says anything about the
*corpus* a run observed; every other field describes the query or the answer.
``served_ledger.append_served_run`` copies it verbatim onto every ledger row, so
if the anchor is a constant then the served-set ledger — the artifact behind
"prove what it served" — binds each row to nothing at all.

MEASURED, on a fresh ``init()`` workspace, before the fix::

    anchor on fresh init: True (genesis)
    hash_chain= 1 evidence rows= 1 attest anchor moved: False
    delete status= 200 hash_chain= 3 evidence rows= 3 attest anchor moved: False
    POSITIVE CONTROL sidecar record_change -> attest anchor moved: True

The resolver read ``.mind-mem-audit/chain.jsonl`` — the field-audit sidecar —
whose only writer is :meth:`~mind_mem.field_audit.FieldAuditor.record_change`.
Governed writes and governed deletes go to ``memory/hash_chain_v2.db`` and the
evidence chain and never touch it, so the anchor sat at
:data:`~mind_mem.recall_attestation.GENESIS_ANCHOR` for the whole life of a
store nobody had run a field audit against. The last line is the positive
control that the *measurement* worked: something could move the anchor, just
never a governed mutation.

This module is the gate on the repointed resolver, and it is deliberately built
out of governed mutations rather than hand-written ledger rows: the claim is
"the anchor moves whenever the gate mints", so every arrow here starts at the
gate. :class:`TestMutationTwin` restores the sidecar walk and shows the same
assertions going red.

Every "moved" assertion is paired with proof the mutation happened (the block is
readable / gone, the ledger grew), and the one "did not move" assertion is
paired with proof the sidecar write it ignores really did land. A negative
assertion whose subject never existed proves nothing.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest
from _ledger_rows import count_chain_authorisations

from mind_mem.block_store import MarkdownBlockStore
from mind_mem.field_audit import FieldAuditor
from mind_mem.governance_gate import evict_gate, get_gate
from mind_mem.hash_chain_v2 import HashChainV2
from mind_mem.http_transport import _handle_delete_memory
from mind_mem.init_workspace import init
from mind_mem.mm_cli import config_set
from mind_mem.preimage import preimage
from mind_mem.recall_attestation import (
    GENESIS_ANCHOR,
    INDEX_ANCHOR_TAG,
    _resolve_index_anchor,
    derive_recall_attestation_for_workspace,
    index_anchor_ledger_path,
)
from mind_mem.recall_digests import query_hash, served_set_digest
from mind_mem.served_ledger import append_served_run, read_served_runs

SIDECAR_RELPATH = os.path.join(".mind-mem-audit", "chain.jsonl")

PIPELINE = "b" * 64


@pytest.fixture()
def workspace(tmp_path: Path) -> str:
    """A real, initialised workspace — the thing a governed write needs."""
    ws = str(tmp_path / "ws")
    init(ws)
    return ws


def _block(bid: str, statement: str) -> dict[str, Any]:
    return {"_id": bid, "Statement": statement, "Date": "2026-09-02", "Status": "active"}


def _governed_write(ws: str, bid: str, statement: str = "a governed statement") -> None:
    """Write one block the only way the store accepts: through the gate."""
    with get_gate(ws).admit_proposal(proposal_id=f"P-{bid}", content="[]", actor="pytest"):
        MarkdownBlockStore(ws).write_block(_block(bid, statement))


def _enable_served_ledger(ws: str) -> None:
    """Turn the opt-in ledger on the way an operator does: ``mm config set``.

    ``init()`` binds the governance spec to the config it wrote, so
    replacing that file without re-attesting makes the very next
    ``admit`` fail closed with a spec drift. ``mm config set`` writes and
    re-attests in one step, so the drift check stays armed instead of
    being routed around. The ``evict_gate`` afterwards is separate and
    still needed: a gate already built for this workspace holds the old
    binding in ``SpecBindingManager``'s cache, which no on-disk write can
    reach.
    """
    config_set(os.path.join(ws, "mind-mem.json"), "served_ledger", {"enabled": True})
    evict_gate(ws)


def _chain_mints(ws: str) -> int:
    """Governed scopes that minted in the ledger, read without touching it.

    Counts *authorisations*, not rows: a scope also appends a close record
    saying whether the write landed, and every assertion here means "the
    gate minted", which is the first. ``tests/_ledger_rows`` holds that
    convention once.
    """
    return count_chain_authorisations(ws, index_anchor_ledger_path(ws))


def _sidecar_rows(ws: str) -> int:
    path = os.path.join(ws, SIDECAR_RELPATH)
    if not os.path.isfile(path):
        return 0
    with open(path, encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _tree(ws: str) -> set[str]:
    return {os.path.relpath(os.path.join(root, name), ws) for root, dirs, files in os.walk(ws) for name in list(dirs) + list(files)}


# ---------------------------------------------------------------------------
# The ledger the anchor reads, and the one it does not
# ---------------------------------------------------------------------------


def test_the_anchor_reads_the_ledger_the_gate_appends_to(workspace: str) -> None:
    """Name the file, so a silent move of the ledger fails here first."""
    assert index_anchor_ledger_path(workspace) == os.path.join(os.path.abspath(workspace), "memory", "hash_chain_v2.db")


def test_an_absent_ledger_is_the_genesis_anchor_and_the_read_creates_nothing(tmp_path: Path) -> None:
    """Rail 2: deriving an attestation may not write, not even a directory."""
    ws = str(tmp_path)
    before = _tree(ws)

    assert _resolve_index_anchor(ws) == GENESIS_ANCHOR

    assert _tree(ws) == before, "resolving the anchor mutated the workspace"
    assert not os.path.exists(index_anchor_ledger_path(ws))
    assert not os.path.isdir(os.path.join(ws, ".mind-mem-audit"))


def test_an_empty_ledger_is_the_genesis_anchor(tmp_path: Path) -> None:
    """A ledger that exists with no rows is still "nothing has been admitted"."""
    ws = str(tmp_path)
    db_path = index_anchor_ledger_path(ws)
    HashChainV2(db_path)  # creates the schema, appends nothing
    assert os.path.isfile(db_path), "positive control: the ledger file exists"
    assert _chain_mints(ws) == 0

    assert _resolve_index_anchor(ws) == GENESIS_ANCHOR


def test_an_unreadable_ledger_is_the_genesis_anchor(tmp_path: Path) -> None:
    """A corrupt ledger degrades to the sentinel rather than raising into recall."""
    ws = str(tmp_path)
    db_path = index_anchor_ledger_path(ws)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    Path(db_path).write_text("this is not a sqlite database", encoding="utf-8")

    assert _resolve_index_anchor(ws) == GENESIS_ANCHOR


# ---------------------------------------------------------------------------
# The property: a governed mutation moves the anchor
# ---------------------------------------------------------------------------


def test_a_governed_write_moves_the_anchor(workspace: str) -> None:
    before = _resolve_index_anchor(workspace)
    assert before == GENESIS_ANCHOR
    assert _chain_mints(workspace) == 0

    _governed_write(workspace, "D-20260902-001")

    # Positive controls: the write really landed, and it really was governed.
    assert MarkdownBlockStore(workspace).get_by_id("D-20260902-001") is not None
    assert _chain_mints(workspace) == 1, "no ledger row means the gate never minted"

    assert _resolve_index_anchor(workspace) != before


def test_a_governed_delete_moves_the_anchor(workspace: str) -> None:
    _governed_write(workspace, "D-20260902-002")
    before = _resolve_index_anchor(workspace)
    rows_before = _chain_mints(workspace)

    status, body = _handle_delete_memory(workspace, "D-20260902-002", actor="pytest")

    # Positive controls: the door answered, and the block is gone.
    assert status == 200, body
    assert MarkdownBlockStore(workspace).get_by_id("D-20260902-002") is None
    assert _chain_mints(workspace) > rows_before

    assert _resolve_index_anchor(workspace) != before


def test_two_writes_give_two_anchors(workspace: str) -> None:
    """Not merely "not genesis" — each mint is its own corpus state."""
    genesis = _resolve_index_anchor(workspace)
    _governed_write(workspace, "D-20260902-003", "first")
    first = _resolve_index_anchor(workspace)
    _governed_write(workspace, "D-20260902-004", "second")
    second = _resolve_index_anchor(workspace)

    assert _chain_mints(workspace) == 2, "positive control: two mints, not one"
    assert len({genesis, first, second}) == 3


def test_the_anchor_is_stable_while_nothing_is_admitted(workspace: str) -> None:
    """It is a corpus-state anchor, not a nonce: no mint, no move."""
    _governed_write(workspace, "D-20260902-005")

    assert _resolve_index_anchor(workspace) == _resolve_index_anchor(workspace)


def test_the_anchor_is_recomputable_from_the_head_an_auditor_can_read(workspace: str) -> None:
    """An anchor nobody can re-derive is a number, not evidence."""
    _governed_write(workspace, "D-20260902-006")

    head = HashChainV2.open_readonly(index_anchor_ledger_path(workspace)).get_latest(n=1)[-1].entry_hash
    expected = hashlib.sha256(preimage(INDEX_ANCHOR_TAG, head)).hexdigest()

    assert _resolve_index_anchor(workspace) == expected
    assert len(head) == 128, "positive control: the ledger really does hash SHA3-512"


def test_the_attestation_binds_the_moving_anchor(workspace: str) -> None:
    """The record, not just the resolver: two corpus states, two attestations."""
    results = [{"_id": "D-20260902-007", "_score": 1.0}]

    _governed_write(workspace, "D-20260902-007")
    first = derive_recall_attestation_for_workspace(results, workspace, vector_requested=False, vector_available=False, query="anchor")
    _governed_write(workspace, "D-20260902-008")
    second = derive_recall_attestation_for_workspace(results, workspace, vector_requested=False, vector_available=False, query="anchor")

    assert first.index_anchor != GENESIS_ANCHOR, "positive control: the anchor resolved at all"
    assert first.index_anchor != second.index_anchor, "same answer, different corpus, one hash"
    assert first.attestation_hash != second.attestation_hash


# ---------------------------------------------------------------------------
# The negative control — the sidecar the resolver used to read
# ---------------------------------------------------------------------------


def test_the_field_audit_sidecar_alone_does_not_move_the_anchor(workspace: str) -> None:
    """The old source, kept as the control it should always have been.

    ``FieldAuditor.record_change`` is the only writer of
    ``.mind-mem-audit/chain.jsonl``. It mints no governance receipt and writes
    no block, so it says nothing about what recall may serve — and it must not
    be able to advance an anchor that claims to describe the corpus.

    The positive control is the row count: without it this test would pass just
    as well if ``record_change`` had silently written nothing, which is exactly
    how the original defect stayed invisible.
    """
    _governed_write(workspace, "D-20260902-009")
    before = _resolve_index_anchor(workspace)
    assert _sidecar_rows(workspace) == 0

    FieldAuditor(workspace).record_change(
        "D-20260902-009",
        "decisions/DECISIONS.md",
        "Status",
        "active",
        "superseded",
        agent="pytest",
    )

    assert _sidecar_rows(workspace) >= 1, "positive control: the sidecar write landed"
    assert _resolve_index_anchor(workspace) == before


# ---------------------------------------------------------------------------
# The served ledger inherits the anchor — including its WIDTH
# ---------------------------------------------------------------------------


def test_the_served_ledger_records_the_resolved_anchor(tmp_path: Path) -> None:
    """The finding's downstream half, end to end.

    ``append_served_run`` copies ``index_anchor`` from the attestation and
    validates it through :func:`~mind_mem.recall_digests.hex64`. The recall
    path swallows every exception from that call and logs it, so an anchor of
    the wrong width would not fail loudly — it would silently stop the ledger
    recording anything at all. Two rows at two corpus states, with the widths
    asserted, is what keeps that coupling honest.
    """
    ws = str(tmp_path / "ws")
    init(ws)
    _enable_served_ledger(ws)

    _governed_write(ws, "D-20260902-010")
    first_anchor = _resolve_index_anchor(ws)
    append_served_run(
        ws,
        query_hash=query_hash("q"),
        served_digest=served_set_digest(["D-20260902-010"]),
        ids=["D-20260902-010"],
        pipeline_hash=PIPELINE,
        index_anchor=first_anchor,
        scoring_instant="2026-09-02",
    )

    _governed_write(ws, "D-20260902-011")
    second_anchor = _resolve_index_anchor(ws)
    append_served_run(
        ws,
        query_hash=query_hash("q"),
        served_digest=served_set_digest(["D-20260902-010"]),
        ids=["D-20260902-010"],
        pipeline_hash=PIPELINE,
        index_anchor=second_anchor,
        scoring_instant="2026-09-02",
    )

    rows = read_served_runs(ws)
    assert len(rows) == 2, "positive control: both appends landed"
    assert [row.index_anchor for row in rows] == [first_anchor, second_anchor]
    assert rows[0].index_anchor != rows[1].index_anchor, "the served proof binds to a constant again"
    assert all(len(row.index_anchor) == 64 for row in rows), "the ledger's width contract"


# ---------------------------------------------------------------------------
# Mutation twin — restore the defect, watch the gate go red
# ---------------------------------------------------------------------------


class TestMutationTwin:
    """Point the resolver back at the sidecar and every claim above fails."""

    @staticmethod
    def _sidecar_resolver(ws: str) -> str:
        """The 5.0.1 walk, verbatim in behaviour: last entry_hash of chain.jsonl."""
        chain_path = os.path.join(os.path.abspath(ws), SIDECAR_RELPATH)
        if not os.path.isfile(chain_path):
            return GENESIS_ANCHOR
        last_line = ""
        with open(chain_path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    last_line = line.strip()
        if not last_line:
            return GENESIS_ANCHOR
        entry = json.loads(last_line)
        head = entry.get("entry_hash") if isinstance(entry, dict) else None
        return str(head) if head else GENESIS_ANCHOR

    def test_the_sidecar_resolver_reproduces_the_constant_anchor(self, workspace: str) -> None:
        """Both directions in one workspace: the mutation is dead, the fix is not."""
        mutated_before = self._sidecar_resolver(workspace)
        fixed_before = _resolve_index_anchor(workspace)

        _governed_write(workspace, "D-20260902-012")
        status, body = _handle_delete_memory(workspace, "D-20260902-012", actor="pytest")
        assert status == 200, body
        assert _chain_mints(workspace) >= 3, "positive control: the gate minted for both"

        assert self._sidecar_resolver(workspace) == mutated_before == GENESIS_ANCHOR
        assert _resolve_index_anchor(workspace) != fixed_before

    def test_the_sidecar_resolver_moves_on_a_field_audit_instead(self, workspace: str) -> None:
        """And it moved on the one event that says nothing about the corpus."""
        _governed_write(workspace, "D-20260902-013")
        mutated_before = self._sidecar_resolver(workspace)

        FieldAuditor(workspace).record_change(
            "D-20260902-013",
            "decisions/DECISIONS.md",
            "Status",
            "active",
            "superseded",
            agent="pytest",
        )

        assert self._sidecar_resolver(workspace) != mutated_before
        assert _resolve_index_anchor(workspace) != GENESIS_ANCHOR
