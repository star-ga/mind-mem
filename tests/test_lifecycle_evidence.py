# Copyright 2026 STARGA, Inc.
"""RA.3 — a block that dies quietly is the one governed act with no receipt.

Every other state change in this product leaves a row somewhere: a write
is admitted, a delete is authorised and reported, a proposal is applied
and can be rolled back. Three transitions were exempt, and all three are
*losses*:

=======================================  ====================================
transition                               what it left behind, before this
=======================================  ====================================
``TierManager.demote``                   an in-process ``emit_event`` and a
                                         log line — no ledger row at all
``TierManager._evict``                   nothing whatsoever
``compaction.archive_completed_blocks``  one *batch* ``MIGRATE`` receipt
                                         naming the run; no record that any
                                         particular block left the served
                                         surface
=======================================  ====================================

So the ledger could say who wrote a block and who killed it outright, and
could not say who quietly stopped it from ever being found. This file
holds the doors to that: :class:`~mind_mem.lifecycle_evidence.LifecycleRecorder`
writes ONE receipt per transition into BOTH the evidence chain (as
``DEMOTE`` / ``ARCHIVE`` / ``FORGET``) and the field-audit sidecar (as
``demote_block`` / ``archive_block`` / ``forget_block``).

Two properties are load-bearing beyond "a row appears", and both are
asserted here rather than argued:

* **the chain still verifies** — a receipt that breaks the ledger it is
  written into is worse than no receipt;
* **the reconciler does not convict the workspace** — ``cross_ledger``
  counts evidence rows that have a hash-chain twin, so a lifecycle row
  must not look like an admission that lost its twin.

Every negative assertion carries a positive control: the ledger is shown
empty before the act, and the act is shown to have really happened (the
tier really moved, the block really left the file), because ``assert no
rows`` passes just as well against a fixture that never demoted anything.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

import pytest

from mind_mem.audit_chain import VALID_OPERATIONS, AuditChain
from mind_mem.compaction import archive_completed_blocks
from mind_mem.evidence_objects import EvidenceAction, EvidenceChain
from mind_mem.governance_gate import evict_gate
from mind_mem.lifecycle_evidence import (
    ARCHIVE_OPERATION,
    ARCHIVE_VERB,
    DEMOTE_OPERATION,
    DEMOTE_VERB,
    EVIDENCE_ACTION_FOR_VERB,
    FORGET_OPERATION,
    FORGET_VERB,
    LIFECYCLE_OPERATIONS,
    LIFECYCLE_VERB_KEY,
    OPERATION_FOR_VERB,
    SUBJECT_BLOCK,
    SUBJECT_TIER_ASSIGNMENT,
    LifecycleRecorder,
    lifecycle_evidence_enabled,
)
from mind_mem.memory_tiers import DemotionReason, MemoryTier, TierManager

CORPUS = ("decisions", "tasks", "entities", "intelligence", "memory")

#: A date comfortably outside every retention window used here.
OLD = "2024-01-01"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_config(ws: Path, *, lifecycle: bool | None) -> None:
    config: dict[str, Any] = {"block_store": {"backend": "markdown"}}
    if lifecycle is not None:
        config["lifecycle_evidence"] = {"enabled": lifecycle}
    (ws / "mind-mem.json").write_text(json.dumps(config), encoding="utf-8")


def _make_workspace(tmp_path: Path, *, lifecycle: bool | None) -> str:
    ws = tmp_path / "ws"
    for sub in CORPUS:
        (ws / sub).mkdir(parents=True, exist_ok=True)
    _write_config(ws, lifecycle=lifecycle)
    (ws / "tasks" / "TASKS.md").write_text("# Tasks\n\n", encoding="utf-8")
    (ws / "decisions" / "DECISIONS.md").write_text("# Decisions\n\n", encoding="utf-8")
    return str(ws)


@pytest.fixture
def workspace(tmp_path: Path) -> Iterator[str]:
    """A workspace that has opted lifecycle receipts IN."""
    ws = _make_workspace(tmp_path, lifecycle=True)
    try:
        yield ws
    finally:
        evict_gate(ws)


@pytest.fixture
def silent_workspace(tmp_path: Path) -> Iterator[str]:
    """A workspace that has said nothing — i.e. the shipped default."""
    ws = _make_workspace(tmp_path, lifecycle=None)
    try:
        yield ws
    finally:
        evict_gate(ws)


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------


def _evidence(ws: str) -> list[dict]:
    path = os.path.join(ws, "memory", "evidence_chain.jsonl")
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sidecar(ws: str) -> list[dict]:
    path = os.path.join(ws, ".mind-mem-audit", "chain.jsonl")
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _rows_with_verb(ws: str, verb: str) -> list[dict]:
    """Rows for one lifecycle verb.

    Selects on ``metadata[LIFECYCLE_VERB_KEY]`` rather than on ``action``,
    because every lifecycle verb writes the SAME action -- ROLLBACK -- so that
    a reader from an older release can still parse the record. Selecting on
    action here would match all three verbs at once and the tests below would
    stop distinguishing them.
    """
    return [
        r
        for r in _evidence(ws)
        if r.get("action") == EvidenceAction.ROLLBACK.value and (r.get("metadata") or {}).get(LIFECYCLE_VERB_KEY) == verb
    ]


def _rows_with_operation(ws: str, operation: str) -> list[dict]:
    return [r for r in _sidecar(ws) if r.get("operation") == operation]


def _tier_manager(ws: str) -> TierManager:
    db_path = os.path.join(ws, "intelligence", "tiers.db")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    return TierManager(db_path, workspace=ws)


def _tier_state(ws: str, block_id: str) -> MemoryTier:
    mgr = TierManager(os.path.join(ws, "intelligence", "tiers.db"))
    try:
        return mgr.get_tier(block_id)
    finally:
        mgr.close()


def _age_tier_row(ws: str, block_id: str, *, days: int) -> None:
    """Backdate a tier row so a decay cycle on the real clock convicts it."""
    stamp = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    conn = sqlite3.connect(os.path.join(ws, "intelligence", "tiers.db"))
    try:
        conn.execute("UPDATE block_tiers SET updated_at = ? WHERE id = ?", (stamp, block_id))
        conn.commit()
    finally:
        conn.close()


def _seed_task(ws: str, bid: str, status: str = "done", date: str = OLD) -> str:
    with open(os.path.join(ws, "tasks", "TASKS.md"), "a", encoding="utf-8") as handle:
        handle.write(f"[{bid}]\nTitle: task {bid}\nDate: {date}\nStatus: {status}\n\n---\n")
    return bid


# ---------------------------------------------------------------------------
# A — the contract: three verbs, three operations, one mapping
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTheVocabulary:
    def test_no_lifecycle_verb_invents_an_evidence_action_member(self) -> None:
        """Forward compatibility, the same contract governing DELETE obeys.

        ``EvidenceObject.from_dict`` does a strict ``EvidenceAction(value)``
        lookup, so a member invented here makes an older reader fail to load a
        chain this release wrote. RA.3's own text asks for DEMOTE/ARCHIVE/
        FORGET members; that text predates the lock and the lock wins.
        """
        assert {m.value for m in EvidenceAction} == {
            "PROPOSE",
            "APPLY",
            "ROLLBACK",
            "CONTRADICT",
            "DRIFT",
            "RESOLVE",
            "VERIFY",
        }

    def test_every_verb_writes_an_action_an_older_reader_knows(self) -> None:
        for verb, action in EVIDENCE_ACTION_FOR_VERB.items():
            assert action is EvidenceAction.ROLLBACK, f"{verb} would break an older reader"

    def test_the_sidecar_accepts_all_three_lifecycle_operations(self) -> None:
        assert LIFECYCLE_OPERATIONS <= VALID_OPERATIONS, sorted(LIFECYCLE_OPERATIONS - VALID_OPERATIONS)

    def test_every_lifecycle_action_maps_to_exactly_one_operation(self) -> None:
        assert OPERATION_FOR_VERB == {
            DEMOTE_VERB: DEMOTE_OPERATION,
            ARCHIVE_VERB: ARCHIVE_OPERATION,
            FORGET_VERB: FORGET_OPERATION,
        }
        assert set(OPERATION_FOR_VERB.values()) == LIFECYCLE_OPERATIONS

    def test_a_lifecycle_operation_is_appendable_to_the_sidecar(self, tmp_path: Path) -> None:
        """The positive control for the gate above: the verb really is accepted."""
        chain = AuditChain(str(tmp_path))
        for operation in sorted(LIFECYCLE_OPERATIONS):
            entry = chain.append(operation, "D-1", agent="tester")
            assert entry.operation == operation
        ok, errors = chain.verify()
        assert ok, errors


# ---------------------------------------------------------------------------
# B — the flag, and what OFF costs
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTheFlag:
    def test_a_workspace_that_says_nothing_is_off(self, silent_workspace: str) -> None:
        assert lifecycle_evidence_enabled(silent_workspace) is False
        assert LifecycleRecorder.for_workspace(silent_workspace) is None

    def test_exactly_one_value_opts_in(self, tmp_path: Path) -> None:
        ws = Path(_make_workspace(tmp_path, lifecycle=True))
        assert lifecycle_evidence_enabled(str(ws)) is True
        for value in (False, "true", 1, None):
            (ws / "mind-mem.json").write_text(json.dumps({"lifecycle_evidence": {"enabled": value}}), encoding="utf-8")
            assert lifecycle_evidence_enabled(str(ws)) is False, value

    def test_no_workspace_is_off_and_reads_nothing(self) -> None:
        assert LifecycleRecorder.for_workspace(None) is None
        assert LifecycleRecorder.for_workspace("") is None

    def test_an_off_workspace_gains_no_ledger_and_no_directory(self, silent_workspace: str) -> None:
        mgr = _tier_manager(silent_workspace)
        try:
            mgr._register_block("D-off", MemoryTier.SHARED)
            assert mgr.demote("D-off", MemoryTier.WORKING, DemotionReason.STALE) is True
        finally:
            mgr.close()
        # Positive control: the demotion really happened.
        assert _tier_state(silent_workspace, "D-off") == MemoryTier.WORKING
        assert _evidence(silent_workspace) == []
        assert not os.path.exists(os.path.join(silent_workspace, ".mind-mem-audit")), (
            "the OFF path must not create the artifact it declined to write"
        )


# ---------------------------------------------------------------------------
# C — the three doors
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDemotionLeavesAReceipt:
    def test_a_demotion_writes_one_row_into_each_ledger(self, workspace: str) -> None:
        assert _evidence(workspace) == [], "the ledger must start empty or the counts below prove nothing"

        mgr = _tier_manager(workspace)
        try:
            mgr._register_block("D-20240101-001", MemoryTier.LONG_TERM)
            assert mgr.demote("D-20240101-001", MemoryTier.SHARED, DemotionReason.STALE) is True
        finally:
            mgr.close()

        assert _tier_state(workspace, "D-20240101-001") == MemoryTier.SHARED, "positive control: the block really was demoted"

        rows = _rows_with_verb(workspace, DEMOTE_VERB)
        assert len(rows) == 1, _evidence(workspace)
        meta = rows[0]["metadata"]
        assert rows[0]["target_block_id"] == "D-20240101-001"
        assert meta["subject"] == SUBJECT_TIER_ASSIGNMENT
        assert meta["from_tier"] == "LONG_TERM"
        assert meta["to_tier"] == "SHARED"
        assert meta["reason_code"] == DemotionReason.STALE.value
        assert meta["door"] == "memory_tiers.TierManager.demote"

        sidecar = _rows_with_operation(workspace, DEMOTE_OPERATION)
        assert len(sidecar) == 1, _sidecar(workspace)
        assert sidecar[0]["target"] == "D-20240101-001"

    def test_a_refused_demotion_leaves_nothing(self, workspace: str) -> None:
        """A receipt is minted for a death, never for an attempt."""
        mgr = _tier_manager(workspace)
        try:
            mgr._register_block("D-20240101-002", MemoryTier.WORKING)
            # A promotion dressed as a demotion — refused by ``demote``.
            assert mgr.demote("D-20240101-002", MemoryTier.VERIFIED, DemotionReason.MANUAL) is False
        finally:
            mgr.close()
        assert _rows_with_verb(workspace, DEMOTE_VERB) == []
        assert _rows_with_operation(workspace, DEMOTE_OPERATION) == []


@pytest.mark.unit
class TestEvictionLeavesAReceipt:
    def test_an_eviction_writes_a_forget_row_naming_what_died(self, workspace: str) -> None:
        mgr = _tier_manager(workspace)
        try:
            mgr._register_block("D-20240101-003", MemoryTier.WORKING)
            future = datetime.now(timezone.utc) + timedelta(days=365)
            _demotions, evicted = mgr.run_decay_cycle(now=future)
        finally:
            mgr.close()

        assert evicted == ["D-20240101-003"], "positive control: the decay cycle really evicted it"

        rows = _rows_with_verb(workspace, FORGET_VERB)
        assert len(rows) == 1, _evidence(workspace)
        meta = rows[0]["metadata"]
        assert rows[0]["target_block_id"] == "D-20240101-003"
        assert meta["subject"] == SUBJECT_TIER_ASSIGNMENT, (
            "an eviction destroys the ladder standing, not the block — the record must not claim the block died"
        )
        assert meta["from_tier"] == "WORKING"
        assert meta["door"] == "memory_tiers.TierManager._evict"
        assert len(_rows_with_operation(workspace, FORGET_OPERATION)) == 1


@pytest.mark.unit
class TestArchiveLeavesAReceiptPerBlock:
    def test_every_archived_block_gets_its_own_row(self, workspace: str) -> None:
        first = _seed_task(workspace, "T-20240101-001")
        second = _seed_task(workspace, "T-20240101-002")
        assert f"[{first}]" in Path(workspace, "tasks", "TASKS.md").read_text(encoding="utf-8")

        actions = archive_completed_blocks(workspace, days=30)

        assert len(actions) == 2, actions
        assert f"[{second}]" in Path(workspace, "tasks", "TASKS_ARCHIVE.md").read_text(encoding="utf-8"), (
            "positive control: the block really did move"
        )

        rows = _rows_with_verb(workspace, ARCHIVE_VERB)
        assert sorted(r["target_block_id"] for r in rows) == [first, second], _evidence(workspace)
        meta = rows[0]["metadata"]
        assert meta["subject"] == SUBJECT_BLOCK
        assert meta["from_file"] == "tasks/TASKS.md"
        assert meta["to_file"] == "tasks/TASKS_ARCHIVE.md"
        assert meta["door"] == "compaction.archive_completed_blocks"
        assert meta["admission_entry_id"], "an archive receipt must name the batch admission that authorised the move"
        assert len(_rows_with_operation(workspace, ARCHIVE_OPERATION)) == 2

    def test_a_dry_run_archives_nothing_and_records_nothing(self, workspace: str) -> None:
        _seed_task(workspace, "T-20240101-003")
        actions = archive_completed_blocks(workspace, days=30, dry_run=True)
        assert actions and all("[dry-run]" in a for a in actions)
        assert _rows_with_verb(workspace, ARCHIVE_VERB) == []


# ---------------------------------------------------------------------------
# D — the receipt must not break the thing it is written into
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTheLedgersStaySound:
    def test_both_chains_still_verify_after_every_lifecycle_verb(self, workspace: str) -> None:
        _seed_task(workspace, "T-20240101-004")
        archive_completed_blocks(workspace, days=30)
        mgr = _tier_manager(workspace)
        try:
            mgr._register_block("D-20240101-004", MemoryTier.SHARED)
            mgr.demote("D-20240101-004", MemoryTier.WORKING, DemotionReason.LOW_CONFIDENCE)
            mgr.run_decay_cycle(now=datetime.now(timezone.utc) + timedelta(days=365))
        finally:
            mgr.close()

        # The verbs live in metadata, not in the action: every lifecycle loss
        # writes ROLLBACK so an older reader can still parse the chain. Assert
        # both halves -- the verbs are all present AND none of them leaked into
        # the action vocabulary, which is the forward-compatibility contract.
        verbs = {
            (r.get("metadata") or {}).get(LIFECYCLE_VERB_KEY)
            for r in _evidence(workspace)
            if (r.get("metadata") or {}).get(LIFECYCLE_VERB_KEY)
        }
        assert {DEMOTE_VERB, ARCHIVE_VERB, FORGET_VERB} <= verbs, verbs
        actions = {r["action"] for r in _evidence(workspace)}
        assert not ({"DEMOTE", "ARCHIVE", "FORGET"} & actions), f"a lifecycle verb reached the action vocabulary: {actions}"

        evidence = EvidenceChain(store_path=os.path.join(workspace, "memory", "evidence_chain.jsonl"))
        ok, broken = evidence.verify_chain()
        assert ok, broken

        sidecar_ok, errors = AuditChain(workspace).verify()
        assert sidecar_ok, errors

    def test_the_reconciler_does_not_read_a_lifecycle_row_as_a_lost_admission(self, workspace: str) -> None:
        """``cross_ledger`` counts evidence rows that must have a hash-chain twin.

        A lifecycle receipt has none — it is written straight to the
        evidence chain — so it must not carry the marker the reconciler
        counts on, or every demotion would convict the workspace of a
        truncated hash chain.
        """
        from mind_mem.cross_ledger import reconcile

        mgr = _tier_manager(workspace)
        try:
            mgr._register_block("D-20240101-005", MemoryTier.VERIFIED)
            for target in (MemoryTier.LONG_TERM, MemoryTier.SHARED, MemoryTier.WORKING):
                assert mgr.demote("D-20240101-005", target, DemotionReason.STALE) is True
        finally:
            mgr.close()

        assert len(_rows_with_verb(workspace, DEMOTE_VERB)) == 3
        verdict = reconcile(workspace)
        assert verdict.ok, verdict.reasons
        assert verdict.admission_rows == 0, "a lifecycle receipt is not an admission row"

    def test_a_receipt_carries_ids_and_enum_names_but_no_block_text(self, workspace: str) -> None:
        """The sidecar and the chain are read by tooling well outside the gate."""
        body = "SECRET-BODY-TEXT-that-must-never-reach-a-ledger"
        _seed_task(workspace, "T-20240101-006", status="done")
        path = Path(workspace, "tasks", "TASKS.md")
        path.write_text(path.read_text(encoding="utf-8").replace("task T-20240101-006", body), encoding="utf-8")
        assert body in path.read_text(encoding="utf-8"), "positive control: the text is really in the corpus"

        archive_completed_blocks(workspace, days=30)

        assert _rows_with_verb(workspace, ARCHIVE_VERB), "positive control: a receipt was written at all"
        assert body not in json.dumps(_evidence(workspace))
        assert body not in json.dumps(_sidecar(workspace))


# ---------------------------------------------------------------------------
# E — the wired path: the console script, not a hand-built manager
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTheCompactionEntryPointRecords:
    def test_the_tier_sweep_that_mind_mem_compact_runs_leaves_receipts(self, workspace: str) -> None:
        """``mind-mem-compact`` -> ``main`` -> ``_run_tier_promotion`` -> decay."""
        from mind_mem.compaction import _run_tier_promotion

        mgr = _tier_manager(workspace)
        try:
            mgr._register_block("D-20240101-007", MemoryTier.SHARED)
            mgr._register_block("D-20240101-008", MemoryTier.WORKING)
        finally:
            mgr.close()
        _age_tier_row(workspace, "D-20240101-007", days=365)
        _age_tier_row(workspace, "D-20240101-008", days=365)

        _run_tier_promotion(workspace)

        assert len(_rows_with_verb(workspace, DEMOTE_VERB)) == 1, _evidence(workspace)
        assert len(_rows_with_verb(workspace, FORGET_VERB)) == 1, _evidence(workspace)
