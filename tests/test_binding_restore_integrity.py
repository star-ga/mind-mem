# Copyright 2026 STARGA, Inc.
"""Governance integrity of the spec binding, every restore door, and proposal states.

Six defects, each with a RED test written against base ``d4a5605d``:

1. Deleting ``.spec_binding.json`` on an armed workspace silently lifted the
   drift check: the gate fell into its "unbound" branch and admitted.
2. The binding verified the absolute path recorded at bind time, so a copied
   workspace attested the SOURCE workspace's config, not its own.
3. ``backup_restore.restore_workspace(force=True)`` rewound ``mind-mem.json``
   (the apply-engine door had been fixed, this one had not), leaving an
   ``enforce`` workspace write-blocked.
4. ``apply_engine._live_config_kept`` let a snapshot's config land when no live
   config existed.
5. ``apply_proposal`` ignored ``_mark_proposal_status``'s return: an apply
   whose status write failed reported success, wrote an ``applied`` receipt
   and left the proposal ``staged`` (so it could be applied again).
6. ``rollback`` restored a receipt whatever its state, so a receipt already
   rolled back was rolled back a second time.

And one labelling defect: ``conflict_resolver`` wrote ``pending-review``
proposals with no Ops that nothing can apply, without saying so.

Every negative assertion carries a positive control.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Iterator

import pytest

from mind_mem import apply_engine
from mind_mem.apply_engine import (
    apply_proposal,
    compute_fingerprint,
    create_snapshot,
    restore_snapshot,
    rollback,
)
from mind_mem.backup_restore import backup_workspace, restore_workspace
from mind_mem.conflict_resolver import ResolutionStrategy, generate_resolution_proposals
from mind_mem.enums import IngestTier
from mind_mem.governance_gate import (
    DRIFT_VERB,
    GovernanceBypassError,
    config_path_for,
    evict_gate,
    get_gate,
)
from mind_mem.init_workspace import init
from mind_mem.mm_cli import config_set
from mind_mem.spec_binding import SpecBindingManager
from mind_mem.storage import get_block_store

BINDING = ".spec_binding.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def _governed_write(ws: str, block_id: str) -> None:
    gate = get_gate(ws)
    store = get_block_store(ws)
    with gate.admit_block("WRITE", block_id, "body", tier=IngestTier.EXTERNAL_INGEST):
        store.write_block({"_id": block_id, "Statement": "body", "Status": "quarantined", "Date": "2026-10-08"})


def _evidence_rows(ws: str) -> list[dict[str, Any]]:
    path = os.path.join(ws, "memory", "evidence_chain.jsonl")
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _disarm_rows(ws: str) -> list[dict[str, Any]]:
    return [
        r
        for r in _evidence_rows(ws)
        if (r.get("metadata") or {}).get("action_verb") == DRIFT_VERB and (r.get("metadata") or {}).get("binding_disarmed")
    ]


@pytest.fixture
def armed_ws(tmp_path: Path) -> Iterator[str]:
    ws = str(tmp_path / "ws")
    init(ws)
    assert os.path.isfile(os.path.join(ws, BINDING)), "init must arm the workspace"
    try:
        yield ws
    finally:
        evict_gate(ws)


# ---------------------------------------------------------------------------
# 1 — deleting the binding is a disarm, not an "unbound" workspace
# ---------------------------------------------------------------------------


class TestDeletedBindingIsADisarm:
    def test_positive_control_an_armed_workspace_admits(self, armed_ws: str) -> None:
        _governed_write(armed_ws, "D-20261008-001")
        assert get_block_store(armed_ws).get_by_id("D-20261008-001") is not None

    def test_same_process_deleted_binding_refuses_next_write(self, armed_ws: str) -> None:
        _governed_write(armed_ws, "D-20261008-001")
        os.remove(os.path.join(armed_ws, BINDING))
        with pytest.raises(GovernanceBypassError):
            _governed_write(armed_ws, "D-20261008-002")
        assert get_block_store(armed_ws).get_by_id("D-20261008-002") is None
        assert len(_disarm_rows(armed_ws)) == 1, "the disarm must be recorded in the ledger"

    def test_fresh_process_deleted_binding_refuses_next_write(self, armed_ws: str) -> None:
        # The armed history lives in the evidence chain: a write admitted
        # under the binding carries its spec_hash.
        _governed_write(armed_ws, "D-20261008-001")
        evict_gate(armed_ws)
        os.remove(os.path.join(armed_ws, BINDING))
        with pytest.raises(GovernanceBypassError):
            _governed_write(armed_ws, "D-20261008-002")
        assert len(_disarm_rows(armed_ws)) == 1
        # And the disarm record itself keeps it refused in the next process.
        evict_gate(armed_ws)
        with pytest.raises(GovernanceBypassError):
            _governed_write(armed_ws, "D-20261008-003")

    def test_rebinding_re_arms_and_writes_resume(self, armed_ws: str) -> None:
        _governed_write(armed_ws, "D-20261008-001")
        os.remove(os.path.join(armed_ws, BINDING))
        with pytest.raises(GovernanceBypassError):
            _governed_write(armed_ws, "D-20261008-002")
        path = config_path_for(armed_ws)
        SpecBindingManager(path).bind(path)
        _governed_write(armed_ws, "D-20261008-003")
        assert get_block_store(armed_ws).get_by_id("D-20261008-003") is not None

    def test_never_armed_workspace_still_admits(self, tmp_path: Path) -> None:
        """Negative control: an unbound workspace with no armed history admits."""
        ws = str(tmp_path / "legacy")
        for sub in ("decisions", "tasks", "memory", "intelligence", "entities", "summaries"):
            os.makedirs(os.path.join(ws, sub), exist_ok=True)
        _write(os.path.join(ws, "mind-mem.json"), json.dumps({"governance_mode": "enforce"}))
        try:
            _governed_write(ws, "D-20261008-001")
            _governed_write(ws, "D-20261008-002")
            assert _disarm_rows(ws) == []
        finally:
            evict_gate(ws)


# ---------------------------------------------------------------------------
# 2 — a copied workspace verifies its OWN config
# ---------------------------------------------------------------------------


class TestCopiedWorkspaceVerifiesItsOwnConfig:
    def test_edit_in_copy_is_drift_in_copy(self, armed_ws: str, tmp_path: Path) -> None:
        copy = str(tmp_path / "copy")
        shutil.copytree(armed_ws, copy)
        path = config_path_for(copy)
        cfg = json.loads(Path(path).read_text(encoding="utf-8"))
        cfg["governance_mode"] = "enforce"
        cfg["tampered"] = True
        _write(path, json.dumps(cfg))
        valid, _reason = SpecBindingManager(path).verify()
        assert not valid, "the copy's own edited config must not verify"
        try:
            with pytest.raises(GovernanceBypassError):
                _governed_write(copy, "D-20261008-001")
        finally:
            evict_gate(copy)

    def test_edit_in_source_is_not_drift_in_copy(self, armed_ws: str, tmp_path: Path) -> None:
        copy = str(tmp_path / "copy")
        shutil.copytree(armed_ws, copy)
        # Positive control: the copy verifies before anything changes.
        assert SpecBindingManager(config_path_for(copy)).verify()[0]
        src = config_path_for(armed_ws)
        cfg = json.loads(Path(src).read_text(encoding="utf-8"))
        cfg["tampered"] = True
        _write(src, json.dumps(cfg))
        assert not SpecBindingManager(src).verify()[0], "the source did drift"
        valid, reason = SpecBindingManager(config_path_for(copy)).verify()
        assert valid, f"an edit to another workspace's config is not drift here: {reason}"


# ---------------------------------------------------------------------------
# 3/4 — every restore door keeps the live, attested config
# ---------------------------------------------------------------------------


class TestRestoreKeepsLiveConfig:
    def test_backup_restore_force_keeps_live_config_and_stays_writable(self, armed_ws: str, tmp_path: Path) -> None:
        path = config_path_for(armed_ws)
        config_set(path, "governance_mode", "enforce")
        _governed_write(armed_ws, "D-20261008-001")
        archive = backup_workspace(armed_ws, str(tmp_path / "b.tar.gz"))
        config_set(path, "proposal_budget.backlog_limit", 500)
        live = Path(path).read_bytes()
        assert b"500" in live  # positive control: the live config differs from the archive's

        restore_workspace(armed_ws, archive, force=True)

        assert Path(path).read_bytes() == live, "a restore rewinds memory, never configuration"
        assert SpecBindingManager(path).verify()[0]
        _governed_write(armed_ws, "D-20261008-002")
        assert get_block_store(armed_ws).get_by_id("D-20261008-002") is not None

    def test_restore_snapshot_does_not_land_config_when_none_live(self, tmp_path: Path) -> None:
        ws = str(tmp_path / "ws")
        for sub in ("decisions", "tasks", "memory", "intelligence", "entities", "summaries"):
            os.makedirs(os.path.join(ws, sub), exist_ok=True)
        _write(os.path.join(ws, "decisions", "DECISIONS.md"), "# Decisions\n\n")
        cfg = os.path.join(ws, "mind-mem.json")
        _write(cfg, json.dumps({"governance_mode": "enforce"}))
        try:
            snap = create_snapshot(ws, "20261008-000000")
            assert os.path.isfile(os.path.join(snap, "mind-mem.json")), "positive control: the snapshot holds a config"
            os.remove(cfg)
            restore_snapshot(ws, snap)
            assert not os.path.exists(cfg), "the snapshot's config must not land where no live config was"
        finally:
            evict_gate(ws)


# ---------------------------------------------------------------------------
# 5/6 — the proposal state machine
# ---------------------------------------------------------------------------

PROPOSAL_ID = "P-20261008-001"
TARGET_ID = "D-20261008-001"
DECISION_FILE = "decisions/DECISIONS.md"
PROPOSED = "intelligence/proposed/DECISIONS_PROPOSED.md"


@pytest.fixture
def staged_ws(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    ws = str(tmp_path / "ws")
    init(ws)
    config_set(config_path_for(ws), "governance_mode", "propose")
    ops = [{"op": "set_status", "file": DECISION_FILE, "target": TARGET_ID, "status": "superseded"}]
    fingerprint = compute_fingerprint({"Type": "edit", "TargetBlock": TARGET_ID, "Ops": ops})
    _write(
        os.path.join(ws, PROPOSED),
        f"\n[{PROPOSAL_ID}]\nProposalId: {PROPOSAL_ID}\nType: edit\nTargetBlock: {TARGET_ID}\n"
        f"Risk: low\nStatus: staged\nEvidence:\n- fixture\nRollback: restore the snapshot\n"
        f"Fingerprint: {fingerprint}\nOps:\n- op: set_status\n  file: {DECISION_FILE}\n"
        f"  target: {TARGET_ID}\n  status: superseded\n",
    )
    _write(os.path.join(ws, DECISION_FILE), f"\n[{TARGET_ID}]\nId: {TARGET_ID}\nStatus: active\nTitle: fixture\n")
    monkeypatch.setattr(apply_engine, "check_preconditions", lambda ws: (True, ["validate: PASS (patched)"]))
    try:
        yield ws
    finally:
        evict_gate(ws)


def _proposal_status(ws: str) -> str:
    for line in Path(ws, PROPOSED).read_text(encoding="utf-8").splitlines():
        if line.startswith("Status:"):
            return line.split(":", 1)[1].strip()
    raise AssertionError("no Status line")


def _receipts(ws: str) -> list[str]:
    applied = os.path.join(ws, "intelligence", "applied")
    return sorted(d for d in os.listdir(applied) if os.path.isfile(os.path.join(applied, d, "APPLY_RECEIPT.md")))


def _last_result(ws: str, ts: str) -> str:
    text = Path(ws, "intelligence", "applied", ts, "APPLY_RECEIPT.md").read_text(encoding="utf-8")
    return [ln.split(":", 1)[1].strip() for ln in text.splitlines() if ln.startswith("Result:")][-1]


class TestStatusWriteFailureFailsTheApply:
    def test_positive_control_apply_marks_applied(self, staged_ws: str) -> None:
        ok, msg = apply_proposal(staged_ws, PROPOSAL_ID)
        assert ok, msg
        assert _proposal_status(staged_ws) == "applied"
        assert _last_result(staged_ws, _receipts(staged_ws)[-1]) == "applied"

    def test_status_write_failure_reports_failure_and_rolls_back(self, staged_ws: str, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(apply_engine, "_mark_proposal_status", lambda *a, **k: False)
        ok, msg = apply_proposal(staged_ws, PROPOSAL_ID)
        assert not ok, f"apply must fail when its status cannot be recorded: {msg}"
        ts = _receipts(staged_ws)[-1]
        assert _last_result(staged_ws, ts) != "applied", "no applied receipt beside a staged proposal"
        assert _proposal_status(staged_ws) == "staged"
        decision = Path(staged_ws, DECISION_FILE).read_text(encoding="utf-8")
        assert "Status: active" in decision, "the ops must be withdrawn"


class TestRollbackRefusesNonApplied:
    def test_second_rollback_of_same_receipt_is_refused(self, staged_ws: str) -> None:
        ok, msg = apply_proposal(staged_ws, PROPOSAL_ID)
        assert ok, msg
        ts = _receipts(staged_ws)[-1]
        assert rollback(staged_ws, ts) is True  # positive control
        assert _last_result(staged_ws, ts) == "rolled_back"
        assert rollback(staged_ws, ts) is False, "a rolled-back receipt cannot be rolled back again"

    def test_in_progress_receipt_is_refused(self, staged_ws: str) -> None:
        ok, msg = apply_proposal(staged_ws, PROPOSAL_ID)
        assert ok, msg
        ts = _receipts(staged_ws)[-1]
        receipt = Path(staged_ws, "intelligence", "applied", ts, "APPLY_RECEIPT.md")
        receipt.write_text(receipt.read_text(encoding="utf-8").replace("Result: applied", "Result: in_progress"), encoding="utf-8")
        decision_before = Path(staged_ws, DECISION_FILE).read_text(encoding="utf-8")
        assert rollback(staged_ws, ts) is False
        assert Path(staged_ws, DECISION_FILE).read_text(encoding="utf-8") == decision_before, "nothing restored"


# ---------------------------------------------------------------------------
# 7 — conflict_resolver output is clearly advisory
# ---------------------------------------------------------------------------


class TestResolutionProposalsAreAdvisory:
    def test_resolution_proposal_is_labelled_advisory(self, tmp_path: Path) -> None:
        resolutions = [
            {
                "strategy": ResolutionStrategy.TIMESTAMP,
                "confidence": "high",
                "contradiction_id": "C-1",
                "block_a": "D-A",
                "block_b": "D-B",
                "hash_a": "a",
                "hash_b": "b",
                "winner_id": "D-B",
                "loser_id": "D-A",
                "rationale": "B is newer",
            }
        ]
        assert generate_resolution_proposals(str(tmp_path), resolutions) == 1
        text = (tmp_path / "intelligence" / "proposed" / "RESOLUTIONS_PROPOSED.md").read_text(encoding="utf-8")
        assert "Advisory: true" in text
        assert "Ops:" not in text
        # And the apply engine indeed cannot apply it: it is not in any file it reads.
        proposal, _src = apply_engine.find_proposal(str(tmp_path), "R-" + text.split("[R-", 1)[1].split("]", 1)[0])
        assert proposal is None
