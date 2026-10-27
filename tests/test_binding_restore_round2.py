# Copyright 2026 STARGA, Inc.
"""Second round of governance-integrity regressions for the binding and restore doors.

Each class reproduces one failure an independent check found on the first
round of fixes:

B1. A workspace armed by ``init`` (or by ``config set`` on top of it) whose
    config was tampered with and whose binding was deleted BEFORE its first
    governed write was admitted: no arming history existed anywhere.
B2. The arming history lived only in ``memory/evidence_chain.jsonl``.
    Deleting that file together with the binding disarmed the workspace.
B3. ``_mark_proposal_status`` reported success when the proposal's
    ``Status:`` line came before its ``ProposalId:`` line (it wrote
    nothing), so apply succeeded, wrote an ``applied`` receipt and left the
    proposal ``staged`` to be applied again.
B4. Deleting ``APPLY_RECEIPT.md`` from an already-rolled-back snapshot let a
    second rollback rewind every later write.
B5. A symlinked receipt skipped the state check; the restore ran and only
    afterwards did rollback report failure.
B6. A snapshot directory with no receipt was restored as a rollback target.

Every negative assertion is paired with a positive control.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterator

import pytest

from mind_mem import apply_engine
from mind_mem.apply_engine import apply_proposal, compute_fingerprint, create_snapshot, rollback
from mind_mem.enums import IngestTier
from mind_mem.governance_gate import GovernanceBypassError, config_path_for, evict_gate, get_gate
from mind_mem.init_workspace import init
from mind_mem.mm_cli import ConfigSetError, config_set
from mind_mem.spec_binding import SpecBindingManager
from mind_mem.storage import get_block_store

BINDING = ".spec_binding.json"
PROPOSAL_ID = "P-20261008-001"
TARGET_ID = "D-20261008-001"
DECISION_FILE = "decisions/DECISIONS.md"
PROPOSED = "intelligence/proposed/DECISIONS_PROPOSED.md"
LATER_ID = "D-20261008-009"


def _write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def _governed_write(ws: str, block_id: str) -> None:
    gate = get_gate(ws)
    store = get_block_store(ws)
    with gate.admit_block("WRITE", block_id, "body", tier=IngestTier.EXTERNAL_INGEST):
        store.write_block({"_id": block_id, "Statement": "body", "Status": "quarantined", "Date": "2026-10-08"})


def _tamper(ws: str, **changes: object) -> None:
    path = config_path_for(ws)
    with open(path, encoding="utf-8") as handle:
        config = json.load(handle)
    config.update(changes)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(config, handle)


def _stored(ws: str, block_id: str) -> bool:
    return get_block_store(ws).get_by_id(block_id) is not None


@pytest.fixture
def fresh_ws(tmp_path: Path) -> Iterator[str]:
    ws = str(tmp_path / "ws")
    init(ws)
    try:
        yield ws
    finally:
        evict_gate(ws)


# ---------------------------------------------------------------------------
# B1 — init arms the workspace durably, before any governed write
# ---------------------------------------------------------------------------


class TestArmedAtInitBeforeFirstWrite:
    def test_positive_control_untampered_first_write_admits(self, fresh_ws: str) -> None:
        evict_gate(fresh_ws)
        _governed_write(fresh_ws, "D-20261008-100")
        assert _stored(fresh_ws, "D-20261008-100")

    def test_init_tamper_delete_binding_before_first_write_is_refused(self, fresh_ws: str) -> None:
        _tamper(fresh_ws, governance_mode="detect_only", tampered=True)
        os.remove(os.path.join(fresh_ws, BINDING))
        evict_gate(fresh_ws)
        with pytest.raises(GovernanceBypassError):
            _governed_write(fresh_ws, "D-20261008-101")
        assert not _stored(fresh_ws, "D-20261008-101")

    def test_config_set_enforce_then_delete_before_first_write_is_refused(self, fresh_ws: str) -> None:
        config_set(config_path_for(fresh_ws), "governance_mode", "enforce")
        evict_gate(fresh_ws)
        _tamper(fresh_ws, governance_mode="detect_only")
        os.remove(os.path.join(fresh_ws, BINDING))
        evict_gate(fresh_ws)
        with pytest.raises(GovernanceBypassError):
            _governed_write(fresh_ws, "D-20261008-102")
        assert not _stored(fresh_ws, "D-20261008-102")

    def test_config_set_refuses_on_armed_workspace_with_binding_missing(self, fresh_ws: str) -> None:
        os.remove(os.path.join(fresh_ws, BINDING))
        with pytest.raises(ConfigSetError):
            config_set(config_path_for(fresh_ws), "governance_mode", "detect_only")
        assert not os.path.exists(os.path.join(fresh_ws, BINDING)), "config set must not silently re-arm either"

    def test_rebind_after_disarm_restores_writes(self, fresh_ws: str) -> None:
        os.remove(os.path.join(fresh_ws, BINDING))
        evict_gate(fresh_ws)
        with pytest.raises(GovernanceBypassError):
            _governed_write(fresh_ws, "D-20261008-103")
        path = config_path_for(fresh_ws)
        SpecBindingManager(path).bind(path)
        _governed_write(fresh_ws, "D-20261008-104")
        assert _stored(fresh_ws, "D-20261008-104")


# ---------------------------------------------------------------------------
# B2 — arming evidence survives deletion of any single file
# ---------------------------------------------------------------------------


class TestArmingEvidenceSurvivesSingleFileDeletion:
    def _armed_with_history(self, ws: str) -> None:
        config_set(config_path_for(ws), "governance_mode", "enforce")
        _governed_write(ws, "D-20261008-201")
        evict_gate(ws)
        _tamper(ws, governance_mode="detect_only")
        os.remove(os.path.join(ws, BINDING))

    def test_binding_and_evidence_ledger_deleted_is_refused(self, fresh_ws: str) -> None:
        self._armed_with_history(fresh_ws)
        os.remove(os.path.join(fresh_ws, "memory", "evidence_chain.jsonl"))
        with pytest.raises(GovernanceBypassError):
            _governed_write(fresh_ws, "D-20261008-202")
        assert not _stored(fresh_ws, "D-20261008-202")

    @pytest.mark.parametrize("which", [0, 1])
    def test_binding_ledger_and_one_arming_record_deleted_is_refused(self, fresh_ws: str, which: int) -> None:
        self._armed_with_history(fresh_ws)
        os.remove(os.path.join(fresh_ws, "memory", "evidence_chain.jsonl"))
        from mind_mem.spec_binding import arming_record_paths

        records = arming_record_paths(config_path_for(fresh_ws), fresh_ws)
        assert len(records) >= 2 and all(os.path.isfile(p) for p in records), "positive control: init recorded arming"
        os.remove(records[which])
        with pytest.raises(GovernanceBypassError):
            _governed_write(fresh_ws, "D-20261008-203")

    def test_every_arming_record_exists_after_init(self, fresh_ws: str) -> None:
        from mind_mem.spec_binding import arming_record_paths

        records = arming_record_paths(config_path_for(fresh_ws), fresh_ws)
        assert len(set(records)) >= 2
        for record in records:
            assert os.path.isfile(record) and not os.path.islink(record)


# ---------------------------------------------------------------------------
# B3 — a status write that does not land fails the apply
# ---------------------------------------------------------------------------


def _staged(ws: str, *, status_first: bool) -> None:
    config_set(config_path_for(ws), "governance_mode", "propose")
    ops = [{"op": "set_status", "file": DECISION_FILE, "target": TARGET_ID, "status": "superseded"}]
    fingerprint = compute_fingerprint({"Type": "edit", "TargetBlock": TARGET_ID, "Ops": ops})
    if status_first:
        head = f"\n[{PROPOSAL_ID}]\nStatus: staged\nProposalId: {PROPOSAL_ID}\n"
    else:
        head = f"\n[{PROPOSAL_ID}]\nProposalId: {PROPOSAL_ID}\nStatus: staged\n"
    _write(
        os.path.join(ws, PROPOSED),
        head + f"Type: edit\nTargetBlock: {TARGET_ID}\nRisk: low\nEvidence:\n- fixture\nRollback: snap\n"
        f"Fingerprint: {fingerprint}\nOps:\n- op: set_status\n  file: {DECISION_FILE}\n"
        f"  target: {TARGET_ID}\n  status: superseded\n",
    )
    _write(os.path.join(ws, DECISION_FILE), f"\n[{TARGET_ID}]\nId: {TARGET_ID}\nStatus: active\nTitle: fixture\n")


def _proposal_status_lines(ws: str) -> list[str]:
    return [ln for ln in Path(ws, PROPOSED).read_text(encoding="utf-8").splitlines() if ln.startswith("Status:")]


def _receipts(ws: str) -> list[str]:
    applied = os.path.join(ws, "intelligence", "applied")
    return sorted(d for d in os.listdir(applied) if os.path.isfile(os.path.join(applied, d, "APPLY_RECEIPT.md")))


def _last_result(ws: str, ts: str) -> str:
    text = Path(ws, "intelligence", "applied", ts, "APPLY_RECEIPT.md").read_text(encoding="utf-8")
    return [ln.split(":", 1)[1].strip() for ln in text.splitlines() if ln.startswith("Result:")][-1]


@pytest.fixture
def patched_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(apply_engine, "check_preconditions", lambda ws: (True, ["validate: PASS (patched)"]))
    monkeypatch.setattr(apply_engine, "check_no_touch_window", lambda ws: (True, "patched: window elapsed"))


class TestStatusLineOrder:
    @pytest.mark.parametrize("status_first", [False, True])
    def test_apply_marks_applied_whatever_the_field_order(self, fresh_ws: str, patched_checks: None, status_first: bool) -> None:
        _staged(fresh_ws, status_first=status_first)
        ok, msg = apply_proposal(fresh_ws, PROPOSAL_ID)
        assert ok, msg
        assert _proposal_status_lines(fresh_ws) == ["Status: applied"]
        ok2, msg2 = apply_proposal(fresh_ws, PROPOSAL_ID)
        assert not ok2, f"an applied proposal cannot be applied again: {msg2}"

    def test_missing_status_line_reports_not_written(self, tmp_path: Path) -> None:
        path = str(tmp_path / "p.md")
        _write(path, f"\n[{PROPOSAL_ID}]\nProposalId: {PROPOSAL_ID}\nType: edit\n")
        assert apply_engine._mark_proposal_status(path, PROPOSAL_ID, "applied") is False

    def test_status_of_a_different_block_is_never_written(self, tmp_path: Path) -> None:
        path = str(tmp_path / "p.md")
        text = f"\n[{PROPOSAL_ID}]\nProposalId: {PROPOSAL_ID}\nType: edit\n\n[P-20261008-002]\nProposalId: P-20261008-002\nStatus: staged\n"
        _write(path, text)
        assert apply_engine._mark_proposal_status(path, PROPOSAL_ID, "applied") is False
        assert Path(path).read_text(encoding="utf-8") == text

    def test_prefix_id_does_not_match_longer_id(self, tmp_path: Path) -> None:
        path = str(tmp_path / "p.md")
        text = "\n[P-20261008-0011]\nProposalId: P-20261008-0011\nStatus: staged\n"
        _write(path, text)
        assert apply_engine._mark_proposal_status(path, PROPOSAL_ID, "applied") is False
        assert Path(path).read_text(encoding="utf-8") == text


# ---------------------------------------------------------------------------
# B4/B5/B6 — rollback needs a real, applied receipt, checked before restore
# ---------------------------------------------------------------------------


def _append_later_block(ws: str) -> None:
    path = os.path.join(ws, DECISION_FILE)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(f"\n[{LATER_ID}]\nId: {LATER_ID}\nStatus: active\nTitle: later\n")


def _later_survives(ws: str) -> bool:
    return LATER_ID in Path(ws, DECISION_FILE).read_text(encoding="utf-8")


@pytest.fixture
def applied_ws(fresh_ws: str, patched_checks: None) -> str:
    _staged(fresh_ws, status_first=False)
    ok, msg = apply_proposal(fresh_ws, PROPOSAL_ID)
    assert ok, msg
    return fresh_ws


class TestRollbackNeedsAnAppliedReceipt:
    def test_positive_control_first_rollback_succeeds(self, applied_ws: str) -> None:
        ts = _receipts(applied_ws)[-1]
        assert rollback(applied_ws, ts) is True
        assert _last_result(applied_ws, ts) == "rolled_back"

    def test_deleted_receipt_after_rollback_is_refused_before_restore(self, applied_ws: str) -> None:
        ts = _receipts(applied_ws)[-1]
        assert rollback(applied_ws, ts) is True
        _append_later_block(applied_ws)
        assert _later_survives(applied_ws), "positive control"
        os.remove(os.path.join(applied_ws, "intelligence", "applied", ts, "APPLY_RECEIPT.md"))
        assert rollback(applied_ws, ts) is False
        assert _later_survives(applied_ws), "a refused rollback must not rewind anything"

    def test_symlinked_receipt_is_refused_before_restore(self, applied_ws: str, tmp_path: Path) -> None:
        ts = _receipts(applied_ws)[-1]
        assert rollback(applied_ws, ts) is True
        _append_later_block(applied_ws)
        receipt = os.path.join(applied_ws, "intelligence", "applied", ts, "APPLY_RECEIPT.md")
        outside = tmp_path / "fake.md"
        outside.write_text(f"Proposal: {PROPOSAL_ID}\nResult: applied\n", encoding="utf-8")
        os.remove(receipt)
        os.symlink(str(outside), receipt)
        assert rollback(applied_ws, ts) is False
        assert _later_survives(applied_ws), "refusal must happen before the restore"

    def test_symlinked_receipt_inside_snapshot_is_refused(self, applied_ws: str) -> None:
        ts = _receipts(applied_ws)[-1]
        snap = os.path.join(applied_ws, "intelligence", "applied", ts)
        receipt = os.path.join(snap, "APPLY_RECEIPT.md")
        real = os.path.join(snap, "REAL.md")
        os.replace(receipt, real)
        os.symlink(real, receipt)
        _append_later_block(applied_ws)
        assert rollback(applied_ws, ts) is False
        assert _later_survives(applied_ws)

    def test_receiptless_snapshot_is_never_a_rollback_target(self, fresh_ws: str, patched_checks: None) -> None:
        _write(os.path.join(fresh_ws, DECISION_FILE), f"\n[{TARGET_ID}]\nId: {TARGET_ID}\nStatus: active\nTitle: f\n")
        create_snapshot(fresh_ws, "20261008-111111", files_touched=[DECISION_FILE])
        _append_later_block(fresh_ws)
        assert rollback(fresh_ws, "20261008-111111") is False
        assert _later_survives(fresh_ws)

    def test_receipt_for_a_rolled_back_proposal_is_refused(self, applied_ws: str) -> None:
        """The receipt says applied but the proposal itself says rolled back: refused."""
        ts = _receipts(applied_ws)[-1]
        assert apply_engine._mark_proposal_status(os.path.join(applied_ws, PROPOSED), PROPOSAL_ID, "rolled_back")
        _append_later_block(applied_ws)
        assert rollback(applied_ws, ts) is False
        assert _later_survives(applied_ws)


class TestArmingRecordIsDeterministic:
    def test_two_inits_write_byte_identical_arming_records(self, tmp_path: Path) -> None:
        from mind_mem.spec_binding import arming_record_paths

        blobs = []
        for name in ("a", "b"):
            ws = str(tmp_path / name)
            init(ws)
            try:
                blobs.append([Path(p).read_bytes() for p in arming_record_paths(config_path_for(ws), ws)])
            finally:
                evict_gate(ws)
        assert blobs[0] and blobs[0] == blobs[1], "no clock in the arming record"
