"""Rolling back a correction restores memory, never the configuration.

Reproduction (internal check 2026-10-08): a correction was applied, then an
unrelated setting was changed through the supported path (re-attested). The
rollback restored the snapshot's ``mind-mem.json`` too, silently undoing the
later setting; its own closing record then failed the spec-binding check
(GovernanceBypassError, "spec-hash drifted") and the workspace stayed
write-blocked.
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import pytest

from mind_mem.apply_engine import apply_proposal, compute_fingerprint, rollback
from mind_mem.block_parser import parse_file
from mind_mem.governance_gate import GovernanceGate, config_path_for
from mind_mem.init_workspace import init
from mind_mem.mm_cli import config_set

PID = "P-20261008-001"
DID = "D-20261001-001"


def _stage(ws: str) -> None:
    with open(os.path.join(ws, "decisions", "DECISIONS.md"), "a", encoding="utf-8") as handle:
        handle.write(
            f"\n[{DID}]\nDate: 2026-10-01\nStatus: active\nScope: project:azariel\n"
            "Statement: Weekly planning call with Ilona is on Tuesday at 10am Arizona time.\n"
            "Rationale: seed.\nSupersedes: none\nTags: schedule\nSources: test seed\n\n"
        )
    ops = [{"op": "set_status", "file": "decisions/DECISIONS.md", "target": DID, "status": "superseded"}]
    fp = compute_fingerprint({"ProposalId": PID, "Type": "edit", "TargetBlock": DID, "Ops": ops})
    proposed = os.path.join(ws, "intelligence", "proposed")
    os.makedirs(proposed, exist_ok=True)
    with open(os.path.join(proposed, "DECISIONS_PROPOSED.md"), "a", encoding="utf-8") as handle:
        handle.write(
            f"\n[{PID}]\nProposalId: {PID}\nType: edit\nTargetBlock: {DID}\nRisk: low\nStatus: staged\n"
            f"Evidence: Ilona correction\nRollback: set status back to active\nFingerprint: {fp}\n"
            f"Ops:\n- op: set_status\n  file: decisions/DECISIONS.md\n  target: {DID}\n  status: superseded\n"
        )


def _status(ws: str) -> str:
    return {b["_id"]: b.get("Status") for b in parse_file(os.path.join(ws, "decisions", "DECISIONS.md"))}[DID]


@pytest.fixture()
def applied(tmp_path: Path) -> tuple[str, str]:
    ws = str(tmp_path / "ws")
    init(ws)
    config_set(config_path_for(ws), "governance_mode", "propose")
    _stage(ws)
    ok, _ = apply_proposal(ws, PID)
    assert ok, "apply failed; the rollback tests below would be vacuous"
    assert _status(ws) == "superseded"
    receipt = os.path.basename(sorted(glob.glob(os.path.join(ws, "intelligence", "applied", "*")))[-1])
    # A later, unrelated, properly attested setting change.
    config_set(config_path_for(ws), "proposal_budget.per_run", 4)
    return ws, receipt


def test_rollback_restores_the_memory(applied):
    ws, receipt = applied
    assert rollback(ws, receipt, reason="undo the correction") is True
    assert _status(ws) == "active"


def test_rollback_keeps_later_config_changes(applied):
    ws, receipt = applied
    rollback(ws, receipt, reason="undo the correction")
    with open(config_path_for(ws), encoding="utf-8") as handle:
        assert json.load(handle)["proposal_budget"]["per_run"] == 4


def test_workspace_accepts_writes_after_rollback(applied):
    ws, receipt = applied
    rollback(ws, receipt, reason="undo the correction")
    gate = GovernanceGate(ws)
    try:
        gate.admit(action="WRITE", block_id="D-20261009-001", content="body", actor="test")
    finally:
        gate.close()
