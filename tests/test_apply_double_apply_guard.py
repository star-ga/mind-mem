"""apply must re-validate proposal status UNDER the workspace lock.

Regression: apply_proposal() validates Status=='staged' BEFORE acquiring
the lock, so two concurrent applies of the same proposal both pass, then
serialize on the lock and the second re-applies an already-applied
proposal (double-apply). _apply_proposal_locked now re-reads the on-disk
proposal under the lock and aborts if it is missing or no longer staged,
BEFORE any WAL replay / snapshot / mutation.
"""

from __future__ import annotations

from pathlib import Path

from mind_mem.apply_engine import (
    _apply_proposal_locked,
    _get_workspace_lock_path,
)
from mind_mem.init_workspace import init
from mind_mem.mind_filelock import FileLock


def test_apply_locked_aborts_when_not_staged_on_disk(tmp_path: Path) -> None:
    ws = str(tmp_path)
    init(ws)

    # A stale snapshot a concurrent caller would hold after passing the
    # pre-lock validation — but on disk there is no such staged proposal
    # (the winning apply already consumed it / it never existed).
    stale = {
        "ProposalId": "P-20260605-001",
        "Type": "edit",
        "TargetBlock": "D-20260605-001",
        "Risk": "low",
        "Status": "staged",
        "Evidence": "x",
        "Rollback": "x",
        "Fingerprint": "deadbeef",
        "Ops": [],
    }

    lock = FileLock(_get_workspace_lock_path(ws), timeout=5.0)
    lock.acquire()
    try:
        ok, msg = _apply_proposal_locked(ws, stale, "P-20260605-001", "proposed/x.md", lock)
    finally:
        lock.release()

    assert ok is False
    assert "no longer" in msg.lower() or "not staged" in msg.lower() or "no longer present" in msg.lower()
