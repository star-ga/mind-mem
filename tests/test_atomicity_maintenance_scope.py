"""v3.2.0 §2.2 — regression test for the ``maintenance/`` atomicity fix.

Before the fix (landed in ``c835a39``), ``apply_engine.create_snapshot``
excluded ``maintenance/`` wholesale from the snapshot. A multi-stage
apply could write to a behavioural state file like
``maintenance/dedup-state.json`` after the corpus snapshot was taken,
crash, roll back the corpus, and leave the dedup hash behind —
causing the next apply to silently skip the rolled-back blocks.

This test pins the fix: files under ``maintenance/tracked/`` survive
the snapshot → mutate → restore round-trip, while files under
``maintenance/append-only/`` are deliberately *not* restored so
append-only observability (reports, logs) isn't clobbered by rollback.

The test uses the apply-engine's public ``create_snapshot`` +
``restore_snapshot`` APIs directly so it exercises the real snapshot
scope, not a re-implementation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mind_mem.apply_engine import create_snapshot, restore_snapshot


@pytest.fixture
def ws(tmp_path: Path) -> Path:
    """Minimal workspace with the v3.2.0 maintenance/ layout in place."""
    (tmp_path / "mind-mem.json").write_text("{}")
    (tmp_path / "decisions").mkdir()
    (tmp_path / "maintenance" / "tracked").mkdir(parents=True)
    (tmp_path / "maintenance" / "append-only").mkdir(parents=True)
    (tmp_path / "intelligence" / "applied").mkdir(parents=True)
    return tmp_path


def _dedup_state_path(ws: Path) -> Path:
    return ws / "maintenance" / "tracked" / "dedup-state.json"


def _audit_log_path(ws: Path) -> Path:
    return ws / "maintenance" / "append-only" / "validation-report.txt"


class TestAtomicityMaintenanceScope:
    def test_tracked_state_is_restored_on_rollback(self, ws: Path) -> None:
        """maintenance/tracked/*.json roundtrips through a snapshot restore.

        Reproduces the §2 scenario from docs/v3.2.0-atomicity-scope-plan.md:
        a dedup-state.json written before a crash must be rolled back so
        the *next* apply doesn't silently skip re-writing rolled-back blocks.
        """
        state = _dedup_state_path(ws)
        pre_content = {"hash": "pre-apply", "seen": ["block-a", "block-b"]}
        state.write_text(json.dumps(pre_content))

        snap_dir = create_snapshot(str(ws), "20260420-000000")
        assert Path(snap_dir).is_dir()

        # Simulate Stage B writing a new hash then the process crashing
        # before the apply commits.
        state.write_text(json.dumps({"hash": "crashed-mid-apply", "seen": ["block-a", "block-b", "block-c"]}))

        restore_snapshot(str(ws), snap_dir)

        # The tracked state file must be restored to pre-apply content —
        # otherwise the "block-c already seen" hash survives the rollback
        # and the next apply skips re-writing it.
        assert state.exists(), "dedup-state.json was not restored from snapshot"
        assert json.loads(state.read_text()) == pre_content

    def test_append_only_log_is_not_restored(self, ws: Path) -> None:
        """maintenance/append-only/*.log is NOT restored.

        Append-only observability output (validation reports, logs)
        must survive a rollback as-is — the whole point of the
        append-only subdir is to preserve signal that rollback would
        otherwise discard.
        """
        # Also populate a tracked file so the snapshot is non-trivial.
        _dedup_state_path(ws).write_text(json.dumps({"hash": "v1"}))

        report = _audit_log_path(ws)
        report.write_text("pre-snapshot line\n")
        snap_dir = create_snapshot(str(ws), "20260420-000001")
        report.write_text("pre-snapshot line\nappended-during-apply\n")

        restore_snapshot(str(ws), snap_dir)

        # Restore must NOT revert the append — the append-only semantic
        # is that rollback preserves whatever was written after the
        # snapshot, because those entries are signal (errors logged during
        # the failed apply, validation progress, etc.).
        assert "appended-during-apply" in report.read_text(), (
            "append-only file was reverted by restore_snapshot — "
            "atomicity exclusion rule violated"
        )

    def test_restore_creates_missing_tracked_file(self, ws: Path) -> None:
        """A tracked file present at snapshot-time is recreated on restore.

        If the state file is deleted (or never written) between
        snapshot and restore, the restore must put it back. Otherwise
        the next apply would start from a clean slate even though the
        rolled-back corpus still references the tracked entries.
        """
        state = _dedup_state_path(ws)
        state.write_text(json.dumps({"hash": "original"}))

        snap_dir = create_snapshot(str(ws), "20260420-000002")

        # Simulate Stage B deleting the state file before crashing.
        state.unlink()

        restore_snapshot(str(ws), snap_dir)

        assert state.exists(), "deleted tracked file was not recreated on restore"
        assert json.loads(state.read_text()) == {"hash": "original"}
