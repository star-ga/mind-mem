"""v3.2.0 §2.2 — tests for maintenance/ subdivision migration."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from mind_mem.maintenance_migrate import (
    already_migrated,
    classify_maintenance_file,
    migrate_maintenance,
)


@pytest.mark.parametrize(
    "name,expected",
    [
        ("validation-report.txt", "append-only"),
        ("compaction-2026-04-01.log", "append-only"),
        ("intel-scan-2026-04-01.ndjson", "append-only"),
        ("something.log", "append-only"),
        ("dedup-state.json", "tracked"),
        ("compaction-checkpoint.json", "tracked"),
        ("workspace.lock", "tracked"),
        ("unknown.json", "tracked"),
        ("README.md", "tracked"),
    ],
)
def test_classify(name: str, expected: str) -> None:
    assert classify_maintenance_file(name) == expected


class TestMigrate:
    def _mkws(self) -> Path:
        d = Path(tempfile.mkdtemp())
        (d / "mind-mem.json").write_text("{}")
        (d / "maintenance").mkdir()
        return d

    def test_empty_workspace(self) -> None:
        d = self._mkws()
        counts = migrate_maintenance(str(d), verbose=False)
        assert counts == {"tracked": 0, "append-only": 0}

    def test_classifies_and_moves(self) -> None:
        d = self._mkws()
        (d / "maintenance" / "dedup-state.json").write_text("{}")
        (d / "maintenance" / "validation-report.txt").write_text("")
        (d / "maintenance" / "mystery.bin").write_text("\x00")

        counts = migrate_maintenance(str(d), verbose=False)
        assert counts["tracked"] >= 2
        assert counts["append-only"] >= 1

        assert (d / "maintenance" / "tracked" / "dedup-state.json").exists()
        assert (d / "maintenance" / "append-only" / "validation-report.txt").exists()
        assert (d / "maintenance" / "tracked" / "mystery.bin").exists()

    def test_idempotent(self) -> None:
        d = self._mkws()
        (d / "maintenance" / "dedup-state.json").write_text("{}")

        migrate_maintenance(str(d), verbose=False)
        # Second call — tracked/append-only subdirs already exist.
        assert already_migrated(str(d))
        counts = migrate_maintenance(str(d), verbose=False)
        assert counts == {"tracked": 0, "append-only": 0}

    def test_does_not_clobber_existing_dst(self) -> None:
        d = self._mkws()
        (d / "maintenance" / "tracked").mkdir()
        # Pre-populate a file at the destination path.
        (d / "maintenance" / "tracked" / "mystery.json").write_text("keep me")
        # ``already_migrated`` returns True so migration is a no-op.
        counts = migrate_maintenance(str(d), verbose=False)
        assert counts == {"tracked": 0, "append-only": 0}
        assert (d / "maintenance" / "tracked" / "mystery.json").read_text() == "keep me"
