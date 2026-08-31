"""v3.2.0 §1.4 PR-3 — MarkdownBlockStore.snapshot / restore / diff tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from mind_mem.block_store import MarkdownBlockStore


@pytest.fixture()
def ws(tmp_path: Path) -> Path:
    """Minimal workspace with a decisions file and a root config file."""
    (tmp_path / "decisions").mkdir()
    (tmp_path / "decisions" / "DECISIONS.md").write_text("[D-001]\nStatement: initial\nStatus: active\n\n---\n")
    (tmp_path / "intelligence").mkdir()
    (tmp_path / "memory").mkdir()
    (tmp_path / "AGENTS.md").write_text("agents config\n")
    return tmp_path


@pytest.fixture()
def snap_dir(tmp_path: Path, ws: Path) -> Path:
    """Pre-created snapshot directory path (not yet populated)."""
    d = tmp_path / "snaps" / "20260419-120000"
    return d


class TestStoreSnapshot:
    def test_snapshot_creates_manifest(self, ws: Path, snap_dir: Path) -> None:
        store = MarkdownBlockStore(str(ws))
        manifest = store.snapshot(str(snap_dir))

        assert snap_dir.is_dir(), "snap_dir should be created"
        assert (snap_dir / "MANIFEST.json").is_file(), "MANIFEST.json must exist"
        assert isinstance(manifest, dict)
        assert "files" in manifest
        assert "version" in manifest

    def test_snapshot_captures_workspace_file(self, ws: Path, snap_dir: Path) -> None:
        store = MarkdownBlockStore(str(ws))
        store.snapshot(str(snap_dir))

        # AGENTS.md is in SNAPSHOT_FILES and must be captured
        assert (snap_dir / "AGENTS.md").is_file()
        assert (snap_dir / "AGENTS.md").read_text() == "agents config\n"


class TestStoreRestore:
    def test_restore_is_idempotent(self, ws: Path, snap_dir: Path) -> None:
        store = MarkdownBlockStore(str(ws))
        store.snapshot(str(snap_dir))

        # Mutate workspace
        (ws / "decisions" / "DECISIONS.md").write_text("[D-001]\nStatement: mutated\n\n---\n")

        # First restore
        store.restore(str(snap_dir))
        content_after_first = (ws / "decisions" / "DECISIONS.md").read_text()
        assert "initial" in content_after_first

        # Second restore (idempotent — should not raise and result must be the same)
        store.restore(str(snap_dir))
        content_after_second = (ws / "decisions" / "DECISIONS.md").read_text()
        assert content_after_second == content_after_first


class TestStoreDiff:
    def test_diff_empty_after_restore(self, ws: Path, snap_dir: Path) -> None:
        store = MarkdownBlockStore(str(ws))
        store.snapshot(str(snap_dir))

        # Mutate, then restore
        (ws / "decisions" / "DECISIONS.md").write_text("[D-001]\nStatement: mutated\n\n---\n")
        assert store.diff(str(snap_dir)) != [], "diff should be non-empty after mutation"

        store.restore(str(snap_dir))
        remaining = store.diff(str(snap_dir))
        assert remaining == [], f"diff should be empty after restore, got: {remaining}"


class TestRestoreWithDamagedSnapshot:
    """A manifest entry whose snapshot copy is gone cannot be restored.

    The old code copied nothing, warned about nothing, and still counted
    the entry in ``file_count`` — so a rollback that left the mutated
    corpus in place logged as a complete restore.
    """

    def _damage(self, ws: Path, snap_dir: Path) -> None:
        store = MarkdownBlockStore(str(ws))
        store.snapshot(str(snap_dir))
        (snap_dir / "decisions" / "DECISIONS.md").unlink()
        (ws / "decisions" / "DECISIONS.md").write_text("[D-001]\nStatement: mutated\n\n---\n")

    def test_missing_source_is_reported_not_counted_as_restored(self, ws: Path, snap_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from mind_mem import block_store as bs

        self._damage(ws, snap_dir)
        events: list[tuple[str, str, dict]] = []
        monkeypatch.setattr(
            bs._log,
            "warning",
            lambda event, **kw: events.append(("warning", event, kw)),
        )
        monkeypatch.setattr(
            bs._log,
            "info",
            lambda event, **kw: events.append(("info", event, kw)),
        )

        MarkdownBlockStore(str(ws)).restore(str(snap_dir))

        warned = [kw for lvl, event, kw in events if lvl == "warning" and event == "restore_missing_snapshot_source"]
        assert [kw["entry"] for kw in warned] == ["decisions/DECISIONS.md"]

        (summary,) = [kw for lvl, event, kw in events if lvl == "info" and event == "block_store_restore"]
        assert summary["missing"] == 1
        assert summary["complete"] is False
        # AGENTS.md really was restored; DECISIONS.md was not. The count must
        # not include the entry nothing was copied for.
        assert summary["file_count"] == 1

    def test_intact_restore_still_reports_complete(self, ws: Path, snap_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from mind_mem import block_store as bs

        store = MarkdownBlockStore(str(ws))
        store.snapshot(str(snap_dir))
        (ws / "decisions" / "DECISIONS.md").write_text("[D-001]\nStatement: mutated\n\n---\n")

        events: list[dict] = []
        monkeypatch.setattr(bs._log, "info", lambda event, **kw: events.append({"event": event, **kw}))
        store.restore(str(snap_dir))

        (summary,) = [e for e in events if e["event"] == "block_store_restore"]
        assert summary["missing"] == 0
        assert summary["complete"] is True
        assert summary["file_count"] == 2
        assert "initial" in (ws / "decisions" / "DECISIONS.md").read_text()

    def test_unrestorable_file_is_not_deleted_as_an_orphan(self, ws: Path, snap_dir: Path) -> None:
        """The live file stays: the snapshot is damaged, not the workspace."""
        self._damage(ws, snap_dir)
        MarkdownBlockStore(str(ws)).restore(str(snap_dir))
        assert (ws / "decisions" / "DECISIONS.md").is_file()
        assert "mutated" in (ws / "decisions" / "DECISIONS.md").read_text()
