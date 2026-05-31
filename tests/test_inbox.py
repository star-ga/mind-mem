"""Tests for the v3.9 inbox folder ingestion."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from mind_mem.inbox import (
    ROUTING_TABLE,
    InboxWatcher,
    classify_file,
    ingest_text_file,
    process_file,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def workspace(tmp_path: Path) -> str:
    ws = tmp_path / "ws"
    (ws / "memory").mkdir(parents=True)
    (ws / "decisions").mkdir(parents=True)
    config = {
        "version": "3.9.0",
        "workspace_path": str(ws),
        "block_store": {"backend": "markdown"},
    }
    (ws / "mind-mem.json").write_text(json.dumps(config))
    return str(ws)


@pytest.fixture
def inbox(tmp_path: Path) -> Path:
    d = tmp_path / "inbox"
    d.mkdir()
    return d


# ---------------------------------------------------------------------------
# classify_file
# ---------------------------------------------------------------------------


class TestClassifyFile:
    def test_text_extensions(self) -> None:
        for ext in (".txt", ".md", ".json", ".csv", ".log", ".xml", ".yaml", ".yml"):
            assert classify_file(f"some/file{ext}") == "text", ext

    def test_image_extensions(self) -> None:
        for ext in (".png", ".jpg", ".jpeg", ".gif", ".webp"):
            assert classify_file(f"some/file{ext}") == "image", ext

    def test_audio_extensions(self) -> None:
        for ext in (".mp3", ".wav", ".flac", ".m4a"):
            assert classify_file(f"some/file{ext}") == "audio", ext

    def test_pdf_extension(self) -> None:
        assert classify_file("doc.pdf") == "pdf"

    def test_unknown_returns_none(self) -> None:
        assert classify_file("file.bin") is None
        assert classify_file("noextension") is None

    def test_case_insensitive(self) -> None:
        assert classify_file("DOC.PDF") == "pdf"
        assert classify_file("Notes.MD") == "text"

    def test_routing_table_complete(self) -> None:
        # Sanity: every routed handler must be one of the four known names.
        assert set(ROUTING_TABLE.values()) <= {"text", "image", "audio", "pdf"}


# ---------------------------------------------------------------------------
# ingest_text_file (real workspace, real BlockStore)
# ---------------------------------------------------------------------------


class TestIngestTextFile:
    def test_writes_block(self, workspace: str, tmp_path: Path) -> None:
        f = tmp_path / "note.md"
        f.write_text("# Hello\nThis is a test note for inbox ingestion.")
        block_id = ingest_text_file(workspace, str(f))
        assert block_id.startswith("INBOX-")
        assert "note" in block_id

    def test_empty_file_still_writes(self, workspace: str, tmp_path: Path) -> None:
        f = tmp_path / "blank.txt"
        f.write_text("")
        block_id = ingest_text_file(workspace, str(f))
        assert block_id.startswith("INBOX-")

    def test_oversized_text_rejected(self, workspace: str, tmp_path: Path) -> None:
        f = tmp_path / "big.txt"
        # Write a bit over the 4 MiB cap.
        f.write_bytes(b"a" * (4 * 1024 * 1024 + 100))
        with pytest.raises(ValueError, match="too large"):
            ingest_text_file(workspace, str(f))

    def test_safe_filename_sanitized(self, workspace: str, tmp_path: Path) -> None:
        f = tmp_path / "weird name & punct!.txt"
        f.write_text("content")
        block_id = ingest_text_file(workspace, str(f))
        # All non-alphanumeric chars (other than - and _) must be replaced.
        suffix = block_id.split("-", 2)[-1]
        assert all(c.isalnum() or c in ("-", "_") for c in suffix), suffix


# ---------------------------------------------------------------------------
# process_file — staging behaviour
# ---------------------------------------------------------------------------


class TestProcessFile:
    def test_success_moves_to_processed(self, workspace: str, inbox: Path) -> None:
        f = inbox / "good.txt"
        f.write_text("ok content")
        processed = inbox / "_processed"
        failed = inbox / "_failed"
        result = process_file(workspace, str(f), processed_dir=str(processed), failed_dir=str(failed))
        assert result.ok is True
        assert result.handler == "text"
        assert result.block_id is not None
        # File no longer at root.
        assert not f.exists()
        # File exists under _processed/<ts>/
        moved = list(processed.rglob("good.txt"))
        assert len(moved) == 1

    def test_unknown_extension_moves_to_failed(self, workspace: str, inbox: Path) -> None:
        f = inbox / "unknown.bin"
        f.write_text("opaque")
        processed = inbox / "_processed"
        failed = inbox / "_failed"
        result = process_file(workspace, str(f), processed_dir=str(processed), failed_dir=str(failed))
        assert result.ok is False
        assert result.handler == "unknown"
        moved = list(failed.rglob("unknown.bin"))
        assert len(moved) == 1
        # Sidecar error file
        errs = list(failed.rglob("unknown.bin.error.txt"))
        assert len(errs) == 1
        assert "unsupported extension" in errs[0].read_text()

    def test_image_handler_fails_with_clear_error(self, workspace: str, inbox: Path) -> None:
        f = inbox / "pic.png"
        f.write_bytes(b"not really an image")
        processed = inbox / "_processed"
        failed = inbox / "_failed"
        result = process_file(workspace, str(f), processed_dir=str(processed), failed_dir=str(failed))
        assert result.ok is False
        assert result.handler == "image"
        assert result.error is not None
        assert "multimodal" in result.error  # points at the optional extra


# ---------------------------------------------------------------------------
# InboxWatcher
# ---------------------------------------------------------------------------


class TestInboxWatcher:
    def test_empty_workspace_rejected(self, inbox: Path) -> None:
        with pytest.raises(ValueError, match="workspace"):
            InboxWatcher("", str(inbox))

    def test_empty_inbox_rejected(self, workspace: str) -> None:
        with pytest.raises(ValueError, match="inbox"):
            InboxWatcher(workspace, "")

    def test_too_short_interval_rejected(self, workspace: str, inbox: Path) -> None:
        with pytest.raises(ValueError, match="interval"):
            InboxWatcher(workspace, str(inbox), interval=0.1)

    def test_directories_created(self, workspace: str, inbox: Path) -> None:
        InboxWatcher(workspace, str(inbox))
        assert (inbox / "_processed").is_dir()
        assert (inbox / "_failed").is_dir()

    def test_process_once_handles_text(self, workspace: str, inbox: Path) -> None:
        (inbox / "first.md").write_text("First memory")
        (inbox / "second.txt").write_text("Second memory")
        watcher = InboxWatcher(workspace, str(inbox))
        results = watcher.process_once()
        assert len(results) == 2
        assert all(r.ok for r in results)

    def test_process_once_skips_staging_dirs(self, workspace: str, inbox: Path) -> None:
        watcher = InboxWatcher(workspace, str(inbox))
        # Drop a file inside _processed that should not be re-ingested.
        (inbox / "_processed" / "old.txt").write_text("already done")
        results = watcher.process_once()
        assert results == []

    def test_callback_fires_per_result(self, workspace: str, inbox: Path) -> None:
        (inbox / "x.md").write_text("hi")
        seen: list[str] = []
        watcher = InboxWatcher(workspace, str(inbox), on_result=lambda r: seen.append(r.handler))
        watcher.process_once()
        assert seen == ["text"]

    def test_callback_exception_does_not_break(self, workspace: str, inbox: Path) -> None:
        (inbox / "y.md").write_text("hi")

        def boom(_r) -> None:
            raise RuntimeError("callback exploded")

        watcher = InboxWatcher(workspace, str(inbox), on_result=boom)
        # Must not raise even though the callback does.
        results = watcher.process_once()
        assert len(results) == 1

    def test_files_processed_in_mtime_order(self, workspace: str, inbox: Path) -> None:
        first = inbox / "a.md"
        first.write_text("A")
        time.sleep(0.05)
        second = inbox / "b.md"
        second.write_text("B")
        watcher = InboxWatcher(workspace, str(inbox))
        results = watcher.process_once()
        assert [Path(r.path).name for r in results] == ["a.md", "b.md"]

    def test_start_stop_lifecycle(self, workspace: str, inbox: Path) -> None:
        watcher = InboxWatcher(workspace, str(inbox), interval=0.5)
        watcher.start()
        # Drop a file after the watcher is running.
        (inbox / "live.md").write_text("live ingest")
        # Wait up to 3 seconds for the watcher to pick it up.
        deadline = time.time() + 3.0
        while time.time() < deadline:
            if not (inbox / "live.md").exists():
                break
            time.sleep(0.1)
        watcher.stop()
        # File should have been ingested + moved.
        assert not (inbox / "live.md").exists()
        moved = list((inbox / "_processed").rglob("live.md"))
        assert len(moved) == 1
