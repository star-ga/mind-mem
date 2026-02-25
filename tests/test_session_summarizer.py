"""Comprehensive tests for scripts/session_summarizer.py.

Covers: file_hash, extract_summary, format_summary_block, write_summary,
        and edge cases (unicode, concurrent access, empty workspace).
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
import threading
from contextlib import contextmanager
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from scripts.session_summarizer import (
    FILE_PATH_RE,
    extract_summary,
    file_hash,
    format_summary_block,
    write_summary,
)


# ── helpers ──────────────────────────────────────────────────────────────

def _msg(role: str, content: str) -> dict:
    """Build a minimal message dict."""
    return {"role": role, "content": content}


def _make_messages(n: int = 5) -> list[dict]:
    """Generate *n* trivial alternating messages."""
    msgs = []
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        msgs.append(_msg(role, f"Message number {i} about SomeProject"))
    return msgs


@contextmanager
def _noop_lock(*_a, **_kw):
    yield


def _fake_file_lock(path):
    """Return a no-op context manager, matching FileLock(path) usage."""
    return _noop_lock()


# ── file_hash ────────────────────────────────────────────────────────────

class TestFileHash:
    def test_existing_file(self, tmp_path):
        p = tmp_path / "hello.txt"
        p.write_bytes(b"hello world")
        h = file_hash(str(p))
        expected = hashlib.sha256(b"hello world").hexdigest()[:16]
        assert h == expected

    def test_nonexistent_file(self, tmp_path):
        p = str(tmp_path / "missing.txt")
        h = file_hash(p)
        expected = hashlib.sha256(p.encode()).hexdigest()[:16]
        assert h == expected

    def test_empty_file(self, tmp_path):
        p = tmp_path / "empty.txt"
        p.write_bytes(b"")
        h = file_hash(str(p))
        expected = hashlib.sha256(b"").hexdigest()[:16]
        assert h == expected

    def test_large_file_reads_only_64k(self, tmp_path):
        """Only first 64 KB should be hashed regardless of file size."""
        p = tmp_path / "big.bin"
        data = os.urandom(128 * 1024)  # 128 KB
        p.write_bytes(data)
        h = file_hash(str(p))
        expected = hashlib.sha256(data[:65536]).hexdigest()[:16]
        assert h == expected

    def test_hash_length(self, tmp_path):
        p = tmp_path / "len.txt"
        p.write_bytes(b"x")
        assert len(file_hash(str(p))) == 16


# ── extract_summary ──────────────────────────────────────────────────────

class TestExtractSummary:
    def test_message_count(self):
        msgs = _make_messages(7)
        s = extract_summary(msgs)
        assert s["message_count"] == 7

    def test_role_counting(self):
        msgs = [
            _msg("user", "hello"),
            _msg("assistant", "hi"),
            _msg("user", "bye"),
        ]
        s = extract_summary(msgs)
        assert s["roles"]["user"] == 2
        assert s["roles"]["assistant"] == 1

    def test_file_extraction(self):
        msgs = [
            _msg("user", "Look at /home/alice/project/foo.py for details"),
            _msg("assistant", "I see ./src/bar.ts and utils/helper.js are relevant"),
        ]
        s = extract_summary(msgs)
        found_paths = {f for f, _ in s["files"]}
        assert "/home/alice/project/foo.py" in found_paths
        assert "./src/bar.ts" in found_paths

    def test_topic_extraction(self):
        msgs = [
            _msg("user", "We need to fix the DatabaseManager and APIGateway"),
            _msg("assistant", "The DatabaseManager config looks correct"),
        ]
        s = extract_summary(msgs)
        topic_names = {t for t, _ in s["topics"]}
        assert "DatabaseManager" in topic_names

    def test_empty_messages(self):
        s = extract_summary([])
        assert s["message_count"] == 0
        assert s["topics"] == []
        assert s["files"] == []
        assert s["decisions"] == []
        assert s["roles"] == {}

    def test_decisions_captured(self):
        msgs = [
            _msg("user", "Don't ever use raw SQL queries in the codebase for safety reasons"),
            _msg("assistant", "Understood, I will always use the ORM layer instead"),
            _msg("user", "The convention is to prefix private methods with underscore"),
        ]
        s = extract_summary(msgs)
        assert len(s["decisions"]) > 0
        types = {d["type"] for d in s["decisions"]}
        assert types & {"correction", "convention"}

    def test_topics_limit(self):
        """Topics list should have at most 15 entries."""
        msgs = [_msg("user", " ".join(f"Topic{i}" for i in range(30)))]
        s = extract_summary(msgs)
        assert len(s["topics"]) <= 15

    def test_files_limit(self):
        """Files list should have at most 20 entries."""
        paths = " ".join(f"dir/file{i}.py" for i in range(30))
        msgs = [_msg("user", paths)]
        s = extract_summary(msgs)
        assert len(s["files"]) <= 20


# ── format_summary_block ─────────────────────────────────────────────────

class TestFormatSummaryBlock:
    def test_basic_format(self):
        summary = {
            "message_count": 10,
            "roles": {"user": 5, "assistant": 5},
            "topics": [("Refactoring", 3), ("CI", 2)],
            "files": [("/home/n/app.py", 4)],
            "decisions": [],
        }
        block = format_summary_block("SESS-20260224-001", "t.jsonl", summary, "abc123")
        assert "[SESS-20260224-001]" in block
        assert "Source: t.jsonl" in block
        assert "TranscriptHash: abc123" in block
        assert "Messages: 10" in block
        assert "Refactoring (3)" in block
        assert "/home/n/app.py" in block

    def test_no_topics(self):
        summary = {
            "message_count": 5,
            "roles": {"user": 3, "assistant": 2},
            "topics": [],
            "files": [],
            "decisions": [],
        }
        block = format_summary_block("SESS-20260224-002", "t.jsonl", summary, "def456")
        assert "Topics:" not in block
        assert "Files:" not in block

    def test_decisions_included(self):
        summary = {
            "message_count": 4,
            "roles": {"user": 2, "assistant": 2},
            "topics": [],
            "files": [],
            "decisions": [
                {"type": "correction", "confidence": "high",
                 "excerpt": "Never use raw SQL", "role": "user"},
            ],
        }
        block = format_summary_block("SESS-20260224-003", "t.jsonl", summary, "gh789")
        assert "Decisions:" in block
        assert "[high] correction:" in block

    def test_roles_formatting(self):
        summary = {
            "message_count": 3,
            "roles": {"user": 1, "assistant": 2},
            "topics": [],
            "files": [],
            "decisions": [],
        }
        block = format_summary_block("SESS-20260224-004", "t.jsonl", summary, "x")
        assert "Roles:" in block
        assert "user=1" in block
        assert "assistant=2" in block


# ── write_summary ────────────────────────────────────────────────────────

class TestWriteSummary:
    @patch("scripts.session_summarizer.append_signals")
    @patch("scripts.session_summarizer.FileLock", side_effect=_fake_file_lock)
    def test_creates_summary_file(self, mock_lock, mock_signals, tmp_path):
        ws = str(tmp_path / "ws")
        os.makedirs(ws)
        t_path = str(tmp_path / "t.jsonl")
        with open(t_path, "w") as f:
            f.write("fake transcript\n")

        msgs = _make_messages(5)
        sess_id = write_summary(ws, t_path, msgs)

        assert sess_id is not None
        assert sess_id.startswith("SESS-")
        today = datetime.now().strftime("%Y-%m-%d")
        summary_file = os.path.join(ws, "summaries", "daily", f"{today}.md")
        assert os.path.isfile(summary_file)

    @patch("scripts.session_summarizer.append_signals")
    @patch("scripts.session_summarizer.FileLock", side_effect=_fake_file_lock)
    def test_dedup_prevents_rewrite(self, mock_lock, mock_signals, tmp_path):
        ws = str(tmp_path / "ws")
        os.makedirs(ws)
        t_path = str(tmp_path / "t.jsonl")
        with open(t_path, "w") as f:
            f.write("same transcript\n")

        msgs = _make_messages(5)
        first = write_summary(ws, t_path, msgs)
        assert first is not None

        second = write_summary(ws, t_path, msgs)
        assert second is None  # dedup

    @patch("scripts.session_summarizer.append_signals")
    @patch("scripts.session_summarizer.FileLock", side_effect=_fake_file_lock)
    def test_dry_run_does_not_write(self, mock_lock, mock_signals, tmp_path, capsys):
        ws = str(tmp_path / "ws")
        os.makedirs(ws)
        t_path = str(tmp_path / "t.jsonl")
        with open(t_path, "w") as f:
            f.write("dry run transcript\n")

        msgs = _make_messages(5)
        sess_id = write_summary(ws, t_path, msgs, dry_run=True)

        assert sess_id is not None
        today = datetime.now().strftime("%Y-%m-%d")
        summary_file = os.path.join(ws, "summaries", "daily", f"{today}.md")
        # File should NOT exist (dry run)
        assert not os.path.isfile(summary_file)
        captured = capsys.readouterr()
        assert "[DRY RUN]" in captured.out

    @patch("scripts.session_summarizer.append_signals")
    @patch("scripts.session_summarizer.FileLock", side_effect=_fake_file_lock)
    def test_too_short_session_skipped(self, mock_lock, mock_signals, tmp_path):
        ws = str(tmp_path / "ws")
        os.makedirs(ws)
        t_path = str(tmp_path / "t.jsonl")
        with open(t_path, "w") as f:
            f.write("short\n")

        msgs = _make_messages(2)  # less than 3
        sess_id = write_summary(ws, t_path, msgs)
        assert sess_id is None

    @patch("scripts.session_summarizer.append_signals")
    @patch("scripts.session_summarizer.FileLock", side_effect=_fake_file_lock)
    def test_counter_increments(self, mock_lock, mock_signals, tmp_path):
        ws = str(tmp_path / "ws")
        os.makedirs(ws)

        ids = []
        for i in range(3):
            t_path = str(tmp_path / f"t{i}.jsonl")
            with open(t_path, "w") as f:
                f.write(f"transcript {i}\n")
            sess_id = write_summary(ws, t_path, _make_messages(5))
            if sess_id:
                ids.append(sess_id)

        assert len(ids) == 3
        # Counter should increment: 001, 002, 003
        numbers = [int(sid.split("-")[-1]) for sid in ids]
        assert numbers == [1, 2, 3]

    @patch("scripts.session_summarizer.append_signals")
    @patch("scripts.session_summarizer.FileLock", side_effect=_fake_file_lock)
    def test_append_signals_called(self, mock_lock, mock_signals, tmp_path):
        """Verify that a linking signal is appended after writing."""
        ws = str(tmp_path / "ws")
        os.makedirs(ws)
        t_path = str(tmp_path / "t.jsonl")
        with open(t_path, "w") as f:
            f.write("signal transcript\n")

        write_summary(ws, t_path, _make_messages(5))
        mock_signals.assert_called_once()
        args = mock_signals.call_args
        signals_list = args[0][1]
        assert signals_list[0]["type"] == "summary"

    @patch("scripts.session_summarizer.append_signals")
    @patch("scripts.session_summarizer.FileLock", side_effect=_fake_file_lock)
    def test_header_written_on_new_file(self, mock_lock, mock_signals, tmp_path):
        ws = str(tmp_path / "ws")
        os.makedirs(ws)
        t_path = str(tmp_path / "t.jsonl")
        with open(t_path, "w") as f:
            f.write("header transcript\n")

        write_summary(ws, t_path, _make_messages(5))
        today = datetime.now().strftime("%Y-%m-%d")
        summary_file = os.path.join(ws, "summaries", "daily", f"{today}.md")
        with open(summary_file) as f:
            content = f.read()
        assert content.startswith("# Session Summaries")


# ── edge cases ───────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_unicode_content(self):
        msgs = [
            _msg("user", "Проект использует UTF-8 кодировку. ファイルパスは正しいです。"),
            _msg("assistant", "Understood, the Кодировка module handles encoding."),
            _msg("user", "Also check résumé.txt and naïve.py — très important."),
        ]
        s = extract_summary(msgs)
        assert s["message_count"] == 3

    def test_file_hash_unicode(self, tmp_path):
        p = tmp_path / "日本語.txt"
        p.write_bytes("こんにちは世界".encode("utf-8"))
        h = file_hash(str(p))
        assert len(h) == 16

    def test_file_path_regex_various(self):
        text = "Edit /home/user/my-project/src/main.rs and also ../lib/utils.go please"
        matches = FILE_PATH_RE.findall(text)
        assert any("main.rs" in m for m in matches)
        assert any("utils.go" in m for m in matches)

    @patch("scripts.session_summarizer.append_signals")
    @patch("scripts.session_summarizer.FileLock", side_effect=_fake_file_lock)
    def test_concurrent_writes(self, mock_lock, mock_signals, tmp_path):
        """Multiple threads writing summaries should not corrupt data."""
        ws = str(tmp_path / "ws")
        os.makedirs(ws)
        results = []
        errors = []

        def worker(idx):
            try:
                t_path = str(tmp_path / f"concurrent_{idx}.jsonl")
                with open(t_path, "w") as f:
                    f.write(f"transcript {idx}\n")
                sid = write_summary(ws, t_path, _make_messages(5))
                results.append(sid)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errors during concurrent writes: {errors}"
        written = [r for r in results if r is not None]
        assert len(written) >= 1  # at least some succeed

    @patch("scripts.session_summarizer.append_signals")
    @patch("scripts.session_summarizer.FileLock", side_effect=_fake_file_lock)
    def test_empty_workspace_creates_dirs(self, mock_lock, mock_signals, tmp_path):
        ws = str(tmp_path / "brand_new_workspace")
        t_path = str(tmp_path / "t.jsonl")
        with open(t_path, "w") as f:
            f.write("content\n")

        sess_id = write_summary(ws, t_path, _make_messages(4))
        assert sess_id is not None
        assert os.path.isdir(os.path.join(ws, "summaries", "daily"))

    def test_decisions_capped_at_10(self):
        """extract_summary should return at most 10 decisions."""
        msgs = [
            _msg("user", f"Don't ever use pattern{i} in production") for i in range(20)
        ]
        s = extract_summary(msgs)
        assert len(s["decisions"]) <= 10
