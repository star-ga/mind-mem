#!/usr/bin/env python3
"""Tests for compaction.py — GC and archival engine."""

import os
import sys
import tempfile
import unittest
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from compaction import (
    _extract_block_text,
    archive_completed_blocks,
    cleanup_daily_logs,
    cleanup_snapshots,
    compact_signals,
)
from init_workspace import init


class TestExtractBlockText(unittest.TestCase):
    def test_extracts_simple_block(self):
        content = "[D-20260101-001]\nStatement: Test decision\nStatus: active\n\n---\n"
        text = _extract_block_text(content, "D-20260101-001")
        self.assertIsNotNone(text)
        self.assertIn("Statement: Test decision", text)

    def test_returns_none_for_missing_block(self):
        content = "[D-20260101-001]\nStatement: Test\n"
        result = _extract_block_text(content, "D-20260101-999")
        self.assertIsNone(result)

    def test_extracts_block_up_to_separator(self):
        content = (
            "[D-20260101-001]\nStatement: First\nStatus: active\n\n---\n"
            "[D-20260101-002]\nStatement: Second\nStatus: done\n"
        )
        text = _extract_block_text(content, "D-20260101-001")
        self.assertIn("First", text)
        self.assertNotIn("Second", text)


class TestArchiveCompletedBlocks(unittest.TestCase):
    def test_dry_run_doesnt_modify(self):
        with tempfile.TemporaryDirectory() as td:
            init(td)
            # Add a completed task with an old date
            tasks_path = os.path.join(td, "tasks", "TASKS.md")
            with open(tasks_path, "a") as f:
                f.write("\n[T-20240101-001]\nTitle: Old task\nDate: 2024-01-01\nStatus: done\n\n---\n")

            actions = archive_completed_blocks(td, days=30, dry_run=True)
            self.assertTrue(len(actions) > 0)
            self.assertTrue(all("[dry-run]" in a for a in actions))
            # Original file should be unchanged
            with open(tasks_path) as f:
                content = f.read()
            self.assertIn("T-20240101-001", content)

    def test_no_archive_for_recent_blocks(self):
        with tempfile.TemporaryDirectory() as td:
            init(td)
            today = datetime.now().strftime("%Y-%m-%d")
            tasks_path = os.path.join(td, "tasks", "TASKS.md")
            with open(tasks_path, "a") as f:
                f.write(f"\n[T-20260215-001]\nTitle: Recent task\nDate: {today}\nStatus: done\n\n---\n")

            actions = archive_completed_blocks(td, days=30, dry_run=False)
            self.assertEqual(len(actions), 0)

    def test_no_archive_for_active_blocks(self):
        with tempfile.TemporaryDirectory() as td:
            init(td)
            tasks_path = os.path.join(td, "tasks", "TASKS.md")
            with open(tasks_path, "a") as f:
                f.write("\n[T-20240101-001]\nTitle: Active old task\nDate: 2024-01-01\nStatus: active\n\n---\n")

            actions = archive_completed_blocks(td, days=30, dry_run=False)
            self.assertEqual(len(actions), 0)


class TestCleanupSnapshots(unittest.TestCase):
    def test_removes_old_snapshots(self):
        with tempfile.TemporaryDirectory() as td:
            init(td)
            old_dir = os.path.join(td, "intelligence", "applied", "20240101-120000")
            os.makedirs(old_dir)
            with open(os.path.join(old_dir, "APPLY_RECEIPT.md"), "w") as f:
                f.write("test receipt\n")

            actions = cleanup_snapshots(td, days=30)
            self.assertTrue(len(actions) > 0)
            self.assertFalse(os.path.exists(old_dir))

    def test_keeps_recent_snapshots(self):
        with tempfile.TemporaryDirectory() as td:
            init(td)
            recent = datetime.now().strftime("%Y%m%d-%H%M%S")
            snap_dir = os.path.join(td, "intelligence", "applied", recent)
            os.makedirs(snap_dir)
            with open(os.path.join(snap_dir, "APPLY_RECEIPT.md"), "w") as f:
                f.write("test\n")

            actions = cleanup_snapshots(td, days=30)
            self.assertEqual(len(actions), 0)
            self.assertTrue(os.path.exists(snap_dir))

    def test_dry_run_snapshots(self):
        with tempfile.TemporaryDirectory() as td:
            init(td)
            old_dir = os.path.join(td, "intelligence", "applied", "20240101-120000")
            os.makedirs(old_dir)

            actions = cleanup_snapshots(td, days=30, dry_run=True)
            self.assertTrue(len(actions) > 0)
            self.assertTrue(os.path.exists(old_dir))  # Not removed in dry-run


class TestCleanupDailyLogs(unittest.TestCase):
    def test_archives_old_logs(self):
        with tempfile.TemporaryDirectory() as td:
            init(td)
            old_log = os.path.join(td, "memory", "2024-01-15.md")
            with open(old_log, "w") as f:
                f.write("# 2024-01-15\n\nOld daily log content.\n")

            actions = cleanup_daily_logs(td, days=30)
            self.assertTrue(len(actions) > 0)
            self.assertFalse(os.path.exists(old_log))
            # Archive should exist
            archive = os.path.join(td, "memory", "archive-2024.md")
            self.assertTrue(os.path.exists(archive))
            with open(archive) as f:
                content = f.read()
            self.assertIn("Old daily log content", content)

    def test_keeps_recent_logs(self):
        with tempfile.TemporaryDirectory() as td:
            init(td)
            today = datetime.now().strftime("%Y-%m-%d")
            log = os.path.join(td, "memory", f"{today}.md")
            with open(log, "w") as f:
                f.write("today's log\n")

            actions = cleanup_daily_logs(td, days=30)
            self.assertEqual(len(actions), 0)
            self.assertTrue(os.path.exists(log))


class TestCompactSignals(unittest.TestCase):
    def test_removes_resolved_old_signals(self):
        with tempfile.TemporaryDirectory() as td:
            init(td)
            signals_path = os.path.join(td, "intelligence", "SIGNALS.md")
            with open(signals_path, "w") as f:
                f.write("# Captured Signals\n\n")
                f.write("[SIG-20240101-001]\nDate: 2024-01-01\nStatus: resolved\nExcerpt: old signal\n\n---\n")
                f.write("[SIG-20260215-001]\nDate: 2026-02-15\nStatus: pending\nExcerpt: new signal\n\n---\n")

            actions = compact_signals(td, days=30)
            self.assertTrue(len(actions) > 0)
            with open(signals_path) as f:
                content = f.read()
            self.assertNotIn("SIG-20240101-001", content)
            self.assertIn("SIG-20260215-001", content)

    def test_keeps_pending_signals(self):
        with tempfile.TemporaryDirectory() as td:
            init(td)
            signals_path = os.path.join(td, "intelligence", "SIGNALS.md")
            with open(signals_path, "w") as f:
                f.write("# Captured Signals\n\n")
                f.write("[SIG-20240101-001]\nDate: 2024-01-01\nStatus: pending\nExcerpt: old but pending\n\n---\n")

            actions = compact_signals(td, days=30)
            self.assertEqual(len(actions), 0)


if __name__ == "__main__":
    unittest.main()
