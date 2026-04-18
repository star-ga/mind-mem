#!/usr/bin/env python3
"""Tests for watcher.py — file change detection for auto-reindex."""

import os
import shutil
import tempfile
import time
import unittest

from mind_mem.watcher import FileWatcher


class TestFileWatcher(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.mkdtemp()
        self.changes: list[set[str]] = []

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def _callback(self, changed: set[str]) -> None:
        self.changes.append(changed)

    def test_detects_new_file(self):
        watcher = FileWatcher(self.td, callback=self._callback, interval=0.1)
        watcher.start()
        try:
            time.sleep(0.2)  # let initial scan happen
            with open(os.path.join(self.td, "test.md"), "w") as f:
                f.write("# New file\n")
            time.sleep(0.3)
            self.assertGreater(len(self.changes), 0)
            found = any("test.md" in str(c) for c in self.changes)
            self.assertTrue(found, f"test.md not found in changes: {self.changes}")
        finally:
            watcher.stop()

    def test_detects_modified_file(self):
        path = os.path.join(self.td, "existing.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write("# Original\n")

        watcher = FileWatcher(self.td, callback=self._callback, interval=0.1)
        watcher.start()
        try:
            time.sleep(0.2)
            with open(path, "a") as f:
                f.write("## Modified\n")
            time.sleep(0.3)
            self.assertGreater(len(self.changes), 0)
        finally:
            watcher.stop()

    def test_detects_deleted_file(self):
        path = os.path.join(self.td, "delete-me.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write("# Will be deleted\n")

        watcher = FileWatcher(self.td, callback=self._callback, interval=0.1)
        watcher.start()
        try:
            time.sleep(0.2)
            os.unlink(path)
            time.sleep(0.3)
            self.assertGreater(len(self.changes), 0)
            found = any("delete-me.md" in str(c) for c in self.changes)
            self.assertTrue(found)
        finally:
            watcher.stop()

    def test_ignores_non_md_files(self):
        watcher = FileWatcher(self.td, callback=self._callback, interval=0.1)
        watcher.start()
        try:
            time.sleep(0.2)
            with open(os.path.join(self.td, "test.txt"), "w") as f:
                f.write("not markdown\n")
            time.sleep(0.3)
            # Should NOT trigger callback for .txt files
            txt_found = any("test.txt" in str(c) for c in self.changes)
            self.assertFalse(txt_found)
        finally:
            watcher.stop()

    def test_ignores_hidden_dirs(self):
        hidden = os.path.join(self.td, ".hidden")
        os.makedirs(hidden)

        watcher = FileWatcher(self.td, callback=self._callback, interval=0.1)
        watcher.start()
        try:
            time.sleep(0.2)
            with open(os.path.join(hidden, "secret.md"), "w") as f:
                f.write("# Hidden\n")
            time.sleep(0.3)
            hidden_found = any("secret.md" in str(c) for c in self.changes)
            self.assertFalse(hidden_found)
        finally:
            watcher.stop()

    def test_stop_actually_stops(self):
        watcher = FileWatcher(self.td, callback=self._callback, interval=0.1)
        watcher.start()
        self.assertTrue(watcher.is_running)
        watcher.stop()
        self.assertFalse(watcher.is_running)

    def test_no_callback_on_unchanged(self):
        path = os.path.join(self.td, "stable.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write("# Stable\n")

        watcher = FileWatcher(self.td, callback=self._callback, interval=0.1)
        watcher.start()
        try:
            time.sleep(0.5)
            # No changes made — should not trigger
            self.assertEqual(len(self.changes), 0)
        finally:
            watcher.stop()

    def test_subdirectory_changes(self):
        subdir = os.path.join(self.td, "decisions")
        os.makedirs(subdir)

        watcher = FileWatcher(self.td, callback=self._callback, interval=0.1)
        watcher.start()
        try:
            time.sleep(0.2)
            with open(os.path.join(subdir, "DECISIONS.md"), "w") as f:
                f.write("# Decisions\n")
            time.sleep(0.3)
            self.assertGreater(len(self.changes), 0)
        finally:
            watcher.stop()

    def test_double_start_is_safe(self):
        watcher = FileWatcher(self.td, callback=self._callback, interval=0.1)
        watcher.start()
        watcher.start()  # should be no-op
        self.assertTrue(watcher.is_running)
        watcher.stop()


if __name__ == "__main__":
    unittest.main()
