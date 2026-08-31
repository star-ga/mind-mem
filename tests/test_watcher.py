#!/usr/bin/env python3
"""Tests for watcher.py — file change detection for auto-reindex."""

import os
import shutil
import tempfile
import threading
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

    def test_clean_stop_reports_true(self):
        watcher = FileWatcher(self.td, callback=self._callback, interval=0.05)
        watcher.start()
        self.assertIs(watcher.stop(), True)
        self.assertFalse(watcher.is_running)

    def test_stop_on_unstarted_watcher_reports_true(self):
        watcher = FileWatcher(self.td, callback=self._callback, interval=0.05)
        self.assertIs(watcher.stop(), True)


class TestStopWaitsForCallback(unittest.TestCase):
    """stop() must not claim "stopped" while the callback is still writing.

    The callback is documented as an incremental reindex, which on a real
    workspace can outlast the join timeout. The old stop() ignored the
    join outcome, dropped the thread handle and logged watcher_stopped,
    leaving no way to observe the still-running writer.
    """

    def setUp(self):
        self.td = tempfile.mkdtemp()
        self.release = threading.Event()
        self.entered = threading.Event()

    def tearDown(self):
        self.release.set()
        shutil.rmtree(self.td, ignore_errors=True)

    def _blocking_callback(self, _changed: set[str]) -> None:
        self.entered.set()
        self.release.wait(10)

    def _watcher_in_callback(self) -> FileWatcher:
        watcher = FileWatcher(self.td, callback=self._blocking_callback, interval=0.05)
        watcher.start()
        with open(os.path.join(self.td, "trigger.md"), "w", encoding="utf-8") as f:
            f.write("# trigger\n")
        self.assertTrue(self.entered.wait(5), "callback never fired")
        return watcher

    def test_stop_reports_false_while_callback_runs(self):
        watcher = self._watcher_in_callback()
        self.assertIs(watcher.stop(timeout=0.1), False)
        self.assertTrue(watcher.is_running, "is_running hid a live callback thread")
        self.release.set()
        self.assertIs(watcher.stop(timeout=5), True)
        self.assertFalse(watcher.is_running)

    def test_start_refuses_while_previous_loop_finishes(self):
        watcher = self._watcher_in_callback()
        thread = watcher._thread
        self.assertIs(watcher.stop(timeout=0.1), False)
        watcher.start()
        self.assertIs(watcher._thread, thread, "a second watch loop was started over the same workspace")
        self.release.set()
        self.assertIs(watcher.stop(timeout=5), True)


if __name__ == "__main__":
    unittest.main()
