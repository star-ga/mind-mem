#!/usr/bin/env python3
"""Tests for filelock.py — cross-platform advisory locking."""

import os
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from filelock import FileLock, LockTimeout


class TestFileLockBasic(unittest.TestCase):
    def test_context_manager_creates_and_removes_lock(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            path = f.name
        try:
            lock_path = path + ".lock"
            with FileLock(path):
                self.assertTrue(os.path.exists(lock_path))
            self.assertFalse(os.path.exists(lock_path))
        finally:
            os.unlink(path)

    @unittest.skipIf(sys.platform == "win32", "Windows holds exclusive lock on .lock fd")
    def test_lock_file_contains_pid(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            path = f.name
        try:
            lock = FileLock(path)
            lock.acquire()
            with open(path + ".lock") as lf:
                pid_str = lf.read().strip()
            self.assertEqual(int(pid_str), os.getpid())
            lock.release()
        finally:
            os.unlink(path)

    def test_double_release_is_safe(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            path = f.name
        try:
            lock = FileLock(path)
            lock.acquire()
            lock.release()
            lock.release()  # Should not raise
        finally:
            os.unlink(path)

    def test_acquire_release_manual(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            path = f.name
        try:
            lock = FileLock(path)
            lock.acquire()
            self.assertTrue(os.path.exists(path + ".lock"))
            lock.release()
            self.assertFalse(os.path.exists(path + ".lock"))
        finally:
            os.unlink(path)

    def test_timeout_zero_raises_immediately(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            path = f.name
        try:
            lock1 = FileLock(path)
            lock1.acquire()
            try:
                lock2 = FileLock(path, timeout=0)
                with self.assertRaises(LockTimeout):
                    lock2.acquire()
            finally:
                lock1.release()
        finally:
            os.unlink(path)

    def test_stale_lock_broken_automatically(self):
        """A lock file from a dead PID should be automatically broken."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            path = f.name
        try:
            lock_path = path + ".lock"
            # Create a lock file with a definitely-dead PID
            with open(lock_path, "w") as f:
                f.write("999999999\n")
            lock = FileLock(path, timeout=2.0)
            lock.acquire()  # Should break stale lock and succeed
            self.assertTrue(os.path.exists(lock_path))
            lock.release()
        finally:
            os.unlink(path)

    def test_repr(self):
        path = os.path.join(tempfile.gettempdir(), "test.md")
        lock = FileLock(path)
        self.assertIn("test.md", repr(lock))


class TestFileLockTimeout(unittest.TestCase):
    def test_timeout_with_short_window(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            path = f.name
        try:
            lock1 = FileLock(path)
            lock1.acquire()
            try:
                lock2 = FileLock(path, timeout=0.2)
                start = time.monotonic()
                with self.assertRaises(LockTimeout):
                    lock2.acquire()
                elapsed = time.monotonic() - start
                self.assertGreaterEqual(elapsed, 0.15)  # Should have waited
            finally:
                lock1.release()
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
