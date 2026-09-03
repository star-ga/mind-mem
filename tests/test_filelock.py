#!/usr/bin/env python3
"""Tests for filelock.py — cross-platform advisory locking."""

import os
import shutil
import sys
import tempfile
import time
import unittest

import mind_mem.mind_filelock as mfl
from mind_mem.mind_filelock import FileLock, LockTimeout


class TestReleaseUnderARefusedUnlink(unittest.TestCase):
    """The Windows release wedge, simulated on any platform.

    On Windows a waiter that merely READS the lockfile (inside
    ``_stale_identity``) makes the holder's post-close ``os.unlink`` fail
    with ``ERROR_SHARING_VIOLATION``; the module swallows that error. CI run
    33707752303 measured the consequence on every Windows row: a lockfile
    naming a live pid that nothing may break, and every waiter timing out.
    ``os.unlink`` is refused here for the lockfile exactly as Windows refuses
    it, so both halves can be shown on Linux: the wedge with the sentinel
    OFF (the positive control -- it is also the mutation proof, since OFF is
    5.0.1's release path byte for byte) and the adoption with it ON.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="mm_refused_unlink_")
        self.path = os.path.join(self.tmp, "target.md")
        self.lock_path = self.path + ".lock"
        self._flag = mfl._UNLINK_MAY_BE_REFUSED
        self._unlink = os.unlink
        lock_path = self.lock_path
        real_unlink = self._unlink

        def refusing_unlink(target, *args, **kwargs):
            if os.fspath(target) == lock_path:
                raise PermissionError(13, "sharing violation (simulated Windows: another process holds the file open)")
            return real_unlink(target, *args, **kwargs)

        os.unlink = refusing_unlink

    def tearDown(self):
        os.unlink = self._unlink
        mfl._UNLINK_MAY_BE_REFUSED = self._flag
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _lockfile_text(self) -> str:
        with open(self.lock_path, encoding="utf-8") as fh:
            return fh.read().strip()

    def test_positive_control_without_the_sentinel_the_refused_unlink_wedges_the_lock(self):
        """5.0.1's release path under Windows unlink semantics: the wedge CI measured."""
        mfl._UNLINK_MAY_BE_REFUSED = False
        with FileLock(self.path, timeout=1.0):
            pass
        self.assertTrue(os.path.exists(self.lock_path), "the simulated refusal did not happen; the control proves nothing")
        self.assertEqual(self._lockfile_text(), str(os.getpid()), "the file names a live pid -- nothing may break it")
        with self.assertRaises(LockTimeout):
            FileLock(self.path, timeout=0.3).acquire()

    def test_with_the_sentinel_the_next_acquirer_adopts_the_refused_file(self):
        mfl._UNLINK_MAY_BE_REFUSED = True
        with FileLock(self.path, timeout=1.0):
            pass
        self.assertTrue(os.path.exists(self.lock_path), "the simulated refusal did not happen")
        self.assertEqual(self._lockfile_text(), mfl._RELEASED_SENTINEL)

        second = FileLock(self.path, timeout=1.0)
        second.acquire()  # adopts: same inode, our pid, our OS lock -- no timeout
        try:
            self.assertEqual(self._lockfile_text(), str(os.getpid()))
            self.assertIsNotNone(second._lock_fd)
        finally:
            second.release()
        self.assertEqual(self._lockfile_text(), mfl._RELEASED_SENTINEL, "the adopter releases the same way it acquired")

    def test_the_sentinel_is_written_under_the_os_lock_not_after_it(self):
        """A reader can never see 'released' beside an adoption in progress."""
        mfl._UNLINK_MAY_BE_REFUSED = True
        order = []
        original_mark, original_unlock = FileLock._mark_released, FileLock._os_unlock
        try:
            FileLock._mark_released = staticmethod(lambda fd: (order.append("mark"), original_mark(fd)))
            FileLock._os_unlock = lambda self, fd: (order.append("unlock"), original_unlock(self, fd))
            with FileLock(self.path, timeout=1.0):
                pass
        finally:
            FileLock._mark_released, FileLock._os_unlock = staticmethod(original_mark), original_unlock
        self.assertEqual(order, ["mark", "unlock"])

    def test_the_posix_release_path_is_untouched_with_the_flag_off(self):
        """OFF must be byte-for-byte the old path: no sentinel write, the file simply unlinked."""
        mfl._UNLINK_MAY_BE_REFUSED = False
        os.unlink = self._unlink  # a platform where the unlink succeeds
        writes = []
        original = FileLock._mark_released
        try:
            FileLock._mark_released = staticmethod(lambda fd: writes.append(fd))
            with FileLock(self.path, timeout=1.0):
                pass
        finally:
            FileLock._mark_released = staticmethod(original)
        self.assertEqual(writes, [])
        self.assertFalse(os.path.exists(self.lock_path))


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
            with open(lock_path, "w", encoding="utf-8") as f:
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
