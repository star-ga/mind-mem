#!/usr/bin/env python3
"""Tests for filelock.py — cross-platform advisory locking."""

import errno
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
            with open(path + ".lock", encoding="utf-8") as lf:
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


class TestAcquireNeverRaisesABareOSError(unittest.TestCase):
    """``acquire`` promises the lock or ``LockTimeout``. Nothing else.

    ``_os_lock``'s docstring says every error other than the
    "unsupported filesystem" errnos "propagates to the caller", and it did:
    a caller writing ``with FileLock(p, timeout=30):`` had no handler for a
    raw ``OSError`` and simply died. Measured against the shipped code by
    injecting the EWOULDBLOCK that arm exists to report: ``acquire``
    propagated ``BlockingIOError`` straight out through the ``with``.

    On POSIX the arm is unreachable -- we hold ``O_EXCL`` on an inode we just
    created, so nobody else can hold its ``flock``. It is reachable where a
    filesystem can hand a breaker the SAME ``(st_dev, st_ino)`` for a NEW
    file as for the corpse it judged, which is ordinary NTFS behaviour: a
    freed MFT record is reused at once, so a ``_break_stale`` racing the
    create passes its identity check against the fresh file and adopts it
    between the ``os.open`` and the lock. ``_break_stale`` holds the OS lock
    across that adoption deliberately, so the arbitration is already right
    and exactly one process is inside the section -- only the loser's
    behaviour was wrong.

    Injected rather than raced, for the reason the class above exists: the
    question is what ``acquire`` does with that answer, and the answer is not
    producible on the platform running this test.
    """

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.target = os.path.join(self.dir, "contested.dat")
        with open(self.target, "w", encoding="utf-8") as fh:
            fh.write("")
        self._real_os_lock = FileLock._os_lock

    def tearDown(self):
        FileLock._os_lock = self._real_os_lock
        shutil.rmtree(self.dir, ignore_errors=True)

    def _refuse_the_os_lock(self):
        def _held_by_someone_else(self, fd):
            raise BlockingIOError(11, "Resource temporarily unavailable")

        FileLock._os_lock = _held_by_someone_else

    def test_a_refused_os_lock_times_out_instead_of_raising(self):
        self._refuse_the_os_lock()
        lock = FileLock(self.target, timeout=0.3, poll_interval=0.01)
        started = time.monotonic()
        with self.assertRaises(LockTimeout):
            lock.acquire()
        # It waited rather than failing fast: the deadline is what ends it.
        self.assertGreaterEqual(time.monotonic() - started, 0.3)

    def test_the_zero_timeout_form_also_answers_LockTimeout(self):
        self._refuse_the_os_lock()
        with self.assertRaises(LockTimeout):
            FileLock(self.target, timeout=0).acquire()

    def test_no_lockfile_is_left_behind(self):
        """A claim we could not keep must not wedge the path for everyone else."""
        self._refuse_the_os_lock()
        with self.assertRaises(LockTimeout):
            FileLock(self.target, timeout=0.1, poll_interval=0.01).acquire()
        self.assertFalse(
            os.path.exists(self.target + ".lock"),
            "the failed acquire left a lockfile naming a live pid",
        )
        # And the path is usable again, by the real lock.
        FileLock._os_lock = self._real_os_lock
        with FileLock(self.target, timeout=2.0):
            pass

    def test_the_thread_lock_is_handed_back(self):
        """Twice in a row: the second attempt must not block on the first."""
        self._refuse_the_os_lock()
        for _ in range(2):
            with self.assertRaises(LockTimeout):
                FileLock(self.target, timeout=0.1, poll_interval=0.01).acquire()

    def test_a_write_failure_still_propagates_by_name(self):
        """Control: only the LOCK attempt polls.

        Without this the change would read as "swallow every OSError in the
        claim", and an ``ENOSPC`` from the pid write would come back as a
        ``LockTimeout`` -- a diagnosis that names the wrong thing.
        """
        real_write = os.write

        def _no_space(fd, data):
            raise OSError(28, "No space left on device")

        os.write = _no_space
        try:
            with self.assertRaises(OSError) as caught:
                FileLock(self.target, timeout=0.3).acquire()
        finally:
            os.write = real_write
        self.assertNotIsInstance(caught.exception, LockTimeout)
        self.assertEqual(caught.exception.errno, 28)

    def test_a_hard_lock_error_still_propagates_by_name(self):
        """Control: only CONTENTION polls. EIO is not contention.

        The boundary this fix must not cross. ``test_medlow_batch12_regressions
        .test_failed_os_lock_leaves_no_lockfile_behind`` was written for a
        synthetic ``EIO`` and asserts the lockfile does not survive it;
        catching every ``OSError`` here turned that hard failure into a
        one-second wait and a ``LockTimeout``, which tells an operator whose
        disk is failing that a lock was busy.
        """

        def _hard_io_error(self, fd):
            raise OSError(errno.EIO, "synthetic I/O error")

        FileLock._os_lock = _hard_io_error
        with self.assertRaises(OSError) as caught:
            FileLock(self.target, timeout=5.0).acquire()
        self.assertNotIsInstance(caught.exception, LockTimeout)
        self.assertEqual(caught.exception.errno, errno.EIO)
        self.assertFalse(
            os.path.exists(self.target + ".lock"),
            "a hard failure left a live-pid lockfile behind",
        )

    def test_the_two_errno_sets_do_not_overlap(self):
        """ "cannot lock" and "somebody else has it" are different answers.

        An errno in both sets would be swallowed by ``_os_lock`` as
        "unsupported filesystem" AND retried here as contention — two
        different beliefs about the same code, and the first one wins
        silently.
        """
        self.assertEqual(
            mfl._CONTENDED_LOCK_ERRNOS & mfl._UNSUPPORTED_LOCK_ERRNOS,
            frozenset(),
            "an errno cannot mean both 'no locking here' and 'lock is taken'",
        )
        self.assertIn(errno.EWOULDBLOCK, mfl._CONTENDED_LOCK_ERRNOS)
        self.assertNotIn(errno.EIO, mfl._CONTENDED_LOCK_ERRNOS)

    def test_an_uncontested_acquire_is_unchanged(self):
        """Control: the real lock still works, and this arm is not on its path."""
        with FileLock(self.target, timeout=2.0) as lock:
            self.assertIsNotNone(lock._lock_fd)
        self.assertFalse(os.path.exists(self.target + ".lock"))


if __name__ == "__main__":
    unittest.main()
