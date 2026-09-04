#!/usr/bin/env python3
"""Tests for filelock.py — cross-platform advisory locking."""

import errno
import os
import shutil
import sys
import tempfile
import time
import unittest

from _platform_compat import chmod_denies_write

import mind_mem.mind_filelock as mfl
from mind_mem.mind_filelock import _BREAK_ADOPTED, _BREAK_NOTHING, FileLock, LockTimeout

#: A pid that cannot be running, so a lockfile naming it is unambiguously a
#: corpse. 2**22 is above every default ``pid_max`` this ships to.
_IMPOSSIBLE_PID = str(2**22)


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


class TestTheWindowsSharingViolationOnCreate(unittest.TestCase):
    """``O_CREAT|O_EXCL`` answering EACCES is contention, not a fault.

    POSIX answers ``FileExistsError`` when the lockfile is already there, and
    for as long as this module existed that was the only answer the create
    arm knew. Windows has another one. Measured on windows-latest 3.12, CI
    run 33904519141 / job 101126178298, with six processes contending for one
    lock: four of them died with

        PermissionError: [Errno 13] Permission denied:
          'C:\\...\\test_no_two_processes_are_ever0\\shared_fixed.dat.lock'

    raised straight out of ``FileLock.acquire`` into a ``with`` statement
    that had no handler for it. The same error killed a worker in
    ``test_served_ledger_concurrency``. The lock itself was correct
    throughout -- zero mutual-exclusion violations were ever reported. Only
    the loser of the race was wrong: it raised where the ``FileExistsError``
    arm one syscall earlier would have waited.

    Note ``[Errno 13]`` and not ``[WinError 32]``. ``OSError.__str__``
    renders the WinError form whenever ``winerror`` is set, so on this path
    it was not: ``os.open`` reaches the CRT's ``_wopen``, which sets only
    ``errno``. A fix that matched on ``winerror`` would have caught nothing,
    which is why these tests inject an errno-only error -- the shape actually
    observed -- rather than the shape that would have been convenient.

    Injected, because a real sharing violation cannot be produced on this
    platform. Labelled honestly: what is proven here is that ``acquire``
    handles that errno correctly. That it is the errno Windows raises is
    established by the CI transcript above, not by this file.
    """

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.target = os.path.join(self.dir, "contested.dat")
        with open(self.target, "w", encoding="utf-8") as fh:
            fh.write("")
        self.lockfile = self.target + ".lock"
        self._real_open = os.open

    def tearDown(self):
        os.open = self._real_open
        shutil.rmtree(self.dir, ignore_errors=True)

    def _refuse_create_with(self, err: int):
        """Make O_CREAT|O_EXCL on the lockfile answer *err*, as Windows does."""
        real = self._real_open
        lockfile = self.lockfile

        def _sharing_violation(path, flags, *a, **k):
            if os.fspath(path) == lockfile and (flags & os.O_CREAT):
                raise PermissionError(err, "Permission denied", lockfile)
            return real(path, flags, *a, **k)

        os.open = _sharing_violation

    # -- the evidenced case -------------------------------------------------

    def test_a_refused_create_beside_a_live_lockfile_waits(self):
        """The exact CI shape: a held lock, and the create answers EACCES."""
        with open(self.lockfile, "w", encoding="utf-8") as fh:
            fh.write(f"{os.getpid()}\n")  # a live pid: not stale, not breakable
        self._refuse_create_with(errno.EACCES)

        started = time.monotonic()
        with self.assertRaises(LockTimeout):
            FileLock(self.target, timeout=0.3, poll_interval=0.01).acquire()
        self.assertGreaterEqual(time.monotonic() - started, 0.3, "it failed fast instead of waiting")

    def test_the_lock_is_still_taken_once_the_holder_leaves(self):
        """It POLLS -- so the ordinary hand-off still completes.

        A fix that merely stopped raising would satisfy the test above by
        sleeping to the deadline forever. This one only passes if the retry
        actually reaches the create again.
        """
        with open(self.lockfile, "w", encoding="utf-8") as fh:
            fh.write(f"{os.getpid()}\n")
        real = self._real_open
        lockfile = self.lockfile
        refusals = [3]  # contended for three attempts, then the holder leaves

        def _briefly_contended(path, flags, *a, **k):
            if os.fspath(path) == lockfile and (flags & os.O_CREAT):
                if refusals[0] > 0:
                    refusals[0] -= 1
                    raise PermissionError(errno.EACCES, "Permission denied", lockfile)
                try:
                    real(lockfile, os.O_WRONLY)  # holder gone; clear its file
                    os.unlink(lockfile)
                except OSError:
                    pass
            return real(path, flags, *a, **k)

        os.open = _briefly_contended
        lock = FileLock(self.target, timeout=5.0, poll_interval=0.01)
        lock.acquire()
        try:
            self.assertEqual(refusals[0], 0, "the retry never got past the refusals")
            self.assertTrue(os.path.exists(lock.lock_path))
        finally:
            lock.release()

    def test_a_dead_owners_lock_is_still_broken_through_this_path(self):
        """It joins the FileExistsError ARM, not a bare sleep loop.

        The contended-create path has to keep the stale-breaking the
        ``FileExistsError`` arm does, or a Windows workspace holding one
        crashed owner's lockfile is wedged until every waiter times out --
        which is the failure this module's ``_break_stale`` exists to end.
        """
        with open(self.lockfile, "w", encoding="utf-8") as fh:
            fh.write(_IMPOSSIBLE_PID + "\n")  # nobody is home
        self._refuse_create_with(errno.EACCES)

        lock = FileLock(self.target, timeout=5.0, poll_interval=0.01)
        lock.acquire()  # must ADOPT the corpse rather than wait it out
        try:
            self.assertIsNotNone(lock._lock_fd)
            with open(self.lockfile, encoding="utf-8") as fh:
                self.assertEqual(fh.read().strip(), str(os.getpid()), "the corpse was not adopted")
        finally:
            lock.release()

    # -- the boundary -------------------------------------------------------

    def test_a_refused_create_with_NO_lockfile_still_propagates(self):
        """EACCES with nothing there is a permissions fault, not contention.

        The disambiguation. Without it, a workspace directory nobody may
        write to would report ``LockTimeout`` -- the same lie as reporting a
        failing disk that way.
        """
        self.assertFalse(os.path.exists(self.lockfile))
        self._refuse_create_with(errno.EACCES)

        with self.assertRaises(PermissionError) as caught:
            FileLock(self.target, timeout=0.3, poll_interval=0.01).acquire()
        self.assertNotIsInstance(caught.exception, LockTimeout)
        self.assertEqual(caught.exception.errno, errno.EACCES)

    def test_a_lockfile_that_vanishes_mid_check_is_not_read_as_a_fault(self):
        """The race the retry closes, made deterministic.

        The create is refused while the lockfile is there, and the holder
        releases before the existence check runs. Judging on that one stat
        alone would read "refused, and nothing is there" as a permissions
        fault and crash a caller for WINNING the race. The second attempt
        resolves it the only honest way: it just succeeds.
        """
        with open(self.lockfile, "w", encoding="utf-8") as fh:
            fh.write(f"{os.getpid()}\n")
        real = self._real_open
        lockfile = self.lockfile
        fired = []

        def _refuse_then_vanish(path, flags, *a, **k):
            if os.fspath(path) == lockfile and (flags & os.O_CREAT) and not fired:
                fired.append(True)
                os.unlink(lockfile)  # the holder releases inside the window
                raise PermissionError(errno.EACCES, "Permission denied", lockfile)
            return real(path, flags, *a, **k)

        os.open = _refuse_then_vanish
        lock = FileLock(self.target, timeout=0.5, poll_interval=0.01)
        lock.acquire()  # must NOT raise PermissionError
        try:
            self.assertEqual(fired, [True], "the injection never fired")
            self.assertIsNotNone(lock._lock_fd)
        finally:
            lock.release()

    def test_a_read_only_directory_still_propagates_unmocked(self):
        """The same boundary with no mock at all, where the OS allows it."""
        if not chmod_denies_write(self.dir):
            self.skipTest("this filesystem does not enforce the write bit")
        sub = os.path.join(self.dir, "locked")
        os.makedirs(sub)
        target = os.path.join(sub, "f.md")
        with open(target, "w", encoding="utf-8") as fh:
            fh.write("x")
        os.chmod(sub, 0o500)
        try:
            with self.assertRaises(OSError) as caught:
                FileLock(target, timeout=0.3, poll_interval=0.01).acquire()
            self.assertNotIsInstance(caught.exception, LockTimeout)
        finally:
            os.chmod(sub, 0o700)

    def test_a_hard_create_error_propagates_however_the_lockfile_looks(self):
        """EIO is not contention even with the lockfile sitting right there."""
        with open(self.lockfile, "w", encoding="utf-8") as fh:
            fh.write(f"{os.getpid()}\n")
        self._refuse_create_with(errno.EIO)

        with self.assertRaises(OSError) as caught:
            FileLock(self.target, timeout=5.0, poll_interval=0.01).acquire()
        self.assertNotIsInstance(caught.exception, LockTimeout)
        self.assertEqual(caught.exception.errno, errno.EIO)

    # -- the winerror supplement -------------------------------------------

    def test_a_winerror_sharing_violation_counts_even_with_an_odd_errno(self):
        """Where CPython DOES supply winerror, it is believed on its own.

        ``os.open`` did not supply one (the CI message rendered ``[Errno 13]``,
        and ``OSError.__str__`` prints ``[WinError n]`` whenever one is set),
        but the Win32-direct calls in this module can. Both vocabularies are
        understood so a future call site is not silently unhandled.
        """
        sharing = OSError(errno.EINVAL, "sharing violation")
        sharing.winerror = 32  # ERROR_SHARING_VIOLATION
        self.assertTrue(mfl._is_lock_contention(sharing))

        lock_violation = OSError(errno.EINVAL, "lock violation")
        lock_violation.winerror = 33  # ERROR_LOCK_VIOLATION
        self.assertTrue(mfl._is_lock_contention(lock_violation))

        # A winerror that is NOT a sharing violation is not waved through.
        disk_full = OSError(errno.ENOSPC, "no space")
        disk_full.winerror = 112  # ERROR_DISK_FULL
        self.assertFalse(mfl._is_lock_contention(disk_full))

        # And the errno vocabulary still works with no winerror at all.
        self.assertTrue(mfl._is_lock_contention(PermissionError(errno.EACCES, "denied")))
        self.assertFalse(mfl._is_lock_contention(OSError(errno.EIO, "io error")))


class TestTheAdoptionRewriteCanBeRefused(unittest.TestCase):
    """``_break_stale``'s ftruncate/lseek/write were unguarded too.

    The adoption is what MAKES the break: the winner truncates the abandoned
    file and writes its own pid into it. Those are ``_chsize_s`` and
    ``_write`` in the CRT, which answer EACCES where Windows means
    ERROR_LOCK_VIOLATION -- and truncating a file whose byte range someone
    still has locked is exactly that. Unguarded, the error left ``_break_stale``
    and then ``acquire``, killing the caller for losing a race.

    An adoption that could not land means no adoption happened, so the honest
    report is ``_BREAK_NOTHING``: nothing was unlinked, the ``finally``
    releases the OS lock and closes the descriptor, and the corpse is left
    exactly as it was for the next waiter.
    """

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.target = os.path.join(self.dir, "contested.dat")
        with open(self.target, "w", encoding="utf-8") as fh:
            fh.write("")
        self.lockfile = self.target + ".lock"
        with open(self.lockfile, "w", encoding="utf-8") as fh:
            fh.write(_IMPOSSIBLE_PID + "\n")  # a corpse, so a break is attempted
        self._real_ftruncate = os.ftruncate

    def tearDown(self):
        os.ftruncate = self._real_ftruncate
        shutil.rmtree(self.dir, ignore_errors=True)

    def _refuse_truncate_with(self, err: int):
        def _refused(fd, length):
            raise OSError(err, "refused")

        os.ftruncate = _refused

    def test_the_control_adopts_when_nothing_refuses(self):
        """Positive control. Without it the refusals below prove nothing:
        a ``_break_stale`` that never adopted would satisfy them too."""
        lock = FileLock(self.target, timeout=1.0)
        self.assertEqual(lock._break_stale(lock._stale_identity()), _BREAK_ADOPTED)
        lock.release()

    def test_a_refused_rewrite_reports_the_break_as_not_taken(self):
        self._refuse_truncate_with(errno.EACCES)
        lock = FileLock(self.target, timeout=1.0)
        self.assertEqual(lock._break_stale(lock._stale_identity()), _BREAK_NOTHING)
        self.assertIsNone(lock._lock_fd, "it kept a descriptor for a lock it does not hold")
        self.assertTrue(os.path.exists(self.lockfile), "the corpse was removed by a break that did not happen")

    def test_acquire_answers_LockTimeout_rather_than_dying(self):
        """End to end: the refusal reaches a caller as the documented answer."""
        self._refuse_truncate_with(errno.EACCES)
        with self.assertRaises(LockTimeout):
            FileLock(self.target, timeout=0.2, poll_interval=0.01).acquire()

    def test_a_hard_rewrite_failure_still_propagates(self):
        """ENOSPC during an adoption is a full disk, not a lost race."""
        self._refuse_truncate_with(errno.ENOSPC)
        lock = FileLock(self.target, timeout=1.0)
        with self.assertRaises(OSError) as caught:
            lock._break_stale(lock._stale_identity())
        self.assertEqual(caught.exception.errno, errno.ENOSPC)


if __name__ == "__main__":
    unittest.main()
