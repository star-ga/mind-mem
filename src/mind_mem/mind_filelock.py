#!/usr/bin/env python3
"""mind-mem file locking — cross-platform advisory locks. Zero external deps.

Provides cooperative file locking for concurrent agent/session writes.
Uses a two-layer approach:
  1. threading.Lock for same-process (thread) contention
  2. O_CREAT|O_EXCL lockfile + OS-level locks for cross-process contention

Usage:
    from filelock import FileLock

    with FileLock("path/to/file.md"):
        # exclusive access to the file
        ...

    # Or manual:
    lock = FileLock("path/to/file.md", timeout=5.0)
    lock.acquire()
    try:
        ...
    finally:
        lock.release()
"""

from __future__ import annotations

import errno
import os
import sys
import threading
import time
from types import TracebackType

#: errnos that mean "this filesystem does not implement advisory locking".
#: Not the same as "the lock is taken" (EWOULDBLOCK/EAGAIN), which must
#: never be swallowed. Some names are platform-specific, hence getattr.
_UNSUPPORTED_LOCK_ERRNOS: frozenset[int] = frozenset(
    code
    for code in (
        getattr(errno, "ENOLCK", None),
        getattr(errno, "EOPNOTSUPP", None),
        getattr(errno, "ENOTSUP", None),
        getattr(errno, "ENOSYS", None),
    )
    if code is not None
)


class LockTimeout(Exception):
    """Raised when lock acquisition times out."""

    pass


class FileLock:
    """Cross-platform advisory file lock.

    Creates a .lock file next to the target. Uses OS-level locking
    where available, falls back to atomic create for portability.
    Includes threading.Lock for intra-process mutual exclusion.

    Parameters:
        path: Path to the file to lock.
        timeout: Max seconds to wait for lock (0 = non-blocking, -1 = infinite).
        poll_interval: Seconds between retry attempts.
    """

    # Class-level thread locks keyed by lock_path for intra-process safety
    _thread_locks: dict = {}
    _thread_lock_guard = threading.Lock()

    def __init__(self, path: str, timeout: float = 10.0, poll_interval: float = 0.05) -> None:
        self.path = os.path.abspath(path)
        self.lock_path = self.path + ".lock"
        self.timeout = timeout
        self.poll_interval = poll_interval
        self._lock_fd: int | None = None
        #: (st_dev, st_ino) of the lockfile this instance created, set once
        #: the acquire succeeds. Only the process holding *this* inode may
        #: remove it: a lockfile is a claim, and unlinking one by path alone
        #: is how a releasing process deletes its successor's claim and puts
        #: two writers inside the same critical section.
        self._lock_identity: tuple[int, int] | None = None
        self._owns_thread_lock = False

    def acquire(self) -> None:
        """Acquire the lock. Raises LockTimeout if timeout exceeded."""
        # Layer 1: Acquire intra-process thread lock
        with self._thread_lock_guard:
            if self.lock_path not in self._thread_locks:
                self._thread_locks[self.lock_path] = threading.Lock()
            tlock = self._thread_locks[self.lock_path]

        start = time.monotonic()
        remaining = self.timeout

        if self.timeout == 0:
            if not tlock.acquire(blocking=False):
                raise LockTimeout(f"Could not acquire lock: {self.lock_path}")
        elif self.timeout < 0:
            tlock.acquire()
        else:
            if not tlock.acquire(timeout=remaining):
                raise LockTimeout(f"Lock timeout ({self.timeout}s) for: {self.lock_path}")
        self._owns_thread_lock = True

        # Layer 2: Acquire cross-process file lock. BaseException, not
        # Exception: a KeyboardInterrupt landing inside the acquire poll
        # loop must still hand back the thread lock, or this lock_path is
        # dead for every other thread in the process.
        try:
            self._acquire_file_lock(start)
        except BaseException:
            self._owns_thread_lock = False
            tlock.release()
            raise

    def _acquire_file_lock(self, start: float) -> None:
        """Acquire the filesystem-level lock."""
        while True:
            try:
                fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                abandoned = self._stale_identity()
                if abandoned is not None:
                    self._break_stale(abandoned)
                    continue

                if self.timeout == 0:
                    raise LockTimeout(f"Could not acquire lock: {self.lock_path}")
                elapsed = time.monotonic() - start
                if self.timeout > 0 and elapsed >= self.timeout:
                    raise LockTimeout(f"Lock timeout ({self.timeout}s) for: {self.lock_path}")
                time.sleep(self.poll_interval)
                continue

            # The lockfile now exists on disk and names THIS live pid, so it
            # can never be judged stale by another acquirer. Anything that
            # fails from here on must undo it, or the lock is wedged for the
            # lifetime of this process — every later acquire would find a
            # live-pid lockfile and block to LockTimeout.
            try:
                os.write(fd, f"{os.getpid()}\n".encode())
                self._os_lock(fd)
            except BaseException:
                self._discard_lock_file(fd)
                raise
            try:
                st = os.fstat(fd)
                self._lock_identity = (st.st_dev, st.st_ino)
            except OSError:
                self._lock_identity = None
            self._lock_fd = fd
            return

    def _discard_lock_file(self, fd: int) -> None:
        """Close ``fd`` and remove the lockfile, ignoring cleanup errors.

        Used on the failure path of :meth:`_acquire_file_lock`, where the
        original error is about to be re-raised and must not be masked.
        """
        self._lock_fd = None
        self._lock_identity = None
        # Identify the lockfile we created BEFORE closing it, so the unlink
        # below removes our own claim and never a successor's.
        try:
            st = os.fstat(fd)
            identity: tuple[int, int] | None = (st.st_dev, st.st_ino)
        except OSError:
            identity = None
        try:
            os.close(fd)
        except OSError:
            pass
        self._unlink_if_ours(identity)

    def release(self) -> None:
        """Release the lock, removing only the lockfile this instance owns.

        The unlink is conditional on identity, not on the path existing.
        Unlinking by path removes whatever lockfile happens to be there
        *now*, which after any hand-off is the next holder's claim — and a
        claim deleted out from under its owner lets the following acquirer
        create a second one, so two processes run the same critical section
        believing they are alone. A second ``release()`` call, or a release
        by an instance that never acquired, therefore removes nothing.
        """
        fd, identity = self._lock_fd, self._lock_identity
        self._lock_fd = None
        self._lock_identity = None
        if fd is not None:
            try:
                self._os_unlock(fd)
            except OSError:
                pass
            finally:
                # close() runs even if the unlock failed — otherwise a
                # filesystem that errors on LOCK_UN leaks a descriptor per
                # release for the lifetime of the process.
                try:
                    os.close(fd)
                except OSError:
                    pass
            self._unlink_if_ours(identity)

        # Release thread lock
        if self._owns_thread_lock:
            self._owns_thread_lock = False
            with self._thread_lock_guard:
                tlock = self._thread_locks.get(self.lock_path)
            if tlock is not None:
                try:
                    tlock.release()
                except RuntimeError:
                    pass

    #: Seconds a lockfile whose contents cannot be read as a pid must sit
    #: untouched before it is treated as abandoned. Long enough that it can
    #: only mean a crash, never a writer mid-handshake.
    _UNREADABLE_LOCK_GRACE_SECONDS = 300

    def _unlink_if_ours(self, identity: tuple[int, int] | None) -> None:
        """Remove the lockfile only while it is still the one in *identity*."""
        if identity is None:
            return
        try:
            st = os.stat(self.lock_path)
        except OSError:
            return  # already gone, or not ours to look at
        if (st.st_dev, st.st_ino) != identity:
            return  # somebody else's claim now — leave it alone
        try:
            os.unlink(self.lock_path)
        except OSError:
            pass

    def _stale_identity(self) -> tuple[int, int] | None:
        """Identify the lockfile only if its owner is *confirmed* gone.

        Returns ``(st_dev, st_ino)`` of a lockfile that may be broken, or
        ``None`` to wait. ``None`` is the answer for every state that is
        merely unknown, and that distinction is the whole point:

        * **empty** — a lockfile is created and then written, so an empty
          one is a live acquirer caught between the two syscalls, not a
          corpse. Reading it as stale let a second process break a lock
          that was being taken and walk straight into the critical section.
        * **missing** — the holder released between our failed create and
          this read. There is nothing to break; the next create attempt
          will simply win.
        * **unreadable / not a pid** — junk on disk is not evidence the
          owner died. Only after
          :attr:`_UNREADABLE_LOCK_GRACE_SECONDS` of no change does it
          become a corpse rather than a mystery.

        Measured before this distinction existed: six processes taking the
        same lock 400 times each recorded **155 mutual-exclusion
        violations** — two holders inside the section at once.
        """
        try:
            st = os.stat(self.lock_path)
            with open(self.lock_path, "r", encoding="utf-8") as f:
                pid_str = f.read().strip()
        except OSError:
            return None  # vanished or unreadable right now: wait, do not break
        identity = (st.st_dev, st.st_ino)

        if not pid_str:
            # Being written right now, or a crash between create and write.
            # Only the clock can tell the two apart.
            return identity if self._older_than_grace(st) else None

        try:
            pid = int(pid_str)
        except ValueError:
            return identity if self._older_than_grace(st) else None

        if sys.platform == "win32":
            return identity if not self._pid_exists_win(pid) else None
        try:
            os.kill(pid, 0)
            return None  # owner is alive
        except ProcessLookupError:
            return identity  # confirmed dead owner
        except PermissionError:
            return None  # alive, just not ours to signal

    def _older_than_grace(self, st: os.stat_result) -> bool:
        """True when *st* has been untouched past the abandonment grace."""
        return (time.time() - st.st_mtime) > self._UNREADABLE_LOCK_GRACE_SECONDS

    def _is_stale(self) -> bool:
        """Whether the existing lockfile is from a process that is gone."""
        return self._stale_identity() is not None

    def _break_stale(self, identity: tuple[int, int] | None = None) -> None:
        """Remove an abandoned lock file — as one atomic winner, not several.

        *identity* is the lockfile :meth:`_stale_identity` judged. Checking
        it before unlinking is necessary and, on its own, not sufficient:
        ``stat`` then ``unlink`` is two syscalls, so two waiters can both
        confirm the dead file, the first can unlink it and create its own
        live claim, and the second's unlink — still aimed at the path —
        removes that claim. Both then create a lockfile and both believe
        they hold the lock. Measured with the identity check alone: four
        overlaps in 4320 acquisitions, down from ~3% but not gone.

        So the break is arbitrated by the OS lock, which is the only thing
        here that is atomic. A breaker must first take the OS lock on the
        abandoned file; the dead owner's is released by the kernel, so it
        succeeds for exactly one waiter, and while it is held nobody else
        can break that inode. The path cannot change underneath it either:
        replacing it requires unlinking this inode first, and that is what
        we hold. A filesystem with no OS locking degrades to the identity
        check alone — the same portability fallback :meth:`_os_lock`
        documents, and no worse than before.
        """
        if identity is None:
            identity = self._stale_identity()
        if identity is None:
            return
        try:
            fd = os.open(self.lock_path, os.O_RDWR)
        except OSError:
            return  # vanished: nothing to break
        try:
            # Identity BEFORE the lock, never after: between judging the
            # file and opening it, its owner may have unlinked it and an
            # acquirer created a fresh one at the same path. Locking that
            # stranger's inode — even for the moment it takes to notice —
            # makes its creator's own OS lock fail EWOULDBLOCK and its
            # perfectly ordinary acquire raise.
            st = os.fstat(fd)
            if (st.st_dev, st.st_ino) != identity:
                return  # not the file we judged
            held = self._try_os_lock(fd)
            if held is None:
                self._unlink_if_ours(identity)  # no OS locking here
                return
            if not held:
                return  # a live process holds it after all
            try:
                on_path = os.stat(self.lock_path)
            except OSError:
                return  # already unlinked by its owner
            if (on_path.st_dev, on_path.st_ino) != identity:
                return  # the path names somebody else's claim now
            try:
                os.unlink(self.lock_path)
            except OSError:
                pass
        finally:
            try:
                self._os_unlock(fd)
            except OSError:
                pass
            try:
                os.close(fd)
            except OSError:
                pass

    def _try_os_lock(self, fd: int) -> bool | None:
        """Take the OS lock without waiting.

        ``True`` when this process now holds it, ``False`` when another
        live process does, and ``None`` when the filesystem does not
        implement advisory locking at all — the three answers
        :meth:`_break_stale` has to tell apart, where :meth:`_os_lock`
        raises for the second and silently succeeds for the third.
        """
        try:
            import fcntl
        except ImportError:
            pass
        else:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return True
            except OSError as e:
                return None if e.errno in _UNSUPPORTED_LOCK_ERRNOS else False
        try:
            import msvcrt
        except ImportError:
            return None
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)  # type: ignore[attr-defined]
            return True
        except OSError as e:
            return None if e.errno in _UNSUPPORTED_LOCK_ERRNOS else False

    @staticmethod
    def _pid_exists_win(pid: int) -> bool:
        """Check if a PID exists on Windows."""
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            handle = kernel32.OpenProcess(0x100000, False, pid)  # SYNCHRONIZE
            if handle:
                kernel32.CloseHandle(handle)
                return True
            return False
        except (OSError, ImportError, AttributeError):
            return False

    def _os_lock(self, fd: int) -> None:
        """Apply OS-level exclusive lock if available.

        OS-level locking is an *upgrade* on top of the O_CREAT|O_EXCL
        lockfile, not the primary mechanism. A filesystem that does not
        implement it (some NFS/FUSE/network mounts return ENOLCK,
        EOPNOTSUPP or ENOSYS) therefore degrades to lockfile-only
        exclusion rather than failing the acquire — the portability
        fallback this module documents. Every other error, including the
        EWOULDBLOCK that means another process really does hold the lock,
        propagates to the caller.
        """
        try:
            import fcntl

            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as e:
                if e.errno not in _UNSUPPORTED_LOCK_ERRNOS:
                    raise
        except ImportError:
            try:
                import msvcrt

                try:
                    msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)  # type: ignore[attr-defined]
                except OSError as e:
                    if e.errno not in _UNSUPPORTED_LOCK_ERRNOS:
                        raise
            except ImportError:
                pass

    def _os_unlock(self, fd: int) -> None:
        """Release OS-level lock."""
        try:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_UN)
        except ImportError:
            try:
                import msvcrt

                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)  # type: ignore[attr-defined]
            except ImportError:
                pass

    def __enter__(self) -> FileLock:
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.release()

    def __repr__(self) -> str:
        return f"FileLock({self.path!r})"


# Aliases for compatibility with huggingface_hub and other packages
# that import BaseFileLock / SoftFileLock from filelock.
BaseFileLock = FileLock
SoftFileLock = FileLock
