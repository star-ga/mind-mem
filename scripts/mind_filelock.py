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

import os
import sys
import threading
import time
from types import TracebackType


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
        self._lock_fd = None
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
                raise LockTimeout(
                    f"Lock timeout ({self.timeout}s) for: {self.lock_path}"
                )
        self._owns_thread_lock = True

        # Layer 2: Acquire cross-process file lock
        try:
            self._acquire_file_lock(start)
        except Exception:
            self._owns_thread_lock = False
            tlock.release()
            raise

    def _acquire_file_lock(self, start: float) -> None:
        """Acquire the filesystem-level lock."""
        while True:
            try:
                fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, f"{os.getpid()}\n".encode())
                self._lock_fd = fd
                self._os_lock(fd)
                return
            except FileExistsError:
                if self._is_stale():
                    self._break_stale()
                    continue

                if self.timeout == 0:
                    raise LockTimeout(f"Could not acquire lock: {self.lock_path}")
                elapsed = time.monotonic() - start
                if self.timeout > 0 and elapsed >= self.timeout:
                    raise LockTimeout(
                        f"Lock timeout ({self.timeout}s) for: {self.lock_path}"
                    )
                time.sleep(self.poll_interval)

    def release(self) -> None:
        """Release the lock."""
        # Release file lock first
        if self._lock_fd is not None:
            try:
                self._os_unlock(self._lock_fd)
                os.close(self._lock_fd)
            except OSError:
                pass
            self._lock_fd = None
        try:
            os.unlink(self.lock_path)
        except OSError:
            pass

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

    def _is_stale(self) -> bool:
        """Check if existing lock file is from a dead process."""
        try:
            with open(self.lock_path, "r") as f:
                pid_str = f.read().strip()
            if not pid_str:
                return True
            pid = int(pid_str)
            if sys.platform == "win32":
                return not self._pid_exists_win(pid)
            else:
                try:
                    os.kill(pid, 0)
                    return False
                except ProcessLookupError:
                    return True
                except PermissionError:
                    return False
        except (OSError, ValueError):
            try:
                age = time.time() - os.path.getmtime(self.lock_path)
                return age > 300
            except OSError:
                return True

    def _break_stale(self) -> None:
        """Remove a stale lock file."""
        try:
            os.unlink(self.lock_path)
        except OSError:
            pass

    @staticmethod
    def _pid_exists_win(pid: int) -> bool:
        """Check if a PID exists on Windows."""
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(0x100000, False, pid)  # SYNCHRONIZE
            if handle:
                kernel32.CloseHandle(handle)
                return True
            return False
        except Exception:
            return False

    def _os_lock(self, fd: int) -> None:
        """Apply OS-level exclusive lock if available."""
        try:
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except ImportError:
            try:
                import msvcrt
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
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
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            except ImportError:
                pass

    def __enter__(self) -> FileLock:
        self.acquire()
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None,
        exc_val: BaseException | None, exc_tb: TracebackType | None,
    ) -> bool:
        self.release()
        return False

    def __repr__(self) -> str:
        return f"FileLock({self.path!r})"


# Aliases for compatibility with huggingface_hub and other packages
# that import BaseFileLock / SoftFileLock from filelock.
BaseFileLock = FileLock
SoftFileLock = FileLock
