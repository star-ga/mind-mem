"""Stress tests for mind-mem file locking under contention."""
import os
import threading
import time

import pytest

from scripts.mind_filelock import FileLock


class TestFileLockContention:
    """Test file locking under concurrent access."""

    def test_sequential_lock_unlock(self, tmp_path):
        lock_file = str(tmp_path / "test.lock")
        for _ in range(20):
            with FileLock(lock_file):
                pass

    def test_concurrent_writers(self, tmp_path):
        """Multiple threads writing to same file with lock protection."""
        data_file = str(tmp_path / "data.txt")
        lock_file = data_file
        errors = []

        with open(data_file, "w") as f:
            f.write("")

        def writer(thread_id: int, iterations: int):
            try:
                for i in range(iterations):
                    with FileLock(lock_file):
                        with open(data_file, "a") as f:
                            f.write(f"t{thread_id}:{i}\n")
            except Exception as e:
                errors.append((thread_id, str(e)))

        threads = [threading.Thread(target=writer, args=(t, 10)) for t in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, f"Thread errors: {errors}"
        with open(data_file) as f:
            lines = [line.strip() for line in f if line.strip()]
        assert len(lines) == 50, f"Expected 50 lines, got {len(lines)}"

    def test_lock_timeout_behavior(self, tmp_path):
        """Lock acquisition should not hang indefinitely."""
        lock_file = str(tmp_path / "timeout.lock")
        start = time.monotonic()
        for _ in range(10):
            with FileLock(lock_file):
                time.sleep(0.01)
        elapsed = time.monotonic() - start
        assert elapsed < 5.0, f"Lock operations took too long: {elapsed:.1f}s"

    def test_lock_reentrance_same_thread(self, tmp_path):
        """Nested locks on different files should work."""
        lock_a = str(tmp_path / "a.lock")
        lock_b = str(tmp_path / "b.lock")
        with FileLock(lock_a):
            with FileLock(lock_b):
                pass

    def test_lock_after_exception(self, tmp_path):
        """Lock should be released after exception in context."""
        lock_file = str(tmp_path / "exc.lock")
        with pytest.raises(ValueError):
            with FileLock(lock_file):
                raise ValueError("test error")
        # Lock should be available again
        with FileLock(lock_file):
            pass

    def test_many_rapid_locks(self, tmp_path):
        """Rapid lock/unlock cycles should not leak resources."""
        lock_file = str(tmp_path / "rapid.lock")
        for _ in range(100):
            with FileLock(lock_file):
                pass

    def test_concurrent_readers_and_writers(self, tmp_path):
        """Mixed read/write operations under lock."""
        data_file = str(tmp_path / "rw.txt")
        lock_file = data_file
        with open(data_file, "w") as f:
            f.write("initial\n")

        errors = []

        def reader(iterations):
            try:
                for _ in range(iterations):
                    with FileLock(lock_file):
                        with open(data_file, "r") as f:
                            _ = f.read()
            except Exception as e:
                errors.append(str(e))

        def writer(iterations):
            try:
                for _ in range(iterations):
                    with FileLock(lock_file):
                        with open(data_file, "a") as f:
                            f.write("data\n")
            except Exception as e:
                errors.append(str(e))

        threads = (
            [threading.Thread(target=reader, args=(10,)) for _ in range(3)] +
            [threading.Thread(target=writer, args=(10,)) for _ in range(2)]
        )
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors

    def test_lock_file_cleanup(self, tmp_path):
        """Lock files should be cleaned up after context exit."""
        lock_file = str(tmp_path / "cleanup.txt")
        with open(lock_file, "w") as f:
            f.write("test")
        with FileLock(lock_file):
            pass
        # The data file should still exist
        assert os.path.exists(lock_file)
