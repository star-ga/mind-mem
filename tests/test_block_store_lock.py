"""v3.2.0 §1.4 PR-4 — MarkdownBlockStore.lock() tests."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from mind_mem.block_store import MarkdownBlockStore
from mind_mem.mind_filelock import LockTimeout


@pytest.fixture
def ws(tmp_path: Path) -> Path:
    for d in ("decisions", "tasks", "entities", "intelligence", "memory"):
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
    return tmp_path


class TestWorkspaceLock:
    def test_lock_is_context_manager(self, ws: Path) -> None:
        store = MarkdownBlockStore(str(ws))
        lock = store.lock()
        with lock:
            # FileLock creates a ``<target>.lock`` sidecar while held.
            assert (ws / ".workspace.lock").is_file()
        # Release removes the sidecar.
        assert not (ws / ".workspace.lock").is_file()

    def test_lock_blocks_concurrent_acquire_non_blocking(self, ws: Path) -> None:
        store_a = MarkdownBlockStore(str(ws))
        store_b = MarkdownBlockStore(str(ws))
        with store_a.lock():
            with pytest.raises(LockTimeout):
                with store_b.lock(blocking=False):
                    pass  # pragma: no cover — should not reach

    def test_lock_blocks_concurrent_acquire_with_timeout(self, ws: Path) -> None:
        store_a = MarkdownBlockStore(str(ws))
        store_b = MarkdownBlockStore(str(ws))
        with store_a.lock():
            with pytest.raises(LockTimeout):
                # Small timeout so the test finishes quickly.
                with store_b.lock(blocking=True, timeout=0.1):
                    pass  # pragma: no cover

    def test_lock_releases_on_exception(self, ws: Path) -> None:
        store = MarkdownBlockStore(str(ws))
        with pytest.raises(RuntimeError):
            with store.lock():
                raise RuntimeError("simulated crash")
        # Lock must be re-acquirable after the exception.
        with store.lock(blocking=False):
            pass

    def test_lock_serialises_threaded_writers(self, ws: Path) -> None:
        """Two threads acquiring the lock serialise their critical sections.

        We record start/end timestamps inside the critical section
        and assert the intervals don't overlap.
        """
        store = MarkdownBlockStore(str(ws))
        intervals: list[tuple[float, float]] = []
        intervals_lock = threading.Lock()
        barrier = threading.Barrier(2)

        def worker() -> None:
            import time as _time

            barrier.wait()
            with store.lock(timeout=5.0):
                start = _time.perf_counter()
                _time.sleep(0.02)
                end = _time.perf_counter()
            with intervals_lock:
                intervals.append((start, end))

        t1 = threading.Thread(target=worker)
        t2 = threading.Thread(target=worker)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert len(intervals) == 2
        first, second = sorted(intervals, key=lambda iv: iv[0])
        # Critical sections must not overlap — second thread starts
        # after first thread ends (within a small scheduling fudge).
        assert second[0] >= first[1] - 0.005, f"overlap: {first} vs {second}"

    def test_workspace_dir_created_on_lock(self, tmp_path: Path) -> None:
        """lock() must create the workspace dir if absent (first-run UX)."""
        ws = tmp_path / "fresh"
        assert not ws.is_dir()
        store = MarkdownBlockStore(str(ws))
        with store.lock(blocking=False):
            assert ws.is_dir()
            # FileLock sidecar appears at ``<target>.lock``.
            assert (ws / ".workspace.lock").is_file()
