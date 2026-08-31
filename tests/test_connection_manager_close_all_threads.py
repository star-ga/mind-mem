"""Regression tests: close() must reach read connections in every thread.

Read connections were stored only in ``threading.local()``, which by
construction resolves to the *calling* thread's slot — so ``close()``
closed at most one of them. Every other reader thread's ``sqlite3``
handle (and its ``.db`` / ``-wal`` / ``-shm`` descriptors) stayed open
behind the manager's back, which makes replacing or deleting the database
after ``close()`` race live handles, and fail outright on Windows.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import tempfile
import threading
import unittest

from mind_mem.connection_manager import ConnectionManager


def _is_closed(conn: sqlite3.Connection) -> bool:
    """True when *conn* has actually been closed."""
    try:
        conn.execute("SELECT 1")
    except sqlite3.ProgrammingError:
        return True
    return False


class TestCloseReachesEveryThread(unittest.TestCase):
    def setUp(self) -> None:
        # Workers park on this until tearDown so they stay ALIVE while a test
        # asserts. A joined worker is a DEAD owner, and a dead owner's
        # connection is reaped and closed on the next get_read_connection --
        # deliberately, since nothing can ever reach it again. Letting the
        # helper join meant a test asserting "not yet closed" was really
        # asserting the fd leak. The property here is that close() reaches
        # OTHER LIVE threads; the reaper gets its own test below.
        self._release = threading.Event()
        self._workers: list[threading.Thread] = []
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test.db")
        self.mgr = ConnectionManager(self.db_path)
        with self.mgr.write_lock:
            wconn = self.mgr.get_write_connection()
            wconn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)")
            wconn.commit()

    def tearDown(self) -> None:
        self._release.set()
        for thread in self._workers:
            thread.join(timeout=10)
        self.mgr.close()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _open_in_threads(self, count: int) -> list[sqlite3.Connection]:
        """Open one read connection per worker thread and hand them back."""
        conns: list[sqlite3.Connection] = []
        errors: list[BaseException] = []
        lock = threading.Lock()

        opened = threading.Semaphore(0)

        def worker() -> None:
            try:
                conn = self.mgr.get_read_connection()
                conn.execute("SELECT COUNT(*) FROM t").fetchone()
                with lock:
                    conns.append(conn)
            except BaseException as exc:  # pragma: no cover - defensive
                with lock:
                    errors.append(exc)
            finally:
                opened.release()
                # Stay alive: a joined thread is a dead owner and its
                # connection is legitimately reaped before the test can look.
                self._release.wait(30)

        threads = [threading.Thread(target=worker) for _ in range(count)]
        self._workers.extend(threads)
        for thread in threads:
            thread.start()
        for _ in range(count):
            self.assertTrue(opened.acquire(timeout=10), "worker did not open a connection")

        self.assertEqual(errors, [])
        self.assertEqual(len(conns), count)
        return conns

    def test_close_closes_other_threads_read_connections(self) -> None:
        conns = self._open_in_threads(3)
        self.assertEqual([_is_closed(c) for c in conns], [False, False, False])

        self.mgr.close()

        self.assertEqual([_is_closed(c) for c in conns], [True, True, True])

    def test_close_closes_the_calling_threads_connection_too(self) -> None:
        own = self.mgr.get_read_connection()
        others = self._open_in_threads(2)

        self.mgr.close()

        self.assertTrue(_is_closed(own))
        self.assertTrue(all(_is_closed(c) for c in others))

    def test_a_thread_gets_a_fresh_connection_after_close(self) -> None:
        """A stale handle must never be handed back out."""
        first = self.mgr.get_read_connection()
        self.mgr.close()

        second = self.mgr.get_read_connection()
        self.assertIsNot(first, second)
        self.assertFalse(_is_closed(second))
        self.assertEqual(second.execute("SELECT COUNT(*) FROM t").fetchone()[0], 0)

    def test_worker_thread_also_recovers_after_close(self) -> None:
        before = self._open_in_threads(1)[0]
        self.mgr.close()
        after = self._open_in_threads(1)[0]
        self.assertTrue(_is_closed(before))
        self.assertFalse(_is_closed(after))

    def test_close_is_still_idempotent(self) -> None:
        self._open_in_threads(2)
        self.mgr.close()
        self.mgr.close()

    def test_reads_still_get_one_connection_per_thread(self) -> None:
        """The registry must not turn per-thread reads into a shared handle."""
        conns = self._open_in_threads(3)
        self.assertEqual(len({id(c) for c in conns}), 3)

    def test_repeat_calls_in_one_thread_reuse_the_connection(self) -> None:
        first = self.mgr.get_read_connection()
        self.assertIs(self.mgr.get_read_connection(), first)


if __name__ == "__main__":
    unittest.main()


class TestDeadOwnersAreReaped(unittest.TestCase):
    """A connection whose owning thread exited is closed without a close().

    The registry that lets ``close()`` reach other threads must not become a
    leak of its own: a plain list holds a STRONG reference to every connection
    ever handed out, so each one outlives its thread and nothing can release it
    until ``close()`` runs. Measured on the list version: +2 descriptors per
    dead thread, unbounded under thread churn -- strictly worse than the
    thread-local-only behaviour it replaced, where the connection died with its
    thread. ``sqlite3.Connection`` does not support weak references, so
    ownership is tracked explicitly and dead owners are reaped.
    """

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "reap.db")
        sqlite3.connect(self.db_path).close()
        self.mgr = ConnectionManager(self.db_path)

    def tearDown(self) -> None:
        self.mgr.close()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _open_and_die(self) -> sqlite3.Connection:
        held: list[sqlite3.Connection] = []
        thread = threading.Thread(target=lambda: held.append(self.mgr.get_read_connection()))
        thread.start()
        thread.join(timeout=10)
        self.assertEqual(len(held), 1)
        return held[0]

    def test_a_dead_owners_connection_is_closed_on_the_next_open(self) -> None:
        orphan = self._open_and_die()
        self.assertFalse(_is_closed(orphan), "the owner has only just exited")

        self.mgr.get_read_connection()  # reaps under the registry lock

        self.assertTrue(_is_closed(orphan), "a dead owner's connection must be closed")

    def test_the_registry_does_not_grow_with_thread_churn(self) -> None:
        """The measured symptom: unbounded growth across short-lived threads."""
        import gc

        gc.disable()
        try:
            for _ in range(30):
                self._open_and_die()
            self.mgr.get_read_connection()
            registry = len(self.mgr._read_conns)
        finally:
            gc.enable()

        # 30 dead owners must not accumulate; only live owners may remain.
        self.assertLessEqual(registry, 2, f"registry grew to {registry} across 30 dead threads")
