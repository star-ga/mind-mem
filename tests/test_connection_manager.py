"""Tests for ConnectionManager — SQLite connection pooling with read/write separation (#466)."""

import os
import sqlite3
import tempfile
import threading
import unittest

from mind_mem.connection_manager import ConnectionManager


class TestConnectionManager(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test.db")
        self.mgr = ConnectionManager(self.db_path)

    def tearDown(self):
        self.mgr.close()
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_db_path_property(self):
        self.assertEqual(self.mgr.db_path, self.db_path)

    def test_write_connection_creates_db(self):
        with self.mgr.write_lock:
            conn = self.mgr.get_write_connection()
            conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)")
            conn.commit()
        self.assertTrue(os.path.isfile(self.db_path))

    def test_read_connection_returns_connection(self):
        # Create the DB first via write
        with self.mgr.write_lock:
            wconn = self.mgr.get_write_connection()
            wconn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)")
            wconn.execute("INSERT INTO t VALUES (1, 'hello')")
            wconn.commit()

        rconn = self.mgr.get_read_connection()
        row = rconn.execute("SELECT val FROM t WHERE id = 1").fetchone()
        self.assertEqual(row[0], "hello")

    def test_read_connection_is_reused(self):
        """Same thread should get the same read connection."""
        conn1 = self.mgr.get_read_connection()
        conn2 = self.mgr.get_read_connection()
        self.assertIs(conn1, conn2)

    def test_write_connection_is_reused(self):
        """Successive calls return the same write connection."""
        with self.mgr.write_lock:
            conn1 = self.mgr.get_write_connection()
        with self.mgr.write_lock:
            conn2 = self.mgr.get_write_connection()
        self.assertIs(conn1, conn2)

    def test_read_and_write_are_different_connections(self):
        rconn = self.mgr.get_read_connection()
        with self.mgr.write_lock:
            wconn = self.mgr.get_write_connection()
        self.assertIsNot(rconn, wconn)

    def test_wal_mode_enabled_write(self):
        with self.mgr.write_lock:
            conn = self.mgr.get_write_connection()
            row = conn.execute("PRAGMA journal_mode").fetchone()
        self.assertEqual(row[0], "wal")

    def test_wal_mode_enabled_read(self):
        # Create the DB first so WAL is established
        with self.mgr.write_lock:
            wconn = self.mgr.get_write_connection()
            wconn.execute("CREATE TABLE t (id INTEGER)")
            wconn.commit()

        conn = self.mgr.get_read_connection()
        row = conn.execute("PRAGMA journal_mode").fetchone()
        self.assertEqual(row[0], "wal")

    def test_read_connection_is_query_only(self):
        """Read connection should reject writes."""
        with self.mgr.write_lock:
            wconn = self.mgr.get_write_connection()
            wconn.execute("CREATE TABLE t (id INTEGER)")
            wconn.commit()

        rconn = self.mgr.get_read_connection()
        with self.assertRaises(sqlite3.OperationalError):
            rconn.execute("INSERT INTO t VALUES (1)")

    def test_busy_timeout_set(self):
        mgr = ConnectionManager(self.db_path, busy_timeout=7000)
        try:
            with mgr.write_lock:
                conn = mgr.get_write_connection()
                row = conn.execute("PRAGMA busy_timeout").fetchone()
            self.assertEqual(row[0], 7000)
        finally:
            mgr.close()

    def test_default_busy_timeout(self):
        with self.mgr.write_lock:
            conn = self.mgr.get_write_connection()
            row = conn.execute("PRAGMA busy_timeout").fetchone()
        self.assertEqual(row[0], 5000)

    def test_synchronous_normal(self):
        with self.mgr.write_lock:
            conn = self.mgr.get_write_connection()
            row = conn.execute("PRAGMA synchronous").fetchone()
        # 1 = NORMAL
        self.assertEqual(row[0], 1)

    def test_close_cleans_up(self):
        self.mgr.get_read_connection()
        with self.mgr.write_lock:
            self.mgr.get_write_connection()
        self.mgr.close()

        # After close, connections should be unusable (or closed)
        # Re-creating manager should work fine
        mgr2 = ConnectionManager(self.db_path)
        with mgr2.write_lock:
            conn = mgr2.get_write_connection()
            conn.execute("CREATE TABLE IF NOT EXISTS t (id INTEGER)")
            conn.commit()
        mgr2.close()

    def test_close_idempotent(self):
        """Calling close() multiple times should not raise."""
        self.mgr.close()
        self.mgr.close()

    def test_thread_safe_reads(self):
        """Multiple threads should get independent read connections."""
        with self.mgr.write_lock:
            wconn = self.mgr.get_write_connection()
            wconn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)")
            for i in range(10):
                wconn.execute("INSERT INTO t VALUES (?, ?)", (i, f"row-{i}"))
            wconn.commit()

        results = {}
        errors = []

        def reader(tid):
            try:
                conn = self.mgr.get_read_connection()
                rows = conn.execute("SELECT COUNT(*) FROM t").fetchone()
                results[tid] = rows[0]
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=reader, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        self.assertEqual(len(errors), 0, f"Errors in reader threads: {errors}")
        for tid, count in results.items():
            self.assertEqual(count, 10)

    def test_write_serialization(self):
        """Concurrent writers should be serialized by write_lock."""
        with self.mgr.write_lock:
            wconn = self.mgr.get_write_connection()
            wconn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)")
            wconn.commit()

        errors = []
        write_order = []

        def writer(tid):
            try:
                with self.mgr.write_lock:
                    conn = self.mgr.get_write_connection()
                    conn.execute("INSERT INTO t VALUES (?)", (tid,))
                    conn.commit()
                    write_order.append(tid)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        self.assertEqual(len(errors), 0, f"Errors in writer threads: {errors}")
        self.assertEqual(len(write_order), 10)

        # Verify all rows written
        with self.mgr.write_lock:
            conn = self.mgr.get_write_connection()
            row = conn.execute("SELECT COUNT(*) FROM t").fetchone()
        self.assertEqual(row[0], 10)

    def test_concurrent_read_write(self):
        """Readers should not block on writers and vice versa (WAL mode)."""
        with self.mgr.write_lock:
            wconn = self.mgr.get_write_connection()
            wconn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)")
            wconn.execute("INSERT INTO t VALUES (1, 'initial')")
            wconn.commit()

        errors = []
        read_results = []

        def reader():
            try:
                conn = self.mgr.get_read_connection()
                row = conn.execute("SELECT val FROM t WHERE id = 1").fetchone()
                if row:
                    read_results.append(row[0])
            except Exception as e:
                errors.append(e)

        def writer():
            try:
                with self.mgr.write_lock:
                    conn = self.mgr.get_write_connection()
                    conn.execute("UPDATE t SET val = 'updated' WHERE id = 1")
                    conn.commit()
            except Exception as e:
                errors.append(e)

        # Start reader and writer concurrently
        rt = threading.Thread(target=reader)
        wt = threading.Thread(target=writer)
        rt.start()
        wt.start()
        rt.join(timeout=10)
        wt.join(timeout=10)

        self.assertEqual(len(errors), 0, f"Errors: {errors}")
        # Reader should have gotten either 'initial' or 'updated' — no error
        self.assertEqual(len(read_results), 1)
        self.assertIn(read_results[0], ("initial", "updated"))

    def test_different_threads_get_different_read_connections(self):
        """Each thread should get its own read connection (thread-local)."""
        # Initialize DB with WAL mode via write connection first
        with self.mgr.write_lock:
            wconn = self.mgr.get_write_connection()
            wconn.execute("CREATE TABLE IF NOT EXISTS t (id INTEGER)")
            wconn.commit()

        connections = {}
        errors = []

        def get_conn(tid):
            try:
                conn = self.mgr.get_read_connection()
                connections[tid] = id(conn)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=get_conn, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        self.assertEqual(len(errors), 0, f"Errors: {errors}")
        # All 3 threads should have different connection objects
        unique_ids = set(connections.values())
        self.assertEqual(len(unique_ids), 3)

    def test_write_lock_is_threading_lock(self):
        self.assertIsInstance(self.mgr.write_lock, type(threading.Lock()))


if __name__ == "__main__":
    unittest.main()
