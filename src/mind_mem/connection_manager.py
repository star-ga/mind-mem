"""SQLite connection manager with read/write separation and WAL mode.

Provides thread-safe connection reuse for mind-mem's SQLite databases.
WAL mode allows concurrent readers while writes are serialized through
a single connection protected by a threading lock.

Usage:
    mgr = ConnectionManager("/path/to/db.sqlite")
    # Reads — one connection per thread, reused
    conn = mgr.get_read_connection()
    row = conn.execute("SELECT ...").fetchone()

    # Writes — single serialized writer
    with mgr.write_lock:
        wconn = mgr.get_write_connection()
        wconn.execute("INSERT ...")
        wconn.commit()

    mgr.close()
"""

from __future__ import annotations

import sqlite3
import threading
from typing import Optional

from .observability import get_logger

_log = get_logger("connection_manager")


class ConnectionManager:
    """Thread-safe SQLite connection manager with WAL-mode read/write separation.

    - Read connections: created per-thread (WAL allows concurrent readers)
    - Write connection: single serialized writer with busy_timeout
    - All connections use WAL mode and busy_timeout pragmas
    """

    def __init__(self, db_path: str, busy_timeout: int = 5000):
        self._db_path = db_path
        self._busy_timeout = busy_timeout
        self._write_lock = threading.Lock()
        self._write_conn: Optional[sqlite3.Connection] = None
        self._local = threading.local()

    @property
    def db_path(self) -> str:
        """Return the database file path."""
        return self._db_path

    def _apply_pragmas(self, conn: sqlite3.Connection, readonly: bool = False) -> None:
        """Apply WAL mode, busy timeout, and synchronous pragmas.

        For read-only connections, journal_mode=WAL may fail if the DB is
        locked by another connection setting WAL concurrently. This is safe
        to ignore — the DB is already in WAL mode if the write connection
        initialized it first.
        """
        try:
            conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.OperationalError:
            if not readonly:
                raise
            # Read connections: WAL already set by write path — safe to skip
        conn.execute(f"PRAGMA busy_timeout={self._busy_timeout}")
        conn.execute("PRAGMA synchronous=NORMAL")

    def get_read_connection(self) -> sqlite3.Connection:
        """Get a thread-local read connection.

        Each thread gets its own connection, reused across calls.
        Read connections have PRAGMA query_only=ON to prevent
        accidental writes.
        """
        conn = getattr(self._local, "read_conn", None)
        if conn is None:
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
            self._apply_pragmas(conn, readonly=True)
            conn.execute("PRAGMA query_only=ON")
            self._local.read_conn = conn
            _log.debug("read_connection_created", thread=threading.current_thread().name)
        return conn

    def get_write_connection(self) -> sqlite3.Connection:
        """Get the single write connection (thread-safe via write_lock).

        Callers MUST hold self.write_lock before calling this method
        and before executing any writes on the returned connection.
        """
        if self._write_conn is None:
            self._write_conn = sqlite3.connect(self._db_path, check_same_thread=False)
            self._apply_pragmas(self._write_conn)
            _log.debug("write_connection_created")
        return self._write_conn

    @property
    def write_lock(self) -> threading.Lock:
        """Return the write serialization lock."""
        return self._write_lock

    def close(self) -> None:
        """Close all managed connections (read + write)."""
        conn = getattr(self._local, "read_conn", None)
        if conn is not None:
            try:
                conn.close()
            except sqlite3.Error:
                pass
            self._local.read_conn = None

        if self._write_conn is not None:
            try:
                self._write_conn.close()
            except sqlite3.Error:
                pass
            self._write_conn = None
        _log.debug("connections_closed")
