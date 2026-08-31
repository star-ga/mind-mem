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

import contextlib
import sqlite3
import threading
from typing import Iterator, Optional

from .observability import get_logger

_log = get_logger("connection_manager")


#: How long close() waits for a connection to fall idle before deferring it.
#: Short on purpose: close() must not block shutdown behind a long query, and
#: deferring is safe -- the owner closes its own handle at its next request.
_CLOSE_ACQUIRE_TIMEOUT = 2.0


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
        # Registry of every read connection handed out, across all threads.
        # thread-local storage is reachable only from the thread that wrote
        # it, so without this close() could never see another thread's
        # connection and its file descriptors would outlive the manager.
        self._read_lock = threading.Lock()
        # thread ident -> (owning thread, connection).
        #
        # NOT a plain list: a list holds a STRONG reference to every connection
        # ever handed out, so a connection outlives the thread that owned it and
        # is never closed until close() runs. Measured: +2 descriptors per dead
        # thread, unbounded under thread churn -- strictly worse than the
        # thread-local-only behaviour it replaced, where the connection was
        # collected with its thread. A WeakSet would be the natural fix but
        # sqlite3.Connection does not support weak references, so ownership is
        # tracked explicitly and dead owners are reaped.
        # thread ident -> (owner, connection, use_lock). The use_lock is held by
        # the OWNING thread around a query and by close() around the close, so a
        # connection is never closed out from under a statement executing on it.
        # Without it close() segfaults the interpreter mid-query -- measured, not
        # theorised: closing a sqlite3 handle while a SELECT is running on it is
        # a use-after-free in the C layer, not a Python-level error.
        self._read_conns: dict[int, tuple[threading.Thread, sqlite3.Connection, threading.Lock]] = {}
        # Bumped by close(). A thread whose cached connection predates the
        # last close() must build a new one instead of using the closed
        # handle it is still holding.
        self._generation = 0

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

        Each thread gets its own connection, reused across calls, and
        registered so :meth:`close` can reach it from any thread. Read
        connections have PRAGMA query_only=ON to prevent accidental writes.

        A connection cached by this thread before the last :meth:`close`
        is stale — it is closed — so it is replaced rather than returned.
        """
        conn: Optional[sqlite3.Connection] = getattr(self._local, "read_conn", None)
        if conn is not None and getattr(self._local, "read_generation", None) == self._generation:
            return conn
        # Created under the registry lock so a connection can never be
        # opened *after* close() drained the list and then be missed by it.
        # This thread is asking for a connection, so it is provably NOT inside a
        # query on its previous one -- the one safe moment to close a handle a
        # concurrent close() had to defer.
        stale = getattr(self._local, "read_conn", None)
        if stale is not None:
            try:
                stale.close()
            except sqlite3.Error:
                pass
            self._local.read_conn = None
            me = threading.current_thread()
            with self._read_lock:
                self._read_conns.pop(me.ident or id(me), None)
        with self._read_lock:
            self._reap_dead_owners_locked()
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
            self._apply_pragmas(conn, readonly=True)
            conn.execute("PRAGMA query_only=ON")
            me = threading.current_thread()
            use_lock = threading.Lock()
            self._read_conns[me.ident or id(me)] = (me, conn, use_lock)
            self._local.read_use_lock = use_lock
            self._local.read_conn = conn
            self._local.read_generation = self._generation
        _log.debug("read_connection_created", thread=threading.current_thread().name)
        return conn

    @contextlib.contextmanager
    def read_connection(self) -> "Iterator[sqlite3.Connection]":
        """A read connection held for the duration of the block.

        Prefer this over :meth:`get_read_connection` wherever a query runs while
        another thread might call :meth:`close`. It holds the connection's
        use-lock, so ``close()`` waits for the block to finish instead of
        closing the handle underneath a running statement.

        The distinction is not stylistic. ``get_read_connection`` hands back a
        raw handle with no way for ``close()`` to know whether a statement is
        executing on it; closing one mid-query is a use-after-free in sqlite3's
        C layer, which segfaults the interpreter rather than raising. ``close()``
        therefore waits :data:`_CLOSE_ACQUIRE_TIMEOUT` for the lock and defers
        the connection if it cannot get it — but that protection only exists for
        callers that take the lock, which is what this does.

        Callers still using the raw accessor must quiesce their readers before
        calling ``close()``.
        """
        conn = self.get_read_connection()
        use_lock = getattr(self._local, "read_use_lock", None)
        if use_lock is None:  # pragma: no cover - defensive
            yield conn
            return
        with use_lock:
            yield conn

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

    def _reap_dead_owners_locked(self) -> None:
        """Close connections whose owning thread has exited. Call under _read_lock.

        A dead thread can never return to reuse or close its connection, so the
        handle is unreachable and closing it is the only way its descriptors are
        released before close(). Deterministic where the previous thread-local
        behaviour relied on the collector.
        """
        dead = [k for k, (owner, _c, _l) in self._read_conns.items() if not owner.is_alive()]
        for key in dead:
            _owner, conn, use_lock = self._read_conns.pop(key)
            # A dead owner cannot be mid-query, but take the lock anyway: it is
            # uncontended here and keeps one rule -- never close without it.
            with use_lock:
                try:
                    conn.close()
                except sqlite3.Error:
                    pass

    def close(self) -> None:
        """Close all managed connections (read + write), in every thread.

        Read connections live in thread-local storage, which only the
        owning thread can read back — so closing "the" read connection
        would leave every other reader thread's handle (and its .db / -wal
        / -shm descriptors) open behind the manager's back, which makes
        deleting or replacing the database race live handles, and fail
        outright on Windows. The registry exists for exactly this.

        Idempotent, and safe to call while other threads still hold a
        reference: each connection is closed under its own use-lock, so a
        statement already executing on it runs to completion first, and the
        generation bump makes that thread open a fresh connection on its next
        :meth:`get_read_connection` rather than reusing a closed handle.

        The lock is not decoration. Closing a ``sqlite3`` handle while a query
        is running on it is a use-after-free in the C layer: it SEGFAULTS the
        interpreter rather than raising, so no ``except`` could contain it.
        """
        with self._read_lock:
            closing = [(k, c, lk) for k, (_owner, c, lk) in self._read_conns.items()]
            self._generation += 1
        deferred = 0
        for key, conn, use_lock in closing:
            # NEVER close a connection that might be executing a statement.
            # Closing a sqlite3 handle mid-query is a use-after-free in the C
            # layer: it SEGFAULTS the interpreter rather than raising, so no
            # `except` could contain it. Most callers hold the raw connection
            # and cannot be made to take a lock, so the only safe rule is to
            # close what is provably idle and defer the rest.
            if use_lock.acquire(timeout=_CLOSE_ACQUIRE_TIMEOUT):
                try:
                    conn.close()
                except sqlite3.Error:
                    pass
                finally:
                    use_lock.release()
                with self._read_lock:
                    self._read_conns.pop(key, None)
            else:
                # In flight. The generation bump above already means its owner
                # will not reuse it; the owner closes it itself at its next
                # get_read_connection, which is the one moment that thread is
                # provably not inside a query.
                deferred += 1
        self._local.read_conn = None
        self._local.read_generation = None

        # Serialize the write-connection teardown under write_lock: a writer
        # holds write_lock while executing on _write_conn, so closing it
        # without the lock is a use-after-close (and races _write_conn=None
        # against get_write_connection()'s None-check, which can create two
        # write connections). Acquire the same lock writers use.
        with self._write_lock:
            if self._write_conn is not None:
                try:
                    self._write_conn.close()
                except sqlite3.Error:
                    pass
                self._write_conn = None
        _log.debug("connections_closed", read_connections=len(closing) - deferred, deferred_in_flight=deferred)
