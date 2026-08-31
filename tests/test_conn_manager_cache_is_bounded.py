# Copyright 2026 STARGA, Inc.
"""The ConnectionManager cache must be bounded, or it leaks fds forever.

``sqlite_index._conn_managers`` is a module-level ``dict[str,
ConnectionManager]``. Each manager holds an open SQLite read connection,
which costs THREE file descriptors on disk: the db, its ``-wal`` and its
``-shm``.

The cache evicts, but only REACTIVELY: a stale entry is dropped when
``_get_conn_manager`` is called again FOR THAT SAME PATH and the file has
vanished. Nothing re-accesses a workspace once its work is done, so the
entry is never reached again and the connection is held for the life of
the process.

Measured before the fix: a full test run reached **2,101 open fds**, with
55 handles on one ``recall.db``, 38 on its ``-wal``, 13 on its ``-shm``,
and many marked ``(deleted)`` -- the workspace gone, the connection still
open. The run then died with ``ValueError: I/O operation on closed file``
and ``lost sys.stderr``: the process crossed the fd limit and pytest's own
stream was the casualty.

The same shape in production is worse than a red suite. A long-running
MCP server serving many workspaces leaks three descriptors per workspace,
permanently, until it can no longer open a file.
"""

from __future__ import annotations

import os

import pytest

from mind_mem import sqlite_index


def _fd_count() -> int | None:
    """Open fds for this process, or None where /proc is absent.

    Returns None -- NOT 0 -- off Linux, so a caller cannot mistake "could
    not measure" for "measured zero".
    """
    try:
        return len(os.listdir("/proc/self/fd"))
    except OSError:
        return None


@pytest.mark.unit
def test_the_manager_cache_does_not_grow_without_bound(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(sqlite_index, "_conn_managers", {}, raising=False)

    cap = getattr(sqlite_index, "_CONN_MANAGER_CACHE_MAX", None)
    assert isinstance(cap, int) and cap > 0, "the cache must declare a bound"

    for i in range(cap * 3):
        ws = tmp_path / f"ws{i}"
        ws.mkdir()
        sqlite_index._get_conn_manager(str(ws))

    assert len(sqlite_index._conn_managers) <= cap, f"cache holds {len(sqlite_index._conn_managers)} managers for a bound of {cap}"


@pytest.mark.unit
def test_eviction_actually_closes_the_connection(tmp_path, monkeypatch) -> None:
    """Bounding the dict is not enough -- an evicted manager must be CLOSED.

    Dropping the reference without closing leaves the descriptors held
    until the garbage collector happens to run, which is exactly the
    non-determinism this leak needs to stop being.
    """
    monkeypatch.setattr(sqlite_index, "_conn_managers", {}, raising=False)
    cap = sqlite_index._CONN_MANAGER_CACHE_MAX

    before = _fd_count()
    if before is None:
        pytest.skip("/proc/self/fd unavailable; cannot measure descriptors here")

    for i in range(cap * 4):
        ws = tmp_path / f"w{i}"
        ws.mkdir()
        mgr = sqlite_index._get_conn_manager(str(ws))
        mgr.get_read_connection()  # force the connection open

    after = _fd_count()
    assert after is not None
    # Each live manager may legitimately hold db + wal + shm. Anything far
    # beyond the bound is the leak.
    assert after - before <= cap * 3 + 20, (
        f"fds grew by {after - before} across {cap * 4} workspaces (bound {cap}); evicted managers are not being closed"
    )
