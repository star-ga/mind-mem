"""The MCP tool decorator must not let a backend DB error crash the server.

Regression guard: a Postgres ``OperationalError`` is NOT a
``sqlite3.OperationalError``, so per-tool ``except sqlite3.OperationalError``
guards miss it and it propagated out of the stdio server, dropping every
tool mid-session (the 2026-06-05 reindex_dirty crash-loop). The decorator
now converts any DB error from either backend into a structured response,
while non-DB exceptions still propagate.

All global state (the ACL-disabled env var and the USER_TOOLS allowlists)
is mutated via ``monkeypatch`` so it auto-restores and cannot pollute other
tests.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import types

import pytest

from mind_mem.mcp.infra import acl as _acl
from mind_mem.mcp.infra import observability as _obs


def _wrap(monkeypatch, tool_name: str, fn):
    """Register *tool_name* in the ACL allowlist and wrap it like the server."""
    monkeypatch.setenv("MIND_MEM_ACL_DISABLED", "true")
    extended = frozenset(_acl.USER_TOOLS | {tool_name})
    # The decorator reads USER_TOOLS bound into its own module at import
    # time, so patch both bindings.
    monkeypatch.setattr(_acl, "USER_TOOLS", extended)
    monkeypatch.setattr(_obs, "USER_TOOLS", extended)
    fn.__name__ = tool_name
    return _obs.mcp_tool_observe(fn)


def test_sqlite_error_becomes_structured_response(monkeypatch) -> None:
    def boom_sqlite():
        raise sqlite3.OperationalError("no such table: blocks")

    wrapped = _wrap(monkeypatch, "boom_sqlite", boom_sqlite)
    out = wrapped()  # must NOT raise
    payload = json.loads(out)
    assert payload["error"] == "database backend error"
    assert payload["error_type"] == "OperationalError"
    assert payload["tool"] == "boom_sqlite"


def test_postgres_branch_is_reached_without_the_driver_installed(monkeypatch) -> None:
    """The Postgres leg of the backstop, exercised on every matrix row.

    ``_is_db_error`` resolves the driver lazily -- ``import psycopg`` inside
    the function, then ``isinstance(exc, psycopg.Error)`` -- so the module it
    consults is whatever ``sys.modules["psycopg"]`` holds at call time. That
    is the seam this test uses: a stand-in module supplies the ``Error`` base
    the check compares against, and the decorator runs its real
    Postgres branch (observability.py ``_is_db_error`` -> ``isinstance`` ->
    the structured-response return) with no ``[postgres]`` extra present.

    Why this exists beside the real-driver test below: psycopg ships only in
    the ``[postgres]`` extra, which no OS/Python matrix row installs, and the
    dedicated "postgres backend" job selects its files by grepping tests/ for
    its DSN environment variable -- a name this file has no reason to
    mention. So the real-driver test runs on ZERO CI rows, and this branch --
    the one the 2026-06-05 crash-loop was actually about -- had no executing
    assertion anywhere. This test moves it to all 15 rows.

    What it deliberately does NOT claim: that real psycopg exceptions are
    shaped this way. That assumption is a single ``issubclass`` line, pinned
    against the real package in the test below wherever the driver exists.
    """
    stub = types.ModuleType("psycopg")

    class _Error(Exception):
        pass

    class _OperationalError(_Error):
        pass

    stub.Error = _Error  # type: ignore[attr-defined]
    stub.OperationalError = _OperationalError  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "psycopg", stub)

    def boom_pg():
        raise _OperationalError("connection failed: password authentication failed")

    wrapped = _wrap(monkeypatch, "boom_pg_stub", boom_pg)
    out = wrapped()  # must NOT raise — this used to crash the server
    payload = json.loads(out)
    assert payload["error"] == "database backend error"
    assert payload["error_type"] == "_OperationalError"
    # The raw message (which can carry DSN/host) must not leak to client.
    assert "password" not in out.lower()

    # Discriminating control: the structured response above must come from the
    # ``isinstance(exc, psycopg.Error)`` branch, not from a decorator that
    # swallows everything. An exception that is NOT a subclass of the stand-in
    # base still has to propagate while the same stand-in module is installed.
    class _NotADbError(Exception):
        pass

    def boom_other():
        raise _NotADbError("not a database error")

    wrapped_other = _wrap(monkeypatch, "boom_other_stub", boom_other)
    with pytest.raises(_NotADbError):
        wrapped_other()


def test_psycopg_operationalerror_becomes_structured_response(monkeypatch) -> None:
    """The exact incident class: a real psycopg error reaching the decorator.

    deferred: this needs psycopg IMPORTABLE, not a live server -- it only
    raises the exception class. But psycopg is in the ``[postgres]`` extra,
    which no matrix row installs, and the dedicated "postgres backend" job
    selects its files by grepping tests/ for its DSN environment variable --
    a name this file has no reason to mention. Net: this assertion runs on
    ZERO CI rows. Upgrade path (one line, in ci.yml or pyproject.toml): add
    psycopg to the ``[test]`` extra, or make the postgres job's file list
    explicit instead of deriving it from a text search. Deliberately NOT
    fixed by writing that variable's name into this docstring: CI selection
    must not turn on prose, and a comment that silently changes which tests
    run is the same defect class in a nicer costume.

    The decorator's own behaviour is now covered on every row by
    ``test_postgres_branch_is_reached_without_the_driver_installed`` above.
    What remains here, and can only be checked against the real package, is
    the class-hierarchy assumption that stand-in encodes.
    """
    psycopg = pytest.importorskip("psycopg")

    # Pins the one fact the stand-in above cannot establish for itself.
    assert issubclass(psycopg.OperationalError, psycopg.Error)

    def boom_pg():
        raise psycopg.OperationalError("connection failed: password authentication failed")

    wrapped = _wrap(monkeypatch, "boom_pg", boom_pg)
    out = wrapped()  # must NOT raise — this used to crash the server
    payload = json.loads(out)
    assert payload["error"] == "database backend error"
    assert payload["error_type"] == "OperationalError"
    # The raw message (which can carry DSN/host) must not leak to client.
    assert "password" not in out.lower()


def test_non_db_exception_still_propagates(monkeypatch) -> None:
    """Contract preserved: a plain ValueError is NOT swallowed."""

    def boom_value():
        raise ValueError("a genuine bug")

    wrapped = _wrap(monkeypatch, "boom_value", boom_value)
    with pytest.raises(ValueError):
        wrapped()
