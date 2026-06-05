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


def test_psycopg_operationalerror_becomes_structured_response(monkeypatch) -> None:
    """The exact incident class: a psycopg error reaching the decorator."""
    psycopg = pytest.importorskip("psycopg")

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
