"""``ToolOutputStore`` on the Postgres backend the module advertises.

Three defects, all of the same family — an answer shaped like success
that did not do the work:

* ``gc()`` returned 0 for any non-sqlite backend without evicting
  anything, and 0 is the value that means "nothing needed evicting", so
  an operator scripting retention was told it had run.
* ``meta()`` returned ``None`` for a non-sqlite backend, which is also
  what it returns for an unknown handle — a stored output was reported
  as missing with no way to tell the two apart.
* the Postgres INSERT omitted the ``ts`` column, so a caller-supplied
  ``ts`` (the documented determinism knob) was dropped for the column's
  ``now()`` default, and its ``ON CONFLICT`` refreshed only the text, so
  a re-store under a different exit code kept the stale one forever
  (the handle hashes source ‖ text, never the exit code).

Exercised against a stand-in connection rather than a live database: the
defects are in what this module asks the database to do, which is
visible in the statements it issues.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any

import pytest

from mind_mem.tool_output import ToolOutputStore


class _FakeCursor:
    def __init__(self, rowcount: int = 0, row: tuple | None = None) -> None:
        self.rowcount = rowcount
        self._row = row

    def fetchone(self) -> tuple | None:
        return self._row


class _FakeConnection:
    """Records every statement; answers DELETE/SELECT with canned results."""

    def __init__(self, *, deleted: int = 0, row: tuple | None = None) -> None:
        self.calls: list[tuple[str, Any]] = []
        self._deleted = deleted
        self._row = row

    def __enter__(self) -> "_FakeConnection":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def execute(self, sql: str, params: Any = None) -> _FakeCursor:
        self.calls.append((" ".join(sql.split()), params))
        if sql.lstrip().upper().startswith("DELETE"):
            return _FakeCursor(rowcount=self._deleted)
        if sql.lstrip().upper().startswith("SELECT"):
            return _FakeCursor(row=self._row)
        return _FakeCursor()

    def statements(self, verb: str) -> list[tuple[str, Any]]:
        return [call for call in self.calls if call[0].upper().startswith(verb.upper())]


@pytest.fixture
def pg_store(monkeypatch: pytest.MonkeyPatch):
    """A postgres-backed store whose connection is a recording stand-in."""

    def _make(**conn_kwargs: Any) -> tuple[ToolOutputStore, _FakeConnection]:
        with tempfile.TemporaryDirectory() as d:
            store = ToolOutputStore(sqlite_path=os.path.join(d, "unused.db"), backend="postgres", max_rows=5)
        con = _FakeConnection(**conn_kwargs)
        monkeypatch.setattr(ToolOutputStore, "_pg", lambda self: con)
        return store, con

    return _make


# ── gc() ──────────────────────────────────────────────────────────────────────


def test_gc_actually_evicts_on_postgres(pg_store) -> None:
    store, con = pg_store(deleted=7)
    assert store.gc() == 7
    deletes = con.statements("DELETE")
    assert len(deletes) == 1
    assert deletes[0][1] == (5,)  # max_rows is the OFFSET


def test_gc_still_evicts_on_sqlite() -> None:
    with tempfile.TemporaryDirectory() as d:
        store = ToolOutputStore(sqlite_path=os.path.join(d, "t.db"), max_rows=2)
        for i in range(5):
            store.store_and_summarize(f"output {i}\n", source="cmd")
        # store_and_summarize already evicts, so a forced gc finds nothing.
        assert store.gc() == 0
        store.max_rows = 1
        assert store.gc() == 1


# ── meta() ────────────────────────────────────────────────────────────────────


def test_meta_reads_the_row_on_postgres(pg_store) -> None:
    store, _con = pg_store(row=("to-abc", "cargo test", 101, "2026-06-01T00:00:00+00:00", 3, 42))
    record = store.meta("to-abc")
    assert record == {
        "handle": "to-abc",
        "source": "cargo test",
        "exit_code": 101,
        "ts": "2026-06-01T00:00:00+00:00",
        "line_count": 3,
        "byte_count": 42,
    }


def test_meta_none_means_unknown_handle_only(pg_store) -> None:
    store, _con = pg_store(row=None)
    assert store.meta("to-missing") is None


# ── store_and_summarize: ts + full upsert ────────────────────────────────────


def _insert(con: _FakeConnection) -> tuple[str, Any]:
    inserts = con.statements("INSERT")
    assert len(inserts) == 1
    return inserts[0]


def test_a_supplied_ts_reaches_the_row_on_postgres(pg_store) -> None:
    store, con = pg_store()
    store.store_and_summarize("hello\n", source="cmd", exit_code=0, ts="2026-06-01T12:00:00+00:00")
    sql, params = _insert(con)
    assert " ts," in sql or "(handle, source, exit_code, ts," in sql
    assert "2026-06-01T12:00:00+00:00" in params


def test_an_empty_ts_leaves_the_storage_clock_to_stamp_the_row(pg_store) -> None:
    store, con = pg_store()
    store.store_and_summarize("hello\n", source="cmd")
    sql, params = _insert(con)
    assert "now()" in sql
    assert None in params  # NULL → COALESCE picks now()


def test_a_re_store_refreshes_exit_code_and_recency(pg_store) -> None:
    store, con = pg_store()
    store.store_and_summarize("hello\n", source="cmd", exit_code=1)
    sql, _params = _insert(con)
    upsert = sql.upper()
    assert "ON CONFLICT" in upsert
    for column in ("exit_code=", "ts=", "line_count=", "byte_count="):
        assert column in sql, f"{column} is never refreshed, so a re-store keeps the first write's value"


def test_the_sqlite_branch_still_writes_the_supplied_ts() -> None:
    with tempfile.TemporaryDirectory() as d:
        store = ToolOutputStore(sqlite_path=os.path.join(d, "t.db"))
        result = store.store_and_summarize("hello\n", source="cmd", exit_code=2, ts="2026-06-01T12:00:00+00:00")
        record = store.meta(result.handle)
        assert record is not None
        assert record["ts"] == "2026-06-01T12:00:00+00:00"
        assert record["exit_code"] == 2
