#!/usr/bin/env python3
"""Backend-parity regression tests for ``sqlite_index`` (audit bugs 4, 9, 13, 14).

These cover the FTS index build / status paths on BOTH backends:

* **SQLite / Markdown** (the default) — must stay byte-for-byte green.
* **Postgres** — ``build_index`` must source rows from the configured
  block store via :func:`mind_mem.storage.iter_active_blocks` (bugs 4 & 9),
  not from the empty Markdown corpus templates, so recall works out of the
  box and no ``sqlite_only_count`` drift is introduced.

The read-only ``index_status`` / ``merkle_leaves`` crash (bugs 13 & 14)
is backend-independent and exercised on the default SQLite path.

Postgres tests use a uniquely-named scratch schema and drop it on
teardown. When psycopg is unavailable or no live Postgres is reachable,
the Postgres tests skip gracefully so the SQLite-only CI stays green.

To run the Postgres leg, point at a live instance::

    MIND_MEM_TEST_PG_DSN="postgresql://mindmem:...@127.0.0.1:5432/mindmem" \\
        pytest tests/test_sqlite_index_backends.py -v
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import uuid
from collections.abc import Generator

import pytest

from mind_mem.sqlite_index import (
    _db_path,
    _index_schema_present,
    build_index,
    index_status,
    merkle_leaves,
    query_index,
)

# ─── Markdown / SQLite default workspace helper ──────────────────────────────

_CORPUS_STUB_FILES = (
    "entities/people.md",
    "entities/tools.md",
    "entities/projects.md",
    "intelligence/CONTRADICTIONS.md",
    "intelligence/DRIFT.md",
    "intelligence/SIGNALS.md",
)


def _make_markdown_workspace(tmpdir: str, decisions: str = "") -> str:
    for d in ("decisions", "tasks", "entities", "intelligence", "memory"):
        os.makedirs(os.path.join(tmpdir, d), exist_ok=True)
    with open(os.path.join(tmpdir, "decisions", "DECISIONS.md"), "w", encoding="utf-8") as f:
        f.write(decisions or "# Decisions\n")
    for fname in _CORPUS_STUB_FILES:
        path = os.path.join(tmpdir, fname)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# {os.path.basename(fname)}\n")
    return tmpdir


# ─── Bugs 13 & 14: read-only status must not CREATE TABLE ─────────────────────


class TestReadonlyStatusNoDDL:
    """index_status / merkle_leaves must never write on a mode=ro connection."""

    def test_index_status_on_calibration_only_db_does_not_crash(self):
        """recall.db created by a side-table (calibration) lacks the FTS
        schema. ``index_status`` opens it mode=ro; it must report
        ``blocks=0`` instead of raising
        ``OperationalError: attempt to write a readonly database``.
        """
        with tempfile.TemporaryDirectory() as ws:
            db = _db_path(ws)
            os.makedirs(os.path.dirname(db), exist_ok=True)
            conn = sqlite3.connect(db)
            conn.execute("CREATE TABLE calibration_feedback (id INTEGER PRIMARY KEY, weight REAL)")
            conn.commit()
            conn.close()

            st = index_status(ws)  # must not raise
            assert st["exists"] is True
            assert st["blocks"] == 0
            assert st.get("schema_built") is False

    def test_merkle_leaves_on_half_initialised_db_returns_empty(self):
        with tempfile.TemporaryDirectory() as ws:
            db = _db_path(ws)
            os.makedirs(os.path.dirname(db), exist_ok=True)
            conn = sqlite3.connect(db)
            conn.execute("CREATE TABLE calibration_feedback (id INTEGER PRIMARY KEY, weight REAL)")
            conn.commit()
            conn.close()

            assert merkle_leaves(ws) == []  # must not raise

    def test_index_schema_present_detects_built_index(self):
        with tempfile.TemporaryDirectory() as ws:
            _make_markdown_workspace(ws, decisions="[D-20260101-001]\nStatement: Use FTS\nStatus: active\n")
            build_index(ws, incremental=False)
            conn = sqlite3.connect(f"file:{_db_path(ws)}?mode=ro", uri=True)
            try:
                assert _index_schema_present(conn) is True
            finally:
                conn.close()


# ─── Bugs 4 & 9: build_index must mirror the SQLite default behaviour ─────────


class TestMarkdownBuildUnchanged:
    """The default markdown path must keep indexing the corpus as before."""

    def test_markdown_build_indexes_corpus_blocks(self):
        with tempfile.TemporaryDirectory() as ws:
            _make_markdown_workspace(
                ws,
                decisions="[D-20260101-001]\nStatement: SQLite is the default backend\nStatus: active\nDate: 2026-01-01\n",
            )
            summary = build_index(ws, incremental=False)
            assert summary["blocks_indexed"] >= 1
            assert summary["total_blocks"] >= 1
            # markdown path keeps the file-driven summary shape
            assert summary["files_checked"] > 0
            assert "source" not in summary  # store-only marker absent on markdown
            hits = query_index(ws, "SQLite", limit=5)
            assert any(h["_id"] == "D-20260101-001" for h in hits)


# ─── Postgres leg (skips when no live DB) ─────────────────────────────────────

psycopg = pytest.importorskip("psycopg", reason="psycopg not installed; skipping Postgres parity tests")

from mind_mem.block_store_postgres import PostgresBlockStore  # noqa: E402

_DEFAULT_DSN = None


def _pg_dsn() -> str | None:
    """Resolve a usable DSN, or None when no live Postgres is reachable."""
    dsn = os.environ.get("MIND_MEM_TEST_PG_DSN")
    try:
        conn = psycopg.connect(dsn, connect_timeout=4)
        conn.close()
    except Exception:
        return None
    return dsn


@pytest.fixture
def pg_workspace() -> Generator[tuple[str, PostgresBlockStore, str], None, None]:
    """Yield (workspace, store, dsn) for a Postgres-backed workspace.

    Uses a uniquely-named scratch schema (``mm_fix_<token>``) and drops
    it on teardown. Skips when no live Postgres is reachable.
    """
    dsn = _pg_dsn()
    if not dsn:
        pytest.skip("no live Postgres reachable — skipping Postgres parity leg")
    schema = f"mm_fix_{uuid.uuid4().hex[:12]}"
    ws = tempfile.mkdtemp(prefix="mm_pg_idx_")
    with open(os.path.join(ws, "mind-mem.json"), "w", encoding="utf-8") as f:
        json.dump({"block_store": {"backend": "postgres", "dsn": dsn, "schema": schema}}, f)
    store = PostgresBlockStore(dsn=dsn, schema=schema, workspace=ws)
    store._ensure_schema()
    try:
        yield ws, store, dsn
    finally:
        try:
            store.close()
        except Exception:
            pass
        try:
            conn = psycopg.connect(dsn)
            conn.autocommit = True
            conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
            conn.close()
        except Exception:
            pass


class TestPostgresBuildSourcesStore:
    """build_index on a PG workspace indexes the store, not empty markdown."""

    def test_build_index_indexes_pg_blocks(self, pg_workspace):
        ws, store, _ = pg_workspace
        store.write_block(
            {
                "_id": "D-20260613-301",
                "_source_file": "decisions/DECISIONS.md",
                "Statement": "The default backend is Postgres for this workspace",
                "Status": "active",
                "Date": "2026-06-13",
            }
        )
        store.write_block(
            {
                "_id": "D-20260613-302",
                "_source_file": "decisions/DECISIONS.md",
                "Statement": "Recall must query the elephant block store directly",
                "Status": "active",
                "Date": "2026-06-13",
            }
        )

        summary = build_index(ws, incremental=False)
        # Bug 4 & 9: was 0 (markdown templates empty); now sources PG.
        assert summary["blocks_indexed"] == 2
        assert summary["total_blocks"] == 2
        assert summary["source"] == "block_store"

        # No sqlite_only drift: exactly the two PG blocks, no template rows.
        st = index_status(ws)
        assert st["blocks"] == 2

        # Each PG block is independently findable through the FTS index.
        assert [h["_id"] for h in query_index(ws, "Postgres", limit=5)] == ["D-20260613-301"]
        assert [h["_id"] for h in query_index(ws, "elephant", limit=5)] == ["D-20260613-302"]

    def test_rebuild_drops_blocks_deleted_from_store(self, pg_workspace):
        ws, store, _ = pg_workspace
        store.write_block(
            {
                "_id": "D-20260613-401",
                "_source_file": "decisions/DECISIONS.md",
                "Statement": "Transient decision that will be removed",
                "Status": "active",
            }
        )
        build_index(ws, incremental=False)
        assert index_status(ws)["blocks"] == 1

        # Remove it from the store, then rebuild — the FTS cache must follow.
        store.delete_block("D-20260613-401")
        summary = build_index(ws, incremental=False)
        assert summary["blocks_indexed"] == 0
        assert index_status(ws)["blocks"] == 0
        assert query_index(ws, "Transient", limit=5) == []

    def test_index_status_does_not_crash_before_first_pg_build(self, pg_workspace):
        """On a fresh PG workspace whose recall.db has only side-tables,
        index_status must report blocks=0 — not raise the readonly DDL crash.
        """
        ws, _store, _ = pg_workspace
        db = _db_path(ws)
        os.makedirs(os.path.dirname(db), exist_ok=True)
        conn = sqlite3.connect(db)
        conn.execute("CREATE TABLE calibration_feedback (id INTEGER PRIMARY KEY, weight REAL)")
        conn.commit()
        conn.close()

        st = index_status(ws)  # must not raise
        assert st["exists"] is True
        assert st["blocks"] == 0
