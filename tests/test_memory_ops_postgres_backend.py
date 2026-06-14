"""Backend-aware memory_ops tools — Postgres parity (audit bug 5).

``export_memory`` / ``get_block`` / ``index_stats`` / ``memory_health`` in
``mind_mem.mcp.tools.memory_ops`` enumerated blocks via ``parse_file`` over
:data:`CORPUS_DIRS`, ignoring the configured ``block_store`` backend. On a
Postgres-backed workspace the corpus files are empty init templates and the
blocks of record live in the ``blocks`` table, so:

* ``export_memory`` exported 0 blocks,
* ``get_block`` returned ``found: false`` for a block that ``get_by_id``
  resolves,
* ``index_stats`` / ``memory_health`` reported an empty workspace.

The tools now branch on :func:`memory_ops._is_markdown_backend` and route the
block enumeration through ``get_block_store(ws).get_all()`` / ``get_by_id()``
on non-Markdown backends.

Coverage:

* **SQLite / Markdown backend** (the zero-config default, no DB) — the four
  tools keep reading the Markdown corpus, guarding the default path against
  regression.
* **Postgres backend** (live DB, isolated scratch schema) — the four tools
  surface PG-resident blocks that the Markdown scan never sees. Each run
  creates and drops a uniquely-named scratch schema; the whole PG section
  skips gracefully when psycopg / a live Postgres is unavailable so
  SQLite-only CI stays green. The production ``mind_mem`` schema is never
  touched.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Generator

import pytest

from mind_mem.mcp.infra.workspace import use_workspace
from mind_mem.mcp.tools.memory_ops import (
    export_memory,
    get_block,
    index_stats,
    memory_health,
)

# ─── Shared workspace scaffolding ─────────────────────────────────────────────


def _write_workspace(ws: Path, *, block_store: dict | None = None) -> None:
    """Scaffold a minimal workspace with corpus directories + config."""
    for sub in ("decisions", "tasks", "entities", "intelligence", "memory"):
        (ws / sub).mkdir(parents=True, exist_ok=True)
    config: dict = {"recall": {"backend": "scan"}}
    if block_store is not None:
        config["block_store"] = block_store
    (ws / "mind-mem.json").write_text(json.dumps(config), encoding="utf-8")


def _md_block(bid: str, statement: str, status: str = "active") -> str:
    return f"[{bid}]\nStatement: {statement}\nStatus: {status}\nDate: 2026-06-13\n\n---\n"


@pytest.fixture(autouse=True)
def _admin_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    """``export_memory`` is admin-scoped; opt the test process in."""
    monkeypatch.setenv("MIND_MEM_SCOPE", "admin")


# ─── SQLite / Markdown default (no DB required) ───────────────────────────────


class TestMarkdownDefaultUnchanged:
    """The zero-config Markdown/SQLite path must not regress."""

    def test_export_memory_returns_markdown_blocks(self, tmp_path: Path) -> None:
        ws = tmp_path / "ws"
        _write_workspace(ws)
        (ws / "decisions" / "DECISIONS.md").write_text(
            _md_block("D-20260613-201", "the default backend is SQLite"),
            encoding="utf-8",
        )
        with use_workspace(str(ws)):
            payload = json.loads(export_memory())
        assert payload["block_count"] == 1
        rows = [json.loads(line) for line in payload["data"].splitlines() if line]
        assert {r["_id"] for r in rows} == {"D-20260613-201"}

    def test_get_block_returns_markdown_block(self, tmp_path: Path) -> None:
        ws = tmp_path / "ws"
        _write_workspace(ws)
        (ws / "decisions" / "DECISIONS.md").write_text(
            _md_block("D-20260613-202", "Markdown lookup works"),
            encoding="utf-8",
        )
        with use_workspace(str(ws)):
            payload = json.loads(get_block("D-20260613-202"))
        assert payload["found"] is True
        assert payload["block"]["_id"] == "D-20260613-202"

    def test_get_block_missing_returns_not_found(self, tmp_path: Path) -> None:
        ws = tmp_path / "ws"
        _write_workspace(ws)
        with use_workspace(str(ws)):
            payload = json.loads(get_block("D-20260613-999"))
        assert payload["found"] is False
        assert "not found in any corpus file" in payload["error"]

    def test_index_stats_counts_markdown_blocks(self, tmp_path: Path) -> None:
        ws = tmp_path / "ws"
        _write_workspace(ws)
        (ws / "decisions" / "DECISIONS.md").write_text(
            _md_block("D-20260613-203", "decision one") + _md_block("D-20260613-204", "decision two"),
            encoding="utf-8",
        )
        with use_workspace(str(ws)):
            stats = json.loads(index_stats())
        # No FTS index built → falls through to the per-corpus markdown count.
        assert stats["fts_index_exists"] is False
        assert stats.get("decisions_blocks") == 2

    def test_memory_health_counts_markdown_blocks(self, tmp_path: Path) -> None:
        ws = tmp_path / "ws"
        _write_workspace(ws)
        (ws / "decisions" / "DECISIONS.md").write_text(
            _md_block("D-20260613-205", "active decision") + _md_block("D-20260613-206", "old decision", status="superseded"),
            encoding="utf-8",
        )
        with use_workspace(str(ws)):
            health = json.loads(memory_health())
        assert health["total_blocks"] == 2
        assert health["total_active"] == 1
        assert health["corpus"]["decisions"] == {"total": 2, "active": 1}


# ─── Postgres backend (live DB; skips cleanly when unavailable) ───────────────

psycopg = pytest.importorskip("psycopg", reason="psycopg not installed; skipping Postgres tests")

from mind_mem.block_store_postgres import PostgresBlockStore  # noqa: E402

# Standard test DSN env var first; fall back to the local audit DSN. The
# schema is always a unique scratch schema we create and drop — the
# production ``mind_mem`` schema is never touched.
_DSN = os.environ.get("MIND_MEM_TEST_PG_DSN")


def _pg_available(dsn: str) -> bool:
    try:
        conn = psycopg.connect(dsn, connect_timeout=3)
        conn.close()
        return True
    except Exception:
        return False


_pg_live = pytest.mark.skipif(
    not _pg_available(_DSN),
    reason="no live Postgres available at the test DSN",
)


@pytest.fixture
def pg_workspace(tmp_path: Path) -> Generator[tuple[str, PostgresBlockStore], None, None]:
    """A workspace configured for Postgres on an isolated scratch schema."""
    schema = f"mm_fix_{uuid.uuid4().hex[:12]}"
    ws = tmp_path / "pgws"
    _write_workspace(ws, block_store={"backend": "postgres", "dsn": _DSN, "schema": schema})

    store = PostgresBlockStore(dsn=_DSN, schema=schema, workspace=str(ws))
    store._ensure_schema()
    try:
        yield str(ws), store
    finally:
        try:
            conn = psycopg.connect(_DSN)
            conn.autocommit = True
            conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
            conn.close()
        except Exception:
            pass
        store.close()


@_pg_live
class TestMemoryOpsPostgresLive:
    def test_export_memory_surfaces_pg_blocks(self, pg_workspace: tuple[str, PostgresBlockStore]) -> None:
        ws, store = pg_workspace
        store.write_block(
            {
                "_id": "D-20260613-301",
                "_source_file": "decisions/DECISIONS.md",
                "Statement": "the default backend is Postgres for this deployment",
                "Status": "active",
                "Date": "2026-06-13",
            }
        )
        store.write_block(
            {
                "_id": "D-20260613-302",
                "_source_file": "decisions/DECISIONS.md",
                "Statement": "Postgres connection pooling is enabled",
                "Status": "active",
                "Date": "2026-06-13",
            }
        )
        # The Markdown corpus on disk is empty — without backend-aware
        # enumeration this exports 0 blocks (the exact audit bug 5).
        with use_workspace(ws):
            payload = json.loads(export_memory())
        rows = [json.loads(line) for line in payload["data"].splitlines() if line]
        ids = {r["_id"] for r in rows}
        assert {"D-20260613-301", "D-20260613-302"} <= ids
        assert payload["block_count"] >= 2

    def test_export_memory_strips_internal_without_metadata(self, pg_workspace: tuple[str, PostgresBlockStore]) -> None:
        ws, store = pg_workspace
        store.write_block(
            {
                "_id": "D-20260613-303",
                "_source_file": "decisions/DECISIONS.md",
                "Statement": "metadata stripping check",
                "Status": "active",
                "Date": "2026-06-13",
            }
        )
        with use_workspace(ws):
            payload = json.loads(export_memory(include_metadata=False))
        rows = [json.loads(line) for line in payload["data"].splitlines() if line]
        row = next(r for r in rows if r["_id"] == "D-20260613-303")
        internal = [k for k in row if k.startswith("_") and k not in ("_id", "_source_file")]
        assert internal == []

    def test_get_block_resolves_pg_block(self, pg_workspace: tuple[str, PostgresBlockStore]) -> None:
        ws, store = pg_workspace
        store.write_block(
            {
                "_id": "D-20260613-304",
                "_source_file": "decisions/DECISIONS.md",
                "Statement": "fetch me from Postgres",
                "Status": "active",
                "Date": "2026-06-13",
            }
        )
        with use_workspace(ws):
            payload = json.loads(get_block("D-20260613-304"))
        assert payload["found"] is True
        assert payload["block"]["_id"] == "D-20260613-304"
        assert payload["block"]["Statement"] == "fetch me from Postgres"

    def test_get_block_missing_on_pg_returns_not_found(self, pg_workspace: tuple[str, PostgresBlockStore]) -> None:
        ws, _store = pg_workspace
        with use_workspace(ws):
            payload = json.loads(get_block("D-20260613-999"))
        assert payload["found"] is False
        assert "block store" in payload["error"]

    def test_index_stats_counts_pg_blocks(self, pg_workspace: tuple[str, PostgresBlockStore]) -> None:
        ws, store = pg_workspace
        for i in range(3):
            store.write_block(
                {
                    "_id": f"D-20260613-31{i}",
                    "_source_file": "decisions/DECISIONS.md",
                    "Statement": f"Postgres block {i}",
                    "Status": "active",
                    "Date": "2026-06-13",
                }
            )
        with use_workspace(ws):
            stats = json.loads(index_stats())
        # No FTS index on a fresh PG workspace → store-backed count.
        assert stats["fts_index_exists"] is False
        assert stats.get("total_blocks") == 3

    def test_memory_health_counts_pg_blocks(self, pg_workspace: tuple[str, PostgresBlockStore]) -> None:
        ws, store = pg_workspace
        store.write_block(
            {
                "_id": "D-20260613-320",
                "_source_file": "decisions/DECISIONS.md",
                "Statement": "active PG decision",
                "Status": "active",
                "Date": "2026-06-13",
            }
        )
        store.write_block(
            {
                "_id": "D-20260613-321",
                "_source_file": "decisions/DECISIONS.md",
                "Statement": "superseded PG decision",
                "Status": "superseded",
                "Date": "2026-06-13",
            }
        )
        with use_workspace(ws):
            health = json.loads(memory_health())
        assert health["total_blocks"] == 2
        assert health["total_active"] == 1
        # Blocks bucket under their source subdir parsed from _source_file.
        assert health["corpus"]["decisions"]["total"] == 2
        assert health["corpus"]["decisions"]["active"] == 1
