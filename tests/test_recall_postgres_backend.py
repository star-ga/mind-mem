"""Backend-aware recall dispatch — Postgres parity (audit bug 1).

The default recall pipeline reads the local Markdown corpus + the local
SQLite FTS index and ignored the configured ``block_store`` backend. On a
Postgres-backed workspace the corpus files are empty init templates and
every real block lives in the ``blocks`` table, so ``recall`` returned
nothing for PG-stored blocks.

``_recall_core._load_backend`` now returns a
:class:`~mind_mem._recall_core.PostgresRecallBackend` when
``block_store.backend == "postgres"`` (unless ``recall.backend`` is
explicitly ``sqlite`` / ``vector``), and ``recall`` dispatches to it —
delegating to ``PostgresBlockStore.hybrid_search``.

Coverage:

* **SQLite / Markdown backend** (the zero-config default, no DB) —
  ``_load_backend`` still returns ``None`` (built-in BM25 scan) and
  ``recall`` keeps returning Markdown-corpus blocks. This guards the
  default path against regression.
* **Postgres backend** (live DB, isolated scratch schema) —
  ``_load_backend`` returns a ``PostgresRecallBackend`` and ``recall``
  surfaces PG-resident blocks that the Markdown scan never sees. Each run
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

from mind_mem._recall_core import PostgresRecallBackend, _load_backend
from mind_mem.recall import recall

# ─── Shared workspace scaffolding ─────────────────────────────────────────────


def _write_workspace(ws: Path, *, block_store: dict | None = None, recall_cfg: dict | None = None) -> None:
    """Scaffold a minimal workspace with corpus directories + config."""
    for sub in ("decisions", "tasks", "entities", "intelligence"):
        (ws / sub).mkdir(parents=True, exist_ok=True)
    config: dict = {"recall": recall_cfg if recall_cfg is not None else {"backend": "scan"}}
    if block_store is not None:
        config["block_store"] = block_store
    (ws / "mind-mem.json").write_text(json.dumps(config), encoding="utf-8")


def _md_block(bid: str, statement: str, status: str = "active") -> str:
    return f"[{bid}]\nStatement: {statement}\nStatus: {status}\nDate: 2026-06-13\n\n---\n"


# ─── SQLite / Markdown default (no DB required) ───────────────────────────────


class TestMarkdownDefaultUnchanged:
    """The zero-config Markdown/SQLite path must not regress."""

    def test_load_backend_markdown_returns_none(self, tmp_path: Path) -> None:
        ws = tmp_path / "ws"
        _write_workspace(ws)
        # Default Markdown block store → built-in BM25 scan (None sentinel).
        assert _load_backend(str(ws)) is None

    def test_load_backend_no_config_returns_none(self, tmp_path: Path) -> None:
        ws = tmp_path / "ws"
        (ws / "decisions").mkdir(parents=True)
        # No mind-mem.json at all → default scan, never the PG backend.
        assert _load_backend(str(ws)) is None

    def test_recall_returns_markdown_blocks(self, tmp_path: Path) -> None:
        ws = tmp_path / "ws"
        _write_workspace(ws)
        (ws / "decisions" / "DECISIONS.md").write_text(
            _md_block("D-20260613-201", "the default backend is SQLite"),
            encoding="utf-8",
        )
        hits = recall(str(ws), "default backend SQLite", limit=5)
        ids = {h["_id"] for h in hits}
        assert "D-20260613-201" in ids

    def test_recall_empty_query_returns_empty(self, tmp_path: Path) -> None:
        ws = tmp_path / "ws"
        _write_workspace(ws)
        assert recall(str(ws), "   ", limit=5) == []


# ─── PostgresRecallBackend unit behaviour (no DB; pure dispatch) ──────────────


class TestPostgresRecallBackendUnit:
    """``_load_backend`` resolution + empty-query short-circuit (no live DB)."""

    def test_load_backend_postgres_returns_pg_backend(self, tmp_path: Path) -> None:
        ws = tmp_path / "ws"
        _write_workspace(ws, block_store={"backend": "postgres", "dsn": "postgresql://x/y"})
        backend = _load_backend(str(ws))
        assert isinstance(backend, PostgresRecallBackend)

    def test_explicit_sqlite_overrides_postgres_store(self, tmp_path: Path) -> None:
        """An explicit recall.backend=sqlite wins over a PG block store."""
        ws = tmp_path / "ws"
        _write_workspace(
            ws,
            block_store={"backend": "postgres", "dsn": "postgresql://x/y"},
            recall_cfg={"backend": "sqlite"},
        )
        assert _load_backend(str(ws)) == "sqlite"

    def test_malformed_recall_config_still_routes_to_postgres(self, tmp_path: Path) -> None:
        """A non-dict recall section must not downgrade a PG workspace."""
        ws = tmp_path / "ws"
        (ws / "decisions").mkdir(parents=True)
        (ws / "mind-mem.json").write_text(
            json.dumps({"block_store": {"backend": "postgres", "dsn": "postgresql://x/y"}, "recall": "oops"}),
            encoding="utf-8",
        )
        assert isinstance(_load_backend(str(ws)), PostgresRecallBackend)

    def test_empty_query_short_circuits_without_db(self, tmp_path: Path) -> None:
        """Empty query returns [] before any store connection is attempted."""
        ws = tmp_path / "ws"
        _write_workspace(ws, block_store={"backend": "postgres", "dsn": "postgresql://invalid"})
        backend = PostgresRecallBackend(str(ws))
        # No DSN connection happens — pure guard.
        assert backend.search(str(ws), "   ", limit=5) == []
        assert backend.search(str(ws), "", limit=5) == []

    def test_index_is_noop(self, tmp_path: Path) -> None:
        backend = PostgresRecallBackend(str(tmp_path))
        assert backend.index(str(tmp_path)) is None


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
class TestPostgresRecallLive:
    def test_recall_surfaces_pg_blocks(self, pg_workspace: tuple[str, PostgresBlockStore]) -> None:
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
        # dispatch this returns [] (the exact audit bug 1).
        hits = recall(ws, "Postgres backend", limit=5)
        ids = {h["_id"] for h in hits}
        assert "D-20260613-301" in ids
        # Every hit carries the canonical recall contract keys.
        for h in hits:
            assert {"_id", "type", "score", "excerpt", "file", "status"} <= set(h)

    def test_recall_empty_store_returns_empty(self, pg_workspace: tuple[str, PostgresBlockStore]) -> None:
        ws, _store = pg_workspace
        assert recall(ws, "nothing was ever written here", limit=5) == []

    def test_recall_respects_limit(self, pg_workspace: tuple[str, PostgresBlockStore]) -> None:
        ws, store = pg_workspace
        for i in range(5):
            store.write_block(
                {
                    "_id": f"D-20260613-31{i}",
                    "_source_file": "decisions/DECISIONS.md",
                    "Statement": f"Postgres backend tuning note number {i}",
                    "Status": "active",
                    "Date": "2026-06-13",
                }
            )
        hits = recall(ws, "Postgres backend tuning", limit=2)
        assert len(hits) <= 2

    def test_recall_date_post_filter_applies_on_pg(self, pg_workspace: tuple[str, PostgresBlockStore]) -> None:
        """The since/until post-filter contract must hold on the PG path."""
        ws, store = pg_workspace
        store.write_block(
            {
                "_id": "D-20260101-320",
                "_source_file": "decisions/DECISIONS.md",
                "Statement": "an old Postgres backend decision",
                "Status": "active",
                "Date": "2026-01-01",
            }
        )
        store.write_block(
            {
                "_id": "D-20260613-321",
                "_source_file": "decisions/DECISIONS.md",
                "Statement": "a recent Postgres backend decision",
                "Status": "active",
                "Date": "2026-06-13",
            }
        )
        hits = recall(ws, "Postgres backend decision", limit=10, since="2026-06-01")
        ids = {h["_id"] for h in hits}
        assert "D-20260613-321" in ids
        assert "D-20260101-320" not in ids
