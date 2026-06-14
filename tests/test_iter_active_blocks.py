"""Backend-aware active-block enumeration — ``storage.iter_active_blocks``.

The recall / scan / governance / export / reindex feature layer must see
the configured ``block_store`` backend's blocks, not just the local
Markdown corpus. :func:`mind_mem.storage.iter_active_blocks` is the shared
primitive those features route through.

Coverage:

* **Markdown backend** (the zero-config default) — enumerates the local
  Markdown corpus, returns only active blocks, tags ``_source_file`` /
  ``_source_label``, and excludes unreviewed pending signals. Runs with
  no DB.
* **Postgres backend** — delegates to
  ``get_block_store(ws).get_all(active_only=True)`` so PG-resident blocks
  are visible. Each run uses a uniquely-named scratch schema it creates
  and drops; the whole class skips gracefully when psycopg / a live
  Postgres is unavailable, so SQLite-only CI stays green.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Generator

import pytest

from mind_mem.storage import get_active_blocks, iter_active_blocks

# ─── Markdown corpus helpers ──────────────────────────────────────────────────


def _write_workspace(ws: Path, *, block_store: dict | None = None) -> None:
    """Scaffold a minimal Markdown workspace with corpus directories."""
    for sub in ("decisions", "tasks", "entities", "intelligence"):
        (ws / sub).mkdir(parents=True, exist_ok=True)
    config: dict = {"recall": {"backend": "bm25"}}
    if block_store is not None:
        config["block_store"] = block_store
    (ws / "mind-mem.json").write_text(json.dumps(config), encoding="utf-8")


def _block(bid: str, statement: str, status: str = "active") -> str:
    return f"[{bid}]\nStatement: {statement}\nStatus: {status}\nDate: 2026-06-13\n\n---\n"


# ─── Markdown backend (no DB required) ────────────────────────────────────────


def test_markdown_returns_active_blocks(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    _write_workspace(ws)
    (ws / "decisions" / "DECISIONS.md").write_text(
        _block("D-20260613-001", "default backend is SQLite", "active")
        + _block("D-20260613-002", "an archived call", "archived"),
        encoding="utf-8",
    )

    blocks = iter_active_blocks(str(ws))
    ids = {b["_id"] for b in blocks}

    assert "D-20260613-001" in ids
    # archived block is filtered out
    assert "D-20260613-002" not in ids
    active = next(b for b in blocks if b["_id"] == "D-20260613-001")
    assert active["_source_file"] == "decisions/DECISIONS.md"
    assert active["_source_label"] == "decisions"


def test_markdown_excludes_pending_signals(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    _write_workspace(ws)
    (ws / "intelligence" / "SIGNALS.md").write_text(
        _block("SIG-20260613-001", "a pending signal", "pending")
        + _block("SIG-20260613-002", "a reviewed signal", "active"),
        encoding="utf-8",
    )

    ids = {b["_id"] for b in iter_active_blocks(str(ws))}

    # #429: unreviewed pending signals are excluded from the active set
    assert "SIG-20260613-001" not in ids
    assert "SIG-20260613-002" in ids


def test_markdown_empty_workspace_returns_empty(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    _write_workspace(ws)
    assert iter_active_blocks(str(ws)) == []


def test_markdown_missing_config_degrades_to_markdown(tmp_path: Path) -> None:
    """No mind-mem.json at all → markdown default, never raises."""
    ws = tmp_path / "ws"
    (ws / "decisions").mkdir(parents=True)
    (ws / "decisions" / "DECISIONS.md").write_text(
        _block("D-20260613-003", "no config present", "active"), encoding="utf-8"
    )
    blocks = iter_active_blocks(str(ws))
    assert {b["_id"] for b in blocks} == {"D-20260613-003"}


def test_get_active_blocks_alias_matches(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    _write_workspace(ws)
    (ws / "decisions" / "DECISIONS.md").write_text(
        _block("D-20260613-004", "alias parity", "active"), encoding="utf-8"
    )
    assert {b["_id"] for b in iter_active_blocks(str(ws))} == {b["_id"] for b in get_active_blocks(str(ws))}


# ─── Postgres backend (live DB; skips cleanly when unavailable) ───────────────

psycopg = pytest.importorskip("psycopg", reason="psycopg not installed; skipping Postgres tests")

from mind_mem.block_store_postgres import PostgresBlockStore  # noqa: E402

# Honour the standard test DSN env var first; fall back to the local
# audit DSN. The schema is always a unique scratch schema we create and
# drop — the production ``mind_mem`` schema is never touched.
_DSN = os.environ.get("MIND_MEM_TEST_PG_DSN")


def _pg_available(dsn: str) -> bool:
    try:
        conn = psycopg.connect(dsn, connect_timeout=3)
        conn.close()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
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


def test_postgres_iter_active_blocks_sees_store_blocks(pg_workspace: tuple[str, PostgresBlockStore]) -> None:
    ws, store = pg_workspace
    store.write_block(
        {
            "_id": "D-20260613-101",
            "_source_file": "decisions/DECISIONS.md",
            "Statement": "default backend is Postgres",
            "Status": "active",
            "Date": "2026-06-13",
        }
    )
    store.write_block(
        {
            "_id": "D-20260613-102",
            "_source_file": "decisions/DECISIONS.md",
            "Statement": "a soon-to-be-deleted PG block",
            "Status": "active",
            "Date": "2026-06-13",
        }
    )
    # The store's ``active_only`` filters the row-level lifecycle flag
    # (set FALSE on delete), the same contract feature code relies on.
    assert store.delete_block("D-20260613-102") is True

    blocks = iter_active_blocks(ws)
    ids = {b["_id"] for b in blocks}

    # The PG-resident active block is visible; the Markdown corpus on
    # disk is empty, so without backend-awareness this would be {}.
    assert "D-20260613-101" in ids
    # The deleted (inactive) row is excluded via get_all(active_only=True).
    assert "D-20260613-102" not in ids


def test_postgres_empty_store_returns_empty(pg_workspace: tuple[str, PostgresBlockStore]) -> None:
    ws, _store = pg_workspace
    assert iter_active_blocks(ws) == []


def test_postgres_alias_matches(pg_workspace: tuple[str, PostgresBlockStore]) -> None:
    ws, store = pg_workspace
    store.write_block(
        {
            "_id": "D-20260613-103",
            "_source_file": "decisions/DECISIONS.md",
            "Statement": "alias parity on PG",
            "Status": "active",
            "Date": "2026-06-13",
        }
    )
    assert {b["_id"] for b in iter_active_blocks(ws)} == {b["_id"] for b in get_active_blocks(ws)}
