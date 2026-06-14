"""Backend-aware governance ``scan`` — audit bugs #3 / #10.

The MCP ``scan`` tool (contradiction / drift / decision integrity scan)
historically read only the local Markdown corpus via ``parse_file`` over
``decisions/DECISIONS.md`` + ``intelligence/*.md``. On a Postgres (or any
non-Markdown) backend those files are the empty init templates, so the
scan silently reported ``decisions.total=0`` / ``contradictions.raw=0``
even when the store held contradictory blocks.

The fix routes ``scan``'s block enumeration through
:func:`mind_mem.storage.iter_active_blocks`, so governance sees the
configured backend's blocks.

Coverage:

* **Markdown backend** (the zero-config SQLite default) — keeps the exact
  legacy file-based behaviour; the decision count still includes archived
  blocks and the contradiction surface still comes from
  ``CONTRADICTIONS.md`` + ``conflict_resolver``. Runs with no DB.
* **Postgres backend** — enumerates store-resident blocks; the decision
  count and statement-level contradiction count both reflect the DB.
  Each run uses a uniquely-named scratch schema it creates and drops; the
  whole Postgres section skips gracefully when psycopg / a live Postgres
  is unavailable, so SQLite-only CI stays green.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Generator

import pytest

from mind_mem.mcp.infra.workspace import use_workspace
from mind_mem.mcp.tools import governance
from mind_mem.mcp.tools.governance import _detect_statement_contradictions, scan

# ─── Pure helper: statement-level contradiction detection ─────────────────────


def test_detect_value_conflict() -> None:
    """Two blocks assigning different values to the same subject conflict."""
    found = _detect_statement_contradictions(
        [
            {"_id": "D-1", "Statement": "default backend is SQLite"},
            {"_id": "D-2", "Statement": "default backend is Postgres"},
        ]
    )
    assert len(found) == 1
    assert {found[0]["block_a"], found[0]["block_b"]} == {"D-1", "D-2"}


def test_detect_affirmation_vs_negation() -> None:
    """A subject asserted affirmatively on one side, negated on the other."""
    found = _detect_statement_contradictions(
        [
            {"_id": "D-201", "Statement": "use Postgres as the only backend"},
            {"_id": "D-202", "Statement": "will not use Postgres; SQLite is the only backend"},
        ]
    )
    assert len(found) == 1


def test_detect_antonym_conflict() -> None:
    found = _detect_statement_contradictions(
        [
            {"_id": "D-3", "Statement": "enable cross-encoder reranking in recall"},
            {"_id": "D-4", "Statement": "disable cross-encoder reranking in recall"},
        ]
    )
    assert len(found) == 1
    assert "enable" in found[0]["reason"]


def test_agreeing_blocks_are_not_contradictions() -> None:
    """Same value, same subject → no conflict (no false positives)."""
    assert (
        _detect_statement_contradictions(
            [
                {"_id": "D-7", "Statement": "default backend is Postgres"},
                {"_id": "D-8", "Statement": "default backend is Postgres for production"},
            ]
        )
        == []
    )


def test_unrelated_blocks_are_not_contradictions() -> None:
    """No topical overlap → never flagged, regardless of negation words."""
    assert (
        _detect_statement_contradictions(
            [
                {"_id": "D-5", "Statement": "use Postgres as the only backend"},
                {"_id": "D-6", "Statement": "schedule the weekly report every Monday"},
            ]
        )
        == []
    )


def test_blocks_without_id_or_text_are_skipped() -> None:
    assert _detect_statement_contradictions([{"Statement": "no id"}, {"_id": "D-9"}]) == []


# ─── Markdown backend (no DB required) ────────────────────────────────────────


def _markdown_workspace(ws: Path) -> None:
    for sub in ("decisions", "tasks", "entities", "intelligence"):
        (ws / sub).mkdir(parents=True, exist_ok=True)
    (ws / "mind-mem.json").write_text(json.dumps({"recall": {"backend": "bm25"}}), encoding="utf-8")


def test_scan_markdown_counts_decisions(tmp_path: Path) -> None:
    """Default path is byte-for-byte legacy: total includes archived."""
    ws = tmp_path / "md"
    _markdown_workspace(ws)
    (ws / "decisions" / "DECISIONS.md").write_text(
        "[D-20260613-001]\nStatement: default backend is SQLite\nStatus: active\nDate: 2026-06-13\n\n---\n"
        "[D-20260613-002]\nStatement: an archived call\nStatus: archived\nDate: 2026-06-13\n\n---\n",
        encoding="utf-8",
    )
    with use_workspace(str(ws)):
        result = json.loads(scan())

    assert result["backend"] == "markdown"
    # Legacy semantics: total counts every block, active only the live one.
    assert result["checks"]["decisions"] == {"total": 2, "active": 1}
    # Markdown contradiction surface stays file/conflict_resolver-driven.
    assert result["checks"]["contradictions"] == {"raw": 0, "resolvable": 0}


def test_scan_markdown_empty_workspace(tmp_path: Path) -> None:
    ws = tmp_path / "md_empty"
    _markdown_workspace(ws)
    with use_workspace(str(ws)):
        result = json.loads(scan())
    assert result["checks"]["decisions"] == {"total": 0, "active": 0}
    assert result["checks"]["contradictions"]["raw"] == 0


# ─── Postgres backend (live DB; skips cleanly when unavailable) ───────────────
#
# Crucially the Markdown / pure-helper tests above carry NO module-level
# skip marker, so they always run in SQLite-only CI (the default path's
# protection must execute, not skip). Only the Postgres tests below are
# guarded, and they are guarded individually so an absent psycopg / DB
# never skips the no-DB tests.

try:
    import psycopg as _psycopg  # type: ignore[import-not-found]

    from mind_mem.block_store_postgres import PostgresBlockStore

    _HAVE_PSYCOPG = True
except Exception:  # pragma: no cover - psycopg not installed
    _psycopg = None  # type: ignore[assignment]
    PostgresBlockStore = None  # type: ignore[assignment,misc]
    _HAVE_PSYCOPG = False

_DSN = os.environ.get("MIND_MEM_TEST_PG_DSN")


def _pg_available(dsn: str) -> bool:
    if not _HAVE_PSYCOPG:
        return False
    try:
        conn = _psycopg.connect(dsn, connect_timeout=3)
        conn.close()
        return True
    except Exception:
        return False


# Per-test guard (NOT a module-level pytestmark) so the no-DB tests above
# always execute.
requires_pg = pytest.mark.skipif(
    not _pg_available(_DSN),
    reason="no live Postgres available at the test DSN",
)


@pytest.fixture
def pg_workspace(tmp_path: Path) -> Generator[tuple[str, PostgresBlockStore], None, None]:
    """A workspace configured for Postgres on an isolated scratch schema."""
    schema = f"mm_fix_{uuid.uuid4().hex[:12]}"
    ws = tmp_path / "pgws"
    for sub in ("decisions", "tasks", "entities", "intelligence"):
        (ws / sub).mkdir(parents=True, exist_ok=True)
    (ws / "mind-mem.json").write_text(
        json.dumps({"block_store": {"backend": "postgres", "dsn": _DSN, "schema": schema}}),
        encoding="utf-8",
    )

    store = PostgresBlockStore(dsn=_DSN, schema=schema, workspace=str(ws))
    store._ensure_schema()
    try:
        yield str(ws), store
    finally:
        try:
            conn = _psycopg.connect(_DSN)
            conn.autocommit = True
            conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
            conn.close()
        except Exception:
            pass


def _decision(store: PostgresBlockStore, bid: str, statement: str, status: str = "active") -> None:
    store.write_block(
        {
            "_id": bid,
            "_source_file": "decisions/DECISIONS.md",
            "Statement": statement,
            "Status": status,
            "Type": "decision",
            "Date": "2026-06-13",
        }
    )


@requires_pg
def test_scan_postgres_sees_store_decisions(pg_workspace: tuple[str, PostgresBlockStore]) -> None:
    ws, store = pg_workspace
    _decision(store, "D-20260613-301", "default backend is Postgres")
    _decision(store, "D-20260613-302", "enable cross-encoder reranking")

    with use_workspace(ws):
        result = json.loads(scan())

    assert result["backend"] == "postgres"
    # Was {total:0, active:0} before the fix — the bug.
    assert result["checks"]["decisions"]["total"] == 2
    assert result["checks"]["decisions"]["active"] == 2


@requires_pg
def test_scan_postgres_detects_contradiction(pg_workspace: tuple[str, PostgresBlockStore]) -> None:
    """The audit's seeded contradictory pair is now surfaced on Postgres."""
    ws, store = pg_workspace
    _decision(store, "D-20260613-201", "use Postgres as the only backend")
    _decision(store, "D-20260613-202", "will not use Postgres; SQLite is the only backend")

    with use_workspace(ws):
        result = json.loads(scan())

    # Was contradictions.raw == 0 before the fix (silent no-op on PG).
    assert result["checks"]["contradictions"]["raw"] >= 1


@requires_pg
def test_scan_postgres_excludes_deleted_blocks(pg_workspace: tuple[str, PostgresBlockStore]) -> None:
    """A soft-deleted (inactive) row is not counted — active_only contract."""
    ws, store = pg_workspace
    _decision(store, "D-20260613-401", "default backend is Postgres")
    _decision(store, "D-20260613-402", "soon-to-be-deleted block")
    assert store.delete_block("D-20260613-402") is True

    with use_workspace(ws):
        result = json.loads(scan())

    assert result["checks"]["decisions"]["total"] == 1


@requires_pg
def test_scan_postgres_clean_workspace_reports_zero(pg_workspace: tuple[str, PostgresBlockStore]) -> None:
    """Agreeing / non-contradictory blocks produce no false contradictions."""
    ws, store = pg_workspace
    _decision(store, "D-20260613-501", "default backend is Postgres")
    _decision(store, "D-20260613-502", "schedule the weekly report every Monday")

    with use_workspace(ws):
        result = json.loads(scan())

    assert result["checks"]["decisions"]["total"] == 2
    assert result["checks"]["contradictions"]["raw"] == 0


def test_scan_resolve_backend_helper_defaults_markdown(tmp_path: Path) -> None:
    """No config / unreadable config degrades to markdown (never raises)."""
    assert governance._resolve_backend(str(tmp_path)) == "markdown"
