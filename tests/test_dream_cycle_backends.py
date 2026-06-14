"""Backend-aware dream-cycle maintenance passes (audit bug 11).

The dream-cycle enrichment passes (entity discovery, citation repair, stale
detection) historically scanned the local Markdown corpus directly, so on a
non-Markdown backend (e.g. Postgres) every pass returned 0 — the deliberate
dangling citation in the audit (``D-999``) was never detected. The passes now
route their block enumeration through
:func:`mind_mem.storage.iter_active_blocks`, so the configured store's blocks
of record are visible.

Coverage:

* **Markdown backend** (the zero-config default) — the file-based scan is
  unchanged; these assertions document parity with the legacy behaviour and
  run with no DB.
* **Postgres backend** — blocks live in the store; each test uses a uniquely
  named scratch schema it creates and drops (the production ``mind_mem``
  schema is never touched). The whole Postgres class skips gracefully when
  psycopg / a live Postgres is unavailable, so SQLite-only CI stays green.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Generator

import pytest

from mind_mem.dream_cycle import (
    pass_citation_repair,
    pass_entity_discovery,
    pass_stale_detection,
    run_dream_cycle,
)

# ─── Markdown helpers ─────────────────────────────────────────────────────────


def _write_markdown_workspace(ws: Path) -> None:
    for sub in ("memory", "entities", "decisions", "tasks", "intelligence"):
        (ws / sub).mkdir(parents=True, exist_ok=True)
    (ws / "mind-mem.json").write_text(json.dumps({"recall": {"backend": "bm25"}}), encoding="utf-8")


def _old_id_date() -> str:
    """A compact YYYYMMDD ~400 days ago (comfortably past the 30-day cutoff)."""
    return (datetime.now() - timedelta(days=400)).strftime("%Y%m%d")


# ─── Markdown backend (no DB required) ────────────────────────────────────────


def test_markdown_citation_repair_detects_dangling(tmp_path: Path) -> None:
    """Parity: the Markdown path still detects a dangling reference."""
    ws = tmp_path / "ws"
    _write_markdown_workspace(ws)
    (ws / "decisions" / "DECISIONS.md").write_text(
        "[D-20260613-001]\nStatement: References nonexistent D-20269999-999 deliberately\nStatus: active\n",
        encoding="utf-8",
    )
    broken = pass_citation_repair(str(ws))
    assert "D-20269999-999" in {b.cited_id for b in broken}


def test_markdown_entity_discovery_scans_logs(tmp_path: Path) -> None:
    """Parity: the Markdown path still scans daily logs for entities."""
    ws = tmp_path / "ws"
    _write_markdown_workspace(ws)
    today = datetime.now().strftime("%Y-%m-%d")
    (ws / "memory" / f"{today}.md").write_text("See https://github.com/star-ga/freshrepo for details.\n", encoding="utf-8")
    proposals = pass_entity_discovery(str(ws))
    assert "freshrepo" in {p.slug for p in proposals}


# ─── Postgres backend (live DB; skips cleanly when unavailable) ───────────────

psycopg = pytest.importorskip("psycopg", reason="psycopg not installed; skipping Postgres tests")

from mind_mem.block_store_postgres import PostgresBlockStore  # noqa: E402

_DSN = os.environ.get("MIND_MEM_TEST_PG_DSN")


def _pg_available(dsn: str) -> bool:
    try:
        conn = psycopg.connect(dsn, connect_timeout=3)
        conn.close()
        return True
    except Exception:
        return False


# Gate only the Postgres class — the Markdown parity tests above need no DB
# and must keep running on SQLite-only CI.
_pg_skip = pytest.mark.skipif(
    not _pg_available(_DSN),
    reason="no live Postgres available at the test DSN",
)


def _write_pg_workspace(ws: Path, schema: str) -> None:
    # The local corpus dirs are intentionally empty init templates — the
    # blocks of record live in Postgres. This is exactly the state in which
    # the markdown-only passes silently returned 0 (the bug).
    for sub in ("memory", "entities", "decisions", "tasks", "intelligence"):
        (ws / sub).mkdir(parents=True, exist_ok=True)
    (ws / "mind-mem.json").write_text(
        json.dumps({"block_store": {"backend": "postgres", "dsn": _DSN, "schema": schema}}),
        encoding="utf-8",
    )


@pytest.fixture
def pg_workspace(tmp_path: Path) -> Generator[tuple[str, PostgresBlockStore], None, None]:
    """A workspace configured for Postgres on an isolated scratch schema."""
    schema = f"mm_fix_{uuid.uuid4().hex[:12]}"
    ws = tmp_path / "pgws"
    _write_pg_workspace(ws, schema)

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


@_pg_skip
def test_pg_citation_repair_detects_dangling(pg_workspace: tuple[str, PostgresBlockStore]) -> None:
    """The deliberate dangling citation (audit bug 11) is detected on Postgres."""
    ws, store = pg_workspace
    store.write_block(
        {
            "_id": "D-20260613-103",
            "_source_file": "decisions/DECISIONS.md",
            "Statement": "References nonexistent D-20269999-999 deliberately",
            "Status": "active",
            "Date": "2026-06-13",
        }
    )
    store.write_block(
        {
            "_id": "D-20260613-104",
            "_source_file": "decisions/DECISIONS.md",
            "Statement": "A real decision with no dangling refs",
            "Status": "active",
            "Date": "2026-06-13",
        }
    )

    broken = pass_citation_repair(ws)
    cited = {b.cited_id for b in broken}
    # The dangling ref is found, and the two valid block IDs are not flagged.
    assert "D-20269999-999" in cited
    assert "D-20260613-103" not in cited
    assert "D-20260613-104" not in cited


@_pg_skip
def test_pg_citation_repair_valid_refs_not_flagged(pg_workspace: tuple[str, PostgresBlockStore]) -> None:
    """A reference to a block that *is* defined in the store is not broken."""
    ws, store = pg_workspace
    store.write_block({"_id": "D-20260613-201", "Statement": "Base decision", "Status": "active", "Date": "2026-06-13"})
    store.write_block(
        {
            "_id": "D-20260613-202",
            "Statement": "Builds on D-20260613-201 which exists",
            "Status": "active",
            "Date": "2026-06-13",
        }
    )
    assert pass_citation_repair(ws) == []


@_pg_skip
def test_pg_entity_discovery_scans_store_blocks(pg_workspace: tuple[str, PostgresBlockStore]) -> None:
    """Entity discovery surfaces untracked entities from store block text."""
    ws, store = pg_workspace
    store.write_block(
        {
            "_id": "D-20260613-301",
            "Statement": "Adopted https://github.com/star-ga/newproject for the build",
            "Status": "active",
            "Date": "2026-06-13",
        }
    )
    proposals = pass_entity_discovery(ws)
    slugs = {p.slug for p in proposals}
    assert "newproject" in slugs


@_pg_skip
def test_pg_entity_discovery_skips_tracked(pg_workspace: tuple[str, PostgresBlockStore]) -> None:
    """An entity already tracked as a PRJ-/TOOL-/PER- block is not re-proposed."""
    ws, store = pg_workspace
    store.write_block({"_id": "PRJ-newproject", "Statement": "tracked project record", "Status": "active", "Date": "2026-06-13"})
    store.write_block(
        {
            "_id": "D-20260613-302",
            "Statement": "Still using https://github.com/star-ga/newproject daily",
            "Status": "active",
            "Date": "2026-06-13",
        }
    )
    proposals = pass_entity_discovery(ws)
    assert "newproject" not in {p.slug for p in proposals}


@_pg_skip
def test_pg_stale_detection_flags_old_block(pg_workspace: tuple[str, PostgresBlockStore]) -> None:
    """A block whose embedded date is well past the cutoff is flagged stale."""
    ws, store = pg_workspace
    old_id = f"D-{_old_id_date()}-001"
    store.write_block({"_id": old_id, "Statement": "An ancient decision", "Status": "active"})
    today_id = "D-" + datetime.now().strftime("%Y%m%d") + "-002"
    store.write_block({"_id": today_id, "Statement": "A fresh decision", "Status": "active"})
    stale = pass_stale_detection(ws, stale_days=30)
    ids = {s.block_id for s in stale}
    assert old_id in ids
    assert today_id not in ids


@_pg_skip
def test_pg_passes_ignore_inactive_blocks(pg_workspace: tuple[str, PostgresBlockStore]) -> None:
    """Deleted (inactive) blocks are invisible to every pass (active_only)."""
    ws, store = pg_workspace
    store.write_block(
        {
            "_id": "D-20260613-401",
            "Statement": "Soon-deleted, cites D-20269999-998",
            "Status": "active",
            "Date": "2026-06-13",
        }
    )
    assert store.delete_block("D-20260613-401") is True
    # The dangling ref inside the deleted block must not surface.
    assert all(b.cited_id != "D-20269999-998" for b in pass_citation_repair(ws))


@_pg_skip
def test_pg_full_cycle_dry_run_finds_dangling(pg_workspace: tuple[str, PostgresBlockStore]) -> None:
    """End-to-end run_dream_cycle on Postgres detects the dangling citation."""
    ws, store = pg_workspace
    store.write_block(
        {
            "_id": "D-20260613-501",
            "Statement": "References nonexistent D-20269999-997 deliberately",
            "Status": "active",
            "Date": "2026-06-13",
        }
    )
    report = run_dream_cycle(ws, dry_run=True)
    assert "D-20269999-997" in {b.cited_id for b in report.broken_citations}
    assert report.total_findings > 0
