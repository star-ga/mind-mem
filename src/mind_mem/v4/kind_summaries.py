"""v4 per-kind global summaries (Group B — GraphRAG-style).

Round 2 multi-LLM audit (3/4 model agreement 2026-05-10) recommended
adding per-kind global summaries so multi-agent systems get a "table
of contents" per knowledge domain without GraphRAG's full graph
construction.

Strategy:

    For each kind, maintain one summary row in ``kind_summaries``.
    When a caller invokes :func:`refresh_summary(kind)`, the planner
    pulls every block of that kind (via ``block_kind_tags`` if
    multi-label is on, else ``blocks.kind``) and produces a summary
    via the configured summariser:

        default     concatenation of the first N tokens of each
                    block's content, capped at ``max_summary_chars``
                    (deterministic, dependency-free)

        pluggable   set_summariser(fn) for production deployments
                    that want an LLM-driven summariser

The summary row carries an ``updated_at`` timestamp so callers can
gate refresh by staleness.

This module ships the planner; the caller decides when to refresh
(write-time hook, periodic batch, on-demand). Pure-function shape
matches :mod:`consolidation_worker`.

Feature-flag gated under ``v4.kind_summaries``.

Copyright STARGA, Inc.
"""

from __future__ import annotations

import datetime as _dt
import sqlite3
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from .feature_flags import flag_config, require_enabled

__all__ = [
    "FLAG",
    "Summariser",
    "KindSummary",
    "DEFAULT_MAX_CHARS",
    "set_summariser",
    "default_summariser",
    "ensure_kind_summary_schema",
    "refresh_summary",
    "get_summary",
    "list_summaries",
]


FLAG: str = "kind_summaries"

#: A summariser maps a list of block contents to one summary string.
Summariser = Callable[[Iterable[str]], str]

DEFAULT_MAX_CHARS: int = 4000


@dataclass(frozen=True)
class KindSummary:
    """Read-only summary record for one kind."""

    kind: str
    summary: str
    block_count: int
    updated_at: str


def default_summariser(blocks: Iterable[str]) -> str:
    """Concatenate truncated heads of each block.

    Deterministic and dependency-free. Each block contributes up to
    160 chars; the total is capped at ``DEFAULT_MAX_CHARS``. Useful
    as a "table of contents" stand-in when no LLM summariser is
    available.
    """
    pieces: list[str] = []
    used = 0
    for content in blocks:
        if not content:
            continue
        head = content.strip().splitlines()[0] if content.strip() else ""
        if len(head) > 160:
            head = head[:157].rstrip() + "..."
        if used + len(head) + 2 > DEFAULT_MAX_CHARS:
            break
        if head:
            pieces.append(head)
            used += len(head) + 2
    return "\n".join(pieces)


_active_summariser: Summariser = default_summariser


def set_summariser(fn: Summariser) -> None:
    """Swap the active summariser (e.g. install an LLM-driven one)."""
    require_enabled(FLAG)
    global _active_summariser
    _active_summariser = fn


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA_SQL: str = """
CREATE TABLE IF NOT EXISTS kind_summaries (
    kind         TEXT PRIMARY KEY,
    summary      TEXT NOT NULL,
    block_count  INTEGER NOT NULL DEFAULT 0,
    updated_at   TEXT NOT NULL
);
"""


def ensure_kind_summary_schema(workspace: str | Path) -> None:
    """Idempotent. Creates the ``kind_summaries`` table."""
    require_enabled(FLAG)
    db = Path(workspace) / "index.db"
    if not db.parent.is_dir():
        db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db) as conn:
        conn.executescript(_SCHEMA_SQL)
        conn.commit()


# ---------------------------------------------------------------------------
# Refresh + read
# ---------------------------------------------------------------------------


def refresh_summary(workspace: str | Path, kind: str) -> KindSummary | None:
    """Rebuild the summary for ``kind`` from current block content.

    Reads from ``blocks(id, content, kind)`` directly (single-label
    path); multi-label callers can pre-aggregate via
    ``block_kind_tags`` and pass the resulting block_ids in via
    :func:`set_summariser` if they want fully-typed inputs.

    Returns the new :class:`KindSummary` or ``None`` if no blocks of
    that kind exist.
    """
    require_enabled(FLAG)
    ensure_kind_summary_schema(workspace)
    db = Path(workspace) / "index.db"
    if not db.is_file():
        return None
    with sqlite3.connect(db) as conn:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(blocks)")}
        if "kind" not in cols:
            return None
        rows = conn.execute(
            "SELECT content FROM blocks WHERE kind = ?",
            (kind,),
        ).fetchall()
    blocks = [r[0] or "" for r in rows]
    if not blocks:
        return None
    summary = _active_summariser(blocks)
    now = _dt.datetime.now(_dt.timezone.utc).isoformat()
    with sqlite3.connect(db) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO kind_summaries (kind, summary, block_count, updated_at) VALUES (?, ?, ?, ?)",
            (kind, summary, len(blocks), now),
        )
        conn.commit()
    return KindSummary(kind=kind, summary=summary, block_count=len(blocks), updated_at=now)


def get_summary(workspace: str | Path, kind: str) -> KindSummary | None:
    """Return the stored summary for ``kind``, or ``None`` if absent."""
    require_enabled(FLAG)
    db = Path(workspace) / "index.db"
    if not db.is_file():
        return None
    with sqlite3.connect(db) as conn:
        if not _table_exists(conn, "kind_summaries"):
            return None
        row = conn.execute(
            "SELECT kind, summary, block_count, updated_at FROM kind_summaries WHERE kind = ?",
            (kind,),
        ).fetchone()
    if row is None:
        return None
    return KindSummary(
        kind=row[0],
        summary=row[1],
        block_count=int(row[2]),
        updated_at=row[3],
    )


def list_summaries(workspace: str | Path) -> list[KindSummary]:
    """Return every stored summary, ordered by kind."""
    require_enabled(FLAG)
    db = Path(workspace) / "index.db"
    if not db.is_file():
        return []
    with sqlite3.connect(db) as conn:
        if not _table_exists(conn, "kind_summaries"):
            return []
        rows = conn.execute("SELECT kind, summary, block_count, updated_at FROM kind_summaries ORDER BY kind").fetchall()
    return [KindSummary(kind=r[0], summary=r[1], block_count=int(r[2]), updated_at=r[3]) for r in rows]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?",
        (name,),
    ).fetchone()
    return row is not None


def _max_chars() -> int:
    raw = flag_config(FLAG)
    if not isinstance(raw, dict):
        return DEFAULT_MAX_CHARS
    v = raw.get("max_chars", DEFAULT_MAX_CHARS)
    try:
        out = int(v)
    except (TypeError, ValueError):
        return DEFAULT_MAX_CHARS
    return max(64, out)
