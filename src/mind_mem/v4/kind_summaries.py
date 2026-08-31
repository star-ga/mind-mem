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
                    block's content, capped at the ``max_chars`` key of
                    the ``v4.kind_summaries`` flag config (default
                    :data:`DEFAULT_MAX_CHARS`; deterministic,
                    dependency-free)

        pluggable   set_summariser(fn) for production deployments
                    that want an LLM-driven summariser. A pluggable
                    summariser receives only the block contents and so
                    owns its own output bound — ``max_chars`` governs
                    :func:`default_summariser`, not the installed one.

The summary row carries an ``updated_at`` timestamp so callers can
gate refresh by staleness.

This module ships the planner; the caller decides when to refresh
(write-time hook, periodic batch, on-demand). The read side
(:func:`get_summary`, :func:`list_summaries`) is read-only, but
:func:`refresh_summary` **writes**: it creates the workspace directory
and the ``kind_summaries`` table if absent and then replaces one row.
Point it only at a workspace you may write to — against a snapshot or a
read-only index it will either mutate it or raise
``sqlite3.OperationalError``.

Feature-flag gated under ``v4.kind_summaries``.

Copyright STARGA, Inc.
"""

from __future__ import annotations

import datetime as _dt
import sqlite3
from collections.abc import Callable, Iterable
from contextlib import closing
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

#: Floor for a configured ``max_chars`` — a cap below this produces
#: summaries too short to be a table of contents at all.
_MIN_MAX_CHARS: int = 64


@dataclass(frozen=True)
class KindSummary:
    """Read-only summary record for one kind."""

    kind: str
    summary: str
    block_count: int
    updated_at: str


def default_summariser(blocks: Iterable[str], max_chars: int | None = None) -> str:
    """Concatenate truncated heads of each block.

    Deterministic and dependency-free. Each block contributes up to
    160 chars; the total is capped at ``max_chars``, which defaults to
    the ``max_chars`` key of the ``v4.kind_summaries`` flag config and
    falls back to :data:`DEFAULT_MAX_CHARS` when unset. Useful as a
    "table of contents" stand-in when no LLM summariser is available.
    """
    cap = _max_chars() if max_chars is None else max(_MIN_MAX_CHARS, int(max_chars))
    pieces: list[str] = []
    used = 0
    for content in blocks:
        if not content:
            continue
        head = content.strip().splitlines()[0] if content.strip() else ""
        if len(head) > 160:
            head = head[:157].rstrip() + "..."
        if used + len(head) + 2 > cap:
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


# Every connection below is opened as ``closing(sqlite3.connect(...)) as conn, conn``.
# Both context managers are load-bearing and the order is not interchangeable:
#
#   * the inner ``conn`` commits on success / rolls back on an exception — that
#     is the *only* thing a bare ``with sqlite3.connect(...) as conn`` does. Its
#     ``__exit__`` never closes the handle;
#   * :func:`contextlib.closing` then closes it. ``close()`` on its own never
#     commits, so it must run *after* the transaction context exits, which is
#     exactly what ``with A, B`` guarantees (B exits first).
#
# Without the close the handle survives the call. Refcounting cannot reclaim it:
# a ``sqlite3.Connection`` owns a prepared-statement cache that refers back to
# the connection, so every one of these sits in a reference cycle and is freed
# only if and when the cyclic collector happens to run. Until then the process
# holds a descriptor on ``index.db`` and on its ``-wal`` / ``-shm`` sidecars —
# an unbounded descriptor leak under a long-lived server, sidecars that never
# get checkpointed away, and on Windows an open handle that makes ``unlink`` /
# ``rmdir`` of the workspace fail outright.
#
# These functions are module-level with no object to hang a ``_session()``
# helper on (cf. :meth:`mind_mem.hash_chain_v2.HashChainV2._session`, the same
# fix in class form), so the close is applied at each call site.


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
    with closing(sqlite3.connect(db, timeout=30)) as conn, conn:
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

    **Writes to the workspace.** Creates the workspace directory and the
    ``kind_summaries`` table when absent, then replaces one row with
    ``INSERT OR REPLACE`` and commits. Not safe to point at a workspace
    you only mean to read.
    """
    require_enabled(FLAG)
    ensure_kind_summary_schema(workspace)
    db = Path(workspace) / "index.db"
    if not db.is_file():
        return None
    with closing(sqlite3.connect(db, timeout=30)) as conn, conn:
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
    with closing(sqlite3.connect(db, timeout=30)) as conn, conn:
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
    with closing(sqlite3.connect(db, timeout=30)) as conn, conn:
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
    with closing(sqlite3.connect(db, timeout=30)) as conn, conn:
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
    """Configured summary cap, or :data:`DEFAULT_MAX_CHARS`.

    Reads ``v4.kind_summaries.max_chars``. A non-numeric or absent value
    falls back to the default; anything below :data:`_MIN_MAX_CHARS` is
    raised to it.
    """
    raw = flag_config(FLAG)
    if not isinstance(raw, dict):
        return DEFAULT_MAX_CHARS
    v = raw.get("max_chars", DEFAULT_MAX_CHARS)
    if isinstance(v, bool):  # bool is an int subclass; not a char count
        return DEFAULT_MAX_CHARS
    try:
        out = int(v)
    except (TypeError, ValueError):
        return DEFAULT_MAX_CHARS
    return max(_MIN_MAX_CHARS, out)
