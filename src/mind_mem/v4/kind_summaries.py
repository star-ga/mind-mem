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
                    that want an LLM-driven summariser. The write door
                    applies the same configured ``max_chars`` cap and
                    prewrite redaction to installed callables; their
                    semantic claims are stored as ``unverified``.

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
import hashlib
import json
import sqlite3
import unicodedata
from collections.abc import Callable, Iterable
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from ..compliance.prewrite import PreWritePolicy, screen
from .feature_flags import flag_config, require_enabled

__all__ = [
    "FLAG",
    "Summariser",
    "KindSummary",
    "SummaryOutputError",
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
    """Read-only summary record for one kind.

    ``enforcement`` is ``verified`` only for the built-in deterministic
    summariser. Installed callables are bounded and screened, but their
    semantic claims cannot be proven by this adapter and are therefore
    exposed as ``unverified`` to every reader.
    """

    kind: str
    summary: str
    block_count: int
    updated_at: str
    source_ids: tuple[str, ...] = ()
    source_digest: str = ""
    enforcement: str = "unverified"


class SummaryOutputError(ValueError):
    """A summary output cannot safely cross the SQLite write boundary."""


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
    updated_at   TEXT NOT NULL,
    source_ids   TEXT NOT NULL DEFAULT '[]',
    source_digest TEXT NOT NULL DEFAULT '',
    enforcement  TEXT NOT NULL DEFAULT 'unverified'
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
        columns = {row[1] for row in conn.execute("PRAGMA table_info(kind_summaries)")}
        # Existing workspaces predate source binding.  Keep them readable,
        # but make their status explicit rather than manufacturing proof.
        for name, ddl in (
            ("source_ids", "TEXT NOT NULL DEFAULT '[]'"),
            ("source_digest", "TEXT NOT NULL DEFAULT ''"),
            ("enforcement", "TEXT NOT NULL DEFAULT 'unverified'"),
        ):
            if name not in columns:
                conn.execute(f"ALTER TABLE kind_summaries ADD COLUMN {name} {ddl}")
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
        id_column = "id" if "id" in cols else "rowid"
        rows = conn.execute(
            f"SELECT {id_column}, content FROM blocks WHERE kind = ? ORDER BY {id_column}",
            (kind,),
        ).fetchall()
    source_rows = [(str(row[0]), row[1] if isinstance(row[1], str) else "") for row in rows]
    if not source_rows:
        return None
    blocks = [content for _, content in source_rows]
    source_ids = tuple(block_id for block_id, _ in source_rows)
    source_digest = _source_digest(source_rows)
    summary = _active_summariser(blocks)
    summary, enforcement = _screen_summary(
        summary,
        workspace=workspace,
        kind=kind,
        source_digest=source_digest,
        trusted=_active_summariser is default_summariser,
    )
    now = _dt.datetime.now(_dt.timezone.utc).isoformat()
    with closing(sqlite3.connect(db, timeout=30)) as conn, conn:
        conn.execute(
            """INSERT OR REPLACE INTO kind_summaries
               (kind, summary, block_count, updated_at, source_ids, source_digest, enforcement)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (kind, summary, len(source_rows), now, json.dumps(source_ids), source_digest, enforcement),
        )
        conn.commit()
    return KindSummary(
        kind=kind,
        summary=summary,
        block_count=len(source_rows),
        updated_at=now,
        source_ids=source_ids,
        source_digest=source_digest,
        enforcement=enforcement,
    )


def get_summary(workspace: str | Path, kind: str) -> KindSummary | None:
    """Return the stored summary for ``kind``, or ``None`` if absent."""
    require_enabled(FLAG)
    db = Path(workspace) / "index.db"
    if not db.is_file():
        return None
    with closing(sqlite3.connect(db, timeout=30)) as conn, conn:
        if not _table_exists(conn, "kind_summaries"):
            return None
        columns = {row[1] for row in conn.execute("PRAGMA table_info(kind_summaries)")}
        if {"source_ids", "source_digest", "enforcement"} <= columns:
            row = conn.execute(
                """SELECT kind, summary, block_count, updated_at, source_ids, source_digest, enforcement
                   FROM kind_summaries WHERE kind = ?""",
                (kind,),
            ).fetchone()
        else:
            old = conn.execute(
                "SELECT kind, summary, block_count, updated_at FROM kind_summaries WHERE kind = ?",
                (kind,),
            ).fetchone()
            row = (*old, "[]", "", "unverified") if old is not None else None
    if row is None:
        return None
    return KindSummary(
        kind=row[0],
        summary=row[1],
        block_count=int(row[2]),
        updated_at=row[3],
        source_ids=_decode_source_ids(row[4]),
        source_digest=str(row[5] or ""),
        enforcement=str(row[6] or "unverified"),
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
        columns = {row[1] for row in conn.execute("PRAGMA table_info(kind_summaries)")}
        if {"source_ids", "source_digest", "enforcement"} <= columns:
            rows = conn.execute(
                """SELECT kind, summary, block_count, updated_at, source_ids, source_digest, enforcement
                   FROM kind_summaries ORDER BY kind"""
            ).fetchall()
        else:
            rows = [
                (*row, "[]", "", "unverified")
                for row in conn.execute("SELECT kind, summary, block_count, updated_at FROM kind_summaries ORDER BY kind").fetchall()
            ]
    return [
        KindSummary(
            kind=r[0],
            summary=r[1],
            block_count=int(r[2]),
            updated_at=r[3],
            source_ids=_decode_source_ids(r[4]),
            source_digest=str(r[5] or ""),
            enforcement=str(r[6] or "unverified"),
        )
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _source_digest(rows: Iterable[tuple[str, str]]) -> str:
    """Hash the exact ordered source ids and text supplied to the plugin."""
    payload = [{"id": block_id, "content": content} for block_id, content in rows]
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _decode_source_ids(raw: object) -> tuple[str, ...]:
    try:
        values = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError):
        return ()
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        return ()
    return tuple(values)


def _screen_summary(
    summary: object,
    *,
    workspace: str | Path,
    kind: str,
    source_digest: str,
    trusted: bool,
) -> tuple[str, str]:
    """Validate and prewrite-screen output before opening the write transaction."""
    if not isinstance(summary, str):
        raise SummaryOutputError(f"summariser must return str, got {type(summary).__name__}")
    cap = _max_chars()
    if len(summary) > cap:
        raise SummaryOutputError(f"summary exceeds configured max_chars ({len(summary)} > {cap})")
    if "\x00" in summary or any((ord(char) < 32 and char not in "\r\n\t") or unicodedata.category(char) == "Cs" for char in summary):
        raise SummaryOutputError("summary contains unsafe control characters")

    # This is the same prewrite door as other governed writes.  The synthetic
    # provenance is about the derived operation itself, never a claim about
    # who authored the source blocks.  ``record=False`` keeps refresh side
    # effects limited to its own SQLite row.
    provenance = {
        "ActorId": "kind_summaries",
        "ActorRole": "derived-summary",
        "SessionId": source_digest,
        "ToolId": "v4.kind_summaries",
        "Purpose": f"refresh kind {kind}",
    }
    screened = screen(
        summary,
        policy=PreWritePolicy.resolve(str(workspace)),
        provenance=provenance,
        target=f"kind_summaries/{kind}",
        record=False,
    )
    return screened.text, "verified" if trusted else "unverified"


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
