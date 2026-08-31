"""v4 self-editing on recall (Group A — MemGPT pattern).

Round 2 multi-LLM audit (2/4 model agreement 2026-05-10) recommended
adding a self-editing surface so the agent can update block content
in place when a recall surfaces a wrong / outdated fact. The pattern
mirrors MemGPT's inline memory edits but stays disciplined: every
edit goes through the existing v3 propose/approve governance flow
rather than mutating directly.

Surface:

    propose_edit(workspace, block_id, new_content, reason) -> edit_id
    approve_edit(edit_id) -> Edit
    list_pending_edits(workspace) -> [Edit]
    list_edit_history(workspace, block_id) -> [Edit]

Schema (idempotent):

    block_edits(
        edit_id      INTEGER PRIMARY KEY AUTOINCREMENT,
        block_id     TEXT NOT NULL,
        old_content  TEXT,
        new_content  TEXT NOT NULL,
        reason       TEXT NOT NULL,
        proposed_at  TEXT NOT NULL,
        status       TEXT NOT NULL DEFAULT 'pending',
        approved_at  TEXT,
        approver     TEXT
    )

Status transitions:
    pending → applied   on approve_edit()
    pending → rejected  on reject_edit()

The ``applied`` status is the green path; the planner does NOT
mutate ``blocks.content`` directly — the audit chain in
:mod:`mind_mem.audit_chain` handles the actual write through the v3
governance layer when an applied edit is detected. This keeps the
self-edit surface compatible with the existing audit trail.

Feature-flag gated under ``v4.self_editing``.

Copyright STARGA, Inc.
"""

from __future__ import annotations

import datetime as _dt
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from ..observability import get_logger
from .feature_flags import require_enabled

__all__ = [
    "FLAG",
    "Edit",
    "EditStatus",
    "ensure_edit_schema",
    "propose_edit",
    "approve_edit",
    "reject_edit",
    "list_pending_edits",
    "list_edit_history",
    "get_edit",
]


_log = get_logger("v4.self_editing")

FLAG: str = "self_editing"


class EditStatus:
    PENDING = "pending"
    APPLIED = "applied"
    REJECTED = "rejected"


@dataclass(frozen=True)
class Edit:
    """Read-only proposed/applied edit record."""

    edit_id: int
    block_id: str
    old_content: str | None
    new_content: str
    reason: str
    proposed_at: str
    status: str
    approved_at: str | None
    approver: str | None


# ---------------------------------------------------------------------------
# Connection discipline
# ---------------------------------------------------------------------------
#
# Every connection in this module is opened as::
#
#     with closing(sqlite3.connect(...)) as conn, conn:
#
# ``with conn`` commits (or rolls back) and leaves the handle OPEN, and a
# ``sqlite3.Connection`` is kept alive by its own prepared-statement cache — a
# reference cycle — so refcounting never reclaims it and the process keeps a
# descriptor on ``index.db`` and its ``-wal``/``-shm`` sidecars until the cyclic
# collector runs. A leak everywhere, and on Windows an open handle makes
# ``os.unlink`` fail, so the workspace directory cannot be removed.
#
# ``closing`` is the OUTER manager on purpose: the managers unwind in reverse,
# so the commit-or-rollback happens BEFORE the close. ``close()`` alone never
# commits and rolls back an open transaction. See ``hash_chain_v2._session``
# for the same discipline expressed as a method.


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA_SQL: str = """
CREATE TABLE IF NOT EXISTS block_edits (
    edit_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    block_id     TEXT NOT NULL,
    old_content  TEXT,
    new_content  TEXT NOT NULL,
    reason       TEXT NOT NULL,
    proposed_at  TEXT NOT NULL,
    status       TEXT NOT NULL DEFAULT 'pending',
    approved_at  TEXT,
    approver     TEXT
);
CREATE INDEX IF NOT EXISTS idx_block_edits_block
    ON block_edits (block_id, status);
CREATE INDEX IF NOT EXISTS idx_block_edits_status
    ON block_edits (status, proposed_at);
"""


def ensure_edit_schema(workspace: str | Path) -> None:
    """Idempotent. Creates the ``block_edits`` table."""
    require_enabled(FLAG)
    db = Path(workspace) / "index.db"
    if not db.parent.is_dir():
        db.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(db, timeout=30)) as conn, conn:
        conn.executescript(_SCHEMA_SQL)
        conn.commit()


# ---------------------------------------------------------------------------
# Write API
# ---------------------------------------------------------------------------


def propose_edit(
    workspace: str | Path,
    block_id: str,
    new_content: str,
    reason: str,
) -> int:
    """Insert a pending edit; return the assigned edit_id.

    The planner does NOT mutate ``blocks.content``; it captures the
    edit for later approval. The reason is required (non-empty) so
    the audit trail carries a rationale even before approval.
    """
    require_enabled(FLAG)
    if not reason or not reason.strip():
        raise ValueError("reason is required for self-edit proposals")
    ensure_edit_schema(workspace)
    db = Path(workspace) / "index.db"
    now = _dt.datetime.now(_dt.timezone.utc).isoformat()
    # Snapshot the CURRENT content into old_content for the audit log. This
    # is the seed of the version chain (block_versioning.block_history makes
    # it version 1), so a NULL here is not benign: content_as_of() then
    # returns None for every instant before the first applied edit, and
    # recall(..., as_of=…) reads that None as "this block has no recorded
    # edits" and serves today's content as historical.
    old_content = _current_block_content(workspace, block_id)
    with closing(sqlite3.connect(db, timeout=30)) as conn, conn:
        if old_content is None:
            # Legacy fallback: the v4 index.db ``blocks`` table. Nothing in
            # this package inserts rows there, so it is normally empty —
            # kept only so a workspace that DOES populate it is not lost.
            old_row = (
                conn.execute(
                    "SELECT content FROM blocks WHERE id = ?",
                    (block_id,),
                ).fetchone()
                if _table_exists(conn, "blocks")
                else None
            )
            old_content = old_row[0] if old_row else None
        if old_content is None:
            # Never silent: the proposal is still recorded (refusing it
            # would block edits on a block the store cannot resolve), but
            # its version chain will start from an unknown revision.
            _log.warning("self_edit_old_content_unavailable", block_id=block_id)
        cursor = conn.execute(
            "INSERT INTO block_edits (block_id, old_content, new_content, reason, proposed_at, status) VALUES (?, ?, ?, ?, ?, ?)",
            (block_id, old_content, new_content, reason, now, EditStatus.PENDING),
        )
        edit_id = int(cursor.lastrowid or 0)
        conn.commit()
    return edit_id


def approve_edit(workspace: str | Path, edit_id: int, approver: str = "system") -> Edit | None:
    """Mark a pending edit applied. Returns the resulting :class:`Edit`
    record or ``None`` if the edit doesn't exist or isn't pending.

    The actual ``blocks.content`` write is left to the v3 audit-chain
    apply path; this function only flips status. Keeps the v3
    governance contract authoritative for the source-of-truth content.
    """
    require_enabled(FLAG)
    return _transition_edit(workspace, edit_id, EditStatus.APPLIED, approver)


def reject_edit(workspace: str | Path, edit_id: int, approver: str = "system") -> Edit | None:
    """Mark a pending edit rejected. Mirror image of :func:`approve_edit`."""
    require_enabled(FLAG)
    return _transition_edit(workspace, edit_id, EditStatus.REJECTED, approver)


def _transition_edit(
    workspace: str | Path,
    edit_id: int,
    new_status: str,
    approver: str,
) -> Edit | None:
    db = Path(workspace) / "index.db"
    if not db.is_file():
        return None
    now = _dt.datetime.now(_dt.timezone.utc).isoformat()
    with closing(sqlite3.connect(db, timeout=10)) as conn, conn:
        if not _table_exists(conn, "block_edits"):
            return None
        cursor = conn.execute(
            "UPDATE block_edits SET status = ?, approved_at = ?, approver = ? WHERE edit_id = ? AND status = ?",
            (new_status, now, approver, edit_id, EditStatus.PENDING),
        )
        if cursor.rowcount != 1:
            return None
        conn.commit()
    return get_edit(workspace, edit_id)


# ---------------------------------------------------------------------------
# Read API
# ---------------------------------------------------------------------------


def get_edit(workspace: str | Path, edit_id: int) -> Edit | None:
    """Return the :class:`Edit` record for an edit_id, or ``None``."""
    require_enabled(FLAG)
    db = Path(workspace) / "index.db"
    if not db.is_file():
        return None
    with closing(sqlite3.connect(db, timeout=30)) as conn, conn:
        if not _table_exists(conn, "block_edits"):
            return None
        row = conn.execute(
            "SELECT edit_id, block_id, old_content, new_content, reason, "
            "proposed_at, status, approved_at, approver "
            "FROM block_edits WHERE edit_id = ?",
            (edit_id,),
        ).fetchone()
    if row is None:
        return None
    return _row_to_edit(row)


def list_pending_edits(workspace: str | Path, *, limit: int = 100) -> list[Edit]:
    """Return up to ``limit`` pending edits, oldest first."""
    require_enabled(FLAG)
    if limit <= 0:
        return []
    db = Path(workspace) / "index.db"
    if not db.is_file():
        return []
    with closing(sqlite3.connect(db, timeout=30)) as conn, conn:
        if not _table_exists(conn, "block_edits"):
            return []
        rows = conn.execute(
            "SELECT edit_id, block_id, old_content, new_content, reason, "
            "proposed_at, status, approved_at, approver "
            "FROM block_edits WHERE status = ? ORDER BY proposed_at ASC LIMIT ?",
            (EditStatus.PENDING, int(limit)),
        ).fetchall()
    return [_row_to_edit(r) for r in rows]


def list_edit_history(workspace: str | Path, block_id: str) -> list[Edit]:
    """Return every edit for a block, oldest first.

    Useful for audit replay: the history is append-only — applied
    and rejected edits stay in the table forever.
    """
    require_enabled(FLAG)
    db = Path(workspace) / "index.db"
    if not db.is_file():
        return []
    with closing(sqlite3.connect(db, timeout=30)) as conn, conn:
        if not _table_exists(conn, "block_edits"):
            return []
        rows = conn.execute(
            "SELECT edit_id, block_id, old_content, new_content, reason, "
            "proposed_at, status, approved_at, approver "
            "FROM block_edits WHERE block_id = ? ORDER BY proposed_at ASC",
            (block_id,),
        ).fetchall()
    return [_row_to_edit(r) for r in rows]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row_to_edit(row: tuple) -> Edit:
    return Edit(
        edit_id=int(row[0]),
        block_id=row[1],
        old_content=row[2],
        new_content=row[3],
        reason=row[4],
        proposed_at=row[5],
        status=row[6],
        approved_at=row[7],
        approver=row[8],
    )


def _current_block_content(workspace: str | Path, block_id: str) -> str | None:
    """Current content of *block_id* from the workspace's blocks of record.

    Reads the configured block store (Markdown corpus by default,
    Postgres when configured) — the only place a block's live content
    actually lives. Returns ``None`` when the block cannot be resolved or
    carries no content field; the caller logs that rather than recording
    an indistinguishable NULL.
    """
    try:
        from ..storage import get_block_store

        block = get_block_store(str(workspace)).get_by_id(block_id)
    except Exception as exc:  # store unreachable / not configured
        _log.debug("self_edit_block_lookup_failed", block_id=block_id, error=str(exc))
        return None
    if not block:
        return None
    # ``Statement`` is the canonical content field of a mind-mem block and
    # what the stores themselves index as content
    # (``block_store_postgres._block_to_row``); ``content`` covers stores
    # that surface the rendered text directly.
    for key in ("Statement", "content"):
        value = block.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?",
        (name,),
    ).fetchone()
    return row is not None
