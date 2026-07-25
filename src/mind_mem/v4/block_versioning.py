#!/usr/bin/env python3
"""Block versioning + time-travel — reconstruct what a block said, and when.

Roadmap Group B: ``block_history(block_id)`` and ``recall(..., as_of=date)``.

This is a **read-only projection** over the ``block_edits`` table already
written by :mod:`mind_mem.v4.self_editing` — no new schema, no new writes, and
therefore no migration. Every applied edit records ``old_content`` and
``new_content``, so the applied edits for a block already *are* its version
chain; this module simply materialises that chain and lets a caller ask what
the content was at an arbitrary instant.

Semantics (deliberately conservative):

* Only edits with status ``applied`` count. A ``pending`` edit never took
  effect and a ``rejected`` one never will, so neither may appear in history —
  otherwise time-travel would report content the workspace never actually had.
* Version 1 is the content *before* the first applied edit — taken from that
  edit's ``old_content``. Its ``valid_from`` is ``None``: the block existed
  with that content at some earlier, unrecorded time. Reporting a fabricated
  timestamp here would be a lie, so the field stays honestly empty.
* Version *n+1* is the ``new_content`` of the *n*-th applied edit, valid from
  that edit's ``approved_at`` (the moment it took effect), falling back to
  ``proposed_at`` only when ``approved_at`` is missing.
* A block with no applied edits has an **empty** history — it has only ever
  had its current content, which lives in the block store, not here. Callers
  treat "empty history" as "current content has always applied".

Timestamps are ISO-8601 strings as written by the rest of the v4 surface, and
are compared lexicographically — correct for a fixed-offset ISO-8601 format,
which is what ``self_editing`` writes.

Feature-flag gated under ``v4.self_editing`` — this is a view over that
surface's data, so it is enabled by exactly the same flag rather than
introducing a second one that could drift out of sync.

Copyright STARGA, Inc.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .feature_flags import require_enabled
from .self_editing import EditStatus

__all__ = [
    "FLAG",
    "BlockVersion",
    "block_history",
    "content_as_of",
    "versioned_block_ids",
]


FLAG: str = "self_editing"


@dataclass(frozen=True)
class BlockVersion:
    """One materialised revision of a block's content.

    ``valid_from`` is the instant this revision took effect, or ``None`` for
    version 1 (the content predates the first recorded edit).
    """

    block_id: str
    version: int
    content: str | None
    valid_from: str | None
    edit_id: int | None
    reason: str | None
    approver: str | None


def block_history(workspace: str | Path, block_id: str) -> list[BlockVersion]:
    """Return the full version chain for ``block_id``, oldest first.

    Empty when the block has no applied edits (it has only ever held its
    current content). Pending and rejected edits are excluded — they never
    took effect.
    """
    require_enabled(FLAG)
    rows = _applied_edits(workspace, block_id)
    if not rows:
        return []

    first_edit_id, _, first_old, _, first_reason, _, _, _ = rows[0]
    versions: list[BlockVersion] = [
        BlockVersion(
            block_id=block_id,
            version=1,
            content=first_old,
            valid_from=None,
            edit_id=None,
            reason=None,
            approver=None,
        )
    ]
    for n, row in enumerate(rows, start=2):
        edit_id, _bid, _old, new_content, reason, proposed_at, approved_at, approver = row
        versions.append(
            BlockVersion(
                block_id=block_id,
                version=n,
                content=new_content,
                valid_from=approved_at or proposed_at,
                edit_id=int(edit_id),
                reason=reason,
                approver=approver,
            )
        )
    return versions


def content_as_of(workspace: str | Path, block_id: str, as_of: str) -> str | None:
    """Return the block's content as it stood at ``as_of`` (ISO-8601).

    Returns ``None`` when the block has no applied edits — the caller should
    fall back to the block's current content, which this module deliberately
    does not read (it owns history, not the live store). ``None`` is also
    returned when the pre-edit content itself was never recorded.
    """
    require_enabled(FLAG)
    if not as_of:
        return None
    history = block_history(workspace, block_id)
    if not history:
        return None

    current: BlockVersion = history[0]
    for version in history[1:]:
        # valid_from is None only for version 1, which is already the seed.
        if version.valid_from is not None and version.valid_from <= as_of:
            current = version
        else:
            break
    return current.content


def versioned_block_ids(workspace: str | Path, *, limit: int = 1000) -> list[str]:
    """Return block_ids that have at least one applied edit, oldest edit first.

    Useful for enumerating what time-travel can actually answer for.
    """
    require_enabled(FLAG)
    if limit <= 0:
        return []
    db = Path(workspace) / "index.db"
    if not db.is_file():
        return []
    with sqlite3.connect(db, timeout=30) as conn:
        if not _table_exists(conn, "block_edits"):
            return []
        rows = conn.execute(
            "SELECT block_id, MIN(COALESCE(approved_at, proposed_at)) AS first_at "
            "FROM block_edits WHERE status = ? "
            "GROUP BY block_id ORDER BY first_at ASC, block_id ASC LIMIT ?",
            (EditStatus.APPLIED, limit),
        ).fetchall()
    return [str(row[0]) for row in rows]


def _applied_edits(workspace: str | Path, block_id: str) -> list[tuple]:
    """Applied edits for a block, oldest-effective first."""
    db = Path(workspace) / "index.db"
    if not db.is_file():
        return []
    with sqlite3.connect(db, timeout=30) as conn:
        if not _table_exists(conn, "block_edits"):
            return []
        return conn.execute(
            "SELECT edit_id, block_id, old_content, new_content, reason, "
            "proposed_at, approved_at, approver "
            "FROM block_edits WHERE block_id = ? AND status = ? "
            "ORDER BY COALESCE(approved_at, proposed_at) ASC, edit_id ASC",
            (block_id, EditStatus.APPLIED),
        ).fetchall()


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?",
        (name,),
    ).fetchone()
    return row is not None
