"""v4 block-kind taxonomy (Group B: knowledge graph).

Promotes blocks from flat to multi-page by attaching a *kind* tag:

    entity      — a person, organization, place, project, system
    concept     — an abstract idea or category that ties entities together
    source      — a citation: paper, post, transcript, code, dataset
    synthesis   — a writeup that ties many sources/entities together
    image       — visual asset (URL + hash; never raw bytes)
    audio       — audio asset (URL + hash; never raw bytes)
    code        — a code-symbol or file reference
    structured  — JSON / table / typed record

The default v3.x flat-block schema has no ``kind`` column. v4 adds one,
zero-downtime: ``ALTER TABLE ... ADD COLUMN kind TEXT NOT NULL DEFAULT
'unspecified'`` makes every existing row legal under the new schema
without any data movement. v3.x reads/writes ignore the column; v4
readers branch on it.

Two retrieval modes coexist downstream (landed in
:mod:`mind_mem.v4.long_context_recall`):

    chunked top-K (current default)
        Fast, low token cost. Returns RRF-ranked chunks across all kinds.

    long-context union (v4 opt-in)
        Returns full ``entity`` / ``concept`` pages whose summaries match.
        Higher token cost; preserves relational understanding.

This module ships the **type + read surface only**:
:class:`BlockKind` enum, :func:`ensure_block_kind_column`,
:func:`get_block_kind`, :func:`list_blocks_by_kind`. The fusion / merge
side (``propose_fuse``, multi-page entity reconciliation) lands in
:mod:`mind_mem.v4.fusion` once block_kinds is stable.

The feature is **off by default**. Calling any public function without
the ``v4.block_kinds`` flag enabled raises
:class:`FeatureDisabledError`. v3.x callers see no behaviour change.

Copyright STARGA, Inc.
"""

from __future__ import annotations

import sqlite3
from enum import Enum
from pathlib import Path
from typing import Iterable

from .feature_flags import require_enabled

__all__ = [
    "FLAG",
    "BlockKind",
    "DEFAULT_KIND",
    "ALLOWED_KINDS",
    "ensure_block_kind_column",
    "get_block_kind",
    "list_blocks_by_kind",
]


#: The feature-flag key in ``mind-mem.json: v4: {...}``.
FLAG: str = "block_kinds"


class BlockKind(str, Enum):
    """Eight typed block kinds plus an explicit ``UNSPECIFIED`` default.

    ``UNSPECIFIED`` is the value v3.x flat blocks get when v4 first
    adds the column. v4 readers can either treat it as a 9th kind or
    coerce it to a v3-compatible default; both paths are legal.
    """

    ENTITY = "entity"
    CONCEPT = "concept"
    SOURCE = "source"
    SYNTHESIS = "synthesis"
    IMAGE = "image"
    AUDIO = "audio"
    CODE = "code"
    STRUCTURED = "structured"
    UNSPECIFIED = "unspecified"


#: Default for v3.x rows when the column is added by an ALTER. Matches
#: the SQL DEFAULT in :data:`_ADD_COLUMN_SQL`.
DEFAULT_KIND: BlockKind = BlockKind.UNSPECIFIED


#: Allowed kind strings — every value of :class:`BlockKind`. Useful for
#: validators that need a fast set membership test.
ALLOWED_KINDS: frozenset[str] = frozenset(k.value for k in BlockKind)


# ---------------------------------------------------------------------------
# Schema migration
# ---------------------------------------------------------------------------

#: Zero-downtime ALTER that makes every v3.x row legal under v4.
_ADD_COLUMN_SQL: str = "ALTER TABLE blocks ADD COLUMN kind TEXT NOT NULL DEFAULT 'unspecified'"

#: Index on the new column so ``list_blocks_by_kind`` doesn't full-scan
#: a million-row blocks table.
_INDEX_SQL: str = "CREATE INDEX IF NOT EXISTS idx_blocks_kind ON blocks (kind)"


def ensure_block_kind_column(workspace: str | Path) -> None:
    """Add the ``kind`` column to ``blocks`` if absent. Idempotent.

    Walks the SQLite ``PRAGMA table_info(blocks)`` cursor and only
    issues the ALTER when ``kind`` is missing — running this on every
    write path is safe. Adds an index on the new column at the same
    time so kind-filtered queries scale.

    Raises :class:`FeatureDisabledError` if the flag is OFF.
    """
    require_enabled(FLAG)
    db = Path(workspace) / "index.db"
    if not db.parent.is_dir():
        db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db) as conn:
        # Ensure the blocks table exists at all (fresh workspaces) so the
        # ALTER below has something to alter.
        conn.execute("CREATE TABLE IF NOT EXISTS blocks (id TEXT PRIMARY KEY, content TEXT)")
        cols = {row[1] for row in conn.execute("PRAGMA table_info(blocks)")}
        if "kind" not in cols:
            conn.execute(_ADD_COLUMN_SQL)
        conn.execute(_INDEX_SQL)
        conn.commit()


# ---------------------------------------------------------------------------
# Reader API
# ---------------------------------------------------------------------------


def get_block_kind(workspace: str | Path, block_id: str) -> BlockKind:
    """Return the kind for a single block.

    Blocks with no row in ``blocks`` (or a missing column) return
    :data:`DEFAULT_KIND`. Unknown stored values also coerce to
    :data:`DEFAULT_KIND` rather than raising — fail-soft so a single
    corrupt row can't kill the recall path.
    """
    require_enabled(FLAG)
    db = Path(workspace) / "index.db"
    if not db.is_file():
        return DEFAULT_KIND
    with sqlite3.connect(db) as conn:
        if not _has_kind_column(conn):
            return DEFAULT_KIND
        row = conn.execute(
            "SELECT kind FROM blocks WHERE id = ?",
            (block_id,),
        ).fetchone()
    if row is None:
        return DEFAULT_KIND
    try:
        return BlockKind(row[0])
    except ValueError:
        return DEFAULT_KIND


def list_blocks_by_kind(
    workspace: str | Path,
    kind: BlockKind | str,
    *,
    limit: int = 100,
) -> list[str]:
    """Return up to ``limit`` block IDs of the given kind.

    Empty list when the schema doesn't exist yet, when no rows match,
    or when ``limit`` is non-positive. Order is the SQLite default
    (insertion / rowid) — callers that need a specific order should
    filter against a separate metadata table.
    """
    require_enabled(FLAG)
    if isinstance(kind, str):
        kind = BlockKind(kind)
    if int(limit) <= 0:
        return []
    db = Path(workspace) / "index.db"
    if not db.is_file():
        return []
    with sqlite3.connect(db) as conn:
        if not _has_kind_column(conn):
            return []
        rows: Iterable[tuple[str]] = conn.execute(
            "SELECT id FROM blocks WHERE kind = ? LIMIT ?",
            (kind.value, int(limit)),
        ).fetchall()
    return [r[0] for r in rows]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _has_kind_column(conn: sqlite3.Connection) -> bool:
    """``PRAGMA table_info`` lookup — True when ``blocks.kind`` exists."""
    try:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(blocks)")}
    except sqlite3.Error:
        return False
    return "kind" in cols
