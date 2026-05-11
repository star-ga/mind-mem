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
    # Multi-label surface (added per v4-audit-2026-05-10, 3/4 model
    # consensus). A block can carry multiple kinds simultaneously
    # (e.g. a Python class is both `entity` and `code`). The legacy
    # single-label `blocks.kind` column stays as the primary kind so
    # v3-compat reads keep working; multi-label callers use the
    # junction-table API below.
    "set_block_kinds",
    "get_block_kind_tags",
    "ensure_block_kind_tags_table",
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
    with sqlite3.connect(db, timeout=30) as conn:
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
    with sqlite3.connect(db, timeout=30) as conn:
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
    with sqlite3.connect(db, timeout=30) as conn:
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


# ---------------------------------------------------------------------------
# Multi-label surface  (v4-audit-2026-05-10, 3/4 model consensus)
# ---------------------------------------------------------------------------
#
# A block can be more than one kind at the same time — e.g. a Python class
# defining a User is both ``code`` (the file/symbol reference) and
# ``entity`` (the User itself). The single-label ``blocks.kind`` column
# stays as the *primary* kind for v3-compat reads; the junction table
# below records every kind a block carries, including the primary.
#
# Schema:
#
#     CREATE TABLE block_kind_tags (
#         block_id TEXT NOT NULL,
#         kind     TEXT NOT NULL,
#         PRIMARY KEY (block_id, kind),
#         FOREIGN KEY (block_id) REFERENCES blocks(id) ON DELETE CASCADE
#     )
#
# Foreign key omitted for SQLite (where FK constraints are off by
# default and turning them on would interfere with v3 ingestion); the
# many-to-many shape is enforced by the composite primary key alone.

_TAGS_SCHEMA_SQL: str = """
CREATE TABLE IF NOT EXISTS block_kind_tags (
    block_id TEXT NOT NULL,
    kind     TEXT NOT NULL,
    PRIMARY KEY (block_id, kind)
);
CREATE INDEX IF NOT EXISTS idx_block_kind_tags_kind
    ON block_kind_tags (kind);
"""


def ensure_block_kind_tags_table(workspace: str | Path) -> None:
    """Create the ``block_kind_tags`` junction table on first call.

    Idempotent. Raises :class:`FeatureDisabledError` if the flag is OFF.
    Safe to call alongside :func:`ensure_block_kind_column`; the two
    surfaces are independent — single-label callers can stay on the
    column, multi-label callers add a tags table next to it.
    """
    require_enabled(FLAG)
    db = Path(workspace) / "index.db"
    if not db.parent.is_dir():
        db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db, timeout=30) as conn:
        conn.executescript(_TAGS_SCHEMA_SQL)
        conn.commit()


def set_block_kinds(
    workspace: str | Path,
    block_id: str,
    kinds: Iterable[BlockKind | str],
) -> set[BlockKind]:
    """Replace the kind tag set for ``block_id`` with ``kinds``.

    Returns the set actually written (de-duplicated, validated). An
    empty ``kinds`` clears every tag for this block. Unknown strings
    raise :class:`ValueError` at the constructor — fail-loud so a
    typo can't silently shrink a block's tag set.

    Multi-label tags are stored in :data:`block_kind_tags`; the
    legacy single-label :data:`blocks.kind` column is left untouched
    by this function so v3-compat reads keep returning the primary
    kind. Callers that want the column synchronised should write to
    it directly alongside this call.
    """
    require_enabled(FLAG)
    validated: set[BlockKind] = set()
    for k in kinds:
        if isinstance(k, str):
            validated.add(BlockKind(k))
        else:
            validated.add(k)
    db = Path(workspace) / "index.db"
    if not db.parent.is_dir():
        db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db, timeout=30) as conn:
        conn.executescript(_TAGS_SCHEMA_SQL)
        conn.execute("DELETE FROM block_kind_tags WHERE block_id = ?", (block_id,))
        if validated:
            conn.executemany(
                "INSERT INTO block_kind_tags (block_id, kind) VALUES (?, ?)",
                [(block_id, k.value) for k in validated],
            )
        conn.commit()
    return validated


def get_block_kind_tags(workspace: str | Path, block_id: str) -> set[BlockKind]:
    """Return every kind tag carried by ``block_id``.

    Empty set when the block has no tags, when the table doesn't
    exist, or when the database is missing — fail-soft same as the
    rest of the v4 read surface. Unknown stored values are silently
    skipped (a corrupt row can't kill the recall path).

    Note: this is the multi-label reader; :func:`get_block_kind`
    returns the *primary* kind from the legacy column for v3 compat.
    Callers that want the union should call both.
    """
    require_enabled(FLAG)
    db = Path(workspace) / "index.db"
    if not db.is_file():
        return set()
    with sqlite3.connect(db, timeout=30) as conn:
        if not _table_exists(conn, "block_kind_tags"):
            return set()
        rows = conn.execute(
            "SELECT kind FROM block_kind_tags WHERE block_id = ?",
            (block_id,),
        ).fetchall()
    out: set[BlockKind] = set()
    for r in rows:
        try:
            out.add(BlockKind(r[0]))
        except ValueError:
            continue
    return out


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?",
        (name,),
    ).fetchone()
    return row is not None
