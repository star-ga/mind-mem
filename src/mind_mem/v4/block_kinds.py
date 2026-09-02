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

**Which database this is.** Every function here operates on
``<workspace>/index.db`` — the v4 side store shared with
:mod:`~mind_mem.v4.federation`, :mod:`~mind_mem.v4.self_editing` and the
rest of the v4 surface. It is **not** the v3 recall index, which lives at
``.mind-mem-index/recall.db`` (see :mod:`mind_mem.sqlite_index`) and is
never opened from this module. Read the ALTER below as *this store's*
schema, not as a migration of the corpus index: on a workspace that has
never used the v4 surface, ``ensure_block_kind_column`` creates an empty
``blocks`` table here and adds the column to that, so nothing is copied and
nothing is at risk — but equally, no existing block gains a kind by calling
it. Saying otherwise (this docstring used to) would promise a data
migration the code does not perform.

``ALTER TABLE ... ADD COLUMN kind TEXT NOT NULL DEFAULT 'unspecified'`` is
still the right shape for the store it does own: any row already present
stays legal under the new schema with no data movement, and readers that
predate the column ignore it.

**The primary column's writer is :func:`set_block_kind`.**
:func:`set_block_kinds` still deliberately writes only ``block_kind_tags``
and leaves the column alone — the two surfaces stay independent — so a
workspace that has never been backfilled reads :data:`DEFAULT_KIND` from
:func:`get_block_kind` and ``[]`` from :func:`list_blocks_by_kind`, and both
are behaving as written rather than failing. What changed in 5.0.1 is that
a backfill now exists to run: :mod:`mind_mem.v4.kind_backfill` calls
:func:`classify_block` over the *admitted* corpus and writes both surfaces.

The deferred note that used to sit here — *"the single-label column has a
reader and no in-package writer"* — is CLOSED as of 5.0.1: the first of its
two named upgrade paths is now implemented as :func:`set_block_kind`, which
writes the ``blocks`` row and its ``kind`` column beside the tag set. Its
caller is :mod:`mind_mem.v4.kind_backfill`, reached from ``mm kinds
backfill``, so ``get_block_kind`` and ``list_blocks_by_kind`` answer from
real data on a backfilled workspace instead of always falling back.

Two retrieval modes coexist downstream (landed in
:mod:`mind_mem.v4.long_context_recall`):

    chunked top-K (current default)
        Fast, low token cost. Returns RRF-ranked chunks across all kinds.

    long-context union (v4 opt-in)
        Returns full ``entity`` / ``concept`` pages whose summaries match.
        Higher token cost; preserves relational understanding.

This module ships the type surface, the read surface
(:class:`BlockKind` enum, :func:`ensure_block_kind_column`,
:func:`get_block_kind`, :func:`list_blocks_by_kind`) and, since 5.0.1, the
write surface (:func:`set_block_kind`, :func:`set_block_kinds`) plus the
deterministic :func:`classify_block` that decides what to write. The fusion / merge
side (``propose_fuse``, multi-page entity reconciliation) lands in
:mod:`mind_mem.v4.fusion` once block_kinds is stable.

The feature is **off by default**. Calling any public function without
the ``v4.block_kinds`` flag enabled raises
:class:`FeatureDisabledError`. v3.x callers see no behaviour change.

Copyright STARGA, Inc.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
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
    # The single-label WRITER the deferred note above asked for, plus the
    # deterministic corpus classifier that feeds it. Wired 5.0.1 through
    # ``mind_mem.v4.kind_backfill`` (`mm kinds backfill`).
    "set_block_kind",
    "classify_block",
    "primary_kind",
    "prune_kind_index",
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
# Schema migration
# ---------------------------------------------------------------------------

#: ALTER that makes every existing row of the v4 ``index.db`` ``blocks`` table
#: legal under the kind schema. Not a v3 migration — see the module docstring.
_ADD_COLUMN_SQL: str = "ALTER TABLE blocks ADD COLUMN kind TEXT NOT NULL DEFAULT 'unspecified'"

#: Index on the new column so ``list_blocks_by_kind`` doesn't full-scan
#: a million-row blocks table.
_INDEX_SQL: str = "CREATE INDEX IF NOT EXISTS idx_blocks_kind ON blocks (kind)"


def ensure_block_kind_column(workspace: str | Path) -> None:
    """Add the ``kind`` column to the v4 ``blocks`` table if absent. Idempotent.

    Operates on ``<workspace>/index.db``, the v4 side store — NOT on the v3
    recall index at ``.mind-mem-index/recall.db``. It therefore does not give
    existing corpus blocks a kind; it prepares this store's schema.

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
    with closing(sqlite3.connect(db, timeout=30)) as conn, conn:
        # The v4 store's own ``blocks`` table. On a workspace that has not used
        # the v4 surface this creates it empty, and the ALTER below then adds
        # the column to an empty table — the honest description of the common
        # case, and the reason this function migrates no corpus data.
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
    """Return the primary kind recorded for a single block in the v4 store.

    Blocks with no row in ``blocks`` (or a missing column) return
    :data:`DEFAULT_KIND`. Unknown stored values also coerce to
    :data:`DEFAULT_KIND` rather than raising — fail-soft so a single
    corrupt row can't kill the recall path.

    No function in this package writes ``blocks.kind``, so on a workspace
    with no external writer this returns :data:`DEFAULT_KIND` for every id.
    That is the fallback doing its job, not a lookup failure — for the tags
    this package does write, use :func:`get_block_kind_tags`.
    """
    require_enabled(FLAG)
    db = Path(workspace) / "index.db"
    if not db.is_file():
        return DEFAULT_KIND
    with closing(sqlite3.connect(db, timeout=30)) as conn, conn:
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

    Reads the same unwritten ``blocks.kind`` column as
    :func:`get_block_kind`, so with no external writer this is empty for
    every kind. An empty result here is "nothing is tagged in the column",
    never "the workspace holds no blocks of that kind".
    """
    require_enabled(FLAG)
    if isinstance(kind, str):
        kind = BlockKind(kind)
    if int(limit) <= 0:
        return []
    db = Path(workspace) / "index.db"
    if not db.is_file():
        return []
    with closing(sqlite3.connect(db, timeout=30)) as conn, conn:
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
    with closing(sqlite3.connect(db, timeout=30)) as conn, conn:
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
    with closing(sqlite3.connect(db, timeout=30)) as conn, conn:
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
    with closing(sqlite3.connect(db, timeout=30)) as conn, conn:
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


# ---------------------------------------------------------------------------
# Single-label writer  (5.0.1 — closes this module's own deferred note)
# ---------------------------------------------------------------------------


def set_block_kind(
    workspace: str | Path,
    block_id: str,
    kind: BlockKind | str,
    *,
    content: str | None = None,
) -> BlockKind:
    """Write the PRIMARY kind for ``block_id`` into ``blocks.kind``.

    This is the writer :func:`get_block_kind` and :func:`list_blocks_by_kind`
    were reading for and never had. It upserts the row in the v4 side store's
    ``blocks`` table (``<workspace>/index.db``, never the corpus and never the
    v3 recall index) and sets its ``kind`` column.

    ``content`` is optional and only ever *added*: passing ``None`` leaves any
    existing content untouched rather than blanking it, so a caller that knows
    a kind but not the text cannot erase text a previous caller stored. That
    matters because :func:`mind_mem.v4.kind_summaries.refresh_summary` reads
    ``blocks.content`` — a kind-only re-run must not empty every summary.

    Unknown ``kind`` strings raise :class:`ValueError` at the enum
    constructor: fail-loud, because a typo that silently became
    ``unspecified`` would be indistinguishable from a block nobody classified.

    Returns the :class:`BlockKind` actually written. Raises
    :class:`FeatureDisabledError` if the flag is OFF.
    """
    require_enabled(FLAG)
    if isinstance(kind, str):
        kind = BlockKind(kind)
    db = Path(workspace) / "index.db"
    if not db.parent.is_dir():
        db.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(db, timeout=30)) as conn, conn:
        conn.execute("CREATE TABLE IF NOT EXISTS blocks (id TEXT PRIMARY KEY, content TEXT)")
        cols = {row[1] for row in conn.execute("PRAGMA table_info(blocks)")}
        if "kind" not in cols:
            conn.execute(_ADD_COLUMN_SQL)
        conn.execute(_INDEX_SQL)
        # INSERT .. ON CONFLICT rather than INSERT OR REPLACE: REPLACE deletes
        # the old row first, which would drop the stored content on a
        # kind-only update and take any ON DELETE CASCADE rows with it.
        conn.execute(
            "INSERT INTO blocks (id, content, kind) VALUES (?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET kind = excluded.kind, "
            "content = COALESCE(excluded.content, blocks.content)",
            (block_id, content, kind.value),
        )
        conn.commit()
    return kind


# ---------------------------------------------------------------------------
# Deterministic corpus classifier
# ---------------------------------------------------------------------------
#
# What a block IS, decided from what the corpus already records about it —
# never from a model call, a clock or a random draw. The backfill has to be
# replayable: running it twice over an unchanged corpus must write the same
# kinds, or `list_blocks_by_kind` becomes a function of when you last ran it.

#: Corpus source label -> primary kind. The label is the directory the block
#: was parsed out of (``storage._iter_markdown_active_blocks`` stamps it), so
#: this is a fact about the corpus layout rather than a guess about content.
_LABEL_KIND: dict[str, BlockKind] = {
    "entities": BlockKind.ENTITY,
    "decisions": BlockKind.SYNTHESIS,
    "intelligence": BlockKind.SYNTHESIS,
    "signals": BlockKind.SOURCE,
    "tasks": BlockKind.STRUCTURED,
    "memory": BlockKind.SOURCE,
    "summaries": BlockKind.SYNTHESIS,
}

#: Block-id prefix -> primary kind, consulted when the source label is absent
#: (a Postgres-backed store hands back rows with no ``_source_label``). Kept
#: deliberately in step with ``mcp.tools.memory_ops._BLOCK_PREFIX_MAP``.
_PREFIX_KIND: dict[str, BlockKind] = {
    "PRJ": BlockKind.ENTITY,
    "PER": BlockKind.ENTITY,
    "TOOL": BlockKind.ENTITY,
    "INC": BlockKind.ENTITY,
    "D": BlockKind.SYNTHESIS,
    "C": BlockKind.SYNTHESIS,
    "DREF": BlockKind.SYNTHESIS,
    "T": BlockKind.STRUCTURED,
    "SIG": BlockKind.SOURCE,
    "INBOX": BlockKind.SOURCE,
    "IMP": BlockKind.SOURCE,
    "INGEST": BlockKind.SOURCE,
    "MSG": BlockKind.SOURCE,
    "TRAJ": BlockKind.STRUCTURED,
}

#: Fields whose presence means the block references code, whatever else it is.
#: This is the multi-label case the junction table exists for: a decision that
#: names a file is both ``synthesis`` and ``code``.
_CODE_FIELDS: tuple[str, ...] = ("File", "Path", "Symbol", "Function", "Module", "Class")

#: ``type:``/``Type:`` values that name a modality outright.
_TYPE_KIND: dict[str, BlockKind] = {
    "image": BlockKind.IMAGE,
    "audio": BlockKind.AUDIO,
    "code": BlockKind.CODE,
    "concept": BlockKind.CONCEPT,
    "entity": BlockKind.ENTITY,
    "source": BlockKind.SOURCE,
    "synthesis": BlockKind.SYNTHESIS,
    "structured": BlockKind.STRUCTURED,
}


def primary_kind(block: dict) -> BlockKind:
    """The single kind that goes in ``blocks.kind``. Pure; no I/O.

    Public because the backfill needs the primary and the tag set from one
    classification, and a private ``_primary_kind`` reached across modules is
    exactly the fragile private-import this slice is elsewhere removing.
    Deliberately NOT flag-gated: it is a pure lookup with no side effect, and
    :func:`classify_block` — the entry point callers use — carries the gate.
    """
    for key in ("type", "Type", "Kind"):
        raw = block.get(key)
        if isinstance(raw, str):
            hit = _TYPE_KIND.get(raw.strip().lower())
            if hit is not None:
                return hit
    label = block.get("_source_label")
    if isinstance(label, str):
        hit = _LABEL_KIND.get(label.strip().lower())
        if hit is not None:
            return hit
    bid = str(block.get("_id", ""))
    prefix = bid.split("-", 1)[0] if "-" in bid else ""
    return _PREFIX_KIND.get(prefix.upper(), DEFAULT_KIND)


def classify_block(block: dict) -> set[BlockKind]:
    """Every kind ``block`` carries, including the :func:`primary_kind` one.

    Pure function of the block dict — no database, no config, no clock — so
    the backfill that calls it is replayable. Returns at least one kind;
    a block nothing matches is :data:`DEFAULT_KIND`, which is a real answer
    ("nobody has classified this") rather than an omission.

    Requires the flag, like every other public function here.
    """
    require_enabled(FLAG)
    kinds = {primary_kind(block)}
    if any(isinstance(block.get(f), str) and block.get(f, "").strip() for f in _CODE_FIELDS):
        kinds.add(BlockKind.CODE)
    return kinds


def prune_kind_index(workspace: str | Path, keep_ids: Iterable[str]) -> int:
    """Drop kind rows for ids not in ``keep_ids``. Returns the count removed.

    Without this the index only ever grows, and it grows in the FAIL-OPEN
    direction: a block that was admitted when the backfill ran and has since
    been quarantined keeps its row, its tags and its stored ``content``, so
    ``list_blocks_by_kind`` goes on naming it and ``blocks.content`` goes on
    holding withheld text. Re-running the backfill has to be able to take
    something away, not just add.

    Scoped on purpose. Only rows this package could have written are eligible:
    a ``blocks`` row is removed only if it carries a non-default ``kind`` or a
    tag, so a row some other v4 caller put in the shared side store is left
    alone even though the corpus does not name it.

    Raises :class:`FeatureDisabledError` if the flag is OFF.
    """
    require_enabled(FLAG)
    keep = {str(i) for i in keep_ids}
    db = Path(workspace) / "index.db"
    if not db.is_file():
        return 0
    removed = 0
    with closing(sqlite3.connect(db, timeout=30)) as conn, conn:
        if not _has_kind_column(conn):
            return 0
        owned: set[str] = {r[0] for r in conn.execute("SELECT id FROM blocks WHERE kind IS NOT NULL AND kind != ?", (DEFAULT_KIND.value,))}
        if _table_exists(conn, "block_kind_tags"):
            owned |= {r[0] for r in conn.execute("SELECT DISTINCT block_id FROM block_kind_tags")}
        stale = sorted(owned - keep)
        for bid in stale:
            conn.execute("DELETE FROM blocks WHERE id = ?", (bid,))
            if _table_exists(conn, "block_kind_tags"):
                conn.execute("DELETE FROM block_kind_tags WHERE block_id = ?", (bid,))
            removed += 1
        conn.commit()
    return removed
