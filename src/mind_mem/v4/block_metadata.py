"""v4 block metadata + schema-validation hooks.

Closes two round-4 audit asks at once:

  - DeepSeek's prior-art borrow: ChromaDB-style ``BlockMetadata``
    (key→value tags + TTL + tenant tags) for production-grade data
    management without touching the core recall path.
  - Grok's prior-art borrow: Weaviate-style ``schema validation
    hooks`` so callers can enforce per-kind invariants pre-write.

Two independent surfaces in one module because they share the same
``block_metadata`` table:

    block_metadata(block_id, tags JSON, ttl_seconds, created_at,
                   PRIMARY KEY(block_id))

API:
    set_block_metadata(workspace, block_id, tags, ttl_seconds=None)
    get_block_metadata(workspace, block_id) -> BlockMetadata | None
    delete_block_metadata(workspace, block_id)
    list_blocks_by_tag(workspace, key, value, limit) -> [block_id]

    register_schema_validator(kind, fn)
    validate_block(kind, payload) -> SchemaValidationResult

Schema validators are pure functions — caller provides them; v4 only
wires registration + dispatch. Returning ``ok=True`` allows the
write; ``ok=False`` rejects with the supplied reason. This keeps the
v3 propose/approve flow authoritative for *what* gets written; this
module only adds a programmatic gate before the propose call.

Feature-flag gated under ``v4.block_metadata``.

Copyright STARGA, Inc.
"""

from __future__ import annotations

import datetime as _dt
import json
import sqlite3
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .feature_flags import require_enabled

__all__ = [
    "FLAG",
    "BlockMetadata",
    "SchemaValidationResult",
    "SchemaValidator",
    "ensure_metadata_schema",
    "set_block_metadata",
    "get_block_metadata",
    "delete_block_metadata",
    "list_blocks_by_tag",
    "register_schema_validator",
    "validate_block",
    "available_validators",
]


FLAG: str = "block_metadata"


@dataclass(frozen=True)
class BlockMetadata:
    """Key-value metadata + optional TTL on a block.

    ``created_at`` is stable across upserts; ``updated_at`` advances on
    every :func:`set_block_metadata` call. Audit/governance callers
    that want "first-touch" timestamps read ``created_at``; tools
    sorting by recency read ``updated_at``.
    """

    block_id: str
    tags: dict[str, str]
    ttl_seconds: int | None
    created_at: str
    updated_at: str = ""


@dataclass(frozen=True)
class SchemaValidationResult:
    ok: bool
    reason: str = ""


SchemaValidator = Callable[[dict[str, Any]], SchemaValidationResult]

_validators: dict[str, SchemaValidator] = {}
_validator_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA_SQL: str = """
CREATE TABLE IF NOT EXISTS block_metadata (
    block_id     TEXT PRIMARY KEY,
    tags         TEXT NOT NULL,
    ttl_seconds  INTEGER,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_block_metadata_created
    ON block_metadata (created_at);
CREATE INDEX IF NOT EXISTS idx_block_metadata_updated
    ON block_metadata (updated_at);
"""


# Older databases (pre-round-4) may have the table without ``updated_at``.
# Idempotent migration runs at every ``ensure_metadata_schema`` call; the
# ALTER is wrapped in a try/except since SQLite raises ``OperationalError``
# when the column already exists.
_MIGRATION_SQL: str = "ALTER TABLE block_metadata ADD COLUMN updated_at TEXT NOT NULL DEFAULT ''"


def ensure_metadata_schema(workspace: str | Path) -> None:
    """Idempotent. Creates the ``block_metadata`` table and runs in-place
    migrations for pre-round-4 schemas missing ``updated_at``."""
    require_enabled(FLAG)
    db = Path(workspace) / "index.db"
    if not db.parent.is_dir():
        db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db) as conn:
        conn.executescript(_SCHEMA_SQL)
        try:
            conn.execute(_MIGRATION_SQL)
        except sqlite3.OperationalError:
            # Column already exists — newer schema; nothing to do.
            pass
        conn.commit()


# ---------------------------------------------------------------------------
# Metadata read/write
# ---------------------------------------------------------------------------


def set_block_metadata(
    workspace: str | Path,
    block_id: str,
    tags: dict[str, str] | None = None,
    *,
    ttl_seconds: int | None = None,
) -> BlockMetadata:
    """Upsert metadata for a block. Empty tags + no TTL is a valid
    "block exists, no extras" record (matches Chroma semantics).

    Uses ``INSERT ... ON CONFLICT DO UPDATE`` so ``created_at`` is
    preserved across updates — the original creation time is
    audit-relevant. ``updated_at`` advances on every call.
    """
    require_enabled(FLAG)
    ensure_metadata_schema(workspace)
    safe_tags = {str(k): str(v) for k, v in (tags or {}).items()}
    now = _dt.datetime.now(_dt.timezone.utc).isoformat()
    db = Path(workspace) / "index.db"
    with sqlite3.connect(db) as conn:
        conn.execute(
            "INSERT INTO block_metadata "
            "(block_id, tags, ttl_seconds, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(block_id) DO UPDATE SET "
            "tags = excluded.tags, "
            "ttl_seconds = excluded.ttl_seconds, "
            "updated_at = excluded.updated_at",
            (block_id, json.dumps(safe_tags), ttl_seconds, now, now),
        )
        # Pull the canonical created_at back out so the returned
        # dataclass reflects the persisted (potentially older) timestamp.
        row = conn.execute(
            "SELECT created_at, updated_at FROM block_metadata WHERE block_id = ?",
            (block_id,),
        ).fetchone()
        conn.commit()
    created_at = row[0] if row else now
    updated_at = row[1] if row else now
    return BlockMetadata(
        block_id=block_id,
        tags=safe_tags,
        ttl_seconds=ttl_seconds,
        created_at=created_at,
        updated_at=updated_at,
    )


def get_block_metadata(workspace: str | Path, block_id: str) -> BlockMetadata | None:
    require_enabled(FLAG)
    db = Path(workspace) / "index.db"
    if not db.is_file():
        return None
    with sqlite3.connect(db) as conn:
        if not _table_exists(conn, "block_metadata"):
            return None
        # ``updated_at`` may be missing on pre-round-4 schemas that
        # haven't run the migration yet — fall back to ``created_at``.
        cols = {row[1] for row in conn.execute("PRAGMA table_info(block_metadata)")}
        has_updated = "updated_at" in cols
        select = (
            "SELECT block_id, tags, ttl_seconds, created_at, updated_at FROM block_metadata WHERE block_id = ?"
            if has_updated
            else "SELECT block_id, tags, ttl_seconds, created_at FROM block_metadata WHERE block_id = ?"
        )
        row = conn.execute(select, (block_id,)).fetchone()
    if row is None:
        return None
    try:
        tags = json.loads(row[1]) if row[1] else {}
    except json.JSONDecodeError:
        tags = {}
    updated_at = row[4] if (has_updated and len(row) > 4 and row[4]) else row[3]
    return BlockMetadata(
        block_id=row[0],
        tags=tags if isinstance(tags, dict) else {},
        ttl_seconds=int(row[2]) if row[2] is not None else None,
        created_at=row[3],
        updated_at=updated_at,
    )


def delete_block_metadata(workspace: str | Path, block_id: str) -> bool:
    require_enabled(FLAG)
    db = Path(workspace) / "index.db"
    if not db.is_file():
        return False
    with sqlite3.connect(db) as conn:
        if not _table_exists(conn, "block_metadata"):
            return False
        cursor = conn.execute("DELETE FROM block_metadata WHERE block_id = ?", (block_id,))
        conn.commit()
    return cursor.rowcount > 0


def list_blocks_by_tag(
    workspace: str | Path,
    key: str,
    value: str,
    *,
    limit: int = 100,
) -> list[str]:
    """Return block IDs whose metadata tags contain key=value.

    Implementation is a JSON-extract over the ``tags`` column. Empty
    list when the schema is missing, when ``limit`` is non-positive,
    or when no rows match.
    """
    require_enabled(FLAG)
    if int(limit) <= 0:
        return []
    db = Path(workspace) / "index.db"
    if not db.is_file():
        return []
    with sqlite3.connect(db) as conn:
        if not _table_exists(conn, "block_metadata"):
            return []
        rows = conn.execute(
            "SELECT block_id FROM block_metadata WHERE json_extract(tags, '$.' || ?) = ? LIMIT ?",
            (key, value, int(limit)),
        ).fetchall()
    return [r[0] for r in rows]


# ---------------------------------------------------------------------------
# Schema validators
# ---------------------------------------------------------------------------


def register_schema_validator(kind: str, fn: SchemaValidator) -> None:
    """Register a validator for ``kind``. Replaces any existing one."""
    require_enabled(FLAG)
    with _validator_lock:
        _validators[kind] = fn


def validate_block(kind: str, payload: dict[str, Any]) -> SchemaValidationResult:
    """Run the validator for ``kind`` against ``payload``.

    Returns ``ok=True`` when no validator is registered (open by
    default — callers register validators only for kinds they want to
    constrain). Validator exceptions are caught and reported as
    ``ok=False`` so the recall path can continue cleanly.
    """
    require_enabled(FLAG)
    with _validator_lock:
        fn = _validators.get(kind)
    if fn is None:
        return SchemaValidationResult(ok=True, reason="no_validator")
    try:
        result = fn(payload)
    except Exception as exc:
        return SchemaValidationResult(ok=False, reason=f"validator_raised: {exc!r}")
    if not isinstance(result, SchemaValidationResult):
        return SchemaValidationResult(ok=False, reason="validator_returned_wrong_type")
    return result


def available_validators() -> list[str]:
    """Return every registered kind."""
    require_enabled(FLAG)
    with _validator_lock:
        return list(_validators.keys())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?",
        (name,),
    ).fetchone()
    return row is not None
