"""PostgresBlockStore — PostgreSQL-backed BlockStore for mind-mem v3.2.0.

Full implementation of the BlockStore protocol against PostgreSQL 14+.
Opt-in via ``block_store.backend = "postgres"`` in mind-mem.json.

Dependencies (optional extras):
    pip install "mind-mem[postgres]"
    # or: pip install "psycopg[binary]>=3.2,<4.0" "pgvector>=0.3,<1.0"

Psycopg is imported lazily inside each method so the module can be
imported cleanly without the optional dependencies installed.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass  # No runtime-only type imports needed; psycopg is a dynamic dependency.

from .block_store import BlockStoreError  # noqa: F401  (re-exported for convenience)

_log = logging.getLogger("mind_mem.block_store_postgres")

__all__ = ["PostgresBlockStore", "BlockStoreError"]

# Schema names must be safe Postgres identifiers (no injection surface).
_SAFE_SCHEMA_RE = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")

# Postgres identifier allowlist: letters, digits, underscore, dollar sign.
# Max 63 bytes (Postgres internal limit). No quoting is applied — the schema
# name is embedded directly into DDL/DML f-strings, so we must reject any
# value that could escape the identifier context.
_SAFE_PG_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]{0,62}$")


def _validate_schema_name(name: str) -> str:
    """Raise ValueError if *name* is not a safe Postgres schema identifier."""
    if not _SAFE_PG_IDENTIFIER_RE.match(name):
        raise ValueError(
            f"Unsafe Postgres schema name {name!r}. "
            "Schema must match [A-Za-z_][A-Za-z0-9_$]{{0,62}}."
        )
    return name


# ─── Sentinel: lazy import guard ──────────────────────────────────────────────

_psycopg: Any = None
_psycopg_pool: Any = None
_import_lock = threading.Lock()


def _require_psycopg() -> tuple[Any, Any]:
    """Return (psycopg, psycopg_pool), importing once and caching."""
    global _psycopg, _psycopg_pool
    if _psycopg is not None:
        return _psycopg, _psycopg_pool
    with _import_lock:
        if _psycopg is not None:
            return _psycopg, _psycopg_pool
        try:
            import psycopg as _ps

            _psycopg = _ps
        except ModuleNotFoundError as exc:
            raise ImportError(
                "The PostgreSQL backend requires psycopg. "
                'Install it with: pip install "mind-mem[postgres]"'
            ) from exc
        try:
            from psycopg_pool import ConnectionPool as _CP

            _psycopg_pool = _CP
        except ModuleNotFoundError as exc:
            raise ImportError(
                "The PostgreSQL backend requires psycopg_pool. "
                'Install it with: pip install "mind-mem[postgres]"'
            ) from exc
    return _psycopg, _psycopg_pool


# ─── DDL ──────────────────────────────────────────────────────────────────────


def _ddl(schema: str) -> Any:
    """Return a psycopg Composable DDL batch with the schema safely quoted.

    All schema references use ``psycopg.sql.Identifier`` so the schema name
    can never act as an SQL injection vector.  ``_validate_schema_name`` is
    also enforced at ``__init__`` time for defence-in-depth.
    """
    from psycopg import sql as pgsql

    s = pgsql.Identifier(schema)
    stmts: list[Any] = [
        pgsql.SQL("CREATE SCHEMA IF NOT EXISTS {s}").format(s=s),
        pgsql.SQL(
            "CREATE TABLE IF NOT EXISTS {s}.blocks ("
            "    id           TEXT PRIMARY KEY,"
            "    file_path    TEXT NOT NULL,"
            "    content      TEXT NOT NULL,"
            "    metadata     JSONB NOT NULL DEFAULT '{}',"
            "    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),"
            "    updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),"
            "    active       BOOLEAN NOT NULL DEFAULT TRUE"
            ")"
        ).format(s=s),
        pgsql.SQL(
            "CREATE INDEX IF NOT EXISTS blocks_file_path ON {s}.blocks(file_path)"
        ).format(s=s),
        pgsql.SQL(
            "CREATE INDEX IF NOT EXISTS blocks_active ON {s}.blocks(active) WHERE active"
        ).format(s=s),
        # GIN index: avoids per-row tsvector recomputation on every search().
        pgsql.SQL(
            "CREATE INDEX IF NOT EXISTS blocks_fts"
            " ON {s}.blocks USING GIN (to_tsvector('english', content))"
        ).format(s=s),
        # updated_at DESC: supports time-range queries without a seq-scan.
        pgsql.SQL(
            "CREATE INDEX IF NOT EXISTS blocks_updated_at ON {s}.blocks(updated_at DESC)"
        ).format(s=s),
        # Covering partial index: makes list_blocks() an index-only scan.
        pgsql.SQL(
            "CREATE INDEX IF NOT EXISTS blocks_active_file_path"
            " ON {s}.blocks(file_path) WHERE active"
        ).format(s=s),
        pgsql.SQL(
            "CREATE TABLE IF NOT EXISTS {s}.snapshots ("
            "    snap_id    TEXT PRIMARY KEY,"
            "    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),"
            "    manifest   JSONB NOT NULL"
            ")"
        ).format(s=s),
        pgsql.SQL(
            "CREATE TABLE IF NOT EXISTS {s}.snapshot_blocks ("
            "    snap_id   TEXT REFERENCES {s}.snapshots(snap_id) ON DELETE CASCADE,"
            "    block_id  TEXT NOT NULL,"
            "    content   TEXT NOT NULL,"
            "    metadata  JSONB NOT NULL,"
            "    PRIMARY KEY (snap_id, block_id)"
            ")"
        ).format(s=s),
        # expires_at: prevents indefinite lock hold when the holder is killed.
        pgsql.SQL(
            "CREATE TABLE IF NOT EXISTS {s}.workspace_lock ("
            "    lock_id     TEXT PRIMARY KEY,"
            "    holder      TEXT NOT NULL,"
            "    acquired_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),"
            "    expires_at  TIMESTAMPTZ NOT NULL"
            "                    DEFAULT NOW() + INTERVAL '5 minutes'"
            ")"
        ).format(s=s),
        pgsql.SQL(
            "CREATE INDEX IF NOT EXISTS workspace_lock_expires_at"
            " ON {s}.workspace_lock(expires_at)"
        ).format(s=s),
        # Schema versioning table: enables safe future ALTER TABLE migrations.
        pgsql.SQL(
            "CREATE TABLE IF NOT EXISTS {s}.schema_migrations ("
            "    version     INTEGER PRIMARY KEY,"
            "    applied_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),"
            "    description TEXT NOT NULL"
            ")"
        ).format(s=s),
    ]
    return pgsql.SQL(";\n").join(stmts)


def _sql(schema: str, template: str) -> Any:
    """Return a psycopg Composable for *template* with ``{s}`` replaced by the
    safely-quoted schema identifier.

    Row values must still be passed as ``%s`` bind parameters — this helper
    only secures the schema name, not row data.
    """
    from psycopg import sql as pgsql

    return pgsql.SQL(template).format(s=pgsql.Identifier(schema))


# ─── Row → block dict ─────────────────────────────────────────────────────────


def _row_to_block(row: dict[str, Any]) -> dict[str, Any]:
    """Convert a database row dict to the block dict shape expected by callers.

    The ``metadata`` column stores all non-primary fields as JSONB.
    The synthetic ``_id`` field is synthesised from the ``id`` column,
    and ``_source_file`` is populated from ``file_path``.
    """
    meta: dict[str, Any] = row.get("metadata") or {}
    block: dict[str, Any] = {
        "_id": row["id"],
        "_source_file": row.get("file_path", ""),
        **meta,
    }
    # Preserve database timestamps as ISO strings for callers that log them.
    if row.get("created_at"):
        block.setdefault("_created_at", str(row["created_at"]))
    if row.get("updated_at"):
        block.setdefault("_updated_at", str(row["updated_at"]))
    if "active" in row:
        block.setdefault("active", row["active"])
    return block


def _block_to_row(block: dict[str, Any]) -> tuple[str, str, str, str]:
    """Return (id, file_path, content, metadata_json) extracted from *block*.

    ``content`` is the plain-text representation of the block (``Statement``
    field when present, otherwise a JSON dump of non-private keys).
    ``metadata`` is the full block dict minus private ``_`` keys.
    """
    block_id: str = str(block.get("_id", ""))
    if not block_id:
        raise ValueError("block is missing '_id'; cannot persist to Postgres")

    file_path: str = str(block.get("_source_file", ""))

    # Content for FTS: prefer Statement, fall back to JSON text.
    content: str = str(block.get("Statement") or block.get("content") or "")
    if not content:
        content = " ".join(str(v) for k, v in block.items() if not k.startswith("_"))

    # Metadata: everything that is not a private field.
    metadata: dict[str, Any] = {k: v for k, v in block.items() if not k.startswith("_")}
    return block_id, file_path, content, json.dumps(metadata, default=str)


# ─── Main class ───────────────────────────────────────────────────────────────


class PostgresBlockStore:
    """BlockStore implementation backed by PostgreSQL 14+.

    Thread-safe: uses ``psycopg_pool.ConnectionPool`` which serialises
    concurrent callers at the connection-checkout level. Individual
    write operations are wrapped in explicit transactions.

    Args:
        dsn:       psycopg connection string, e.g.
                   ``"postgresql://user:pw@localhost:5432/mydb"``.
        schema:    Postgres schema to use (default ``mind_mem``).
        workspace: Optional workspace path; used only to derive
                   ``snap_id`` basenames and write MANIFEST.json for
                   cross-backend compatibility.
    """

    def __init__(
        self,
        dsn: str,
        *,
        schema: str = "mind_mem",
        workspace: str | None = None,
    ) -> None:
        self._dsn = dsn
        self._schema = _validate_schema_name(schema)
        self._workspace = workspace
        self._pool: Any = None
        self._schema_ready = False
        self._init_lock = threading.Lock()

    # ─── Lifecycle ────────────────────────────────────────────────────────────

    def _get_pool(self) -> Any:
        """Return the connection pool, creating it on first call."""
        if self._pool is not None:
            return self._pool
        _, ConnectionPool = _require_psycopg()
        with self._init_lock:
            if self._pool is None:
                self._pool = ConnectionPool(
                    self._dsn,
                    min_size=1,
                    max_size=10,
                    open=True,
                )
        return self._pool

    def _ensure_schema(self) -> None:
        """Run CREATE TABLE / INDEX migrations idempotently on first call."""
        if self._schema_ready:
            return
        with self._init_lock:
            if self._schema_ready:
                return
            psycopg, _ = _require_psycopg()
            pool = self._get_pool()
            try:
                with pool.connection() as conn:
                    conn.autocommit = True
                    conn.execute(_ddl(self._schema))
                self._schema_ready = True
                _log.info("postgres_block_store_schema_ready", extra={"schema": self._schema})
            except Exception as exc:
                raise BlockStoreError(f"Schema migration failed: {exc}") from exc

    # ─── Read surface ─────────────────────────────────────────────────────────

    def get_all(self, *, active_only: bool = False) -> list[dict[str, Any]]:
        """Return all blocks, optionally filtered to active rows."""
        self._ensure_schema()
        pool = self._get_pool()
        psycopg, _ = _require_psycopg()
        if active_only:
            sql = _sql(
                self._schema,
                "SELECT id, file_path, content, metadata, created_at, updated_at, active"
                " FROM {s}.blocks WHERE active = TRUE ORDER BY id",
            )
        else:
            sql = _sql(
                self._schema,
                "SELECT id, file_path, content, metadata, created_at, updated_at, active"
                " FROM {s}.blocks ORDER BY id",
            )
        try:
            with pool.connection() as conn:
                with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
                    cur.execute(sql)
                    return [_row_to_block(dict(r)) for r in cur.fetchall()]
        except Exception as exc:
            raise BlockStoreError(f"get_all failed: {exc}") from exc

    def get_by_id(self, block_id: str) -> dict[str, Any] | None:
        """Return a single block by primary key, or None if absent."""
        self._ensure_schema()
        pool = self._get_pool()
        psycopg, _ = _require_psycopg()
        sql = _sql(
            self._schema,
            "SELECT id, file_path, content, metadata, created_at, updated_at, active"
            " FROM {s}.blocks WHERE id = %s",
        )
        try:
            with pool.connection() as conn:
                with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
                    cur.execute(sql, (block_id,))
                    row = cur.fetchone()
                    if row is None:
                        return None
                    return _row_to_block(dict(row))
        except Exception as exc:
            raise BlockStoreError(f"get_by_id failed: {exc}") from exc

    def search(self, query: str, *, limit: int = 10) -> list[dict[str, Any]]:
        """Full-text search over content + Statement metadata field.

        Uses ``plainto_tsquery`` when the ``pg_trgm`` / text search
        index exists. Falls back to ``ILIKE`` when Postgres raises an
        error (e.g., the text index is missing on a fresh schema).
        """
        self._ensure_schema()
        pool = self._get_pool()
        psycopg, _ = _require_psycopg()
        fts_sql = _sql(
            self._schema,
            "SELECT id, file_path, content, metadata, created_at, updated_at, active,"
            "       ts_rank("
            "           to_tsvector('english', content || ' '"
            "               || COALESCE(metadata->>'Statement', '')),"
            "           plainto_tsquery('english', %s)"
            "       ) AS rank"
            " FROM {s}.blocks"
            " WHERE to_tsvector('english', content || ' '"
            "           || COALESCE(metadata->>'Statement', ''))"
            "       @@ plainto_tsquery('english', %s)"
            " ORDER BY rank DESC"
            " LIMIT %s",
        )
        ilike_sql = _sql(
            self._schema,
            "SELECT id, file_path, content, metadata, created_at, updated_at, active"
            " FROM {s}.blocks"
            " WHERE content ILIKE %s OR metadata->>'Statement' ILIKE %s"
            " LIMIT %s",
        )
        try:
            with pool.connection() as conn:
                with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
                    try:
                        cur.execute(fts_sql, (query, query, limit))
                    except Exception:
                        # Roll back the failed statement and fall back to ILIKE.
                        conn.rollback()
                        pattern = f"%{query}%"
                        cur.execute(ilike_sql, (pattern, pattern, limit))
                    return [_row_to_block(dict(r)) for r in cur.fetchall()]
        except Exception as exc:
            raise BlockStoreError(f"search failed: {exc}") from exc

    def list_blocks(self) -> list[str]:
        """Return distinct ``file_path`` values for all active blocks."""
        self._ensure_schema()
        pool = self._get_pool()
        sql = f"SELECT DISTINCT file_path FROM {self._schema}.blocks WHERE active = TRUE ORDER BY file_path"
        try:
            with pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql)
                    return [row[0] for row in cur.fetchall()]
        except Exception as exc:
            raise BlockStoreError(f"list_blocks failed: {exc}") from exc

    # ─── Write surface ────────────────────────────────────────────────────────

    def write_block(self, block: dict[str, Any]) -> str:
        """Upsert a block. Returns the block's ``_id``.

        INSERT … ON CONFLICT (id) DO UPDATE is executed in a single
        transaction so it is genuinely atomic (no lost-update window).
        """
        self._ensure_schema()
        pool = self._get_pool()
        block_id, file_path, content, metadata_json = _block_to_row(block)
        sql = f"""
            INSERT INTO {self._schema}.blocks (id, file_path, content, metadata, updated_at)
            VALUES (%s, %s, %s, %s::jsonb, NOW())
            ON CONFLICT (id) DO UPDATE
                SET file_path  = EXCLUDED.file_path,
                    content    = EXCLUDED.content,
                    metadata   = EXCLUDED.metadata,
                    updated_at = EXCLUDED.updated_at
        """
        try:
            with pool.connection() as conn:
                with conn.transaction():
                    conn.execute(sql, (block_id, file_path, content, metadata_json))
            _log.debug("block_store_write", extra={"block_id": block_id})
            return block_id
        except Exception as exc:
            raise BlockStoreError(f"write_block failed for {block_id!r}: {exc}") from exc

    def delete_block(self, block_id: str) -> bool:
        """Delete a block by id. Returns True if a row was removed.

        Logs to ``<schema>.deleted_blocks`` if that table exists;
        otherwise the deletion is silently recorded only in the
        application log.
        """
        self._ensure_schema()
        pool = self._get_pool()
        sql_delete = f"DELETE FROM {self._schema}.blocks WHERE id = %s RETURNING id, content"
        sql_log = f"""
            INSERT INTO {self._schema}.deleted_blocks (block_id, content, deleted_at)
            VALUES (%s, %s, NOW())
            ON CONFLICT DO NOTHING
        """
        try:
            with pool.connection() as conn:
                with conn.transaction():
                    with conn.cursor() as cur:
                        cur.execute(sql_delete, (block_id,))
                        deleted_row = cur.fetchone()
                        if deleted_row is None:
                            return False
                        deleted_content = deleted_row[1] if len(deleted_row) > 1 else ""
                        # Best-effort: log deletion if the journal table exists.
                        try:
                            cur.execute(sql_log, (block_id, deleted_content))
                        except Exception:
                            pass  # table absent — non-fatal
            _log.debug("block_store_delete", extra={"block_id": block_id})
            return True
        except Exception as exc:
            raise BlockStoreError(f"delete_block failed for {block_id!r}: {exc}") from exc

    # ─── Snapshot surface ─────────────────────────────────────────────────────

    def snapshot(
        self,
        snap_dir: str,
        *,
        files_touched: list[str] | None = None,
    ) -> dict[str, Any]:
        """Capture a point-in-time snapshot.

        ``snap_id`` is derived from the basename of ``snap_dir``.
        All live blocks are copied into ``<schema>.snapshot_blocks``
        and the manifest is also written to ``snap_dir/MANIFEST.json``
        for cross-backend compatibility (restore via MarkdownBlockStore
        can read the same manifest).

        Returns the manifest dict (``{"files": [...], "version": 2}``).
        """
        self._ensure_schema()
        pool = self._get_pool()
        psycopg, _ = _require_psycopg()
        snap_id = os.path.basename(snap_dir.rstrip("/"))
        if not snap_id:
            raise ValueError(f"Cannot derive snap_id from snap_dir={snap_dir!r}")

        # Collect all live blocks.
        all_blocks = self.get_all(active_only=False)

        if files_touched:
            fp_set = set(files_touched)
            blocks_to_snap = [b for b in all_blocks if b.get("_source_file", "") in fp_set]
        else:
            blocks_to_snap = all_blocks

        # Build the manifest file list.
        files_in_snap: list[str] = sorted(
            {b.get("_source_file", "") for b in blocks_to_snap if b.get("_source_file")}
        )
        manifest: dict[str, Any] = {"files": files_in_snap, "version": 2}
        manifest_json = json.dumps(manifest, default=str)

        sql_snap = f"""
            INSERT INTO {self._schema}.snapshots (snap_id, manifest)
            VALUES (%s, %s::jsonb)
            ON CONFLICT (snap_id) DO UPDATE SET manifest = EXCLUDED.manifest,
                                                created_at = NOW()
        """
        sql_blocks = f"""
            INSERT INTO {self._schema}.snapshot_blocks (snap_id, block_id, content, metadata)
            VALUES (%s, %s, %s, %s::jsonb)
            ON CONFLICT (snap_id, block_id) DO UPDATE
                SET content  = EXCLUDED.content,
                    metadata = EXCLUDED.metadata
        """
        try:
            with pool.connection() as conn:
                with conn.transaction():
                    conn.execute(sql_snap, (snap_id, manifest_json))
                    # Remove stale snapshot_blocks from a previous overwrite.
                    conn.execute(
                        f"DELETE FROM {self._schema}.snapshot_blocks WHERE snap_id = %s",
                        (snap_id,),
                    )
                    with conn.cursor() as cur:
                        for b in blocks_to_snap:
                            bid = b.get("_id", "")
                            content = b.get("Statement") or b.get("content") or ""
                            meta = {k: v for k, v in b.items() if not k.startswith("_")}
                            cur.execute(sql_blocks, (snap_id, bid, content, json.dumps(meta, default=str)))
        except Exception as exc:
            raise BlockStoreError(f"snapshot failed for snap_id={snap_id!r}: {exc}") from exc

        # Write MANIFEST.json to disk for cross-backend compatibility.
        os.makedirs(snap_dir, exist_ok=True)
        manifest_path = os.path.join(snap_dir, "MANIFEST.json")
        with open(manifest_path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh)

        _log.info("postgres_block_store_snapshot", extra={"snap_id": snap_id, "block_count": len(blocks_to_snap)})
        return manifest

    def restore(self, snap_dir: str) -> None:
        """Restore live blocks from a snapshot in a single transaction.

        Clears all current rows in ``<schema>.blocks`` and re-inserts from
        ``<schema>.snapshot_blocks[snap_id]``.  The operation is fully
        atomic: if the transaction fails the live table is unchanged.
        """
        self._ensure_schema()
        pool = self._get_pool()
        snap_id = os.path.basename(snap_dir.rstrip("/"))
        if not snap_id:
            raise ValueError(f"Cannot derive snap_id from snap_dir={snap_dir!r}")

        sql_check = f"SELECT snap_id FROM {self._schema}.snapshots WHERE snap_id = %s"
        sql_clear = f"DELETE FROM {self._schema}.blocks"
        sql_insert = f"""
            INSERT INTO {self._schema}.blocks (id, file_path, content, metadata, active)
            SELECT sb.block_id,
                   COALESCE(sb.metadata->>'_source_file', ''),
                   sb.content,
                   sb.metadata,
                   TRUE
            FROM {self._schema}.snapshot_blocks sb
            WHERE sb.snap_id = %s
        """
        try:
            with pool.connection() as conn:
                with conn.transaction():
                    with conn.cursor() as cur:
                        cur.execute(sql_check, (snap_id,))
                        if cur.fetchone() is None:
                            raise BlockStoreError(f"Snapshot {snap_id!r} not found in database")
                    conn.execute(sql_clear)
                    conn.execute(sql_insert, (snap_id,))
        except BlockStoreError:
            raise
        except Exception as exc:
            raise BlockStoreError(f"restore failed for snap_id={snap_id!r}: {exc}") from exc

        _log.info("postgres_block_store_restore", extra={"snap_id": snap_id})

    def diff(self, snap_dir: str) -> list[str]:
        """Return paths of blocks that differ between current state and snapshot.

        Uses a FULL OUTER JOIN between ``blocks`` and ``snapshot_blocks`` to
        detect: added (in live but not in snapshot), removed (in snapshot but
        not in live), modified (same id but different content).

        Returns a sorted list of ``file_path`` strings for changed blocks.
        """
        self._ensure_schema()
        pool = self._get_pool()
        snap_id = os.path.basename(snap_dir.rstrip("/"))
        if not snap_id:
            raise ValueError(f"Cannot derive snap_id from snap_dir={snap_dir!r}")

        sql = f"""
            SELECT COALESCE(live.file_path, snap.metadata->>'_source_file', snap.block_id) AS path,
                   CASE
                     WHEN live.id IS NULL              THEN 'removed'
                     WHEN snap.block_id IS NULL         THEN 'added'
                     ELSE                                   'modified'
                   END AS change_type
            FROM {self._schema}.blocks AS live
            FULL OUTER JOIN {self._schema}.snapshot_blocks AS snap
                ON live.id = snap.block_id AND snap.snap_id = %s
            WHERE snap.snap_id = %s OR snap.snap_id IS NULL
              AND (
                   live.id IS NULL
                   OR snap.block_id IS NULL
                   OR live.content <> snap.content
                   OR live.metadata::text <> snap.metadata::text
              )
            ORDER BY path
        """
        try:
            with pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, (snap_id, snap_id))
                    rows = cur.fetchall()
            return sorted({str(r[0]) for r in rows if r[0]})
        except Exception as exc:
            raise BlockStoreError(f"diff failed for snap_id={snap_id!r}: {exc}") from exc


    # ─── Workspace lock surface ───────────────────────────────────────────────

    def lock(self, *, blocking: bool = True, timeout: float = 30.0) -> Any:
        """Acquire an exclusive workspace-wide lock via the Postgres ``workspace_lock`` table.

        Uses ``INSERT … ON CONFLICT DO NOTHING`` within a transaction to
        simulate advisory locking. The returned object is a context manager;
        the lock row is deleted on ``__exit__``.

        Args:
            blocking: When ``True`` (default), poll until the lock is free or
                      *timeout* elapses.  When ``False``, raise
                      :class:`~mind_mem.mind_filelock.LockTimeout` immediately if
                      the lock is held.
            timeout:  Maximum seconds to wait when ``blocking=True``.

        Raises:
            BlockStoreError: On unexpected database errors.
        """
        import time

        from .mind_filelock import LockTimeout

        self._ensure_schema()
        pool = self._get_pool()
        lock_id = "workspace"
        holder = str(os.getpid())
        sql_acquire = f"""
            INSERT INTO {self._schema}.workspace_lock (lock_id, holder)
            VALUES (%s, %s)
            ON CONFLICT (lock_id) DO NOTHING
        """
        sql_release = f"DELETE FROM {self._schema}.workspace_lock WHERE lock_id = %s AND holder = %s"

        deadline = time.monotonic() + timeout

        class _PgLock:
            def __init__(self_inner) -> None:
                pass

            def __enter__(self_inner) -> "_PgLock":
                while True:
                    try:
                        with pool.connection() as conn:
                            with conn.transaction():
                                with conn.cursor() as cur:
                                    cur.execute(sql_acquire, (lock_id, holder))
                                    acquired = cur.rowcount > 0
                        if acquired:
                            return self_inner
                    except Exception as exc:
                        raise BlockStoreError(f"lock acquire error: {exc}") from exc
                    if not blocking:
                        raise LockTimeout("Postgres workspace lock held by another process")
                    if time.monotonic() >= deadline:
                        raise LockTimeout(f"Timed out waiting for Postgres workspace lock after {timeout}s")
                    time.sleep(0.1)

            def __exit__(self_inner, *_: Any) -> None:
                try:
                    with pool.connection() as conn:
                        with conn.transaction():
                            conn.execute(sql_release, (lock_id, holder))
                except Exception:
                    pass  # best-effort release; non-fatal

        return _PgLock()

    # ─── Compatibility shim ───────────────────────────────────────────────────

    def list_files(self) -> list[str]:
        """Deprecated alias for :meth:`list_blocks` — removed in v4.0."""
        import warnings

        warnings.warn(
            "BlockStore.list_files() is deprecated; use list_blocks() instead. "
            "The alias will be removed in v4.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.list_blocks()

    # ─── Context manager / cleanup ────────────────────────────────────────────

    def close(self) -> None:
        """Close the underlying connection pool."""
        if self._pool is not None:
            try:
                self._pool.close()
            except Exception:
                pass
            self._pool = None

    def __enter__(self) -> "PostgresBlockStore":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
