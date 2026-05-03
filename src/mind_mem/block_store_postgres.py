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
        raise ValueError(f"Unsafe Postgres schema name {name!r}. Schema must match [A-Za-z_][A-Za-z0-9_$]{{{{0,62}}}}.")
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
            raise ImportError('The PostgreSQL backend requires psycopg. Install it with: pip install "mind-mem[postgres]"') from exc
        try:
            from psycopg_pool import ConnectionPool as _CP

            _psycopg_pool = _CP
        except ModuleNotFoundError as exc:
            raise ImportError('The PostgreSQL backend requires psycopg_pool. Install it with: pip install "mind-mem[postgres]"') from exc
    return _psycopg, _psycopg_pool


# ─── DDL ──────────────────────────────────────────────────────────────────────


DEFAULT_EMBEDDING_DIM = 1024  # mxbai-embed-large native dim; matches workspace default.


def _ddl(schema: str) -> Any:
    """Return a psycopg Composable DDL batch with the schema safely quoted.

    All schema references use ``psycopg.sql.Identifier`` so the schema name
    can never act as an SQL injection vector.  ``_validate_schema_name`` is
    also enforced at ``__init__`` time for defence-in-depth.

    Note: this DDL only creates the base schema. The pgvector embedding
    column + IVFFlat index are added by ``_ddl_pgvector()`` after a
    server-side capability check, so deployments without pgvector still
    get a working BM25-only schema.
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
            # '{{}}' escapes psycopg's positional {} placeholder so
            # the JSONB literal '{}' reaches the server unmangled.
            "    metadata     JSONB NOT NULL DEFAULT '{{}}',"
            "    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),"
            "    updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),"
            "    active       BOOLEAN NOT NULL DEFAULT TRUE"
            ")"
        ).format(s=s),
        pgsql.SQL("CREATE INDEX IF NOT EXISTS blocks_file_path ON {s}.blocks(file_path)").format(s=s),
        pgsql.SQL("CREATE INDEX IF NOT EXISTS blocks_active ON {s}.blocks(active) WHERE active").format(s=s),
        # GIN index over the SAME tsvector expression that search() and
        # hybrid_search() use in their WHERE clauses — content + the
        # Statement field from JSONB metadata. Without character-for-
        # character match the planner falls back to a sequential scan
        # with per-row tsvector recomputation. v3.8.13 had a
        # ``to_tsvector('english', content)`` index that never matched.
        pgsql.SQL(
            "CREATE INDEX IF NOT EXISTS blocks_fts ON {s}.blocks USING GIN ("
            "to_tsvector('english', content || ' ' || COALESCE(metadata->>'Statement', ''))"
            ")"
        ).format(s=s),
        # updated_at DESC: supports time-range queries without a seq-scan.
        pgsql.SQL("CREATE INDEX IF NOT EXISTS blocks_updated_at ON {s}.blocks(updated_at DESC)").format(s=s),
        # Covering partial index: makes list_blocks() an index-only scan.
        pgsql.SQL("CREATE INDEX IF NOT EXISTS blocks_active_file_path ON {s}.blocks(file_path) WHERE active").format(s=s),
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
        pgsql.SQL("CREATE INDEX IF NOT EXISTS workspace_lock_expires_at ON {s}.workspace_lock(expires_at)").format(s=s),
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


def _try_create_extension_vector(conn: Any) -> bool:
    """Attempt ``CREATE EXTENSION IF NOT EXISTS vector``. Returns True on
    success, False when the extension is unavailable on the server. The
    failure path leaves the transaction in a clean state for the caller.
    """
    try:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        return True
    except Exception:
        # Roll back the failed statement so the next exec inherits a
        # clean transaction state. Conn is autocommit in our caller so
        # this is a no-op there but defensive in case that changes.
        try:
            conn.rollback()
        except Exception:
            pass
        return False


def _ddl_pgvector(schema: str, embedding_dim: int = DEFAULT_EMBEDDING_DIM) -> Any:
    """Return DDL that adds the embedding column + HNSW cosine index.

    Run this only after :func:`_try_create_extension_vector` confirmed
    pgvector is available on the server. The ALTER TABLE is idempotent
    (``ADD COLUMN IF NOT EXISTS``) so re-runs are safe.

    Index choice: HNSW (pgvector >= 0.5.0). HNSW builds incrementally
    on insert so the index quality is independent of when we run the
    DDL relative to row population — fixing the IVFFlat-on-empty-table
    bug v3.8.13 had where centroids were built before any rows existed.
    HNSW is also more forgiving on small corpora than IVFFlat.
    Defaults ``m=16, ef_construction=64`` are pgvector's documented
    starting point.

    The integer dimension is rendered through ``pgsql.Literal`` so the
    SQL composition stays inside the Composable API; we never glue
    user-controlled values into raw SQL text.
    """
    from psycopg import sql as pgsql

    s = pgsql.Identifier(schema)
    dim_lit = pgsql.Literal(int(embedding_dim))
    stmts: list[Any] = [
        pgsql.SQL("ALTER TABLE {s}.blocks ADD COLUMN IF NOT EXISTS embedding VECTOR({d})").format(s=s, d=dim_lit),
        # HNSW with cosine. lists/m tuning replaces the brittle
        # IVFFlat lists=100 default that mis-bucketed any corpus
        # smaller than ~100k rows.
        pgsql.SQL(
            "CREATE INDEX IF NOT EXISTS blocks_embedding "
            "ON {s}.blocks USING hnsw (embedding vector_cosine_ops) "
            "WITH (m=16, ef_construction=64)"
        ).format(s=s),
    ]
    return pgsql.SQL(";\n").join(stmts)


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


def _embedding_to_pg(embedding: list[float]) -> str:
    """Render a Python float list as the pgvector text literal ``[a,b,...]``.

    pgvector accepts ``'[1,2,3]'::vector`` and parses without requiring
    the pgvector psycopg adapter to be installed — keeps the dependency
    surface to plain ``psycopg``.

    Rejects NaN/Inf at the boundary. Postgres + pgvector silently accept
    ``'[nan,inf]'::vector`` but every cosine distance involving such a
    row returns NaN, which silently poisons RRF ranking in
    :meth:`PostgresBlockStore.hybrid_search`. Catching here is cheaper
    than chasing NaN-tainted recall scores later.
    """
    import math

    for x in embedding:
        if not math.isfinite(float(x)):
            raise ValueError(f"non-finite value in embedding: {x!r}")
    return "[" + ",".join(format(float(x), ".7g") for x in embedding) + "]"


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
        # RLock so _ensure_schema() can call _get_pool() while still
        # holding this lock without self-deadlocking. Both methods
        # acquire it for the same purpose (one-time init), and the
        # cost of re-entrance is zero compared to the cost of the
        # bug it fixes (any first call from a single thread hangs).
        self._init_lock = threading.RLock()
        # v3.8.13: pgvector wiring. Default dim matches mxbai-embed-large
        # (1024). If the deployment uses a different embedder, set this
        # before _ensure_schema() runs (constructor arg or attr write).
        self._embedding_dim: int = DEFAULT_EMBEDDING_DIM
        self._has_vector: bool = False

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
        """Run CREATE TABLE / INDEX migrations idempotently on first call.

        v3.8.13: also probes for pgvector. Sets ``self._has_vector`` so
        downstream paths (write_block, hybrid_search) can route on it.
        """
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
                    self._has_vector = _try_create_extension_vector(conn)
                    if self._has_vector:
                        try:
                            conn.execute(_ddl_pgvector(self._schema, self._embedding_dim))
                        except Exception as vec_exc:
                            # Vector ext present but DDL failed (e.g. dim
                            # mismatch with an existing column). Don't
                            # fail the whole migration; degrade to
                            # BM25-only and surface in the log.
                            self._has_vector = False
                            _log.warning(
                                "postgres_pgvector_ddl_failed",
                                extra={"schema": self._schema, "error": str(vec_exc)[:200]},
                            )
                self._schema_ready = True
                _log.info(
                    "postgres_block_store_schema_ready",
                    extra={"schema": self._schema, "has_vector": self._has_vector},
                )
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
                "SELECT id, file_path, content, metadata, created_at, updated_at, active FROM {s}.blocks WHERE active = TRUE ORDER BY id",
            )
        else:
            sql = _sql(
                self._schema,
                "SELECT id, file_path, content, metadata, created_at, updated_at, active FROM {s}.blocks ORDER BY id",
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
            "SELECT id, file_path, content, metadata, created_at, updated_at, active FROM {s}.blocks WHERE id = %s",
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

    def hybrid_search(
        self,
        query: str,
        *,
        query_embedding: list[float] | None = None,
        limit: int = 10,
        rrf_k: int = 60,
        candidate_pool: int = 100,
    ) -> list[dict[str, Any]]:
        """Hybrid BM25 + cosine recall fused with reciprocal rank fusion.

        v3.8.13 — runs both halves server-side in a single CTE batch:
          * BM25-style: ``ts_rank`` over the indexed tsvector.
          * Vector:     ``embedding <=> query_embedding`` cosine distance.

        Each half retrieves the top-``candidate_pool`` rows; ranks are
        fused per-id with RRF (``1 / (rrf_k + rank)``). Returns the
        ``limit`` highest-scored merged rows.

        Falls back to BM25-only when:
          * pgvector is not installed on the server, OR
          * ``query_embedding`` is None, OR
          * the column is empty (every row has NULL embedding — no
            backfill yet).

        Search semantics intentionally match the canonical mind-mem
        retrieval pipeline so callers can swap backends without
        re-tuning weights.
        """
        self._ensure_schema()
        psycopg, _ = _require_psycopg()
        pool = self._get_pool()

        # Bound caller-controlled values so a runaway agent can't trigger
        # an OOM via huge candidate_pool / limit values. The caps are
        # generous (5x default) but finite. Enforced before SQL.
        limit = min(int(limit), 200)
        candidate_pool = min(int(candidate_pool), 500)

        # Treat None and empty list distinctly: empty-list is a caller bug
        # (likely an embedder regression) and should fail loudly rather
        # than silently degrade to BM25-only.
        if query_embedding is not None and len(query_embedding) != self._embedding_dim:
            raise BlockStoreError(f"query_embedding dim mismatch: got {len(query_embedding)}, schema expects {self._embedding_dim}")
        do_vector = query_embedding is not None and self._has_vector

        bm25_sql = _sql(
            self._schema,
            "SELECT id, ts_rank("
            "    to_tsvector('english', content || ' ' || COALESCE(metadata->>'Statement', '')),"
            "    plainto_tsquery('english', %s)"
            ") AS rank"
            " FROM {s}.blocks"
            " WHERE active"
            "   AND to_tsvector('english', content || ' ' || COALESCE(metadata->>'Statement', ''))"
            "       @@ plainto_tsquery('english', %s)"
            " ORDER BY rank DESC"
            " LIMIT %s",
        )
        cos_sql = _sql(
            self._schema,
            "SELECT id, (embedding <=> %s::vector) AS dist"
            " FROM {s}.blocks"
            " WHERE active AND embedding IS NOT NULL"
            " ORDER BY embedding <=> %s::vector"
            " LIMIT %s",
        )
        fetch_sql = _sql(
            self._schema,
            "SELECT id, file_path, content, metadata, created_at, updated_at, active FROM {s}.blocks WHERE id = ANY(%s)",
        )

        try:
            with pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(bm25_sql, (query, query, candidate_pool))
                    bm25_rows = cur.fetchall()  # [(id, rank), ...]
                    cos_rows: list[tuple[str, float]] = []
                    if do_vector:
                        emb_lit = _embedding_to_pg(query_embedding or [])
                        cur.execute(cos_sql, (emb_lit, emb_lit, candidate_pool))
                        cos_rows = cur.fetchall()

                    # RRF fusion. Both halves contribute 1/(k+rank); rank
                    # is 1-based to match the standard formulation. Ties
                    # broken by sum of contributions (already handled by
                    # accumulation).
                    fused: dict[str, float] = {}
                    for rank, (bid, _r) in enumerate(bm25_rows, start=1):
                        fused[bid] = fused.get(bid, 0.0) + 1.0 / (rrf_k + rank)
                    for rank, (cid, _d) in enumerate(cos_rows, start=1):
                        fused[cid] = fused.get(cid, 0.0) + 1.0 / (rrf_k + rank)

                    if not fused:
                        return []
                    top = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:limit]
                    top_ids = [bid for bid, _ in top]
                    score_by_id = dict(top)

                    with conn.cursor(row_factory=psycopg.rows.dict_row) as fcur:
                        fcur.execute(fetch_sql, (top_ids,))
                        rows_by_id = {r["id"]: dict(r) for r in fcur.fetchall()}

                    out: list[dict[str, Any]] = []
                    for bid in top_ids:
                        row = rows_by_id.get(bid)
                        if row is None:
                            continue  # raced delete; skip silently.
                        block = _row_to_block(row)
                        block["_score"] = score_by_id[bid]
                        block["_retrieval_source"] = "hybrid_pgvector" if do_vector else "bm25_only"
                        out.append(block)
                    return out
        except Exception as exc:
            raise BlockStoreError(f"hybrid_search failed: {exc}") from exc

    def list_blocks(self) -> list[str]:
        """Return distinct ``file_path`` values for all active blocks."""
        self._ensure_schema()
        pool = self._get_pool()
        sql = _sql(
            self._schema,
            "SELECT DISTINCT file_path FROM {s}.blocks WHERE active = TRUE ORDER BY file_path",
        )
        try:
            with pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql)
                    return [row[0] for row in cur.fetchall()]
        except Exception as exc:
            raise BlockStoreError(f"list_blocks failed: {exc}") from exc

    # ─── Write surface ────────────────────────────────────────────────────────

    def write_block(self, block: dict[str, Any], *, embedding: list[float] | None = None) -> str:
        """Upsert a block. Returns the block's ``_id``.

        INSERT … ON CONFLICT (id) DO UPDATE is executed in a single
        transaction so it is genuinely atomic (no lost-update window).

        v3.8.13: when ``embedding`` is provided AND the schema has the
        pgvector column (``self._has_vector``), the embedding is upserted
        atomically with the row. Without an embedding the column stays
        NULL — backfill via ``backfill_embedding`` later.
        """
        self._ensure_schema()
        pool = self._get_pool()
        block_id, file_path, content, metadata_json = _block_to_row(block)
        if embedding is not None and self._has_vector:
            if len(embedding) != self._embedding_dim:
                raise BlockStoreError(f"embedding dim mismatch: got {len(embedding)}, schema expects {self._embedding_dim}")
            sql = _sql(
                self._schema,
                "INSERT INTO {s}.blocks (id, file_path, content, metadata, embedding, updated_at)"
                " VALUES (%s, %s, %s, %s::jsonb, %s::vector, NOW())"
                " ON CONFLICT (id) DO UPDATE"
                "     SET file_path  = EXCLUDED.file_path,"
                "         content    = EXCLUDED.content,"
                "         metadata   = EXCLUDED.metadata,"
                "         embedding  = EXCLUDED.embedding,"
                "         updated_at = EXCLUDED.updated_at",
            )
            params = (block_id, file_path, content, metadata_json, _embedding_to_pg(embedding))
        else:
            sql = _sql(
                self._schema,
                "INSERT INTO {s}.blocks (id, file_path, content, metadata, updated_at)"
                " VALUES (%s, %s, %s, %s::jsonb, NOW())"
                " ON CONFLICT (id) DO UPDATE"
                "     SET file_path  = EXCLUDED.file_path,"
                "         content    = EXCLUDED.content,"
                "         metadata   = EXCLUDED.metadata,"
                "         updated_at = EXCLUDED.updated_at",
            )
            params = (block_id, file_path, content, metadata_json)
        try:
            with pool.connection() as conn:
                with conn.transaction():
                    conn.execute(sql, params)
            _log.debug("block_store_write", extra={"block_id": block_id, "embedded": embedding is not None})
            return block_id
        except Exception as exc:
            raise BlockStoreError(f"write_block failed for {block_id!r}: {exc}") from exc

    def backfill_embedding(self, block_id: str, embedding: list[float]) -> None:
        """Set the embedding for an existing row without touching content.

        Used by ``mm migrate-store --with-embeddings`` to populate the
        vector column post-hoc. Idempotent — overwrites whatever is
        there. Raises ``BlockStoreError`` if pgvector is not available.
        """
        self._ensure_schema()
        if not self._has_vector:
            raise BlockStoreError("pgvector not available; cannot backfill embeddings")
        if len(embedding) != self._embedding_dim:
            raise BlockStoreError(f"embedding dim mismatch: got {len(embedding)}, schema expects {self._embedding_dim}")
        pool = self._get_pool()
        sql = _sql(
            self._schema,
            "UPDATE {s}.blocks SET embedding = %s::vector, updated_at = NOW() WHERE id = %s",
        )
        try:
            with pool.connection() as conn:
                with conn.transaction():
                    conn.execute(sql, (_embedding_to_pg(embedding), block_id))
        except Exception as exc:
            raise BlockStoreError(f"backfill_embedding failed for {block_id!r}: {exc}") from exc

    def delete_block(self, block_id: str) -> bool:
        """Delete a block by id. Returns True if a row was removed.

        Logs to ``<schema>.deleted_blocks`` if that table exists;
        otherwise the deletion is silently recorded only in the
        application log.
        """
        self._ensure_schema()
        pool = self._get_pool()
        sql_delete = _sql(
            self._schema,
            "DELETE FROM {s}.blocks WHERE id = %s RETURNING id, content",
        )
        sql_log = _sql(
            self._schema,
            "INSERT INTO {s}.deleted_blocks (block_id, content, deleted_at) VALUES (%s, %s, NOW()) ON CONFLICT DO NOTHING",
        )
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
                        except Exception as exc:
                            _log.debug("deletion_journal_skipped block_id=%s: %s", block_id, exc)  # journal table absent — non-fatal
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
        files_in_snap: list[str] = sorted({b.get("_source_file", "") for b in blocks_to_snap if b.get("_source_file")})
        manifest: dict[str, Any] = {"files": files_in_snap, "version": 2}
        manifest_json = json.dumps(manifest, default=str)

        sql_snap = _sql(
            self._schema,
            "INSERT INTO {s}.snapshots (snap_id, manifest)"
            " VALUES (%s, %s::jsonb)"
            " ON CONFLICT (snap_id) DO UPDATE"
            "     SET manifest   = EXCLUDED.manifest,"
            "         created_at = NOW()",
        )
        sql_del_blocks = _sql(
            self._schema,
            "DELETE FROM {s}.snapshot_blocks WHERE snap_id = %s",
        )
        sql_blocks = _sql(
            self._schema,
            "INSERT INTO {s}.snapshot_blocks (snap_id, block_id, content, metadata)"
            " VALUES (%s, %s, %s, %s::jsonb)"
            " ON CONFLICT (snap_id, block_id) DO UPDATE"
            "     SET content  = EXCLUDED.content,"
            "         metadata = EXCLUDED.metadata",
        )
        try:
            with pool.connection() as conn:
                with conn.transaction():
                    conn.execute(sql_snap, (snap_id, manifest_json))
                    # Remove stale snapshot_blocks from a previous overwrite.
                    conn.execute(sql_del_blocks, (snap_id,))
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

        sql_check = _sql(
            self._schema,
            "SELECT snap_id FROM {s}.snapshots WHERE snap_id = %s",
        )
        sql_clear = _sql(self._schema, "DELETE FROM {s}.blocks")
        sql_insert = _sql(
            self._schema,
            "INSERT INTO {s}.blocks (id, file_path, content, metadata, active)"
            " SELECT sb.block_id,"
            "        COALESCE(sb.metadata->>'_source_file', ''),"
            "        sb.content,"
            "        sb.metadata,"
            "        TRUE"
            " FROM {s}.snapshot_blocks sb"
            " WHERE sb.snap_id = %s",
        )
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

        sql = _sql(
            self._schema,
            "SELECT COALESCE(live.file_path, snap.metadata->>'_source_file', snap.block_id) AS path,"
            "       CASE"
            "         WHEN live.id IS NULL       THEN 'removed'"
            "         WHEN snap.block_id IS NULL  THEN 'added'"
            "         ELSE                            'modified'"
            "       END AS change_type"
            " FROM {s}.blocks AS live"
            " FULL OUTER JOIN {s}.snapshot_blocks AS snap"
            "     ON live.id = snap.block_id AND snap.snap_id = %s"
            " WHERE snap.snap_id = %s OR snap.snap_id IS NULL"
            "   AND ("
            "        live.id IS NULL"
            "        OR snap.block_id IS NULL"
            "        OR live.content <> snap.content"
            "        OR live.metadata::text <> snap.metadata::text"
            "   )"
            " ORDER BY path",
        )
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
        sql_acquire = _sql(
            self._schema,
            "INSERT INTO {s}.workspace_lock (lock_id, holder) VALUES (%s, %s) ON CONFLICT (lock_id) DO NOTHING",
        )
        sql_release = _sql(
            self._schema,
            "DELETE FROM {s}.workspace_lock WHERE lock_id = %s AND holder = %s",
        )

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
                except Exception as exc:
                    _log.debug("pg_advisory_lock_release_failed lock_id=%s: %s", lock_id, exc)  # best-effort release; non-fatal

        return _PgLock()

    # ─── Compatibility shim ───────────────────────────────────────────────────

    def list_files(self) -> list[str]:
        """Deprecated alias for :meth:`list_blocks` — removed in v4.0."""
        import warnings

        warnings.warn(
            "BlockStore.list_files() is deprecated; use list_blocks() instead. The alias will be removed in v4.0.",
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
            except Exception as exc:
                _log.debug("pg_pool_close_failed: %s", exc)
            self._pool = None

    def __enter__(self) -> "PostgresBlockStore":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
