"""v4 HNSW kind-filtered ANN index (Group D).

Multi-LLM v4 audit (3/4 model consensus 2026-05-10) flagged the
``list_blocks_by_kind`` path as O(n) full-table scan and recommended
adding an HNSW index keyed by ``(kind, embedding)`` so kind-filtered
ANN runs in O(log n) instead.

Interface:

    register_block_embedding(workspace, block_id, kind, embedding)
    knn_by_kind(workspace, kind, query_embedding, k=10) -> [(bid, dist)]

deferred: the ANN backend is NOT built yet. ``knn_by_kind`` runs a
brute-force cosine scan over the kind partition on every install — the
correct answer, at O(n). An earlier revision of this module detected
sqlite-vec, created a placeholder ``vec0`` table, then returned the
brute-force result anyway, so ``backend_status`` advertised an ANN
backend that never ran a single query; the placeholder is gone and the
status surface now reports what actually serves the query.
upgrade path: a real vec0 backend needs (1) a per-dimension virtual
table — vec0 schemas are dim-specific — populated on the *write* path
in ``register_block_embedding``, never lazily on the read path, (2) a
sync watermark so rows registered while the extension was unloadable
can't silently vanish from the ANN answer, and (3) an equivalence gate
against ``_knn_brute_force`` over random vectors. Until all three
exist, a vec0 query would be a faster wrong answer.

``backend_status`` reports ``sqlite_vec_available`` separately, so a
deployment can still see whether the extension is installed and ready
for that work.

Feature-flag gated under ``v4.hnsw_kind_index``. v3.x callers see no
behaviour change.

Copyright STARGA, Inc.
"""

from __future__ import annotations

import logging
import math
import sqlite3
import struct
from collections.abc import Sequence
from contextlib import closing
from pathlib import Path

from .feature_flags import require_enabled

__all__ = [
    "FLAG",
    "register_block_embedding",
    "knn_by_kind",
    "ensure_hnsw_schema",
    "backend_status",
]


FLAG: str = "hnsw_kind_index"

_log = logging.getLogger("mind_mem.v4.hnsw_kind_index")


# Every connection below is opened as ``with closing(sqlite3.connect(...)) as
# conn:`` wrapping an inner ``with conn:``. Both halves are load-bearing and the
# nesting order is not arbitrary:
#
# * the inner ``with conn`` commits on success / rolls back on an exception —
#   ``close()`` alone does neither, and closing a connection with an open write
#   transaction discards it;
# * the outer ``closing`` actually releases the descriptor. A bare
#   ``with sqlite3.connect(...) as conn`` — what this module used to do — commits
#   and then leaves the handle open, which its ``__exit__`` documents. Nothing
#   reclaims it afterwards either: a ``sqlite3.Connection`` and its
#   prepared-statement cache reference each other, so the object is unreachable
#   only to the cyclic collector, not to refcounting. Until a collection happens
#   the process holds an open descriptor on ``index.db`` and on its ``-wal`` /
#   ``-shm`` sidecars, and on Windows those handles make the workspace
#   undeletable.
#
# These are module-level functions called once per operation, so there is no
# long-lived connection here to keep open deliberately — every one of them is
# opened, used, and closed within a single call.


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------


def _try_load_sqlite_vec(conn: sqlite3.Connection) -> bool:
    """Attempt to load the sqlite-vec extension. Returns True on success."""
    try:
        import sqlite_vec  # type: ignore[import-not-found]
    except ImportError:
        return False
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
    except (sqlite3.OperationalError, AttributeError):
        return False
    return True


def backend_status(workspace: str | Path) -> dict[str, object]:
    """Report which backend actually runs kNN, and what is available.

    ``backend`` is the one that will serve the next ``knn_by_kind``
    call. Today that is always ``brute_force``: no ANN backend is
    implemented (see the module docstring), so reporting anything else
    would tell a deployment health check that an index is doing work
    nothing does.

    ``sqlite_vec_available`` is the separate, honest signal — whether
    the extension loads here — for callers tracking readiness for the
    ANN work rather than current behaviour.
    """
    require_enabled(FLAG)
    db = Path(workspace) / "index.db"
    if not db.parent.is_dir():
        db.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(db, timeout=30)) as conn:
        with conn:
            available = _try_load_sqlite_vec(conn)
    return {
        "backend": "brute_force",
        "sqlite_vec_available": available,
        "workspace": str(workspace),
    }


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA_SQL: str = """
CREATE TABLE IF NOT EXISTS block_kind_embeddings (
    block_id TEXT PRIMARY KEY,
    kind     TEXT NOT NULL,
    payload  BLOB NOT NULL,
    dim      INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_block_kind_embeddings_kind
    ON block_kind_embeddings (kind);
"""


def ensure_hnsw_schema(workspace: str | Path) -> None:
    """Idempotent. Creates the ``block_kind_embeddings`` table that
    :func:`knn_by_kind` scans. It is also the source-of-truth a future
    vec0 backend would be populated from — see the module docstring for
    what that backend still needs before it can be trusted."""
    require_enabled(FLAG)
    db = Path(workspace) / "index.db"
    if not db.parent.is_dir():
        db.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(db, timeout=30)) as conn:
        with conn:
            conn.executescript(_SCHEMA_SQL)
            conn.commit()


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------


def register_block_embedding(
    workspace: str | Path,
    block_id: str,
    kind: str,
    embedding: Sequence[float],
) -> None:
    """Store an embedding for a (block_id, kind) pair.

    The embedding is packed as ``<{N}f`` (little-endian float32). On
    re-register with the same block_id, the row is replaced (INSERT
    OR REPLACE) — last-writer-wins semantics matching the rest of
    the v4 surface's update model.
    """
    require_enabled(FLAG)
    if not embedding:
        return
    ensure_hnsw_schema(workspace)
    db = Path(workspace) / "index.db"
    payload = struct.pack(f"<{len(embedding)}f", *embedding)
    with closing(sqlite3.connect(db, timeout=30)) as conn:
        with conn:
            conn.execute(
                "INSERT OR REPLACE INTO block_kind_embeddings (block_id, kind, payload, dim) VALUES (?, ?, ?, ?)",
                (block_id, kind, payload, len(embedding)),
            )
            conn.commit()


# ---------------------------------------------------------------------------
# kNN
# ---------------------------------------------------------------------------


def knn_by_kind(
    workspace: str | Path,
    kind: str,
    query: Sequence[float],
    *,
    k: int = 10,
) -> list[tuple[str, float]]:
    """Return up to ``k`` (block_id, cosine_distance) pairs for blocks
    of the given kind, ordered by ascending distance.

    Distance is ``1 - cos_sim``; range ``[0, 2]``. A brute-force
    sequential scan over the kind partition — see the module docstring
    for why there is no ANN path yet.

    Empty result for: missing schema, no embeddings of that kind,
    non-positive k, a zero-norm query, or every stored vector in the
    partition being skipped (dimension mismatch against the query, or
    zero norm). The last case is the one that looks like an empty
    partition but isn't — it is logged as a warning naming the counts,
    because after an embedder swap it is the whole index, not one row.

    Read-only: this path never writes to ``index.db``.
    """
    require_enabled(FLAG)
    if k <= 0 or not query:
        return []
    db = Path(workspace) / "index.db"
    if not db.is_file():
        return []
    # Read-only path: the inner ``with conn`` has nothing to commit, but it
    # is kept so the shape is identical everywhere and a future statement here
    # cannot silently become an uncommitted write.
    with closing(sqlite3.connect(db, timeout=30)) as conn:
        with conn:
            if not _table_exists(conn, "block_kind_embeddings"):
                return []
            return _knn_brute_force(conn, kind, query, k)


def _knn_brute_force(
    conn: sqlite3.Connection,
    kind: str,
    query: Sequence[float],
    k: int,
) -> list[tuple[str, float]]:
    """Sequential cosine-distance scan over the kind partition."""
    rows = conn.execute(
        "SELECT block_id, payload, dim FROM block_kind_embeddings WHERE kind = ?",
        (kind,),
    ).fetchall()
    scored: list[tuple[str, float]] = []
    qlen = len(query)
    qnorm = math.sqrt(sum(v * v for v in query))
    if qnorm == 0.0:
        return []
    skipped_dim = 0
    skipped_unusable = 0  # unpack failure or zero norm — no direction to score
    for bid, payload, dim in rows:
        if int(dim) != qlen:
            skipped_dim += 1
            continue
        try:
            vec = struct.unpack(f"<{dim}f", payload)
        except struct.error:
            skipped_unusable += 1
            continue
        vnorm = math.sqrt(sum(v * v for v in vec))
        if vnorm == 0.0:
            skipped_unusable += 1
            continue
        dot = sum(a * b for a, b in zip(query, vec))
        cos_sim = dot / (qnorm * vnorm)
        cos_sim = max(-1.0, min(1.0, cos_sim))
        scored.append((bid, 1.0 - cos_sim))
    if skipped_dim:
        # A dimension mismatch returns fewer rows — or, when it takes the
        # whole partition, an empty list indistinguishable from "no such
        # kind". After an embedder swap that is the whole index, not one
        # row, so name the counts rather than letting it read as empty.
        _log.warning(
            "%s kind=%s query_dim=%d rows=%d skipped_dim_mismatch=%d skipped_unusable=%d",
            "hnsw_knn_all_rows_skipped" if not scored else "hnsw_knn_rows_skipped",
            kind,
            qlen,
            len(rows),
            skipped_dim,
            skipped_unusable,
        )
    scored.sort(key=lambda r: r[1])
    return scored[:k]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?",
        (name,),
    ).fetchone()
    return row is not None
