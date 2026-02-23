"""Retrieval logger + co-retrieval graph for usage-based score propagation.

Logs every recall() invocation (query, returned IDs, scores) into SQLite.
Builds co-retrieval edges (blocks frequently returned together get linked).
Propagates scores across the co-retrieval graph via damped PageRank-like
iteration to surface "hidden" relevant blocks.

Tables (created in recall.db alongside FTS5 index):
    retrieval_log: per-query log (query_text, mem_ids, scores, timestamp)
    co_retrieval:  weighted undirected edges between co-returned blocks
    hard_negatives: blocks that BM25 liked but cross-encoder rejected

Zero external deps — sqlite3, json, hashlib (all stdlib).
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3

from observability import get_logger

_log = get_logger("retrieval_graph")

__all__ = [
    "ensure_graph_tables",
    "log_retrieval",
    "propagate_scores",
    "record_hard_negatives",
    "get_hard_negative_ids",
]

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS retrieval_log (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    query_text TEXT NOT NULL,
    query_hash TEXT NOT NULL,
    mem_ids    TEXT NOT NULL,
    scores     TEXT NOT NULL,
    top_k      INTEGER,
    timestamp  TEXT DEFAULT (datetime('now')),
    feedback   REAL DEFAULT 0.0
);
CREATE INDEX IF NOT EXISTS idx_rlog_qhash ON retrieval_log(query_hash);
CREATE INDEX IF NOT EXISTS idx_rlog_ts ON retrieval_log(timestamp);

CREATE TABLE IF NOT EXISTS co_retrieval (
    mem1_id    TEXT NOT NULL,
    mem2_id    TEXT NOT NULL,
    weight     REAL DEFAULT 0.0,
    hit_count  INTEGER DEFAULT 0,
    updated_at TEXT,
    PRIMARY KEY (mem1_id, mem2_id)
);
CREATE INDEX IF NOT EXISTS idx_co_ret_weight ON co_retrieval(weight);

CREATE TABLE IF NOT EXISTS hard_negatives (
    mem_id      TEXT NOT NULL,
    query_hash  TEXT NOT NULL,
    bm25_score  REAL,
    ce_score    REAL,
    timestamp   TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (mem_id, query_hash)
);
"""


def _db_path(workspace: str) -> str:
    return os.path.join(os.path.abspath(workspace), ".mind-mem-index", "recall.db")


def _connect(workspace: str) -> sqlite3.Connection:
    path = _db_path(workspace)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path, timeout=5)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=3000")
    conn.row_factory = sqlite3.Row
    return conn


def ensure_graph_tables(workspace: str) -> None:
    """Create retrieval_log, co_retrieval, hard_negatives tables if missing."""
    conn = _connect(workspace)
    conn.executescript(_SCHEMA_SQL)
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Retrieval logging
# ---------------------------------------------------------------------------

def log_retrieval(
    workspace: str,
    query: str,
    results: list[dict],
) -> None:
    """Log a recall query and its results. Updates co-retrieval edges.

    Called after every recall() — best-effort (never raises).
    """
    if not results:
        return
    conn = None
    try:
        conn = _connect(workspace)
        conn.executescript(_SCHEMA_SQL)

        mem_ids = [r.get("_id", "") for r in results if r.get("_id")]
        scores = [r.get("score", 0) for r in results]
        qhash = hashlib.sha256(query.encode()).hexdigest()[:16]

        conn.execute(
            "INSERT INTO retrieval_log (query_text, query_hash, mem_ids, scores, top_k) "
            "VALUES (?, ?, ?, ?, ?)",
            (query, qhash, json.dumps(mem_ids), json.dumps(scores), len(results)),
        )

        # Update co-retrieval edges (undirected: always store lo < hi)
        edge_weight = 1.0 / max(len(mem_ids), 1)
        for i, a in enumerate(mem_ids):
            for b in mem_ids[i + 1:]:
                lo, hi = (a, b) if a < b else (b, a)
                conn.execute(
                    "INSERT INTO co_retrieval (mem1_id, mem2_id, weight, hit_count, updated_at) "
                    "VALUES (?, ?, ?, 1, datetime('now')) "
                    "ON CONFLICT(mem1_id, mem2_id) DO UPDATE SET "
                    "weight = weight + ?, hit_count = hit_count + 1, "
                    "updated_at = datetime('now')",
                    (lo, hi, edge_weight, edge_weight),
                )

        conn.commit()
        _log.debug("retrieval_logged", query_hash=qhash, results=len(results))
    except Exception as e:
        _log.debug("retrieval_log_failed", error=str(e))
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Score propagation (PageRank-like)
# ---------------------------------------------------------------------------

def propagate_scores(
    workspace: str,
    initial_scores: dict[str, float],
    *,
    iterations: int = 3,
    damping: float = 0.3,
    min_edge: float = 0.1,
) -> dict[str, float]:
    """PageRank-like score propagation across co-retrieval graph.

    Args:
        workspace: Workspace root path.
        initial_scores: {block_id: score} from current recall results.
        iterations: Number of propagation rounds.
        damping: Fraction of score transferred per edge per iteration.
        min_edge: Minimum edge weight to consider.

    Returns:
        Updated {block_id: score} with propagated boosts.
    """
    conn = None
    try:
        conn = _connect(workspace)
        conn.executescript(_SCHEMA_SQL)
    except Exception:
        if conn:
            conn.close()
        return dict(initial_scores)

    try:
        adj: dict[str, list[tuple[str, float]]] = {}
        for row in conn.execute(
            "SELECT mem1_id, mem2_id, weight FROM co_retrieval WHERE weight > ?",
            (min_edge,),
        ):
            m1, m2, w = row["mem1_id"], row["mem2_id"], row["weight"]
            adj.setdefault(m1, []).append((m2, w))
            adj.setdefault(m2, []).append((m1, w))
    except Exception:
        return dict(initial_scores)
    finally:
        if conn:
            conn.close()

    if not adj:
        return dict(initial_scores)

    scores = dict(initial_scores)
    for _ in range(iterations):
        updates: dict[str, float] = {}
        for mid, score in scores.items():
            for neighbor, w in adj.get(mid, []):
                boost = score * damping * min(w, 1.0)
                if boost > updates.get(neighbor, 0):
                    updates[neighbor] = boost
        for mid, boost in updates.items():
            scores[mid] = max(scores.get(mid, 0), boost)

    return scores


# ---------------------------------------------------------------------------
# Hard negative recording (Feature 5: abstention-guided)
# ---------------------------------------------------------------------------

def record_hard_negatives(
    workspace: str,
    query: str,
    candidates: list[dict],
    *,
    bm25_threshold: float = 0.1,
    ce_threshold: float = 0.3,
) -> int:
    """Log near-miss blocks when abstention fires or cross-encoder rejects.

    A hard negative is a block that BM25 scored highly but the cross-encoder
    (or abstention classifier) rejected — indicating it's superficially
    relevant but actually misleading.

    Args:
        workspace: Workspace root path.
        query: Original query text.
        candidates: Result dicts with score (BM25) and optionally ce_score.
        bm25_threshold: Min BM25 score to be considered a near-miss.
        ce_threshold: Max cross-encoder score to be flagged as negative.

    Returns:
        Number of hard negatives recorded.
    """
    count = 0
    conn = None
    try:
        conn = _connect(workspace)
        conn.executescript(_SCHEMA_SQL)
        qhash = hashlib.sha256(query.encode()).hexdigest()[:16]

        for cand in candidates:
            bm25 = cand.get("score", 0)
            ce = cand.get("ce_score", 1.0)
            if bm25 > bm25_threshold and ce < ce_threshold:
                conn.execute(
                    "INSERT OR IGNORE INTO hard_negatives "
                    "(mem_id, query_hash, bm25_score, ce_score) "
                    "VALUES (?, ?, ?, ?)",
                    (cand.get("_id", ""), qhash, bm25, ce),
                )
                count += 1

        conn.commit()
        if count:
            _log.debug("hard_negatives_recorded", count=count)
    except Exception as e:
        _log.debug("hard_negative_record_failed", error=str(e))
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass
    return count


def get_hard_negative_ids(workspace: str, *, max_age_days: int = 30) -> set[str]:
    """Return set of block IDs flagged as hard negatives within max_age_days."""
    conn = None
    try:
        conn = _connect(workspace)
        conn.executescript(_SCHEMA_SQL)
        rows = conn.execute(
            "SELECT DISTINCT mem_id FROM hard_negatives "
            "WHERE timestamp > datetime('now', ?)",
            (f"-{max_age_days} days",),
        ).fetchall()
        return {row["mem_id"] for row in rows}
    except Exception:
        return set()
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass
