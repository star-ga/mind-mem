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

from .observability import get_logger

_log = get_logger("retrieval_graph")

# Retention cleanup counter — prune old entries every 100th log_retrieval call (#472)
_retention_counter: int = 0

__all__ = [
    "ensure_graph_tables",
    "log_retrieval",
    "propagate_scores",
    "record_hard_negatives",
    "get_hard_negative_ids",
    "retrieval_diagnostics",
]

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS retrieval_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    query_text   TEXT NOT NULL,
    query_hash   TEXT NOT NULL,
    mem_ids      TEXT NOT NULL,
    scores       TEXT NOT NULL,
    top_k        INTEGER,
    timestamp    TEXT DEFAULT (datetime('now')),
    feedback     REAL DEFAULT 0.0,
    intent_type  TEXT DEFAULT '',
    stage_counts TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_rlog_qhash ON retrieval_log(query_hash);
CREATE INDEX IF NOT EXISTS idx_rlog_ts ON retrieval_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_rlog_intent ON retrieval_log(intent_type);

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


def _migrate_schema(conn: sqlite3.Connection) -> None:
    """Add columns from schema v2 (#428/#430) if missing."""
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(retrieval_log)").fetchall()}
    except Exception:
        return
    if "intent_type" not in cols:
        conn.execute("ALTER TABLE retrieval_log ADD COLUMN intent_type TEXT DEFAULT ''")
    if "stage_counts" not in cols:
        conn.execute("ALTER TABLE retrieval_log ADD COLUMN stage_counts TEXT DEFAULT '{}'")
    try:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_rlog_intent ON retrieval_log(intent_type)")
    except Exception:
        pass


def ensure_graph_tables(workspace: str) -> None:
    """Create retrieval_log, co_retrieval, hard_negatives tables if missing."""
    conn = _connect(workspace)
    conn.executescript(_SCHEMA_SQL)
    _migrate_schema(conn)
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Retrieval logging
# ---------------------------------------------------------------------------


def log_retrieval(
    workspace: str,
    query: str,
    results: list[dict],
    *,
    intent_type: str = "",
    stage_counts: dict | None = None,
) -> None:
    """Log a recall query and its results. Updates co-retrieval edges.

    Called after every recall() — best-effort (never raises).

    Args:
        workspace: Workspace root path.
        query: Original query text.
        results: Final result dicts.
        intent_type: IntentRouter classification (e.g. "WHY", "WHEN").
        stage_counts: Per-stage candidate counts from the pipeline.
    """
    if not results:
        return
    conn = None
    try:
        conn = _connect(workspace)
        conn.executescript(_SCHEMA_SQL)
        _migrate_schema(conn)

        mem_ids = [r.get("_id", "") for r in results if r.get("_id")]
        scores = [r.get("score", 0) for r in results]
        qhash = hashlib.sha256(query.encode()).hexdigest()[:16]

        conn.execute(
            "INSERT INTO retrieval_log "
            "(query_text, query_hash, mem_ids, scores, top_k, intent_type, stage_counts) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                query,
                qhash,
                json.dumps(mem_ids),
                json.dumps(scores),
                len(results),
                intent_type,
                json.dumps(stage_counts or {}),
            ),
        )

        # Update co-retrieval edges (undirected: always store lo < hi)
        edge_weight = 1.0 / max(len(mem_ids), 1)
        for i, a in enumerate(mem_ids):
            for b in mem_ids[i + 1 :]:
                lo, hi = (a, b) if a < b else (b, a)
                conn.execute(
                    "INSERT INTO co_retrieval (mem1_id, mem2_id, weight, hit_count, updated_at) "
                    "VALUES (?, ?, ?, 1, datetime('now')) "
                    "ON CONFLICT(mem1_id, mem2_id) DO UPDATE SET "
                    "weight = weight + ?, hit_count = hit_count + 1, "
                    "updated_at = datetime('now')",
                    (lo, hi, edge_weight, edge_weight),
                )

        # Retention policy: prune entries older than 30 days every 100th call (#472)
        global _retention_counter
        _retention_counter += 1
        if _retention_counter % 100 == 0:
            try:
                conn.execute("DELETE FROM retrieval_log WHERE timestamp < datetime('now', '-30 days')")
                conn.execute("DELETE FROM co_retrieval WHERE updated_at < datetime('now', '-30 days')")
                _log.debug("retention_cleanup", counter=_retention_counter)
            except Exception as cleanup_exc:
                _log.debug("retention_cleanup_failed", error=str(cleanup_exc))

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
    max_hops: int = 2,
) -> dict[str, float]:
    """PageRank-like score propagation across co-retrieval graph.

    Args:
        workspace: Workspace root path.
        initial_scores: {block_id: score} from current recall results.
        iterations: Number of propagation rounds.
        damping: Fraction of score transferred per edge per iteration.
        min_edge: Minimum edge weight to consider.
        max_hops: Maximum propagation depth from seed nodes (#472).

    Returns:
        Updated {block_id: score} with propagated boosts.
    """
    conn = None
    try:
        conn = _connect(workspace)
        conn.executescript(_SCHEMA_SQL)
    except Exception as exc:
        _log.debug("propagate_scores_failed", error=str(exc))
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
    except Exception as exc:
        _log.debug("propagate_scores_failed", error=str(exc))
        return dict(initial_scores)
    finally:
        if conn:
            conn.close()

    if not adj:
        return dict(initial_scores)

    # Bound propagation to max_hops from seed nodes (#472)
    # Only allow neighbors within max_hops of the original seed set.
    seed_ids = set(initial_scores.keys())
    reachable: set[str] = set(seed_ids)
    frontier: set[str] = set(seed_ids)
    for _hop in range(max_hops):
        next_frontier: set[str] = set()
        for mid in frontier:
            for neighbor, _w in adj.get(mid, []):
                if neighbor not in reachable:
                    reachable.add(neighbor)
                    next_frontier.add(neighbor)
        frontier = next_frontier
        if not frontier:
            break

    scores = dict(initial_scores)
    for _ in range(iterations):
        updates: dict[str, float] = {}
        for mid, score in scores.items():
            for neighbor, w in adj.get(mid, []):
                if neighbor not in reachable:
                    continue  # beyond max_hops — skip
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
            "SELECT DISTINCT mem_id FROM hard_negatives WHERE timestamp > datetime('now', ?)",
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


# ---------------------------------------------------------------------------
# Retrieval diagnostics (#428)
# ---------------------------------------------------------------------------


def retrieval_diagnostics(
    workspace: str,
    *,
    last_n: int = 50,
    max_age_days: int = 7,
) -> dict:
    """Aggregate pipeline diagnostics from recent retrieval logs.

    Returns per-stage rejection rates, intent distribution, confidence
    histogram, and hard negative summary.

    Args:
        workspace: Workspace root path.
        last_n: Number of recent queries to analyze.
        max_age_days: Only consider queries within this age window.

    Returns:
        Dict with stage_rejection_rates, intent_distribution,
        score_distribution, hard_negative_summary.
    """
    conn = None
    try:
        conn = _connect(workspace)
        conn.executescript(_SCHEMA_SQL)
        _migrate_schema(conn)

        # --- Stage counts aggregation ---
        rows = conn.execute(
            "SELECT query_text, intent_type, stage_counts, scores FROM retrieval_log "
            "WHERE timestamp > datetime('now', ?) "
            "ORDER BY id DESC LIMIT ?",
            (f"-{max_age_days} days", last_n),
        ).fetchall()

        intent_dist: dict[str, int] = {}
        intent_quality: dict[str, list[float]] = {}  # #430: per-intent quality
        intent_confidence: dict[str, list[float]] = {}  # #430: per-intent confidence
        stage_totals: dict[str, list[int]] = {}
        all_top_scores: list[float] = []
        all_final_counts: list[int] = []
        low_confidence_queries: list[dict] = []

        for row in rows:
            intent = row["intent_type"] or "unknown"
            intent_dist[intent] = intent_dist.get(intent, 0) + 1

            try:
                sc = json.loads(row["stage_counts"]) if row["stage_counts"] else {}
            except (json.JSONDecodeError, TypeError):
                sc = {}

            for stage, count in sc.items():
                if isinstance(count, (int, float)):
                    stage_totals.setdefault(stage, []).append(int(count))

            # #430: Track intent confidence from stage_counts
            conf = sc.get("intent_confidence")
            if conf is not None:
                intent_confidence.setdefault(intent, []).append(float(conf))
                if float(conf) < 0.3:
                    low_confidence_queries.append(
                        {
                            "query": (row["query_text"] or "")[:80],
                            "intent": intent,
                            "confidence": float(conf),
                        }
                    )

            try:
                scores = json.loads(row["scores"]) if row["scores"] else []
            except (json.JSONDecodeError, TypeError):
                scores = []
            if scores:
                top_score = max(scores)
                all_top_scores.append(top_score)
                all_final_counts.append(len(scores))
                # #430: Per-intent quality signal (top score as proxy)
                intent_quality.setdefault(intent, []).append(top_score)

        # Compute per-stage averages and rejection rates
        stage_stats: dict[str, dict] = {}
        ordered_stages = [
            "corpus_loaded",
            "bm25_passed",
            "graph_boosted",
            "rm3_expanded",
            "temporal_filtered",
            "wide_candidates",
            "deduped",
            "reranked",
            "hard_neg_penalized",
            "knee_cutoff",
            "final",
        ]
        for stage in ordered_stages:
            counts = stage_totals.get(stage, [])
            if counts:
                stage_stats[stage] = {
                    "avg": round(sum(counts) / len(counts), 1),
                    "min": min(counts),
                    "max": max(counts),
                    "samples": len(counts),
                }

        # Rejection rates between consecutive stages
        rejection_rates: dict[str, float] = {}
        prev_stage: str | None = None
        for stage in ordered_stages:
            if stage in stage_stats and prev_stage is not None and prev_stage in stage_stats:
                prev_avg = stage_stats[prev_stage]["avg"]
                curr_avg = stage_stats[stage]["avg"]
                if prev_avg > 0:
                    rejection_rates[f"{prev_stage}_to_{stage}"] = round(1.0 - curr_avg / prev_avg, 3)
            if stage in stage_stats:
                prev_stage = stage

        # Score distribution
        score_dist = {}
        if all_top_scores:
            all_top_scores.sort()
            score_dist = {
                "p25": round(all_top_scores[len(all_top_scores) // 4], 4),
                "p50": round(all_top_scores[len(all_top_scores) // 2], 4),
                "p75": round(all_top_scores[3 * len(all_top_scores) // 4], 4),
                "avg_final_count": round(sum(all_final_counts) / len(all_final_counts), 1),
            }

        # --- Hard negatives summary ---
        hn_rows = conn.execute(
            "SELECT mem_id, bm25_score, ce_score FROM hard_negatives WHERE timestamp > datetime('now', ?)",
            (f"-{max_age_days} days",),
        ).fetchall()
        hn_summary = {
            "total": len(hn_rows),
            "unique_blocks": len({r["mem_id"] for r in hn_rows}),
        }
        if hn_rows:
            bm25_scores = [r["bm25_score"] for r in hn_rows if r["bm25_score"] is not None]
            ce_scores = [r["ce_score"] for r in hn_rows if r["ce_score"] is not None]
            if bm25_scores:
                hn_summary["avg_bm25"] = round(sum(bm25_scores) / len(bm25_scores), 4)
            if ce_scores:
                hn_summary["avg_ce"] = round(sum(ce_scores) / len(ce_scores), 4)

        # #430: Per-intent quality breakdown
        intent_quality_summary: dict[str, dict] = {}
        for intent, scores_list in intent_quality.items():
            if scores_list:
                scores_list.sort()
                intent_quality_summary[intent] = {
                    "queries": len(scores_list),
                    "avg_top_score": round(sum(scores_list) / len(scores_list), 4),
                    "p50_top_score": round(scores_list[len(scores_list) // 2], 4),
                }
                confs = intent_confidence.get(intent, [])
                if confs:
                    intent_quality_summary[intent]["avg_confidence"] = round(sum(confs) / len(confs), 3)

        return {
            "queries_analyzed": len(rows),
            "intent_distribution": intent_dist,
            "intent_quality": intent_quality_summary,
            "low_confidence_queries": low_confidence_queries[:10],
            "stage_stats": stage_stats,
            "rejection_rates": rejection_rates,
            "score_distribution": score_dist,
            "hard_negatives": hn_summary,
        }

    except Exception as exc:
        _log.debug("retrieval_diagnostics_failed", error=str(exc))
        return {"error": str(exc), "queries_analyzed": 0}
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass
