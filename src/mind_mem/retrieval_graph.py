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
from datetime import date, timedelta
from typing import Any

from .observability import get_logger
from .scoring_instant import resolve_scoring_instant

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
    "feedback_quality_credit",
    "recall_sufficiency",
    "graph_db_path",
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
    stage_counts TEXT DEFAULT '{}',
    credits      TEXT DEFAULT '{}'
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


def graph_db_path(workspace: str) -> str:
    """Where this module writes the ``co_retrieval`` graph — the public name.

    Readers of the graph (``v4.kernels``' lineage / contradicts / graph_walk
    strategies) resolve the path through the WRITER rather than rebuilding it,
    so the reader can never drift away from wherever the writer puts it. They
    were reaching for the private ``_db_path`` to do it; this is the same
    function under a name that is part of the contract.
    """
    return _db_path(workspace)


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
    except Exception as exc:
        _log.debug("schema_migrate_skipped", error=str(exc))
        return
    if "intent_type" not in cols:
        conn.execute("ALTER TABLE retrieval_log ADD COLUMN intent_type TEXT DEFAULT ''")
    if "stage_counts" not in cols:
        conn.execute("ALTER TABLE retrieval_log ADD COLUMN stage_counts TEXT DEFAULT '{}'")
    if "credits" not in cols:
        conn.execute("ALTER TABLE retrieval_log ADD COLUMN credits TEXT DEFAULT '{}'")
    try:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_rlog_intent ON retrieval_log(intent_type)")
    except Exception as exc:
        _log.debug("retrieval_log_index_skipped", error=str(exc))


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

    **Deliberately NOT admitted, and this is the argument (ROW-7).** The
    other two writers into this graph — ``block_lineage.add_block_edge``
    and ``causal_graph.CausalGraph.add_edge`` — are *assertion* doors:
    they take a whole ``(src, kind, dst)`` triple from a USER-scope tool
    argument, so they can name any two blocks and any relation, and they
    open one ``admit_artifact`` scope each. This is not that door.

    Three properties, and the third is the one that matters:

    * it is **telemetry over content already served** — every id it can
      write is one the ``recall()`` it is logging just returned;
    * it **cannot choose a kind**. The ``INSERT`` above names no ``kind``
      column, so a new row takes the schema default ``'cooccurrence'``,
      and the ``ON CONFLICT`` arm does not update ``kind`` — a typed
      edge another door asserted keeps its type. A co-occurrence row
      therefore cannot become a ``contradicts`` edge here, which is the
      only kind that fires staleness propagation; and
    * it runs on **every recall**, writing O(k²) upserts. Admitting it
      would put one evidence-chain row per query into the ledger that
      exists to record decisions ABOUT content — drowning real
      governance events in retrieval noise and growing the audit file
      without bound. That is a real cost for no gain against a door that
      can neither name a block nor choose a relation.

    The claim is bounded, not waived: the confinement is asserted by
    ``tests/test_governed_artifact_writes.py``
    (``TestTheTelemetrySinkIsNotAnAssertionDoor``), so an edit that lets
    this function write a typed kind fails the build and this argument
    stops being available.

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
        credits = {r["_id"]: r["feedback_credit"] for r in results if r.get("_id") and isinstance(r.get("feedback_credit"), dict)}

        conn.execute(
            "INSERT INTO retrieval_log "
            "(query_text, query_hash, mem_ids, scores, top_k, intent_type, stage_counts, credits) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                query,
                qhash,
                json.dumps(mem_ids),
                json.dumps(scores),
                len(results),
                intent_type,
                json.dumps(stage_counts or {}),
                json.dumps(credits),
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
            except Exception as exc:
                _log.debug("retrieval_log_conn_close_failed", error=str(exc))


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
                    "INSERT OR IGNORE INTO hard_negatives (mem_id, query_hash, bm25_score, ce_score) VALUES (?, ?, ?, ?)",
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
            except Exception as exc:
                _log.debug("hard_negative_conn_close_failed", error=str(exc))
    return count


def get_hard_negative_ids(
    workspace: str,
    *,
    max_age_days: int = 30,
    scoring_instant: date | str | None = None,
) -> set[str]:
    """Block ids flagged as hard negatives within *max_age_days* of the run.

    The window is measured from *scoring_instant* — the run's pinned
    recency input — and NOT from the wall clock.

    It used to read ``datetime('now')`` inside the SQL, which put a
    hidden clock on the scoring path: this set demotes scores, so with
    the identical corpus, config, query *and* pinned ``scoring_instant``,
    the same recall returned a different answer once a row aged past the
    window. That falsified the purity contract recall states
    (``recall`` is a function of corpus, config and ``scoring_instant``)
    and made an attested run unreplayable — the attestation binds
    ``scoring_instant``, so replaying it must not consult a second,
    unbound clock.

    ``None`` still resolves to today in UTC, so an unpinned caller is
    unchanged; the cutoff is just computed once, in Python, from the
    instant the run is scoring against.
    """
    cutoff = resolve_scoring_instant(scoring_instant) - timedelta(days=max_age_days)
    conn = None
    try:
        conn = _connect(workspace)
        conn.executescript(_SCHEMA_SQL)
        rows = conn.execute(
            # Lexicographic comparison against the stored
            # ``YYYY-MM-DD HH:MM:SS`` timestamps, which is why the cutoff
            # is rendered at midnight in the same layout.
            "SELECT DISTINCT mem_id FROM hard_negatives WHERE timestamp > ?",
            (f"{cutoff.isoformat()} 00:00:00",),
        ).fetchall()
        return {row["mem_id"] for row in rows}
    except Exception as exc:
        _log.debug("hard_negative_ids_failed", error=str(exc))
        return set()
    finally:
        if conn:
            try:
                conn.close()
            except Exception as exc2:
                _log.debug("hard_negative_ids_conn_close_failed", error=str(exc2))


# ---------------------------------------------------------------------------
# Feedback-quality credit (Group I, flag-gated)
# ---------------------------------------------------------------------------


def feedback_quality_credit(
    hits: list[dict[str, Any]],
    workspace: str,
    cfg: dict[str, Any],
) -> None:
    """Annotate every hit with a four-component feedback-quality credit.

    Group I: per-hit credit {informative, valid, non_redundant, retained},
    each round(x, 4) in [0, 1]. Mutates ``hits`` in place (Stage 3.1),
    mirroring the ``apply_validity_gate`` idiom. A complete no-op (no
    annotation, no DB reads) unless ``cfg["feedback_credit"]["enabled"]``
    is truthy. ``cfg`` is the ``recall`` section of mind-mem.json.

    Deterministic by construction: reads only the hit list plus unwindowed
    stored state (contradiction log, staleness scores) — no clock, no
    randomness on the scored preimage.
    """
    fc_cfg = cfg.get("feedback_credit")
    if not isinstance(fc_cfg, dict) or not fc_cfg.get("enabled", False):
        return
    if not hits:
        return

    # Lazy imports — lineage_staleness (pulled in by validity_gate)
    # imports this module, so top-level imports would be circular.
    from ._recall_constants import LIFECYCLE_RETENTION
    from .dedup import _cosine_similarity, _get_result_text, _term_vector, _text_tokens
    from .validity_gate import (
        _load_contradicted_party_ids,
        _status_component,
        validity_components,
    )

    # `valid` — batch the two stored-state reads only when some hit lacks
    # the Stage 2.65 annotation; either way the math is the ONE shared
    # validity_components helper.
    unannotated = [h for h in hits if not isinstance(h.get("validity"), dict)]
    contradicted_ids: set[str] = set()
    staleness: dict[str, float] = {}
    if unannotated:
        from .lineage_staleness import list_staleness_scores

        contradicted_ids = _load_contradicted_party_ids(workspace)
        block_ids = [h.get("_id", "") for h in unannotated if h.get("_id")]
        staleness = list_staleness_scores(workspace, block_ids)

    top_score = max(float(h.get("score", 0.0)) for h in hits)
    kept_vectors: list[dict[str, int]] = []

    for hit in hits:
        score = float(hit.get("score", 0.0))
        if top_score > 0:
            informative = round(min(1.0, max(0.0, score / top_score)), 4)
        else:
            informative = 1.0  # no score signal -> neutral

        validity = hit.get("validity")
        if isinstance(validity, dict) and isinstance(validity.get("score"), (int, float)):
            valid = round(float(validity["score"]), 4)
        else:
            valid = float(validity_components(hit, contradicted_ids, staleness)["score"])

        vec = _term_vector(_text_tokens(_get_result_text(hit)))
        max_sim = max((_cosine_similarity(vec, kv) for kv in kept_vectors), default=0.0)
        non_redundant = round(max(0.0, 1.0 - max_sim), 4)
        kept_vectors.append(vec)

        lifecycle = str(hit.get("Lifecycle") or hit.get("lifecycle") or "durable").strip().lower()
        retained = round(_status_component(hit) * LIFECYCLE_RETENTION.get(lifecycle, 1.0), 4)

        hit["feedback_credit"] = {
            "informative": informative,
            "valid": valid,
            "non_redundant": non_redundant,
            "retained": retained,
        }

    _log.debug("feedback_credit_applied", hits=len(hits))


# ---------------------------------------------------------------------------
# Recall-sufficiency score (Group I item 2, flag-gated)
# ---------------------------------------------------------------------------


def recall_sufficiency(
    hits: list[dict[str, Any]],
    intent_type: str,
) -> dict[str, Any] | None:
    """ONE [0,1] float: did this recall deliver enough on-task durable
    context for this query class (Group I item 2).

    Sums each credited hit's useful-context mass (the product of its four
    Stage 3.1 credit components) and normalizes by INTENT_DEMAND for the
    routed class. Returns {"score", "effective_hits", "demand",
    "intent_type"}, or None when no hit carries a ``feedback_credit``
    dict (Stage 3.1 off) — so flag-off stays byte-identical.

    Deterministic: pure arithmetic over the already-deterministic credits
    plus a constant-table lookup — no clock, no rand, no I/O.
    """
    from ._recall_constants import DEFAULT_INTENT_DEMAND, INTENT_DEMAND

    credited = [c for c in (h.get("feedback_credit") for h in hits) if isinstance(c, dict)]
    if not credited:
        return None
    effective = 0.0
    for c in credited:
        mass = 1.0
        for key in ("informative", "valid", "non_redundant", "retained"):
            v = c.get(key)
            mass *= float(v) if isinstance(v, (int, float)) else 1.0
        effective += max(0.0, min(1.0, mass))
    demand = INTENT_DEMAND.get(str(intent_type or "").upper(), DEFAULT_INTENT_DEMAND)
    return {
        "score": round(min(1.0, effective / demand), 4),
        "effective_hits": round(effective, 4),
        "demand": demand,
        "intent_type": str(intent_type or ""),
    }


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
            "SELECT query_text, intent_type, stage_counts, scores, credits FROM retrieval_log "
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
        # Group I item 2: recall-sufficiency score, when Stage 3.2 ran.
        suff_scores: list[float] = []
        suff_by_intent: dict[str, list[float]] = {}
        latest_suff: dict[str, Any] | None = None

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

            # Group I item 2: recall-sufficiency score (rows are newest-first,
            # so the first hit populates `latest_suff`).
            suff = sc.get("sufficiency")
            if isinstance(suff, (int, float)):
                suff_scores.append(float(suff))
                suff_by_intent.setdefault(intent, []).append(float(suff))
                if latest_suff is None:
                    latest_suff = {
                        "score": float(suff),
                        "effective_hits": float(sc.get("sufficiency_effective_hits", 0.0)),
                        "demand": float(sc.get("sufficiency_demand", 0.0)),
                        "intent": intent,
                    }

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

        # --- Feedback-quality credit aggregation (Group I, #Group-I) ---
        # Second pass over `rows` (ORDER BY id DESC, i.e. newest first):
        # accumulate per-component sums and capture the most recent
        # non-empty credits dict for `latest_per_block`.
        credit_queries = 0
        credit_hits = 0
        credit_sums: dict[str, float] = {}
        latest_per_block: dict[str, dict] = {}
        for row in rows:
            try:
                credits = json.loads(row["credits"]) if row["credits"] else {}
            except (json.JSONDecodeError, TypeError):
                credits = {}
            if not isinstance(credits, dict) or not credits:
                continue

            credit_queries += 1
            for block_id, components in credits.items():
                if not isinstance(components, dict):
                    continue
                credit_hits += 1
                for k, v in components.items():
                    if isinstance(v, (int, float)):
                        credit_sums[k] = credit_sums.get(k, 0.0) + float(v)

            if not latest_per_block:
                latest_per_block = credits

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

        # #Group-I: per-hit feedback-quality credit surfaced in diagnostics.
        feedback_quality: dict[str, Any] = {
            "queries_with_credits": credit_queries,
            "hits_credited": credit_hits,
        }
        if credit_hits > 0:
            feedback_quality["avg"] = {k: round(v / credit_hits, 4) for k, v in credit_sums.items()}
            feedback_quality["latest_per_block"] = latest_per_block

        # Group I item 2: recall-sufficiency score summary, additive key.
        # `queries_scored: 0` (and nothing else) when no logged query
        # carried the score — flag-off diagnostics stay a structural no-op.
        from ._recall_constants import SUFFICIENCY_STARVED_THRESHOLD

        recall_sufficiency_summary: dict[str, Any] = {"queries_scored": len(suff_scores)}
        if suff_scores:
            sorted_suff = sorted(suff_scores)
            starved = sum(1 for s in suff_scores if s < SUFFICIENCY_STARVED_THRESHOLD)
            recall_sufficiency_summary.update(
                {
                    "avg": round(sum(suff_scores) / len(suff_scores), 4),
                    "p50": round(sorted_suff[len(sorted_suff) // 2], 4),
                    "min": round(min(suff_scores), 4),
                    "starved_rate": round(starved / len(suff_scores), 4),
                    "by_intent": {
                        i: {"queries": len(scores), "avg": round(sum(scores) / len(scores), 4)} for i, scores in suff_by_intent.items()
                    },
                    "latest": latest_suff,
                }
            )

        return {
            "queries_analyzed": len(rows),
            "intent_distribution": intent_dist,
            "intent_quality": intent_quality_summary,
            "low_confidence_queries": low_confidence_queries[:10],
            "stage_stats": stage_stats,
            "rejection_rates": rejection_rates,
            "score_distribution": score_dist,
            "hard_negatives": hn_summary,
            "feedback_quality": feedback_quality,
            "recall_sufficiency": recall_sufficiency_summary,
        }

    except Exception as exc:
        _log.debug("retrieval_diagnostics_failed", error=str(exc))
        return {"error": str(exc), "queries_analyzed": 0}
    finally:
        if conn:
            try:
                conn.close()
            except Exception as exc2:
                _log.debug("diagnostics_conn_close_failed", error=str(exc2))
