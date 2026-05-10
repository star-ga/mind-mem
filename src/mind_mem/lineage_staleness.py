"""Lineage→staleness propagation (v3.12.0, Theme C).

When a new block is recorded as a ``contradicts`` of an existing
block, the existing block (and its dependents along the lineage
graph) are no longer reliable. v3.12 wires this together end-to-end:

    1. Caller adds a ``contradicts`` edge via
       :func:`mind_mem.block_lineage.add_block_edge`.
    2. Caller invokes :func:`propagate_lineage_staleness` with the new
       block's id as the seed.
    3. The propagator walks the lineage graph (built from the v3.11.0
       typed edges) with a bounded BFS — capped at
       :data:`mind_mem.block_lineage.LINEAGE_DEPTH_CAP` (3 hops) —
       and writes per-block penalty scores into a new
       ``block_staleness`` table.
    4. The recall reranker (and ``_explain.staleness_penalty``) reads
       this table to demote stale blocks at retrieval time.

The penalty for a block at hop *h* through edge of kind *k* is:

    penalty = KIND_DECAY[k] × HOP_DECAY[h]

with ``HOP_DECAY = (1.0, 0.9, 0.5, 0.2)``. Identical to the v2.6.0
scheme; the difference is that the *seed* is now the contradicted
block (so the seed itself gets ``1.0``) and the *kind multipliers*
come from the edge that pointed at it.

Persistence is idempotent: re-running the propagator with the same
seed updates ``decayed_at`` but does not double-count the score. Each
``(block_id, source_id)`` pair owns at most one row.

Stdlib only (sqlite3, datetime). No external dependencies.
"""

from __future__ import annotations

import datetime as _dt
from typing import Iterable

from .block_lineage import (
    KIND_DECAY,
    LINEAGE_DEPTH_CAP,
    ensure_lineage_schema,
    lineage_adjacency,
)
from .retrieval_graph import _connect, ensure_graph_tables
from .staleness import propagate_staleness as _bfs_propagate

__all__ = [
    "ensure_block_staleness_schema",
    "get_staleness_score",
    "list_staleness_scores",
    "propagate_lineage_staleness",
]


_BLOCK_STALENESS_DDL = """\
CREATE TABLE IF NOT EXISTS block_staleness (
    block_id   TEXT NOT NULL,
    source_id  TEXT NOT NULL,
    score      REAL NOT NULL,
    decayed_at TEXT NOT NULL,
    PRIMARY KEY (block_id, source_id)
);
CREATE INDEX IF NOT EXISTS idx_block_staleness_block ON block_staleness (block_id);
"""


def ensure_block_staleness_schema(workspace: str) -> None:
    """Create the ``block_staleness`` table if missing (idempotent)."""

    ensure_graph_tables(workspace)
    ensure_lineage_schema(workspace)
    conn = _connect(workspace)
    try:
        conn.executescript(_BLOCK_STALENESS_DDL)
        conn.commit()
    finally:
        conn.close()


def _classify_seed_neighbours(workspace: str, source_id: str) -> dict[str, str]:
    """Return ``{neighbour_id: edge_kind}`` for edges *out of* ``source_id``.

    These are the immediate seeds — the blocks contradicted (or
    refined / cited / implemented) by ``source_id``. The kind dictates
    how aggressively staleness propagates from each.
    """

    ensure_lineage_schema(workspace)
    conn = _connect(workspace)
    try:
        rows = conn.execute(
            "SELECT mem2_id, kind FROM co_retrieval WHERE mem1_id = ? "
            "UNION ALL "
            "SELECT mem1_id, kind FROM co_retrieval WHERE mem2_id = ? AND kind != 'cooccurrence'",
            (source_id, source_id),
        ).fetchall()
    finally:
        conn.close()
    out: dict[str, str] = {}
    for nid, kind in rows:
        if nid == source_id:
            continue
        # First-write-wins: the strongest signal kind shows up first
        # because of the SELECT order; the dict-set keeps it.
        out.setdefault(nid, kind)
    return out


def propagate_lineage_staleness(
    workspace: str,
    source_id: str,
    *,
    max_hops: int | None = None,
) -> dict[str, float]:
    """Walk the lineage graph from ``source_id`` and persist penalties.

    Args:
        workspace: Path to the workspace root.
        source_id: The newly-added block whose edges identify the seeds.
            Typically a block of `kind="contradicts"` was just written
            with this id as ``mem1_id``.
        max_hops: Hard cap on hop distance, clamped to
            ``[1, LINEAGE_DEPTH_CAP]``.

    Returns:
        A flat ``{block_id: score}`` map of every block that received
        a non-zero penalty in this pass.
    """

    if not source_id:
        raise ValueError("source_id must be non-empty")

    cap = max_hops if max_hops is not None else LINEAGE_DEPTH_CAP
    cap = max(1, min(int(cap), LINEAGE_DEPTH_CAP))

    ensure_block_staleness_schema(workspace)

    seeds = _classify_seed_neighbours(workspace, source_id)
    if not seeds:
        return {}

    # Build an adjacency map of the lineage graph excluding the
    # source itself; the BFS propagates from each seed independently
    # using kind-specific multipliers.
    adj = lineage_adjacency(workspace)
    adj.pop(source_id, None)
    for v in adj.values():
        while source_id in v:
            v.remove(source_id)

    aggregate: dict[str, float] = {}

    for seed_id, kind in seeds.items():
        kind_mul = KIND_DECAY.get(kind, 0.5)
        plan = _bfs_propagate([seed_id], adj, max_hops=cap)
        for bid, hop_score in plan.scores.items():
            scaled = hop_score * kind_mul
            if scaled > aggregate.get(bid, 0.0):
                aggregate[bid] = scaled

    if not aggregate:
        return aggregate

    now = _dt.datetime.now(_dt.timezone.utc).isoformat()
    conn = _connect(workspace)
    try:
        conn.executemany(
            """
            INSERT INTO block_staleness (block_id, source_id, score, decayed_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(block_id, source_id) DO UPDATE SET
                score = excluded.score,
                decayed_at = excluded.decayed_at
            """,
            [(bid, source_id, score, now) for bid, score in aggregate.items()],
        )
        conn.commit()
    finally:
        conn.close()

    return aggregate


def get_staleness_score(workspace: str, block_id: str) -> float:
    """Return the maximum persisted staleness score for ``block_id``.

    A block may carry penalties from multiple ``source_id`` seeds; the
    most-stale wins. ``0.0`` if no entry exists.
    """

    ensure_block_staleness_schema(workspace)
    conn = _connect(workspace)
    try:
        row = conn.execute(
            "SELECT MAX(score) FROM block_staleness WHERE block_id = ?",
            (block_id,),
        ).fetchone()
    finally:
        conn.close()
    if not row or row[0] is None:
        return 0.0
    return float(row[0])


def list_staleness_scores(workspace: str, block_ids: Iterable[str]) -> dict[str, float]:
    """Return ``{block_id: max_score}`` for the requested ids.

    Missing blocks default to ``0.0``. Cheaper than calling
    :func:`get_staleness_score` once per id.
    """

    ids = [bid for bid in block_ids if bid]
    if not ids:
        return {}
    ensure_block_staleness_schema(workspace)
    # ``placeholders`` is N copies of ``?`` derived from len(ids); no
    # user input is interpolated into the SQL string. The actual ids
    # are bound as parameters below.
    placeholders = ",".join("?" * len(ids))
    conn = _connect(workspace)
    try:
        rows = conn.execute(
            f"SELECT block_id, MAX(score) FROM block_staleness "  # nosec B608
            f"WHERE block_id IN ({placeholders}) GROUP BY block_id",
            ids,
        ).fetchall()
    finally:
        conn.close()
    out = {bid: 0.0 for bid in ids}
    for bid, score in rows:
        out[bid] = float(score) if score is not None else 0.0
    return out
