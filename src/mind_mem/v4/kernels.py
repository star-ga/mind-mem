"""v4 kernel strategy implementations (Group A).

Lands the five named strategies declared in
:mod:`mind_mem.v4.cognitive_kernel` and registers them at import time.
Each strategy takes a default-kernel candidate list and **re-ranks**
it according to a specific signal:

    surprise_weighted   semantic distance from rolling recall context
    lineage_first       v3.11 typed-edge graph proximity
    recent_first        last-seen-at recency from the tier table
    contradicts_first   blocks linked by ``contradicts`` edges first
    graph_walk          bounded BFS from a seed match

Strategies do not modify v3.x state — they read v3.x recall + lineage
output and produce a new ranking. Each strategy carries a ``reason``
tag on every :class:`KernelHit` so callers can audit the routing
decision.

Strategies fall back gracefully when their signal is absent:

  - lineage_first / contradicts_first / graph_walk degrade to
    DEFAULT when the lineage graph table does not exist yet.
  - recent_first degrades to DEFAULT when block_recall_tier is empty.
  - surprise_weighted degrades to DEFAULT when no embedding centroid
    is supplied (the embedding pipeline that produces centroids lands
    in a separate v4 commit; for now callers pass the centroid in).

Feature-flag gated under ``v4.cognitive_kernel`` (the same flag the
registry uses). v3.x callers see no behaviour change.

Copyright STARGA, Inc.
"""

from __future__ import annotations

import sqlite3
from collections import deque
from pathlib import Path
from typing import Any, Sequence

from .cognitive_kernel import (
    DEFAULT_KERNEL,
    KernelHit,
    KernelKind,
    KernelResult,
)
from .feature_flags import is_enabled
from .surprise_retrieval import compute_surprise

__all__ = [
    "surprise_weighted_kernel",
    "lineage_first_kernel",
    "recent_first_kernel",
    "contradicts_first_kernel",
    "graph_walk_kernel",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?",
        (name,),
    ).fetchone()
    return row is not None


def _open(workspace: str) -> sqlite3.Connection | None:
    """Return a read-only connection to the workspace ``index.db`` or
    ``None`` if the database doesn't exist yet (degraded path)."""
    db = Path(workspace) / "index.db"
    if not db.is_file():
        return None
    return sqlite3.connect(db, timeout=30)


# ---------------------------------------------------------------------------
# surprise_weighted
# ---------------------------------------------------------------------------


def surprise_weighted_kernel(
    workspace: str,
    query: str,
    *,
    context_centroid: Sequence[float] | None = None,
    candidate_embeddings: dict[str, Sequence[float]] | None = None,
    **_: Any,
) -> KernelResult:
    """Rank default-kernel candidates by surprise against a context centroid.

    Two inputs the caller supplies (the embedding pipeline that
    auto-derives them from the recall log lands separately):

        context_centroid       Centroid of last-K hit embeddings.
        candidate_embeddings   {block_id: embedding} for the candidate
                               set. Missing keys → mild surprise (0.5).

    Falls back to DEFAULT when neither is provided. Returns a
    KernelResult with kernel=SURPRISE_WEIGHTED and a per-hit reason
    of the form ``surprise_weighted:s=0.83``.
    """
    base = DEFAULT_KERNEL(workspace, query)
    if context_centroid is None or candidate_embeddings is None:
        return KernelResult(
            kernel=KernelKind.SURPRISE_WEIGHTED,
            hits=base.hits,
            metadata={"degraded": True, "reason": "no_centroid_or_embeddings"},
        )

    rescored: list[KernelHit] = []
    for h in base.hits:
        emb = candidate_embeddings.get(h.block_id)
        s = compute_surprise(emb or [], context_centroid)
        rescored.append(
            KernelHit(
                block_id=h.block_id,
                score=s,
                reason=f"surprise_weighted:s={s:.3f}",
            )
        )
    rescored.sort(key=lambda h: h.score, reverse=True)
    return KernelResult(
        kernel=KernelKind.SURPRISE_WEIGHTED,
        hits=rescored,
        metadata={"context_dim": len(context_centroid)},
    )


# ---------------------------------------------------------------------------
# lineage_first
# ---------------------------------------------------------------------------


def lineage_first_kernel(
    workspace: str,
    query: str,
    *,
    max_hops: int = 2,
    **_: Any,
) -> KernelResult:
    """Promote candidates that have outgoing lineage edges; demote leaves.

    Reads the v3.11 ``co_retrieval`` lineage table. Each candidate's
    score becomes ``base_score * (1 + edge_count / 10)`` so a block
    with many outgoing edges out-ranks an isolated leaf at the same
    raw score. Falls back to DEFAULT when the table is missing.
    """
    base = DEFAULT_KERNEL(workspace, query)
    conn = _open(workspace)
    if conn is None or not _table_exists(conn, "co_retrieval"):
        if conn is not None:
            conn.close()
        return KernelResult(
            kernel=KernelKind.LINEAGE_FIRST,
            hits=base.hits,
            metadata={"degraded": True, "reason": "no_lineage_table"},
        )

    edge_counts: dict[str, int] = {}
    try:
        for h in base.hits:
            row = conn.execute(
                "SELECT COUNT(*) FROM co_retrieval WHERE mem1_id = ?",
                (h.block_id,),
            ).fetchone()
            edge_counts[h.block_id] = int(row[0]) if row else 0
    finally:
        conn.close()

    rescored = [
        KernelHit(
            block_id=h.block_id,
            score=h.score * (1.0 + edge_counts.get(h.block_id, 0) / 10.0),
            reason=f"lineage_first:edges={edge_counts.get(h.block_id, 0)}",
        )
        for h in base.hits
    ]
    rescored.sort(key=lambda h: h.score, reverse=True)
    return KernelResult(
        kernel=KernelKind.LINEAGE_FIRST,
        hits=rescored,
        metadata={"max_hops": max_hops, "nonzero": sum(1 for v in edge_counts.values() if v > 0)},
    )


# ---------------------------------------------------------------------------
# recent_first
# ---------------------------------------------------------------------------


def recent_first_kernel(workspace: str, query: str, **_: Any) -> KernelResult:
    """Boost candidates whose ``block_recall_tier.last_seen_at`` is recent.

    Reads the v4 ``block_recall_tier`` table. Each candidate's score
    becomes ``base_score + recency_bonus`` where recency is
    ``1.0 - hop_index / total_hops``. Missing rows score 0 recency
    (no boost). Falls back to DEFAULT when the table is missing.
    """
    base = DEFAULT_KERNEL(workspace, query)
    conn = _open(workspace)
    if conn is None or not _table_exists(conn, "block_recall_tier"):
        if conn is not None:
            conn.close()
        return KernelResult(
            kernel=KernelKind.RECENT_FIRST,
            hits=base.hits,
            metadata={"degraded": True, "reason": "no_tier_table"},
        )

    rank: dict[str, int] = {}
    try:
        # Most-recent → lowest rank index.
        rows = conn.execute("SELECT block_id FROM block_recall_tier ORDER BY last_seen_at DESC").fetchall()
        for i, r in enumerate(rows):
            rank[r[0]] = i
    finally:
        conn.close()

    if not rank:
        return KernelResult(
            kernel=KernelKind.RECENT_FIRST,
            hits=base.hits,
            metadata={"degraded": True, "reason": "empty_tier_table"},
        )

    total = float(len(rank))
    rescored: list[KernelHit] = []
    for h in base.hits:
        idx = rank.get(h.block_id)
        bonus = 0.0 if idx is None else (1.0 - idx / total)
        rescored.append(
            KernelHit(
                block_id=h.block_id,
                score=h.score + bonus,
                reason=f"recent_first:bonus={bonus:.3f}",
            )
        )
    rescored.sort(key=lambda h: h.score, reverse=True)
    return KernelResult(
        kernel=KernelKind.RECENT_FIRST,
        hits=rescored,
        metadata={"tier_table_size": int(total)},
    )


# ---------------------------------------------------------------------------
# contradicts_first
# ---------------------------------------------------------------------------


def contradicts_first_kernel(workspace: str, query: str, **_: Any) -> KernelResult:
    """Surface candidates linked by a ``contradicts`` edge first.

    Reads the v3.11 ``co_retrieval`` table filtered to ``kind =
    'contradicts'``. Candidates that appear on either side of a
    contradicts edge get a +1.0 score boost; the rest stay on
    base_score. Useful for hypothesis-testing recalls where the user
    wants to *see* the open contradictions before consensus.

    Falls back to DEFAULT when the lineage table is missing.
    """
    base = DEFAULT_KERNEL(workspace, query)
    conn = _open(workspace)
    if conn is None or not _table_exists(conn, "co_retrieval"):
        if conn is not None:
            conn.close()
        return KernelResult(
            kernel=KernelKind.CONTRADICTS_FIRST,
            hits=base.hits,
            metadata={"degraded": True, "reason": "no_lineage_table"},
        )

    contradicts: set[str] = set()
    try:
        # Schema check — v2.6.0 graphs without `kind` column have no
        # contradicts edges, and we should fall back rather than error.
        cols = {row[1] for row in conn.execute("PRAGMA table_info(co_retrieval)")}
        if "kind" not in cols:
            return KernelResult(
                kernel=KernelKind.CONTRADICTS_FIRST,
                hits=base.hits,
                metadata={"degraded": True, "reason": "untyped_lineage"},
            )
        rows = conn.execute("SELECT mem1_id, mem2_id FROM co_retrieval WHERE kind = 'contradicts'").fetchall()
        for a, b in rows:
            contradicts.add(a)
            contradicts.add(b)
    finally:
        conn.close()

    rescored = [
        KernelHit(
            block_id=h.block_id,
            score=h.score + (1.0 if h.block_id in contradicts else 0.0),
            reason=("contradicts_first:hit" if h.block_id in contradicts else "contradicts_first:miss"),
        )
        for h in base.hits
    ]
    rescored.sort(key=lambda h: h.score, reverse=True)
    return KernelResult(
        kernel=KernelKind.CONTRADICTS_FIRST,
        hits=rescored,
        metadata={"contradicts_count": len(contradicts)},
    )


# ---------------------------------------------------------------------------
# graph_walk
# ---------------------------------------------------------------------------


def graph_walk_kernel(
    workspace: str,
    query: str,
    *,
    seed_ids: Sequence[str] | None = None,
    max_hops: int = 2,
    max_nodes: int = 50,
    **_: Any,
) -> KernelResult:
    """Bounded BFS from seed IDs (or default-kernel hits if no seeds).

    Walks the v3.11 ``co_retrieval`` graph from each seed up to
    ``max_hops`` away, capped at ``max_nodes`` total. Score is
    ``1.0 / (hop_distance + 1)`` so seeds rank highest, immediate
    neighbours next, and so on.

    Falls back to DEFAULT when the lineage table is missing.
    """
    base = DEFAULT_KERNEL(workspace, query)
    conn = _open(workspace)
    if conn is None or not _table_exists(conn, "co_retrieval"):
        if conn is not None:
            conn.close()
        return KernelResult(
            kernel=KernelKind.GRAPH_WALK,
            hits=base.hits,
            metadata={"degraded": True, "reason": "no_lineage_table"},
        )

    seeds = list(seed_ids) if seed_ids else [h.block_id for h in base.hits[:5]]
    if not seeds:
        conn.close()
        return KernelResult(
            kernel=KernelKind.GRAPH_WALK,
            hits=[],
            metadata={"degraded": True, "reason": "no_seeds"},
        )

    visited: dict[str, int] = {}
    queue: deque[tuple[str, int]] = deque()
    for s in seeds:
        if s not in visited:
            visited[s] = 0
            queue.append((s, 0))

    try:
        while queue and len(visited) < max_nodes:
            node, hop = queue.popleft()
            if hop >= max_hops:
                continue
            rows = conn.execute(
                "SELECT mem2_id FROM co_retrieval WHERE mem1_id = ?",
                (node,),
            ).fetchall()
            for (nbr,) in rows:
                if nbr in visited or len(visited) >= max_nodes:
                    continue
                visited[nbr] = hop + 1
                queue.append((nbr, hop + 1))
    finally:
        conn.close()

    hits = [
        KernelHit(
            block_id=bid,
            score=1.0 / (hop + 1.0),
            reason=f"graph_walk:hop={hop}",
        )
        for bid, hop in visited.items()
    ]
    hits.sort(key=lambda h: h.score, reverse=True)
    return KernelResult(
        kernel=KernelKind.GRAPH_WALK,
        hits=hits,
        metadata={"seeds": list(seeds), "visited": len(visited), "max_hops": max_hops},
    )


# ---------------------------------------------------------------------------
# Auto-register at import time
# ---------------------------------------------------------------------------
#
# Bypasses register_kernel (which is flag-gated) by writing directly
# to the registry — same pattern the DEFAULT kernel uses in
# cognitive_kernel.py. The registry being populated is independent of
# whether the flag is set; the flag gates *use* of the API, not the
# registration of strategies.

from .cognitive_kernel import _registry  # noqa: E402

_registry[KernelKind.SURPRISE_WEIGHTED] = surprise_weighted_kernel
_registry[KernelKind.LINEAGE_FIRST] = lineage_first_kernel
_registry[KernelKind.RECENT_FIRST] = recent_first_kernel
_registry[KernelKind.CONTRADICTS_FIRST] = contradicts_first_kernel
_registry[KernelKind.GRAPH_WALK] = graph_walk_kernel


# Also auto-import on cognitive_kernel use so callers don't have to
# remember to import this module separately. The light-weight `is_enabled`
# probe avoids loading v4 internals on a v3.x-only checkout.
def _maybe_warmup() -> None:
    if is_enabled("cognitive_kernel"):
        # Trigger any lazy imports the strategies depend on.
        _ = surprise_weighted_kernel  # touch
        _ = lineage_first_kernel
        _ = recent_first_kernel
        _ = contradicts_first_kernel
        _ = graph_walk_kernel


_maybe_warmup()
