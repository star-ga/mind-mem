"""v4 kernel strategy implementations (Group A).

Lands the four named strategies declared in
:mod:`mind_mem.v4.cognitive_kernel` and registers them at import time.
Each strategy takes a default-kernel candidate list and **re-ranks**
it according to a specific signal:

    surprise_weighted   semantic distance from rolling recall context
    lineage_first       v3.11 typed-edge graph proximity
    contradicts_first   blocks linked by ``contradicts`` edges first
    graph_walk          bounded BFS from a seed match

Strategies do not modify v3.x state — they read v3.x recall + lineage
output and produce a new ranking. Each strategy carries a ``reason``
tag on every :class:`KernelHit` so callers can audit the routing
decision.

Strategies fall back gracefully when their signal is absent:

  - lineage_first / contradicts_first / graph_walk degrade to
    DEFAULT when the lineage graph table does not exist yet. They read
    the ``co_retrieval`` graph from the workspace
    ``.mind-mem-index/recall.db`` written by mind_mem.retrieval_graph /
    mind_mem.block_lineage.
  - surprise_weighted degrades to DEFAULT when no embedding centroid
    is supplied (the embedding pipeline that produces centroids lands
    in a separate v4 commit; for now callers pass the centroid in).

Feature-flag gated under ``v4.cognitive_kernel`` (the same flag the
registry uses). v3.x callers see no behaviour change.

**Importing this module is what registers the strategies.** The
registry lives in :mod:`mind_mem.v4.cognitive_kernel`, and that module
does not import this one, so::

    from mind_mem.v4.cognitive_kernel import mind_recall
    mind_recall(ws, "q", kernel="graph_walk")   # KeyError

raises until ``mind_mem.v4.kernels`` has been imported at least once.
Callers that want the four named strategies must import it explicitly::

    import mind_mem.v4.kernels  # noqa: F401 — registers the strategies
    from mind_mem.v4.cognitive_kernel import mind_recall

Copyright STARGA, Inc.
"""

from __future__ import annotations

import sqlite3
from collections import deque
from pathlib import Path
from typing import Any, Sequence

from ..retrieval_graph import _db_path as _lineage_db_path
from .cognitive_kernel import (
    DEFAULT_KERNEL,
    KernelHit,
    KernelKind,
    KernelResult,
)
from .surprise_retrieval import compute_surprise

__all__ = [
    "surprise_weighted_kernel",
    "lineage_first_kernel",
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


def _open_lineage(workspace: str) -> sqlite3.Connection | None:
    """Return a connection to the workspace lineage graph database, or
    ``None`` if it doesn't exist yet (degraded path).

    The typed ``co_retrieval`` graph is **not** in the ``index.db`` the
    other v4 modules use for block state -- it is written by
    :mod:`mind_mem.retrieval_graph` / :mod:`mind_mem.block_lineage` into
    ``<workspace>/.mind-mem-index/recall.db``. The path is resolved
    through the writer's own helper so this reader can never drift away
    from wherever the writer puts it.

    The path is only ever *read*: unlike the writer's ``_connect`` this
    never creates the directory or the file, so calling a lineage
    strategy on a workspace that has no graph yet degrades instead of
    leaving an empty database behind.
    """
    db = Path(_lineage_db_path(workspace))
    if not db.is_file():
        return None
    return sqlite3.connect(db, timeout=30)


def _open_lineage_graph(workspace: str) -> sqlite3.Connection | None:
    """Open the lineage database *and* confirm ``co_retrieval`` is present.

    Returns ``None`` on the two degraded cases the strategies document —
    no database file, or a database without the graph table — and a live
    connection otherwise.

    The table probe used to run inside each caller's guard expression
    (``if conn is None or not _table_exists(conn, ...)``), i.e. *before*
    the ``try/finally`` that closes the connection. A corrupt
    ``recall.db`` makes that first query raise, so every call leaked one
    connection — and with it the ``.db``/``-wal``/``-shm`` descriptors,
    which on Windows blocks deleting or replacing the workspace. Opening
    and probing are one operation here so there is no window in which a
    connection exists outside a handler that closes it.
    """
    conn = _open_lineage(workspace)
    if conn is None:
        return None
    try:
        has_table = _table_exists(conn, "co_retrieval")
    except BaseException:
        conn.close()
        raise
    if not has_table:
        conn.close()
        return None
    return conn


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

    Reads the v3.11 ``co_retrieval`` lineage table (in the workspace
    ``.mind-mem-index/recall.db``, not ``index.db``). Each candidate's
    score becomes ``base_score * (1 + edge_count / 10)`` so a block
    with many outgoing edges out-ranks an isolated leaf at the same
    raw score. Falls back to DEFAULT when the table is missing.
    """
    base = DEFAULT_KERNEL(workspace, query)
    conn = _open_lineage_graph(workspace)
    if conn is None:
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
# contradicts_first
# ---------------------------------------------------------------------------


def contradicts_first_kernel(workspace: str, query: str, **_: Any) -> KernelResult:
    """Surface candidates linked by a ``contradicts`` edge first.

    Reads the v3.11 ``co_retrieval`` table (in the workspace
    ``.mind-mem-index/recall.db``, not ``index.db``) filtered to ``kind =
    'contradicts'``. Candidates that appear on either side of a
    contradicts edge get a +1.0 score boost; the rest stay on
    base_score. Useful for hypothesis-testing recalls where the user
    wants to *see* the open contradictions before consensus.

    Falls back to DEFAULT when the lineage table is missing.
    """
    base = DEFAULT_KERNEL(workspace, query)
    conn = _open_lineage_graph(workspace)
    if conn is None:
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

    Walks the v3.11 ``co_retrieval`` graph (in the workspace
    ``.mind-mem-index/recall.db``, not ``index.db``) from each seed up to
    ``max_hops`` away, capped at ``max_nodes`` total. Score is
    ``1.0 / (hop_distance + 1)`` so seeds rank highest, immediate
    neighbours next, and so on.

    Falls back to DEFAULT when the lineage table is missing.
    """
    base = DEFAULT_KERNEL(workspace, query)
    conn = _open_lineage_graph(workspace)
    if conn is None:
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
_registry[KernelKind.CONTRADICTS_FIRST] = contradicts_first_kernel
_registry[KernelKind.GRAPH_WALK] = graph_walk_kernel


# There is deliberately no "warm-up" hook below this point. A previous
# ``_maybe_warmup()`` was defined *in this module* and called *from this
# module*, so it could only ever run once this module had already been
# imported — the auto-import its own comment promised was structurally
# impossible, and its body only rebound four already-bound names. It has
# been removed rather than left as a hook that looks load-bearing.
#
# Registration is therefore import-driven, and importing
# ``mind_mem.v4.cognitive_kernel`` alone is NOT enough: nothing in that
# module (or in ``mind_mem.v4.__init__``) imports this one, so
# ``available_kernels()`` returns ``[DEFAULT]`` and ``mind_recall(...,
# kernel="graph_walk")`` raises KeyError until a caller does::
#
#     import mind_mem.v4.kernels  # noqa: F401 — registers the strategies
#
# See the module docstring.
