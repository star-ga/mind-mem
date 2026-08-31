"""Typed block-lineage edges + bounded BFS reader (v3.11.0+, Pattern 3).

The v2.6.0 ``co_retrieval`` graph stores undirected, weighted edges
implicitly typed as **co-occurrence** (two blocks were returned in the
same recall pass). v3.11.0 extends that table with an explicit
``kind`` column so callers can record *semantic* lineage edges:

    * ``cites``         — block A explicitly references block B.
    * ``implements``    — block A is the concrete realisation of B.
    * ``refines``       — block A is a tightening or correction of B.
    * ``contradicts``   — block A asserts the negation of B.
    * ``cooccurrence``  — original v2.6.0 default, unchanged.
    * ``supports``      — block A provides evidence or context that
                          strengthens the claim in block B.
    * ``derived_from``  — block A was derived or synthesised from block B
                          (e.g. a summary, a transformed view, an
                          agent-generated distillation).

The migration is zero-downtime: ``ALTER TABLE ... ADD COLUMN kind TEXT
NOT NULL DEFAULT 'cooccurrence'`` makes every existing edge legal under
the new schema without any data movement. Postgres parity is emitted
by the SQLite-first migration in ``schema_migrations.py``.

Two read tools sit on top:

    * :func:`add_block_edge` — write a typed lineage edge.
    * :func:`block_lineage` — bounded BFS that returns
      ``[{block_id, kind, distance, confidence}]`` ordered by ascending
      distance, capped at ``max_depth`` (≤3 by contract) and at
      ``LINEAGE_NODE_CAP`` total nodes (1000 by contract).

Kind-specific decay multipliers feed the existing
:func:`mind_mem.staleness.propagate_staleness` propagator: when a
``contradicts`` edge fires it propagates with full hop-decay; ``cites``
edges scale at 0.8; ``implements`` at 0.6; ``refines`` at 0.4; ``supports``
at 0.7 (strong positive signal, slightly weaker than a direct citation);
``derived_from`` at 0.5 (a derivation is one inferential step removed
from the source). The multiplier is applied **at adjacency-construction
time**, leaving the propagator's contract untouched.

Optional edge-aware recall boost
----------------------------------
:func:`edge_aware_boost` computes a recall-score addend for a block
given the typed edges anchored to it.  The boost is **off by default**
(``weight=0.0``) so existing recall behaviour is fully preserved.  Callers
that opt in pass a non-zero weight:

    boosts = edge_aware_boost(workspace, block_ids, weight=0.1)

The boost is additive: ``final_score = bm25_score + boosts.get(block_id, 0)``.

The module is dependency-free (stdlib + ``mind_mem.retrieval_graph``)
and SQLite-only — Postgres replicas are read-only paths and don't need
the lineage write API today.
"""

from __future__ import annotations

import datetime as _dt
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable

from .project_key import resolve_project_key
from .retrieval_graph import _connect, ensure_graph_tables

__all__ = [
    "ALLOWED_KINDS",
    "KIND_DECAY",
    "LINEAGE_DEPTH_CAP",
    "LINEAGE_NODE_CAP",
    "LineageEdge",
    "LineageResult",
    "NO_ORIGIN_PROJECT",
    "add_block_edge",
    "block_lineage",
    "distinct_project_count",
    "distinct_project_counts",
    "edge_aware_boost",
    "ensure_lineage_schema",
    "lineage_adjacency",
]

ALLOWED_KINDS: frozenset[str] = frozenset({"cites", "implements", "refines", "contradicts", "cooccurrence", "supports", "derived_from"})

KIND_DECAY: dict[str, float] = {
    "contradicts": 1.0,
    "cites": 0.8,
    "supports": 0.7,
    "implements": 0.6,
    "derived_from": 0.5,
    "cooccurrence": 0.5,
    "refines": 0.4,
}

LINEAGE_DEPTH_CAP: int = 3
LINEAGE_NODE_CAP: int = 1000

#: Value stored in ``co_retrieval.origin_project`` when an edge carries no
#: provenance.  It is the column default, so it is also what every row
#: written before the provenance migration holds.  Rows carrying it are
#: excluded from breadth counts — unknown provenance is not a project.
NO_ORIGIN_PROJECT: str = ""


@dataclass(frozen=True)
class LineageEdge:
    """One step in a lineage traversal."""

    block_id: str
    kind: str
    distance: int
    confidence: float

    def to_dict(self) -> dict[str, object]:
        return {
            "block_id": self.block_id,
            "kind": self.kind,
            "distance": self.distance,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class LineageResult:
    """The full bounded-BFS result rooted at ``root``."""

    root: str
    edges: list[LineageEdge] = field(default_factory=list)
    truncated: bool = False
    max_depth: int = LINEAGE_DEPTH_CAP

    def to_dict(self) -> dict[str, object]:
        return {
            "root": self.root,
            "edges": [e.to_dict() for e in self.edges],
            "truncated": self.truncated,
            "max_depth": self.max_depth,
            "count": len(self.edges),
        }


def ensure_lineage_schema(workspace: str) -> None:
    """Add the ``kind`` and ``origin_project`` columns to ``co_retrieval``.

    Idempotent — safe to call on every process startup.  Both migrations
    are ``ALTER TABLE ... ADD COLUMN`` with a default, so every row that
    predates them stays legal and readable without any data movement:
    an edge written before provenance existed simply carries the empty
    :data:`NO_ORIGIN_PROJECT` marker.
    """

    ensure_graph_tables(workspace)
    conn = _connect(workspace)
    try:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(co_retrieval)").fetchall()}
        if "kind" not in cols:
            conn.execute("ALTER TABLE co_retrieval ADD COLUMN kind TEXT NOT NULL DEFAULT 'cooccurrence'")
        if "origin_project" not in cols:
            conn.execute("ALTER TABLE co_retrieval ADD COLUMN origin_project TEXT NOT NULL DEFAULT ''")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_co_ret_kind_src ON co_retrieval (kind, mem1_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_co_ret_kind_dst ON co_retrieval (kind, mem2_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_co_ret_origin_dst ON co_retrieval (mem2_id, origin_project)")
        conn.commit()
    finally:
        conn.close()


def add_block_edge(
    workspace: str,
    src: str,
    dst: str,
    kind: str,
    *,
    weight: float = 1.0,
    origin_project: str | None = None,
) -> None:
    """Record an explicit typed lineage edge from ``src`` to ``dst``.

    Edges are deduplicated by ``(mem1_id, mem2_id, kind)``: re-adding
    the same edge bumps ``hit_count`` and refreshes ``updated_at``.

    Args:
        origin_project: Which project this assertion originates from —
            the provenance corroboration *breadth* is counted over.
            ``None`` (default) resolves it from the process working
            directory via :func:`mind_mem.project_key.resolve_project_key`,
            because the originating project is the repository the writing
            session is working in, not the workspace the store lives in
            (a fleet sharing one store would otherwise report one
            project for every writer).  Pass an explicit key to override,
            or :data:`NO_ORIGIN_PROJECT` to record no provenance at all.
            Resolution never raises into this write path.
    """

    if kind not in ALLOWED_KINDS:
        raise ValueError(f"kind must be one of {sorted(ALLOWED_KINDS)}, got {kind!r}")
    if not src or not dst:
        raise ValueError("src and dst must be non-empty block ids")
    if src == dst:
        raise ValueError("src and dst must differ (no self-loops)")

    project = resolve_project_key() if origin_project is None else str(origin_project)

    ensure_lineage_schema(workspace)
    conn = _connect(workspace)
    now = _dt.datetime.now(_dt.timezone.utc).isoformat()
    try:
        # `origin_project` follows the same first-writer-wins rule as `kind`:
        # a row that has no provenance yet (pre-migration rows, and rows
        # written by the co-occurrence logger) adopts the incoming key, and a
        # row that already names a project keeps it.
        #
        # deferred: an edge re-asserted from a *second* project keeps only the
        # first key, so breadth under-counts that case — the conservative
        # direction. It is not the ONLY direction available -- see the resolution
        # ladder in project_key: an unmarked non-repository tree can still read
        # two sibling directories as two projects -- but it is the one this path
        # takes. Upgrade
        # path: carry the set in a side table keyed (mem1_id, mem2_id, project)
        # rather than widening this row, which would change the primary key.
        conn.execute(
            """
            INSERT INTO co_retrieval (mem1_id, mem2_id, weight, hit_count, updated_at, kind, origin_project)
            VALUES (?, ?, ?, 1, ?, ?, ?)
            ON CONFLICT(mem1_id, mem2_id) DO UPDATE SET
                hit_count = hit_count + 1,
                updated_at = excluded.updated_at,
                weight = MAX(weight, excluded.weight),
                kind = CASE
                    WHEN co_retrieval.kind = 'cooccurrence' THEN excluded.kind
                    ELSE co_retrieval.kind
                END,
                origin_project = CASE
                    WHEN co_retrieval.origin_project = '' THEN excluded.origin_project
                    ELSE co_retrieval.origin_project
                END
            """,
            (src, dst, float(weight), now, kind, project),
        )
        conn.commit()
    finally:
        conn.close()


def distinct_project_counts(
    workspace: str,
    block_ids: list[str] | None = None,
    *,
    kind_filter: str | None = None,
) -> dict[str, int]:
    """Count distinct originating projects among each block's incoming edges.

    "Incoming" has the same meaning as in :func:`edge_aware_boost`: a row
    whose ``mem2_id`` is the block, i.e. another block pointing at this
    one.  Edges with no recorded provenance (:data:`NO_ORIGIN_PROJECT`)
    are excluded — unknown provenance is not evidence of a project.

    Args:
        workspace: Workspace root path.
        block_ids: Restrict to these blocks.  ``None`` (default) counts
            every block that has at least one provenance-carrying
            incoming edge — and is the right call for a whole-corpus
            sweep, since each id here becomes a bind parameter and
            SQLite caps how many one statement may carry.
        kind_filter: Restrict to one edge kind.  ``None`` counts all
            kinds, matching the untyped incoming-edge count that
            :func:`mind_mem.block_maturity.maturity_score` is given.

    Returns:
        ``block_id -> distinct project count``.  A block with no
        provenance-carrying incoming edge is absent from the mapping
        (equivalently, counts zero).
    """
    if kind_filter is not None and kind_filter not in ALLOWED_KINDS:
        raise ValueError(f"kind_filter must be one of {sorted(ALLOWED_KINDS)} or None, got {kind_filter!r}")
    if block_ids is not None and not block_ids:
        return {}

    ensure_lineage_schema(workspace)
    conn = _connect(workspace)
    try:
        clauses = ["origin_project != ?"]
        params: list[object] = [NO_ORIGIN_PROJECT]
        if kind_filter is not None:
            clauses.append("kind = ?")
            params.append(kind_filter)
        if block_ids is not None:
            id_set = sorted(set(block_ids))
            clauses.append(f"mem2_id IN ({','.join('?' * len(id_set))})")
            params.extend(id_set)
        rows = conn.execute(
            "SELECT mem2_id, COUNT(DISTINCT origin_project) FROM co_retrieval "  # nosec B608 — every interpolated fragment is "?" bind placeholders; values go in params
            f"WHERE {' AND '.join(clauses)} GROUP BY mem2_id",
            tuple(params),
        ).fetchall()
    finally:
        conn.close()

    return {str(dst): int(count) for dst, count in rows}


def distinct_project_count(
    workspace: str,
    block_id: str,
    *,
    kind_filter: str | None = None,
) -> int:
    """Distinct originating projects among *block_id*'s incoming edges."""
    if not block_id:
        raise ValueError("block_id must be non-empty")
    return distinct_project_counts(workspace, [block_id], kind_filter=kind_filter).get(block_id, 0)


def _outgoing(workspace: str, block_id: str, *, kind_filter: str | None = None) -> list[tuple[str, str]]:
    """Return list of ``(neighbour_id, kind)`` outgoing from ``block_id``."""
    ensure_lineage_schema(workspace)
    conn = _connect(workspace)
    try:
        if kind_filter is None:
            rows = conn.execute(
                "SELECT mem2_id, kind FROM co_retrieval WHERE mem1_id = ? "
                "UNION ALL "
                "SELECT mem1_id, kind FROM co_retrieval WHERE mem2_id = ?",
                (block_id, block_id),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT mem2_id, kind FROM co_retrieval WHERE mem1_id = ? AND kind = ? "
                "UNION ALL "
                "SELECT mem1_id, kind FROM co_retrieval WHERE mem2_id = ? AND kind = ?",
                (block_id, kind_filter, block_id, kind_filter),
            ).fetchall()
        return [(r[0], r[1]) for r in rows]
    finally:
        conn.close()


def block_lineage(
    workspace: str,
    block_id: str,
    max_depth: int = LINEAGE_DEPTH_CAP,
    *,
    kind_filter: str | None = None,
    node_cap: int = LINEAGE_NODE_CAP,
) -> LineageResult:
    """BFS-traverse the lineage graph rooted at ``block_id``.

    Bounded by:
        * ``max_depth`` — clamped to ``[1, LINEAGE_DEPTH_CAP]``.
        * ``node_cap`` — hard cap on total nodes returned. Reaching the
          cap sets ``LineageResult.truncated`` so callers know the
          traversal was incomplete.

    Edges are returned in **ascending distance** (1-hop first), and
    within a distance bucket, in deterministic insertion order.
    """

    if not block_id:
        raise ValueError("block_id must be non-empty")
    if kind_filter is not None and kind_filter not in ALLOWED_KINDS:
        raise ValueError(f"kind_filter must be one of {sorted(ALLOWED_KINDS)} or None, got {kind_filter!r}")

    depth = max(1, min(int(max_depth), LINEAGE_DEPTH_CAP))
    cap = max(1, int(node_cap))

    visited: set[str] = {block_id}
    queue: deque[tuple[str, int, str]] = deque()
    edges: list[LineageEdge] = []
    truncated = False

    for neighbour, kind in _outgoing(workspace, block_id, kind_filter=kind_filter):
        if neighbour == block_id or neighbour in visited:
            continue
        visited.add(neighbour)
        queue.append((neighbour, 1, kind))
        edges.append(
            LineageEdge(
                block_id=neighbour,
                kind=kind,
                distance=1,
                confidence=KIND_DECAY.get(kind, 0.5),
            )
        )
        if len(edges) >= cap:
            truncated = True
            break

    while queue and not truncated:
        node, hop, _kind = queue.popleft()
        if hop >= depth:
            continue
        for neighbour, n_kind in _outgoing(workspace, node, kind_filter=kind_filter):
            if neighbour in visited:
                continue
            visited.add(neighbour)
            next_hop = hop + 1
            queue.append((neighbour, next_hop, n_kind))
            confidence = KIND_DECAY.get(n_kind, 0.5) * (0.5 ** (next_hop - 1))
            edges.append(
                LineageEdge(
                    block_id=neighbour,
                    kind=n_kind,
                    distance=next_hop,
                    confidence=confidence,
                )
            )
            if len(edges) >= cap:
                truncated = True
                break

    edges.sort(key=lambda e: (e.distance, e.block_id))
    return LineageResult(
        root=block_id,
        edges=edges,
        truncated=truncated,
        max_depth=depth,
    )


def lineage_adjacency(
    workspace: str,
    *,
    kind_filter: str | None = None,
) -> dict[str, list[str]]:
    """Build a flat undirected adjacency map for the staleness propagator.

    Strips the ``kind`` from each edge — the kind-specific decay
    is applied separately by callers via :data:`KIND_DECAY`.
    """

    ensure_lineage_schema(workspace)
    conn = _connect(workspace)
    try:
        if kind_filter is None:
            rows: Iterable[tuple[str, str]] = conn.execute("SELECT mem1_id, mem2_id FROM co_retrieval").fetchall()
        else:
            rows = conn.execute(
                "SELECT mem1_id, mem2_id FROM co_retrieval WHERE kind = ?",
                (kind_filter,),
            ).fetchall()
        adj: dict[str, list[str]] = {}
        for src, dst in rows:
            adj.setdefault(src, []).append(dst)
            adj.setdefault(dst, []).append(src)
        return adj
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Edge-aware recall boost (Group H typed-edges, behavior-preserving default)
# ---------------------------------------------------------------------------

#: Per-kind boost multipliers applied inside :func:`edge_aware_boost`.
#: ``supports`` and ``derived_from`` receive positive (but modest) boosts;
#: ``contradicts`` receives zero so contradiction edges do not inflate
#: the score of the target block.
EDGE_BOOST_WEIGHT: dict[str, float] = {
    "supports": 1.0,
    "cites": 0.8,
    "derived_from": 0.6,
    "implements": 0.5,
    "refines": 0.4,
    "cooccurrence": 0.2,
    "contradicts": 0.0,
}


def edge_aware_boost(
    workspace: str,
    block_ids: list[str],
    *,
    weight: float = 0.0,
) -> dict[str, float]:
    """Compute an additive score boost for each block based on typed edges.

    The boost is **off by default** (``weight=0.0``) so callers that do
    not opt in receive the same scores as before this feature existed.
    Callers that want edge-aware ranking pass a small positive weight
    (e.g. ``0.05``–``0.15``).

    The boost for a block is the weighted sum of incoming ``supports`` and
    ``derived_from`` edges (which indicate that other blocks actively
    reinforce this block's claim), plus a smaller contribution from other
    kinds.  ``contradicts`` edges contribute zero — they do not inflate
    the score of the target.

    Args:
        workspace: Workspace root path.
        block_ids: Block IDs whose boosts are requested.
        weight: Global scale factor.  ``0.0`` (default) disables the
            feature entirely; ``0.1`` is a reasonable starting point.

    Returns:
        Dict mapping block_id → additive boost value.  Missing keys mean
        zero boost.  All values are non-negative.
    """
    if weight == 0.0 or not block_ids:
        return {}

    ensure_lineage_schema(workspace)
    conn = _connect(workspace)
    try:
        # Build a set for fast membership check; bind each as "?" in the query.
        id_set = set(block_ids)
        placeholders = ",".join("?" * len(id_set))
        # Query edges where any of the target blocks appears as the *destination*
        # (mem2_id).  An incoming edge means another block points at this block,
        # which is the signal we boost on (being cited / supported / derived-from
        # by others is a quality signal).
        rows = conn.execute(
            f"SELECT mem2_id, kind FROM co_retrieval WHERE mem2_id IN ({placeholders})",  # nosec B608 — placeholders are "?" bind params; id_set values are passed as the params tuple
            tuple(id_set),
        ).fetchall()
    finally:
        conn.close()

    boosts: dict[str, float] = {}
    for dst, kind in rows:
        per_kind = EDGE_BOOST_WEIGHT.get(kind, 0.0)
        if per_kind > 0.0:
            boosts[dst] = boosts.get(dst, 0.0) + per_kind * weight

    return boosts
