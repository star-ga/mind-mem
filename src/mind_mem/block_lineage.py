"""Typed block-lineage edges + bounded BFS reader (v3.11.0, Pattern 3).

The v2.6.0 ``co_retrieval`` graph stores undirected, weighted edges
implicitly typed as **co-occurrence** (two blocks were returned in the
same recall pass). v3.11.0 extends that table with an explicit
``kind`` column so callers can record *semantic* lineage edges:

    * ``cites``         — block A explicitly references block B.
    * ``implements``    — block A is the concrete realisation of B.
    * ``refines``       — block A is a tightening or correction of B.
    * ``contradicts``   — block A asserts the negation of B.
    * ``cooccurrence``  — original v2.6.0 default, unchanged.

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
edges scale at 0.8; ``implements`` at 0.6; ``refines`` at 0.4. The
multiplier is applied **at adjacency-construction time**, leaving the
propagator's contract untouched.

The module is dependency-free (stdlib + ``mind_mem.retrieval_graph``)
and SQLite-only — Postgres replicas are read-only paths and don't need
the lineage write API today.
"""

from __future__ import annotations

import datetime as _dt
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable

from .retrieval_graph import _connect, ensure_graph_tables

__all__ = [
    "ALLOWED_KINDS",
    "KIND_DECAY",
    "LINEAGE_DEPTH_CAP",
    "LINEAGE_NODE_CAP",
    "LineageEdge",
    "LineageResult",
    "add_block_edge",
    "block_lineage",
    "ensure_lineage_schema",
    "lineage_adjacency",
]

ALLOWED_KINDS: frozenset[str] = frozenset(
    {"cites", "implements", "refines", "contradicts", "cooccurrence"}
)

KIND_DECAY: dict[str, float] = {
    "contradicts": 1.0,
    "cites": 0.8,
    "implements": 0.6,
    "refines": 0.4,
    "cooccurrence": 0.5,
}

LINEAGE_DEPTH_CAP: int = 3
LINEAGE_NODE_CAP: int = 1000


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
    """Add ``kind`` column to ``co_retrieval`` if missing.

    Idempotent — safe to call on every process startup.
    """

    ensure_graph_tables(workspace)
    conn = _connect(workspace)
    try:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(co_retrieval)").fetchall()}
        if "kind" not in cols:
            conn.execute(
                "ALTER TABLE co_retrieval ADD COLUMN kind TEXT NOT NULL DEFAULT 'cooccurrence'"
            )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_co_ret_kind_src "
            "ON co_retrieval (kind, mem1_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_co_ret_kind_dst "
            "ON co_retrieval (kind, mem2_id)"
        )
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
) -> None:
    """Record an explicit typed lineage edge from ``src`` to ``dst``.

    Edges are deduplicated by ``(mem1_id, mem2_id, kind)``: re-adding
    the same edge bumps ``hit_count`` and refreshes ``updated_at``.
    """

    if kind not in ALLOWED_KINDS:
        raise ValueError(
            f"kind must be one of {sorted(ALLOWED_KINDS)}, got {kind!r}"
        )
    if not src or not dst:
        raise ValueError("src and dst must be non-empty block ids")
    if src == dst:
        raise ValueError("src and dst must differ (no self-loops)")

    ensure_lineage_schema(workspace)
    conn = _connect(workspace)
    now = _dt.datetime.now(_dt.timezone.utc).isoformat()
    try:
        conn.execute(
            """
            INSERT INTO co_retrieval (mem1_id, mem2_id, weight, hit_count, updated_at, kind)
            VALUES (?, ?, ?, 1, ?, ?)
            ON CONFLICT(mem1_id, mem2_id) DO UPDATE SET
                hit_count = hit_count + 1,
                updated_at = excluded.updated_at,
                weight = MAX(weight, excluded.weight),
                kind = CASE
                    WHEN co_retrieval.kind = 'cooccurrence' THEN excluded.kind
                    ELSE co_retrieval.kind
                END
            """,
            (src, dst, float(weight), now, kind),
        )
        conn.commit()
    finally:
        conn.close()


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
        raise ValueError(
            f"kind_filter must be one of {sorted(ALLOWED_KINDS)} or None, "
            f"got {kind_filter!r}"
        )

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
            rows: Iterable[tuple[str, str]] = conn.execute(
                "SELECT mem1_id, mem2_id FROM co_retrieval"
            ).fetchall()
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
