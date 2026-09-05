#!/usr/bin/env python3
"""mind-mem Temporal Causal Graph — directed dependency tracking with staleness.

Extends the co-retrieval graph concept with:
- Directed edges (A depends_on B, A supersedes B)
- Timestamps on all edges
- Staleness propagation — when a block changes, downstream dependents are flagged
- Cycle detection to prevent circular dependencies
- Causal chain queries (full dependency paths)

Usage:
    from .causal_graph import CausalGraph
    graph = CausalGraph(workspace)
    graph.add_edge("D-002", "D-001", "depends_on")
    stale = graph.propagate_staleness("D-001")
    chain = graph.causal_chain("D-003")

Zero external deps — sqlite3, json, os (all stdlib).
"""

from __future__ import annotations

import json
import os
import sqlite3
from collections import deque

from .admission import OP_WRITE, require_admission
from .observability import get_logger, metrics

_log = get_logger("causal_graph")

#: Id namespace a causal edge is admitted under.
#:
#: ``CE`` routes to no corpus file, so a receipt minted for a causal edge
#: cannot be spent on a block. Kept in step with
#: ``governance_gate.ARTIFACT_ID_PREFIXES`` by
#: ``tests/test_governed_artifact_writes.py``.
EDGE_ID_PREFIX = "CE-"


def causal_edge_id(source_id: str, target_id: str, edge_type: str) -> str:
    """The admission subject for the ``(source, type, target)`` edge.

    Deterministic, so re-asserting an edge is recognisably the same
    subject in the chain. The triple is joined on a separator none of
    the three may contain, so two triples cannot collide on one id.
    """
    import hashlib  # noqa: PLC0415 — stdlib, kept off the module import path

    digest = hashlib.sha256("\x00".join((str(source_id), str(edge_type), str(target_id))).encode("utf-8")).hexdigest()[:16]
    return f"{EDGE_ID_PREFIX}{digest}"


# Edge types
EDGE_DEPENDS_ON = "depends_on"
EDGE_SUPERSEDES = "supersedes"
EDGE_INFORMS = "informs"
EDGE_CONTRADICTS = "contradicts"
EDGE_EXTENDS = "extends"

VALID_EDGE_TYPES = frozenset(
    {
        EDGE_DEPENDS_ON,
        EDGE_SUPERSEDES,
        EDGE_INFORMS,
        EDGE_CONTRADICTS,
        EDGE_EXTENDS,
    }
)


def _db_path(workspace: str) -> str:
    return os.path.join(os.path.abspath(workspace), ".mind-mem-index", "causal.db")


def _connect(workspace: str) -> sqlite3.Connection:
    path = _db_path(workspace)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path, timeout=5)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=3000")
    conn.row_factory = sqlite3.Row
    return conn


_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS causal_edges (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id   TEXT NOT NULL,
    target_id   TEXT NOT NULL,
    edge_type   TEXT NOT NULL,
    weight      REAL DEFAULT 1.0,
    created_at  TEXT DEFAULT (datetime('now')),
    updated_at  TEXT DEFAULT (datetime('now')),
    metadata    TEXT DEFAULT '{}',
    UNIQUE(source_id, target_id, edge_type)
);
CREATE INDEX IF NOT EXISTS idx_ce_source ON causal_edges(source_id);
CREATE INDEX IF NOT EXISTS idx_ce_target ON causal_edges(target_id);
CREATE INDEX IF NOT EXISTS idx_ce_type ON causal_edges(edge_type);

CREATE TABLE IF NOT EXISTS staleness_flags (
    block_id    TEXT PRIMARY KEY,
    stale_since TEXT DEFAULT (datetime('now')),
    reason      TEXT DEFAULT '',
    source_id   TEXT DEFAULT ''
);
"""


def _write_causal_edge(
    conn: sqlite3.Connection,
    edge_id: str,
    source_id: str,
    target_id: str,
    edge_type: str,
    weight: float,
    metadata: dict | None,
) -> None:
    """Upsert one causal edge — the seam, and the only writer.

    Calls :func:`~mind_mem.admission.require_admission` before the SQL
    runs, so a caller with no open scope fails closed rather than
    landing the edge and being caught later by a scan.
    """
    require_admission(edge_id, status=None, operation=OP_WRITE)
    conn.execute(
        "INSERT INTO causal_edges (source_id, target_id, edge_type, weight, metadata) "
        "VALUES (?, ?, ?, ?, ?) "
        "ON CONFLICT(source_id, target_id, edge_type) DO UPDATE SET "
        "weight = ?, updated_at = datetime('now'), metadata = ?",
        (
            source_id,
            target_id,
            edge_type,
            weight,
            json.dumps(metadata or {}),
            weight,
            json.dumps(metadata or {}),
        ),
    )


class CausalEdge:
    """A directed edge in the causal graph."""

    __slots__ = ("source_id", "target_id", "edge_type", "weight", "created_at", "updated_at")

    def __init__(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
        weight: float = 1.0,
        created_at: str = "",
        updated_at: str = "",
    ):
        self.source_id = source_id
        self.target_id = target_id
        self.edge_type = edge_type
        self.weight = weight
        self.created_at = created_at
        self.updated_at = updated_at

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type,
            "weight": self.weight,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class CausalGraph:
    """Directed temporal causal graph for block dependencies."""

    def __init__(self, workspace: str) -> None:
        self.workspace = os.path.realpath(workspace)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        conn = _connect(self.workspace)
        try:
            conn.executescript(_SCHEMA_SQL)
            conn.commit()
        finally:
            conn.close()

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
        *,
        weight: float = 1.0,
        metadata: dict | None = None,
    ) -> CausalEdge:
        """Add a directed edge: source depends_on/supersedes/etc. target.

        Args:
            source_id: The block that has the dependency.
            target_id: The block being depended on.
            edge_type: One of VALID_EDGE_TYPES.
            weight: Edge weight (default 1.0).
            metadata: Optional metadata dict.

        Returns:
            The created CausalEdge.

        **Governed since 5.0.2 (ROW-7).** A causal edge is content: it
        is asserted from outside, stored, and served back through
        ``causal_chain`` / ``dependencies`` / ``stale_blocks``, and a
        ``supersedes`` or ``contradicts`` edge changes how downstream
        blocks are treated. Measured before this change: one call moved
        the evidence chain +0 and the hash chain +0. One
        :meth:`~mind_mem.governance_gate.GovernanceGate.admit_artifact`
        scope per edge, keyed on :func:`causal_edge_id`; a refused
        authorisation writes nothing and the transaction rolls back.

        Raises:
            ValueError: If edge_type is invalid or would create a cycle.
            GovernanceBypassError: The gate refused the admission.
        """
        from .governance_gate import get_gate  # noqa: PLC0415 — the gate imports half the package

        if edge_type not in VALID_EDGE_TYPES:
            raise ValueError(f"Invalid edge type '{edge_type}'. Must be one of: {sorted(VALID_EDGE_TYPES)}")

        if source_id == target_id:
            raise ValueError("Self-loops are not allowed")

        edge_id = causal_edge_id(source_id, target_id, edge_type)

        # Cycle-check and insert share a single BEGIN IMMEDIATE
        # transaction so concurrent callers cannot both pass the
        # cycle check and then race to insert complementary edges.
        conn = _connect(self.workspace)
        try:
            conn.execute("BEGIN IMMEDIATE")
            if self._would_create_cycle_on_conn(conn, source_id, target_id):
                conn.rollback()
                raise ValueError(f"Adding edge {source_id} → {target_id} would create a cycle")
            # The scope opens INSIDE the transaction, so a refusal leaves
            # the BEGIN IMMEDIATE unc­ommitted and the edge is not there.
            with get_gate(self.workspace).admit_artifact(
                edge_id,
                f"{source_id}\t{edge_type}\t{target_id}\t{weight}",
                target_file=".mind-mem-index/causal.db",
                metadata={
                    "door": "causal_graph.CausalGraph.add_edge",
                    "edge_type": edge_type,
                    "source_id": str(source_id),
                    "target_id": str(target_id),
                },
            ):
                _write_causal_edge(conn, edge_id, source_id, target_id, edge_type, weight, metadata)
                conn.commit()
        finally:
            conn.close()

        _log.info("causal_edge_added", source=source_id, target=target_id, type=edge_type)
        metrics.inc("causal_edges_added")

        return CausalEdge(source_id, target_id, edge_type, weight)

    def remove_edge(self, source_id: str, target_id: str, edge_type: str) -> bool:
        """Remove a directed edge.

        Returns True if an edge was removed, False if it didn't exist.
        """
        conn = _connect(self.workspace)
        try:
            cursor = conn.execute(
                "DELETE FROM causal_edges WHERE source_id = ? AND target_id = ? AND edge_type = ?",
                (source_id, target_id, edge_type),
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def _would_create_cycle(self, source_id: str, target_id: str) -> bool:
        """Check if adding source → target would create a cycle.

        A cycle exists if target can already reach source via existing edges.
        """
        conn = _connect(self.workspace)
        try:
            return self._would_create_cycle_on_conn(conn, source_id, target_id)
        finally:
            conn.close()

    def _would_create_cycle_on_conn(
        self,
        conn: sqlite3.Connection,
        source_id: str,
        target_id: str,
    ) -> bool:
        """BFS cycle check sharing *conn* with the caller.

        Used by ``add_edge`` so the check and the subsequent insert run
        in the same BEGIN IMMEDIATE transaction.
        """
        visited: set[str] = set()
        queue: deque[str] = deque([target_id])

        while queue:
            current = queue.popleft()
            if current == source_id:
                return True
            if current in visited:
                continue
            visited.add(current)

            rows = conn.execute(
                "SELECT target_id FROM causal_edges WHERE source_id = ?",
                (current,),
            ).fetchall()
            for row in rows:
                if row["target_id"] not in visited:
                    queue.append(row["target_id"])

        return False

    def dependents(self, block_id: str) -> list[CausalEdge]:
        """Get all blocks that depend on this block (incoming depends_on edges)."""
        conn = _connect(self.workspace)
        try:
            rows = conn.execute(
                "SELECT * FROM causal_edges WHERE target_id = ?",
                (block_id,),
            ).fetchall()
        finally:
            conn.close()

        return [
            CausalEdge(
                row["source_id"],
                row["target_id"],
                row["edge_type"],
                row["weight"],
                row["created_at"],
                row["updated_at"],
            )
            for row in rows
        ]

    def dependencies(self, block_id: str) -> list[CausalEdge]:
        """Get all blocks this block depends on (outgoing edges)."""
        conn = _connect(self.workspace)
        try:
            rows = conn.execute(
                "SELECT * FROM causal_edges WHERE source_id = ?",
                (block_id,),
            ).fetchall()
        finally:
            conn.close()

        return [
            CausalEdge(
                row["source_id"],
                row["target_id"],
                row["edge_type"],
                row["weight"],
                row["created_at"],
                row["updated_at"],
            )
            for row in rows
        ]

    def causal_chain(self, block_id: str, *, max_depth: int = 10) -> list[list[str]]:
        """Trace all causal chains from a block back to its roots.

        Returns a list of paths, each path being a list of block IDs
        from the given block back to a root (block with no dependencies).

        Opens a single SQLite connection for the whole traversal
        rather than one per recursion step; deep graphs would
        otherwise exhaust connection limits and produce inconsistent
        snapshot reads across calls.
        """
        chains: list[list[str]] = []
        conn = _connect(self.workspace)
        try:

            def _adj(current: str) -> list[str]:
                rows = conn.execute(
                    "SELECT target_id FROM causal_edges WHERE source_id = ?",
                    (current,),
                ).fetchall()
                return [row["target_id"] for row in rows]

            def _dfs(current: str, path: list[str], depth: int) -> None:
                if depth > max_depth:
                    chains.append(list(path))
                    return
                deps = _adj(current)
                if not deps:
                    chains.append(list(path))
                    return
                for nxt in deps:
                    if nxt not in path:
                        path.append(nxt)
                        _dfs(nxt, path, depth + 1)
                        path.pop()
                    else:
                        chains.append(list(path))

            _dfs(block_id, [block_id], 0)
        finally:
            conn.close()
        return chains

    def propagate_staleness(self, changed_block_id: str, *, reason: str = "") -> list[str]:
        """When a block changes, mark all downstream dependents as stale.

        Uses BFS to find all blocks that transitively depend on the
        changed block.

        Args:
            changed_block_id: The block that was modified.
            reason: Why the change makes dependents stale.

        Returns:
            List of block IDs marked as stale.
        """
        stale_ids: list[str] = []
        visited: set[str] = set()
        queue: deque[str] = deque([changed_block_id])

        conn = _connect(self.workspace)
        try:
            while queue:
                current = queue.popleft()
                if current in visited:
                    continue
                visited.add(current)

                # Find blocks that depend on current
                rows = conn.execute(
                    "SELECT source_id FROM causal_edges WHERE target_id = ?",
                    (current,),
                ).fetchall()

                for row in rows:
                    dep_id = row["source_id"]
                    if dep_id not in visited:
                        queue.append(dep_id)
                        stale_ids.append(dep_id)

            # Mark all stale blocks
            for sid in stale_ids:
                conn.execute(
                    "INSERT OR REPLACE INTO staleness_flags (block_id, reason, source_id) VALUES (?, ?, ?)",
                    (sid, reason or f"Upstream block {changed_block_id} was modified", changed_block_id),
                )
            conn.commit()
        finally:
            conn.close()

        if stale_ids:
            _log.info(
                "staleness_propagated",
                changed=changed_block_id,
                stale_count=len(stale_ids),
            )
            metrics.inc("staleness_propagations")

        return stale_ids

    def get_stale_blocks(self) -> list[dict]:
        """Get all blocks currently flagged as stale.

        Returns:
            List of dicts with block_id, stale_since, reason, source_id.
        """
        conn = _connect(self.workspace)
        try:
            rows = conn.execute("SELECT * FROM staleness_flags ORDER BY stale_since DESC").fetchall()
        finally:
            conn.close()

        return [dict(row) for row in rows]

    def clear_staleness(self, block_id: str) -> bool:
        """Clear the staleness flag for a block (after review/update).

        Returns True if a flag was cleared.
        """
        conn = _connect(self.workspace)
        try:
            cursor = conn.execute(
                "DELETE FROM staleness_flags WHERE block_id = ?",
                (block_id,),
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def all_edges(self, *, edge_type: str | None = None) -> list[CausalEdge]:
        """Get all edges in the graph, optionally filtered by type."""
        conn = _connect(self.workspace)
        try:
            if edge_type:
                rows = conn.execute(
                    "SELECT * FROM causal_edges WHERE edge_type = ? ORDER BY created_at",
                    (edge_type,),
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM causal_edges ORDER BY created_at").fetchall()
        finally:
            conn.close()

        return [
            CausalEdge(
                row["source_id"],
                row["target_id"],
                row["edge_type"],
                row["weight"],
                row["created_at"],
                row["updated_at"],
            )
            for row in rows
        ]

    def summary(self) -> dict:
        """Get graph summary statistics."""
        conn = _connect(self.workspace)
        try:
            total = conn.execute("SELECT COUNT(*) as cnt FROM causal_edges").fetchone()["cnt"]
            by_type = {}
            for row in conn.execute("SELECT edge_type, COUNT(*) as cnt FROM causal_edges GROUP BY edge_type"):
                by_type[row["edge_type"]] = row["cnt"]

            nodes = set()
            for row in conn.execute("SELECT DISTINCT source_id FROM causal_edges"):
                nodes.add(row["source_id"])
            for row in conn.execute("SELECT DISTINCT target_id FROM causal_edges"):
                nodes.add(row["target_id"])

            stale = conn.execute("SELECT COUNT(*) as cnt FROM staleness_flags").fetchone()["cnt"]
        finally:
            conn.close()

        return {
            "total_edges": total,
            "unique_nodes": len(nodes),
            "edges_by_type": by_type,
            "stale_blocks": stale,
        }
