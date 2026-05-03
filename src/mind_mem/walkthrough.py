"""Dependency-ordered walkthrough — `compile_walkthrough` (v3.9.0 candidate).

For *teach me / explain* queries, agents and humans both want a *learning
order* — foundations first, then derived context, then current state.
``recall`` and ``hybrid_search`` already return blocks ranked by
relevance; this module re-orders them so the result is a sequence
("step 1, step 2, ...") instead of a flat scored set.

Algorithm:

1. Call ``recall(topic, limit=N)`` to get candidate blocks.
2. Build a *dependency graph* over the candidates:

   * **Chronological backbone** — derived from each block's id (the
     mind-mem id format embeds a YYYYMMDD prefix). Earlier dates are
     foundations; later dates depend on them.
   * **Co-retrieval edges** — when the
     ``intelligence/state/retrieval_graph.db`` co-retrieval table
     reports a weighted edge between two candidate blocks, that
     reinforces the chronological direction.

3. Topo-sort with Kahn's algorithm. Cycles (rare; only happen when
   two blocks share a date and a co-retrieval edge) are broken on
   the lowest-weight edge so the output is always a clean sequence.

4. Tag each step with a *role*:

   * ``foundation`` — first ~30% of the walkthrough
   * ``context``    — middle ~40%
   * ``current``    — last ~30%

5. If the topic produces no candidates, return an empty list. If no
   graph structure emerges (single isolated block), fall back to
   relevance order with all steps tagged ``context``.

Returns a list of ``{step, block_id, role, score, subject}`` dicts.
The walkthrough is read-only — it never mutates the workspace, the
co-retrieval graph, or any block.
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
from typing import Any

__all__ = [
    "compile_walkthrough",
    "Step",
]

_log = logging.getLogger("mind_mem.walkthrough")

# block id format: PREFIX-YYYYMMDD-NNN (regex pulled from block_store)
_DATE_RE = re.compile(r"^[A-Z]+-(\d{8})-")

# Type alias kept loose to avoid coupling to TypedDict; callers see plain dicts.
Step = dict[str, Any]

DEFAULT_LIMIT = 25
MAX_LIMIT = 100


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _block_id(block: dict[str, Any]) -> str | None:
    bid = block.get("_id") or block.get("id") or block.get("block_id")
    return str(bid) if bid else None


def _block_subject(block: dict[str, Any]) -> str:
    for k in ("Subject", "subject", "Statement", "statement", "content"):
        v = block.get(k)
        if isinstance(v, str) and v.strip():
            line = v.split("\n", 1)[0].strip()
            return line[:120]
    return ""


def _date_key(block_id: str) -> int:
    """Extract YYYYMMDD as int for ordering. Returns 0 when absent."""
    m = _DATE_RE.match(block_id)
    if not m:
        return 0
    try:
        return int(m.group(1))
    except ValueError:
        return 0


def _co_retrieval_edges(workspace: str, ids: set[str]) -> dict[tuple[str, str], float]:
    """Return ``{(a, b): weight}`` for every co-retrieval edge inside *ids*.

    Edges are normalised so ``a < b`` (lexicographic); the dict has at
    most ``len(ids) * (len(ids) - 1) / 2`` entries.
    """
    if not ids:
        return {}
    db_path = os.path.join(workspace, "intelligence", "state", "retrieval_graph.db")
    if not os.path.isfile(db_path):
        return {}
    edges: dict[tuple[str, str], float] = {}
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            for row in conn.execute("SELECT mem1_id, mem2_id, weight FROM co_retrieval WHERE weight > 0"):
                m1, m2 = row["mem1_id"], row["mem2_id"]
                if m1 not in ids or m2 not in ids:
                    continue
                a, b = (m1, m2) if m1 < m2 else (m2, m1)
                w = float(row["weight"])
                if (a, b) not in edges or edges[(a, b)] < w:
                    edges[(a, b)] = w
        finally:
            conn.close()
    except sqlite3.Error as exc:
        _log.debug("co_retrieval_query_failed", extra={"error": str(exc)})
        return {}
    return edges


def _topo_sort(nodes: list[str], edges: list[tuple[str, str]]) -> list[str]:
    """Kahn's algorithm. *edges* are directed (src -> dst).

    On a cycle the lowest-degree node is force-emitted, breaking the
    cycle deterministically. Result is stable: same input → same output.
    """
    if not nodes:
        return []
    incoming: dict[str, set[str]] = {n: set() for n in nodes}
    outgoing: dict[str, set[str]] = {n: set() for n in nodes}
    for src, dst in edges:
        if src not in incoming or dst not in incoming:
            continue
        if src == dst:
            continue
        incoming[dst].add(src)
        outgoing[src].add(dst)

    out: list[str] = []
    # Sort the seed set by node id for determinism.
    ready = sorted([n for n in nodes if not incoming[n]])
    remaining = set(nodes)
    while remaining:
        if ready:
            n = ready.pop(0)
            remaining.discard(n)
            out.append(n)
            for m in sorted(outgoing[n]):
                incoming[m].discard(n)
                if not incoming[m] and m in remaining:
                    ready.append(m)
            ready.sort()
            continue
        # Cycle: take the remaining node with fewest incoming edges, break ties by id.
        n = sorted(remaining, key=lambda x: (len(incoming[x]), x))[0]
        remaining.discard(n)
        out.append(n)
        for m in sorted(outgoing[n]):
            incoming[m].discard(n)
            if not incoming[m] and m in remaining and m not in ready:
                ready.append(m)
        ready.sort()
    return out


def _role_for(step_idx: int, total: int) -> str:
    if total <= 1:
        return "context"
    pos = step_idx / max(total - 1, 1)
    if pos <= 0.30:
        return "foundation"
    if pos >= 0.70:
        return "current"
    return "context"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compile_walkthrough(
    workspace: str,
    topic: str,
    *,
    limit: int = DEFAULT_LIMIT,
    active_only: bool = False,
    agent_id: str | None = None,
) -> list[Step]:
    """Return blocks for *topic* in dependency order (foundations → current).

    Args:
        workspace: Workspace root.
        topic: Search query.
        limit: Maximum candidates to consider (>=1, <=100).
        active_only: Only return blocks with active status.
        agent_id: Optional agent id for namespace ACL filtering.

    Returns:
        Ordered list of ``{step, block_id, role, score, subject}`` dicts,
        with ``step`` 1-based. Empty when *topic* yields no candidates.

    Raises:
        ValueError: *workspace* / *topic* empty or *limit* out of range.
    """
    if not workspace:
        raise ValueError("workspace must be a non-empty path")
    if not isinstance(topic, str) or not topic.strip():
        raise ValueError("topic must be a non-empty string")
    if not 1 <= limit <= MAX_LIMIT:
        raise ValueError(f"limit must be in [1, {MAX_LIMIT}]")

    from ._recall_core import recall as _recall

    candidates = _recall(
        workspace=workspace,
        query=topic,
        limit=limit,
        active_only=active_only,
        agent_id=agent_id,
    )
    if not candidates:
        return []

    # Index candidates by id for fast lookup, drop those without an id.
    by_id: dict[str, dict[str, Any]] = {}
    order_in: list[str] = []
    for c in candidates:
        bid = _block_id(c)
        if bid is None:
            continue
        by_id[bid] = c
        order_in.append(bid)

    if not by_id:
        return []
    if len(by_id) == 1:
        # Trivial graph: one block, no edges. Fall back to relevance order.
        only_id = order_in[0]
        block = by_id[only_id]
        return [
            {
                "step": 1,
                "block_id": only_id,
                "role": "context",
                "score": block.get("score") or block.get("_score"),
                "subject": _block_subject(block),
            }
        ]

    # ---- Build dependency edges --------------------------------------------
    ids = set(by_id.keys())
    co_edges = _co_retrieval_edges(workspace, ids)

    # Chronological backbone: earlier date → later date is a default
    # dependency. Two blocks with the same date use lex order on their
    # full id to keep the graph deterministic.
    directed: list[tuple[str, str]] = []
    sorted_ids = sorted(ids, key=lambda b: (_date_key(b), b))
    for i in range(len(sorted_ids) - 1):
        a, b = sorted_ids[i], sorted_ids[i + 1]
        directed.append((a, b))

    # Co-retrieval edges reinforce the chronological direction. When
    # both endpoints share a date, the lex-lower id is the foundation.
    for (a, b), _w in co_edges.items():
        # Normalise so chronologically-earlier id is the source.
        d_a, d_b = _date_key(a), _date_key(b)
        if (d_a, a) < (d_b, b):
            directed.append((a, b))
        else:
            directed.append((b, a))

    ordered = _topo_sort(sorted_ids, directed)

    total = len(ordered)
    out: list[Step] = []
    for i, bid in enumerate(ordered):
        block = by_id[bid]
        out.append(
            {
                "step": i + 1,
                "block_id": bid,
                "role": _role_for(i, total),
                "score": block.get("score") or block.get("_score"),
                "subject": _block_subject(block),
            }
        )
    return out
