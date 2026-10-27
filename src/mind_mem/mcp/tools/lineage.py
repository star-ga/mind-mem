"""MCP wrapping for the v3.11.0 typed block-lineage graph (Pattern 3).

Exposes the :mod:`mind_mem.block_lineage` write+read primitives as MCP
tools so AI clients can record explicit semantic lineage between
blocks (cites / implements / refines / contradicts / supersedes) and
traverse the result with a bounded BFS.

Two tools:

* :func:`block_lineage` — read-only BFS rooted at a block, capped at
  three hops and 1000 nodes by contract.
* :func:`add_block_edge` — write a single typed edge.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any

from ..infra.observability import mcp_tool_observe
from ..infra.workspace import _check_workspace, _workspace
from ._helpers import _read_scope_or_denied, get_logger

_log = get_logger("mcp_server")

__all__ = ["add_block_edge", "block_lineage", "register"]


@mcp_tool_observe
def block_lineage(
    block_id: str,
    max_depth: int = 3,
    kind_filter: str | None = None,
    node_cap: int = 1000,
) -> str:
    """BFS-traverse the block-lineage graph rooted at ``block_id``.

    Args:
        block_id: Root block id.
        max_depth: Maximum hop distance, clamped to ``[1, 3]``.
        kind_filter: Optional restriction to a single edge kind
            (``cites`` / ``implements`` / ``refines`` / ``contradicts``
            / ``cooccurrence`` / ``supersedes``); ``None`` returns every kind.
        node_cap: Total-node cap; reaching this sets ``truncated``
            in the response.

    Returns:
        JSON string of the
        :class:`mind_mem.block_lineage.LineageResult` payload.
    """

    ws = _workspace()
    ws_err = _check_workspace(ws)
    if ws_err:
        return ws_err

    from mind_mem.block_lineage import block_lineage as _read

    scope, denied = _read_scope_or_denied(ws)
    if denied is not None:
        return denied
    # A restricted caller gets the lineage of the VISIBLE subgraph: the walk
    # never enters or steps through a block the caller may not read, so a
    # visible block reachable only via a hidden one is absent and every
    # distance is one the caller could derive from blocks they may read. A root
    # the caller cannot read answers exactly like a root with no edges.
    walkable = (not scope.restricted) or scope.allows(block_id)
    # An unreadable root walks a surrogate id that carries no edges, so the
    # work (and latency) of a denied root is that of an edgeless/missing one.
    walk_root = str(block_id) if walkable else "\x01-unreadable-root"
    try:
        result = _read(
            ws,
            walk_root,
            max_depth=int(max_depth),
            kind_filter=kind_filter,
            node_cap=int(node_cap),
            visible=scope.allows if scope.restricted else None,
        )
    except ValueError as exc:
        return json.dumps({"error": str(exc)})
    if not walkable:
        result = dataclasses.replace(result, root=str(block_id), edges=[], truncated=False)

    return json.dumps(result.to_dict(), indent=2)


@mcp_tool_observe
def add_block_edge(
    src: str,
    dst: str,
    kind: str,
    weight: float = 1.0,
) -> str:
    """Record an explicit typed lineage edge from ``src`` to ``dst``.

    Idempotent — re-adding the same ``(src, dst, kind)`` triple bumps
    the hit-count instead of inserting a duplicate row.
    """

    ws = _workspace()
    ws_err = _check_workspace(ws)
    if ws_err:
        return ws_err

    from mind_mem.block_lineage import add_block_edge as _write

    try:
        _write(ws, str(src), str(dst), str(kind), weight=float(weight))
    except ValueError as exc:
        return json.dumps({"error": str(exc)})

    return json.dumps({"ok": True, "src": src, "dst": dst, "kind": kind})


def register(mcp: Any) -> None:
    """Wire lineage tools onto *mcp*."""

    mcp.tool(block_lineage)
    mcp.tool(add_block_edge)
