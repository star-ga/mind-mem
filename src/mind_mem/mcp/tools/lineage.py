"""MCP wrapping for the v3.11.0 typed block-lineage graph (Pattern 3).

Exposes the :mod:`mind_mem.block_lineage` write+read primitives as MCP
tools so AI clients can record explicit semantic lineage between
blocks (cites / implements / refines / contradicts) and traverse the
result with a bounded BFS.

Two tools:

* :func:`block_lineage` — read-only BFS rooted at a block, capped at
  three hops and 1000 nodes by contract.
* :func:`add_block_edge` — write a single typed edge.
"""

from __future__ import annotations

import json
from typing import Any

from ..infra.observability import mcp_tool_observe
from ..infra.workspace import _check_workspace, _workspace
from ._helpers import get_logger

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
            / ``cooccurrence``); ``None`` returns every kind.
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

    try:
        result = _read(
            ws,
            str(block_id),
            max_depth=int(max_depth),
            kind_filter=kind_filter,
            node_cap=int(node_cap),
        )
    except ValueError as exc:
        return json.dumps({"error": str(exc)})

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
