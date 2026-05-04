"""MCP wrapping for pipeline-hash inspection + dirty-block re-extraction.

v3.9.0 follow-up to the v3.9 ``pipeline_hash.py`` module. v3.9
shipped the inspection primitive (which blocks were extracted by an
older pipeline) and the ``mm pipeline-status`` CLI; this surfaces
both inspection and the new write-side helper through MCP so AI
clients can trigger targeted re-stamping without dropping to the
shell.

Two new tools:

* :func:`pipeline_status` — read-only summary: current hash, dirty-
  block count, inputs that produced the hash. Equivalent of the CLI
  but JSON-shaped for tool-call use.

* :func:`reindex_dirty` — write-side trigger that re-stamps blocks
  whose ``TransformHash`` is stale. Supports ``dry_run`` and
  ``limit`` for safe operator workflows.
"""

from __future__ import annotations

import json
from typing import Any

from ..infra.observability import mcp_tool_observe
from ..infra.workspace import _check_workspace, _workspace
from ._helpers import get_logger

_log = get_logger("mcp_server")


__all__ = [
    "pipeline_status",
    "reindex_dirty",
    "register",
]


@mcp_tool_observe
def pipeline_status() -> str:
    """Return current pipeline hash + dirty-block summary.

    The pipeline hash is a deterministic SHA-256 over the package
    version, extractor backend, model id, on-disk extractor source,
    and prompt template. Blocks whose stored ``TransformHash`` is
    different from the current hash are flagged as ``dirty`` and
    can be refreshed via :func:`reindex_dirty`.

    Read-only; no writes occur.
    """
    ws = _workspace()
    ws_err = _check_workspace(ws)
    if ws_err:
        return ws_err

    from mind_mem.pipeline_hash import current_pipeline_hash, pipeline_dirty_blocks

    digest, inputs = current_pipeline_hash(ws, return_inputs=True)
    dirty = pipeline_dirty_blocks(ws)
    return json.dumps(
        {
            "current_hash": digest,
            "inputs": inputs.as_dict(),
            "dirty_count": len(dirty),
            "dirty_ids": dirty[:25],  # head only — full list via reindex_dirty(dry_run=true)
            "truncated": len(dirty) > 25,
        },
        indent=2,
        default=str,
    )


@mcp_tool_observe
def reindex_dirty(
    limit: int = 0,
    dry_run: bool = False,
) -> str:
    """Re-stamp blocks whose pipeline hash is stale.

    v3.9.0 replaces v3.9's read-only inspection with a targeted
    write: only blocks flagged as dirty by :func:`pipeline_status`
    are touched, and only their ``TransformHash`` field is
    refreshed (block content is *not* re-extracted — that requires
    invoking the LLM extractor and is left to a future revision).

    Args:
        limit: Maximum number of blocks to touch. ``0`` (default)
            means "all dirty blocks". Caps allow safe staged
            rollouts.
        dry_run: When True, report which blocks *would* be touched
            without writing.

    Returns:
        ``{processed, skipped, dry_run, ids}`` summary.
    """
    if limit < 0:
        return json.dumps({"error": "limit must be >= 0"})

    ws = _workspace()
    ws_err = _check_workspace(ws)
    if ws_err:
        return ws_err

    from mind_mem.pipeline_hash import reextract_dirty_blocks

    try:
        result = reextract_dirty_blocks(
            ws,
            limit=limit if limit > 0 else None,
            dry_run=bool(dry_run),
        )
    except ValueError as exc:
        return json.dumps({"error": str(exc)})

    return json.dumps(result, indent=2, default=str)


def register(mcp: Any) -> None:
    """Wire pipeline-hash tools onto *mcp*."""
    mcp.tool(pipeline_status)
    mcp.tool(reindex_dirty)
