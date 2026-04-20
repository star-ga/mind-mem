"""Context-core MCP tools — ``.mmcore`` bundle lifecycle.

Extracted from ``mcp_server.py`` per docs/v3.2.0-mcp-decomposition-plan.md
(PR-3 slice, core domain). Four tools wrap the
:class:`CoreRegistry` singleton for building, loading, unloading,
and listing portable block + knowledge-graph archives.
"""

from __future__ import annotations

import json
import os
from typing import Any

from ..infra.observability import mcp_tool_observe
from ..infra.workspace import _check_workspace, _workspace
from ._helpers import _core_dir, _core_registry, _kg_path


@mcp_tool_observe
def build_core(namespace: str, version: str, filename: str = "") -> str:
    """Build a .mmcore bundle from the active workspace's blocks.

    Snapshots the current block index + knowledge graph into a portable
    `.mmcore` archive. Downstream callers can load it into another
    mind-mem instance via `load_core`.

    Args:
        namespace: Identifier used to prefix blocks when loaded.
        version: Caller-facing semver recorded in the manifest.
        filename: Optional output filename (defaults to
            ``<namespace>-<version>.mmcore`` under ``memory/cores/``).

    Returns:
        JSON envelope with the bundle path and manifest summary.
    """
    from mind_mem.context_core import build_core as _build_core
    from mind_mem.knowledge_graph import KnowledgeGraph

    ws = _workspace()
    ws_err = _check_workspace(ws)
    if ws_err:
        return ws_err

    if not isinstance(namespace, str) or not namespace.strip():
        return json.dumps({"error": "namespace must be a non-empty string"})
    if not isinstance(version, str) or not version.strip():
        return json.dumps({"error": "version must be a non-empty string"})

    blocks: list[dict] = []
    try:
        from mind_mem.sqlite_index import merkle_leaves as _leaves

        for bid, content_hash in _leaves(ws):
            blocks.append({"_id": bid, "content_hash": content_hash})
    except (ImportError, AttributeError):
        pass

    edges: list[dict] = []
    kg_file = _kg_path(ws)
    if os.path.isfile(kg_file):
        kg = KnowledgeGraph(kg_file)
        try:
            fallback_edges: list[Any] = []
            for e in kg.edges_from("__all__") if False else fallback_edges:
                edges.append(e.as_dict())
            rows = kg._conn.execute(
                "SELECT subject, predicate, object, source_block_id, confidence, valid_from, valid_until, metadata FROM edges"
            ).fetchall()
            for row in rows:
                edges.append(
                    {
                        "subject": row["subject"],
                        "predicate": row["predicate"],
                        "object": row["object"],
                        "source_block_id": row["source_block_id"],
                        "confidence": row["confidence"],
                        "valid_from": row["valid_from"],
                        "valid_until": row["valid_until"],
                        "metadata": row["metadata"],
                    }
                )
        finally:
            kg.close()

    out_name = filename.strip() or f"{namespace.strip()}-{version.strip()}.mmcore"
    if any(ch in out_name for ch in "/\\"):
        return json.dumps({"error": "filename must not contain path separators"})
    out_path = os.path.join(_core_dir(ws), out_name)

    try:
        manifest = _build_core(
            out_path,
            namespace=namespace,
            version=version,
            blocks=blocks,
            edges=edges,
        )
    except ValueError as exc:
        return json.dumps({"error": str(exc)})

    return json.dumps(
        {"path": out_path, "manifest": manifest.as_dict(), "_schema_version": "1.0"},
        indent=2,
    )


@mcp_tool_observe
def load_core(filename: str, verify: bool = True) -> str:
    """Load a .mmcore bundle from the workspace's cores/ directory.

    Args:
        filename: Core filename relative to ``memory/cores/``.
        verify: Recompute and compare the content hash (default True).
    """
    from mind_mem.context_core import CoreLoadError

    ws = _workspace()
    ws_err = _check_workspace(ws)
    if ws_err:
        return ws_err

    if not isinstance(filename, str) or not filename.strip():
        return json.dumps({"error": "filename must be a non-empty string"})
    if any(ch in filename for ch in "/\\"):
        return json.dumps({"error": "filename must not contain path separators"})

    path = os.path.join(_core_dir(ws), filename.strip())
    try:
        loaded = _core_registry().load(path, verify=verify)
    except CoreLoadError as exc:
        return json.dumps({"error": str(exc)})
    except RuntimeError as exc:
        return json.dumps({"error": str(exc)})
    return json.dumps(
        {
            "loaded": True,
            "namespace": loaded.manifest.namespace,
            "blocks": loaded.block_count(),
            "edges": loaded.edge_count(),
            "content_hash": loaded.manifest.content_hash,
            "_schema_version": "1.0",
        },
        indent=2,
    )


@mcp_tool_observe
def unload_core(namespace: str) -> str:
    """Unload a previously-loaded core by namespace."""
    ws = _workspace()
    ws_err = _check_workspace(ws)
    if ws_err:
        return ws_err
    if not isinstance(namespace, str) or not namespace.strip():
        return json.dumps({"error": "namespace must be a non-empty string"})
    ok = _core_registry().unload(namespace.strip())
    return json.dumps({"unloaded": bool(ok)})


@mcp_tool_observe
def list_cores() -> str:
    """List every currently-loaded .mmcore bundle (namespace + stats)."""
    ws = _workspace()
    ws_err = _check_workspace(ws)
    if ws_err:
        return ws_err
    return json.dumps({"cores": _core_registry().stats(), "_schema_version": "1.0"}, indent=2)


def register(mcp) -> None:
    """Wire the core tools onto *mcp*."""
    mcp.tool(build_core)
    mcp.tool(load_core)
    mcp.tool(unload_core)
    mcp.tool(list_cores)
