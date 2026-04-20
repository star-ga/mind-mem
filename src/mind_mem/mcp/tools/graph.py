"""Knowledge-graph + causal-graph MCP tools.

Extracted from ``mcp_server.py`` per docs/v3.2.0-mcp-decomposition-plan.md
(PR-3 slice, graph domain). Two surfaces land here:

* ``graph_add_edge`` / ``graph_query`` / ``graph_stats`` — typed
  knowledge-graph edges + N-hop traversal + aggregate stats.
* ``traverse_graph`` — causal-dependency graph navigation for
  impact analysis on block IDs.
"""

from __future__ import annotations

import json
import os
import re as _re_mod
import sqlite3
from typing import Any

from mind_mem.observability import get_logger, metrics

from ..infra.constants import MCP_SCHEMA_VERSION
from ..infra.observability import _is_db_locked, _sqlite_busy_error, mcp_tool_observe
from ..infra.workspace import _check_workspace, _workspace
from ._helpers import _kg_path

_log = get_logger("mcp_server")


@mcp_tool_observe
def graph_add_edge(
    subject: str,
    predicate: str,
    object: str,
    source_block_id: str,
    confidence: float = 1.0,
) -> str:
    """Record a typed relationship in the knowledge graph."""
    from mind_mem.knowledge_graph import KnowledgeGraph, Predicate

    ws = _workspace()
    ws_err = _check_workspace(ws)
    if ws_err:
        return ws_err

    if not isinstance(subject, str) or not subject.strip():
        return json.dumps({"error": "subject must be a non-empty string"})
    if not isinstance(object, str) or not object.strip():
        return json.dumps({"error": "object must be a non-empty string"})
    if len(subject) > 512 or len(object) > 512:
        return json.dumps({"error": "entity names must be ≤512 chars"})
    try:
        pred = Predicate.from_str(predicate)
    except ValueError as exc:
        return json.dumps({"error": str(exc)})

    kg = KnowledgeGraph(_kg_path(ws))
    try:
        edge = kg.add_edge(
            subject,
            pred,
            object,
            source_block_id=source_block_id,
            confidence=float(confidence),
        )
    except ValueError as exc:
        return json.dumps({"error": str(exc)})
    finally:
        kg.close()
    return json.dumps({**edge.as_dict(), "_schema_version": "1.0"}, indent=2)


@mcp_tool_observe
def graph_query(
    entity: str,
    depth: int = 1,
    predicate: str = "",
    direction: str = "outgoing",
    limit: int = 64,
) -> str:
    """N-hop traversal from *entity*."""
    from mind_mem.knowledge_graph import KnowledgeGraph, Predicate

    ws = _workspace()
    ws_err = _check_workspace(ws)
    if ws_err:
        return ws_err

    if not isinstance(entity, str) or not entity.strip():
        return json.dumps({"error": "entity must be a non-empty string"})
    if len(entity) > 512:
        return json.dumps({"error": "entity name must be ≤512 chars"})
    if not (1 <= depth <= 8):
        return json.dumps({"error": "depth must be in [1, 8]"})
    if not (1 <= limit <= 256):
        return json.dumps({"error": "limit must be in [1, 256]"})
    pred_obj = None
    if predicate.strip():
        try:
            pred_obj = Predicate.from_str(predicate)
        except ValueError as exc:
            return json.dumps({"error": str(exc)})
    if direction not in {"outgoing", "incoming", "both"}:
        return json.dumps({"error": "direction must be 'outgoing' / 'incoming' / 'both'"})

    kg = KnowledgeGraph(_kg_path(ws))
    try:
        neighbours = kg.neighbors(
            entity,
            depth=depth,
            predicate=pred_obj,
            direction=direction,
            max_results=limit,
        )
    except ValueError as exc:
        return json.dumps({"error": str(exc)})
    finally:
        kg.close()
    return json.dumps(
        {
            "entity": entity,
            "depth": depth,
            "direction": direction,
            "neighbors": neighbours,
            "_schema_version": "1.0",
        },
        indent=2,
    )


@mcp_tool_observe
def graph_stats() -> str:
    """Aggregated knowledge-graph stats for the active workspace."""
    from mind_mem.knowledge_graph import KnowledgeGraph

    ws = _workspace()
    ws_err = _check_workspace(ws)
    if ws_err:
        return ws_err

    kg_file = _kg_path(ws)
    if not os.path.isfile(kg_file):
        return json.dumps(
            {"entities": 0, "edges": 0, "predicates": {}, "orphan_entities": 0, "_schema_version": "1.0"},
            indent=2,
        )
    kg = KnowledgeGraph(kg_file)
    try:
        stats = kg.stats().as_dict()
    finally:
        kg.close()
    return json.dumps({**stats, "_schema_version": "1.0"}, indent=2)


@mcp_tool_observe
def traverse_graph(block_id: str, depth: int = 2, direction: str = "both") -> str:
    """Navigate the causal dependency graph from a block."""
    if not _re_mod.match(r"^[A-Z]+-[a-zA-Z0-9_.-]+$", block_id):
        return json.dumps(
            {
                "_schema_version": MCP_SCHEMA_VERSION,
                "error": f"Invalid block_id format: {block_id}",
            }
        )

    depth = max(1, min(depth, 5))

    if direction not in ("upstream", "downstream", "both"):
        return json.dumps(
            {
                "_schema_version": MCP_SCHEMA_VERSION,
                "error": f"Invalid direction: {direction}. Use 'upstream', 'downstream', or 'both'.",
            }
        )

    ws = _workspace()
    try:
        from mind_mem.causal_graph import CausalGraph

        cg = CausalGraph(ws)

        result: dict[str, Any] = {
            "_schema_version": MCP_SCHEMA_VERSION,
            "block_id": block_id,
            "direction": direction,
            "depth": depth,
        }

        if direction in ("upstream", "both"):
            chains = cg.causal_chain(block_id, max_depth=depth)
            deps = cg.dependencies(block_id)
            result["upstream"] = {
                "direct_dependencies": [
                    {
                        "target": e.target_id,
                        "edge_type": e.edge_type,
                        "weight": e.weight,
                    }
                    for e in deps
                ],
                "causal_chains": chains,
            }

        if direction in ("downstream", "both"):
            dependents = cg.dependents(block_id)
            downstream_nodes: list[dict] = []
            visited: set[str] = {block_id}
            current_layer = [block_id]
            for d in range(depth):
                next_layer: list[str] = []
                for node_id in current_layer:
                    node_deps = cg.dependents(node_id)
                    for e in node_deps:
                        if e.source_id not in visited:
                            visited.add(e.source_id)
                            next_layer.append(e.source_id)
                            downstream_nodes.append(
                                {
                                    "block_id": e.source_id,
                                    "depends_on": node_id,
                                    "edge_type": e.edge_type,
                                    "depth": d + 1,
                                }
                            )
                current_layer = next_layer
                if not current_layer:
                    break

            result["downstream"] = {
                "direct_dependents": [
                    {
                        "source": e.source_id,
                        "edge_type": e.edge_type,
                        "weight": e.weight,
                    }
                    for e in dependents
                ],
                "reachable_nodes": downstream_nodes,
            }

        summary = cg.summary()
        result["graph_summary"] = summary

        metrics.inc("mcp_traverse_graph")
        _log.info("mcp_traverse_graph", block_id=block_id, direction=direction, depth=depth)
        return json.dumps(result, indent=2, default=str)

    except ImportError:
        return json.dumps(
            {
                "_schema_version": MCP_SCHEMA_VERSION,
                "error": "causal_graph module not available",
                "block_id": block_id,
            },
            indent=2,
        )
    except sqlite3.OperationalError as exc:
        if _is_db_locked(exc):
            return _sqlite_busy_error()
        raise
    except (OSError, ValueError) as exc:
        _log.warning("traverse_graph_failed", block_id=block_id, error=str(exc))
        return json.dumps(
            {
                "_schema_version": MCP_SCHEMA_VERSION,
                "error": f"Graph traversal failed: {exc}",
                "block_id": block_id,
            },
            indent=2,
        )


def register(mcp) -> None:
    """Wire the graph tools onto *mcp*."""
    mcp.tool(graph_add_edge)
    mcp.tool(graph_query)
    mcp.tool(graph_stats)
    mcp.tool(traverse_graph)
