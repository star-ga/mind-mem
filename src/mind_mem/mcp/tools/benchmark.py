"""Benchmark + category-summary MCP tools.

Extracted from ``mcp_server.py`` per docs/v3.2.0-mcp-decomposition-plan.md
(PR-3 slice, benchmark domain). Two tools live here:

* ``governance_health_bench`` — exercises contradiction detection,
  audit completeness, drift, and scalability probes.
* ``category_summary`` — category distiller lookup driven by the
  configurable ``max_category_results`` limit and the
  ``categories.extra_categories`` config block.
"""

from __future__ import annotations

import json

from ..infra.config import _get_limits, _load_extra_categories
from ..infra.constants import MCP_SCHEMA_VERSION
from ..infra.observability import mcp_tool_observe
from ..infra.workspace import _workspace
from ._helpers import get_logger, metrics

_log = get_logger("mcp_server")


@mcp_tool_observe
def governance_health_bench() -> str:
    """Run the governance health benchmark suite.

    Exercises contradiction detection, audit completeness, drift
    detection, and scalability probes against the current workspace.

    Returns:
        JSON report covering all bench sub-suites and aggregated
        pass/fail counts.
    """
    ws = _workspace()
    try:
        from mind_mem.governance_bench import GovernanceBench

        report = GovernanceBench(ws).run_all()
    except Exception as exc:
        return json.dumps(
            {
                "_schema_version": MCP_SCHEMA_VERSION,
                "error": f"bench run failed: {exc}",
            }
        )
    out = {"_schema_version": MCP_SCHEMA_VERSION, **report}
    return json.dumps(out, indent=2, default=str)


@mcp_tool_observe
def category_summary(topic: str, limit: int = 3) -> str:
    """Returns category summaries relevant to a given topic.

    Uses the category distiller to find and return thematic summary files
    matching the topic. Categories are auto-generated from memory blocks.

    Args:
        topic: Topic or query to find relevant categories for.
        limit: Maximum number of category summaries to return (default: 3).

    Returns:
        Concatenated category summaries with block references.
    """
    ws = _workspace()
    limits = _get_limits(ws)
    try:
        from mind_mem.category_distiller import CategoryDistiller

        extra_cats = _load_extra_categories(ws)
        distiller = CategoryDistiller(extra_categories=extra_cats if extra_cats else None)
        context = distiller.get_category_context(topic, ws, limit=max(1, min(limit, limits["max_category_results"])))
        cats = distiller.get_categories_for_query(topic)
        metrics.inc("mcp_category_summary")
        _log.info("mcp_category_summary", topic=topic, matched_categories=cats[:limit])
        if not context:
            return json.dumps(
                {
                    "_schema_version": MCP_SCHEMA_VERSION,
                    "topic": topic,
                    "status": "no_categories",
                    "hint": "Run reindex to generate category files, or add blocks with matching tags.",
                },
                indent=2,
            )
        return json.dumps(
            {
                "_schema_version": MCP_SCHEMA_VERSION,
                "topic": topic,
                "matched_categories": cats[:limit],
                "content": context,
            },
            indent=2,
        )
    except ImportError:
        return json.dumps(
            {
                "_schema_version": MCP_SCHEMA_VERSION,
                "error": "category_distiller module not available",
            }
        )
    except (OSError, ValueError, KeyError) as exc:
        _log.warning("category_summary_failed", topic=topic, error=str(exc))
        return json.dumps(
            {
                "_schema_version": MCP_SCHEMA_VERSION,
                "error": "Category summary lookup failed",
                "topic": topic,
            },
            indent=2,
        )


def register(mcp) -> None:
    """Wire the benchmark tools onto *mcp*."""
    mcp.tool(governance_health_bench)
    mcp.tool(category_summary)
