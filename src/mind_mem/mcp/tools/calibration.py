"""Calibration feedback MCP tools — ``calibration_feedback`` + ``calibration_stats``.

Extracted from ``mcp_server.py`` per docs/v3.2.0-mcp-decomposition-plan.md
(PR-3 slice, calibration domain). Both tools delegate to
``mind_mem.calibration.CalibrationManager``; the retrieval scoring
loop records per-block feedback and reports accuracy by query type.
"""

from __future__ import annotations

import json

from ..infra.constants import MCP_SCHEMA_VERSION
from ..infra.observability import mcp_tool_observe
from ..infra.workspace import _workspace
from ._helpers import get_logger, metrics

_log = get_logger("mcp_server")


@mcp_tool_observe
def calibration_feedback(
    query_id: str,
    block_ids_useful: list[str] | None = None,
    block_ids_not_useful: list[str] | None = None,
    feedback_type: str = "accepted",
) -> str:
    """Record retrieval quality feedback for calibration.

    After a recall query, report which blocks were useful and which were not.
    This feeds a calibration loop that adjusts block ranking over time:
    consistently useful blocks get boosted, consistently unhelpful blocks
    get demoted.

    Args:
        query_id: The query_id from a previous recall result envelope.
        block_ids_useful: Block IDs that were useful/relevant.
        block_ids_not_useful: Block IDs that were not useful/irrelevant.
        feedback_type: Feedback kind — "accepted" (user used results),
            "rejected" (results were wrong), or "ignored" (user skipped).

    Returns:
        JSON confirmation with recorded feedback counts.
    """
    ws = _workspace()

    if feedback_type not in ("accepted", "rejected", "ignored"):
        return json.dumps(
            {
                "_schema_version": MCP_SCHEMA_VERSION,
                "error": f"Invalid feedback_type: {feedback_type}. Must be accepted/rejected/ignored.",
            }
        )

    useful = block_ids_useful or []
    not_useful = block_ids_not_useful or []

    if not useful and not not_useful:
        return json.dumps(
            {
                "_schema_version": MCP_SCHEMA_VERSION,
                "error": "At least one of block_ids_useful or block_ids_not_useful must be provided.",
            }
        )

    try:
        from mind_mem.calibration import CalibrationManager

        cal = CalibrationManager(ws)
        result = cal.record_feedback(
            query_id=query_id,
            block_ids_useful=useful,
            block_ids_not_useful=not_useful,
            feedback_type=feedback_type,
        )
    except ImportError:
        return json.dumps(
            {
                "_schema_version": MCP_SCHEMA_VERSION,
                "error": "Calibration module not available.",
            }
        )
    except Exception as exc:
        _log.warning("calibration_feedback_failed", error=str(exc))
        return json.dumps(
            {
                "_schema_version": MCP_SCHEMA_VERSION,
                "error": f"Failed to record feedback: {exc}",
            }
        )

    metrics.inc("mcp_calibration_feedback")
    return json.dumps(
        {
            "_schema_version": MCP_SCHEMA_VERSION,
            "status": "recorded",
            **result,
        },
        indent=2,
    )


@mcp_tool_observe
def calibration_stats() -> str:
    """Report calibration health — per-block scores, per-query-type accuracy.

    Shows which blocks are being boosted or demoted by the calibration
    feedback loop, and how accurate retrieval is for different query types
    (WHAT, WHEN, WHO, HOW, etc.).

    Returns:
        JSON report with calibration health metrics.
    """
    ws = _workspace()

    try:
        from mind_mem.calibration import CalibrationManager

        cal = CalibrationManager(ws)
        stats = cal.get_calibration_stats()
    except ImportError:
        return json.dumps(
            {
                "_schema_version": MCP_SCHEMA_VERSION,
                "error": "Calibration module not available.",
            }
        )
    except Exception as exc:
        _log.warning("calibration_stats_failed", error=str(exc))
        return json.dumps(
            {
                "_schema_version": MCP_SCHEMA_VERSION,
                "error": f"Failed to retrieve calibration stats: {exc}",
            }
        )

    metrics.inc("mcp_calibration_stats")
    return json.dumps(
        {
            "_schema_version": MCP_SCHEMA_VERSION,
            **stats,
        },
        indent=2,
    )


def register(mcp) -> None:
    """Wire the calibration tools onto *mcp*."""
    mcp.tool(calibration_feedback)
    mcp.tool(calibration_stats)
