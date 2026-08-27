"""Calibration MCP tools — feedback (quality) + outcome attribution (utility).

Extracted from ``mcp_server.py`` per docs/v3.2.0-mcp-decomposition-plan.md
(PR-3 slice, calibration domain). Both tools delegate to
``mind_mem.calibration.CalibrationManager``; the retrieval scoring
loop records per-block feedback and reports accuracy by query type.

``report_outcome`` / ``outcome_stats`` close the other half of the loop:
not "was this block a good match?" but "did acting on it actually work?".
Both write to the same calibration store — see
``mind_mem.outcome_attribution``.
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


@mcp_tool_observe
def report_outcome(
    block_ids: list[str] | None = None,
    outcome: str = "",
    query_id: str = "",
    task_id: str = "",
    actor_id: str = "",
    session_id: str = "",
    tool_id: str = "",
    evidence: str = "",
    project_to_calibration: bool = False,
) -> str:
    """Report whether recalled blocks actually *helped* — utility, not relevance.

    Call this once the real-world verdict is known: the tests went green
    after a fix guided by these blocks (``success``), the change they
    justified broke the build (``failure``), or the outcome is not
    attributable to them (``neutral``).

    Repeatedly implicated blocks are deterministically demoted by the
    recall validity gate; blocks in successful outcomes are treated as
    corroborated. Recording an outcome never edits block content — it
    appends to the calibration store only. Corpus changes still go
    through ``propose_update``.

    Replaying the same report is a no-op: the outcome id is the SHA-256
    of its canonical payload.

    Args:
        block_ids: Blocks that were recalled and acted upon.
        outcome: "success", "failure", or "neutral".
        query_id: The query_id from the recall result that produced them.
        task_id: What was being done (build id, ticket, test name).
        actor_id: Who/what is reporting.
        session_id: Session provenance.
        tool_id: Reporting tool provenance.
        evidence: Free-text proof, e.g. a test summary line.
        project_to_calibration: Also feed these verdicts into the
            per-block calibration weight loop (default off — that loop
            moves recall scores for everyone, opted in or not).

    Returns:
        JSON with the outcome id, payload hash, and recorded/duplicate counts.
    """
    ws = _workspace()

    try:
        from mind_mem.outcome_attribution import report_outcome as _report

        result = _report(
            ws,
            block_ids or [],
            outcome,
            query_id=query_id,
            task_id=task_id,
            actor_id=actor_id,
            session_id=session_id,
            tool_id=tool_id,
            evidence=evidence,
            project_to_calibration=project_to_calibration,
        )
    except ValueError as exc:
        return json.dumps(
            {
                "_schema_version": MCP_SCHEMA_VERSION,
                "error": str(exc),
            }
        )
    except ImportError:
        return json.dumps(
            {
                "_schema_version": MCP_SCHEMA_VERSION,
                "error": "Outcome attribution module not available.",
            }
        )
    except Exception as exc:
        _log.warning("report_outcome_failed", error=str(exc))
        return json.dumps(
            {
                "_schema_version": MCP_SCHEMA_VERSION,
                "error": f"Failed to record outcome: {exc}",
            }
        )

    metrics.inc("mcp_report_outcome")
    return json.dumps(
        {
            "_schema_version": MCP_SCHEMA_VERSION,
            "status": "recorded",
            **result,
        },
        indent=2,
    )


@mcp_tool_observe
def outcome_stats(block_id: str = "", task_id: str = "", top_n: int = 20) -> str:
    """Query recorded outcomes — which memories actually earned their keep.

    With no arguments this returns the utility health report: totals, the
    blocks corroborated by successful outcomes, and the blocks most
    implicated in failures (with their deterministic utility factor).
    Passing ``block_id`` or ``task_id`` instead lists the individual
    outcome records, with full provenance, for that block or task.

    Args:
        block_id: List outcomes attributed to this block.
        task_id: List outcomes recorded for this task.
        top_n: Max entries per section of the health report.

    Returns:
        JSON health report, or JSON outcome listing when filtered.
    """
    ws = _workspace()

    try:
        from mind_mem.calibration import CalibrationManager

        cal = CalibrationManager(ws)
        if block_id or task_id:
            outcomes = cal.list_outcomes(block_id=block_id, task_id=task_id, limit=top_n)
            payload = {
                "block_id": block_id,
                "task_id": task_id,
                "count": len(outcomes),
                "outcomes": outcomes,
            }
        else:
            payload = cal.get_outcome_stats(top_n=top_n)
    except ValueError as exc:
        return json.dumps(
            {
                "_schema_version": MCP_SCHEMA_VERSION,
                "error": str(exc),
            }
        )
    except ImportError:
        return json.dumps(
            {
                "_schema_version": MCP_SCHEMA_VERSION,
                "error": "Calibration module not available.",
            }
        )
    except Exception as exc:
        _log.warning("outcome_stats_failed", error=str(exc))
        return json.dumps(
            {
                "_schema_version": MCP_SCHEMA_VERSION,
                "error": f"Failed to retrieve outcome stats: {exc}",
            }
        )

    metrics.inc("mcp_outcome_stats")
    return json.dumps(
        {
            "_schema_version": MCP_SCHEMA_VERSION,
            **payload,
        },
        indent=2,
    )


def register(mcp) -> None:
    """Wire the calibration tools onto *mcp*."""
    mcp.tool(calibration_feedback)
    mcp.tool(calibration_stats)
    mcp.tool(report_outcome)
    mcp.tool(outcome_stats)
