"""Interaction-signal MCP tools — ``observe_signal`` + ``signal_stats``.

Extracted from ``mcp_server.py`` per docs/v3.2.0-mcp-decomposition-plan.md
(PR-3 slice, signal domain). ``observe_signal`` captures re-query /
refinement / correction feedback; ``signal_stats`` reports the
aggregated counters that operators watch for recall regressions.
"""

from __future__ import annotations

import json

from ..infra.observability import mcp_tool_observe
from ..infra.workspace import _check_workspace, _workspace
from ._helpers import _signal_store_path


@mcp_tool_observe
def observe_signal(
    session_id: str,
    previous_query: str,
    new_query: str,
    previous_results: str = "",
) -> str:
    """Capture a re-query / refinement / correction feedback signal.

    Called by the recall pipeline (or an external agent) when the user
    rephrases or retries a query within the same session. The classifier
    decides whether the pair is a genuine feedback event and, if so,
    appends a structured :class:`Signal` to the append-only store.

    Args:
        session_id: Conversation / agent session identifier.
        previous_query: The earlier query the user issued.
        new_query: The follow-up query.
        previous_results: Comma-separated block ids returned by the
            earlier recall. Used by the A/B eval harness to score future
            retrieval changes against what the user already saw.

    Returns:
        JSON with either a captured signal (``signal_id``, ``type``,
        ``similarity``) or ``{"captured": false}`` when the two queries
        are unrelated and no feedback was recorded.
    """
    from mind_mem.interaction_signals import SignalStore

    ws = _workspace()
    ws_err = _check_workspace(ws)
    if ws_err:
        return ws_err

    if not isinstance(session_id, str) or not session_id.strip():
        return json.dumps({"error": "session_id must be a non-empty string"})
    if not isinstance(previous_query, str) or not isinstance(new_query, str):
        return json.dumps({"error": "queries must be strings"})
    if len(previous_query) > 8192 or len(new_query) > 8192:
        return json.dumps({"error": "queries must be ≤8192 chars"})

    prev_ids: list[str] = []
    if previous_results.strip():
        prev_ids = [bid.strip() for bid in previous_results.split(",") if bid.strip()][:64]

    store = SignalStore(_signal_store_path(ws))
    result = store.observe_pair(
        session_id=session_id,
        previous_query=previous_query,
        new_query=new_query,
        previous_results=prev_ids,
    )
    if result is None:
        return json.dumps({"captured": False, "reason": "unrelated or duplicate"})
    return json.dumps(
        {
            "captured": True,
            "signal_id": result.signal_id,
            "signal_type": result.signal_type.value,
            "similarity": round(result.similarity, 4),
            "_schema_version": "1.0",
        }
    )


@mcp_tool_observe
def signal_stats() -> str:
    """Return aggregated interaction-signal counts for the workspace.

    Useful for operators monitoring how often users re-query / correct /
    refine — a spike in corrections usually flags a regression in the
    underlying recall pipeline.
    """
    from mind_mem.interaction_signals import SignalStore

    ws = _workspace()
    ws_err = _check_workspace(ws)
    if ws_err:
        return ws_err

    store = SignalStore(_signal_store_path(ws))
    return json.dumps(
        {**store.stats().as_dict(), "_schema_version": "1.0"},
        indent=2,
    )


def register(mcp) -> None:
    """Wire the signal tools onto *mcp*."""
    mcp.tool(observe_signal)
    mcp.tool(signal_stats)
