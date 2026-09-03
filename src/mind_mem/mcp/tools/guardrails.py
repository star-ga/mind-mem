# Copyright 2026 STARGA, Inc.
"""MCP surface for GUARDRAIL blocks — trigger-fired constraints.

Two read-only tools:

* ``check_guardrails`` — pure trigger evaluation.  Given what the agent is
  about to do (tool / command / intent / paths), return the constraints
  that apply.  No query, no ranker, no model: a client can call this
  before every risky action and get a deterministic answer.
* ``recall_with_guardrails`` — ordinary recall for *query*, with the
  constraints firing for the same context surfaced ahead of the ranked
  hits.  Bounded by ``recall.guardrails.max_surfaced``.

Both are ``USER_TOOLS`` scope and never write.  Neither can mint a
guardrail: ``[GR-...]`` blocks are operator-authored in
``guardrails/GUARDRAILS.md`` and read back read-only, and a block carrying
external-ingest / imported provenance is refused recognition outright
(``mind_mem.guardrails.guardrail_provenance_refusal``).

The recall cache is deliberately bypassed here — its key does not include
the guardrail context, so a cached envelope from a different context would
answer with the wrong constraints.  Correctness beats the cache hit on a
prohibition.
"""

from __future__ import annotations

import json
from typing import Any, Sequence

from mind_mem.guardrail_surface import guardrail_hits
from mind_mem.guardrails import GuardrailContext, GuardrailPolicy, GuardrailSpecError
from mind_mem.recall import recall as recall_engine

from ..infra.config import _load_config
from ..infra.constants import MCP_SCHEMA_VERSION
from ..infra.observability import mcp_tool_observe
from ..infra.workspace import _check_workspace, _workspace
from ._helpers import get_logger

_log = get_logger("mcp_server")

__all__ = ["check_guardrails", "recall_with_guardrails", "register"]

_MAX_QUERY_LEN = 8192
_MAX_PATHS = 64


def _build_context(
    tool: str,
    command: str,
    intent: str,
    paths: Sequence[str] | str | None,
) -> GuardrailContext:
    """Validate the MCP argument boundary into a context.

    Raises:
        GuardrailSpecError: an argument has the wrong type or is oversized.
    """
    for name, value in (("tool", tool), ("command", command), ("intent", intent)):
        if not isinstance(value, str):
            raise GuardrailSpecError(f"{name} must be a string, got {type(value).__name__}")
    if isinstance(paths, str):
        paths = [paths]
    elif paths is None:
        paths = []
    elif not isinstance(paths, (list, tuple)):
        raise GuardrailSpecError(f"paths must be a string or list, got {type(paths).__name__}")
    if len(paths) > _MAX_PATHS:
        raise GuardrailSpecError(f"paths may declare at most {_MAX_PATHS} entries")
    for entry in paths:
        if not isinstance(entry, str):
            raise GuardrailSpecError(f"paths entries must be strings, got {type(entry).__name__}")
    return GuardrailContext(
        tool=tool,
        command=command,
        intent=intent,
        paths=tuple(p for p in paths if p.strip()),
    )


@mcp_tool_observe
def check_guardrails(
    tool: str = "",
    command: str = "",
    intent: str = "",
    paths: list[str] | None = None,
) -> str:
    """Return the guardrails that fire for an about-to-happen action.

    Args:
        tool: Tool the agent is about to invoke (e.g. ``"Bash"``).
        command: Command line / call payload the tool would run.
        intent: Intent class for the action (e.g. ``"destructive_actions"``).
        paths: Files the action would touch.

    Returns:
        JSON: ``{"count": N, "context": {...}, "guardrails": [...]}``.
        Each entry carries ``guardrail_constraint`` (the rule text),
        ``guardrail_severity`` and ``guardrail_triggers`` (which
        dimensions matched).  Matching is declarative and deterministic —
        the same context always yields the same list, in the same order.
    """
    ws = _workspace()
    ws_err = _check_workspace(ws)
    if ws_err:
        return ws_err
    try:
        context = _build_context(tool, command, intent, paths)
    except GuardrailSpecError as exc:
        return json.dumps({"error": str(exc)})

    policy = GuardrailPolicy.from_config(_load_config(ws))
    hits = guardrail_hits(ws, context, policy)
    _log.info("mcp_check_guardrails", tool=context.tool, intent=context.intent, count=len(hits))
    return json.dumps(
        {
            "_schema_version": MCP_SCHEMA_VERSION,
            "count": len(hits),
            "max_surfaced": policy.max_surfaced,
            "context": {
                "tool": context.tool,
                "command": context.command,
                "intent": context.intent,
                "paths": list(context.paths),
            },
            "guardrails": hits,
        },
        indent=2,
        default=str,
    )


@mcp_tool_observe
def recall_with_guardrails(
    query: str,
    tool: str = "",
    command: str = "",
    intent: str = "",
    paths: list[str] | None = None,
    limit: int = 10,
    active_only: bool = False,
) -> str:
    """Recall *query*, with the constraints for this context surfaced first.

    Identical to ``recall`` except that guardrail blocks whose triggers
    match the supplied context are returned ahead of the ranked hits,
    regardless of similarity score.  Guardrail hits are marked
    ``"guardrail": true`` so a client can render them as constraints
    rather than as evidence.

    Returns:
        JSON: ``{"query": ..., "count": N, "guardrail_count": G,
        "results": [...], "attestation": {...}}``.

    ``attestation`` is the ``RECALL_ATTEST_v2`` record for this run, derived by
    the serving entry this tool calls and committing to the ids in
    ``results`` — guardrail-surfaced blocks included, because they were served.
    """
    if not isinstance(query, str):
        return json.dumps({"error": "query must be a string"})
    if len(query) > _MAX_QUERY_LEN:
        return json.dumps({"error": f"query must be ≤{_MAX_QUERY_LEN} characters"})
    ws = _workspace()
    ws_err = _check_workspace(ws)
    if ws_err:
        return ws_err
    try:
        context = _build_context(tool, command, intent, paths)
    except GuardrailSpecError as exc:
        return json.dumps({"error": str(exc)})
    try:
        bounded_limit = max(1, min(int(limit), 100))
    except (TypeError, ValueError):
        return json.dumps({"error": "limit must be an integer"})

    results: list[dict[str, Any]] = recall_engine(
        ws,
        query,
        limit=bounded_limit,
        active_only=bool(active_only),
        guardrail_context=context,
    )
    guardrail_count = sum(1 for r in results if r.get("guardrail"))
    _log.info("mcp_recall_with_guardrails", query=query, results=len(results), guardrails=guardrail_count)
    return json.dumps(
        {
            "_schema_version": MCP_SCHEMA_VERSION,
            "query": query,
            "count": len(results),
            "guardrail_count": guardrail_count,
            "results": results,
            "attestation": getattr(results, "attestation", None),
        },
        indent=2,
        default=str,
    )


def register(mcp: Any) -> None:
    """Wire the guardrail tools onto *mcp*."""

    mcp.tool(check_guardrails)
    mcp.tool(recall_with_guardrails)
