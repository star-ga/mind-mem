# Copyright 2026 STARGA, Inc.
"""MCP surface for TASK-FRAME + DEAD-END blocks — continuity and negative memory.

Two read-only tools, both deterministic and both model-free:

* ``resume_brief`` — what session N+1 needs before it does anything: the
  goal, the plan steps, what was tried, what is believed, what remains,
  and the dead ends that overlap this task.  Call it once at session
  start instead of re-deriving context from the repository.
* ``check_dead_ends`` — the about-to-act check.  Given what the agent is
  about to do (tool / command / intent / paths), return the recorded
  failures that overlap it.  No query, no ranker: a client can call this
  before every expensive approach and get the same answer every time.

Neither tool writes.  ``[TF-...]`` and ``[DE-...]`` blocks are authored on
the governed proposal route like every other block kind and only ever read
back here; a block carrying external-ingest provenance is refused
recognition outright (``mind_mem.guardrails.guardrail_provenance_refusal``).

A dead end is **evidence, never a prohibition**: these tools report, they
never refuse an action and never filter a plan.  The operator decides.
"""

from __future__ import annotations

import json
from typing import Any, Sequence

from mind_mem.dead_ends import load_dead_ends_with_rejections, match_surface
from mind_mem.guardrails import GuardrailContext, GuardrailSpecError
from mind_mem.resume_brief import resume_brief as resume_brief_engine
from mind_mem.task_frames import ApproachSurface, FramePolicy, FrameSpecError

from ..infra.config import _load_config
from ..infra.constants import MCP_SCHEMA_VERSION
from ..infra.observability import mcp_tool_observe
from ..infra.workspace import _check_workspace, _workspace
from ._helpers import get_logger

_log = get_logger("mcp_server")

__all__ = ["check_dead_ends", "register", "resume_brief"]

_MAX_FRAME_ID_LEN = 128
_MAX_PATHS = 64


def _build_context(
    tool: str,
    command: str,
    intent: str,
    paths: Sequence[str] | str | None,
) -> GuardrailContext:
    """Validate the MCP argument boundary into a single-action context.

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
def resume_brief(frame_id: str = "") -> str:
    """Return the resume brief for the active task frame.

    Args:
        frame_id: A specific ``TF-...`` id.  Empty (the default) selects
            the active frame — the live frame with the highest block ID.

    Returns:
        JSON: ``{"frame_id": ..., "goal": ..., "steps": [...],
        "tried": [...], "believed": [...], "remaining": [...],
        "dead_ends": [...], "dead_end_count": N}``.  ``frame_id`` is ``""``
        when the workspace declares no live frame — an empty brief, not an
        error, so a session can ask unconditionally.

        Each ``dead_ends`` entry carries ``approach``, ``why_failed``,
        ``outcome``, ``evidence`` and ``matched`` (which declared
        dimensions overlapped).  They are **warnings, not prohibitions**:
        re-running a recorded failure is sometimes right, and this tool has
        no standing to refuse it.

        ``dead_end_total`` / ``dead_ends_elided`` say how many actually
        fired versus how many fit under ``max_warnings``, and ``rejected``
        names every ``[TF-...]`` / ``[DE-...]`` block the workspace
        declared that could not be read.  An empty ``frame_id`` with a
        non-empty ``rejected`` means "continuity exists but is
        unreadable", which is not the same answer as "no continuity".
    """
    if not isinstance(frame_id, str):
        return json.dumps({"error": "frame_id must be a string"})
    if len(frame_id) > _MAX_FRAME_ID_LEN:
        return json.dumps({"error": f"frame_id must be ≤{_MAX_FRAME_ID_LEN} characters"})
    ws = _workspace()
    ws_err = _check_workspace(ws)
    if ws_err:
        return ws_err

    policy = FramePolicy.from_config(_load_config(ws))
    try:
        brief = resume_brief_engine(ws, frame_id.strip(), policy=policy)
    except FrameSpecError as exc:
        return json.dumps({"error": str(exc)})

    payload: dict[str, Any] = {"_schema_version": MCP_SCHEMA_VERSION, **brief.to_dict()}
    payload["dead_end_count"] = len(brief.dead_ends)
    payload["max_warnings"] = policy.max_warnings
    _log.info(
        "mcp_resume_brief",
        frame_id=brief.frame_id,
        dead_ends=len(brief.dead_ends),
        elided=brief.dead_ends_elided,
        rejected=len(brief.rejected),
    )
    return json.dumps(payload, indent=2, default=str)


@mcp_tool_observe
def check_dead_ends(
    tool: str = "",
    command: str = "",
    intent: str = "",
    paths: list[str] | None = None,
) -> str:
    """Return the recorded failures that overlap an about-to-happen action.

    Args:
        tool: Tool the agent is about to invoke (e.g. ``"Bash"``).
        command: Command line / call payload the tool would run.
        intent: Intent class for the action (e.g. ``"prove_floor"``).
        paths: Files the action would touch.

    Returns:
        JSON: ``{"count": N, "total_matched": M, "elided": M-N,
        "context": {...}, "dead_ends": [...], "rejected": [...]}``.
        Matching is declarative and deterministic — the same context always
        yields the same list, in the same order — and advisory: a dead end
        warns, it never blocks.  An empty context matches nothing.

        ``elided`` is how many fired but did not fit under
        ``max_warnings``, and ``rejected`` names every ``[DE-...]`` block
        that could not be read.  Negative memory is never dropped
        silently: a caller that sees a short list is told it is short.
    """
    ws = _workspace()
    ws_err = _check_workspace(ws)
    if ws_err:
        return ws_err
    try:
        context = _build_context(tool, command, intent, paths)
    except GuardrailSpecError as exc:
        return json.dumps({"error": str(exc)})

    policy = FramePolicy.from_config(_load_config(ws))
    surface = ApproachSurface.from_context(context)
    registry, rejected = load_dead_ends_with_rejections(ws, policy)
    matched = match_surface(surface, registry)
    warnings = matched[: policy.max_warnings]
    _log.info(
        "mcp_check_dead_ends",
        tool=context.tool,
        intent=context.intent,
        count=len(warnings),
        elided=len(matched) - len(warnings),
    )
    return json.dumps(
        {
            "_schema_version": MCP_SCHEMA_VERSION,
            "count": len(warnings),
            "total_matched": len(matched),
            "elided": len(matched) - len(warnings),
            "max_warnings": policy.max_warnings,
            "rejected": [block.to_dict() for block in rejected],
            "context": {
                "tool": context.tool,
                "command": context.command,
                "intent": context.intent,
                "paths": list(context.paths),
            },
            "dead_ends": [warning.to_dict() for warning in warnings],
        },
        indent=2,
        default=str,
    )


def register(mcp: Any) -> None:
    """Wire the task-frame tools onto *mcp*."""

    mcp.tool(resume_brief)
    mcp.tool(check_dead_ends)
