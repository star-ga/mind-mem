"""MCP wrapping for v3.9 walkthrough + persona projection.

v3.10.0 follow-up to the v3.9 ``walkthrough.py`` + ``personas.py``
modules. v3.9 wired both into the HTTP REST adapter; this surfaces
them through the MCP server so MCP-native clients (Claude Code,
Codex CLI, Gemini CLI, Cursor, Windsurf, Zed) can call them
without going through HTTP.

Two new tools:

* :func:`compile_truth_walkthrough` — wraps
  :func:`mind_mem.walkthrough.compile_walkthrough`. Returns a
  dependency-ordered learning sequence
  (``foundation`` → ``context`` → ``current``).

* :func:`recall_with_persona` — additive recall variant that runs
  ``recall(query, limit, active_only)`` and then applies
  :func:`mind_mem.personas.apply_persona` to the result list.
  Kept separate from the existing ``recall`` so the long-stable
  envelope schema is not broken — clients that want persona
  projection opt in by name.
"""

from __future__ import annotations

import json
from typing import Any

from ..infra.observability import mcp_tool_observe
from ..infra.workspace import _check_workspace, _workspace
from ._helpers import get_logger

_log = get_logger("mcp_server")


__all__ = [
    "compile_truth_walkthrough",
    "recall_with_persona",
    "register",
]


_DEFAULT_LIMIT = 25
_MAX_LIMIT = 100
_MAX_TOPIC_LEN = 4096


@mcp_tool_observe
def compile_truth_walkthrough(
    topic: str,
    limit: int = _DEFAULT_LIMIT,
    active_only: bool = False,
) -> str:
    """Return *topic* memories in dependency order (foundations → current).

    Re-orders the recall result for *topic* into a learning sequence
    so agents and humans can read step 1, step 2, ... instead of a
    flat ranked list. Step roles:

    * ``foundation`` — first ~30%
    * ``context``    — middle ~40%
    * ``current``    — last ~30%

    Algorithm: chronological backbone from each block's id
    (YYYYMMDD prefix) reinforced by the workspace co-retrieval
    graph. Topo-sort with Kahn's algorithm; cycles broken
    deterministically. See :mod:`mind_mem.walkthrough`.

    Args:
        topic: Search query.
        limit: Maximum candidates to consider (1..100).
        active_only: Only emit blocks with active status.

    Returns:
        JSON envelope with ``steps`` (list of
        ``{step, block_id, role, score, subject}`` dicts) and
        ``count``. Empty steps list when *topic* yields no matches.
    """
    if not isinstance(topic, str) or not topic.strip():
        return json.dumps({"error": "topic must be a non-empty string"})
    if len(topic) > _MAX_TOPIC_LEN:
        return json.dumps({"error": f"topic must be ≤{_MAX_TOPIC_LEN} characters"})
    if not 1 <= limit <= _MAX_LIMIT:
        return json.dumps({"error": f"limit must be in [1, {_MAX_LIMIT}]"})

    ws = _workspace()
    ws_err = _check_workspace(ws)
    if ws_err:
        return ws_err

    from mind_mem.walkthrough import compile_walkthrough

    try:
        steps = compile_walkthrough(
            workspace=ws,
            topic=topic,
            limit=int(limit),
            active_only=bool(active_only),
        )
    except ValueError as exc:
        return json.dumps({"error": str(exc)})

    return json.dumps(
        {
            "topic": topic,
            "count": len(steps),
            "steps": steps,
        },
        indent=2,
        default=str,
    )


@mcp_tool_observe
def recall_with_persona(
    query: str,
    persona: str = "detailed",
    limit: int = 10,
    active_only: bool = False,
) -> str:
    """Recall memories and project the result list through *persona*.

    Three personas:

    * ``brief``     — ``{id, score, subject}`` only; fits routing
                      layers, Slack snippets, status panels.
    * ``detailed``  — full block (matches the existing ``recall``
                      default).
    * ``technical`` — full block plus promoted governance fields
                      (``axis_scores``, ``governance_state``,
                      ``provenance_hash``, ``source_span``,
                      ``transform_hash``); fits audit consumers.

    See :mod:`mind_mem.personas` for the projection rules.

    Args:
        query: Search query.
        persona: One of ``brief``, ``detailed``, ``technical``.
        limit: Result cap (1..500).
        active_only: Only emit blocks with active status.

    Returns:
        JSON envelope ``{persona, query, count, results}``.
    """
    if not isinstance(query, str) or not query.strip():
        return json.dumps({"error": "query must be a non-empty string"})
    if not 1 <= limit <= 500:
        return json.dumps({"error": "limit must be in [1, 500]"})

    from mind_mem.personas import PERSONAS, PersonaError, apply_persona

    if persona not in PERSONAS:
        return json.dumps(
            {"error": f"unknown persona {persona!r}; must be one of {list(PERSONAS)}"}
        )

    ws = _workspace()
    ws_err = _check_workspace(ws)
    if ws_err:
        return ws_err

    # Reuse the recall MCP impl so cache, axes, observability all flow through.
    from .recall import _recall_impl

    raw = _recall_impl(query, limit=int(limit), active_only=bool(active_only))
    try:
        envelope = json.loads(raw)
    except json.JSONDecodeError:
        return raw  # _recall_impl already returned an error JSON

    if not isinstance(envelope, dict) or "results" not in envelope:
        return raw

    results: list[dict[str, Any]] = list(envelope.get("results", []))
    try:
        projected = apply_persona(results, persona)
    except PersonaError as exc:
        return json.dumps({"error": str(exc)})

    return json.dumps(
        {
            "persona": persona,
            "query": query,
            "count": len(projected),
            "results": projected,
        },
        indent=2,
        default=str,
    )


def register(mcp: Any) -> None:
    """Wire walkthrough + persona tools onto *mcp*."""
    mcp.tool(compile_truth_walkthrough)
    mcp.tool(recall_with_persona)
