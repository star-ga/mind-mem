"""Chat surface — grounded question answering over the workspace.

Wraps :func:`mind_mem.chat_memory.chat_with_memory` as a single MCP
tool. Unlike ``recall``, which hands back ranked blocks for the caller
to read, ``chat_with_memory`` returns a *sentence-level cited answer*:
every claim carries a ``[[block_id]]`` that has been resolved against
the workspace before the tool returns.

Default posture is deliberately conservative:

* ``generator="extractive"`` — deterministic, offline, quotes the
  evidence verbatim. ``"service"`` opts into the local generation
  service.
* ``on_invalid="reject"`` — an ungrounded answer is replaced by the
  literal ``"no record found"`` and flagged, rather than raising
  through the MCP boundary.
* Empty recall short-circuits to ``"no record found"`` without invoking
  any generator.
"""

from __future__ import annotations

import json
from typing import Any

from ..infra.observability import mcp_tool_observe
from ..infra.workspace import _check_workspace, _workspace
from ._helpers import get_logger

_log = get_logger("mcp_server")

__all__ = ["chat_with_memory", "register"]


_MAX_QUESTION_LEN = 8192


@mcp_tool_observe
def chat_with_memory(
    question: str,
    limit: int = 8,
    generator: str = "extractive",
    on_invalid: str = "reject",
    require_in_evidence: bool = False,
) -> str:
    """Answer *question* from workspace memory with verified citations.

    Args:
        question: Natural-language question (≤8192 chars).
        limit: Max blocks to recall (1–50).
        generator: ``"extractive"`` (default, offline + deterministic)
            or ``"service"`` (local generation service).
        on_invalid: ``"reject"`` (default) returns the no-record string
            with the violation attached; ``"raise"`` surfaces the
            grounding failure as a structured error.
        require_in_evidence: Also reject citations that resolve in the
            workspace but were not part of the recalled evidence.

    Returns:
        JSON string of the :class:`~mind_mem.chat_memory.ChatAnswer`
        payload — ``answer``, ``citations``, ``evidence``, ``report``.
        Empty recall yields ``answer == "no record found"``.
    """
    if not isinstance(question, str) or not question.strip():
        return json.dumps({"error": "question must be a non-empty string"})
    if len(question) > _MAX_QUESTION_LEN:
        return json.dumps({"error": f"question must be ≤{_MAX_QUESTION_LEN} characters"})

    ws = _workspace()
    ws_err = _check_workspace(ws)
    if ws_err:
        return ws_err

    from mind_mem.chat_citations import CitationError
    from mind_mem.chat_generators import GeneratorError, resolve_generator
    from mind_mem.chat_memory import chat_with_memory as _chat

    try:
        gen = resolve_generator(generator)
    except ValueError as exc:
        return json.dumps({"error": str(exc)})

    try:
        result = _chat(
            ws,
            question,
            generator=gen,
            limit=limit,
            on_invalid=on_invalid,
            require_in_evidence=bool(require_in_evidence),
        )
    except CitationError as exc:
        _log.warning("chat_with_memory_ungrounded", reason=str(exc))
        return json.dumps({"error": "ungrounded answer rejected", "report": exc.report.to_dict()})
    except (ValueError, TypeError) as exc:
        return json.dumps({"error": str(exc)})
    except GeneratorError as exc:
        return json.dumps({"error": str(exc)})

    return json.dumps(result.to_dict(), indent=2, default=str)


def register(mcp: Any) -> None:
    """Wire the chat tool onto *mcp*."""

    mcp.tool(chat_with_memory)
