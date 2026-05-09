"""MCP wrapping for the v3.11.0 deterministic quality gate.

Exposes the :mod:`mind_mem.quality_gate` rule engine as a single MCP
tool, ``validate_block``, that AI clients can call to pre-flight a
candidate block before staging it through ``propose_update``.

The MCP tool is *advisory by default*; AI clients that ignore the
verdict still write through, but the gate's reasoning is now visible
to the caller. Strict-mode opt-in is via the tool's ``strict`` arg or
the workspace config (``mind-mem.json``: ``quality_gate_mode``).
"""

from __future__ import annotations

import json
from typing import Any

from ..infra.observability import mcp_tool_observe
from ._helpers import get_logger

_log = get_logger("mcp_server")


__all__ = ["validate_block", "register"]


@mcp_tool_observe
def validate_block(
    text: str,
    strict: bool = False,
    force: bool = False,
) -> str:
    """Run the deterministic quality gate against ``text``.

    Args:
        text: Candidate block content to validate.
        strict: When ``True``, fired rules reject. Default ``False``
            keeps the call advisory (rules are reported but the
            verdict still ``accept``\\ s).
        force: Escape hatch — when ``True``, the verdict is forced to
            ``accept=True`` even if rules fire. The ``forced`` flag in
            the response confirms this. Use only when caller has
            already validated the input out-of-band.

    Returns:
        JSON string of the
        :class:`mind_mem.quality_gate.QualityGateVerdict` `to_dict`
        payload.
    """

    from mind_mem.quality_gate import validate_block as _validate

    if not isinstance(text, str):
        return json.dumps({"error": "text must be a string", "got": type(text).__name__})

    verdict = _validate(text, strict=bool(strict), force=bool(force))
    payload = verdict.to_dict()
    payload["text_chars"] = len(text)
    payload["text_non_ws_chars"] = sum(1 for c in text if not c.isspace())
    return json.dumps(payload, indent=2)


def register(mcp: Any) -> None:
    """Wire quality-gate tools onto *mcp*."""

    mcp.tool(validate_block)
