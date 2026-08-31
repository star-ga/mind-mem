"""MCP wrapping for the v3.11.0 deterministic quality gate.

Exposes the :mod:`mind_mem.quality_gate` rule engine as a single MCP
tool, ``validate_block``, that AI clients can call to pre-flight a
candidate block before staging it through ``propose_update``.

The MCP tool is *advisory by default*; AI clients that ignore the
verdict still write through, but the gate's reasoning is now visible
to the caller. Strict mode is opted into by the tool's ``strict`` arg
or by the workspace config — ``mind-mem.json``:
``{"quality_gate": {"mode": "strict"}}``, the same key
``propose_update`` enforces on (``mcp.infra.config._get_quality_gate_mode``).

The tool exists to pre-flight a write, so it reads that config rather
than assuming advisory: a preview that answers ``accept: true`` for a
statement ``propose_update`` will reject with ``quality_gate_rejection``
is a preview of nothing.
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

    The workspace's configured ``quality_gate.mode`` is honoured, so the
    verdict matches what ``propose_update`` would do with the same text:
    a workspace in ``strict`` mode previews as strict without the caller
    having to know that. ``strict=True`` additionally forces strict
    evaluation in a workspace that is not.

    The near-duplicate rule is evaluated against the same recent-proposal
    window the enforcer uses (``governance._recent_statements``), so a
    statement that would be rejected as a duplicate previews as one.

    Args:
        text: Candidate block content to validate.
        strict: When ``True``, fired rules reject regardless of the
            workspace mode. Default ``False`` defers to the workspace:
            advisory (rules reported, verdict still ``accept``\\ s)
            unless the config says ``strict``.
        force: Escape hatch — when ``True``, the verdict is forced to
            ``accept=True`` even if rules fire. The ``forced`` flag in
            the response confirms this. Use only when caller has
            already validated the input out-of-band.

    Returns:
        JSON string of the
        :class:`mind_mem.quality_gate.QualityGateVerdict` `to_dict`
        payload, plus the ``mode`` that was applied and whether the
        evaluation was ``strict``.
    """

    from mind_mem.mcp.infra.config import _get_quality_gate_mode
    from mind_mem.mcp.infra.workspace import _workspace
    from mind_mem.mcp.tools.governance import _recent_statements
    from mind_mem.quality_gate import validate_block as _validate

    if not isinstance(text, str):
        return json.dumps({"error": "text must be a string", "got": type(text).__name__})

    ws = _workspace()
    mode = _get_quality_gate_mode(ws)
    # ``off`` means propose_update never runs the gate, so a preview must
    # not reject either; it still reports which rules fired.
    effective_strict = bool(strict) or mode == "strict"

    # Same window ``propose_update`` builds, from the same function — a
    # preview that omits it silently drops rule 6 (near_duplicate) and then
    # answers ``accept: true`` for a statement the enforcer will reject as a
    # duplicate. That is the failure this tool exists to prevent.
    verdict = _validate(text, strict=effective_strict, force=bool(force), recent=_recent_statements(ws))
    payload = verdict.to_dict()
    payload["mode"] = mode
    payload["strict"] = effective_strict
    payload["text_chars"] = len(text)
    payload["text_non_ws_chars"] = sum(1 for c in text if not c.isspace())
    return json.dumps(payload, indent=2)


def register(mcp: Any) -> None:
    """Wire quality-gate tools onto *mcp*."""

    mcp.tool(validate_block)
