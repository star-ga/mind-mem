# Copyright 2026 STARGA, Inc.
"""MCP wrapping for the deterministic corpus lint + its governed repair path.

Two tools, deliberately split by what they can do rather than by what they
are about:

* :func:`lint` — read-only. Walks the corpus and returns content-addressed
  findings (``LF-xxxxxxxx``). Writes nothing, so it is USER-scope.
* :func:`lint_autofix` — hands ONE finding to
  :mod:`mind_mem.lint_autofix`, which stages a governance *proposal* in
  ``intelligence/proposed/EDITS_PROPOSED.md`` and stops. The block of record
  still only changes under ``approve_apply``. Staging a proposal is the same
  class of act as ``propose_update``, so it is ADMIN-scope.

Both are gated on the ``v4.lint`` flag (default OFF), probed through
:func:`mind_mem.lint.flag_enabled` — a quiet, fail-closed read, so a call
made while the surface is off is answered without emitting anything the
unwired build would not.

This is NOT the quality gate. ``validate_block`` (``mcp/tools/quality.py``)
filters a *candidate* before it is staged; the lint scans the corpus that is
already stored. Zero rule overlap, opposite ends of the write path — they are
complementary and must stay separate tools.
"""

from __future__ import annotations

import json
from typing import Any

from ..infra.constants import MCP_SCHEMA_VERSION
from ..infra.observability import mcp_tool_observe
from ..infra.workspace import _check_workspace, _workspace
from ._helpers import get_logger

_log = get_logger("mcp_server")

__all__ = ["lint", "lint_autofix", "register"]


def _disabled_payload(ws: str) -> str:
    """The one answer both tools give when ``v4.lint`` is off."""
    return json.dumps(
        {
            "_schema_version": MCP_SCHEMA_VERSION,
            "workspace": ws,
            "error": "v4.lint is disabled",
            "enable": 'mind-mem.json: "v4": { "lint": { "enabled": true } }',
        },
        indent=2,
    )


@mcp_tool_observe
def lint(rule: str = "") -> str:
    """Report deterministic corpus defects (read-only, USER-scope).

    Walks the governed Markdown corpus and returns every finding, ordered
    deterministically by ``(file, rule, block_id, finding_id)``. Each finding
    carries a stable, content-addressed ``finding_id`` (``LF-xxxxxxxx``) that
    is a pure function of the defect, not of when the scan ran — the same
    corpus yields the same ids on any machine, which is what makes a finding
    id safe to quote back to :func:`lint_autofix`.

    Nothing is written and nothing is proposed. Requires the ``v4.lint`` flag.

    Args:
        rule: Optional single rule to run — ``stale_date``,
            ``missing_metadata`` or ``duplicate_block``. Empty (the default)
            runs all three.

    Returns:
        JSON with ``count``, ``autofixable``, and the ``findings`` list.
    """
    from mind_mem.lint import RULES, LintError, flag_enabled
    from mind_mem.lint import lint as _lint

    ws = _workspace()
    ws_err = _check_workspace(ws)
    if ws_err:
        return ws_err

    if not flag_enabled(ws):
        return _disabled_payload(ws)

    if not isinstance(rule, str):
        return json.dumps({"error": "rule must be a string", "got": type(rule).__name__})
    selected = [rule] if rule.strip() else None
    if selected is not None and rule.strip() not in RULES:
        return json.dumps({"error": f"unknown lint rule: {rule!r}", "known": list(RULES)})

    try:
        findings = _lint(ws, rules=selected)
    except LintError as exc:
        return json.dumps({"error": str(exc)})

    return json.dumps(
        {
            "_schema_version": MCP_SCHEMA_VERSION,
            "workspace": ws,
            "rules": list(selected) if selected else list(RULES),
            "count": len(findings),
            "autofixable": sum(1 for f in findings if f.autofixable),
            "findings": [f.as_dict() for f in findings],
        },
        indent=2,
    )


@mcp_tool_observe
def lint_autofix(finding_id: str) -> str:
    """Stage a repair PROPOSAL for one lint finding (ADMIN-scope).

    The corpus is not modified. The single file written is
    ``intelligence/proposed/EDITS_PROPOSED.md``, where the repair lands as a
    fingerprinted, ``staged`` proposal for a human to review. Applying it is a
    separate, explicitly human act — ``approve_apply``.

    Admin-scoped for the same reason ``propose_update`` is: it puts an item in
    front of the operator's approval gate. It cannot bypass that gate, and a
    finding with no deterministic, content-free repair is refused rather than
    guessed at — the lint never invents prose to fill a field.

    Args:
        finding_id: A ``LF-xxxxxxxx`` id from :func:`lint`.

    Returns:
        JSON with the staged ``proposal_id``, or a structured ``error``.
    """
    from mind_mem.lint import flag_enabled
    from mind_mem.lint_autofix import LintAutofixError
    from mind_mem.lint_autofix import lint_autofix as _autofix

    ws = _workspace()
    ws_err = _check_workspace(ws)
    if ws_err:
        return ws_err

    if not flag_enabled(ws):
        return _disabled_payload(ws)

    if not isinstance(finding_id, str) or not finding_id.strip():
        return json.dumps({"error": "finding_id must be a non-empty string"})

    try:
        proposal_id = _autofix(ws, finding_id.strip())
    except LintAutofixError as exc:
        return json.dumps({"finding_id": finding_id, "error": str(exc), "error_kind": type(exc).__name__})

    _log.info("mcp_lint_autofix", finding_id=finding_id, proposal_id=proposal_id)
    return json.dumps(
        {
            "_schema_version": MCP_SCHEMA_VERSION,
            "workspace": ws,
            "finding_id": finding_id,
            "proposal_id": proposal_id,
            "status": "staged",
            "next_step": f"Nothing has changed yet. Review it, then call approve_apply('{proposal_id}').",
        },
        indent=2,
    )


def register(mcp: Any) -> None:
    """Wire the lint tools onto *mcp*."""

    mcp.tool(lint)
    mcp.tool(lint_autofix)
