# Copyright 2026 STARGA, Inc.
"""Workspace builders for the ``mm review`` test suite.

Builds a real, ``validate_py``-clean workspace with real staged proposals
rendered by the production serialiser, so the review surface is exercised
against blocks the apply engine would actually accept. Nothing here writes
to the corpus outside the temp directory it is handed.
"""

from __future__ import annotations

import json
import os
from typing import Any, Iterator, Mapping, Sequence

import pytest

DECISION_FILE = "decisions/DECISIONS.md"
PROPOSAL_FILE = "intelligence/proposed/EDITS_PROPOSED.md"


def _decision_block(index: int) -> str:
    return (
        f"\n[D-20260801-{index:03d}]\n"
        f"Type: decision\n"
        f"Status: active\n"
        f"Statement: Baseline decision number {index} about the storage backend.\n"
        f"Scope: global\n"
        f"Rationale: seeded for the review-queue tests.\n"
        f"Supersedes: none\n"
        f"Date: 2026-08-01\n"
        f"Tags: baseline\n"
        f"Sources:\n- seed\n"
    )


def build_proposal(index: int, *, created: str = "", day: str = "20260829") -> dict[str, Any]:
    """One valid ``edit`` proposal retagging decision *index*."""
    from mind_mem.apply_engine import compute_fingerprint

    proposal: dict[str, Any] = {
        "ProposalId": f"P-{day}-{index:03d}",
        "Type": "edit",
        "TargetBlock": f"D-20260801-{index:03d}",
        "Risk": "low",
        "Evidence": [f"observed drift in decision {index}"],
        "Rollback": "restore from snapshot",
        "Ops": [
            {
                "op": "update_field",
                "file": DECISION_FILE,
                "target": f"D-20260801-{index:03d}",
                "field": "Tags",
                "value": f"baseline,reviewed-{index}",
            }
        ],
        "Status": "staged",
        "FilesTouched": [DECISION_FILE],
        "Sources": [f"D-20260801-{index:03d}"],
    }
    proposal["Fingerprint"] = compute_fingerprint(proposal)
    if created:
        proposal["Created"] = created
    return proposal


def _render_proposal_block(proposal: Mapping[str, Any]) -> str:
    """Serialise *proposal* to the canonical proposal-block markdown.

    Inlined here in 5.0.0. This was ``mind_mem.lint_autofix.render_proposal_block``
    until that module was removed as unreachable -- nothing in the product
    imported it; only this fixture did. The function is pure formatting with no
    dependencies, and eight review test files build their corpora through it, so
    it moves to the test support layer rather than keeping a production module
    alive for test-only use. The wire format below is the on-disk proposal block
    the review CLI parses, so it must stay byte-compatible with the parser.
    """
    ops_lines: list[str] = []
    for op in proposal.get("Ops", []):
        ops_lines.append(f"- op: {op['op']}")
        for key in ("file", "target", "field", "value", "status"):
            if key in op:
                ops_lines.append(f"  {key}: {op[key]}")
    evidence = "\n".join(f"- {line}" for line in proposal["Evidence"])
    touched = "\n".join(f"- {line}" for line in proposal["FilesTouched"])
    sources = "\n".join(f"- {line}" for line in proposal["Sources"])
    return (
        f"\n[{proposal['ProposalId']}]\n"
        f"ProposalId: {proposal['ProposalId']}\n"
        f"Type: {proposal['Type']}\n"
        f"TargetBlock: {proposal['TargetBlock']}\n"
        f"Risk: {proposal['Risk']}\n"
        f"Evidence:\n{evidence}\n"
        f"Rollback: {proposal['Rollback']}\n"
        f"Ops:\n" + "\n".join(ops_lines) + "\n"
        f"Fingerprint: {proposal['Fingerprint']}\n"
        f"Status: {proposal['Status']}\n"
        f"FilesTouched:\n{touched}\n"
        f"Sources:\n{sources}\n"
    )


def render(proposal: Mapping[str, Any]) -> str:
    """Serialise one proposal, carrying ``Created`` through when present."""
    text = _render_proposal_block(proposal)
    created = proposal.get("Created")
    if created:
        text = text.replace(
            f"Status: {proposal['Status']}\n",
            f"Status: {proposal['Status']}\nCreated: {created}\n",
        )
    return text


def build_workspace(
    root: str,
    count: int = 3,
    *,
    created: Sequence[str] | None = None,
    backlog_limit: int = 500,
) -> tuple[str, ...]:
    """Initialise *root* with *count* decisions and *count* staged proposals.

    Returns the proposal ids in file order.
    """
    from mind_mem.init_workspace import init

    init(root)
    _relax_gates(root, backlog_limit=backlog_limit)

    decisions = "".join(_decision_block(i) for i in range(1, count + 1))
    _write(os.path.join(root, DECISION_FILE), "# Decisions\n" + decisions)

    proposals = [build_proposal(i, created=(created[i - 1] if created else "")) for i in range(1, count + 1)]
    _write(
        os.path.join(root, PROPOSAL_FILE),
        "# Proposed Edits\n" + "".join(render(p) for p in proposals),
    )
    return tuple(str(p["ProposalId"]) for p in proposals)


def _relax_gates(root: str, *, backlog_limit: int) -> None:
    """Enable applies and lift the backlog ceiling for the fixture workspace."""
    state_path = os.path.join(root, "memory/intel-state.json")
    with open(state_path, encoding="utf-8") as handle:
        state = json.load(handle)
    state["governance_mode"] = "propose_apply"
    state.pop("last_apply_ts", None)
    _write(state_path, json.dumps(state, indent=2))

    config_path = os.path.join(root, "mind-mem.json")
    with open(config_path, encoding="utf-8") as handle:
        config = json.load(handle)
    config.setdefault("proposal_budget", {})["backlog_limit"] = backlog_limit
    config["governance_mode"] = "propose_apply"
    _write(config_path, json.dumps(config, indent=2))


def clear_no_touch_window(root: str) -> None:
    """Drop ``last_apply_ts`` so the next real apply is not rate-limited."""
    state_path = os.path.join(root, "memory/intel-state.json")
    with open(state_path, encoding="utf-8") as handle:
        state = json.load(handle)
    state.pop("last_apply_ts", None)
    _write(state_path, json.dumps(state, indent=2))


def proposal_status(root: str, proposal_id: str) -> str:
    """Current ``Status`` of *proposal_id*, or ``""`` when absent."""
    from mind_mem.apply_engine import find_proposal

    proposal, _source = find_proposal(root, proposal_id)
    return str(proposal.get("Status", "")) if proposal else ""


def _write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


@pytest.fixture(autouse=True)
def mcp_budget() -> Iterator[None]:
    """Return the MCP rate-limit budget this test consumes.

    ``mcp_tool_observe`` enforces a **process-global** sliding window of
    120 tool calls per 60 seconds, keyed on ``pid-<pid>`` — one bucket
    for an entire pytest session. The review tests make real
    ``approve_apply`` / ``reject_proposal`` / ``verify_chain`` calls, so
    left alone they drain that bucket and every *later* test in the
    session silently reads ``{"error": "Rate limit exceeded"}`` instead
    of its tool's payload. That surfaces as unrelated ``KeyError``s in
    files that have nothing to do with review, which is a miserable
    thing to debug.

    Resetting the limiter around each test keeps the failure local. The
    durable fix is the same reset as an autouse fixture in
    ``tests/conftest.py`` so it covers every MCP-touching test file, not
    only these; that is a shared surface and is left to its owner.
    """
    _reset_rate_limiters()
    try:
        yield
    finally:
        _reset_rate_limiters()


def _reset_rate_limiters() -> None:
    from mind_mem.mcp.infra import rate_limit

    with rate_limit._rate_limiters_lock:
        rate_limit._rate_limiters.clear()
