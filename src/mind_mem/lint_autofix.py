# Copyright 2026 STARGA, Inc.
"""Turn one lint finding into a staged repair **proposal**.

``lint_autofix(workspace, finding_id)`` is the write-adjacent half of
:mod:`mind_mem.lint`. It never touches the corpus. It stages a normal
governance proposal in ``intelligence/proposed/EDITS_PROPOSED.md`` —
``Status: staged``, fingerprinted with
:func:`mind_mem.apply_engine.compute_fingerprint` — and returns the
proposal id. The block of record changes only when a human runs
``approve_apply`` / ``apply_proposal``, which is where the snapshot,
the contradiction check, the WAL and the rollback receipt live.

Why the proposal file and not ``propose_update``: ``propose_update``
stages a *statement* (a new decision/task) as a SIGNAL for a human to
formalise; a lint repair is an *edit* to an existing block, and
``approve_apply`` only resolves ids out of the ``intelligence/proposed/``
files. Staging there is what makes the repair reviewable **and**
applicable through the same HITL gate — a signal would be neither.

Failure is typed, never a traceback: :class:`UnknownFindingError`,
:class:`NotAutofixableError`, :class:`LintAutofixError`.

Zero external deps — stdlib plus ``apply_engine`` / ``block_parser``.
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Any, Final, Mapping

from .apply_engine import compute_fingerprint, validate_proposal
from .lint import (
    RULE_DUPLICATE_BLOCK,
    RULE_MISSING_METADATA,
    RULE_STALE_DATE,
    Finding,
    is_finding_id,
    lint,
    require_lint_enabled,
)
from .mind_filelock import FileLock
from .observability import get_logger, metrics

__all__ = [
    "LintAutofixError",
    "NotAutofixableError",
    "PROPOSAL_FILE",
    "UnknownFindingError",
    "build_proposal",
    "lint_autofix",
    "render_proposal_block",
]

_log = get_logger("lint_autofix")

#: Every lint repair is an edit, so it stages in the edit proposal file.
PROPOSAL_FILE: Final = "intelligence/proposed/EDITS_PROPOSED.md"

#: Blast radius per rule, in the apply engine's vocabulary.
_RULE_RISK: Final[Mapping[str, str]] = {
    RULE_STALE_DATE: "low",
    RULE_MISSING_METADATA: "low",
    RULE_DUPLICATE_BLOCK: "high",
}

_PROPOSAL_ID_RE: Final = re.compile(r"^P-(\d{8})-(\d{3})$")
_UNSAFE_RE: Final = re.compile(r"[\r\n]+")


class LintAutofixError(RuntimeError):
    """Base class for every autofix failure. Callers catch this one."""


class UnknownFindingError(LintAutofixError):
    """The finding id is malformed, or no longer present in the corpus."""


class NotAutofixableError(LintAutofixError):
    """The finding is real but has no content-free deterministic repair."""


def _one_line(text: str) -> str:
    """Collapse to a single markdown-safe line (blocks header injection)."""
    return _UNSAFE_RE.sub(" ", text).replace("[", "(").replace("]", ")").strip()[:280]


def _next_proposal_id(existing: str, date_compact: str) -> str:
    """Allocate ``P-<date>-NNN`` one past the highest id already staged."""
    used = [int(m) for m in re.findall(rf"P-{re.escape(date_compact)}-(\d{{3}})", existing)]
    nxt = (max(used) + 1) if used else 1
    if nxt > 999:
        raise LintAutofixError(f"proposal id space exhausted for {date_compact} (999 per day)")
    return f"P-{date_compact}-{nxt:03d}"


def build_proposal(finding: Finding, proposal_id: str) -> dict[str, Any]:
    """Build the (unwritten) proposal dict for *finding*.

    Pure: no filesystem access, no clock. Exposed so a caller — or a
    golden-diff test — can inspect exactly what would be staged.
    """
    if not finding.autofixable or not finding.repair:
        raise NotAutofixableError(
            f"finding {finding.finding_id} ({finding.rule} on {finding.block_id}) has no deterministic repair; it needs a human decision"
        )
    if not _PROPOSAL_ID_RE.match(proposal_id):
        raise LintAutofixError(f"invalid proposal id: {proposal_id!r} (expected P-YYYYMMDD-NNN)")

    op = finding.repair_op()
    proposal: dict[str, Any] = {
        "ProposalId": proposal_id,
        "Type": "edit",
        "TargetBlock": finding.block_id,
        "Risk": _RULE_RISK.get(finding.rule, "high"),
        "Evidence": [_one_line(f"lint {finding.rule}: {finding.detail}")],
        "Rollback": "restore_snapshot",
        "Ops": [op],
        "FilesTouched": [finding.file],
        "Status": "staged",
        "Sources": [f"lint:{finding.rule}", f"finding:{finding.finding_id}"],
    }
    proposal["Fingerprint"] = compute_fingerprint(proposal)

    errors = validate_proposal(proposal)
    if errors:  # pragma: no cover — guards against a malformed repair op
        raise LintAutofixError(f"generated proposal failed validation: {errors}")
    return proposal


def render_proposal_block(proposal: Mapping[str, Any]) -> str:
    """Serialise *proposal* to the canonical proposal-block markdown."""
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


def lint_autofix(workspace: str, finding_id: str, *, now: datetime | None = None) -> str:
    """Stage a repair proposal for *finding_id* and return its proposal id.

    The corpus is not modified. The only file written is
    :data:`PROPOSAL_FILE`; applying the repair stays a separate,
    human-gated ``approve_apply`` call.

    Args:
        workspace: Workspace root.
        finding_id: A ``LF-xxxxxxxx`` id from :func:`mind_mem.lint.lint`.
        now: Clock override for the proposal-id date (tests/replay).

    Returns:
        The staged proposal id, e.g. ``"P-20260827-001"``.

    Raises:
        UnknownFindingError: malformed id, or no such finding.
        NotAutofixableError: the finding needs a human decision.
        LintAutofixError: workspace/proposal-file problems.
        FeatureDisabledError: the ``v4.lint`` flag is OFF.
    """
    if not workspace or not isinstance(workspace, str):
        raise LintAutofixError("workspace must be a non-empty path string")
    if not isinstance(finding_id, str) or not is_finding_id(finding_id):
        raise UnknownFindingError(f"malformed finding id: {finding_id!r} (expected LF-xxxxxxxx)")

    ws = os.path.abspath(workspace)
    if not os.path.isdir(ws):
        raise LintAutofixError(f"workspace does not exist: {ws}")
    require_lint_enabled(ws)

    findings = lint(ws)
    match = next((f for f in findings if f.finding_id == finding_id), None)
    if match is None:
        raise UnknownFindingError(f"no such finding: {finding_id} ({len(findings)} finding(s) currently reported)")

    proposal_path = os.path.join(ws, PROPOSAL_FILE)
    if not os.path.isfile(proposal_path):
        raise LintAutofixError(f"missing proposal file: {PROPOSAL_FILE} (run mind-mem-init on this workspace)")

    stamp = now or datetime.now()
    date_compact = stamp.strftime("%Y%m%d")

    with FileLock(proposal_path):
        with open(proposal_path, "r", encoding="utf-8") as handle:
            existing = handle.read()
        proposal_id = _next_proposal_id(existing, date_compact)
        proposal = build_proposal(match, proposal_id)
        fingerprint = str(proposal["Fingerprint"])
        if f"Fingerprint: {fingerprint}" in existing:
            raise LintAutofixError(f"an identical repair is already staged (fingerprint {fingerprint})")
        with open(proposal_path, "a", encoding="utf-8") as handle:
            handle.write(render_proposal_block(proposal))

    metrics.inc("lint_autofix_proposals")
    _log.info(
        "lint_autofix_staged",
        finding_id=finding_id,
        rule=match.rule,
        block_id=match.block_id,
        proposal_id=proposal_id,
    )
    return proposal_id
