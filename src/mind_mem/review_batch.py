# Copyright 2026 STARGA, Inc.
"""Batch approve/reject over the existing governed apply path.

``mm review`` is a **front end**. Every approval in this module goes out
through :func:`mind_mem.mcp.tools.governance.approve_apply` and every
rejection through ``reject_proposal`` — the same two entry points the MCP
server exposes, with the same contradiction check, the same snapshot,
the same receipt and the same rate limit. There is no second write path
here, and adding one would defeat the purpose of the surface.

**No proposal is ever approved without an operator action.** A
:class:`ReviewDecision` cannot be constructed without an ``origin`` drawn
from :data:`OPERATOR_ORIGINS`, and :func:`run_batch` acts on the
decisions it is handed and on nothing else — a proposal nobody decided on
is left staged. There is deliberately no risk-based shortcut, no
"approve the safe ones", no unattended mode. Batch review makes
approving *fast*; a human still approves. ``tests/test_review_no_auto
approve.py`` fails the build if that ever stops being true.

Atomicity is **per proposal**, not per batch. If proposal 7 of 30 fails,
proposals 1-6 stay applied, 8-30 still run, and 7 is reported. A
half-applied batch that silently rolled back six good applies is worse
than a slow one, and rolling back an already-receipted apply behind the
operator's back would break the audit chain's meaning.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

from .review_metrics import ApprovalEvent, SessionMetrics, summarise
from .review_queue import ReviewItem, load_queue

__all__ = [
    "ACTIONS",
    "OPERATOR_ORIGINS",
    "BatchOutcome",
    "BatchReport",
    "ReviewBatchError",
    "ReviewDecision",
    "governed_approve",
    "governed_reject",
    "run_batch",
]

#: The closed decision vocabulary. There is no third action.
ACTIONS: frozenset[str] = frozenset({"approve", "reject"})

#: Where a decision may come from. Every member is a human doing
#: something: a keystroke in the review session, an explicit ``--approve``
#: flag, or a line the operator piped in. A scheduler, a policy engine or
#: a confidence score is not on this list and must never be added — that
#: is the auto-approval path the surface exists without.
OPERATOR_ORIGINS: frozenset[str] = frozenset({"keypress", "cli-flag", "stdin"})

#: Minimum length of a rejection rationale, matching ``reject_proposal``.
MIN_REASON_CHARS = 8

_PROPOSAL_ID_RE = re.compile(r"^P-\d{8}-\d{3}$")


class ReviewBatchError(ValueError):
    """A malformed decision, or a decision set that cannot be executed."""


@dataclass(frozen=True)
class ReviewDecision:
    """One operator decision about one proposal.

    Validated at construction so an ill-formed decision can never reach
    the governed path. ``origin`` records *which human action* produced
    it and is checked against :data:`OPERATOR_ORIGINS`.
    """

    proposal_id: str
    action: str
    origin: str
    reason: str = ""

    def __post_init__(self) -> None:
        if not _PROPOSAL_ID_RE.match(self.proposal_id or ""):
            raise ReviewBatchError(f"malformed proposal id: {self.proposal_id!r} (expected P-YYYYMMDD-NNN)")
        if self.action not in ACTIONS:
            raise ReviewBatchError(f"unknown action {self.action!r}; expected one of {sorted(ACTIONS)}")
        if self.origin not in OPERATOR_ORIGINS:
            raise ReviewBatchError(
                f"decision origin {self.origin!r} is not an operator action; "
                f"expected one of {sorted(OPERATOR_ORIGINS)}. Approval requires a human."
            )
        if self.action == "reject" and len(self.reason.strip()) < MIN_REASON_CHARS:
            raise ReviewBatchError(
                f"rejecting {self.proposal_id} needs a written reason of at least "
                f"{MIN_REASON_CHARS} characters; rejections without one leave no audit trail"
            )


@dataclass(frozen=True)
class BatchOutcome:
    """What happened to one proposal in the batch."""

    proposal_id: str
    action: str
    succeeded: bool
    message: str
    age_seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "action": self.action,
            "succeeded": self.succeeded,
            "message": self.message,
            "age_seconds": self.age_seconds,
        }


@dataclass(frozen=True)
class BatchReport:
    """Per-proposal outcomes plus the published throughput metric."""

    outcomes: tuple[BatchOutcome, ...] = ()
    metrics: SessionMetrics | None = None
    #: Wall time the governed applies took.
    elapsed_seconds: float = 0.0
    #: Wall time the whole operator session took — deciding *and*
    #: applying. This is the span the published rate covers, because
    #: "proposals per minute" over the applies alone measures the apply
    #: engine and calls it approval throughput.
    session_seconds: float = 0.0

    @property
    def applied(self) -> tuple[BatchOutcome, ...]:
        return tuple(o for o in self.outcomes if o.action == "approve" and o.succeeded)

    @property
    def rejected(self) -> tuple[BatchOutcome, ...]:
        return tuple(o for o in self.outcomes if o.action == "reject" and o.succeeded)

    @property
    def failed(self) -> tuple[BatchOutcome, ...]:
        return tuple(o for o in self.outcomes if not o.succeeded)

    def to_dict(self) -> dict[str, Any]:
        return {
            "applied": [o.to_dict() for o in self.applied],
            "rejected": [o.to_dict() for o in self.rejected],
            "failed": [o.to_dict() for o in self.failed],
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "apply_seconds": round(self.elapsed_seconds, 3),
            "session_seconds": round(self.session_seconds, 3),
        }


ApproveHook = Callable[..., tuple[bool, str]]
RejectHook = Callable[..., tuple[bool, str]]


def governed_approve(workspace: str, proposal_id: str, *, dry_run: bool) -> tuple[bool, str]:
    """Approve one proposal through the governed MCP entry point.

    The single choke point: this is the only function in the review
    surface permitted to reach ``approve_apply``, which keeps the
    contradiction check, the snapshot, the receipt, the recall-cache
    invalidation and the apply rate limit on the path.
    """
    from .mcp.infra.workspace import use_workspace
    from .mcp.tools import governance

    with use_workspace(workspace):
        payload = governance.approve_apply(proposal_id, dry_run=dry_run)
    return _outcome_of(payload)


def governed_reject(workspace: str, proposal_id: str, reason: str) -> tuple[bool, str]:
    """Reject one proposal through the governed MCP entry point."""
    from .mcp.infra.workspace import use_workspace
    from .mcp.tools import governance

    with use_workspace(workspace):
        payload = governance.reject_proposal(proposal_id, reason)
    return _outcome_of(payload)


def run_batch(
    workspace: str,
    decisions: Sequence[ReviewDecision],
    *,
    approve_hook: ApproveHook | None = None,
    reject_hook: RejectHook | None = None,
    now_iso: str = "",
    session_started: float = 0.0,
) -> BatchReport:
    """Execute *decisions*, one governed call each, in the order given.

    Args:
        workspace: Workspace root.
        decisions: Explicit operator decisions. Proposals absent from
            this sequence are left untouched — there is no default
            action and no implicit approval.
        approve_hook: Failure-injection seam for tests; defaults to
            :func:`governed_approve`. A hook can lie about a result but
            cannot produce a write — the corpus only changes on the
            governed path.
        reject_hook: The same seam for rejections.
        now_iso: Clock for the age metric. Empty means ages are unknown
            and the published median is ``None`` rather than invented.
        session_started: ``time.perf_counter()`` reading from when the
            operator's session began — before they were shown the queue,
            not when the applies started. The span is closed here, at
            the end of the batch, and becomes the denominator of the
            published rate. That is the honest denominator: an operator
            spends most of a review reading, and a rate measured over
            the applies alone flatters the surface publishing it.
            Omitted (the default) the rate covers the applies only, so
            existing callers are unchanged.

    Raises:
        ReviewBatchError: two decisions name the same proposal. Refused
            before anything is applied, because "approve then reject"
            has no defensible resolution.
    """
    ordered = tuple(decisions)
    _reject_duplicates(ordered)
    approve = approve_hook or governed_approve
    reject = reject_hook or governed_reject
    ages = _ages(workspace, ordered, now_iso=now_iso)

    started = time.perf_counter()
    outcomes = tuple(_execute_one(workspace, d, approve, reject, ages.get(d.proposal_id)) for d in ordered)
    elapsed = time.perf_counter() - started

    events: list[ApprovalEvent] = [
        ApprovalEvent(
            proposal_id=o.proposal_id,
            action=o.action,
            succeeded=o.succeeded,
            age_seconds=o.age_seconds,
            decided_at=float(index),
        )
        for index, o in enumerate(outcomes)
    ]
    span = max(elapsed, time.perf_counter() - session_started) if session_started else elapsed
    return BatchReport(
        outcomes=outcomes,
        metrics=summarise(events, elapsed_seconds=span),
        elapsed_seconds=elapsed,
        session_seconds=span,
    )


def _execute_one(
    workspace: str,
    decision: ReviewDecision,
    approve: ApproveHook,
    reject: RejectHook,
    age_seconds: float | None,
) -> BatchOutcome:
    """One proposal, one governed call, failures contained.

    The containment is the point: an exception here becomes this
    proposal's reported failure and the batch continues. Nothing that
    already succeeded is undone.
    """
    try:
        if decision.action == "approve":
            ok, message = approve(workspace, decision.proposal_id, dry_run=False)
        else:
            ok, message = reject(workspace, decision.proposal_id, decision.reason)
    except Exception as exc:  # noqa: BLE001 — one proposal's failure must not end the batch
        return BatchOutcome(decision.proposal_id, decision.action, False, f"{type(exc).__name__}: {exc}", age_seconds)
    return BatchOutcome(decision.proposal_id, decision.action, bool(ok), str(message), age_seconds)


def _ages(workspace: str, decisions: Iterable[ReviewDecision], *, now_iso: str) -> dict[str, float | None]:
    """Age of each decided proposal, snapshotted before anything is applied."""
    if not now_iso:
        return {}
    wanted = {d.proposal_id for d in decisions}
    try:
        items: tuple[ReviewItem, ...] = load_queue(workspace)
    except Exception:  # noqa: BLE001 — the metric must never break the batch
        return {}
    return {item.proposal_id: item.age_seconds(now_iso=now_iso) for item in items if item.proposal_id in wanted}


def _reject_duplicates(decisions: Sequence[ReviewDecision]) -> None:
    seen: set[str] = set()
    for decision in decisions:
        if decision.proposal_id in seen:
            raise ReviewBatchError(
                f"{decision.proposal_id} appears twice in one batch; resolve the conflicting decisions before running it"
            )
        seen.add(decision.proposal_id)


def _outcome_of(payload: str) -> tuple[bool, str]:
    """Read ``(success, message)`` out of a governance tool's JSON reply."""
    try:
        parsed = json.loads(payload)
    except (TypeError, ValueError):
        return False, f"unparseable governance response: {str(payload)[:200]}"
    if not isinstance(parsed, dict):
        return False, f"unexpected governance response: {str(payload)[:200]}"
    if "error" in parsed:
        return False, str(parsed["error"])
    message = str(parsed.get("message") or parsed.get("status") or "")
    if "success" in parsed:
        return bool(parsed["success"]), message
    return parsed.get("status") in {"rejected", "applied"}, message
