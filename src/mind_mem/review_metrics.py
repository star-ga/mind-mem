# Copyright 2026 STARGA, Inc.
"""Approval throughput — the metric the review surface exists to move.

Two numbers get published, and they are computed from measured session
data rather than asserted:

* **median proposal age at approval** — how long a proposal waited
  before a human acted on it. This is the churn signal: a queue whose
  median age is days old is a queue the agent has already routed around.
* **proposals per minute** — how fast an operator can clear the queue.

Age is reported *only* for proposals that actually carry a ``Created``
timestamp, and the fraction that did is published alongside it as
``age_coverage``. A fabricated age would make the headline number a lie,
and a metric nobody can trust is worse than no metric.

Every value here is derived from the events handed in. Nothing in this
module reads a clock, so the same event list summarises identically on
any machine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

__all__ = ["ApprovalEvent", "SessionMetrics", "summarise"]

SECONDS_PER_MINUTE = 60.0


@dataclass(frozen=True)
class ApprovalEvent:
    """One operator decision and what came of it."""

    proposal_id: str
    action: str
    succeeded: bool
    age_seconds: float | None
    decided_at: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "action": self.action,
            "succeeded": self.succeeded,
            "age_seconds": self.age_seconds,
            "decided_at": self.decided_at,
        }


@dataclass(frozen=True)
class SessionMetrics:
    """The published summary of one review session."""

    decisions: int
    applied: int
    rejected: int
    failed: int
    elapsed_seconds: float
    proposals_per_minute: float | None
    applied_per_minute: float | None
    median_age_at_approval_seconds: float | None
    aged_sample: int
    age_coverage: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "decisions": self.decisions,
            "applied": self.applied,
            "rejected": self.rejected,
            "failed": self.failed,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "proposals_per_minute": _round(self.proposals_per_minute),
            "applied_per_minute": _round(self.applied_per_minute),
            "median_age_at_approval_seconds": _round(self.median_age_at_approval_seconds),
            "aged_sample": self.aged_sample,
            "age_coverage": round(self.age_coverage, 3),
        }


def summarise(events: Iterable[ApprovalEvent], *, elapsed_seconds: float) -> SessionMetrics:
    """Summarise one review session.

    Args:
        events: One entry per operator decision, in decision order.
        elapsed_seconds: Wall time the session took. ``0`` yields
            ``None`` rates rather than a division by zero.
    """
    ordered = tuple(events)
    approvals = tuple(e for e in ordered if e.action == "approve")
    applied = tuple(e for e in approvals if e.succeeded)
    rejected = tuple(e for e in ordered if e.action == "reject" and e.succeeded)
    failed = tuple(e for e in ordered if not e.succeeded)

    ages = tuple(e.age_seconds for e in applied if e.age_seconds is not None)
    coverage = (len(ages) / len(applied)) if applied else 0.0

    return SessionMetrics(
        decisions=len(ordered),
        applied=len(applied),
        rejected=len(rejected),
        failed=len(failed),
        elapsed_seconds=float(elapsed_seconds),
        proposals_per_minute=_rate(len(ordered), elapsed_seconds),
        applied_per_minute=_rate(len(applied), elapsed_seconds),
        median_age_at_approval_seconds=median(ages),
        aged_sample=len(ages),
        age_coverage=coverage,
    )


def median(values: Sequence[float]) -> float | None:
    """Median of *values*, or ``None`` for an empty sample.

    ``None`` and not ``0.0``: an empty sample means "unknown", and
    reporting an unknown median as zero would read as instant approval.
    """
    if not values:
        return None
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[midpoint])
    return (float(ordered[midpoint - 1]) + float(ordered[midpoint])) / 2.0


def _rate(count: int, elapsed_seconds: float) -> float | None:
    """Per-minute rate, or ``None`` when no time elapsed."""
    if elapsed_seconds <= 0:
        return None if count else 0.0
    return count * SECONDS_PER_MINUTE / elapsed_seconds


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 3)
