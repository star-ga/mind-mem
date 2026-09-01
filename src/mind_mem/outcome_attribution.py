"""Outcome attribution — did the recalled memory actually *help*?

Every other quality signal in mind-mem measures **retrieval quality**
(was the block a good match for the query?). This module measures
**retrieval utility**: after an agent — or a CI job — acted on a set of
recalled blocks, did the downstream work succeed or fail?

The consumer reports back once the real-world verdict is known::

    from mind_mem.outcome_attribution import report_outcome

    report_outcome(
        workspace,
        block_ids=["D-20260301-001"],
        outcome="failure",
        task_id="fix-lint-gate",
        evidence="pytest tests/test_lint.py :: 3 failed",
    )

Storage is the **existing calibration store** — the same
``.mind-mem-index/recall.db`` and the same
:class:`mind_mem.calibration.CalibrationManager` that already owns the
``calibration_feedback`` table. Outcomes land in a sibling
``recall_outcome`` table and are *projected* into
``calibration_feedback`` (success -> ``accepted``, failure ->
``rejected``, neutral -> ``ignored``) so the pre-existing calibration
weight loop picks utility up for free. No parallel subsystem.

Determinism
-----------
The scored path is pure arithmetic over **unwindowed stored counts**:
no clock, no rolling window, no randomness, no learned parameters.
:func:`outcome_factor` and :func:`is_corroborated` are total functions of
two integers. The only clock value in the module (``recorded_at``) is
provenance metadata that never reaches a score, and it is injectable.

Idempotency
-----------
An outcome's identity is the SHA-256 of its canonical payload (sorted
block ids + verdict + provenance), so replaying the same report is a
no-op: the insert conflicts and the stored row — including its original
``recorded_at`` — is returned unchanged.

Bounded influence
-----------------
Two limits stop one reporter from moving a block without bound:

* the counts the scored path reads are **clamped** at twice
  :data:`MIN_OUTCOME_EVIDENCE` — a thousand reports of one verdict read as
  six, by contract and not by accident; and
* the opt-in ``calibration_feedback`` projection is keyed on the reporting
  ``actor_id``, so one reporter is worth at most one vote per block and
  verdict however many distinct reports it files.

Neither is a rate limit, and the gate is deliberately sensitive: three
*distinct* failure reports take a block's factor to ``0.6``, under the
validity gate's ``0.8`` threshold, which halves its recall score. Utility
evidence is meant to bite early — operators should know it bites that early.

Governed write
--------------
Recording an outcome **never** mutates block content. It appends to the
sidecar index database only. When an outcome is strong enough to warrant
changing the corpus, :func:`outcome_proposal` builds a payload for
``propose_update`` so the change goes through the HITL gate like every
other write.

Flag
----
The validity-gate consumer is opt-in and default-OFF::

    {"recall": {"validity_gate": {"enabled": true,
                                  "outcome_attribution": {"enabled": true}}}}

With the sub-flag absent the gate's four-criteria output is byte-identical
to the pre-outcome pipeline.

Copyright (c) STARGA, Inc.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from .observability import get_logger
from .trajectory import TRAJECTORY_FLAG

_log = get_logger("outcome_attribution")

__all__ = [
    "MIN_OUTCOME_EVIDENCE",
    "OUTCOME_FAILURE",
    "OUTCOME_FLOOR",
    "OUTCOME_NEUTRAL",
    "OUTCOME_SUCCESS",
    "VALID_OUTCOMES",
    "OutcomeSignal",
    "bounded_field",
    "canonical_outcome_id",
    "is_corroborated",
    "load_outcome_signals",
    "normalize_block_ids",
    "outcome_factor",
    "outcome_proposal",
    "outcome_stats",
    "report_outcome",
    "validate_outcome",
]

# --- vocabulary -------------------------------------------------------------

#: The block helped: the work that used it succeeded.
OUTCOME_SUCCESS = "success"
#: The block is implicated in a failure: the work that used it failed.
OUTCOME_FAILURE = "failure"
#: The block was recalled but the verdict is not attributable either way.
OUTCOME_NEUTRAL = "neutral"

VALID_OUTCOMES: frozenset[str] = frozenset({OUTCOME_SUCCESS, OUTCOME_FAILURE, OUTCOME_NEUTRAL})

# --- scoring constants (fixed, never learned) -------------------------------

#: Attributed outcomes required before utility evidence is allowed to move
#: a score at all. Mirrors ``calibration.MIN_FEEDBACK_THRESHOLD``.
MIN_OUTCOME_EVIDENCE = 3

#: Lower bound of the utility factor. A block can be halved by a failure
#: history, never zeroed — the gate demotes, it does not delete.
OUTCOME_FLOOR = 0.5

# --- boundary limits --------------------------------------------------------

#: Max blocks attributable to one outcome report.
MAX_OUTCOME_BLOCKS = 256
#: Max length of any single provenance/evidence field.
MAX_OUTCOME_FIELD_LEN = 512


# ---------------------------------------------------------------------------
# Validation (boundary)
# ---------------------------------------------------------------------------


def validate_outcome(outcome: str) -> str:
    """Return the canonical verdict string, or raise :class:`ValueError`."""
    if not isinstance(outcome, str):
        raise ValueError(f"outcome must be a string, got {type(outcome).__name__}")
    normalized = outcome.strip().lower()
    if normalized not in VALID_OUTCOMES:
        raise ValueError(f"outcome must be one of {sorted(VALID_OUTCOMES)}, got {outcome!r}")
    return normalized


def normalize_block_ids(block_ids: Iterable[str] | None) -> tuple[str, ...]:
    """Return deduplicated, sorted, non-empty block ids.

    Sorting is what makes an outcome's identity independent of the order
    the caller happened to list its blocks in — two reports of the same
    verdict over the same blocks collapse to one row.
    """
    if block_ids is None:
        raise ValueError("block_ids is required")
    if isinstance(block_ids, (str, bytes)):
        raise ValueError("block_ids must be a sequence of ids, not a single string")

    cleaned = {str(bid).strip() for bid in block_ids}
    cleaned.discard("")
    if not cleaned:
        raise ValueError("block_ids must contain at least one non-empty id")
    if len(cleaned) > MAX_OUTCOME_BLOCKS:
        raise ValueError(f"block_ids exceeds {MAX_OUTCOME_BLOCKS} entries (got {len(cleaned)})")
    return tuple(sorted(cleaned))


def bounded_field(name: str, value: str | None) -> str:
    """Return a trimmed provenance field, or raise when it is oversized."""
    if value is None:
        return ""
    text = str(value).strip()
    if len(text) > MAX_OUTCOME_FIELD_LEN:
        raise ValueError(f"{name} exceeds {MAX_OUTCOME_FIELD_LEN} chars (got {len(text)})")
    return text


# ---------------------------------------------------------------------------
# Identity (deterministic, provenance-bearing)
# ---------------------------------------------------------------------------

#: Field separator inside a canonical payload; record separator between ids.
_FIELD_SEP = "\x1f"
_RECORD_SEP = "\x1e"


def canonical_outcome_id(
    block_ids: Sequence[str],
    outcome: str,
    *,
    task_id: str = "",
    actor_id: str = "",
    session_id: str = "",
    tool_id: str = "",
    evidence: str = "",
) -> tuple[str, str]:
    """Return ``(outcome_id, payload_hash)`` for one outcome report.

    ``payload_hash`` is the full SHA-256 over the canonical payload — the
    tamper-evident record of exactly what was reported. ``outcome_id`` is
    its short, human-pasteable prefix and the idempotency key.
    """
    payload = _FIELD_SEP.join(
        (
            outcome,
            task_id,
            actor_id,
            session_id,
            tool_id,
            evidence,
            _RECORD_SEP.join(block_ids),
        )
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"out-{digest[:24]}", digest


# ---------------------------------------------------------------------------
# Pure scoring math
# ---------------------------------------------------------------------------


def outcome_factor(success: int, failure: int) -> float:
    """Utility factor in ``[OUTCOME_FLOOR, 1.0]`` from attributed counts.

    Governing rule, inherited from the validity gate: *absence of a signal
    is neutral; only affirmative evidence of invalidity debits.* So:

    * no failures at all -> ``1.0`` (successes never debit, and a block
      with a clean record is never punished for being unused);
    * fewer than :data:`MIN_OUTCOME_EVIDENCE` attributed outcomes -> ``1.0``
      (one bad day is not a pattern);
    * otherwise a Laplace-smoothed success ratio linearly mapped onto
      ``[OUTCOME_FLOOR, 1.0]``.

    Pure: same two integers always yield the same float.
    """
    success = max(0, int(success))
    failure = max(0, int(failure))
    if failure == 0:
        return 1.0
    total = success + failure
    if total < MIN_OUTCOME_EVIDENCE:
        return 1.0
    ratio = (success + 1) / (total + 2)
    factor = OUTCOME_FLOOR + (1.0 - OUTCOME_FLOOR) * ratio
    return round(max(OUTCOME_FLOOR, min(1.0, factor)), 4)


def is_corroborated(success: int, failure: int) -> bool:
    """True when successful outcomes corroborate the block.

    Requires a *pattern* of successes (>= :data:`MIN_OUTCOME_EVIDENCE`)
    that outweighs the failures. A corroborated block is treated by the
    validity gate as confirmed by an independent, real-world source.
    """
    success = max(0, int(success))
    failure = max(0, int(failure))
    return success >= MIN_OUTCOME_EVIDENCE and success > failure


@dataclass(frozen=True)
class OutcomeSignal:
    """Immutable per-block utility evidence (unwindowed stored counts)."""

    block_id: str
    success: int = 0
    failure: int = 0
    neutral: int = 0

    @property
    def total(self) -> int:
        return self.success + self.failure + self.neutral

    @property
    def factor(self) -> float:
        """Deterministic utility factor — see :func:`outcome_factor`."""
        return outcome_factor(self.success, self.failure)

    @property
    def corroborated(self) -> bool:
        """Whether successes corroborate this block — see :func:`is_corroborated`."""
        return is_corroborated(self.success, self.failure)

    def as_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "success": self.success,
            "failure": self.failure,
            "neutral": self.neutral,
            "total": self.total,
            "factor": self.factor,
            "corroborated": self.corroborated,
        }


# ---------------------------------------------------------------------------
# Public API (thin wrappers over the calibration store)
# ---------------------------------------------------------------------------


def report_outcome(
    workspace: str,
    block_ids: Iterable[str],
    outcome: str,
    *,
    query_id: str = "",
    task_id: str = "",
    actor_id: str = "",
    session_id: str = "",
    tool_id: str = "",
    evidence: str = "",
    recorded_at: str | None = None,
    project_to_calibration: bool = False,
) -> dict[str, Any]:
    """Record whether recalled blocks actually helped. Idempotent.

    Args:
        workspace: Workspace root (the calibration DB lives under it).
        block_ids: Blocks that were recalled and acted upon.
        outcome: ``success`` | ``failure`` | ``neutral``.
        query_id: Originating recall ``query_id``, when known.
        task_id: What the consumer was doing (build id, ticket, test name).
        actor_id: Who/what reported the outcome.
        session_id: Session provenance.
        tool_id: Reporting tool provenance.
        evidence: Free-text proof (e.g. a test summary line).
        recorded_at: Injectable ISO-8601 timestamp; defaults to now. It is
            provenance only and never reaches a score.
        project_to_calibration: Also write the verdicts into the existing
            ``calibration_feedback`` loop (success -> accepted, failure ->
            rejected, neutral -> ignored). Default-off: the recall pipeline
            applies calibration weights unconditionally, so projecting by
            default would move scores for callers who never opted in. The
            projected row is keyed on ``actor_id``, so one reporter casts at
            most one vote per block and verdict; reports with no ``actor_id``
            share a single anonymous vote.

    Returns:
        Dict with ``outcome_id``, ``payload_hash``, ``block_ids``,
        ``recorded`` / ``duplicate`` counts and ``idempotent``.

    Raises:
        ValueError: on an invalid verdict, empty/oversized block list, or
            an oversized provenance field (validated at the boundary).
    """
    from .calibration import CalibrationManager

    result = CalibrationManager(workspace).record_outcome(
        block_ids=block_ids,
        outcome=outcome,
        query_id=query_id,
        task_id=task_id,
        actor_id=actor_id,
        session_id=session_id,
        tool_id=tool_id,
        evidence=evidence,
        recorded_at=recorded_at,
        project_to_calibration=project_to_calibration,
    )
    trajectory_path = _capture_trajectory(workspace, result)
    return result if trajectory_path is None else {**result, "trajectory": trajectory_path}


def _capture_trajectory(workspace: str, result: dict[str, Any]) -> str | None:
    """Mirror a recorded outcome into the trajectory sidecar, if enabled.

    Gated on the v4 ``trajectory`` flag, default-OFF. The probe is
    ``is_enabled_quiet``: ``is_enabled`` warns ``v4_config_unreadable`` on a
    malformed config, and a probe that logs on an OFF path makes the
    flag-off build observably different from the build that never had the
    feature. With the flag off this function reads the config file and
    returns — no trajectory directory, no log line, and the dict the caller
    gets back is the one ``record_outcome`` returned, unchanged.

    Runs only AFTER the outcome is durably recorded, so a report that
    ``record_outcome`` rejects leaves nothing behind.
    """
    from .v4.feature_flags import is_enabled_quiet

    if not is_enabled_quiet(TRAJECTORY_FLAG):
        return None
    from .trajectory import capture_from_outcome

    return capture_from_outcome(workspace, result)


def load_outcome_signals(workspace: str, block_ids: Sequence[str]) -> dict[str, OutcomeSignal]:
    """Batch-load per-block utility evidence. Never raises.

    A missing table, an unreadable database or an empty id list all yield
    ``{}`` — which the validity gate reads as "no signal", i.e. neutral.
    """
    if not block_ids:
        return {}
    try:
        from .calibration import CalibrationManager

        return CalibrationManager(workspace).get_outcome_signals(list(block_ids))
    except Exception as exc:  # pragma: no cover — degradation path
        _log.warning("outcome_signals_unavailable", error=str(exc))
        return {}


def outcome_stats(workspace: str, top_n: int = 20) -> dict[str, Any]:
    """Utility health report: totals, most-corroborated, most-implicated."""
    from .calibration import CalibrationManager

    return CalibrationManager(workspace).get_outcome_stats(top_n=top_n)


# ---------------------------------------------------------------------------
# Governed write helper
# ---------------------------------------------------------------------------


def outcome_proposal(signal: OutcomeSignal, *, task_id: str = "") -> dict[str, str]:
    """Build a ``propose_update`` payload for a block with a failure record.

    Outcome recording itself never touches block content. When the
    evidence says a block should change, the change must travel the
    normal governed route: this returns the argument dict for
    ``propose_update`` (which writes a SIGNAL for human review), and
    nothing here writes anything.
    """
    # deferred: no auto-proposal trigger — a caller decides when evidence is
    # strong enough and passes this to propose_update. Upgrade path: a
    # threshold-driven sweep that stages proposals for review, never applies.
    verdict = "corroborated by" if signal.corroborated else "implicated in"
    scope = f" while working on {task_id}" if task_id else ""
    statement = (
        f"Block {signal.block_id} is {verdict} downstream outcomes: "
        f"{signal.success} success / {signal.failure} failure / {signal.neutral} neutral"
        f"{scope}."
    )
    rationale = (
        f"Outcome attribution recorded {signal.total} attributed outcomes for "
        f"{signal.block_id}; deterministic utility factor is {signal.factor}. "
        "Review whether the block's content or status should change."
    )
    return {
        "block_type": "decision",
        "statement": statement[:2000],
        "rationale": rationale[:2000],
        "tags": "outcome-attribution,memory-utility",
    }
