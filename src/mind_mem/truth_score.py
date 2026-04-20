"""Probabilistic truth score for memory blocks (v3.3.0).

A memory system that keeps conflicting blocks around has to give the
caller a principled way to distinguish "we used to think X" from
"we currently believe X". ``truth_score`` turns the governance
signal (contradiction votes, status transitions, access patterns)
into a single [0, 1] float the recall pipeline can surface alongside
BM25 rank.

The update is a Bayesian posterior over two latent states for each
block::

    p(TRUE | evidence) ∝ p(evidence | TRUE) · p(TRUE)

where evidence aggregates:

* **Status transitions** — ``active`` adds prior mass, ``superseded``
  drains it, ``verified`` caps near 1.0, ``draft`` / ``deprecated``
  drain it.
* **Contradiction votes** — each ``contradicts`` edge from another
  block subtracts weighted mass (weight = other block's own
  ``truth_score``, so high-confidence contradictions bite harder).
* **Age** — temporal decay on a 180-day half-life; older blocks
  without re-verification lose prior mass.
* **Access freshness** — blocks the recall layer surfaces without
  user correction gain a small mass (implicit "this was useful").

Pure function — no I/O. The caller feeds in whatever contradiction
graph it already has (``governance_gate`` already maintains one).

Exposed on recall results as ``block.truth_score`` when
``retrieval.truth_score.enabled`` is true in ``mind-mem.json``.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

from .observability import get_logger

_log = get_logger("truth_score")


# Status → prior mass. Applied at the start of the Bayesian update.
_STATUS_PRIOR: dict[str, float] = {
    "verified": 0.92,
    "active": 0.75,
    "draft": 0.55,
    "deferred": 0.50,
    "superseded": 0.20,
    "deprecated": 0.15,
    "rejected": 0.05,
}

# Default prior when Status is missing / unrecognised.
_UNKNOWN_STATUS_PRIOR = 0.60

# Half-life (in days) for the age decay term. Matches
# ``retrieval.temporal_half_life_days`` default so the two signals
# agree by default.
_DEFAULT_AGE_HALF_LIFE_DAYS = 180.0

# How much each contradiction vote bites into the score. The actual
# subtracted mass is ``votes[i].weight * _CONTRADICTION_WEIGHT``.
_CONTRADICTION_WEIGHT = 0.18


def _days_since(date_str: str | None) -> float | None:
    if not date_str:
        return None
    try:
        dt = datetime.strptime(str(date_str)[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None
    delta = datetime.now(timezone.utc) - dt
    return max(0.0, delta.days)


def _prior_for_status(status: str | None) -> float:
    if not status:
        return _UNKNOWN_STATUS_PRIOR
    return _STATUS_PRIOR.get(str(status).lower(), _UNKNOWN_STATUS_PRIOR)


def _age_factor(days: float, half_life_days: float) -> float:
    """Exponential decay factor ∈ (0, 1]. Newer → closer to 1."""
    if half_life_days <= 0:
        return 1.0
    return float(0.5 ** (days / half_life_days))


def truth_score(
    block: dict,
    *,
    contradiction_votes: list[dict] | None = None,
    age_half_life_days: float = _DEFAULT_AGE_HALF_LIFE_DAYS,
) -> float:
    """Compute a [0, 1] probabilistic truth estimate for a block.

    Args:
        block: Block dict. Reads ``Status`` / ``Created`` / ``Date`` /
            ``_access_count``. All fields optional.
        contradiction_votes: List of ``{"weight": float}`` dicts, one
            per contradicting block. ``weight`` should be the other
            block's own truth score (so confident contradictions bite
            harder). Empty / None skips the contradiction term.
        age_half_life_days: Half-life for the age decay term. Shorter
            values push older blocks to lower scores faster.

    Returns:
        A value in [0.01, 0.99] — clamped so downstream callers can
        safely multiply or take ``log`` without hitting edge cases.
    """
    prior = _prior_for_status(block.get("Status"))

    # Age term — decay from the block's own date field.
    created = block.get("Created") or block.get("Date")
    days = _days_since(created)
    age_mult = _age_factor(days, age_half_life_days) if days is not None else 1.0

    # Contradiction term — each vote subtracts proportional mass.
    contradiction_mass = 0.0
    if contradiction_votes:
        for vote in contradiction_votes:
            if not isinstance(vote, dict):
                continue
            w = vote.get("weight", 0.5)
            if isinstance(w, (int, float)) and w > 0:
                contradiction_mass += float(w) * _CONTRADICTION_WEIGHT
        contradiction_mass = min(contradiction_mass, prior * 0.9)

    # Access freshness — gentle bump so blocks that get re-used without
    # correction gain confidence. Log-scaled so 10 accesses ≈ +0.07.
    access_count = block.get("_access_count", 0)
    if isinstance(access_count, int) and access_count > 0:
        access_bonus = min(0.10, math.log1p(access_count) / 30.0)
    else:
        access_bonus = 0.0

    score = prior * age_mult - contradiction_mass + access_bonus

    # Clamp to (0.01, 0.99) for numerical friendliness.
    if score < 0.01:
        score = 0.01
    elif score > 0.99:
        score = 0.99
    return round(float(score), 4)


def annotate_results(
    results: list[dict],
    *,
    contradiction_graph: dict[str, list[dict]] | None = None,
    age_half_life_days: float = _DEFAULT_AGE_HALF_LIFE_DAYS,
) -> list[dict]:
    """Attach ``truth_score`` to every result in-place (and return).

    ``contradiction_graph`` maps ``block_id`` → list of vote dicts for
    blocks that contradict it. Callers with an existing governance
    graph pass theirs; the scorer doesn't build its own.
    """
    if not results:
        return results
    votes_for: dict[str, list[dict]] = contradiction_graph or {}
    annotated = 0
    for r in results:
        bid = str(r.get("_id") or "")
        r["truth_score"] = truth_score(
            r,
            contradiction_votes=votes_for.get(bid, []),
            age_half_life_days=age_half_life_days,
        )
        annotated += 1
    _log.info("truth_score_annotated", count=annotated)
    return results


def is_truth_score_enabled(config: dict[str, Any] | None) -> bool:
    if not config or not isinstance(config, dict):
        return False
    retrieval = config.get("retrieval", {})
    if not isinstance(retrieval, dict):
        return False
    ts = retrieval.get("truth_score", {})
    if not isinstance(ts, dict):
        return False
    return bool(ts.get("enabled", False))


__all__ = [
    "truth_score",
    "annotate_results",
    "is_truth_score_enabled",
]
