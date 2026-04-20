"""Quorum-based consensus voting on contradictions (v3.3.0).

Extends :mod:`conflict_resolver`'s pairwise analysis with multi-agent
quorum: N agents each independently score a contradiction, and the
resolver picks the winner when the weighted majority passes a
threshold. Avoids human review in the common case where multiple
agents converge on the same resolution.

Each agent has a ``trust_weight`` (from their namespace config) that
scales their vote. A verified-tier agent with trust_weight=2.0
counts twice as much as a default trust_weight=1.0 agent.

Pure function: caller feeds in the vote records it already has, this
returns the resolved winner + confidence. No I/O, no governance
state mutation — the caller persists the resolution.

Config::

    {
      "governance": {
        "consensus": {
          "enabled": false,
          "quorum_threshold": 0.66,
          "min_votes": 2
        }
      },
      "namespaces": {
        "agent-alice": {"trust_weight": 1.5},
        "agent-bot":   {"trust_weight": 0.8}
      }
    }
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .observability import get_logger

_log = get_logger("consensus_vote")


# Outcomes callers use to decide what to write into the audit chain.
@dataclass
class ConsensusDecision:
    """One-shot result of a quorum vote.

    ``winner`` is ``None`` when the quorum isn't met — the caller
    should fall back to human review. ``margin`` is the weighted-vote
    share of the winner (0.0..1.0); ``confidence`` is margin scaled
    against the minimum-quorum threshold so 1.0 means "exactly hit
    the threshold" and higher means "clear win".
    """

    winner: str | None
    margin: float
    confidence: float
    reason: str  # "quorum" | "below_threshold" | "insufficient_votes"
    vote_counts: dict[str, float]


@dataclass
class Vote:
    """One agent's vote. ``choice`` is the block ID / key being chosen."""

    agent_id: str
    choice: str
    trust_weight: float = 1.0
    rationale: str | None = None


def _resolve_trust_weight(namespace_config: dict[str, Any] | None, agent_id: str) -> float:
    """Trust weight for an agent, from ``namespaces.<id>.trust_weight``.

    A namespace trust_weight of ``0`` explicitly excludes the agent —
    returns 0 instead of falling back to the 1.0 default. Negative
    values still fall back (they're configuration errors).
    """
    if not namespace_config or not isinstance(namespace_config, dict):
        return 1.0
    ns = namespace_config.get(agent_id)
    if isinstance(ns, dict) and "trust_weight" in ns:
        w = ns["trust_weight"]
        if isinstance(w, (int, float)) and w >= 0:
            return float(w)
    return 1.0


def tally_votes(
    votes: list[Vote],
    *,
    namespace_config: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Return ``{choice: weighted_total}``.

    Votes without a ``trust_weight`` pull their weight from
    ``namespace_config[agent_id].trust_weight``, defaulting to 1.0.
    Agent-level duplicate votes (same agent voting twice for the same
    choice) are counted once — the highest trust_weight wins, so
    a later re-vote can only strengthen a prior conviction.
    """
    # Collapse duplicate (agent, choice) pairs to the max trust_weight.
    latest: dict[tuple[str, str], float] = {}
    for v in votes:
        key = (v.agent_id, v.choice)
        weight = v.trust_weight if v.trust_weight > 0 else _resolve_trust_weight(namespace_config, v.agent_id)
        if weight <= 0:
            continue
        if key not in latest or weight > latest[key]:
            latest[key] = weight

    tally: dict[str, float] = {}
    for (_, choice), weight in latest.items():
        tally[choice] = tally.get(choice, 0.0) + weight
    return tally


def reach_consensus(
    votes: list[Vote],
    *,
    quorum_threshold: float = 0.66,
    min_votes: int = 2,
    namespace_config: dict[str, Any] | None = None,
) -> ConsensusDecision:
    """Decide the winner of a contradiction by weighted quorum.

    Args:
        votes: All agents' votes for this contradiction.
        quorum_threshold: Winner must have ≥ this share of total
            weighted votes (default 0.66 ≈ 2/3 majority).
        min_votes: Minimum number of distinct agents required. Below
            this, ``reason="insufficient_votes"`` and caller falls
            back to human review.
        namespace_config: Maps ``agent_id`` → namespace settings. Used
            to look up ``trust_weight`` when the ``Vote`` itself omits.

    Returns:
        :class:`ConsensusDecision` — caller audits ``reason`` before
        acting on ``winner``.
    """
    unique_agents = {v.agent_id for v in votes}
    if len(unique_agents) < min_votes:
        return ConsensusDecision(
            winner=None,
            margin=0.0,
            confidence=0.0,
            reason="insufficient_votes",
            vote_counts={},
        )

    tally = tally_votes(votes, namespace_config=namespace_config)
    if not tally:
        return ConsensusDecision(
            winner=None,
            margin=0.0,
            confidence=0.0,
            reason="insufficient_votes",
            vote_counts={},
        )

    total = sum(tally.values())
    winner, top_weight = max(tally.items(), key=lambda kv: kv[1])
    margin = top_weight / total if total > 0 else 0.0

    if margin < quorum_threshold:
        _log.info(
            "consensus_below_threshold",
            winner=winner,
            margin=round(margin, 3),
            threshold=quorum_threshold,
        )
        return ConsensusDecision(
            winner=None,
            margin=round(margin, 4),
            confidence=0.0,
            reason="below_threshold",
            vote_counts=tally,
        )

    # Confidence: how cleanly the margin beats the threshold.
    # 1.0 = exactly threshold; >1.0 = clear win.
    confidence = (margin - quorum_threshold) / max(1.0 - quorum_threshold, 1e-6) + 1.0
    _log.info(
        "consensus_reached",
        winner=winner,
        margin=round(margin, 3),
        confidence=round(confidence, 3),
        agents=len(unique_agents),
    )
    return ConsensusDecision(
        winner=winner,
        margin=round(margin, 4),
        confidence=round(min(confidence, 2.0), 4),
        reason="quorum",
        vote_counts=tally,
    )


def resolve_consensus_config(config: dict[str, Any] | None) -> dict[str, Any]:
    """Resolve ``governance.consensus`` knobs with safe defaults."""
    defaults: dict[str, Any] = {
        "enabled": False,
        "quorum_threshold": 0.66,
        "min_votes": 2,
    }
    if not config or not isinstance(config, dict):
        return defaults
    gov = config.get("governance", {})
    if not isinstance(gov, dict):
        return defaults
    cons = gov.get("consensus", {})
    if not isinstance(cons, dict):
        return defaults
    out = dict(defaults)
    if isinstance(cons.get("enabled"), bool):
        out["enabled"] = cons["enabled"]
    if isinstance(cons.get("quorum_threshold"), (int, float)) and 0 < cons["quorum_threshold"] <= 1:
        out["quorum_threshold"] = float(cons["quorum_threshold"])
    if isinstance(cons.get("min_votes"), int) and cons["min_votes"] >= 1:
        out["min_votes"] = int(cons["min_votes"])
    return out


__all__ = [
    "Vote",
    "ConsensusDecision",
    "tally_votes",
    "reach_consensus",
    "resolve_consensus_config",
]
