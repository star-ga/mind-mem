"""Regression tests: configured namespace trust actually reaches the tally.

``Vote.trust_weight`` used to default to 1.0, so an ordinary
``Vote(agent_id, choice)`` never consulted ``namespace_config`` — the
lookup was dead for exactly the votes it exists to serve, and a
configured 0 (an excluded agent) still counted at full weight.
"""

from __future__ import annotations

import pytest

from mind_mem.consensus_vote import Vote, reach_consensus, tally_votes

_NAMESPACES = {
    "agent-alice": {"trust_weight": 5.0},
    "agent-bob": {"trust_weight": 0},
}


class TestNamespaceTrustReachesTally:
    def test_default_vote_uses_configured_weight(self) -> None:
        votes = [Vote("agent-alice", "A"), Vote("agent-bob", "B")]
        assert tally_votes(votes, namespace_config=_NAMESPACES) == {"A": 5.0}

    def test_configured_zero_excludes_the_agent(self) -> None:
        assert tally_votes([Vote("agent-bob", "B")], namespace_config=_NAMESPACES) == {}

    def test_unconfigured_agent_still_defaults_to_one(self) -> None:
        assert tally_votes([Vote("agent-carl", "C")], namespace_config=_NAMESPACES) == {"C": 1.0}

    def test_explicit_weight_on_the_vote_still_wins(self) -> None:
        votes = [Vote("agent-alice", "A", trust_weight=2.0)]
        assert tally_votes(votes, namespace_config=_NAMESPACES) == {"A": 2.0}

    def test_quorum_follows_configured_trust(self) -> None:
        """Weight-5 alice outvotes two default-weight agents."""
        votes = [Vote("agent-alice", "A"), Vote("agent-carl", "B"), Vote("agent-dana", "B")]
        decision = reach_consensus(votes, quorum_threshold=0.66, namespace_config=_NAMESPACES)
        assert decision.winner == "A"
        assert decision.margin == pytest.approx(5.0 / 7.0, abs=1e-4)
