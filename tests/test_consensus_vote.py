"""v3.3.0 — quorum-based consensus voting on contradictions."""

from __future__ import annotations

import pytest

from mind_mem.consensus_vote import (
    ConsensusDecision,
    Vote,
    reach_consensus,
    resolve_consensus_config,
    tally_votes,
)


def _v(agent: str, choice: str, weight: float = 1.0) -> Vote:
    return Vote(agent_id=agent, choice=choice, trust_weight=weight)


class TestTallyVotes:
    def test_basic_tally(self) -> None:
        votes = [_v("a", "x"), _v("b", "x"), _v("c", "y")]
        assert tally_votes(votes) == {"x": 2.0, "y": 1.0}

    def test_duplicate_votes_collapsed_to_max(self) -> None:
        """An agent that re-votes for the same choice doesn't double-count."""
        votes = [_v("a", "x", 1.0), _v("a", "x", 1.5)]
        assert tally_votes(votes) == {"x": 1.5}

    def test_namespace_trust_weight_applies(self) -> None:
        votes = [Vote(agent_id="a", choice="x", trust_weight=0)]  # 0 → use namespace lookup
        namespaces = {"a": {"trust_weight": 2.5}}
        assert tally_votes(votes, namespace_config=namespaces) == {"x": 2.5}

    def test_zero_weight_falls_back_to_namespace_default(self) -> None:
        """trust_weight=0 in the Vote → look up in namespace config.

        When no namespace config is provided either, the default of
        1.0 applies. To truly exclude an agent, set the namespace's
        trust_weight to 0 as well.
        """
        votes = [Vote(agent_id="a", choice="x", trust_weight=0)]
        assert tally_votes(votes) == {"x": 1.0}
        # With namespace trust_weight=0, agent is excluded.
        assert tally_votes(votes, namespace_config={"a": {"trust_weight": 0}}) == {}


class TestReachConsensus:
    def test_insufficient_votes(self) -> None:
        d = reach_consensus([_v("a", "x")], min_votes=2)
        assert d.winner is None
        assert d.reason == "insufficient_votes"

    def test_clear_quorum_wins(self) -> None:
        votes = [_v("a", "x"), _v("b", "x"), _v("c", "y")]  # 2/3 = 66.6%
        d = reach_consensus(votes, quorum_threshold=0.6)
        assert d.winner == "x"
        assert d.reason == "quorum"
        assert d.margin > 0.6

    def test_below_threshold_no_winner(self) -> None:
        votes = [_v("a", "x"), _v("b", "y"), _v("c", "z")]  # 33% each
        d = reach_consensus(votes, quorum_threshold=0.66)
        assert d.winner is None
        assert d.reason == "below_threshold"
        # Tally was populated even though threshold wasn't met
        assert set(d.vote_counts.keys()) == {"x", "y", "z"}

    def test_trust_weighted_win(self) -> None:
        """Two default-weight agents vs one heavy-weight agent — heavy wins."""
        votes = [
            _v("a", "x", weight=1.0),
            _v("b", "x", weight=1.0),
            _v("c", "y", weight=3.0),
        ]
        d = reach_consensus(votes, quorum_threshold=0.55)
        assert d.winner == "y"
        assert d.margin == pytest.approx(3.0 / 5.0)

    def test_confidence_scales_with_margin(self) -> None:
        """Confidence ≥ 1.0 at threshold, higher for cleaner wins."""
        tight = reach_consensus(
            [_v("a", "x"), _v("b", "x"), _v("c", "y")],
            quorum_threshold=0.66,
        )
        clear = reach_consensus(
            [_v("a", "x"), _v("b", "x"), _v("c", "x"), _v("d", "y")],
            quorum_threshold=0.66,
        )
        assert tight.confidence <= clear.confidence

    def test_unanimous_top_confidence(self) -> None:
        votes = [_v("a", "x"), _v("b", "x"), _v("c", "x")]
        d = reach_consensus(votes, quorum_threshold=0.5)
        assert d.winner == "x"
        # Unanimous = margin 1.0, confidence ≥ 1.0 and capped at 2.0
        assert d.confidence >= 1.0


class TestConfigResolution:
    def test_defaults(self) -> None:
        out = resolve_consensus_config(None)
        assert out == {"enabled": False, "quorum_threshold": 0.66, "min_votes": 2}

    def test_custom(self) -> None:
        out = resolve_consensus_config({"governance": {"consensus": {"enabled": True, "quorum_threshold": 0.75, "min_votes": 3}}})
        assert out == {"enabled": True, "quorum_threshold": 0.75, "min_votes": 3}

    def test_invalid_threshold_falls_back(self) -> None:
        out = resolve_consensus_config({"governance": {"consensus": {"quorum_threshold": 1.5}}})
        assert out["quorum_threshold"] == 0.66

    def test_invalid_min_votes_falls_back(self) -> None:
        out = resolve_consensus_config({"governance": {"consensus": {"min_votes": 0}}})
        assert out["min_votes"] == 2
