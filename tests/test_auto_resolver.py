"""Tests for mind-mem auto contradiction resolution (auto_resolver.py)."""

import os

import pytest

from mind_mem.auto_resolver import AutoResolver, ResolutionSuggestion
from mind_mem.conflict_resolver import ResolutionStrategy


@pytest.fixture
def workspace(tmp_path):
    ws = str(tmp_path)
    os.makedirs(os.path.join(ws, "decisions"), exist_ok=True)
    os.makedirs(os.path.join(ws, "intelligence"), exist_ok=True)
    return ws


@pytest.fixture
def resolver(workspace):
    return AutoResolver(workspace)


class TestResolutionSuggestion:
    def test_to_dict(self):
        s = ResolutionSuggestion(
            contradiction_id="C-001",
            block_a="D-001",
            block_b="D-002",
            strategy="timestamp_priority",
            confidence_score=0.85,
            winner_id="D-002",
            loser_id="D-001",
            rationale="Newer decision wins",
            side_effects=["D-003 depends on D-001"],
        )
        d = s.to_dict()
        assert d["confidence_score"] == 0.85
        assert d["winner_id"] == "D-002"
        assert len(d["side_effects"]) == 1


class TestPreferences:
    def test_record_preference(self, resolver):
        resolver.record_preference("timestamp_priority", chosen=True)
        resolver.record_preference("timestamp_priority", chosen=True)
        resolver.record_preference("timestamp_priority", chosen=False)

        summary = resolver.preference_summary()
        assert "timestamp_priority" in summary
        assert summary["timestamp_priority"]["chosen"] == 2
        assert summary["timestamp_priority"]["rejected"] == 1

    def test_domain_preference(self, resolver):
        resolver.record_preference("confidence_priority", domain="security", chosen=True)
        resolver.record_preference("confidence_priority", domain="api", chosen=False)

        summary = resolver.preference_summary()
        assert "confidence_priority:security" in summary
        assert summary["confidence_priority:security"]["chosen"] == 1

    def test_acceptance_rate(self, resolver):
        for _ in range(8):
            resolver.record_preference("scope_priority", chosen=True)
        for _ in range(2):
            resolver.record_preference("scope_priority", chosen=False)

        summary = resolver.preference_summary()
        assert summary["scope_priority"]["acceptance_rate"] == 0.8

    def test_preference_boost(self, resolver):
        # Record strong preference
        for _ in range(10):
            resolver.record_preference("timestamp_priority", chosen=True)

        boost = resolver._get_preference_boost("timestamp_priority")
        assert boost > 0  # Should have positive boost

    def test_no_preference_no_boost(self, resolver):
        boost = resolver._get_preference_boost("unknown_strategy")
        assert boost == 0.0


class TestSuggestions:
    def test_empty_workspace(self, resolver):
        suggestions = resolver.suggest_resolutions()
        assert suggestions == []

    def test_accept_suggestion(self, resolver):
        s = ResolutionSuggestion(
            contradiction_id="C-001",
            block_a="D-001",
            block_b="D-002",
            strategy="timestamp_priority",
            confidence_score=0.8,
            winner_id="D-002",
            loser_id="D-001",
            rationale="Test",
        )
        resolver.accept_suggestion(s)

        # Should record preference
        summary = resolver.preference_summary()
        assert "timestamp_priority" in summary
        assert summary["timestamp_priority"]["chosen"] == 1

        # Should log to audit chain
        ok, errors = resolver._chain.verify()
        assert ok

    def test_reject_suggestion(self, resolver):
        s = ResolutionSuggestion(
            contradiction_id="C-001",
            block_a="D-001",
            block_b="D-002",
            strategy="scope_priority",
            confidence_score=0.6,
            winner_id="D-001",
            loser_id="D-002",
            rationale="Test reject",
        )
        resolver.reject_suggestion(s, reason="Incorrect analysis")

        summary = resolver.preference_summary()
        assert summary["scope_priority"]["rejected"] == 1


class TestSideEffects:
    def test_no_dependents(self, resolver):
        effects = resolver._analyze_side_effects("D-001")
        assert effects == []

    def test_with_dependents(self, resolver):
        from mind_mem.causal_graph import EDGE_DEPENDS_ON
        resolver._graph.add_edge("D-002", "D-001", EDGE_DEPENDS_ON)
        resolver._graph.add_edge("D-003", "D-001", EDGE_DEPENDS_ON)

        effects = resolver._analyze_side_effects("D-001")
        assert len(effects) == 2
        assert any("D-002" in e for e in effects)
        assert any("D-003" in e for e in effects)
