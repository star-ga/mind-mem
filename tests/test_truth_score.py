"""v3.3.0 — probabilistic truth score.

Pure function: ``truth_score(block, contradiction_votes=...,
age_half_life_days=...)`` → float in (0.01, 0.99).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from mind_mem.truth_score import (
    annotate_results,
    is_truth_score_enabled,
    truth_score,
)


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _days_ago(n: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=n)).strftime("%Y-%m-%d")


class TestStatusPriors:
    def test_verified_caps_high(self) -> None:
        score = truth_score({"Status": "verified", "Created": _today()})
        assert score >= 0.90

    def test_active_above_default(self) -> None:
        score = truth_score({"Status": "active", "Created": _today()})
        assert score >= 0.70

    def test_superseded_well_below_active(self) -> None:
        today = _today()
        a = truth_score({"Status": "active", "Created": today})
        s = truth_score({"Status": "superseded", "Created": today})
        assert s < a - 0.4

    def test_unknown_status_uses_default_prior(self) -> None:
        score = truth_score({"Status": "mystery", "Created": _today()})
        # Default unknown prior is 0.6; no decay today → ~0.6.
        assert 0.55 <= score <= 0.65

    def test_missing_status_uses_default(self) -> None:
        score = truth_score({"Created": _today()})
        assert 0.55 <= score <= 0.65


class TestAgeDecay:
    def test_today_no_decay(self) -> None:
        score = truth_score({"Status": "active", "Created": _today()})
        assert score >= 0.74

    def test_half_life_old_loses_roughly_half(self) -> None:
        hl = 180
        today_score = truth_score({"Status": "active", "Created": _today()}, age_half_life_days=hl)
        old_score = truth_score({"Status": "active", "Created": _days_ago(hl)}, age_half_life_days=hl)
        # Old ≈ today / 2 (± a bit for rounding).
        assert old_score == pytest.approx(today_score / 2, abs=0.05)

    def test_missing_date_no_decay(self) -> None:
        """No date → no decay, use the raw prior."""
        today_score = truth_score({"Status": "active", "Created": _today()})
        no_date_score = truth_score({"Status": "active"})
        assert abs(today_score - no_date_score) < 0.01


class TestContradictionVotes:
    def test_no_votes_unchanged(self) -> None:
        base = truth_score({"Status": "active", "Created": _today()})
        with_empty = truth_score({"Status": "active", "Created": _today()}, contradiction_votes=[])
        assert with_empty == base

    def test_one_vote_reduces_score(self) -> None:
        base = truth_score({"Status": "active", "Created": _today()})
        vote = truth_score(
            {"Status": "active", "Created": _today()},
            contradiction_votes=[{"weight": 0.9}],
        )
        assert vote < base

    def test_multiple_votes_compound(self) -> None:
        base = truth_score({"Status": "active", "Created": _today()})
        one_vote = truth_score(
            {"Status": "active", "Created": _today()},
            contradiction_votes=[{"weight": 0.5}],
        )
        many_votes = truth_score(
            {"Status": "active", "Created": _today()},
            contradiction_votes=[{"weight": 0.5}] * 5,
        )
        assert many_votes < one_vote < base

    def test_contradiction_cap_prevents_negative(self) -> None:
        """Extreme contradictions can't push below the clamp floor."""
        score = truth_score(
            {"Status": "active", "Created": _today()},
            contradiction_votes=[{"weight": 1.0}] * 100,
        )
        assert score >= 0.01


class TestAccessBonus:
    def test_access_count_bumps_score(self) -> None:
        base = truth_score({"Status": "active", "Created": _today()})
        accessed = truth_score({"Status": "active", "Created": _today(), "_access_count": 25})
        assert accessed > base

    def test_zero_access_no_bonus(self) -> None:
        base = truth_score({"Status": "active", "Created": _today()})
        zero = truth_score({"Status": "active", "Created": _today(), "_access_count": 0})
        assert base == zero


class TestClamps:
    def test_upper_clamp(self) -> None:
        score = truth_score({"Status": "verified", "Created": _today(), "_access_count": 1000})
        assert score <= 0.99

    def test_lower_clamp(self) -> None:
        score = truth_score(
            {"Status": "rejected", "Created": _days_ago(1000)},
            contradiction_votes=[{"weight": 1.0}] * 10,
        )
        assert score >= 0.01


class TestAnnotateResults:
    def test_empty_input(self) -> None:
        assert annotate_results([]) == []

    def test_annotates_each(self) -> None:
        results = [
            {"_id": "D-1", "Status": "active", "Created": _today()},
            {"_id": "D-2", "Status": "superseded", "Created": _today()},
        ]
        out = annotate_results(results)
        assert "truth_score" in out[0]
        assert "truth_score" in out[1]
        assert out[0]["truth_score"] > out[1]["truth_score"]

    def test_contradiction_graph_threads_through(self) -> None:
        results = [
            {"_id": "D-1", "Status": "active", "Created": _today()},
            {"_id": "D-2", "Status": "active", "Created": _today()},
        ]
        # D-2 has three contradictions; D-1 has none.
        graph = {"D-2": [{"weight": 0.9}] * 3}
        out = annotate_results(results, contradiction_graph=graph)
        assert out[0]["truth_score"] > out[1]["truth_score"]


class TestEnableResolution:
    def test_off_without_config(self) -> None:
        assert is_truth_score_enabled(None) is False
        assert is_truth_score_enabled({}) is False

    def test_off_by_default(self) -> None:
        assert is_truth_score_enabled({"retrieval": {"truth_score": {}}}) is False

    def test_explicit_on(self) -> None:
        assert is_truth_score_enabled({"retrieval": {"truth_score": {"enabled": True}}}) is True
