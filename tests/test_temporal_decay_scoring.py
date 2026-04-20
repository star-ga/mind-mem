"""v3.3.0 Tier 1 #3 — half-life decay on block ``Created``/``Date`` field.

Pre-v3.3.0: ``date_score`` returned a linear 1.0..0.1 value over a
fixed 365-day window — usable as a filter but too coarse to meaningfully
rank within a recall result set.

v3.3.0: exponential half-life decay on the same field.
``score = 0.5 ** (age_days / half_life_days)`` with
``half_life_days`` read from ``retrieval.temporal_half_life_days``
(default 90). Used as a ranking signal, not a filter — the existing
date-score callers pick this up automatically.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from mind_mem._recall_scoring import (
    _resolve_half_life_days,
    temporal_decay_score,
)


class TestTemporalDecay:
    def test_fresh_block_scores_1(self) -> None:
        today = datetime.now().strftime("%Y-%m-%d")
        block = {"Created": today}
        assert temporal_decay_score(block) == pytest.approx(1.0, abs=0.01)

    def test_half_life_scores_0_5(self) -> None:
        half_life = 90
        past = (datetime.now() - timedelta(days=half_life)).strftime("%Y-%m-%d")
        block = {"Created": past}
        assert temporal_decay_score(block, half_life_days=half_life) == pytest.approx(0.5, abs=0.02)

    def test_two_half_lives_scores_0_25(self) -> None:
        half_life = 90
        past = (datetime.now() - timedelta(days=half_life * 2)).strftime("%Y-%m-%d")
        block = {"Created": past}
        assert temporal_decay_score(block, half_life_days=half_life) == pytest.approx(0.25, abs=0.02)

    def test_missing_date_returns_neutral(self) -> None:
        """Blocks without a date get 0.5 — avoids penalising undated content."""
        assert temporal_decay_score({}) == pytest.approx(0.5, abs=0.01)

    def test_malformed_date_returns_neutral(self) -> None:
        assert temporal_decay_score({"Created": "not-a-date"}) == pytest.approx(0.5, abs=0.01)

    def test_future_date_clamps_to_1(self) -> None:
        future = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
        assert temporal_decay_score({"Created": future}) == pytest.approx(1.0, abs=0.01)

    def test_falls_back_to_date_field(self) -> None:
        """``Date`` field also honoured when ``Created`` is missing."""
        today = datetime.now().strftime("%Y-%m-%d")
        assert temporal_decay_score({"Date": today}) == pytest.approx(1.0, abs=0.01)

    def test_custom_half_life(self) -> None:
        """Short half-life compresses decay."""
        past_30 = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        block = {"Created": past_30}
        assert temporal_decay_score(block, half_life_days=30) == pytest.approx(0.5, abs=0.02)


class TestHalfLifeResolution:
    def test_default_is_90_days(self) -> None:
        assert _resolve_half_life_days(None) == 90

    def test_reads_from_retrieval_config(self) -> None:
        cfg = {"retrieval": {"temporal_half_life_days": 45}}
        assert _resolve_half_life_days(cfg) == 45

    def test_ignores_non_int(self) -> None:
        cfg = {"retrieval": {"temporal_half_life_days": "nope"}}
        assert _resolve_half_life_days(cfg) == 90

    def test_clamps_negative(self) -> None:
        cfg = {"retrieval": {"temporal_half_life_days": -5}}
        assert _resolve_half_life_days(cfg) == 90

    def test_top_level_missing_retrieval(self) -> None:
        assert _resolve_half_life_days({"unrelated": {}}) == 90
