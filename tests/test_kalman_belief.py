# Copyright 2026 STARGA, Inc.
"""Tests for KalmanBeliefUpdater and BeliefStore (kalman_belief.py).

Covers: initial state, time decay, Kalman update math, source reliability
tracking, stale belief detection, serialization, and SQLite persistence.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from mind_mem.kalman_belief import (
    BeliefState,
    BeliefStore,
    KalmanBeliefUpdater,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 4, 5, 12, 0, 0, tzinfo=timezone.utc)


def _state(
    block_id: str = "D-001",
    estimate: float = 0.8,
    variance: float = 0.1,
    observation_count: int = 1,
    source_reliability: dict[str, float] | None = None,
    last_updated: datetime | None = None,
) -> BeliefState:
    return BeliefState(
        block_id=block_id,
        estimate=estimate,
        variance=variance,
        last_updated=last_updated or _NOW,
        observation_count=observation_count,
        source_reliability=source_reliability or {},
    )


# ---------------------------------------------------------------------------
# BeliefState — dataclass contract
# ---------------------------------------------------------------------------


class TestBeliefState:
    def test_fields_present(self):
        s = _state()
        assert s.block_id == "D-001"
        assert s.estimate == 0.8
        assert s.variance == 0.1
        assert isinstance(s.last_updated, datetime)
        assert s.observation_count == 1
        assert isinstance(s.source_reliability, dict)

    def test_defaults_are_sensible(self):
        # Freshly created belief with default updater params
        updater = KalmanBeliefUpdater()
        s = updater.initial_state("X-001")
        assert 0.0 <= s.estimate <= 1.0
        assert s.variance > 0.0
        assert s.observation_count == 0

    def test_estimate_bounds_respected_after_update(self):
        updater = KalmanBeliefUpdater()
        s = _state(estimate=0.95, variance=0.05)
        # Observation of 1.0 should nudge towards 1.0 but not exceed it
        s2 = updater.update(s, observation=1.0, source="oracle")
        assert 0.0 <= s2.estimate <= 1.0

    def test_estimate_floor_after_update(self):
        updater = KalmanBeliefUpdater()
        s = _state(estimate=0.05, variance=0.05)
        s2 = updater.update(s, observation=0.0, source="oracle")
        assert 0.0 <= s2.estimate <= 1.0


# ---------------------------------------------------------------------------
# KalmanBeliefUpdater — predict (time decay)
# ---------------------------------------------------------------------------


class TestPredict:
    def test_variance_grows_with_time(self):
        updater = KalmanBeliefUpdater(process_noise=0.01)
        s = _state(variance=0.1)
        s2 = updater.predict(s, dt_hours=1.0)
        assert s2.variance > s.variance

    def test_variance_grows_proportional_to_dt(self):
        updater = KalmanBeliefUpdater(process_noise=0.01)
        s = _state(variance=0.1)
        s1h = updater.predict(s, dt_hours=1.0)
        s10h = updater.predict(s, dt_hours=10.0)
        # 10-hour decay must have grown variance more than 1-hour
        assert s10h.variance > s1h.variance

    def test_estimate_unchanged_by_predict(self):
        updater = KalmanBeliefUpdater()
        s = _state(estimate=0.75, variance=0.05)
        s2 = updater.predict(s, dt_hours=24.0)
        assert s2.estimate == pytest.approx(0.75)

    def test_predict_is_immutable(self):
        updater = KalmanBeliefUpdater()
        s = _state(variance=0.1)
        s2 = updater.predict(s, dt_hours=5.0)
        assert s.variance == 0.1  # original unchanged
        assert s2 is not s

    def test_zero_dt_no_change(self):
        updater = KalmanBeliefUpdater(process_noise=0.01)
        s = _state(variance=0.1)
        s2 = updater.predict(s, dt_hours=0.0)
        assert s2.variance == pytest.approx(s.variance)

    def test_variance_capped_at_one(self):
        updater = KalmanBeliefUpdater(process_noise=1.0)
        s = _state(variance=0.9)
        s2 = updater.predict(s, dt_hours=1.0)
        assert s2.variance <= 1.0


# ---------------------------------------------------------------------------
# KalmanBeliefUpdater — update (Kalman correction step)
# ---------------------------------------------------------------------------


class TestUpdate:
    def test_update_is_immutable(self):
        updater = KalmanBeliefUpdater()
        s = _state(estimate=0.5, variance=0.2)
        s2 = updater.update(s, observation=0.9, source="wiki")
        assert s.estimate == 0.5  # original untouched
        assert s2 is not s

    def test_observation_count_increments(self):
        updater = KalmanBeliefUpdater()
        s = _state(observation_count=3)
        s2 = updater.update(s, observation=0.8, source="wiki")
        assert s2.observation_count == 4

    def test_reliable_source_moves_estimate_strongly(self):
        updater = KalmanBeliefUpdater()
        updater.register_source("oracle", reliability=1.0)
        s = _state(estimate=0.5, variance=0.5)
        s2 = updater.update(s, observation=1.0, source="oracle")
        # High reliability → high Kalman gain → estimate moves far towards 1.0
        assert s2.estimate > 0.8

    def test_unreliable_source_moves_estimate_weakly(self):
        updater = KalmanBeliefUpdater()
        updater.register_source("rumor", reliability=0.05)
        s = _state(estimate=0.5, variance=0.5)
        s2 = updater.update(s, observation=1.0, source="rumor")
        # Low reliability → low Kalman gain (≈0.2) → estimate barely moves
        # Math: R = 0.1/0.05 = 2.0, K = 0.5/2.5 = 0.2, Δ = 0.2*0.5 = 0.1 → 0.6
        assert s2.estimate <= 0.65

    def test_variance_decreases_after_update(self):
        updater = KalmanBeliefUpdater()
        s = _state(variance=0.5)
        s2 = updater.update(s, observation=0.8, source="default")
        assert s2.variance < s.variance

    def test_unknown_source_uses_default_reliability(self):
        updater = KalmanBeliefUpdater(default_source_reliability=0.7)
        s = _state(estimate=0.5, variance=0.5)
        s2 = updater.update(s, observation=1.0, source="unknown_source")
        # Should succeed without error, using default 0.7
        assert 0.5 < s2.estimate < 1.0

    def test_explicit_observation_noise_overrides_default(self):
        updater = KalmanBeliefUpdater()
        s = _state(estimate=0.5, variance=0.5)
        # High explicit noise → smaller gain → smaller shift
        s_high_noise = updater.update(s, observation=1.0, source="default", observation_noise=10.0)
        s_low_noise = updater.update(s, observation=1.0, source="default", observation_noise=0.01)
        assert s_low_noise.estimate > s_high_noise.estimate

    def test_source_reliability_stored_in_state(self):
        updater = KalmanBeliefUpdater()
        updater.register_source("agent_a", reliability=0.9)
        s = _state()
        s2 = updater.update(s, observation=0.8, source="agent_a")
        assert "agent_a" in s2.source_reliability
        assert s2.source_reliability["agent_a"] == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# KalmanBeliefUpdater — get_confidence / should_review
# ---------------------------------------------------------------------------


class TestConfidenceAndReview:
    def test_high_estimate_low_variance_gives_high_confidence(self):
        updater = KalmanBeliefUpdater()
        s = _state(estimate=0.9, variance=0.01)
        assert updater.get_confidence(s) > 0.8

    def test_low_estimate_gives_low_confidence(self):
        updater = KalmanBeliefUpdater()
        s = _state(estimate=0.1, variance=0.05)
        assert updater.get_confidence(s) < 0.3

    def test_high_variance_reduces_confidence(self):
        updater = KalmanBeliefUpdater()
        s_low_var = _state(estimate=0.8, variance=0.01)
        s_high_var = _state(estimate=0.8, variance=0.9)
        assert updater.get_confidence(s_low_var) > updater.get_confidence(s_high_var)

    def test_should_review_below_threshold(self):
        updater = KalmanBeliefUpdater()
        s = _state(estimate=0.1, variance=0.9)
        assert updater.should_review(s, threshold=0.3) is True

    def test_should_not_review_above_threshold(self):
        updater = KalmanBeliefUpdater()
        s = _state(estimate=0.9, variance=0.01)
        assert updater.should_review(s, threshold=0.3) is False


# ---------------------------------------------------------------------------
# Source reliability tracking (EMA)
# ---------------------------------------------------------------------------


class TestSourceReliability:
    def test_register_source(self):
        updater = KalmanBeliefUpdater()
        updater.register_source("journal", reliability=0.95)
        assert updater._source_reliability["journal"] == pytest.approx(0.95)

    def test_correct_outcome_increases_reliability(self):
        updater = KalmanBeliefUpdater()
        updater.register_source("agent", reliability=0.5)
        updater.update_source_reliability("agent", was_correct=True)
        assert updater._source_reliability["agent"] > 0.5

    def test_incorrect_outcome_decreases_reliability(self):
        updater = KalmanBeliefUpdater()
        updater.register_source("agent", reliability=0.5)
        updater.update_source_reliability("agent", was_correct=False)
        assert updater._source_reliability["agent"] < 0.5

    def test_reliability_clamped_0_to_1(self):
        updater = KalmanBeliefUpdater()
        updater.register_source("perfect", reliability=0.99)
        for _ in range(50):
            updater.update_source_reliability("perfect", was_correct=True)
        assert updater._source_reliability["perfect"] <= 1.0

    def test_unknown_source_auto_registers_on_update_reliability(self):
        updater = KalmanBeliefUpdater(default_source_reliability=0.7)
        updater.update_source_reliability("new_source", was_correct=True)
        assert "new_source" in updater._source_reliability


# ---------------------------------------------------------------------------
# BeliefStore
# ---------------------------------------------------------------------------


class TestBeliefStore:
    def test_get_belief_returns_initial_for_unknown_block(self):
        store = BeliefStore()
        s = store.get_belief("D-new")
        assert s.block_id == "D-new"
        assert 0.0 <= s.estimate <= 1.0

    def test_update_belief_persists(self):
        store = BeliefStore()
        s1 = store.update_belief("D-001", observation=1.0, source="oracle")
        s2 = store.get_belief("D-001")
        assert s2.estimate == pytest.approx(s1.estimate)

    def test_decay_all_grows_variance(self):
        store = BeliefStore()
        store.get_belief("D-001")  # ensure entry exists
        store.get_belief("D-002")
        variances_before = {bid: store.get_belief(bid).variance for bid in ("D-001", "D-002")}
        store.decay_all(hours_elapsed=10.0)
        for bid in ("D-001", "D-002"):
            assert store.get_belief(bid).variance >= variances_before[bid]

    def test_decay_all_returns_stale_ids(self):
        store = BeliefStore()
        # Force a low-confidence block
        store._beliefs["D-stale"] = BeliefState(
            block_id="D-stale",
            estimate=0.1,
            variance=0.9,
            last_updated=_NOW,
            observation_count=1,
            source_reliability={},
        )
        stale_ids = store.decay_all(hours_elapsed=1.0)
        assert "D-stale" in stale_ids

    def test_get_stale_beliefs_filters_correctly(self):
        store = BeliefStore()
        store._beliefs["D-good"] = _state(estimate=0.9, variance=0.01)
        store._beliefs["D-bad"] = _state(block_id="D-bad", estimate=0.1, variance=0.9)
        stale = store.get_stale_beliefs(threshold=0.3)
        ids = [s.block_id for s in stale]
        assert "D-bad" in ids
        assert "D-good" not in ids

    def test_serialization_round_trip(self):
        store = BeliefStore()
        store.update_belief("D-001", observation=0.9, source="a")
        store.update_belief("D-002", observation=0.3, source="b")
        data = store.to_dict()
        store2 = BeliefStore()
        store2.from_dict(data)
        for bid in ("D-001", "D-002"):
            orig = store.get_belief(bid)
            restored = store2.get_belief(bid)
            assert orig.estimate == pytest.approx(restored.estimate, abs=1e-9)
            assert orig.variance == pytest.approx(restored.variance, abs=1e-9)
            assert orig.observation_count == restored.observation_count

    def test_sqlite_persistence(self, tmp_path):
        db_path = str(tmp_path / "beliefs.db")
        store1 = BeliefStore(db_path=db_path)
        store1.update_belief("D-001", observation=0.8, source="persist_test")
        estimate1 = store1.get_belief("D-001").estimate

        # New store instance, same db — should reload
        store2 = BeliefStore(db_path=db_path)
        estimate2 = store2.get_belief("D-001").estimate
        assert estimate2 == pytest.approx(estimate1, abs=1e-9)

    def test_from_dict_with_sqlite_persists(self, tmp_path):
        db_path = str(tmp_path / "restore.db")
        store1 = BeliefStore(db_path=db_path)
        store1.update_belief("D-010", observation=0.7, source="s1")
        data = store1.to_dict()

        # Restore into a fresh db-backed store using from_dict
        db_path2 = str(tmp_path / "restore2.db")
        store2 = BeliefStore(db_path=db_path2)
        store2.from_dict(data)

        # Reload from db to confirm persistence fired inside from_dict
        store3 = BeliefStore(db_path=db_path2)
        assert store3.get_belief("D-010").estimate == pytest.approx(store1.get_belief("D-010").estimate, abs=1e-9)


class TestInputValidation:
    def test_register_source_rejects_out_of_range(self):
        updater = KalmanBeliefUpdater()
        with pytest.raises(ValueError, match="reliability must be in"):
            updater.register_source("bad", reliability=1.5)

    def test_predict_rejects_negative_dt(self):
        updater = KalmanBeliefUpdater()
        s = _state()
        with pytest.raises(ValueError, match="dt_hours must be non-negative"):
            updater.predict(s, dt_hours=-1.0)

    def test_init_rejects_negative_process_noise(self):
        with pytest.raises(ValueError, match="process_noise must be non-negative"):
            KalmanBeliefUpdater(process_noise=-0.01)

    def test_init_rejects_out_of_range_default_reliability(self):
        with pytest.raises(ValueError, match="default_source_reliability"):
            KalmanBeliefUpdater(default_source_reliability=1.5)
