# Copyright 2026 STARGA, Inc.
"""mind-mem Kalman Filter Belief Updater — confidence estimation for memory blocks.

Models each memory block's "truth confidence" as a state estimate that evolves
over time using a scalar Kalman filter.  Variance (uncertainty) grows during
predict steps (time decay) and shrinks during update steps (new evidence).

Mathematical model (scalar Kalman, no external deps):
    Predict:  variance += process_noise * dt
    Update:   R      = observation_noise / source_reliability
              K      = variance / (variance + R)
              estimate = estimate + K * (observation - estimate)
              variance = (1 - K) * variance

Integration points (by convention — no imports required):
    drift_detector:       observation = 0.5  (uncertain after drift detection)
    contradiction_detector: observation = 0.0  (conflicting evidence)
    approve_apply:        observation = 1.0  (proposal confirmed)
    rollback:             observation = 0.0  (proposal rejected)

Usage:
    from .kalman_belief import BeliefState, BeliefStore, KalmanBeliefUpdater

    updater = KalmanBeliefUpdater()
    store = BeliefStore(db_path="/tmp/beliefs.db")
    state = store.update_belief("D-001", observation=1.0, source="approve_apply")

Zero external deps — all stdlib (dataclasses, datetime, json, sqlite3).
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default process noise: how quickly uncertainty grows per hour without updates.
DEFAULT_PROCESS_NOISE: float = 0.01

# Default source reliability when no prior information is available.
DEFAULT_SOURCE_RELIABILITY: float = 0.7

# Default observation noise baseline (before scaling by source reliability).
DEFAULT_OBSERVATION_NOISE: float = 0.1

# EMA alpha for source reliability updates.
_RELIABILITY_EMA_ALPHA: float = 0.2

# Confidence threshold below which a belief flags for review.
_DEFAULT_REVIEW_THRESHOLD: float = 0.3

# Initial estimate for newly created belief states.
_INITIAL_ESTIMATE: float = 0.5

# Initial variance for newly created belief states (high uncertainty).
_INITIAL_VARIANCE: float = 0.5


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class BeliefState:
    """Belief state for one memory block.

    Attributes:
        block_id:           Unique block identifier (e.g. "D-20260405-001").
        estimate:           Current confidence in this belief (0.0–1.0).
        variance:           Uncertainty — higher means less certain.
        last_updated:       Timezone-aware UTC timestamp of last update.
        observation_count:  Total number of observations incorporated.
        source_reliability: Map of source_name → reliability score (0.0–1.0).
    """

    block_id: str
    estimate: float
    variance: float
    last_updated: datetime
    observation_count: int
    source_reliability: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# KalmanBeliefUpdater
# ---------------------------------------------------------------------------


class KalmanBeliefUpdater:
    """Scalar Kalman filter for evolving belief confidence over time.

    Args:
        process_noise:             Variance growth rate per hour (predict step).
        default_source_reliability: Reliability assumed for unregistered sources.
    """

    def __init__(
        self,
        process_noise: float = DEFAULT_PROCESS_NOISE,
        default_source_reliability: float = DEFAULT_SOURCE_RELIABILITY,
    ) -> None:
        if process_noise < 0:
            raise ValueError("process_noise must be non-negative")
        if not 0.0 <= default_source_reliability <= 1.0:
            raise ValueError("default_source_reliability must be in [0, 1]")
        self._process_noise = process_noise
        self._default_reliability = default_source_reliability
        self._source_reliability: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def initial_state(self, block_id: str) -> BeliefState:
        """Return a fresh belief state with default priors."""
        return BeliefState(
            block_id=block_id,
            estimate=_INITIAL_ESTIMATE,
            variance=_INITIAL_VARIANCE,
            last_updated=_utcnow(),
            observation_count=0,
            source_reliability={},
        )

    def predict(self, state: BeliefState, dt_hours: float) -> BeliefState:
        """Time-decay step: variance grows proportionally with elapsed time.

        Args:
            state:     Current belief state.
            dt_hours:  Hours elapsed since last prediction.

        Returns:
            New BeliefState with increased variance; estimate unchanged.
        """
        if dt_hours < 0:
            raise ValueError("dt_hours must be non-negative")
        new_variance = min(state.variance + self._process_noise * dt_hours, 1.0)
        return BeliefState(
            block_id=state.block_id,
            estimate=state.estimate,
            variance=new_variance,
            last_updated=state.last_updated,
            observation_count=state.observation_count,
            source_reliability=dict(state.source_reliability),
        )

    def update(
        self,
        state: BeliefState,
        observation: float,
        source: str,
        observation_noise: float | None = None,
    ) -> BeliefState:
        """Kalman correction step: incorporate new evidence.

        The effective observation noise is scaled by the source's reliability:
            R = observation_noise / source_reliability

        A reliability of 1.0 gives a noise exactly equal to observation_noise.
        A reliability of 0.01 inflates R 100×, heavily damping the update.

        Args:
            state:            Current belief state.
            observation:      Observed value (0.0–1.0).
            source:           Name of the evidence source.
            observation_noise: Override noise; defaults to DEFAULT_OBSERVATION_NOISE.

        Returns:
            New BeliefState with updated estimate and reduced variance.
        """
        reliability = self._source_reliability.get(source, self._default_reliability)
        reliability = max(reliability, 1e-6)  # guard against division by zero

        base_noise = observation_noise if observation_noise is not None else DEFAULT_OBSERVATION_NOISE
        r = base_noise / reliability  # effective measurement noise

        divisor = state.variance + r
        k = 0.0 if divisor == 0.0 else state.variance / divisor  # Kalman gain
        new_estimate = state.estimate + k * (observation - state.estimate)
        new_variance = (1.0 - k) * state.variance

        # Clamp to [0, 1] — physical bounds for probability-like values.
        new_estimate = max(0.0, min(1.0, new_estimate))
        new_variance = max(0.0, min(1.0, new_variance))

        # Snapshot current reliability into state for provenance.
        updated_reliability = dict(state.source_reliability)
        updated_reliability[source] = reliability

        return BeliefState(
            block_id=state.block_id,
            estimate=new_estimate,
            variance=new_variance,
            last_updated=_utcnow(),
            observation_count=state.observation_count + 1,
            source_reliability=updated_reliability,
        )

    def get_confidence(self, state: BeliefState) -> float:
        """Effective confidence accounting for both estimate and uncertainty.

        Returns estimate scaled down by uncertainty:
            confidence = estimate * (1 - variance)

        A state with estimate=0.9 but variance=0.9 has low real confidence.

        Returns:
            Float in [0.0, 1.0].
        """
        return state.estimate * (1.0 - state.variance)

    def should_review(self, state: BeliefState, threshold: float = _DEFAULT_REVIEW_THRESHOLD) -> bool:
        """Return True when the effective confidence has decayed below threshold.

        Args:
            state:     Belief state to evaluate.
            threshold: Review if confidence < threshold (default 0.3).
        """
        return self.get_confidence(state) < threshold

    def register_source(self, source: str, reliability: float) -> None:
        """Set a source's reliability score.

        Args:
            source:      Source identifier.
            reliability: Score in [0.0, 1.0]; 0 = unreliable, 1 = perfect.
        """
        if not 0.0 <= reliability <= 1.0:
            raise ValueError(f"reliability must be in [0, 1]; got {reliability}")
        self._source_reliability[source] = reliability

    def update_source_reliability(self, source: str, was_correct: bool) -> None:
        """Adjust source reliability using EMA based on observed outcome.

        Args:
            source:      Source identifier.
            was_correct: True if the source's information proved accurate.
        """
        current = self._source_reliability.get(source, self._default_reliability)
        target = 1.0 if was_correct else 0.0
        updated = current + _RELIABILITY_EMA_ALPHA * (target - current)
        self._source_reliability[source] = max(0.0, min(1.0, updated))


# ---------------------------------------------------------------------------
# BeliefStore
# ---------------------------------------------------------------------------

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS beliefs (
    block_id           TEXT PRIMARY KEY,
    estimate           REAL NOT NULL,
    variance           REAL NOT NULL,
    last_updated       TEXT NOT NULL,
    observation_count  INTEGER NOT NULL,
    source_reliability TEXT NOT NULL
)
"""

_UPSERT_SQL = """
INSERT INTO beliefs (block_id, estimate, variance, last_updated, observation_count, source_reliability)
VALUES (?, ?, ?, ?, ?, ?)
ON CONFLICT(block_id) DO UPDATE SET
    estimate           = excluded.estimate,
    variance           = excluded.variance,
    last_updated       = excluded.last_updated,
    observation_count  = excluded.observation_count,
    source_reliability = excluded.source_reliability
"""


class BeliefStore:
    """Manages Kalman belief states for all memory blocks.

    In-memory by default; pass db_path for SQLite persistence.

    Args:
        db_path:  Optional path to an SQLite database file.  If provided,
                  beliefs are loaded on init and persisted on every write.
        updater:  KalmanBeliefUpdater instance; a default is created if omitted.
    """

    def __init__(
        self,
        db_path: str | None = None,
        updater: KalmanBeliefUpdater | None = None,
    ) -> None:
        self._updater = updater or KalmanBeliefUpdater()
        self._beliefs: dict[str, BeliefState] = {}
        self._db_path = db_path
        if db_path:
            self._init_db(db_path)
            self._load_from_db()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_belief(self, block_id: str) -> BeliefState:
        """Return current belief state for block_id; create initial if absent."""
        if block_id not in self._beliefs:
            self._beliefs[block_id] = self._updater.initial_state(block_id)
            self._persist(block_id)
        return self._beliefs[block_id]

    def update_belief(self, block_id: str, observation: float, source: str) -> BeliefState:
        """Incorporate a new observation for block_id.

        Args:
            block_id:    Target memory block identifier.
            observation: Evidence strength (0.0–1.0).
            source:      Identifier of the evidence source.

        Returns:
            Updated BeliefState.
        """
        current = self.get_belief(block_id)
        updated = self._updater.update(current, observation=observation, source=source)
        self._beliefs[block_id] = updated
        self._persist(block_id)
        return updated

    def decay_all(self, hours_elapsed: float) -> list[str]:
        """Apply time decay to all tracked beliefs.

        Args:
            hours_elapsed: Hours since the last decay pass.

        Returns:
            List of block_ids whose confidence has fallen below the review
            threshold after decay.
        """
        stale_ids: list[str] = []
        for block_id, state in self._beliefs.items():
            decayed = self._updater.predict(state, dt_hours=hours_elapsed)
            self._beliefs[block_id] = decayed
            self._persist(block_id)
            if self._updater.should_review(decayed):
                stale_ids.append(block_id)
        return stale_ids

    def get_stale_beliefs(self, threshold: float = _DEFAULT_REVIEW_THRESHOLD) -> list[BeliefState]:
        """Return all beliefs whose confidence is below threshold.

        Args:
            threshold: Confidence floor (default 0.3).
        """
        return [s for s in self._beliefs.values() if self._updater.should_review(s, threshold=threshold)]

    def to_dict(self) -> dict:
        """Serialize all beliefs to a JSON-compatible dict."""
        return {block_id: _state_to_dict(state) for block_id, state in self._beliefs.items()}

    def from_dict(self, data: dict) -> None:
        """Restore beliefs from a dict produced by to_dict().

        Existing in-memory beliefs are replaced.
        """
        self._beliefs = {block_id: _state_from_dict(raw) for block_id, raw in data.items()}
        if self._db_path:
            for block_id in self._beliefs:
                self._persist(block_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_db(self, db_path: str) -> None:
        with sqlite3.connect(db_path) as conn:
            conn.execute(_CREATE_TABLE_SQL)
            conn.commit()

    def _load_from_db(self) -> None:
        if not self._db_path:
            return
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                "SELECT block_id, estimate, variance, last_updated, observation_count, source_reliability FROM beliefs"
            ).fetchall()
        for row in rows:
            block_id, estimate, variance, last_updated_str, obs_count, sr_json = row
            last_updated = datetime.fromisoformat(last_updated_str)
            source_reliability = json.loads(sr_json)
            self._beliefs[block_id] = BeliefState(
                block_id=block_id,
                estimate=estimate,
                variance=variance,
                last_updated=last_updated,
                observation_count=obs_count,
                source_reliability=source_reliability,
            )

    def _persist(self, block_id: str) -> None:
        if not self._db_path:
            return
        state = self._beliefs[block_id]
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                _UPSERT_SQL,
                (
                    state.block_id,
                    state.estimate,
                    state.variance,
                    state.last_updated.isoformat(),
                    state.observation_count,
                    json.dumps(state.source_reliability),
                ),
            )
            conn.commit()


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC time as timezone-aware datetime."""
    return datetime.now(tz=timezone.utc)


def _state_to_dict(state: BeliefState) -> dict:
    return {
        "block_id": state.block_id,
        "estimate": state.estimate,
        "variance": state.variance,
        "last_updated": state.last_updated.isoformat(),
        "observation_count": state.observation_count,
        "source_reliability": state.source_reliability,
    }


def _state_from_dict(data: dict) -> BeliefState:
    return BeliefState(
        block_id=data["block_id"],
        estimate=data["estimate"],
        variance=data["variance"],
        last_updated=datetime.fromisoformat(data["last_updated"]),
        observation_count=data["observation_count"],
        source_reliability=data.get("source_reliability", {}),
    )
