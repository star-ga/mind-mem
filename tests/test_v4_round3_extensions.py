"""Tests for round-3 audit extensions: observability + eviction."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from mind_mem.v4 import FeatureDisabledError
from mind_mem.v4.eviction import (
    DEFAULT_EVICTION_LIMIT,
    DEFAULT_LOW_SURPRISE_THRESHOLD,
    EvictionPlan,
    EvictionPolicy,
    available_policies,
    plan_eviction,
    register_policy,
)
from mind_mem.v4.eviction import FLAG as EVICT_FLAG
from mind_mem.v4.observability import FLAG as OBS_FLAG
from mind_mem.v4.observability import (
    Counter,
    Gauge,
    Histogram,
    MetricEvent,
    counter,
    gauge,
    histogram,
    reset_for_tests,
    set_exporter,
    snapshot,
    time_block,
    timed,
)


def _cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **flags: bool) -> Path:
    block = {k: {"enabled": v} for k, v in flags.items()}
    (tmp_path / "mind-mem.json").write_text(json.dumps({"v4": block}), encoding="utf-8")
    monkeypatch.setenv("MIND_MEM_CONFIG", str(tmp_path / "mind-mem.json"))
    return tmp_path


@pytest.fixture(autouse=True)
def _clean_obs_state() -> None:
    """Each test starts with a fresh observability registry."""
    reset_for_tests()
    yield
    reset_for_tests()


# ===========================================================================
# observability.py
# ===========================================================================


@pytest.fixture
def obs_on(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    return _cfg(tmp_path, monkeypatch, **{OBS_FLAG: True})


@pytest.fixture
def obs_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    return _cfg(tmp_path, monkeypatch, **{OBS_FLAG: False})


@pytest.mark.unit
def test_obs_flag_off_blocks_set_exporter(obs_off: Path) -> None:
    with pytest.raises(FeatureDisabledError):
        set_exporter(lambda _e: None)


@pytest.mark.unit
def test_counter_increments_atomically(obs_on: Path) -> None:
    c = counter("v4.test.cnt")
    c.inc()
    c.inc(5)
    c.inc()
    assert c.value == 7


@pytest.mark.unit
def test_counter_returns_same_instance(obs_on: Path) -> None:
    a = counter("v4.test.same")
    b = counter("v4.test.same")
    a.inc(3)
    assert b.value == 3
    assert a is b


@pytest.mark.unit
def test_gauge_set_overwrites(obs_on: Path) -> None:
    g = gauge("v4.test.gauge")
    g.set(1.0)
    g.set(99.0)
    g.set(0.5)
    assert g.value == 0.5


@pytest.mark.unit
def test_histogram_records_running_stats(obs_on: Path) -> None:
    h = histogram("v4.test.hist")
    for v in (1.0, 2.0, 3.0, 4.0, 5.0):
        h.observe(v)
    assert h.count == 5
    assert h.sum_v == 15.0
    assert h.min_v == 1.0
    assert h.max_v == 5.0


@pytest.mark.unit
def test_snapshot_returns_flat_view(obs_on: Path) -> None:
    counter("v4.test.c").inc(7)
    gauge("v4.test.g").set(42.5)
    histogram("v4.test.h").observe(3.0)
    histogram("v4.test.h").observe(5.0)
    snap = snapshot()
    assert snap["v4.test.c"] == 7
    assert snap["v4.test.g"] == 42.5
    h = snap["v4.test.h"]
    assert h["count"] == 2
    assert h["mean"] == 4.0
    assert h["min"] == 3.0
    assert h["max"] == 5.0


@pytest.mark.unit
def test_snapshot_works_when_flag_off() -> None:
    """Snapshot is a read-only path; should not require the flag."""
    # Manually populate the registry without the flag.
    counter("v4.test.read_only").inc()  # writes silently no-op vs exporter
    snap = snapshot()
    assert "v4.test.read_only" in snap


@pytest.mark.unit
def test_exporter_receives_events(obs_on: Path) -> None:
    captured: list[MetricEvent] = []

    def cap(e: MetricEvent) -> None:
        captured.append(e)

    set_exporter(cap)
    counter("v4.test.exp").inc(3)
    gauge("v4.test.exp_g").set(1.5)
    histogram("v4.test.exp_h").observe(0.5)

    kinds = {e.kind for e in captured}
    assert kinds == {"counter", "gauge", "histogram"}
    assert any(e.value == 3.0 and e.kind == "counter" for e in captured)


@pytest.mark.unit
def test_exporter_failure_does_not_crash(obs_on: Path) -> None:
    """Exporter raising must not break the recall path."""

    def bad(_e: MetricEvent) -> None:
        raise RuntimeError("boom")

    set_exporter(bad)
    # Should not raise.
    counter("v4.test.exporter_fail").inc()


@pytest.mark.unit
def test_exporter_silent_when_flag_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Disabled flag → exporter is never invoked."""
    captured: list[MetricEvent] = []

    # Briefly enable to set the exporter.
    cfg_on = _cfg(tmp_path, monkeypatch, **{OBS_FLAG: True})
    set_exporter(lambda e: captured.append(e))

    # Now flip flag off.
    _cfg(tmp_path, monkeypatch, **{OBS_FLAG: False})
    counter("v4.test.silent").inc()
    # No event reached the exporter while the flag was off.
    assert all(e.name != "v4.test.silent" for e in captured)
    _ = cfg_on  # keep ref


@pytest.mark.unit
def test_timed_decorator_records_latency(obs_on: Path) -> None:
    @timed("v4.test.dec_lat_ms")
    def slow() -> int:
        return sum(range(1000))

    slow()
    slow()
    snap = snapshot()
    h = snap["v4.test.dec_lat_ms"]
    assert h["count"] == 2
    # Wall time is small but positive.
    assert h["min"] >= 0


@pytest.mark.unit
def test_timed_decorator_records_latency_on_error(obs_on: Path) -> None:
    @timed("v4.test.err_ms")
    def boom() -> None:
        raise ValueError("kaboom")

    with pytest.raises(ValueError):
        boom()
    snap = snapshot()
    assert snap["v4.test.err_ms"]["count"] == 1


@pytest.mark.unit
def test_time_block_context_manager(obs_on: Path) -> None:
    with time_block("v4.test.ctx_ms"):
        sum(range(100))
    snap = snapshot()
    assert snap["v4.test.ctx_ms"]["count"] == 1


@pytest.mark.unit
def test_metric_types_are_distinct() -> None:
    assert Counter("c").name == "c"
    assert Gauge("g").name == "g"
    assert Histogram("h").name == "h"


# ===========================================================================
# eviction.py
# ===========================================================================


@pytest.fixture
def evict_on(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    return _cfg(tmp_path, monkeypatch, **{EVICT_FLAG: True})


@pytest.fixture
def evict_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    return _cfg(tmp_path, monkeypatch, **{EVICT_FLAG: False})


def _seed_cold(workspace: Path, rows: list[tuple[str, str, float | None]]) -> None:
    """rows: (block_id, last_seen_at, last_surprise)."""
    db = workspace / "index.db"
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS block_recall_tier ("
            "block_id TEXT PRIMARY KEY, tier TEXT NOT NULL DEFAULT 'cold', "
            "last_seen_at TEXT NOT NULL, promoted_count INTEGER NOT NULL DEFAULT 0, "
            "last_surprise REAL, block_version INTEGER NOT NULL DEFAULT 0)"
        )
        conn.executemany(
            "INSERT INTO block_recall_tier (block_id, tier, last_seen_at, last_surprise) VALUES (?, 'cold', ?, ?)",
            rows,
        )
        conn.commit()


@pytest.mark.unit
def test_evict_flag_off_blocks_plan(evict_off: Path) -> None:
    with pytest.raises(FeatureDisabledError):
        plan_eviction(evict_off)


@pytest.mark.unit
def test_evict_returns_empty_when_db_missing(evict_on: Path) -> None:
    p = plan_eviction(evict_on)
    assert isinstance(p, EvictionPlan)
    assert p.count == 0


@pytest.mark.unit
def test_evict_lru_returns_oldest_first(evict_on: Path) -> None:
    _seed_cold(
        evict_on,
        [
            ("B-newer", "2026-05-10T05:00:00Z", 0.4),
            ("B-oldest", "2026-05-09T05:00:00Z", 0.4),
            ("B-mid", "2026-05-09T18:00:00Z", 0.4),
        ],
    )
    p = plan_eviction(evict_on, EvictionPolicy.LRU, limit=10)
    ids = [bid for bid, _r in p.candidates]
    assert ids == ["B-oldest", "B-mid", "B-newer"]
    # Reasons carry the LRU tag.
    assert all(r.startswith("lru:") for _b, r in p.candidates)


@pytest.mark.unit
def test_evict_lru_respects_limit(evict_on: Path) -> None:
    _seed_cold(
        evict_on,
        [(f"B-{i:02}", f"2026-05-{i + 1:02}T00:00:00Z", 0.5) for i in range(20)],
    )
    p = plan_eviction(evict_on, EvictionPolicy.LRU, limit=3)
    assert p.count == 3


@pytest.mark.unit
def test_evict_low_surprise_filters(evict_on: Path) -> None:
    _seed_cold(
        evict_on,
        [
            ("B-bored", "2026-05-10T00:00:00Z", 0.05),
            ("B-active", "2026-05-10T00:00:00Z", 0.8),
            ("B-medium", "2026-05-10T00:00:00Z", 0.4),
        ],
    )
    p = plan_eviction(evict_on, EvictionPolicy.LOW_SURPRISE, threshold=0.1)
    ids = {bid for bid, _r in p.candidates}
    assert ids == {"B-bored"}


@pytest.mark.unit
def test_evict_age_filters_by_cutoff(evict_on: Path) -> None:
    _seed_cold(
        evict_on,
        [
            ("B-old", "2024-01-01T00:00:00Z", 0.5),
            ("B-mid", "2025-06-01T00:00:00Z", 0.5),
            ("B-recent", "2026-05-10T00:00:00Z", 0.5),
        ],
    )
    p = plan_eviction(evict_on, EvictionPolicy.AGE, cutoff_iso="2026-01-01T00:00:00Z")
    ids = {bid for bid, _r in p.candidates}
    assert ids == {"B-old", "B-mid"}


@pytest.mark.unit
def test_evict_composite_unions_dedupes(evict_on: Path) -> None:
    _seed_cold(
        evict_on,
        [
            ("B-old-bored", "2024-01-01T00:00:00Z", 0.05),  # old AND bored
            ("B-old", "2024-06-01T00:00:00Z", 0.5),
            ("B-bored", "2026-01-01T00:00:00Z", 0.05),
            ("B-fine", "2026-05-10T00:00:00Z", 0.9),
        ],
    )
    p = plan_eviction(
        evict_on,
        EvictionPolicy.COMPOSITE,
        policies=[EvictionPolicy.AGE.value, EvictionPolicy.LOW_SURPRISE.value],
        cutoff_iso="2025-01-01T00:00:00Z",
        threshold=0.1,
    )
    ids = [bid for bid, _r in p.candidates]
    # B-old-bored appears in both, should not be duplicated.
    assert ids.count("B-old-bored") == 1
    assert "B-fine" not in ids


@pytest.mark.unit
def test_evict_unknown_policy_returns_empty(evict_on: Path) -> None:
    p = plan_eviction(evict_on, "no_such_policy")
    assert p.count == 0


@pytest.mark.unit
def test_register_policy_runs_custom_fn(evict_on: Path) -> None:
    def always_one(_w: object, **_: object) -> list[tuple[str, str]]:
        return [("B-special", "custom:always_pick_me")]

    register_policy("custom_pick", always_one)
    p = plan_eviction(evict_on, "custom_pick")
    assert p.candidates == [("B-special", "custom:always_pick_me")]


@pytest.mark.unit
def test_available_policies_includes_builtins(evict_on: Path) -> None:
    out = available_policies()
    assert EvictionPolicy.LRU in out
    assert EvictionPolicy.LOW_SURPRISE in out
    assert EvictionPolicy.AGE in out
    assert EvictionPolicy.COMPOSITE in out


@pytest.mark.unit
def test_default_constants_documented() -> None:
    """Public constants stay where production tuners look for them."""
    assert DEFAULT_EVICTION_LIMIT == 100
    assert DEFAULT_LOW_SURPRISE_THRESHOLD == 0.1


@pytest.mark.unit
def test_eviction_plan_count_property() -> None:
    p = EvictionPlan(
        policy=EvictionPolicy.LRU,
        candidates=[("B-1", "x"), ("B-2", "y"), ("B-3", "z")],
    )
    assert p.count == 3
