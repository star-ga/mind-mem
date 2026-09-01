"""Tests for round-3 audit extensions: observability.

The eviction half went with RA.0 — every one of its policies was a query
against ``block_recall_tier``, the deleted ladder's table.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mind_mem.v4 import FeatureDisabledError
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
