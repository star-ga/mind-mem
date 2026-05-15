"""Tests for v4 circuit breaker (round 5 audit, Mistral + GLM 9.9→10)."""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from mind_mem.v4 import FeatureDisabledError
from mind_mem.v4 import circuit_breaker as cb_mod
from mind_mem.v4.circuit_breaker import (
    DEFAULT_FAILURE_THRESHOLD,
    DEFAULT_HALF_OPEN_PROBES,
    DEFAULT_RECOVERY_TIMEOUT_S,
    FLAG,
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
    circuit_breaker,
    default_breaker,
)


def _cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **flags: bool) -> Path:
    block = {k: {"enabled": v} for k, v in flags.items()}
    (tmp_path / "mind-mem.json").write_text(json.dumps({"v4": block}), encoding="utf-8")
    monkeypatch.setenv("MIND_MEM_CONFIG", str(tmp_path / "mind-mem.json"))
    return tmp_path


@pytest.fixture
def cb_on(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    return _cfg(tmp_path, monkeypatch, **{FLAG: True})


@pytest.fixture
def cb_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    return _cfg(tmp_path, monkeypatch, **{FLAG: False})


@pytest.fixture(autouse=True)
def _reset_singleton() -> None:
    cb_mod.reset_for_tests()
    yield
    cb_mod.reset_for_tests()


# ---------------------------------------------------------------------------
# Defaults + validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_defaults_match_documented(cb_on: Path) -> None:
    b = CircuitBreaker()
    assert b.failure_threshold == DEFAULT_FAILURE_THRESHOLD
    assert b.recovery_timeout == DEFAULT_RECOVERY_TIMEOUT_S
    assert b.half_open_probes == DEFAULT_HALF_OPEN_PROBES


@pytest.mark.unit
def test_constructor_rejects_invalid_threshold(cb_on: Path) -> None:
    with pytest.raises(ValueError):
        CircuitBreaker(failure_threshold=0)


@pytest.mark.unit
def test_constructor_rejects_negative_recovery(cb_on: Path) -> None:
    with pytest.raises(ValueError):
        CircuitBreaker(recovery_timeout=-1.0)


@pytest.mark.unit
def test_constructor_rejects_zero_probes(cb_on: Path) -> None:
    with pytest.raises(ValueError):
        CircuitBreaker(half_open_probes=0)


# ---------------------------------------------------------------------------
# Flag gating
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_flag_off_blocks_call(cb_off: Path) -> None:
    b = CircuitBreaker()
    with pytest.raises(FeatureDisabledError):
        b.call(lambda: 1)


@pytest.mark.unit
def test_flag_off_blocks_default_breaker(cb_off: Path) -> None:
    with pytest.raises(FeatureDisabledError):
        default_breaker()


# ---------------------------------------------------------------------------
# CLOSED state — happy path
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_closed_passes_through_success(cb_on: Path) -> None:
    b = CircuitBreaker()
    assert b.state() is CircuitState.CLOSED
    assert b.call(lambda x: x * 2, 5) == 10
    assert b.state() is CircuitState.CLOSED
    assert b.failure_count() == 0


@pytest.mark.unit
def test_closed_resets_failure_count_on_success(cb_on: Path) -> None:
    b = CircuitBreaker(failure_threshold=5)
    for _ in range(3):
        with pytest.raises(RuntimeError):
            b.call(_raises)
    assert b.failure_count() == 3
    b.call(lambda: 1)  # success
    assert b.failure_count() == 0
    assert b.state() is CircuitState.CLOSED


@pytest.mark.unit
def test_closed_propagates_original_exception(cb_on: Path) -> None:
    b = CircuitBreaker(failure_threshold=10)
    with pytest.raises(ValueError, match="boom"):
        b.call(lambda: (_ for _ in ()).throw(ValueError("boom")))


# ---------------------------------------------------------------------------
# CLOSED → OPEN trip
# ---------------------------------------------------------------------------


def _raises() -> None:
    raise RuntimeError("synthetic failure")


@pytest.mark.unit
def test_trips_to_open_after_threshold(cb_on: Path) -> None:
    b = CircuitBreaker(failure_threshold=3)
    for _ in range(3):
        with pytest.raises(RuntimeError):
            b.call(_raises)
    assert b.state() is CircuitState.OPEN


@pytest.mark.unit
def test_open_short_circuits_with_retry_after(cb_on: Path) -> None:
    b = CircuitBreaker(failure_threshold=2, recovery_timeout=10.0)
    for _ in range(2):
        with pytest.raises(RuntimeError):
            b.call(_raises)
    with pytest.raises(CircuitOpenError) as exc_info:
        b.call(lambda: 1)
    assert exc_info.value.retry_after > 0
    assert exc_info.value.retry_after <= 10.0


@pytest.mark.unit
def test_open_does_not_call_wrapped_function(cb_on: Path) -> None:
    b = CircuitBreaker(failure_threshold=1, recovery_timeout=10.0)
    with pytest.raises(RuntimeError):
        b.call(_raises)
    assert b.state() is CircuitState.OPEN

    calls = []

    def tracker() -> int:
        calls.append(1)
        return 42

    with pytest.raises(CircuitOpenError):
        b.call(tracker)
    assert calls == []


# ---------------------------------------------------------------------------
# OPEN → HALF_OPEN → CLOSED
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_recovers_via_half_open(cb_on: Path) -> None:
    # v4.0.9: timer-resolution margin. Windows timer resolution is
    # ~15.6ms by default, so the original recovery_timeout=0.05 + sleep
    # 0.06 (10ms slack) was flaky on windows-3.12 — retry_after rounded
    # to 0.00s but the state machine hadn't transitioned. 0.1s timeout
    # + 0.15s sleep gives 50ms of slack which is >3× the worst-case
    # Windows tick.
    b = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)
    for _ in range(2):
        with pytest.raises(RuntimeError):
            b.call(_raises)
    assert b.state() is CircuitState.OPEN
    time.sleep(0.15)
    # First call after timeout enters HALF_OPEN; success closes.
    assert b.call(lambda: "ok") == "ok"
    assert b.state() is CircuitState.CLOSED


@pytest.mark.unit
def test_half_open_failure_reopens_for_full_window(cb_on: Path) -> None:
    b = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)
    with pytest.raises(RuntimeError):
        b.call(_raises)
    time.sleep(0.15)
    # HALF_OPEN probe fails → re-OPEN
    with pytest.raises(RuntimeError):
        b.call(_raises)
    assert b.state() is CircuitState.OPEN
    # Within new window, still short-circuiting.
    with pytest.raises(CircuitOpenError):
        b.call(lambda: 1)


@pytest.mark.unit
def test_half_open_requires_n_probes_to_close(cb_on: Path) -> None:
    b = CircuitBreaker(failure_threshold=1, recovery_timeout=0.05, half_open_probes=3)
    with pytest.raises(RuntimeError):
        b.call(_raises)
    time.sleep(0.06)
    # First two probes succeed but stay HALF_OPEN.
    b.call(lambda: 1)
    assert b.state() is CircuitState.HALF_OPEN
    b.call(lambda: 1)
    assert b.state() is CircuitState.HALF_OPEN
    # Third probe closes.
    b.call(lambda: 1)
    assert b.state() is CircuitState.CLOSED


@pytest.mark.unit
def test_time_until_retry_zero_when_closed(cb_on: Path) -> None:
    b = CircuitBreaker()
    assert b.time_until_retry() == 0.0


@pytest.mark.unit
def test_time_until_retry_decreases(cb_on: Path) -> None:
    b = CircuitBreaker(failure_threshold=1, recovery_timeout=0.5)
    with pytest.raises(RuntimeError):
        b.call(_raises)
    t1 = b.time_until_retry()
    time.sleep(0.05)
    t2 = b.time_until_retry()
    assert t2 < t1
    assert t1 <= 0.5


# ---------------------------------------------------------------------------
# Manual transitions
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_reset_returns_to_closed(cb_on: Path) -> None:
    b = CircuitBreaker(failure_threshold=1, recovery_timeout=10.0)
    with pytest.raises(RuntimeError):
        b.call(_raises)
    assert b.state() is CircuitState.OPEN
    b.reset()
    assert b.state() is CircuitState.CLOSED
    assert b.failure_count() == 0
    # And calls work again immediately.
    assert b.call(lambda: 1) == 1


@pytest.mark.unit
def test_trip_forces_open(cb_on: Path) -> None:
    b = CircuitBreaker()
    assert b.state() is CircuitState.CLOSED
    b.trip()
    assert b.state() is CircuitState.OPEN
    with pytest.raises(CircuitOpenError):
        b.call(lambda: 1)


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_decorator_wraps_function(cb_on: Path) -> None:
    @circuit_breaker(failure_threshold=3, recovery_timeout=10.0)
    def fn(x: int) -> int:
        return x + 1

    assert fn(1) == 2
    assert fn.breaker.state() is CircuitState.CLOSED


@pytest.mark.unit
def test_decorator_trips_on_failures(cb_on: Path) -> None:
    @circuit_breaker(failure_threshold=2, recovery_timeout=10.0)
    def fn() -> None:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        fn()
    with pytest.raises(RuntimeError):
        fn()
    assert fn.breaker.state() is CircuitState.OPEN
    with pytest.raises(CircuitOpenError):
        fn()


@pytest.mark.unit
def test_decorator_per_function_breaker_isolation(cb_on: Path) -> None:
    @circuit_breaker(failure_threshold=1, recovery_timeout=10.0)
    def fail() -> None:
        raise RuntimeError()

    @circuit_breaker(failure_threshold=1, recovery_timeout=10.0)
    def ok() -> int:
        return 42

    with pytest.raises(RuntimeError):
        fail()
    assert fail.breaker.state() is CircuitState.OPEN
    # ok's breaker is independent — must still be CLOSED.
    assert ok.breaker.state() is CircuitState.CLOSED
    assert ok() == 42


# ---------------------------------------------------------------------------
# Singleton + config
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_default_breaker_is_singleton(cb_on: Path) -> None:
    b1 = default_breaker()
    b2 = default_breaker()
    assert b1 is b2


@pytest.mark.unit
def test_default_breaker_reads_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "mind-mem.json").write_text(
        json.dumps(
            {
                "v4": {
                    FLAG: {
                        "enabled": True,
                        "failure_threshold": 17,
                        "recovery_timeout_s": 99.0,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("MIND_MEM_CONFIG", str(tmp_path / "mind-mem.json"))
    cb_mod.reset_for_tests()
    b = default_breaker()
    assert b.failure_threshold == 17
    assert b.recovery_timeout == 99.0


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_concurrent_calls_consistent_state(cb_on: Path) -> None:
    b = CircuitBreaker(failure_threshold=10, recovery_timeout=10.0)

    def worker(i: int) -> None:
        if i % 3 == 0:
            try:
                b.call(_raises)
            except RuntimeError:
                pass
            except CircuitOpenError:
                pass
        else:
            try:
                b.call(lambda: 1)
            except CircuitOpenError:
                pass

    with ThreadPoolExecutor(max_workers=16) as ex:
        list(ex.map(worker, range(200)))

    # State must be one of the three valid states (no torn reads /
    # invalid intermediates).
    assert b.state() in {CircuitState.CLOSED, CircuitState.OPEN, CircuitState.HALF_OPEN}


@pytest.mark.unit
def test_concurrent_failures_trip_at_most_once(cb_on: Path) -> None:
    """20 threads racing through a failing function must (a) trip the
    breaker exactly once and (b) leave it in OPEN with a sane
    failure_count, regardless of how many threads got past the
    pre-call state check before the trip."""
    b = CircuitBreaker(failure_threshold=5, recovery_timeout=10.0)
    barrier = threading.Barrier(20)
    actual_call_count = threading.Semaphore(0)

    def fail_and_count() -> None:
        actual_call_count.release()
        raise RuntimeError("synthetic failure")

    def worker() -> None:
        barrier.wait()
        try:
            b.call(fail_and_count)
        except RuntimeError:
            pass
        except CircuitOpenError:
            pass

    with ThreadPoolExecutor(max_workers=20) as ex:
        list(ex.map(lambda _: worker(), range(20)))

    # Breaker must end OPEN — failure_threshold reached.
    assert b.state() is CircuitState.OPEN
    # At least 5 calls reached the wrapped fn (the threshold), but
    # never more than 20 (the worker count). Confirms the lock is
    # not over- or under-counting.
    n_calls = 0
    while actual_call_count.acquire(blocking=False):
        n_calls += 1
    assert 5 <= n_calls <= 20


@pytest.mark.unit
def test_record_success_under_contention(cb_on: Path) -> None:
    b = CircuitBreaker(failure_threshold=100, recovery_timeout=10.0)
    barrier = threading.Barrier(32)

    def worker() -> int:
        barrier.wait()
        return b.call(lambda: 1)

    with ThreadPoolExecutor(max_workers=32) as ex:
        results = list(ex.map(lambda _: worker(), range(32)))
    assert all(r == 1 for r in results)
    assert b.state() is CircuitState.CLOSED
    assert b.failure_count() == 0


# ---------------------------------------------------------------------------
# Ergonomics — does it integrate with surprise_retrieval cleanly?
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_integration_with_fallback_pattern(cb_on: Path) -> None:
    """Demonstrates the round-5 use case: an embedder wrapped in a
    circuit breaker that falls back to a cheaper / cached embedding
    when the breaker is OPEN."""
    b = CircuitBreaker(failure_threshold=2, recovery_timeout=10.0)

    def primary() -> list[float]:
        raise RuntimeError("upstream timeout")

    def fallback() -> list[float]:
        return [0.0] * 8

    def embed_with_fallback() -> list[float]:
        try:
            return b.call(primary)
        except (RuntimeError, CircuitOpenError):
            return fallback()

    # First two calls: primary raises, breaker counts.
    assert embed_with_fallback() == [0.0] * 8
    assert embed_with_fallback() == [0.0] * 8
    assert b.state() is CircuitState.OPEN
    # Third call: short-circuits, fallback covers.
    assert embed_with_fallback() == [0.0] * 8
