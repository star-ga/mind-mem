"""Tests for src/mind_mem/telemetry.py.

Covers graceful degradation (no OTel/Prometheus), decorator correctness,
counter increments, and idempotent init functions.
"""

from __future__ import annotations

import sys
import threading
import types
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# 1. Import without opentelemetry installed
# ---------------------------------------------------------------------------


def test_import_without_opentelemetry() -> None:
    """telemetry must import cleanly and expose all public symbols.

    When opentelemetry IS installed on the host _HAS_OTEL will be True,
    which is correct behaviour.  We just verify the three public names are
    present and callable regardless of whether the package is installed.
    """
    from mind_mem import telemetry

    assert hasattr(telemetry, "init_tracing")
    assert hasattr(telemetry, "init_prometheus")
    assert hasattr(telemetry, "traced")
    assert callable(telemetry.init_tracing)
    assert callable(telemetry.init_prometheus)
    assert callable(telemetry.traced)
    # _HAS_OTEL and _HAS_PROM must be booleans (not None / missing)
    assert isinstance(telemetry._HAS_OTEL, bool)
    assert isinstance(telemetry._HAS_PROM, bool)


def test_import_exposes_all_public_symbols() -> None:
    """After a normal import the three key names are present."""
    from mind_mem.telemetry import init_prometheus, init_tracing, traced  # noqa: F401

    assert callable(init_tracing)
    assert callable(init_prometheus)
    assert callable(traced)


# ---------------------------------------------------------------------------
# 2. @traced does not corrupt return values
# ---------------------------------------------------------------------------


def test_traced_preserves_return_value_no_deps() -> None:
    """@traced must pass through the return value unchanged."""
    from mind_mem import telemetry

    @telemetry.traced("scan")
    def _compute(x: int) -> int:
        return x * 2

    assert _compute(7) == 14


def test_traced_preserves_return_value_none() -> None:
    from mind_mem import telemetry

    @telemetry.traced("scan")
    def _noop() -> None:
        return None

    assert _noop() is None


def test_traced_preserves_return_value_complex() -> None:
    from mind_mem import telemetry

    @telemetry.traced("recall")
    def _returns_list() -> list[int]:
        return [1, 2, 3]

    assert _returns_list() == [1, 2, 3]


def test_traced_re_raises_exceptions() -> None:
    """@traced must propagate exceptions unchanged."""
    import pytest

    from mind_mem import telemetry

    @telemetry.traced("scan")
    def _boom() -> None:
        raise ValueError("expected")

    with pytest.raises(ValueError, match="expected"):
        _boom()


# ---------------------------------------------------------------------------
# 3. Prometheus counters increment
# ---------------------------------------------------------------------------


def test_prometheus_counters_increment_with_mock() -> None:
    """_record_* helpers call Prometheus metric objects when installed."""
    from mind_mem import telemetry

    # Build lightweight mock metric objects
    counter_mock = MagicMock()
    histogram_mock = MagicMock()

    # Save originals so we can restore them; never set to None to avoid
    # duplicate-registration errors from prometheus_client's global registry.
    original_prom = telemetry._HAS_PROM
    orig_recall_total = telemetry._recall_total
    orig_recall_duration = telemetry._recall_duration
    orig_propose = telemetry._propose_update_total
    orig_scan = telemetry._scan_total
    try:
        telemetry._HAS_PROM = True  # type: ignore[assignment]
        telemetry._recall_total = counter_mock
        telemetry._recall_duration = histogram_mock
        telemetry._propose_update_total = counter_mock
        telemetry._scan_total = counter_mock

        telemetry._record_recall(0.05)
        counter_mock.inc.assert_called()
        histogram_mock.observe.assert_called_with(0.05)

        telemetry._record_propose_update()
        telemetry._record_scan()
        assert counter_mock.inc.call_count >= 3
    finally:
        telemetry._HAS_PROM = original_prom  # type: ignore[assignment]
        telemetry._recall_total = orig_recall_total
        telemetry._recall_duration = orig_recall_duration
        telemetry._propose_update_total = orig_propose
        telemetry._scan_total = orig_scan


def test_record_helpers_noop_without_prometheus() -> None:
    """_record_* helpers silently skip when prometheus_client absent."""
    from mind_mem import telemetry

    original = telemetry._HAS_PROM
    try:
        telemetry._HAS_PROM = False  # type: ignore[assignment]
        # These must not raise
        telemetry._record_recall(0.1)
        telemetry._record_propose_update()
        telemetry._record_scan()
        telemetry._record_apply()
        telemetry._record_apply_rollback()
    finally:
        telemetry._HAS_PROM = original  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# 4. init_tracing / init_prometheus are idempotent
# ---------------------------------------------------------------------------


def test_init_tracing_idempotent_noop() -> None:
    """init_tracing is safe to call multiple times without OTel installed."""
    from mind_mem import telemetry

    original_flag = telemetry._HAS_OTEL
    original_init = telemetry._otel_initialized
    try:
        telemetry._HAS_OTEL = False  # type: ignore[assignment]
        telemetry._otel_initialized = False
        # Must not raise
        telemetry.init_tracing()
        telemetry.init_tracing(endpoint="http://localhost:4317")
        telemetry.init_tracing()
    finally:
        telemetry._HAS_OTEL = original_flag  # type: ignore[assignment]
        telemetry._otel_initialized = original_init


def test_init_prometheus_idempotent_noop() -> None:
    """init_prometheus is safe to call multiple times without prometheus_client."""
    from mind_mem import telemetry

    original_flag = telemetry._HAS_PROM
    original_started = telemetry._prom_started
    try:
        telemetry._HAS_PROM = False  # type: ignore[assignment]
        telemetry._prom_started = False
        telemetry.init_prometheus(port=19090)
        telemetry.init_prometheus(port=29090)  # second call — must not raise
    finally:
        telemetry._HAS_PROM = original_flag  # type: ignore[assignment]
        telemetry._prom_started = original_started


def test_init_prometheus_idempotent_with_mock() -> None:
    """init_prometheus calls start_http_server exactly once even if called twice."""
    from mind_mem import telemetry

    original_flag = telemetry._HAS_PROM
    original_started = telemetry._prom_started
    # Save originals; restore rather than null so we don't break prometheus_client's
    # global registry for tests that run after this one.
    orig_rd = telemetry._recall_duration
    orig_rt = telemetry._recall_total
    orig_pu = telemetry._propose_update_total
    orig_sc = telemetry._scan_total
    orig_ap = telemetry._apply_total
    orig_rb = telemetry._apply_rollback_total
    try:
        # Force metrics to None so _init_prom_metrics uses the fake module
        telemetry._recall_duration = None
        telemetry._recall_total = None
        telemetry._propose_update_total = None
        telemetry._scan_total = None
        telemetry._apply_total = None
        telemetry._apply_rollback_total = None

        telemetry._HAS_PROM = True  # type: ignore[assignment]
        telemetry._prom_started = False

        start_mock = MagicMock()

        fake_prom = types.ModuleType("prometheus_client")
        fake_prom.start_http_server = start_mock  # type: ignore[attr-defined]
        fake_prom.Counter = MagicMock(return_value=MagicMock())  # type: ignore[attr-defined]
        fake_prom.Histogram = MagicMock(return_value=MagicMock())  # type: ignore[attr-defined]

        with patch.dict(sys.modules, {"prometheus_client": fake_prom}):
            telemetry.init_prometheus(port=19091)
            telemetry.init_prometheus(port=19091)  # second call
            telemetry.init_prometheus(port=19091)  # third call

        assert start_mock.call_count == 1
    finally:
        telemetry._HAS_PROM = original_flag  # type: ignore[assignment]
        telemetry._prom_started = original_started
        # Restore original metric objects (never set to None after real init)
        telemetry._recall_duration = orig_rd
        telemetry._recall_total = orig_rt
        telemetry._propose_update_total = orig_pu
        telemetry._scan_total = orig_sc
        telemetry._apply_total = orig_ap
        telemetry._apply_rollback_total = orig_rb


# ---------------------------------------------------------------------------
# 5. _fire_metric dispatches correctly
# ---------------------------------------------------------------------------


def test_fire_metric_unknown_span_is_noop() -> None:
    """_fire_metric with an unknown span name must not raise."""
    from mind_mem import telemetry

    telemetry._fire_metric("unknown_span", 0.1)  # must not raise


def test_fire_metric_recall_passes_duration() -> None:
    from mind_mem import telemetry

    original_flag = telemetry._HAS_PROM
    orig_duration = telemetry._recall_duration
    orig_total = telemetry._recall_total
    try:
        telemetry._HAS_PROM = True  # type: ignore[assignment]
        duration_mock = MagicMock()
        counter_mock = MagicMock()
        telemetry._recall_duration = duration_mock
        telemetry._recall_total = counter_mock

        telemetry._fire_metric("recall", 0.123)

        counter_mock.inc.assert_called_once()
        duration_mock.observe.assert_called_once_with(0.123)
    finally:
        telemetry._HAS_PROM = original_flag  # type: ignore[assignment]
        telemetry._recall_duration = orig_duration
        telemetry._recall_total = orig_total


# ---------------------------------------------------------------------------
# 6. Thread-safety: concurrent traced calls don't crash
# ---------------------------------------------------------------------------


def test_traced_concurrent_calls() -> None:
    """Concurrent @traced calls must not corrupt return values or raise.

    Prometheus metrics are disabled for this test to avoid duplicate-registration
    errors from the prometheus_client global registry across test runs.
    """
    from mind_mem import telemetry

    original_prom = telemetry._HAS_PROM
    try:
        telemetry._HAS_PROM = False  # type: ignore[assignment]

        results: list[int] = []
        lock = threading.Lock()

        @telemetry.traced("scan")
        def _work(n: int) -> int:
            return n * n

        def _run(n: int) -> None:
            val = _work(n)
            with lock:
                results.append(val)

        threads = [threading.Thread(target=_run, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert sorted(results) == sorted([i * i for i in range(20)])
    finally:
        telemetry._HAS_PROM = original_prom  # type: ignore[assignment]
