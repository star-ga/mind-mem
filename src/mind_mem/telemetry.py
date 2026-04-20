"""mind-mem Telemetry — OpenTelemetry traces + Prometheus metrics.

Optional-dep module. Gracefully no-ops when opentelemetry-api or
prometheus_client are not installed.

Usage:
    from mind_mem.telemetry import init_tracing, init_prometheus, traced

    init_tracing(endpoint="http://jaeger:4317")
    init_prometheus(port=9090)

    @traced("recall")
    def recall(workspace, query, ...):
        ...

When OTel / Prometheus packages are absent every call is a no-op and
import cost is negligible.
"""

from __future__ import annotations

import functools
import importlib.util
import threading
import time
from typing import Any, Callable, TypeVar

# ---------------------------------------------------------------------------
# Availability probes (zero-import-cost checks)
# ---------------------------------------------------------------------------

_HAS_OTEL = (
    importlib.util.find_spec("opentelemetry") is not None
    and importlib.util.find_spec("opentelemetry.trace") is not None
)
_HAS_PROM = importlib.util.find_spec("prometheus_client") is not None

# ---------------------------------------------------------------------------
# Prometheus metrics (lazy-init on first use)
# ---------------------------------------------------------------------------

_prom_lock = threading.Lock()
_prom_metrics_lock = threading.Lock()
_prom_started = False

_recall_duration: Any = None
_recall_total: Any = None
_propose_update_total: Any = None
_scan_total: Any = None
_apply_total: Any = None
_apply_rollback_total: Any = None


def _init_prom_metrics() -> None:
    """Create Prometheus metric objects exactly once (thread-safe)."""
    global _recall_duration, _recall_total, _propose_update_total
    global _scan_total, _apply_total, _apply_rollback_total

    if not _HAS_PROM:
        return
    if _recall_duration is not None:
        return  # fast path — already done, no lock needed

    with _prom_metrics_lock:
        # Double-checked locking: re-test inside lock to handle contention
        if _recall_duration is not None:
            return

        from prometheus_client import Counter, Histogram  # type: ignore[import-untyped]

        _recall_duration = Histogram(
            "recall_duration_seconds",
            "Time spent in recall()",
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
        )
        _recall_total = Counter("recall_total", "Total recall() invocations")
        _propose_update_total = Counter("propose_update_total", "Total propose_update() invocations")
        _scan_total = Counter("scan_total", "Total scan() invocations")
        _apply_total = Counter("apply_total", "Total approve_apply() invocations")
        _apply_rollback_total = Counter("apply_rollback_total", "Total rollback_proposal() invocations")


def init_prometheus(port: int = 9090) -> None:
    """Start Prometheus HTTP server on *port*.

    Idempotent — subsequent calls with the same or different port are no-ops
    after the first successful start.  Silently skips when prometheus_client
    is not installed.
    """
    global _prom_started

    if not _HAS_PROM:
        return

    # Initialise metric objects before taking the server-start lock so
    # _prom_metrics_lock and _prom_lock are never held concurrently.
    _init_prom_metrics()

    with _prom_lock:
        if _prom_started:
            return
        from prometheus_client import start_http_server  # type: ignore[import-untyped]

        start_http_server(port)
        _prom_started = True


# ---------------------------------------------------------------------------
# OpenTelemetry tracer (lazy-init)
# ---------------------------------------------------------------------------

_tracer: Any = None
_otel_lock = threading.Lock()
_otel_initialized = False


def init_tracing(endpoint: str | None = None) -> None:
    """Configure the global OTel tracer.

    If *endpoint* is provided and opentelemetry-exporter-otlp is installed,
    an OTLP gRPC exporter is configured targeting that endpoint.  Otherwise
    the SDK's NoOp tracer is used (zero overhead).

    Idempotent — repeated calls are no-ops.
    """
    global _tracer, _otel_initialized

    if not _HAS_OTEL:
        return

    with _otel_lock:
        if _otel_initialized:
            return

        from opentelemetry import trace  # type: ignore[import-untyped]
        from opentelemetry.sdk.resources import Resource  # type: ignore[import-untyped]
        from opentelemetry.sdk.trace import TracerProvider  # type: ignore[import-untyped]
        from opentelemetry.sdk.trace.export import BatchSpanProcessor  # type: ignore[import-untyped]

        resource = Resource.create({"service.name": "mind-mem"})
        provider = TracerProvider(resource=resource)

        if endpoint:
            _otlp_spec = importlib.util.find_spec("opentelemetry.exporter.otlp.proto.grpc")
            if _otlp_spec is not None:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter  # type: ignore[import-untyped]

                exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
                provider.add_span_processor(BatchSpanProcessor(exporter))

        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer("mind_mem")
        _otel_initialized = True


def _get_tracer() -> Any:
    """Return the tracer, initialising with NoOp if not yet set up."""
    global _tracer

    if not _HAS_OTEL:
        return None

    if _tracer is None:
        with _otel_lock:
            if _tracer is None:
                from opentelemetry import trace  # type: ignore[import-untyped]

                _tracer = trace.get_tracer("mind_mem")
    return _tracer


# ---------------------------------------------------------------------------
# Metric helpers — called by the decorator and by external code
# ---------------------------------------------------------------------------


def _record_recall(duration_seconds: float) -> None:
    """Record a recall invocation in Prometheus (no-op if not installed)."""
    if not _HAS_PROM:
        return
    _init_prom_metrics()
    if _recall_total is not None:
        _recall_total.inc()
    if _recall_duration is not None:
        _recall_duration.observe(duration_seconds)


def _record_propose_update() -> None:
    if not _HAS_PROM:
        return
    _init_prom_metrics()
    if _propose_update_total is not None:
        _propose_update_total.inc()


def _record_scan() -> None:
    if not _HAS_PROM:
        return
    _init_prom_metrics()
    if _scan_total is not None:
        _scan_total.inc()


def _record_apply() -> None:
    if not _HAS_PROM:
        return
    _init_prom_metrics()
    if _apply_total is not None:
        _apply_total.inc()


def _record_apply_rollback() -> None:
    if not _HAS_PROM:
        return
    _init_prom_metrics()
    if _apply_rollback_total is not None:
        _apply_rollback_total.inc()


# ---------------------------------------------------------------------------
# @traced decorator
# ---------------------------------------------------------------------------

_F = TypeVar("_F", bound=Callable[..., Any])

# Map span names to their per-metric recorder
_METRIC_RECORDERS: dict[str, Callable[..., None]] = {
    "recall": _record_recall,
    "propose_update": lambda: _record_propose_update(),
    "scan": lambda: _record_scan(),
    "approve_apply": lambda: _record_apply(),
    "rollback_proposal": lambda: _record_apply_rollback(),
}


def traced(span_name: str) -> Callable[[_F], _F]:
    """Decorator that wraps a function in an OTel span and records Prometheus metrics.

    When neither OTel nor Prometheus is installed, the wrapped function is
    returned unmodified (zero overhead beyond the attribute lookup).

    Args:
        span_name: The OTel span name / metric label.

    Example:
        @traced("recall")
        def recall(workspace, query, ...):
            ...
    """

    def decorator(fn: _F) -> _F:
        if not _HAS_OTEL and not _HAS_PROM:
            # Fast path: absolutely no wrapping overhead
            return fn

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.monotonic()
            tracer = _get_tracer()

            if tracer is not None:
                from opentelemetry.trace import StatusCode  # type: ignore[import-untyped]

                with tracer.start_as_current_span(span_name) as span:
                    try:
                        result = fn(*args, **kwargs)
                        span.set_status(StatusCode.OK)
                        return result
                    except Exception as exc:
                        span.set_status(StatusCode.ERROR, str(exc))
                        span.record_exception(exc)
                        raise
                    finally:
                        duration = time.monotonic() - start
                        _fire_metric(span_name, duration)
            else:
                try:
                    return fn(*args, **kwargs)
                finally:
                    duration = time.monotonic() - start
                    _fire_metric(span_name, duration)

        return wrapper  # type: ignore[return-value]

    return decorator


def _fire_metric(span_name: str, duration: float) -> None:
    """Dispatch to the correct Prometheus recorder for *span_name*."""
    recorder = _METRIC_RECORDERS.get(span_name)
    if recorder is None:
        return
    if span_name == "recall":
        recorder(duration)
    else:
        recorder()
