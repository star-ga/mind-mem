"""mind-mem Telemetry — OpenTelemetry traces + Prometheus metrics.

Optional-dep module. Gracefully no-ops when opentelemetry-api,
opentelemetry-sdk or prometheus_client are not installed — the api and
the sdk are separate distributions, and each is probed separately.

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
import os
import threading
import time
from typing import Any, Callable, TypeVar

# ---------------------------------------------------------------------------
# Availability probes (zero-import-cost checks)
# ---------------------------------------------------------------------------


# ``opentelemetry.trace`` ships in the opentelemetry-api distribution and is
# all ``_get_tracer`` needs. ``init_tracing`` additionally imports
# ``opentelemetry.sdk.*``, which is a SEPARATE distribution
# (opentelemetry-sdk) and a separate pyproject requirement — api-without-sdk
# is a supported install shape and a common transitive state, so the two
# capabilities get two probes. One flag for both made ``init_tracing()``
# raise ModuleNotFoundError on an api-only host.
def _has_module(name: str) -> bool:
    """True when *name* is importable, False on ANY failure.

    ``importlib.util.find_spec`` is not a total function: it propagates
    ModuleNotFoundError when a parent package is absent, and any exception a
    custom meta-path finder raises. Calling it bare at import time turns an
    optional dependency being missing into this module failing to import at
    all -- which is strictly worse than the ``init_tracing()`` that this file
    already guards, and worse than the single-probe version it replaced.
    An availability probe must never be able to raise.
    """
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:  # noqa: BLE001 — see below
        # Deliberately broad, and the docstring's "ANY failure" now matches the
        # code. A narrower tuple contradicted the sentence directly above it:
        # find_spec executes arbitrary meta-path finders, which may raise
        # anything at all, and this runs at MODULE SCOPE — an escaping
        # exception makes the module unimportable, which is the exact failure
        # this helper exists to prevent. An availability probe that can raise
        # is not a probe.
        return False


_HAS_OTEL = _has_module("opentelemetry") and _has_module("opentelemetry.trace")
_HAS_OTEL_SDK = _HAS_OTEL and _has_module("opentelemetry.sdk.trace")
_HAS_PROM = _has_module("prometheus_client")

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
# Evaluated once on first access — _get_tracer() is on the hot path of
# every @traced function, so avoid an os.environ lookup per call.
_telemetry_disabled: bool | None = None


def init_tracing(endpoint: str | None = None) -> None:
    """Configure the global OTel tracer.

    If *endpoint* is provided and opentelemetry-exporter-otlp is installed,
    an OTLP gRPC exporter is configured targeting that endpoint.  Otherwise
    the SDK's NoOp tracer is used (zero overhead).

    No-ops when opentelemetry-sdk is absent — including the api-installed
    but sdk-missing shape, where the ``opentelemetry.sdk.*`` imports below
    would otherwise raise ModuleNotFoundError.

    Idempotent — repeated calls are no-ops.
    """
    global _tracer, _otel_initialized

    if not _HAS_OTEL_SDK:
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
            # Through _has_module, never bare. find_spec RAISES when a parent
            # package is missing, and this is the deepest dotted probe in the
            # file: a host with opentelemetry-api and -sdk but no gRPC OTLP
            # exporter (a real shape -- the `all` extra does not pull `otel`)
            # would raise ModuleNotFoundError out of a function documented to
            # no-op gracefully. This is the same defect _has_module was written
            # to fix, 120 lines above.
            if _has_module("opentelemetry.exporter.otlp.proto.grpc"):
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter  # type: ignore[import-untyped]

                exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
                provider.add_span_processor(BatchSpanProcessor(exporter))

        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer("mind_mem")
        _otel_initialized = True


def _get_tracer() -> Any:
    """Return the tracer, initialising with NoOp if not yet set up.

    Set ``MIND_MEM_DISABLE_TELEMETRY=1`` to force-disable tracing
    regardless of installed exporters. Tracing is pure instrumentation
    with no effect on retrieval results; disabling it is the supported
    configuration for benchmarks and latency-sensitive runs.
    """
    global _tracer, _telemetry_disabled

    if _telemetry_disabled is None:
        _telemetry_disabled = bool(os.environ.get("MIND_MEM_DISABLE_TELEMETRY"))
    if not _HAS_OTEL or _telemetry_disabled:
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
