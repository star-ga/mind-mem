#!/usr/bin/env python3
"""mind-mem Observability Module. Zero external deps.

Provides:
- Structured JSON logging via stdlib logging
- In-process metrics counters
- Timing context manager for latency tracking

Usage:
    from .observability import get_logger, metrics, timed

    log = get_logger("capture")
    log.info("scan_complete", signals=5, duration_ms=120)

    metrics.inc("signals_captured", 5)
    metrics.observe("scan_duration_ms", 120.3)

    with timed("recall_query"):
        results = recall(query)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Structured JSON Formatter
# ---------------------------------------------------------------------------


def _safe_sanitize(obj, _depth=0, _seen=None):
    """Make an arbitrary object JSON-safe without unbounded recursion.

    Structured log payloads can include caller-supplied objects that are
    deeply nested or contain reference cycles. ``json.dumps`` on those
    raises ``RecursionError`` (or hangs) — which, from inside a logging
    handler, would crash the process. This bounds depth and breaks
    cycles, replacing offending sub-trees with a short repr.
    """
    if _seen is None:
        _seen = set()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if _depth >= 12:
        return f"<max-depth {type(obj).__name__}>"
    oid = id(obj)
    if oid in _seen:
        return "<cycle>"
    # ``_seen`` is a *visited* set, not a path set: once an object has
    # been rendered it is never descended into again (subsequent
    # occurrences in a DAG render as "<cycle>"). This bounds total work
    # to O(nodes) instead of O(2**depth) for diamond-shaped graphs.
    if isinstance(obj, dict):
        _seen.add(oid)
        return {str(k): _safe_sanitize(v, _depth + 1, _seen) for k, v in list(obj.items())[:200]}
    if isinstance(obj, (list, tuple, set)):
        _seen.add(oid)
        return [_safe_sanitize(v, _depth + 1, _seen) for v in list(obj)[:200]]
    # Do NOT call str(obj)/repr(obj) here: a caller object's __str__
    # /__repr__ may itself emit a structured log, which re-enters this
    # formatter and recurses without bound. Primitives/containers are
    # handled above; everything else is rendered by type only.
    return f"<{type(obj).__name__}>"


class JSONFormatter(logging.Formatter):
    """Emit log records as single-line JSON."""

    def format(self, record):
        # Compute the timestamp once so the fallback path cannot fail
        # even if the clock call were to (the formatter must never
        # crash the caller — see _log).
        try:
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        except Exception:
            ts = "1970-01-01T00:00:00.000000Z"
        try:
            entry = {
                "ts": ts,
                "level": record.levelname.lower(),
                "component": getattr(record, "component", record.name),
                "event": record.getMessage(),
            }
            # v4.logging_context: ``StructuredLogFilter`` (installed only
            # while the flag is ON — see sync_log_context) attaches the
            # active correlation-ID / key-value stack as ``record.ctx``.
            # Nothing else sets that attribute, so with the flag OFF this
            # branch is never taken and the emitted JSON is unchanged.
            ctx = getattr(record, "ctx", None)
            if ctx:
                entry["ctx"] = _safe_sanitize(ctx)
            # Merge extra data passed via log.info("event", extra={...})
            if hasattr(record, "data") and record.data:
                entry["data"] = _safe_sanitize(record.data)
            return json.dumps(entry, default=str)
        except (RecursionError, ValueError, TypeError) as exc:
            # A logging formatter must never crash the caller.
            return json.dumps(
                {
                    "ts": ts,
                    "level": getattr(record, "levelname", "ERROR").lower(),
                    "component": getattr(record, "component", record.name),
                    "event": "log_format_error",
                    "data": {"error": type(exc).__name__},
                }
            )


class StructuredLogger:
    """Logger that supports keyword arguments as structured data."""

    def __init__(self, name):
        self.name = name
        self._logger = logging.getLogger(f"mind-mem.{name}")
        if not self._logger.handlers:
            handler = logging.StreamHandler(sys.stderr)
            handler.setFormatter(JSONFormatter())
            self._logger.addHandler(handler)
            self._logger.setLevel(getattr(logging, os.environ.get("MIND_MEM_LOG_LEVEL", "INFO").upper(), logging.INFO))
            self._logger.propagate = False
            # v4.logging_context plug point. The filter goes on OUR handler,
            # never the root logger: ``propagate = False`` two lines up means a
            # root-logger install (what the v4 module's own docstring suggests)
            # never sees a single mind-mem record.
            _register_handler(self._logger.name, handler)

    def _log(self, level, event, **kwargs):
        # A logging call must never raise into the caller. The stdlib
        # logging path swallows handler/format errors via handleError;
        # StructuredLogger drives makeRecord+handle by hand, so it must
        # provide the same guarantee explicitly (otherwise a bad payload
        # or a near-limit call stack turns logging into a process crash).
        if not self._logger.isEnabledFor(level):
            return
        try:
            record = self._logger.makeRecord(
                name=self._logger.name,
                level=level,
                fn="",
                lno=0,
                msg=event,
                args=(),
                exc_info=None,
            )
            record.component = self.name
            record.data = kwargs if kwargs else None
            self._logger.handle(record)
        except Exception:  # nosec B110 — a logger must never raise into
            # its caller; this mirrors the stdlib logging contract
            # (logging.Handler.handleError swallows formatting/emit
            # errors). Re-raising here would turn any bad log payload or
            # a near-limit call stack into a process crash.
            pass

    def debug(self, event: str, **kwargs) -> None:
        self._log(logging.DEBUG, event, **kwargs)

    def info(self, event: str, **kwargs) -> None:
        self._log(logging.INFO, event, **kwargs)

    def warning(self, event: str, **kwargs) -> None:
        self._log(logging.WARNING, event, **kwargs)

    def error(self, event: str, **kwargs) -> None:
        self._log(logging.ERROR, event, **kwargs)


def get_logger(component: str) -> StructuredLogger:
    """Get a structured logger for a component."""
    return StructuredLogger(component)


# ---------------------------------------------------------------------------
# v4 structured logging context (flag: v4.logging_context, default OFF)
# ---------------------------------------------------------------------------

#: v4 feature flag gating correlation-ID / key-value context propagation.
_LOG_CONTEXT_FLAG = "logging_context"

#: The stderr handlers this module created, keyed by logger name. Only our own
#: handlers are eligible for the context filter — see the note in
#: ``StructuredLogger.__init__``.
_owned_handlers: dict[str, logging.Handler] = {}

#: Guards ``_owned_handlers`` / ``_context_filter`` / ``_context_active``.
#: Reentrant because arming can import the flag registry, which builds a
#: logger of its own and re-enters ``_register_handler``.
_context_lock = threading.RLock()

_context_filter: logging.Filter | None = None
_context_active = False


def log_context_active() -> bool:
    """True iff the v4 structured-log context filter is currently installed.

    A pure in-memory read. Hot callers (the MCP tool decorator) use it to
    decide whether minting a correlation ID would surface anywhere at all,
    without paying a ``mind-mem.json`` read per call — so the flag-OFF path
    costs one attribute load and behaves exactly as it did before v4
    ``logging_context`` was wired.
    """
    return _context_active


def _log_context_flag_enabled() -> bool:
    """Read ``v4.logging_context`` from the active config, fail-closed and QUIET.

    Deliberately does NOT call ``feature_flags.is_enabled``. That helper logs
    ``v4_config_unreadable`` when the config will not parse, and this probe runs
    unconditionally on every new logger — so with the flag OFF and a malformed
    ``mind-mem.json`` it emitted a stderr line the unwired build does not. That
    is a default-off behaviour difference, and "flag-off is byte-identical" is
    the constraint this whole restoration is landing under.

    So: parse the one key directly, swallow everything, log nothing. A probe
    that decides whether a feature is on must not itself be observable when the
    answer is no.

    deferred: there is no config-RELOAD re-arm path. ``_context_active`` is
    re-evaluated only when a new logger name registers a handler, or on an
    explicit ``sync_log_context()``. A process that creates all its loggers
    before the config becomes readable stays un-armed and mints no correlation
    ids. Upgrade path: call ``sync_log_context()`` from the config-reload hook
    when one exists.
    """
    try:
        import json as _json

        # Reuse the CANONICAL resolver -- it honours MIND_MEM_CONFIG and the
        # workspace search order, and a second resolution path here would drift
        # from it. Only the logging is skipped, not the lookup.
        from .v4.feature_flags import _config_path

        path = _config_path()
        if not path.is_file():
            return False
        block = _json.loads(path.read_text(encoding="utf-8")).get("v4") or {}
        # Same interpretation as feature_flags.is_enabled: the flag is ON only
        # for the nested {"enabled": true} shape, never a bare truthy value.
        # Fail-closed, so a typo cannot switch it on.
        sub = block.get(_LOG_CONTEXT_FLAG)
        return isinstance(sub, dict) and sub.get("enabled") is True
    except Exception:
        # Missing, unreadable, or malformed config; a non-dict v4 block; or this
        # module still mid-import. Every one of them means "off".
        return False


def sync_log_context() -> bool:
    """(Re)evaluate ``v4.logging_context`` and install/remove the filter.

    Called once per distinct logger name, when this module creates that
    logger's handler — so a process started with the flag ON is fully armed by
    import time. Exported so a config reload can re-arm loggers that already
    exist. Returns the resulting active state.
    """
    global _context_filter, _context_active
    enabled = _log_context_flag_enabled()
    with _context_lock:
        if enabled and _context_filter is None:
            try:
                from .v4.logging_context import StructuredLogFilter
            except Exception:
                _context_active = False
                return False
            _context_filter = StructuredLogFilter()
        filt = _context_filter
        if filt is not None:
            for handler in _owned_handlers.values():
                installed = filt in handler.filters
                if enabled and not installed:
                    handler.addFilter(filt)
                elif not enabled and installed:
                    handler.removeFilter(filt)
        _context_active = enabled and filt is not None
        return _context_active


def _register_handler(logger_name: str, handler: logging.Handler) -> None:
    """Record a handler we own and arm it if the flag is ON."""
    with _context_lock:
        _owned_handlers[logger_name] = handler
    sync_log_context()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class Metrics:
    """Simple in-process metrics collector.

    Tracks counters and observations (for histograms/gauges).
    Can be dumped as JSON for external collection.
    """

    def __init__(self) -> None:
        self._counters: dict[str, int | float] = {}
        self._observations: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def inc(self, name: str, value: int | float = 1) -> None:
        """Increment a counter (thread-safe)."""
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + value

    def observe(self, name: str, value: float) -> None:
        """Record an observation (thread-safe)."""
        with self._lock:
            if name not in self._observations:
                self._observations[name] = []
            self._observations[name].append(value)

    def get(self, name: str) -> int | float:
        """Get counter value."""
        return self._counters.get(name, 0)

    def samples(self, name: str) -> list[float]:
        """A copy of the raw observations recorded under *name* (thread-safe).

        :meth:`summary` reduces each series to count/min/max/avg, which
        cannot answer a percentile question — an SLI on p99 latency needs
        the readings themselves. Returns a copy so a caller cannot mutate
        the live series, and an empty list for an unknown name.
        """
        with self._lock:
            return list(self._observations.get(name, ()))

    def summary(self) -> dict:
        """Return metrics summary as dict."""
        result: dict[str, object] = {"counters": dict(self._counters)}
        for name, values in self._observations.items():
            if values:
                obs: dict[str, object] = result.setdefault("observations", {})  # type: ignore[assignment]
                obs[name] = {
                    "count": len(values),
                    "min": min(values),
                    "max": max(values),
                    "avg": sum(values) / len(values),
                }
        return result

    def reset(self) -> None:
        """Reset all metrics."""
        self._counters.clear()
        self._observations.clear()


# Global metrics instance
metrics = Metrics()


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


@contextmanager
def timed(operation: str, logger: StructuredLogger | None = None) -> Generator[None, None, None]:
    """Context manager that times an operation and records the metric.

    Usage:
        with timed("recall_query", log):
            results = recall(query)
    """
    start = time.monotonic()
    try:
        yield
    finally:
        elapsed_ms = (time.monotonic() - start) * 1000
        metrics.observe(f"{operation}_ms", elapsed_ms)
        if logger:
            logger.debug(f"{operation}_complete", duration_ms=round(elapsed_ms, 2))
