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
        return {
            str(k): _safe_sanitize(v, _depth + 1, _seen)
            for k, v in list(obj.items())[:200]
        }
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
            # Merge extra data passed via log.info("event", extra={...})
            if hasattr(record, "data") and record.data:
                entry["data"] = _safe_sanitize(record.data)
            return json.dumps(entry, default=str)
        except (RecursionError, ValueError, TypeError) as exc:
            # A logging formatter must never crash the caller.
            return json.dumps({
                "ts": ts,
                "level": getattr(record, "levelname", "ERROR").lower(),
                "component": getattr(record, "component", record.name),
                "event": "log_format_error",
                "data": {"error": type(exc).__name__},
            })


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
        except Exception:  # logging must never crash the caller
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
