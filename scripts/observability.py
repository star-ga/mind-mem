#!/usr/bin/env python3
"""mind-mem Observability Module. Zero external deps.

Provides:
- Structured JSON logging via stdlib logging
- In-process metrics counters
- Timing context manager for latency tracking

Usage:
    from observability import get_logger, metrics, timed

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
import time
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Structured JSON Formatter
# ---------------------------------------------------------------------------

class JSONFormatter(logging.Formatter):
    """Emit log records as single-line JSON."""

    def format(self, record):
        entry = {
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "level": record.levelname.lower(),
            "component": getattr(record, "component", record.name),
            "event": record.getMessage(),
        }
        # Merge extra data passed via log.info("event", extra={...})
        if hasattr(record, "data") and record.data:
            entry["data"] = record.data
        return json.dumps(entry, default=str)


class StructuredLogger:
    """Logger that supports keyword arguments as structured data."""

    def __init__(self, name):
        self.name = name
        self._logger = logging.getLogger(f"mind-mem.{name}")
        if not self._logger.handlers:
            handler = logging.StreamHandler(sys.stderr)
            handler.setFormatter(JSONFormatter())
            self._logger.addHandler(handler)
            self._logger.setLevel(
                getattr(logging, os.environ.get("MIND_MEM_LOG_LEVEL", "INFO").upper(), logging.INFO)
            )
            self._logger.propagate = False

    def _log(self, level, event, **kwargs):
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

    def inc(self, name: str, value: int | float = 1) -> None:
        """Increment a counter."""
        self._counters[name] = self._counters.get(name, 0) + value

    def observe(self, name: str, value: float) -> None:
        """Record an observation (e.g., latency, size)."""
        if name not in self._observations:
            self._observations[name] = []
        self._observations[name].append(value)

    def get(self, name: str) -> int | float:
        """Get counter value."""
        return self._counters.get(name, 0)

    def summary(self) -> dict:
        """Return metrics summary as dict."""
        result = {"counters": dict(self._counters)}
        for name, values in self._observations.items():
            if values:
                result.setdefault("observations", {})[name] = {
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
