"""Back-pressure-aware streaming ingest (v3.3.0).

Drop-in replacement for ``capture --stdin`` when the rate of memory
blocks coming in exceeds the rate the writer pool can process.
Implements a bounded mpsc-style queue: producers ``enqueue()`` blocks
(memories) and get immediate back-pressure signal (accepted / dropped)
instead of silently filling RAM.

Policy::

    Queue full  →  drop-oldest (keep the newest signal)
    Per-client  →  token-bucket rate limit

This runs in-process (no asyncio / threading surprises — the queue
uses a ``collections.deque`` with a ``threading.Lock``). Callers can
use it directly from the websocket handler or from a Unix-socket
producer. Not wired into the MCP entry points yet — ships as a
standalone module so operators can adopt it incrementally.

Config::

    {
      "streaming": {
        "enabled": false,
        "capacity": 1024,
        "drop_policy": "oldest",
        "rate_limit": {
          "tokens_per_second": 20,
          "burst": 40
        }
      }
    }
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterator

from .observability import get_logger

_log = get_logger("streaming")


@dataclass
class IngestEvent:
    """One unit of work on the back-pressure queue.

    ``payload`` is the block to ingest; ``client_id`` is used for
    per-client rate limiting and telemetry attribution.
    """

    payload: dict[str, Any]
    client_id: str = "anonymous"
    received_at_monotonic: float = field(default_factory=time.monotonic)


@dataclass
class EnqueueResult:
    """Outcome of an ``enqueue`` call — producers should surface this
    back to the remote side so the client adapts its send rate."""

    accepted: bool
    reason: str  # "ok" | "rate_limited" | "queue_full_dropped_oldest"
    dropped_event: IngestEvent | None = None


class _TokenBucket:
    """Minimal mpsc-safe token bucket."""

    def __init__(self, tokens_per_second: float, burst: float):
        if tokens_per_second <= 0 or burst <= 0:
            raise ValueError("tokens_per_second and burst must be > 0")
        self._rate = float(tokens_per_second)
        self._burst = float(burst)
        self._tokens = float(burst)
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def try_consume(self, n: float = 1.0) -> bool:
        with self._lock:
            now = time.monotonic()
            self._tokens = min(self._burst, self._tokens + (now - self._last_refill) * self._rate)
            self._last_refill = now
            if self._tokens >= n:
                self._tokens -= n
                return True
            return False


class StreamingIngestQueue:
    """Bounded mpsc queue with drop-oldest back-pressure.

    Thread-safe for multi-producer / single-consumer usage. Consumer
    side calls :meth:`drain` or iterates via :meth:`drain_iter`; no
    explicit ``get()`` to discourage single-item blocking reads that
    would defeat the back-pressure design.
    """

    def __init__(
        self,
        capacity: int = 1024,
        *,
        rate_limit: _TokenBucket | None = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self._capacity = int(capacity)
        self._queue: deque[IngestEvent] = deque()
        self._lock = threading.Lock()
        self._rate_limit = rate_limit

    @property
    def capacity(self) -> int:
        return self._capacity

    def __len__(self) -> int:
        return len(self._queue)

    def enqueue(self, event: IngestEvent) -> EnqueueResult:
        # Rate-limit first — denied producers don't even touch the queue.
        if self._rate_limit is not None and not self._rate_limit.try_consume():
            _log.info("streaming_rate_limited", client=event.client_id)
            return EnqueueResult(accepted=False, reason="rate_limited")

        with self._lock:
            dropped: IngestEvent | None = None
            if len(self._queue) >= self._capacity:
                dropped = self._queue.popleft()
            self._queue.append(event)

        if dropped is not None:
            _log.warning(
                "streaming_dropped_oldest",
                dropped_client=dropped.client_id,
                dropped_age_seconds=round(time.monotonic() - dropped.received_at_monotonic, 3),
                queue_capacity=self._capacity,
            )
            return EnqueueResult(accepted=True, reason="queue_full_dropped_oldest", dropped_event=dropped)
        return EnqueueResult(accepted=True, reason="ok")

    def drain(self, max_items: int | None = None) -> list[IngestEvent]:
        """Drain up to ``max_items`` events; None → everything available."""
        drained: list[IngestEvent] = []
        with self._lock:
            while self._queue and (max_items is None or len(drained) < max_items):
                drained.append(self._queue.popleft())
        if drained:
            _log.debug("streaming_drained", count=len(drained))
        return drained

    def drain_iter(self) -> Iterator[IngestEvent]:
        """Yield events as long as the queue has anything. Non-blocking."""
        while True:
            with self._lock:
                if not self._queue:
                    return
                event = self._queue.popleft()
            yield event


def build_queue_from_config(config: dict[str, Any] | None) -> StreamingIngestQueue | None:
    """Construct a :class:`StreamingIngestQueue` from ``streaming`` config.

    Returns ``None`` when streaming is disabled or config is missing —
    callers fall back to synchronous ingest in that case.
    """
    if not config or not isinstance(config, dict):
        return None
    streaming = config.get("streaming")
    if not isinstance(streaming, dict) or not streaming.get("enabled", False):
        return None
    capacity = int(streaming.get("capacity", 1024))
    rl_cfg = streaming.get("rate_limit") or {}
    bucket: _TokenBucket | None = None
    if isinstance(rl_cfg, dict) and rl_cfg:
        try:
            bucket = _TokenBucket(
                tokens_per_second=float(rl_cfg.get("tokens_per_second", 20)),
                burst=float(rl_cfg.get("burst", 40)),
            )
        except ValueError as exc:
            _log.warning("streaming_rate_limit_disabled", error=str(exc))
    return StreamingIngestQueue(capacity=capacity, rate_limit=bucket)


__all__ = [
    "IngestEvent",
    "EnqueueResult",
    "StreamingIngestQueue",
    "build_queue_from_config",
]
