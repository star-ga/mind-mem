"""Back-pressure-aware streaming ingest (v3.3.0).

Drop-in replacement for ``capture --stdin`` when the rate of memory
blocks coming in exceeds the rate the writer pool can process.
Implements a bounded mpsc-style queue: producers ``enqueue()`` blocks
(memories) and get immediate back-pressure signal (accepted / dropped)
instead of silently filling RAM.

Policy::

    Queue full  →  drop-oldest (keep the newest signal)
    Per-client  →  token-bucket rate limit, one bucket per ``client_id``

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
          "burst": 40,
          "max_clients": 1024
        }
      }
    }

``tokens_per_second`` / ``burst`` are the allowance **each** client gets.
``max_clients`` bounds how many distinct ``client_id`` values keep their
own bucket — see :class:`_PerClientRateLimiter` for what that bound is and
is not.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any, Iterator

from .observability import get_logger

_log = get_logger("streaming")


@dataclass
class IngestEvent:
    """One unit of work on the back-pressure queue.

    ``payload`` is the block to ingest; ``client_id`` keys the rate
    limiter (one token bucket per client) and attributes telemetry.
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


#: Default cap on how many distinct ``client_id`` values keep their own
#: bucket. See :class:`_PerClientRateLimiter`.
DEFAULT_MAX_TRACKED_CLIENTS = 1024


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

    @property
    def rate(self) -> float:
        """Refill rate in tokens per second."""
        return self._rate

    @property
    def burst(self) -> float:
        """Maximum tokens the bucket holds."""
        return self._burst

    def try_consume(self, n: float = 1.0) -> bool:
        with self._lock:
            now = time.monotonic()
            self._tokens = min(self._burst, self._tokens + (now - self._last_refill) * self._rate)
            self._last_refill = now
            if self._tokens >= n:
                self._tokens -= n
                return True
            return False


class _PerClientRateLimiter:
    """One token bucket per ``client_id``, all minted from one template.

    The queue is multi-producer. A single shared bucket therefore counts
    every producer's traffic against every other producer, so one noisy
    client rate-limits the whole fleet — the cross-client denial of
    service ``client_id`` exists to prevent. Each client gets its own
    allowance instead.

    The key space is producer-supplied, so it is bounded: past
    ``max_clients`` distinct ids the least-recently-used bucket is
    evicted. That bound is a **memory** guard, not an authentication one.
    A producer free to invent a fresh ``client_id`` per event is not
    throttled by per-client accounting at all — it never was, under any
    keying — so authenticate the id upstream if that matters.
    """

    def __init__(self, template: _TokenBucket, *, max_clients: int = DEFAULT_MAX_TRACKED_CLIENTS) -> None:
        if max_clients <= 0:
            raise ValueError("max_clients must be > 0")
        self._rate = template.rate
        self._burst = template.burst
        self._max_clients = int(max_clients)
        self._buckets: OrderedDict[str, _TokenBucket] = OrderedDict()
        self._lock = threading.Lock()

    @property
    def tracked_clients(self) -> int:
        """How many clients currently hold a bucket."""
        with self._lock:
            return len(self._buckets)

    def try_consume(self, client_id: str, n: float = 1.0) -> bool:
        with self._lock:
            bucket = self._buckets.get(client_id)
            if bucket is None:
                if len(self._buckets) >= self._max_clients:
                    self._buckets.popitem(last=False)
                bucket = _TokenBucket(self._rate, self._burst)
            self._buckets[client_id] = bucket
            self._buckets.move_to_end(client_id)
        # Consume outside the registry lock — the bucket carries its own.
        return bucket.try_consume(n)


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
        max_clients: int = DEFAULT_MAX_TRACKED_CLIENTS,
    ) -> None:
        """Build the queue.

        ``rate_limit`` is a *template*: its rate and burst become the
        allowance handed to each ``client_id`` separately, not one
        allowance shared across all producers.
        """
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self._capacity = int(capacity)
        self._queue: deque[IngestEvent] = deque()
        self._lock = threading.Lock()
        self._rate_limit = None if rate_limit is None else _PerClientRateLimiter(rate_limit, max_clients=max_clients)

    @property
    def capacity(self) -> int:
        return self._capacity

    def __len__(self) -> int:
        return len(self._queue)

    def enqueue(self, event: IngestEvent) -> EnqueueResult:
        # Rate-limit first — denied producers don't even touch the queue.
        # Keyed by client so one producer cannot spend another's allowance.
        if self._rate_limit is not None and not self._rate_limit.try_consume(event.client_id):
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
    max_clients = DEFAULT_MAX_TRACKED_CLIENTS
    if isinstance(rl_cfg, dict) and rl_cfg:
        try:
            bucket = _TokenBucket(
                tokens_per_second=float(rl_cfg.get("tokens_per_second", 20)),
                burst=float(rl_cfg.get("burst", 40)),
            )
        except ValueError as exc:
            _log.warning("streaming_rate_limit_disabled", error=str(exc))
        # Parsed separately from the bucket on purpose: a bad client cap is
        # a reason to fall back to the default cap, never a reason to drop
        # the rate limit that a valid bucket above just established.
        try:
            max_clients = int(rl_cfg.get("max_clients", DEFAULT_MAX_TRACKED_CLIENTS))
        except (TypeError, ValueError) as exc:
            _log.warning("streaming_max_clients_defaulted", error=str(exc))
            max_clients = DEFAULT_MAX_TRACKED_CLIENTS
        if max_clients <= 0:
            _log.warning("streaming_max_clients_defaulted", error=f"max_clients must be > 0, got {max_clients}")
            max_clients = DEFAULT_MAX_TRACKED_CLIENTS
    return StreamingIngestQueue(capacity=capacity, rate_limit=bucket, max_clients=max_clients)


__all__ = [
    "IngestEvent",
    "EnqueueResult",
    "StreamingIngestQueue",
    "build_queue_from_config",
]
