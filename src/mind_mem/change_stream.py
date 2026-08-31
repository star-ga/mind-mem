# Copyright 2026 STARGA, Inc.
"""In-process change stream (v2.5.0).

A simple pub/sub primitive for broadcasting block / edge lifecycle
events to subscribers in the same Python process. The HTTP webhook
endpoint + cross-process bus remain deferred (they pull in aiohttp
or similar).

Event shape — intentionally minimal so downstream consumers can
evolve independently:

    {
        "type":      "block.created" | "block.updated" | "block.deleted"
                     | "edge.added" | "edge.removed" | "custom",
        "timestamp": ISO 8601 UTC,
        "payload":   <arbitrary JSON-serialisable dict>,
    }

Delivery is two-sided, and the difference matters when reading the
counters:

* **push** — ``publish`` calls every listener synchronously, in the
  publishing thread, while the bus lock is held. Every published event
  reaches every live listener; none is ever dropped on this path, and a
  slow listener DOES stall the bus (the bounded queue below does not
  protect against that — only a listener that returns promptly does).
* **pull** — the same event is also retained in the subscriber's bounded
  queue, which :meth:`ChangeStream.poll` drains. This is the half the
  ``dropped`` / ``queue_depth`` counters describe: an event that ages out
  of a full queue before it was polled is shed (oldest first) and counted.

So a subscriber that consumes by polling shows real lag in ``dropped``,
while a callback-only subscriber never polls and will accumulate shed
events as a matter of course — its consumption is the callback, not the
queue. Read ``dropped`` against ``poll`` usage, never as "events the
listener missed".
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional


@dataclass(frozen=True)
class ChangeEvent:
    """Immutable record broadcast through the change stream."""

    type: str
    timestamp: str
    payload: dict

    def as_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "timestamp": self.timestamp,
            "payload": self.payload,
        }


# Callback shape for subscribers: receives (event) and returns None.
Listener = Callable[[ChangeEvent], None]


@dataclass
class _Subscription:
    """Per-subscriber queue + stats used by :class:`ChangeStream`."""

    listener: Listener
    queue: deque
    received: int = 0
    dropped: int = 0
    listener_errors: int = 0


@dataclass(frozen=True)
class StreamStats:
    """Snapshot of stream-wide counters for observability.

    ``delivered`` counts synchronous listener invocations — the push half,
    where nothing is ever dropped. ``dropped`` and ``queue_depth`` describe
    the pull half only: events shed from a full subscriber queue before
    :meth:`ChangeStream.poll` drained it, and the events still waiting
    there. A subscriber that never polls sheds one event per publish past
    the queue cap by design, so ``dropped`` is a lag signal only for
    polling subscribers.
    """

    subscribers: int
    published: int
    delivered: int
    dropped: int
    listener_errors: int
    queue_depth: int  # sum across all subscriber queues

    def as_dict(self) -> dict[str, int]:
        return {
            "subscribers": self.subscribers,
            "published": self.published,
            "delivered": self.delivered,
            "dropped": self.dropped,
            "listener_errors": self.listener_errors,
            "queue_depth": self.queue_depth,
        }


class ChangeStream:
    """Thread-safe in-process pub/sub bus.

    Listeners register a callback; `publish` delivers the event to every
    listener under a lock, and also queues it for :meth:`poll`. Callbacks
    that raise are isolated: the exception is counted in
    ``listener_errors`` (not in ``dropped`` — the event was delivered) and
    is not propagated to other subscribers.

    Args:
        max_queue_depth: Per-subscriber backlog cap for the pull surface.
            Events beyond the cap shed the oldest element first so the
            newest data always reaches a recovering subscriber.
    """

    def __init__(self, *, max_queue_depth: int = 1024) -> None:
        if max_queue_depth < 1:
            raise ValueError("max_queue_depth must be >= 1")
        self._max_depth = int(max_queue_depth)
        self._lock = threading.RLock()
        self._subs: list[_Subscription | None] = []
        self._published = 0
        self._delivered = 0
        self._dropped = 0
        self._listener_errors = 0

    # ------------------------------------------------------------------
    # Subscribers
    # ------------------------------------------------------------------

    def subscribe(self, listener: Listener) -> int:
        """Register *listener*; the returned id addresses :meth:`poll` and
        :meth:`unsubscribe`."""
        with self._lock:
            self._subs.append(_Subscription(listener=listener, queue=deque(maxlen=self._max_depth)))
            return len(self._subs) - 1

    def unsubscribe(self, sub_id: int) -> bool:
        """Remove the subscription at *sub_id*. Idempotent."""
        with self._lock:
            if 0 <= sub_id < len(self._subs) and self._subs[sub_id] is not None:
                self._subs[sub_id] = None  # keep ids stable
                return True
            return False

    def poll(self, sub_id: int, max_events: int | None = None) -> list[ChangeEvent]:
        """Drain up to *max_events* queued events for *sub_id*, oldest first.

        The missing half of the bounded-queue design: without a drain the
        queue could only ever fill, so ``dropped`` counted one shed event
        per publish past the cap for every subscriber — a function of
        publish volume that said nothing about subscriber health. A
        subscriber that polls keeps its queue short, and a ``dropped``
        above zero then means it genuinely fell behind.

        Returns an empty list for an unknown or unsubscribed id (polling a
        subscription that is gone is not an error).
        """
        with self._lock:
            if not (0 <= sub_id < len(self._subs)):
                return []
            sub = self._subs[sub_id]
            if sub is None:
                return []
            if max_events is None:
                count = len(sub.queue)
            else:
                count = max(0, min(int(max_events), len(sub.queue)))
            return [sub.queue.popleft() for _ in range(count)]

    # ------------------------------------------------------------------
    # Publish
    # ------------------------------------------------------------------

    def publish(self, event_type: str, payload: Optional[Mapping[str, Any]] = None) -> ChangeEvent:
        """Build and broadcast an event. Returns the created event."""
        ev = ChangeEvent(
            type=event_type,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            payload=dict(payload or {}),
        )
        with self._lock:
            self._published += 1
            for sub in self._subs:
                if sub is None:
                    continue
                # deque maxlen auto-sheds; count the drop exactly once
                # per shed (before the append performs the actual shed).
                # This is a PULL-side drop: the shed event was still
                # delivered to the listener below. See the module
                # docstring — it is a lag signal only for a subscriber
                # that drains its queue with poll().
                if len(sub.queue) == sub.queue.maxlen:
                    sub.dropped += 1
                    self._dropped += 1
                sub.queue.append(ev)
                sub.received += 1
                self._delivered += 1
                try:
                    sub.listener(ev)
                except Exception:  # pragma: no cover — listener isolation
                    # A listener raising isn't a queue drop (the event
                    # was queued and delivered). Track separately so
                    # operators can distinguish backpressure from buggy
                    # subscribers.
                    sub.listener_errors += 1
                    self._listener_errors += 1
        return ev

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> StreamStats:
        with self._lock:
            active = [s for s in self._subs if s is not None]
            return StreamStats(
                subscribers=len(active),
                published=self._published,
                delivered=self._delivered,
                dropped=self._dropped,
                listener_errors=self._listener_errors,
                queue_depth=sum(len(s.queue) for s in active),
            )


__all__ = ["ChangeEvent", "ChangeStream", "StreamStats", "Listener"]
