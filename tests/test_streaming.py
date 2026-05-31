"""v3.3.0 — back-pressure-aware streaming ingest queue."""

from __future__ import annotations

import time

import pytest

from mind_mem.streaming import (
    IngestEvent,
    StreamingIngestQueue,
    _TokenBucket,
    build_queue_from_config,
)


class TestTokenBucket:
    def test_rejects_zero_rate(self) -> None:
        with pytest.raises(ValueError):
            _TokenBucket(tokens_per_second=0, burst=10)

    def test_consumes_within_burst(self) -> None:
        bucket = _TokenBucket(tokens_per_second=1, burst=3)
        assert bucket.try_consume() is True
        assert bucket.try_consume() is True
        assert bucket.try_consume() is True
        # 4th should fail — bucket empty, refill too slow to matter here.
        assert bucket.try_consume() is False

    def test_refills_over_time(self) -> None:
        bucket = _TokenBucket(tokens_per_second=1000, burst=1)
        bucket.try_consume()
        time.sleep(0.05)  # ~50 tokens worth of refill
        assert bucket.try_consume() is True


class TestEnqueue:
    def test_single_event_accepted(self) -> None:
        q = StreamingIngestQueue(capacity=3)
        r = q.enqueue(IngestEvent(payload={"hello": "world"}))
        assert r.accepted is True
        assert r.reason == "ok"
        assert len(q) == 1

    def test_full_queue_drops_oldest(self) -> None:
        q = StreamingIngestQueue(capacity=2)
        q.enqueue(IngestEvent(payload={"n": 1}))
        q.enqueue(IngestEvent(payload={"n": 2}))
        r = q.enqueue(IngestEvent(payload={"n": 3}))
        assert r.accepted is True
        assert r.reason == "queue_full_dropped_oldest"
        assert r.dropped_event is not None
        assert r.dropped_event.payload == {"n": 1}
        # Queue retains the newest two.
        drained = [e.payload for e in q.drain()]
        assert drained == [{"n": 2}, {"n": 3}]

    def test_rate_limited_rejected(self) -> None:
        bucket = _TokenBucket(tokens_per_second=1, burst=1)
        q = StreamingIngestQueue(capacity=10, rate_limit=bucket)
        first = q.enqueue(IngestEvent(payload={"n": 1}))
        second = q.enqueue(IngestEvent(payload={"n": 2}))
        assert first.accepted is True
        assert second.accepted is False
        assert second.reason == "rate_limited"
        # Rejected producers don't touch the queue.
        assert len(q) == 1


class TestDrain:
    def test_drain_returns_events_in_fifo_order(self) -> None:
        q = StreamingIngestQueue(capacity=5)
        for i in range(3):
            q.enqueue(IngestEvent(payload={"i": i}))
        events = q.drain()
        assert [e.payload["i"] for e in events] == [0, 1, 2]
        assert len(q) == 0

    def test_drain_max_items(self) -> None:
        q = StreamingIngestQueue(capacity=10)
        for i in range(5):
            q.enqueue(IngestEvent(payload={"i": i}))
        first_two = q.drain(max_items=2)
        assert [e.payload["i"] for e in first_two] == [0, 1]
        # Rest still queued.
        assert len(q) == 3

    def test_drain_iter_yields_and_stops(self) -> None:
        q = StreamingIngestQueue(capacity=5)
        q.enqueue(IngestEvent(payload={"a": 1}))
        q.enqueue(IngestEvent(payload={"b": 2}))
        seen = [e.payload for e in q.drain_iter()]
        assert seen == [{"a": 1}, {"b": 2}]
        assert len(q) == 0


class TestBuildFromConfig:
    def test_disabled_returns_none(self) -> None:
        assert build_queue_from_config(None) is None
        assert build_queue_from_config({}) is None
        assert build_queue_from_config({"streaming": {"enabled": False}}) is None

    def test_enabled_without_rate_limit(self) -> None:
        q = build_queue_from_config({"streaming": {"enabled": True, "capacity": 128}})
        assert q is not None
        assert q.capacity == 128

    def test_enabled_with_rate_limit(self) -> None:
        q = build_queue_from_config(
            {
                "streaming": {
                    "enabled": True,
                    "capacity": 64,
                    "rate_limit": {"tokens_per_second": 50, "burst": 100},
                }
            }
        )
        assert q is not None
        assert q.capacity == 64
        # Rate limit wired — burst of 100 fills cleanly.
        for i in range(100):
            r = q.enqueue(IngestEvent(payload={"i": i}))
            assert r.accepted, f"burst slot {i} rejected"

    def test_invalid_rate_limit_falls_open(self) -> None:
        """Bad rate-limit config doesn't prevent the queue from existing."""
        q = build_queue_from_config(
            {
                "streaming": {
                    "enabled": True,
                    "capacity": 4,
                    "rate_limit": {"tokens_per_second": -5, "burst": 10},
                }
            }
        )
        assert q is not None
        # Without rate-limit, all enqueues succeed until capacity hit.
        for _ in range(4):
            assert q.enqueue(IngestEvent(payload={})).accepted


class TestConcurrency:
    def test_multi_producer_safe(self) -> None:
        """Two threads enqueueing don't corrupt the deque."""
        import threading

        q = StreamingIngestQueue(capacity=2000)

        def producer(start: int) -> None:
            for i in range(start, start + 500):
                q.enqueue(IngestEvent(payload={"i": i}, client_id=f"p{start}"))

        threads = [threading.Thread(target=producer, args=(s,)) for s in (0, 500, 1000, 1500)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        drained = q.drain()
        # All 2000 events should survive (capacity is 2000, no drops).
        assert len(drained) == 2000
        ids = sorted(e.payload["i"] for e in drained)
        assert ids == list(range(2000))
