"""The change stream's backpressure counters, and the drain that gives
them meaning.

``dropped`` / ``queue_depth`` are published to operators (the
``stream_status`` MCP tool) as a subscriber-lag signal, but nothing in
the package ever removed an event from a subscriber's queue: the
listener is called with the event itself, never from the queue. Past the
queue cap every publish therefore shed one event for every subscriber
regardless of its health, so the counters were a function of publish
volume alone — a healthy subscriber and a stalled one reported the same
numbers.

:meth:`ChangeStream.poll` is the missing half. These pin what the
counters mean now: a subscriber that drains stays at zero, and one that
does not still sheds (that is genuine backlog loss on the pull surface,
and the older overflow test rests on it), while the push surface never
drops anything either way.
"""

from __future__ import annotations

import pytest

from mind_mem.change_stream import ChangeStream


@pytest.mark.unit
def test_a_polling_subscriber_reports_no_backlog() -> None:
    stream = ChangeStream(max_queue_depth=2)
    sub = stream.subscribe(lambda ev: None)
    drained = []
    for i in range(10):
        stream.publish("block.created", {"n": i})
        drained.extend(stream.poll(sub))

    stats = stream.stats()
    assert stats.published == 10
    assert [ev.payload["n"] for ev in drained] == list(range(10))
    # The queue never filled, so nothing was shed: dropped is now a
    # statement about this subscriber, not about how much was published.
    assert stats.dropped == 0
    assert stats.queue_depth == 0


@pytest.mark.unit
def test_a_subscriber_that_falls_behind_still_sheds_and_says_so() -> None:
    stream = ChangeStream(max_queue_depth=2)
    sub = stream.subscribe(lambda ev: None)
    for i in range(5):
        stream.publish("block.created", {"n": i})

    stats = stream.stats()
    assert stats.dropped == 3  # 5 published, 2 retained
    assert stats.queue_depth == 2
    # The newest events are the ones kept, so a recovering subscriber
    # resumes from current state rather than replaying stale history.
    assert [ev.payload["n"] for ev in stream.poll(sub)] == [3, 4]
    assert stream.stats().queue_depth == 0


@pytest.mark.unit
def test_polling_is_per_subscriber() -> None:
    stream = ChangeStream()
    first = stream.subscribe(lambda ev: None)
    second = stream.subscribe(lambda ev: None)
    stream.publish("block.created", {"n": 1})

    assert len(stream.poll(first)) == 1
    assert len(stream.poll(first)) == 0  # drained
    assert len(stream.poll(second)) == 1  # its own queue is untouched


@pytest.mark.unit
def test_poll_respects_its_limit_and_keeps_the_rest() -> None:
    stream = ChangeStream()
    sub = stream.subscribe(lambda ev: None)
    for i in range(4):
        stream.publish("block.created", {"n": i})

    assert [ev.payload["n"] for ev in stream.poll(sub, 2)] == [0, 1]
    assert stream.stats().queue_depth == 2
    assert [ev.payload["n"] for ev in stream.poll(sub)] == [2, 3]


@pytest.mark.unit
def test_polling_a_gone_subscription_is_not_an_error() -> None:
    stream = ChangeStream()
    sub = stream.subscribe(lambda ev: None)
    stream.publish("block.created", {})
    stream.unsubscribe(sub)
    assert stream.poll(sub) == []
    assert stream.poll(999) == []
    assert stream.poll(-1) == []


@pytest.mark.unit
def test_every_listener_is_still_called_for_every_event() -> None:
    """The push surface drops nothing, whatever the queue counters say."""
    stream = ChangeStream(max_queue_depth=2)
    seen: list[int] = []
    stream.subscribe(lambda ev: seen.append(ev.payload["n"]))
    for i in range(10):
        stream.publish("block.created", {"n": i})

    assert seen == list(range(10))
    assert stream.stats().delivered == 10
