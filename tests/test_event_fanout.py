"""v4.0 prep — governance event fan-out."""

from __future__ import annotations

from mind_mem.event_fanout import (
    EVENT_BLOCK_PROMOTED,
    EVENT_CONTRADICTION_DETECTED,
    Event,
    EventFanout,
    LoggingPublisher,
    create_fanout,
    register_publisher,
)


class _StubPublisher:
    name = "stub"

    def __init__(self, name: str = "stub") -> None:
        self.name = name
        self.published: list[Event] = []
        self.closed = False

    def publish(self, event: Event) -> None:
        self.published.append(event)

    def close(self) -> None:
        self.closed = True


class _FailingPublisher:
    name = "failing"

    def publish(self, event: Event) -> None:
        raise RuntimeError("boom")

    def close(self) -> None:
        pass


class TestEvent:
    def test_to_wire_shape(self) -> None:
        e = Event(
            kind=EVENT_BLOCK_PROMOTED,
            payload={"block_id": "D-1", "from": "SHARED", "to": "LONG_TERM"},
            workspace="/tmp/ws",
        )
        wire = e.to_wire()
        assert wire["kind"] == EVENT_BLOCK_PROMOTED
        assert wire["payload"]["block_id"] == "D-1"
        assert wire["workspace"] == "/tmp/ws"
        assert isinstance(wire["ts_wall"], float)


class TestEventFanout:
    def test_publishes_to_all(self) -> None:
        a, b = _StubPublisher("a"), _StubPublisher("b")
        fanout = EventFanout([a, b])
        event = Event(kind=EVENT_CONTRADICTION_DETECTED, payload={"ids": ["D-1", "D-2"]})
        fanout.publish(event)
        assert len(a.published) == 1
        assert len(b.published) == 1
        assert a.published[0] is event

    def test_failing_publisher_doesnt_block_others(self) -> None:
        bad = _FailingPublisher()
        good = _StubPublisher("good")
        fanout = EventFanout([bad, good])
        fanout.publish(Event(kind=EVENT_BLOCK_PROMOTED, payload={}))
        assert len(good.published) == 1

    def test_close_closes_all(self) -> None:
        a, b = _StubPublisher("a"), _StubPublisher("b")
        EventFanout([a, b]).close()
        assert a.closed and b.closed

    def test_non_canonical_kind_still_published(self, caplog) -> None:
        """Custom event kinds warn but still fan out."""
        stub = _StubPublisher("stub")
        fanout = EventFanout([stub])
        fanout.publish(Event(kind="custom_event_kind", payload={}))
        assert len(stub.published) == 1


class TestLoggingPublisher:
    def test_publishes_to_logger(self) -> None:
        pub = LoggingPublisher()
        pub.publish(Event(kind=EVENT_BLOCK_PROMOTED, payload={"a": 1}))
        # No assertion on log output — just verify no exception.


class TestCreateFanout:
    def test_disabled_returns_none(self) -> None:
        assert create_fanout(None) is None
        assert create_fanout({}) is None
        assert create_fanout({"events": {"enabled": False}}) is None

    def test_default_logging_publisher(self) -> None:
        fanout = create_fanout({"events": {"enabled": True}})
        assert fanout is not None
        # At least the logging publisher wired in.
        assert len(fanout._publishers) == 1
        assert fanout._publishers[0].name == "logging"

    def test_unknown_publisher_skipped(self) -> None:
        """Unknown name warns, logging publisher still available."""
        fanout = create_fanout({"events": {"enabled": True, "publishers": ["logging", "not_registered"]}})
        assert fanout is not None
        assert [p.name for p in fanout._publishers] == ["logging"]

    def test_all_unknown_returns_none(self) -> None:
        """All requested publishers unresolvable → no fanout."""
        fanout = create_fanout({"events": {"enabled": True, "publishers": ["not_a_thing"]}})
        assert fanout is None

    def test_custom_publisher_via_register(self) -> None:
        captured: list[Event] = []

        class _Capturing:
            name = "capturing"

            def __init__(self, cfg):
                self.cfg = cfg

            def publish(self, e):
                captured.append(e)

            def close(self):
                pass

        register_publisher("capturing", _Capturing)
        fanout = create_fanout(
            {
                "events": {
                    "enabled": True,
                    "publishers": ["capturing"],
                    "capturing": {"greeting": "hi"},
                }
            }
        )
        assert fanout is not None
        fanout.publish(Event(kind=EVENT_BLOCK_PROMOTED, payload={"x": 1}))
        assert len(captured) == 1
        assert captured[0].payload == {"x": 1}
