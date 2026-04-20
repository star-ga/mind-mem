"""Governance event fan-out (v4.0 prep).

Exposes mind-mem's governance events (``contradiction_detected``,
``block_promoted``, ``snapshot_created``, ``proposal_applied``,
``rollback_executed``, ``audit_chain_verified``) as a publish-subscribe
stream. External systems (Kafka, NATS, Redis Streams, webhook
aggregators) subscribe once instead of polling governance endpoints.

The module ships a minimal pluggable publisher interface plus two
built-in publishers:

* :class:`LoggingPublisher` — zero-dep; emits events to the
  structured logger. Always available.
* :class:`RedisStreamPublisher` — writes to a Redis stream when
  ``redis`` is importable. Cross-worker fan-out with at-least-once
  semantics via consumer groups.

Additional publishers (Kafka, NATS, SNS, custom webhook) plug in by
implementing :class:`Publisher` and registering via
:func:`register_publisher`. The module intentionally keeps the
interface tiny so downstream deployments can ship their own
publisher adapters without touching mind-mem.

Config::

    {
      "events": {
        "enabled": false,
        "publishers": ["logging", "redis"],
        "redis": {"url": "redis://localhost:6379/0", "stream": "mind-mem:events"}
      }
    }
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

from .observability import get_logger

_log = get_logger("event_fanout")


# Canonical event names — callers pass strings matching these values.
EVENT_CONTRADICTION_DETECTED = "contradiction_detected"
EVENT_BLOCK_PROMOTED = "block_promoted"
EVENT_SNAPSHOT_CREATED = "snapshot_created"
EVENT_PROPOSAL_APPLIED = "proposal_applied"
EVENT_ROLLBACK_EXECUTED = "rollback_executed"
EVENT_AUDIT_CHAIN_VERIFIED = "audit_chain_verified"
EVENT_TIER_PROMOTED = "tier_promoted"
EVENT_TIER_DEMOTED = "tier_demoted"


_CANONICAL_EVENTS: frozenset[str] = frozenset(
    {
        EVENT_CONTRADICTION_DETECTED,
        EVENT_BLOCK_PROMOTED,
        EVENT_SNAPSHOT_CREATED,
        EVENT_PROPOSAL_APPLIED,
        EVENT_ROLLBACK_EXECUTED,
        EVENT_AUDIT_CHAIN_VERIFIED,
        EVENT_TIER_PROMOTED,
        EVENT_TIER_DEMOTED,
    }
)


@dataclass
class Event:
    """A governance event.

    ``kind`` should be one of the canonical event strings above.
    Non-canonical kinds are accepted and logged as a warning — callers
    can extend the taxonomy without patching this module.
    """

    kind: str
    payload: dict[str, Any]
    workspace: str | None = None
    ts_monotonic: float = field(default_factory=time.monotonic)
    ts_wall: float = field(default_factory=time.time)

    def to_wire(self) -> dict[str, Any]:
        """JSON-safe wire format for transport publishers."""
        return {
            "kind": self.kind,
            "payload": self.payload,
            "workspace": self.workspace,
            "ts_wall": self.ts_wall,
        }


# ---------------------------------------------------------------------------
# Publisher protocol + registry
# ---------------------------------------------------------------------------


@runtime_checkable
class Publisher(Protocol):
    """Implementations publish events to a single downstream target."""

    name: str

    def publish(self, event: Event) -> None: ...

    def close(self) -> None: ...


_REGISTRY: dict[str, Callable[[dict[str, Any]], Publisher]] = {}


def register_publisher(name: str, factory: Callable[[dict[str, Any]], Publisher]) -> None:
    """Register a publisher factory. Callable takes the ``events.<name>``
    config dict and returns a :class:`Publisher` instance.
    """
    _REGISTRY[name] = factory


# ---------------------------------------------------------------------------
# Built-in publishers
# ---------------------------------------------------------------------------


class LoggingPublisher:
    """Zero-dep publisher — emits events as structured log lines."""

    name = "logging"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._level = (config or {}).get("level", "info").lower()

    def publish(self, event: Event) -> None:
        log_fn = getattr(_log, self._level, _log.info)
        log_fn("event_fanout", kind=event.kind, payload=event.payload, workspace=event.workspace)

    def close(self) -> None:  # pragma: no cover
        return None


class RedisStreamPublisher:
    """Publishes events to a Redis stream. Fails open on network errors."""

    name = "redis"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        import redis  # type: ignore

        cfg = config or {}
        self._stream = cfg.get("stream", "mind-mem:events")
        self._maxlen = int(cfg.get("maxlen", 10000))
        url = cfg.get("url", "redis://localhost:6379/0")
        self._client = redis.from_url(url, decode_responses=True, socket_timeout=1.0)

    def publish(self, event: Event) -> None:
        try:
            payload = json.dumps(event.to_wire(), default=str)
            self._client.xadd(
                self._stream,
                {"data": payload},
                maxlen=self._maxlen,
                approximate=True,
            )
        except Exception as exc:
            _log.warning("event_redis_publish_failed", error=str(exc))

    def close(self) -> None:  # pragma: no cover
        try:
            self._client.close()
        except Exception:
            return None


# Pre-register the built-ins so ``create_fanout`` can resolve them.
register_publisher("logging", LoggingPublisher)


def _redis_factory(config: dict[str, Any]) -> Publisher:
    """Deferred-import factory so ``redis`` stays an optional dep."""
    return RedisStreamPublisher(config)


register_publisher("redis", _redis_factory)


# ---------------------------------------------------------------------------
# Fanout orchestrator
# ---------------------------------------------------------------------------


class EventFanout:
    """Routes a single :class:`Event` to every configured publisher.

    Publisher failures never block the event — each publisher's error
    is logged and the next publisher runs. Callers get at-least-once
    delivery to whichever publishers are online.
    """

    def __init__(self, publishers: list[Publisher]) -> None:
        self._publishers = list(publishers)

    def publish(self, event: Event) -> None:
        if event.kind not in _CANONICAL_EVENTS:
            _log.debug("event_kind_non_canonical", kind=event.kind)
        for pub in self._publishers:
            try:
                pub.publish(event)
            except Exception as exc:
                _log.warning("event_publish_failed", publisher=pub.name, error=str(exc))

    def close(self) -> None:
        for pub in self._publishers:
            try:
                pub.close()
            except Exception:  # pragma: no cover
                continue


def create_fanout(config: dict[str, Any] | None) -> EventFanout | None:
    """Build an :class:`EventFanout` from ``events`` config.

    Returns ``None`` when events are disabled or no publishers could
    be constructed. Callers check for None and skip fan-out in that
    case — zero-cost when disabled.
    """
    if not config or not isinstance(config, dict):
        return None
    events = config.get("events", {})
    if not isinstance(events, dict) or not events.get("enabled", False):
        return None
    names = events.get("publishers") or ["logging"]
    publishers: list[Publisher] = []
    for name in names:
        factory = _REGISTRY.get(name)
        if factory is None:
            _log.warning("event_publisher_unknown", name=name)
            continue
        pub_cfg = events.get(name, {}) if isinstance(events.get(name), dict) else {}
        try:
            pub = factory(pub_cfg)
        except Exception as exc:
            _log.warning("event_publisher_build_failed", name=name, error=str(exc))
            continue
        publishers.append(pub)
    if not publishers:
        return None
    return EventFanout(publishers)


__all__ = [
    "Event",
    "EventFanout",
    "Publisher",
    "LoggingPublisher",
    "RedisStreamPublisher",
    "register_publisher",
    "create_fanout",
    "EVENT_CONTRADICTION_DETECTED",
    "EVENT_BLOCK_PROMOTED",
    "EVENT_SNAPSHOT_CREATED",
    "EVENT_PROPOSAL_APPLIED",
    "EVENT_ROLLBACK_EXECUTED",
    "EVENT_AUDIT_CHAIN_VERIFIED",
    "EVENT_TIER_PROMOTED",
    "EVENT_TIER_DEMOTED",
]
