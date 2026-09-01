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

Config (``mind-mem.json``), **default off**::

    {
      "events": {
        "enabled": false,
        "publishers": ["logging", "redis"],
        "redis": {"url": "redis://localhost:6379/0", "stream": "mind-mem:events"}
      }
    }

Product callers do not build a fanout themselves — they call
:func:`emit_event`, which reads the flag, reuses a per-workspace fanout and
swallows every failure. See "The product seam" below, and
:func:`scrub_payload` for the rule that keeps block content off the wire:
**payloads carry ids and hashes only, enforced by the Event type itself.**
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from collections.abc import Mapping
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


# ---------------------------------------------------------------------------
# Payload discipline — the leak guard
# ---------------------------------------------------------------------------
#
# ``LoggingPublisher`` writes ``payload`` verbatim into the structured log and
# ``RedisStreamPublisher`` JSON-dumps it onto a stream any subscriber can read.
# Both are OUTSIDE the admission gate. So a payload carrying block text is a
# content-egress hole that quarantine cannot see: the block never becomes
# recallable, and its text leaves anyway.
#
# The rule this module enforces is therefore "ids and hashes only", and it is
# enforced STRUCTURALLY — in ``Event.__post_init__``, so every Event ever
# constructed is scrubbed, whoever built it and whichever publisher it reaches.
# A convention that emit sites "should" pass ids does not survive the first
# careless caller; a type that cannot hold prose does.
#
# The rule, fail-closed:
#
#   * numbers, booleans and ``None`` pass under ANY well-formed key — a float
#     cannot carry a Statement;
#   * strings and flat lists of strings pass ONLY under an id-bearing key
#     (``_ID_BEARING_KEYS`` / ``_ID_BEARING_SUFFIXES``), and only when short
#     and single-line, which is what an id, a hash or an enum name looks like;
#   * everything else — nested dicts, long strings, multi-line strings, strings
#     under a free-form key — is DROPPED, not truncated. Truncation IS the leak:
#     the first 128 characters of a Statement is still a Statement.
#
# Dropped keys are listed under ``_dropped`` so the omission is auditable
# rather than silent.
_MAX_PAYLOAD_KEYS = 32
_MAX_TEXTUAL_VALUE_CHARS = 128
_MAX_LIST_ITEMS = 32

_PAYLOAD_KEY_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")

#: Exact key names whose values may be short text (ids, enum names, modes).
_ID_BEARING_KEYS: frozenset[str] = frozenset(
    {
        "id",
        "ids",
        "actor",
        "backend",
        "from_tier",
        "kind",
        "mode",
        "publisher",
        "reason_code",
        "scheme",
        "status",
        "tier",
        "to_tier",
    }
)

#: Key suffixes whose values may be short text.
_ID_BEARING_SUFFIXES: tuple[str, ...] = (
    "_id",
    "_ids",
    "_hash",
    "_digest",
    "_ts",
    "_tier",
    "_status",
    "_kind",
    "_code",
)


def _is_id_bearing_key(key: str) -> bool:
    """True when *key* is allowed to carry short text rather than a number."""
    return key in _ID_BEARING_KEYS or key.endswith(_ID_BEARING_SUFFIXES)


def _is_opaque_scalar(value: Any) -> bool:
    """Numbers/bools/None — shapes that cannot carry prose."""
    return value is None or isinstance(value, (bool, int, float))


def _is_id_shaped_text(value: Any) -> bool:
    """Short, single-line, control-character-free text: id / hash / enum name."""
    if not isinstance(value, str) or len(value) > _MAX_TEXTUAL_VALUE_CHARS:
        return False
    return all(ord(ch) >= 0x20 and ord(ch) != 0x7F for ch in value)


def scrub_payload(payload: Any) -> dict[str, Any]:
    """Return a copy of *payload* carrying ids and hashes only.

    Never raises and never mutates its argument. Anything it refuses is
    listed under ``_dropped`` (sorted, so the wire form is deterministic).
    """
    if not isinstance(payload, Mapping):
        return {"_dropped": ["<non_mapping_payload>"]}

    clean: dict[str, Any] = {}
    dropped: list[str] = []
    for key, value in payload.items():
        name = key if isinstance(key, str) else repr(key)
        if not isinstance(key, str) or not _PAYLOAD_KEY_RE.match(key) or len(clean) >= _MAX_PAYLOAD_KEYS:
            dropped.append(name)
            continue
        if _is_opaque_scalar(value):
            clean[key] = value
            continue
        if not _is_id_bearing_key(key):
            dropped.append(name)
            continue
        if _is_id_shaped_text(value):
            clean[key] = value
            continue
        if (
            isinstance(value, (list, tuple))
            and len(value) <= _MAX_LIST_ITEMS
            and all(_is_opaque_scalar(item) or _is_id_shaped_text(item) for item in value)
        ):
            clean[key] = list(value)
            continue
        dropped.append(name)

    if dropped:
        clean["_dropped"] = sorted(dropped)
    return clean


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

    def __post_init__(self) -> None:
        # The leak guard, applied at the type boundary rather than at the
        # call sites: an Event simply cannot hold block content, so no emit
        # site — present or future, ours or a downstream fork's — can put
        # corpus text on a log line or a Redis stream through this module.
        self.payload = scrub_payload(self.payload)

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
            except Exception as exc:  # pragma: no cover
                _log.debug("publisher_close_failed", publisher=pub.name, error=str(exc))
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


# ---------------------------------------------------------------------------
# The product seam — workspace-scoped emission
# ---------------------------------------------------------------------------
#
# The module shipped with ``create_fanout`` and no caller: a publish-subscribe
# stream nothing published to. This section is the connection, and it carries
# three properties the governance surface depends on.
#
# 1. DEFAULT OFF, and inert when off. ``emit_event`` reads the flag FIRST and
#    returns before an :class:`Event` is constructed, so with ``events.enabled``
#    unset there is no clock read, no log record, no publisher build and no
#    payload evaluation — the flag probe is not itself observable. The probe
#    deliberately parses ``mind-mem.json`` itself rather than reusing
#    ``mcp.infra.config._load_config``, because that helper LOGS on a malformed
#    config, and a flag-off build must not gain a log line it did not have.
#
# 2. FAN-OUT CANNOT FAIL A GOVERNED WRITE. ``emit_event`` swallows every
#    ``Exception``: a dead Redis, a broken third-party publisher, a config that
#    stopped parsing. An apply that succeeded must not be reported as failed
#    because a downstream subscriber was offline — the store is the source of
#    truth, the stream is a notification.
#
# 3. PAYLOADS ARE LAZY. ``payload`` may be a callable; it is invoked only after
#    the flag check passes, so a caller can hand over a digest it would rather
#    not compute on the default path.
_FANOUT_LOCK = threading.Lock()
_FANOUT_BY_WORKSPACE: dict[str, EventFanout] = {}


def _read_enabled_config(workspace: str) -> dict[str, Any] | None:
    """Return the parsed config when events are ON for *workspace*, else None.

    Silent by construction — no logging on any failure path. An unreadable or
    malformed config means "off", which is the same answer as no config at all.
    """
    try:
        with open(os.path.join(workspace, "mind-mem.json"), encoding="utf-8") as handle:
            config = json.load(handle)
    except Exception:
        return None
    if not isinstance(config, dict):
        return None
    events = config.get("events")
    if not isinstance(events, dict) or events.get("enabled") is not True:
        return None
    return config


def is_fanout_enabled(workspace: str | None) -> bool:
    """True when ``events.enabled`` is exactly ``true`` in the workspace config."""
    if not workspace:
        return False
    return _read_enabled_config(workspace) is not None


def _fanout_for(workspace: str, config: dict[str, Any]) -> EventFanout | None:
    """Cached per-workspace fanout. Publishers are built once, not per event."""
    cached = _FANOUT_BY_WORKSPACE.get(workspace)
    if cached is not None:
        return cached
    with _FANOUT_LOCK:
        cached = _FANOUT_BY_WORKSPACE.get(workspace)
        if cached is not None:
            return cached
        built = create_fanout(config)
        if built is None:
            return None
        _FANOUT_BY_WORKSPACE[workspace] = built
        return built


def reset_fanout_cache() -> None:
    """Drop every cached fanout, closing each one. For tests and shutdown.

    Publisher CONFIG is read once per workspace and cached; the ENABLED flag is
    re-read on every emit, so turning events off takes effect immediately while
    changing publishers needs this call (or a restart).
    """
    with _FANOUT_LOCK:
        stale = list(_FANOUT_BY_WORKSPACE.values())
        _FANOUT_BY_WORKSPACE.clear()
    for fanout in stale:
        try:
            fanout.close()
        except Exception:  # nosec B112  # pragma: no cover
            # Swallowed DELIBERATELY. This runs during shutdown/reset over
            # every registered fanout; one publisher whose socket is already
            # gone must not stop the others from closing, and there is no
            # caller left to report to. Narrowing the except would mean
            # enumerating every transport's failure mode (redis, http, log)
            # and would silently start propagating the one we forgot.
            continue


def emit_event(
    workspace: str | None,
    kind: str,
    payload: Mapping[str, Any] | Callable[[], Mapping[str, Any]],
) -> bool:
    """Publish one governance event. Returns True only if it was fanned out.

    NEVER raises. Callers are governed writes; a notification must not be able
    to turn a completed apply into a failure.
    """
    if not workspace:
        return False
    try:
        config = _read_enabled_config(workspace)
        if config is None:
            return False
        fanout = _fanout_for(workspace, config)
        if fanout is None:
            return False
        resolved = payload() if callable(payload) else payload
        fanout.publish(Event(kind=kind, payload=dict(resolved), workspace=workspace))
        return True
    except Exception as exc:
        try:
            _log.warning("event_emit_failed", kind=kind, error=str(exc))
        except Exception:  # nosec B110  # pragma: no cover
            # Swallowed DELIBERATELY. This is the error path OF the error
            # path: we are already handling a failed emit, and fan-out must
            # never be able to fail a governed write. A logger that raises
            # here would turn "the event did not send" into "the apply
            # crashed", which is the exact coupling this whole leg exists to
            # avoid. The caller still gets False.
            pass
        return False


__all__ = [
    "Event",
    "EventFanout",
    "Publisher",
    "LoggingPublisher",
    "RedisStreamPublisher",
    "register_publisher",
    "create_fanout",
    "emit_event",
    "is_fanout_enabled",
    "reset_fanout_cache",
    "scrub_payload",
    "EVENT_CONTRADICTION_DETECTED",
    "EVENT_BLOCK_PROMOTED",
    "EVENT_SNAPSHOT_CREATED",
    "EVENT_PROPOSAL_APPLIED",
    "EVENT_ROLLBACK_EXECUTED",
    "EVENT_AUDIT_CHAIN_VERIFIED",
    "EVENT_TIER_PROMOTED",
    "EVENT_TIER_DEMOTED",
]
