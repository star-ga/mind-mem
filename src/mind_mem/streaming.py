"""Rate-limited front gate for the ingest webhook (v3.3.0, wired 5.0.1).

Two halves, and only one of them is live.

**Live: the per-client rate limiter.** :class:`_PerClientRateLimiter` fronts
``POST /ingest`` (:func:`mind_mem.ingestion_pipeline.serve_webhook`) through
:class:`StreamRateGate`, so one noisy producer cannot spend another
producer's allowance, cannot flood the queue behind it, and cannot fill the
write-ahead log on everyone else's behalf. A refused request gets HTTP 429
before its body reaches the WAL or the queue.

**This module writes nothing.** It imports no BlockStore, opens no admission
scope, and contains no ``write_block`` -- the ingest door has exactly ONE
write funnel, ``ingestion_pipeline._write_admitted``, which admits every
event under ``IngestTier.EXTERNAL_INGEST`` and therefore mints
``Status.QUARANTINED``. Streamed content lands invisible to recall and stays
that way until a human releases it through the governed propose → apply
path. A second governed writer here would have been a second thing to audit;
the gate decides *who may knock*, the funnel decides *what is stored*.

**Deprecated: this module's own queue.** :class:`StreamingIngestQueue` lost
the queue role to :class:`~mind_mem.ingestion_pipeline.IngestionQueue` -- see
the DEPRECATED note on the class. Kept, not deleted: it works, it is tested,
and deleting a working thing because a better one exists is precisely the
5.0.0 mistake this release reverses.

Policy::

    Rate limit  →  token bucket, one bucket per ``client_id``, 429 on empty
    Queue full  →  reject-new (``IngestionQueue``), never silent drop-oldest
    Every write →  ingestion_pipeline, EXTERNAL_INGEST, Status: quarantined

Config — **the gate is OFF unless ``streaming.enabled`` is true**::

    {
      "streaming": {
        "enabled": false,
        "rate_limit": {
          "tokens_per_second": 20,
          "burst": 40,
          "max_clients": 1024
        }
      }
    }

``tokens_per_second`` / ``burst`` are the allowance **each** client gets.
``max_clients`` bounds how many distinct ``client_id`` values keep their own
bucket — see :class:`_PerClientRateLimiter` for what that bound is and is
not. ``capacity`` and ``drop_policy`` are read only by the DEPRECATED
:func:`build_queue_from_config`; the live door's queue size is
``open_ingest_door(capacity=...)``, so setting them here changes nothing
about the webhook. With the flag off, :func:`build_stream_gate` returns ``None`` before
reading any other key, registers nothing and logs nothing, the webhook keeps
no rate-limit leg, and ``stream_status`` reports exactly the keys it always
reported — the flag probe itself is unobservable.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final, Iterator, Mapping

from .observability import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .ingestion_pipeline import IngestionQueue

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
    """DEPRECATED — superseded by :class:`mind_mem.ingestion_pipeline.IngestionQueue`.

    deprecated: the queue role belongs to ``IngestionQueue``; this class
    is retained for its existing API surface and tests, and is used by no
    live path — upgrade path: construct
    :class:`~mind_mem.ingestion_pipeline.IngestionQueue` — in practice via
    :func:`~mind_mem.ingestion_pipeline.open_ingest_door`, which builds one
    and binds it to the governed drain — and take this module's per-client
    rate limiting from :meth:`StreamRateGate.admit_client`.

    Why the survivor won, stated plainly so nobody re-litigates it from
    the class name alone: this queue is **drop-oldest**. When it is full
    it deletes the oldest event and returns ``accepted=True``. For a
    telemetry buffer that is the right policy — the newest sample is the
    valuable one. For a *governed store* it is data loss that reports
    itself as success, and the operator learns about it only from a log
    line. ``IngestionQueue`` is reject-new: a full queue answers ``False``
    (HTTP 503), the producer keeps its event and retries, and the
    write-ahead log holds anything already accepted across a crash.

    Not deleted, deliberately: "nothing imports it" is evidence about
    wiring, never about worth, and this class works and is tested.

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


def _parse_rate_limit(streaming: Mapping[str, Any]) -> tuple[_TokenBucket | None, int]:
    """Parse ``streaming.rate_limit`` into ``(bucket_template, max_clients)``.

    Shared by the deprecated queue and by :func:`build_rate_limiter_from_config`
    so both read one config with one set of warnings. Callers must have resolved the
    ``streaming.enabled`` flag first: this function logs, and a flag probe
    that logs is an observable probe.
    """
    rl_cfg = streaming.get("rate_limit") or {}
    bucket: _TokenBucket | None = None
    max_clients = DEFAULT_MAX_TRACKED_CLIENTS
    if isinstance(rl_cfg, Mapping) and rl_cfg:
        try:
            bucket = _TokenBucket(
                tokens_per_second=float(rl_cfg.get("tokens_per_second", 20)),
                burst=float(rl_cfg.get("burst", 40)),
            )
        except (TypeError, ValueError) as exc:
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
    return bucket, max_clients


def build_queue_from_config(config: dict[str, Any] | None) -> StreamingIngestQueue | None:
    """Construct a :class:`StreamingIngestQueue` from ``streaming`` config.

    Returns ``None`` when streaming is disabled or config is missing —
    callers fall back to synchronous ingest in that case.

    deprecated: builds the superseded drop-oldest queue — upgrade path:
    :func:`~mind_mem.ingestion_pipeline.open_ingest_door`, which builds the
    governed door over ``IngestionQueue`` with this module's rate limiter in
    front of it.
    """
    if not config or not isinstance(config, Mapping):
        return None
    streaming = config.get("streaming")
    if not isinstance(streaming, Mapping) or not streaming.get("enabled", False):
        return None
    capacity = int(streaming.get("capacity", 1024))
    bucket, max_clients = _parse_rate_limit(streaming)
    return StreamingIngestQueue(capacity=capacity, rate_limit=bucket, max_clients=max_clients)


# ---------------------------------------------------------------------------
# The front gate — per-client rate limiting for the ingest webhook
# ---------------------------------------------------------------------------
#
# This module writes NOTHING. It never imports a BlockStore, never opens an
# admission scope, and has no `write_block` anywhere in it -- deliberately,
# and it is the strongest safety property of this wiring. The webhook's ONE
# write funnel is `ingestion_pipeline._write_admitted`, which admits every
# event under `IngestTier.EXTERNAL_INGEST` (=> `Status.QUARANTINED`). A
# second governed writer here would have been a second thing to audit and a
# second thing to get wrong, so this half of the door does exactly one job:
# decide, per client, whether a request is allowed to reach that funnel.

#: Config key that arms the rate limiter. Absent / false => no gate at all.
STREAM_DOOR_FLAG: Final = "streaming.enabled"

#: Request header carrying the producer's identity for rate limiting.
CLIENT_ID_HEADER: Final = "X-Client-Id"

#: Bucket a producer that sends no id falls into.
ANONYMOUS_CLIENT: Final = "anonymous"

#: Longest client id kept. The id is producer-supplied and becomes a dict
#: key and a log field; the LRU bounds how MANY ids exist, this bounds each.
MAX_CLIENT_ID_CHARS: Final = 128


def is_stream_door_enabled(config: Mapping[str, Any] | None) -> bool:
    """Resolve the ``streaming.enabled`` flag. Pure, silent, default False.

    No logging, no env read, no clock, no file. A flag probe that leaves a
    trace is observable with the flag off, which is exactly what
    "flag-off behaviour is byte-identical" forbids.
    """
    if not isinstance(config, Mapping):
        return False
    streaming = config.get("streaming")
    if not isinstance(streaming, Mapping):
        return False
    return bool(streaming.get("enabled", False))


def _enabled_streaming_config(config: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    """The ``streaming`` sub-config when the flag is on, else ``None``.

    One narrowing helper rather than a flag check plus an unchecked re-read:
    the second lookup is where a "the flag said yes so this must be a dict"
    assumption would live, and assumptions like that are how a probe ends up
    running with the flag off.
    """
    if not is_stream_door_enabled(config):
        return None
    streaming = config.get("streaming") if isinstance(config, Mapping) else None
    return streaming if isinstance(streaming, Mapping) else None


def normalise_client_id(client_id: object) -> str:
    """Bound and flatten a producer-supplied client id.

    Producer-controlled input that becomes a dict key and a log field, so it
    is truncated and stripped of control characters -- a raw newline would
    let a producer forge extra log lines. Empty resolves to
    :data:`ANONYMOUS_CLIENT` rather than to ``""``, so "sent no id" is one
    named bucket instead of a bucket called nothing.
    """
    if not isinstance(client_id, str):
        return ANONYMOUS_CLIENT
    flattened = "".join(ch if ch.isprintable() else "_" for ch in client_id.strip())
    return flattened[:MAX_CLIENT_ID_CHARS] or ANONYMOUS_CLIENT


def build_rate_limiter_from_config(config: Mapping[str, Any] | None) -> _PerClientRateLimiter | None:
    """Build the per-client limiter, or ``None`` when there is not one.

    ``None`` means no 429 leg at all: the flag is off, or the operator
    configured no ``rate_limit`` block. The flag is resolved before anything
    is parsed, so nothing is logged on the off path.
    """
    streaming = _enabled_streaming_config(config)
    if streaming is None:
        return None
    bucket, max_clients = _parse_rate_limit(streaming)
    if bucket is None:
        return None
    return _PerClientRateLimiter(bucket, max_clients=max_clients)


@dataclass(frozen=True)
class StreamGateSnapshot:
    """Counters ``stream_status`` publishes about the ingest door.

    Counters only -- no block ids and no block content -- so a user-scope
    caller learns that content arrived and how fast, never what it said.
    """

    enabled: bool
    rate_limited: int
    tracked_clients: int
    limiter_configured: bool
    queue_depth: int | None = None
    queue_capacity: int | None = None
    accepted: int | None = None
    rejected: int | None = None
    backpressure_drops: int | None = None
    applied: int | None = None

    def as_dict(self) -> dict[str, Any]:
        """Drop the queue keys entirely when no queue is bound.

        Absent rather than ``null``: a client can tell "no queue is attached"
        from "a queue is attached and empty", which ``0`` would not.
        """
        out: dict[str, Any] = {
            "enabled": self.enabled,
            "rate_limited": self.rate_limited,
            "tracked_clients": self.tracked_clients,
            "limiter_configured": self.limiter_configured,
        }
        for key in ("queue_depth", "queue_capacity", "accepted", "rejected", "backpressure_drops", "applied"):
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        return out


class StreamRateGate:
    """Per-client rate limiting for ``POST /ingest``, plus its telemetry.

    Holds a limiter and (once bound) a read-only reference to the queue the
    webhook feeds, so ``stream_status`` can report depth. It offers no
    enqueue method and no write method on purpose: an event this gate lets
    through goes to ``ingestion_pipeline``'s queue and WAL, and reaches the
    corpus only through that module's single admitted write funnel.
    """

    def __init__(self, limiter: _PerClientRateLimiter | None = None) -> None:
        self._limiter = limiter
        self._queue: "IngestionQueue | None" = None
        self._lock = threading.Lock()
        self._rate_limited = 0

    @property
    def limiter_configured(self) -> bool:
        return self._limiter is not None

    @property
    def queue(self) -> "IngestionQueue | None":
        return self._queue

    def bind_queue(self, queue: "IngestionQueue") -> None:
        """Attach the webhook's queue so depth becomes observable."""
        self._queue = queue

    def admit_client(self, client_id: object = ANONYMOUS_CLIENT) -> bool:
        """The front door: may this client spend one request?

        ``True`` when no limiter is configured -- the gate does not invent a
        limit the operator did not ask for. Keyed per client, so a flooding
        producer exhausts its own bucket and nobody else's.
        """
        if self._limiter is None:
            return True
        client = normalise_client_id(client_id)
        if self._limiter.try_consume(client):
            return True
        with self._lock:
            self._rate_limited += 1
        _log.info("streaming_rate_limited", client=client)
        return False

    @property
    def rate_limited(self) -> int:
        """How many requests this gate has refused with 429."""
        with self._lock:
            return self._rate_limited

    def snapshot(self) -> StreamGateSnapshot:
        queue = self._queue
        if queue is None:
            return StreamGateSnapshot(
                enabled=True,
                rate_limited=self.rate_limited,
                tracked_clients=self._limiter.tracked_clients if self._limiter is not None else 0,
                limiter_configured=self.limiter_configured,
            )
        stats = queue.stats()
        return StreamGateSnapshot(
            enabled=True,
            rate_limited=self.rate_limited,
            tracked_clients=self._limiter.tracked_clients if self._limiter is not None else 0,
            limiter_configured=self.limiter_configured,
            queue_depth=queue.depth,
            queue_capacity=queue.capacity,
            accepted=stats.accepted,
            rejected=stats.rejected,
            backpressure_drops=stats.backpressure_drops,
            applied=stats.applied,
        )


def build_stream_gate(
    workspace: str,
    config: Mapping[str, Any] | None = None,
    *,
    register: bool = True,
) -> StreamRateGate | None:
    """Build the front gate from workspace config, or ``None`` if disabled.

    ``None`` is the default answer: :data:`STREAM_DOOR_FLAG` defaults to
    false, and with it false this reads no further key, opens no file,
    registers nothing and logs nothing.
    """
    if config is None:
        from .init_workspace import load_config

        config = load_config(workspace)
    limiter = build_rate_limiter_from_config(config)
    if limiter is None and _enabled_streaming_config(config) is None:
        return None
    gate = StreamRateGate(limiter)
    if register:
        register_stream_gate(gate)
    return gate


def client_admission_hook(workspace: str, config: Mapping[str, Any] | None = None) -> Any:
    """The ``admit_client`` callable for ``serve_webhook``, or ``None``.

    The single call ``ingestion_pipeline.open_ingest_door`` makes into this
    module. ``None`` (the default, flag off) leaves the webhook with no
    rate-limit leg at all, byte-identical to having no hook parameter.
    """
    gate = build_stream_gate(workspace, config)
    return None if gate is None else gate.admit_client


# ---------------------------------------------------------------------------
# Process-wide registry — how ``stream_status`` finds a running gate
# ---------------------------------------------------------------------------

_GATE_LOCK = threading.Lock()
_ACTIVE_GATE: StreamRateGate | None = None


def register_stream_gate(gate: StreamRateGate) -> None:
    """Publish *gate* as this process's gate (``stream_status`` reads it)."""
    global _ACTIVE_GATE
    with _GATE_LOCK:
        _ACTIVE_GATE = gate


def clear_stream_gate() -> None:
    """Forget the registered gate. Restores the flag-off observable state."""
    global _ACTIVE_GATE
    with _GATE_LOCK:
        _ACTIVE_GATE = None


def current_stream_gate() -> StreamRateGate | None:
    """The registered gate, or ``None`` when no door is running."""
    with _GATE_LOCK:
        return _ACTIVE_GATE


def stream_door_snapshot() -> dict[str, Any] | None:
    """Gate counters for ``stream_status``, or ``None`` when there is no gate.

    ``None`` is what keeps the flag-off contract: ``stream_status`` adds no
    key at all, so its payload stays byte-identical to the pre-5.0.1 one.
    """
    gate = current_stream_gate()
    return None if gate is None else gate.snapshot().as_dict()


__all__ = [
    "ANONYMOUS_CLIENT",
    "CLIENT_ID_HEADER",
    "EnqueueResult",
    "IngestEvent",
    "MAX_CLIENT_ID_CHARS",
    "STREAM_DOOR_FLAG",
    "StreamGateSnapshot",
    "StreamRateGate",
    "StreamingIngestQueue",
    "build_queue_from_config",
    "build_rate_limiter_from_config",
    "build_stream_gate",
    "clear_stream_gate",
    "client_admission_hook",
    "current_stream_gate",
    "is_stream_door_enabled",
    "normalise_client_id",
    "register_stream_gate",
    "stream_door_snapshot",
]
