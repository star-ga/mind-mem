# Copyright 2026 STARGA, Inc.
"""Event-driven ingestion + WAL + webhook endpoint + the governed drain.

Four pieces:

1. :class:`IngestionQueue` — bounded event queue with backpressure.
2. :class:`WriteAheadLog` — JSONL WAL so crashes don't lose un-indexed
   writes.
3. :func:`serve_webhook` — stdlib-only HTTP endpoint that POSTs to the
   queue; no aiohttp dependency.
4. **The drain consumer** (:func:`drain_once` / :func:`replay_wal`) —
   the piece that turns an accepted event into a block.

Until 5.0.1 the first three shipped with **no consumer at all**: events
reached a queue and a WAL and stopped there. That is why the door was
safe, and it is also why it did nothing. Writing the consumer is the
whole risk of this module, because a consumer is a new way for content
to enter the store.

**The drain path IS the gate.** Every event becomes a block through
exactly one funnel, :func:`_write_admitted`, which opens
``get_gate(ws).admit_block(..., tier=IngestTier.EXTERNAL_INGEST)`` and
writes inside that scope — the ``inbox.py`` pattern verbatim. That tier's
row in :data:`~mind_mem.enums.INITIAL_STATUS` mints
:attr:`~mind_mem.enums.Status.QUARANTINED`, so an ingested event is inert
on arrival: :func:`mind_mem.recall.recall` will not return it, and only a
governed release proposal (``propose_import_release`` → ``approve_apply``)
can make it servable. No new ingest tier is introduced, and no caller can
request a status — ``admit_block`` refuses any tier that mints a servable
one, so the property holds by construction rather than by this module
remembering to stamp a field.

**The whole door is flag-gated, default OFF** (``v4.ingest_serve``; see
:func:`flag_enabled`). With the flag off nothing here writes: the probe
reads the config quietly — no log line, no file created — so a flag-off
build is indistinguishable from the build that never had the consumer.

**Determinism.** A block id is the SHA3-free SHA-256 of the canonical JSON
of the event that produced it (:func:`event_block_id`) — no clock, no
counter, no randomness. Three consequences, all load-bearing:

* WAL replay after a kill is **idempotent**: the same event re-derives the
  same id, and ``write_block`` replaces in place, so "apply, crash before
  checkpoint, replay" writes the identical block rather than a duplicate.
* A producer that retries a 503 (backpressure) does not double-write.
* Nothing on the scored path reads a clock. Arrival *time* is recorded
  where it belongs — the tamper-evident chain entry the gate appends.

**Crash safety.** ``serve_webhook`` fsyncs the event into the WAL *before*
offering it to the in-memory queue, and the drain consumes **from the
WAL**, by a checkpoint offset (:meth:`WriteAheadLog.advance`), whenever a
WAL is configured. The queue is then the backpressure gauge, drained in
lockstep to release capacity. So a kill at any point loses nothing: the
event is either already applied, or still pending in the WAL. Without a
WAL the drain falls back to the queue, which is at-most-once — documented,
not silent.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import queue
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from typing import Any, Callable, Final, Iterable, Mapping, Optional

from .codepoint_sanitize import sanitize_structure

_log = logging.getLogger("mind_mem.ingestion_pipeline")


@dataclass
class IngestionStats:
    """Counters an operator watches to tell a quiet queue from a broken one.

    ``rejected`` counts events the endpoint refused outright — unknown
    path, oversized body, undecodable or non-object JSON, over-nested
    structure. It is distinct from ``backpressure_drops``, which counts
    well-formed events dropped because the queue was full.
    """

    accepted: int = 0
    rejected: int = 0
    backpressure_drops: int = 0
    applied: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "accepted": self.accepted,
            "rejected": self.rejected,
            "backpressure_drops": self.backpressure_drops,
            "applied": self.applied,
        }


class IngestionQueue:
    """Bounded event queue with explicit backpressure semantics."""

    def __init__(self, *, capacity: int = 1024) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self._q: queue.Queue = queue.Queue(maxsize=capacity)
        self._stats = IngestionStats()
        self._lock = threading.RLock()

    @property
    def depth(self) -> int:
        return self._q.qsize()

    @property
    def capacity(self) -> int:
        return self._q.maxsize

    def offer(self, event: Mapping[str, Any]) -> bool:
        """Non-blocking enqueue. Returns False when backpressure engages."""
        try:
            self._q.put_nowait(dict(event))
        except queue.Full:
            with self._lock:
                self._stats.backpressure_drops += 1
            return False
        with self._lock:
            self._stats.accepted += 1
        return True

    def reject(self, count: int = 1) -> None:
        """Record *count* refused events (malformed / oversized / unroutable).

        Callers that turn an event away before it can be offered must
        record it here — otherwise a producer emitting nothing but
        malformed events leaves ``rejected`` at 0 and the queue reads as
        merely idle.
        """
        with self._lock:
            self._stats.rejected += int(count)

    def drain(self, max_items: int = 64) -> list[dict]:
        drained: list[dict] = []
        for _ in range(max_items):
            try:
                drained.append(self._q.get_nowait())
            except queue.Empty:
                break
        return drained

    def stats(self) -> IngestionStats:
        with self._lock:
            return IngestionStats(
                accepted=self._stats.accepted,
                rejected=self._stats.rejected,
                backpressure_drops=self._stats.backpressure_drops,
                applied=self._stats.applied,
            )

    def mark_applied(self, count: int) -> None:
        with self._lock:
            self._stats.applied += int(count)


# ---------------------------------------------------------------------------
# Write-ahead log
# ---------------------------------------------------------------------------


class WriteAheadLog:
    """Append-only JSONL WAL for ingestion events.

    Every ``append`` fsyncs the file so a crash before indexing
    doesn't lose pending writes. ``replay`` yields un-applied records
    so the indexer can catch up after restart.
    """

    def __init__(self, path: str) -> None:
        self._path = os.path.abspath(path)
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        self._lock = threading.RLock()

    @property
    def path(self) -> str:
        return self._path

    def append(self, event: Mapping[str, Any]) -> None:
        with self._lock:
            with open(self._path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(event, separators=(",", ":"), default=str) + "\n")
                fh.flush()
                os.fsync(fh.fileno())

    def replay(self) -> list[dict]:
        if not os.path.isfile(self._path):
            return []
        out: list[dict] = []
        with open(self._path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    obj = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    out.append(obj)
        return out

    # ─── applied-offset checkpoint (5.0.1) ───────────────────────────
    #
    # ``replay()`` alone cannot say what a previous run already applied,
    # so a restart either re-applies the whole log or the operator
    # truncates it and hopes. Neither is crash-safe. The checkpoint is a
    # single integer — the number of *parsed records* the drain consumer
    # has finished with — written atomically beside the log.
    #
    # It is deliberately a lower bound, never an upper one: it advances
    # only AFTER the blocks are on disk, so a kill in the window between
    # the write and the checkpoint re-applies those records on restart.
    # That is safe precisely because block ids are content-addressed
    # (:func:`event_block_id`) and ``write_block`` replaces in place, so a
    # re-apply is a byte-identical rewrite rather than a duplicate.

    @property
    def checkpoint_path(self) -> str:
        """Path of the applied-offset sidecar (``<wal>.applied``)."""
        return self._path + ".applied"

    def applied_count(self) -> int:
        """Records the drain consumer has finished with. Absent → 0."""
        try:
            with open(self.checkpoint_path, "r", encoding="utf-8") as fh:
                return max(0, int(fh.read().strip() or 0))
        except (OSError, ValueError):
            # No checkpoint, an unreadable one, or a corrupt one all mean
            # the same thing to a fail-safe consumer: assume nothing was
            # applied and re-apply. Idempotent ids make that free.
            return 0

    def pending(self) -> list[dict]:
        """Records at or after the checkpoint — the un-applied tail."""
        return self.replay()[self.applied_count() :]

    def advance(self, count: int) -> int:
        """Mark *count* further records applied. Returns the new offset."""
        with self._lock:
            new_offset = self.applied_count() + max(0, int(count))
            tmp = self.checkpoint_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                fh.write(str(new_offset))
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.checkpoint_path)
            return new_offset

    def truncate(self) -> None:
        with self._lock:
            with open(self._path, "w", encoding="utf-8"):
                pass
            # The checkpoint is an offset INTO this file. Leaving it
            # behind after emptying the log would make ``pending()``
            # skip the next N records written — silent data loss, the
            # exact failure the WAL exists to prevent.
            try:
                os.unlink(self.checkpoint_path)
            except FileNotFoundError:
                pass


# ---------------------------------------------------------------------------
# HTTP webhook endpoint (stdlib-only)
# ---------------------------------------------------------------------------


class _ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def serve_webhook(
    port: int,
    ingestion: IngestionQueue,
    *,
    wal: Optional[WriteAheadLog] = None,
    host: str = "127.0.0.1",
    sanitize: bool = True,
    admit_client: Optional[Callable[[str], bool]] = None,
    client_id_header: str = "X-Client-Id",
) -> tuple[threading.Thread, Callable[[], None]]:
    """Start a stdlib HTTP server accepting POST /ingest with a JSON body.

    Returns ``(server_thread, stop_fn)``. Callers invoke ``stop_fn()``
    to shut the server down cleanly. Requests that exceed 1 MiB are
    refused with HTTP 413 so the endpoint cannot be used as a memory
    DoS vector. Every refusal (404 / 413 / 400) bumps
    ``ingestion.stats().rejected`` so a misconfigured producer is
    visible in the counters rather than looking like an idle queue.

    ``sanitize`` (default ``True``) strips invisible-Unicode codepoints
    (zero-width chars, Unicode tag chars, bidi controls — a
    prompt-injection channel) from every string in the event before it
    reaches the WAL or the queue; see
    :mod:`mind_mem.codepoint_sanitize`. Events nested deeper than 64
    levels are refused with HTTP 400. Callers wiring this from a
    workspace config should pass
    ``sanitize=is_sanitize_enabled(config)``.

    ``admit_client`` is the per-client rate-limit hook. Left ``None`` (the
    default) there is no rate-limit leg at all and this endpoint behaves
    exactly as it did before the hook existed. Passed a callable -- in
    practice :meth:`mind_mem.streaming.StreamingIngestDoor.admit_client` --
    it is consulted once per request with the value of *client_id_header*,
    falling back to the peer address when the producer sends no header, and
    a ``False`` answer refuses the request with HTTP 429 before the body
    reaches the WAL or the queue. One client's flood therefore cannot spend
    another client's allowance.

    The header is producer-supplied and unauthenticated: it identifies a
    cooperating producer for fair queuing, it does not authenticate one. A
    producer inventing a fresh id per request is not throttled by per-client
    accounting under any keying -- authenticate upstream if that matters.
    """

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # silence default
            return

        def do_POST(self) -> None:
            if self.path != "/ingest":
                ingestion.reject()
                self.send_response(404)
                self.end_headers()
                return
            length = int(self.headers.get("Content-Length", "0") or 0)
            if length > 1_048_576:
                ingestion.reject()
                self.send_response(413)
                self.end_headers()
                return
            raw = self.rfile.read(length) if length > 0 else b""
            # Rate-limit AFTER draining the body (so the 429 is delivered on a
            # clean socket rather than racing a reset) and BEFORE the WAL and
            # the queue -- a refused request must leave no trace downstream.
            if admit_client is not None:
                client = (self.headers.get(client_id_header) or "").strip() or str(self.client_address[0])
                if not admit_client(client):
                    self.send_response(429)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Retry-After", "1")
                    self.end_headers()
                    self.wfile.write(json.dumps({"accepted": False, "reason": "rate_limited"}).encode("utf-8"))
                    return
            try:
                event = json.loads(raw.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                ingestion.reject()
                self.send_response(400)
                self.end_headers()
                return
            if not isinstance(event, dict):
                ingestion.reject()
                self.send_response(400)
                self.end_headers()
                return
            if sanitize:
                try:
                    event = sanitize_structure(event)
                except ValueError:  # nesting deeper than the sanitizer's cap
                    ingestion.reject()
                    self.send_response(400)
                    self.end_headers()
                    return
            if wal is not None:
                wal.append(event)
            ok = ingestion.offer(event)
            self.send_response(202 if ok else 503)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"accepted": ok}).encode("utf-8"))

    httpd = _ThreadingHTTPServer((host, port), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    def _stop() -> None:
        httpd.shutdown()
        httpd.server_close()

    return thread, _stop


# ---------------------------------------------------------------------------
# The governed drain consumer — the door IS the gate
# ---------------------------------------------------------------------------

#: ``mind-mem.json`` sub-flag under ``v4`` that arms the whole door.
INGEST_SERVE_FLAG: Final[str] = "ingest_serve"

#: Block-id prefix. Routed to ``memory/INGEST.md`` by
#: ``block_store._BLOCK_PREFIX_MAP``; because that file lives under
#: ``memory/``, ``admissibility._releasable_id_pattern`` accepts these ids
#: in a release decision, so a governed proposal can admit them later.
INGEST_BLOCK_PREFIX: Final[str] = "INGEST"

#: Event keys searched, in order, for the block's ``Statement``.
_TEXT_FIELDS: Final[tuple[str, ...]] = ("text", "statement", "content", "body", "message")

#: Event keys searched, in order, for the producer's own timestamp. Recorded
#: verbatim as ``EventTime`` and never trusted: it is attacker-controlled,
#: it is not used for ordering, and it is not the arrival time (the gate's
#: chain entry is). Reading a clock here would break replay determinism.
_TIME_FIELDS: Final[tuple[str, ...]] = ("timestamp", "time", "ts", "event_time")

#: Matches ``serve_webhook``'s 1 MiB body cap, so the direct API cannot be
#: used to write a block the HTTP door would have refused.
_MAX_TEXT_CHARS: Final[int] = 1_048_576

_DEFAULT_ACTOR: Final[str] = "ingest-serve"
_ID_DIGEST_CHARS: Final[int] = 32


class IngestRejected(ValueError):
    """An event the drain consumer refuses to turn into a block.

    A *terminal* refusal — malformed, empty, oversized. Retrying it would
    fail identically, so the drain counts it, records the reason and moves
    the checkpoint past it. Anything that might succeed on a retry (a disk
    error, a governance refusal) is NOT this exception and propagates, so
    the checkpoint stays put and the record is re-applied on restart.
    """


@dataclass(frozen=True)
class DrainOutcome:
    """What one drain pass did. Immutable; ``rejected`` holds reasons."""

    written: tuple[str, ...] = ()
    rejected: tuple[str, ...] = ()

    @property
    def processed(self) -> int:
        return len(self.written) + len(self.rejected)

    def as_dict(self) -> dict[str, Any]:
        return {
            "written": list(self.written),
            "rejected": list(self.rejected),
            "processed": self.processed,
        }


@dataclass(frozen=True)
class IngestDoor:
    """A running webhook door plus its bound drain and stop callables."""

    queue: IngestionQueue
    wal: Optional[WriteAheadLog]
    host: str
    port: int
    drain: Callable[[], DrainOutcome]
    stop: Callable[[], None]


# ---------------------------------------------------------------------------
# Flag probe — quiet by construction
# ---------------------------------------------------------------------------


def _ambient_flag_enabled() -> bool:
    """``v4.ingest_serve`` from the AMBIENT config, read fail-closed and QUIET.

    Deliberately NOT ``feature_flags.is_enabled``: that helper logs
    ``v4_config_unreadable`` on a malformed config, so on a workspace with a
    broken ``mind-mem.json`` and the flag OFF the wired build would emit a
    line the unwired build never emitted. A probe deciding whether a feature
    is on must not be observable when the answer is no. The canonical
    resolver and the canonical ``{"enabled": true}`` interpretation are
    reused, so ``MIND_MEM_CONFIG``, the workspace search order, and the
    refusal of a bare truthy value all still apply.
    """
    try:
        from .v4.feature_flags import _config_path

        path = _config_path()
        if not path.is_file():
            return False
        data = json.loads(path.read_text(encoding="utf-8"))
        block = data.get("v4") if isinstance(data, dict) else None
        sub = block.get(INGEST_SERVE_FLAG) if isinstance(block, dict) else None
        return isinstance(sub, dict) and sub.get("enabled") is True
    except Exception:
        return False


def flag_enabled(workspace: str) -> bool:
    """``v4.ingest_serve`` state for *workspace*, ambient config as fallback.

    Reads only, logs nothing, raises nothing, creates nothing. Unset (the
    default) → ``False``: no server, no WAL, no block.
    """
    config_path = os.path.join(workspace, "mind-mem.json")
    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return _ambient_flag_enabled()
    block = data.get("v4") if isinstance(data, dict) else None
    if isinstance(block, dict):
        sub = block.get(INGEST_SERVE_FLAG)
        if isinstance(sub, dict):
            return sub.get("enabled") is True
    return _ambient_flag_enabled()


def require_ingest_serve_enabled(workspace: str) -> None:
    """Raise :class:`FeatureDisabledError` when the door is OFF."""
    if not flag_enabled(workspace):
        from .v4.feature_flags import FeatureDisabledError

        raise FeatureDisabledError(
            'mind-mem surface \'ingest_serve\' is disabled. Enable via mind-mem.json: "v4": { "ingest_serve": { "enabled": true } }'
        )


# ---------------------------------------------------------------------------
# Event → block (pure, deterministic)
# ---------------------------------------------------------------------------


def canonical_event(event: Mapping[str, Any]) -> str:
    """Canonical JSON for *event* — the pre-image of its block id."""
    return json.dumps(dict(event), sort_keys=True, separators=(",", ":"), default=str, ensure_ascii=True)


def event_block_id(event: Mapping[str, Any]) -> str:
    """Deterministic, content-addressed block id for *event*.

    No clock, no counter, no randomness — the same event always yields the
    same id, which is what makes WAL replay and producer retries idempotent
    instead of duplicating blocks.
    """
    digest = hashlib.sha256(canonical_event(event).encode("utf-8")).hexdigest()
    return f"{INGEST_BLOCK_PREFIX}-{digest[:_ID_DIGEST_CHARS]}"


def event_text(event: Mapping[str, Any]) -> str:
    """Return the event's text payload, or raise :class:`IngestRejected`."""
    for field in _TEXT_FIELDS:
        value = event.get(field)
        if isinstance(value, str) and value.strip():
            if len(value) > _MAX_TEXT_CHARS:
                raise IngestRejected(f"text field {field!r} is {len(value)} chars (max {_MAX_TEXT_CHARS})")
            return value
    raise IngestRejected(f"event carries no non-empty text field (looked for {', '.join(_TEXT_FIELDS)})")


def _one_line(value: object, *, limit: int = 200) -> str:
    """Flatten a producer-supplied scalar into one safe block-field line.

    CR / LF / NUL are stripped: these values land in ``Key: value`` lines of
    a Markdown block file, and a newline in one of them would let a producer
    forge additional fields.
    """
    text = str(value).replace("\r", " ").replace("\n", " ").replace("\x00", "")
    return text[:limit].strip()


def build_block(workspace: str, event: Mapping[str, Any]) -> dict[str, Any]:
    """Render *event* as a quarantined block dict. Pure but for sanitize config.

    The ``Status`` / ``IngestTier`` pair is the same one the inbox drop
    folder stamps. It is belt-and-braces rather than the mechanism: the
    receipt's tier is what actually decides servability, and ``write_block``
    refuses a block whose status outranks it.
    """
    from .codepoint_sanitize import sanitize_text_for_ingest
    from .importers.quarantine import QUARANTINE_STATUS, QUARANTINE_TIER, TIER_FIELD

    if not isinstance(event, Mapping):
        raise IngestRejected(f"event must be a JSON object, got {type(event).__name__}")

    text = event_text(event)
    block_id = event_block_id(event)
    source = _one_line(event.get("source") or "webhook:/ingest")
    subject = _one_line(event.get("subject") or f"Ingest: {source}")

    # Security: strip invisible-Unicode (zero-width, tag chars, bidi
    # controls) before the text becomes a block — a prompt-injection
    # channel. The HTTP door already sanitizes the whole structure; this
    # covers the direct-API and replayed-from-a-foreign-WAL paths too.
    # Applied AFTER the id is derived, so the id stays a function of the
    # event as it was received.
    statement = sanitize_text_for_ingest(text, workspace, source=source)

    block: dict[str, Any] = {
        "_id": block_id,
        "type": "INGEST_EVENT",
        "Subject": subject,
        "Statement": statement,
        "Source": source,
        # The webhook is a DROP DOOR: anything a producer POSTs is untrusted
        # input, exactly like a file dropped in the inbox. It arrives
        # quarantined and stays invisible to recall until a governance
        # release admits it.
        "Status": QUARANTINE_STATUS,
        TIER_FIELD: QUARANTINE_TIER,
    }
    for field in _TIME_FIELDS:
        value = event.get(field)
        if isinstance(value, (str, int, float)) and str(value).strip():
            block["EventTime"] = _one_line(value, limit=64)
            break
    return block


# ---------------------------------------------------------------------------
# The single write funnel
# ---------------------------------------------------------------------------


def _write_admitted(workspace: str, event: Mapping[str, Any], *, actor: str) -> str:
    """Write one event as a block INSIDE a governance admission scope.

    The only place in this module that touches a BlockStore. Every drain
    path calls it, so "the drain path is the gate" is checkable by reading
    one function rather than by auditing every caller.
    """
    from .enums import IngestTier
    from .governance_gate import get_gate
    from .pipeline_hash import stamp_transform_hash
    from .storage import get_block_store

    block = build_block(workspace, event)
    block_id = str(block["_id"])
    statement = str(block["Statement"])
    store = get_block_store(workspace)
    with get_gate(workspace).admit_block(
        action="INGEST",
        block_id=block_id,
        content=statement,
        tier=IngestTier.EXTERNAL_INGEST,
        actor=actor,
        metadata={"source": str(block.get("Source", "")), "door": "ingest-serve"},
    ):
        written_id = store.write_block(stamp_transform_hash(workspace, block))
    _log.info("ingest_event_written", extra={"block_id": written_id, "source": block.get("Source", "")})
    return str(written_id)


def write_events(workspace: str, events: Iterable[Mapping[str, Any]], *, actor: str = _DEFAULT_ACTOR) -> DrainOutcome:
    """Write every event in *events* as a quarantined block."""
    require_ingest_serve_enabled(workspace)
    written: list[str] = []
    rejected: list[str] = []
    for event in events:
        try:
            written.append(_write_admitted(workspace, event, actor=actor))
        except IngestRejected as exc:
            rejected.append(str(exc))
            _log.warning("ingest_event_rejected", extra={"reason": str(exc)})
    return DrainOutcome(tuple(written), tuple(rejected))


def replay_wal(
    workspace: str,
    wal: WriteAheadLog,
    *,
    actor: str = _DEFAULT_ACTOR,
    max_items: Optional[int] = None,
) -> DrainOutcome:
    """Apply the WAL's un-checkpointed tail, then advance the checkpoint.

    The checkpoint advances by the number of records **finished with**, in
    order, and it advances in a ``finally`` — so an unexpected failure on
    record *k* leaves the offset at *k*, and the next run resumes exactly
    there rather than skipping it.
    """
    require_ingest_serve_enabled(workspace)
    pending = wal.pending()
    if max_items is not None:
        pending = pending[: max(0, int(max_items))]
    written: list[str] = []
    rejected: list[str] = []
    finished = 0
    try:
        for event in pending:
            try:
                written.append(_write_admitted(workspace, event, actor=actor))
            except IngestRejected as exc:
                rejected.append(str(exc))
                _log.warning("ingest_event_rejected", extra={"reason": str(exc)})
            finished += 1
    finally:
        if finished:
            wal.advance(finished)
    return DrainOutcome(tuple(written), tuple(rejected))


def drain_once(
    workspace: str,
    *,
    ingestion: IngestionQueue,
    wal: Optional[WriteAheadLog] = None,
    max_items: int = 64,
    actor: str = _DEFAULT_ACTOR,
) -> DrainOutcome:
    """Run one drain pass and return what it wrote.

    With a WAL configured the WAL is the source of record and the queue is
    drained in lockstep purely to release capacity — ``serve_webhook``
    fsyncs to the WAL *before* offering to the queue, so a queue-sourced
    drain would have a window where an accepted event exists only in
    memory. Without a WAL the queue is the source, which is at-most-once.
    """
    require_ingest_serve_enabled(workspace)
    if wal is not None:
        outcome = replay_wal(workspace, wal, actor=actor, max_items=max_items)
        if outcome.processed:
            ingestion.drain(outcome.processed)
            ingestion.mark_applied(len(outcome.written))
        return outcome
    outcome = write_events(workspace, ingestion.drain(max_items), actor=actor)
    ingestion.mark_applied(len(outcome.written))
    if outcome.rejected:
        ingestion.reject(len(outcome.rejected))
    return outcome


def default_wal_path(workspace: str) -> str:
    """Where ``mm ingest-serve`` keeps its WAL when none is given."""
    return os.path.join(workspace, "memory", "ingest-wal.jsonl")


def open_ingest_door(
    workspace: str,
    *,
    port: int,
    host: str = "127.0.0.1",
    wal_path: Optional[str] = None,
    capacity: int = 1024,
    max_items: int = 64,
    actor: str = _DEFAULT_ACTOR,
    rate_limiter: Optional[Callable[[str], bool]] = None,
) -> IngestDoor:
    """Start the webhook and return it bound to a governed drain.

    Raises :class:`~mind_mem.v4.feature_flags.FeatureDisabledError` before
    binding a socket when the flag is off — a disabled door does not listen.

    ``rate_limiter`` is the per-client admission hook. Left ``None`` it is
    resolved from :func:`mind_mem.streaming.client_admission_hook`, which
    returns ``None`` unless the workspace sets ``streaming.enabled`` — so
    the default is still no rate-limit leg, and a workspace that configures
    one gets it without a second wiring step. The gate that hook belongs to
    is also handed this door's queue, which is how ``stream_status`` reports
    queue depth. Pass a callable to override, or a lambda returning ``True``
    to force the limiter off for one door.
    """
    require_ingest_serve_enabled(workspace)
    from .codepoint_sanitize import sanitize_enabled_for_workspace

    ingestion = IngestionQueue(capacity=capacity)
    wal = WriteAheadLog(wal_path) if wal_path else None
    if rate_limiter is None:
        # Late import: `streaming` reads this module for its queue type, so a
        # module-level import here would close the cycle.
        from .streaming import build_stream_gate

        gate = build_stream_gate(workspace)
        if gate is not None:
            gate.bind_queue(ingestion)
            rate_limiter = gate.admit_client
    _thread, stop = serve_webhook(
        port,
        ingestion,
        wal=wal,
        host=host,
        sanitize=sanitize_enabled_for_workspace(workspace),
        admit_client=rate_limiter,
    )

    def _drain() -> DrainOutcome:
        return drain_once(workspace, ingestion=ingestion, wal=wal, max_items=max_items, actor=actor)

    return IngestDoor(queue=ingestion, wal=wal, host=host, port=port, drain=_drain, stop=stop)


__all__ = [
    "INGEST_BLOCK_PREFIX",
    "INGEST_SERVE_FLAG",
    "DrainOutcome",
    "IngestDoor",
    "IngestRejected",
    "IngestionQueue",
    "IngestionStats",
    "WriteAheadLog",
    "build_block",
    "canonical_event",
    "default_wal_path",
    "drain_once",
    "event_block_id",
    "event_text",
    "flag_enabled",
    "open_ingest_door",
    "replay_wal",
    "require_ingest_serve_enabled",
    "serve_webhook",
    "write_events",
]
