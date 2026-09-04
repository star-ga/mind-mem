"""HTTP transport adapter for mind-mem (v3.9.0 candidate).

Stdlib-only HTTP server exposing the v3.9 endpoint surface so non-MCP
clients (Slack bots, web dashboards, monitoring tools, Streamlit /
Gradio frontends) can talk to a workspace without speaking MCP. All
endpoints are 5–20-line wrappers around existing public APIs:

* ``GET  /status``           — health, memory count, last-scan timestamp
* ``POST /query``            — natural-language search
                               (wraps ``recall`` / ``hybrid_search``)
* ``GET  /memories``         — list / browse with filtering
* ``POST /consolidate``      — trigger dream cycle on demand
* ``DELETE /memories/{id}``  — remove a specific memory
* ``POST /clear``            — wipe workspace
                               (governance-protected; requires rationale)

Auth — bearer-token via ``X-MindMem-Token`` header (matches the MCP
HTTP transport convention). The token is read from
``MIND_MEM_TOKEN`` env at server startup. Localhost-only by default;
loopback binds skip auth when
``--allow-unauthenticated-localhost`` is set (matches the existing
MCP HTTP transport flag).

Body limit — every JSON-bodied endpoint refuses payloads larger than
1 MiB with HTTP 413 so the surface cannot be used as a memory-DoS
vector. (Same posture as
``ingestion_pipeline.serve_webhook``.)

Read admission — the egress half of the governance seam applies here,
not only on the MCP surface. Two rules, both structural rather than
remembered:

* **The route table is the surface.** :data:`ROUTES` is the only thing
  the dispatcher consults, and every :class:`Route` carries a ``verdict``
  with no default, so a handler cannot become reachable without someone
  deciding whether its response can carry workspace block content.
  ``tests/test_http_read_admission.py`` sweeps every content route with a
  quarantined canary and asserts the measured reach set equals the
  declared one.
* **One reader.** Corpus rows reach a handler only through
  :func:`_admitted_blocks`, which runs ``admission.admit_read`` — the same
  predicate the recall legs use — and returns the withheld count so a
  short answer is visibly short. ``store.get_all`` / ``store.get_by_id``
  are called directly by exactly three functions in this module, all of
  them on the *delete* path where reaching a withheld block is the point;
  the test enumerates them against an allowlist and fails the build on a
  fourth.

Usage::

    from mind_mem.http_transport import serve_http

    thread, stop = serve_http(
        workspace="/path/to/workspace",
        port=8765,
        token="secret",
    )
    # ... later ...
    stop()
"""

from __future__ import annotations

import hashlib
import hmac
import inspect
import json
import logging
import os
import socket
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from typing import Any, Callable

from .admission import admit_read
from .protection import AUTH_HEADER


def _corpus_encoding_response(exc: Any, workspace: str, **extra: Any) -> tuple[int, dict[str, Any]]:
    """The deliberate answer for a corpus file that is not valid UTF-8.

    500, not 4xx: the request was well formed and the fault is entirely on
    this side -- a file in the operator's own workspace holds bytes this
    server cannot read. But a *bare* 500 saying "internal block store error"
    is the wrong 500. It is indistinguishable from a defect, so an operator
    whose only problem is one legacy-encoded file has nothing to act on, and
    the traceback that would tell them lands in the server log they are not
    reading. This body names the file and what is wrong with it.

    404 would be worse than unhelpful, it would be a lie: the block may well
    be IN the file that could not be read, and "not found" for a block that
    exists is the one answer a memory product must never give.

    The path is reported relative to the workspace. Absolute paths are the
    server's filesystem layout, and the caller already knows the workspace.
    """
    path = getattr(exc, "path", "") or ""
    try:
        shown = os.path.relpath(path, workspace)
    except (ValueError, TypeError):  # different drive on Windows, or no path
        shown = os.path.basename(path)
    body: dict[str, Any] = {
        "error": "corpus file is not valid UTF-8",
        "code": "corpus_encoding",
        "file": shown,
        "detail": getattr(exc, "reason", str(exc)),
    }
    body.update(extra)
    return (500, body)


__all__ = [
    "ANONYMOUS_ACTORS",
    "CONTENT",
    "DIRECT_CALL_ACTOR",
    "HTTP_TOKEN_ACTOR_PREFIX",
    "HTTP_UNAUTHENTICATED_ACTOR",
    "MAX_BODY_BYTES",
    "NO_CONTENT",
    "ROUTES",
    "Route",
    "mutating_routes",
    "serve_http",
    "build_handler",
]


def _safe_log(value: Any, max_len: int = 200) -> str:
    """Sanitize a user-controlled value for log emission.

    Strips CR / LF / NUL so a hostile caller cannot inject log
    lines or split a single record across multiple log entries.
    Truncates to *max_len* characters so a megabyte-scale payload
    can't bloat the log file.
    """
    s = str(value)
    s = s.replace("\r", " ").replace("\n", " ").replace("\x00", "")
    if len(s) > max_len:
        s = s[: max_len - 1] + "…"
    return s


_log = logging.getLogger("mind_mem.http_transport")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_BODY_BYTES = 1_048_576  # 1 MiB cap on POST/DELETE bodies
DEFAULT_PORT = 8765
DEFAULT_HOST = "127.0.0.1"

# Per-IP sliding-window rate limit (audit S-2). The HTTP transport exposes
# expensive paths (POST /consolidate triggers a full dream cycle,
# POST /query runs vector search) so a single authenticated caller could
# rack up CPU + LLM cost in a tight loop. Defaults are intentionally
# generous; operators can tune via MIND_MEM_HTTP_RATE_*.
DEFAULT_RATE_MAX_CALLS = 120
DEFAULT_RATE_WINDOW_SECS = 60
MAX_TRACKED_CLIENTS = 1024  # LRU cap to bound limiter memory under attack


# -- Token rotation primitive (roadmap v4.0.x) -----------------------------
# ``MIND_MEM_TOKENS`` (comma-separated) supports N-of-K active tokens with
# a grace window during rotation. Old single-token deployments keep using
# ``MIND_MEM_TOKEN``; setting ``MIND_MEM_TOKENS`` overrides + extends.
#
# Rotation flow: operator runs ``mm token rotate``, which appends a new
# token to ``MIND_MEM_TOKENS`` (the new one becomes the canonical write
# token); old tokens stay valid through the grace window so in-flight
# clients don't break, then the operator removes the old entry. Server
# reads on every request (no restart needed).
def _active_tokens(fallback: str | None = None) -> list[str]:
    """Return the set of currently-active tokens.

    Reads ``MIND_MEM_TOKENS`` (comma-separated) on every call so a
    rotation via ``mm token rotate`` lands without restart. Falls
    back to ``MIND_MEM_TOKEN`` (single-token deployments) and then
    to the handler-bound *fallback* (the server's startup-time token).
    Whitespace + empty entries are stripped.
    """
    multi = os.environ.get("MIND_MEM_TOKENS", "").strip()
    if multi:
        toks = [t.strip() for t in multi.split(",") if t.strip()]
        if toks:
            return toks
    single = os.environ.get("MIND_MEM_TOKEN", "").strip()
    if single:
        return [single]
    if fallback:
        return [fallback]
    return []


# Endpoint paths — kept as constants so tests can import them.
PATH_STATUS = "/status"
PATH_QUERY = "/query"
PATH_MEMORIES = "/memories"
PATH_CONSOLIDATE = "/consolidate"
PATH_CLEAR = "/clear"
PATH_WALKTHROUGH = "/walkthrough"
_MEMORY_ID_PREFIX = "/memories/"

# v4.0.0 federation wire transport (flag-gated; requires v4.federation flag
# in mind-mem.json). Foundation primitives ship alongside (vclock, conflict
# log, MergeStrategy); these endpoints add the over-the-wire sync layer so
# two mind-mem hosts can exchange version vectors and resolve conflicts.
PATH_FED_VCLOCK = "/federation/vclock"
PATH_FED_CONFLICTS = "/federation/conflicts"
PATH_FED_WRITE = "/federation/write"
PATH_FED_RESOLVE = "/federation/resolve"
_FED_VCLOCK_PREFIX = "/federation/vclock/"

# Loopback addresses that may skip auth when the operator opts in.
_LOOPBACK_ADDRS = frozenset({"127.0.0.1", "::1", "localhost"})


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class _ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    # Bound the per-request handler thread join at shutdown so stop() is
    # deterministic and never blocks on a stuck client connection.
    block_on_close = True


# Server-startup readiness handshake (deterministic boot). HTTPServer.__init__
# binds + listens synchronously, but serve_forever()'s accept loop runs in a
# background thread that may not have been scheduled yet when serve_http()
# returns. On a loaded CI runner (notably Windows) a client that connects in
# that window can stall until its own connect timeout fires, surfacing as a
# spurious socket TimeoutError. We close the window by polling a real TCP
# connect against the listener until it accepts, before serve_http() returns.
_READY_TIMEOUT_SECS = 10.0
_READY_POLL_INTERVAL_SECS = 0.01


def _wait_until_accepting(host: str, port: int, *, deadline: float) -> None:
    """Block until ``(host, port)`` accepts a TCP connection or ``deadline``.

    Raises :class:`TimeoutError` if the listener never accepts within the
    deadline so a genuinely-broken bind fails loudly instead of leaking a
    half-started server. The probe connection is opened and immediately
    closed; it does not issue an HTTP request, so it does not perturb the
    rate limiter or any application state.
    """
    # 0.0.0.0 / :: are bind-all sentinels that are not necessarily
    # connectable as destinations; probe loopback instead.
    connect_host = host
    if host in ("", "0.0.0.0", "::"):  # nosec B104 — not a bind; remaps bind-all sentinels to loopback for the readiness probe
        connect_host = "127.0.0.1"
    last_err: OSError | None = None
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"http_transport listener on {host}:{port} did not start accepting within {_READY_TIMEOUT_SECS:.1f}s"
            ) from last_err
        try:
            with socket.create_connection((connect_host, port), timeout=min(remaining, 1.0)):
                return
        except OSError as exc:
            last_err = exc
            time.sleep(min(_READY_POLL_INTERVAL_SECS, max(remaining, 0.0)))


class _PerClientRateLimiter:
    """Bounded LRU per-client sliding-window rate limiter (audit S-2).

    Each remote address gets its own request-timestamp deque. The map is
    LRU-bounded so an attacker spamming distinct addresses can't grow
    unbounded memory. Thread-safe — protected by a single lock for the
    map and per-entry lists.
    """

    def __init__(self, max_calls: int, window_seconds: int, max_clients: int):
        from collections import OrderedDict

        self._max_calls = max_calls
        self._window = float(window_seconds)
        self._max_clients = max_clients
        self._clients: "OrderedDict[str, list[float]]" = OrderedDict()
        self._lock = threading.Lock()

    def allow(self, client_key: str) -> tuple[bool, float]:
        """Return (allowed, retry_after_seconds). retry_after is 0 when allowed."""
        now = time.monotonic()
        with self._lock:
            timestamps = self._clients.get(client_key)
            if timestamps is None:
                if len(self._clients) >= self._max_clients:
                    # Drop the least-recently-used to bound memory.
                    self._clients.popitem(last=False)
                timestamps = []
                self._clients[client_key] = timestamps
            else:
                self._clients.move_to_end(client_key)
            cutoff = now - self._window
            # Single-pass purge of expired entries.
            i = 0
            while i < len(timestamps) and timestamps[i] <= cutoff:
                i += 1
            if i:
                del timestamps[:i]
            if len(timestamps) >= self._max_calls:
                retry_after = timestamps[0] + self._window - now
                return (False, max(retry_after, 0.1))
            timestamps.append(now)
            return (True, 0.0)


def _is_loopback(host: str) -> bool:
    return host in _LOOPBACK_ADDRS


def _read_body(handler: BaseHTTPRequestHandler) -> tuple[bytes | None, int]:
    """Return ``(body, status)``. ``status`` is non-zero on error."""
    try:
        length = int(handler.headers.get("Content-Length", "0") or 0)
    except ValueError:
        return (None, 400)
    if length < 0:
        return (None, 400)
    if length > MAX_BODY_BYTES:
        return (None, 413)
    if length == 0:
        return (b"", 0)
    return (handler.rfile.read(length), 0)


def _write_json(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, sort_keys=True).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _write_status(handler: BaseHTTPRequestHandler, status: int, message: str = "") -> None:
    payload = {"status": status, "error": message} if message else {"status": status}
    _write_json(handler, status, payload)


def _parse_query_params(path: str) -> tuple[str, dict[str, str]]:
    """Split ``/foo?a=1&b=2`` into ``("/foo", {"a": "1", "b": "2"})``."""
    if "?" not in path:
        return (path, {})
    base, _, qs = path.partition("?")
    params: dict[str, str] = {}
    for chunk in qs.split("&"):
        if not chunk:
            continue
        key, _, value = chunk.partition("=")
        if key:
            params[key] = value
    return (base, params)


# ---------------------------------------------------------------------------
# The read seam — the only way a handler here obtains corpus rows
# ---------------------------------------------------------------------------


def _admitted_blocks(workspace: str, *, active_only: bool, surface: str) -> tuple[list[dict[str, Any]], int]:
    """The corpus rows this transport may serve, and how many it may not.

    ``store.get_all`` answers with everything the store can read —
    quarantined and pending blocks included — which is correct for a
    storage adapter and wrong for a transport. Every content-serving
    handler in this module therefore reads through here, and here calls
    :func:`~mind_mem.admission.admit_read`: the *same* egress decision the
    recall legs make, rather than a status check written a second time.
    A local ``if status != "active"`` would drift from it, and the first
    thing lost to drift would be the release set — an operator-approved
    readmission that admission resolves and a hand-rolled check cannot.

    ``workspace`` is passed on deliberately. It buys the two legs that
    make the answer current rather than cached: a status refresh (the
    index caches ``status`` and goes stale in the fail-OPEN direction, so
    a block quarantined after it was indexed still reads ``active``
    there) and the release set. Omitting it is the permissive direction.

    Args:
        workspace: Workspace root.
        active_only: Passed to the store. A *caller's* filter, never the
            governance decision — admission runs whatever this says, so
            ``active_only=false`` widens the listing by exactly nothing.
        surface: Recorded on the withheld metric, for the same reason
            ``admit_leg`` takes ``leg``.

    Returns:
        ``(admitted, withheld)`` — the rows a caller may be shown, order
        preserved, and the number the gate dropped. The count is
        returned rather than logged because a listing that is silently
        short reads as a complete corpus to whoever gets it.

    Raises:
        Whatever the store or the status refresh raises. Deliberately not
        swallowed: a surface that cannot confirm a status must fail
        rather than serve the copy it could not check.
    """
    from .storage import get_block_store

    store = get_block_store(workspace)
    blocks = store.get_all(active_only=active_only)
    admission = admit_read(blocks, workspace=workspace, surface=surface)
    return admission.admitted, admission.withheld


#: Fields a block summary is drawn from, most specific first.
#:
#: The listing used to read ``id``/``type``/``subject``/``timestamp``,
#: none of which is a field any store emits — blocks carry ``_id`` and the
#: canonical capitalised names (``block_store._CANONICAL_FIELD_ORDER``),
#: on the Markdown, encrypted, Postgres and sharded backends alike. Every
#: summary was therefore ``{"id": null, "type": null, ...}``: the endpoint
#: answered 200 with a list of empty shapes. The subject chain mirrors
#: ``memory_index._SUMMARY_FIELDS`` so a block's one-line summary is the
#: same string here as everywhere else it is rendered.
_SUMMARY_SUBJECT_FIELDS: tuple[str, ...] = ("Statement", "Title", "Name", "Subject", "content")
_SUMMARY_TYPE_FIELDS: tuple[str, ...] = ("Type", "type", "block_type")
_SUMMARY_TIMESTAMP_FIELDS: tuple[str, ...] = ("Timestamp", "Date", "timestamp", "_created_at")


def _first_field(block: dict[str, Any], fields: tuple[str, ...]) -> Any:
    """First present, non-empty value among *fields*, else ``None``."""
    for key in fields:
        value = block.get(key)
        if value not in (None, "", [], {}):
            return value
    return None


def _summarise(block: dict[str, Any]) -> dict[str, Any]:
    """One listing row: id, type, category, subject, timestamp.

    The wire keys are unchanged from 5.0.1 — only the fields they are
    read from are corrected — so a client that parsed the old shape keeps
    parsing this one, and starts getting values in it.
    """
    return {
        "id": block.get("_id"),
        "type": _first_field(block, _SUMMARY_TYPE_FIELDS),
        # No store emits a category field; the category *distiller* files
        # blocks into ``categories/<name>.md`` without stamping the block.
        # Kept as a declared null rather than dropped: removing a key is a
        # breaking change for a client that reads it, and inventing a
        # value from the id prefix would be reporting a guess as data.
        "category": block.get("Category"),
        "subject": _first_field(block, _SUMMARY_SUBJECT_FIELDS),
        "timestamp": _first_field(block, _SUMMARY_TIMESTAMP_FIELDS),
    }


# ---------------------------------------------------------------------------
# Endpoint handlers — pure functions of (workspace, body, params)
# ---------------------------------------------------------------------------


def _handle_status(workspace: str) -> tuple[int, dict[str, Any]]:
    """``GET /status`` — health + memory count + last-scan timestamp."""
    from .storage import get_block_store

    try:
        store = get_block_store(workspace)
        # deferred: this is the count of block-containing *artifacts*, not
        # of memories — the same list_blocks type confusion that made
        # POST /clear delete nothing (see _corpus_block_ids). Measured: a
        # 7-block corpus in one file reports memory_count 1. Not corrected
        # here because the honest count costs a full corpus parse on a
        # polled, rate-limited health endpoint, which is a cost decision
        # rather than a bug fix. Upgrade path: read the count off the
        # recall index (sqlite_index) instead of parsing the corpus.
        block_ids = store.list_blocks()
        memory_count = len(block_ids)
    except Exception as exc:
        _log.warning("status_block_store_unavailable", extra={"error": str(exc)})
        memory_count = -1

    last_scan_path = os.path.join(workspace, "intelligence", "state", "last_scan.json")
    last_scan_ts: str | None = None
    if os.path.isfile(last_scan_path):
        try:
            with open(last_scan_path, encoding="utf-8") as fh:
                last_scan_ts = json.load(fh).get("timestamp")
        except (OSError, json.JSONDecodeError):
            last_scan_ts = None

    return (
        200,
        {
            "ok": True,
            # Audit S-4: return the workspace basename only. The absolute
            # filesystem path was visible to every authenticated caller
            # (and to unauthenticated-localhost callers) — useful operator
            # info but also a free filesystem-layout disclosure for any
            # caller who reaches the endpoint. Basename keeps the
            # human-readable signal without leaking the layout.
            "workspace": os.path.basename(os.path.abspath(workspace)),
            "memory_count": memory_count,
            "last_scan_timestamp": last_scan_ts,
            "server_time": int(time.time()),
        },
    )


def _handle_query(workspace: str, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """``POST /query`` — natural-language search.

    Body schema::

        {
          "query":       "...",
          "limit"?:      int,
          "active_only"?: bool,
          "agent_id"?:    str,
          "persona"?:     "brief" | "detailed" | "technical"
        }
    """
    query = body.get("query")
    if not isinstance(query, str) or not query.strip():
        return (400, {"error": "query is required and must be a non-empty string"})
    try:
        limit = int(body.get("limit", 10))
    except (TypeError, ValueError):
        return (400, {"error": "limit must be an integer"})
    if limit < 1 or limit > 200:
        return (400, {"error": "limit must be in [1, 200]"})
    active_only = bool(body.get("active_only", False))
    agent_id = body.get("agent_id")
    if agent_id is not None and not isinstance(agent_id, str):
        return (400, {"error": "agent_id must be a string"})
    persona = body.get("persona")
    if persona is not None and not isinstance(persona, str):
        return (400, {"error": "persona must be a string"})

    from .personas import PERSONAS, PersonaError, apply_persona
    from .recall import recall as _recall

    if persona is not None and persona not in PERSONAS:
        return (
            400,
            {"error": f"unknown persona {persona!r}; must be one of {list(PERSONAS)}"},
        )

    try:
        results = _recall(
            workspace=workspace,
            query=query,
            limit=limit,
            active_only=active_only,
            agent_id=agent_id,
        )
    except Exception as exc:
        _log.error("query_failed", extra={"error": str(exc)})
        return (500, {"error": "internal recall error"})

    try:
        projected = apply_persona(results, persona)
    except PersonaError as exc:  # belt + suspenders; PERSONAS check above already gates this
        return (400, {"error": str(exc)})

    payload: dict[str, Any] = {
        "query": query,
        "results": projected,
        "count": len(projected),
        # The proof travels with the answer. This route reached the engine
        # directly until 5.0.2 and served block content that nothing recorded:
        # the attestation was bound on one caller (the MCP recall handler), so
        # "mind-mem can prove what it served" was a property of that handler
        # rather than of the product. It now calls the serving entry, which
        # derives the record and appends the served-ledger row before
        # returning, and the record is surfaced here so an HTTP client can
        # replay the run from its ``scoring_instant`` and check the served
        # digest against what it received.
        #
        # ``persona`` is a PROJECTION applied after the fact — it rewrites the
        # per-hit fields a client sees, never the set or the order — so the
        # record still commits to exactly the ids in ``results``. Deriving it
        # from the projected list instead would bind a shape rather than a
        # ranking.
        "attestation": getattr(results, "attestation", None),
    }
    if persona is not None:
        payload["persona"] = persona
    return (200, payload)


def _handle_list_memories(workspace: str, params: dict[str, str]) -> tuple[int, dict[str, Any]]:
    """``GET /memories?limit=N&active_only=true`` — the admitted listing.

    Reads through :func:`_admitted_blocks`, so a quarantined or pending
    block cannot leave through this door whatever ``active_only`` says.
    That parameter is a caller's convenience filter and was never a
    governance control: it defaults to ``false``, and this endpoint
    served ``get_all(active_only=False)`` straight onto the wire.

    Three counts, because collapsing them hides the interesting one:

    ``count``
        rows in this response, after ``limit``.
    ``total``
        rows the caller may see, before ``limit`` — the admitted set, not
        the corpus. It used to be the corpus size, which quietly told
        every caller how many blocks were being withheld from them.
    ``withheld``
        rows admission dropped. Always present, including as ``0``: a key
        that appears only when something was held back is a key readers
        learn to ignore, and "silently short" is the failure this endpoint
        had.

    There is no ``include_withheld`` parameter and adding one would put
    the leak back behind a keyword — the full-fidelity read is
    ``snapshot()``, which is not a transport concern.
    """
    try:
        limit = int(params.get("limit", "100"))
    except ValueError:
        return (400, {"error": "limit must be an integer"})
    if limit < 1 or limit > 1000:
        return (400, {"error": "limit must be in [1, 1000]"})
    active_only_str = params.get("active_only", "false").lower()
    active_only = active_only_str in ("1", "true", "yes")

    try:
        blocks, withheld = _admitted_blocks(workspace, active_only=active_only, surface="http:GET /memories")
    except Exception as exc:
        _log.error("list_memories_failed", extra={"error": _safe_log(exc)})
        return (500, {"error": "internal block store error"})

    summaries = [_summarise(b) for b in blocks[:limit]]
    return (
        200,
        {
            "count": len(summaries),
            "total": len(blocks),
            "withheld": withheld,
            "memories": summaries,
        },
    )


def _handle_walkthrough(workspace: str, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """``POST /walkthrough`` — dependency-ordered walkthrough.

    Body schema::

        {"topic": "...", "limit"?: int, "active_only"?: bool, "agent_id"?: str}
    """
    topic = body.get("topic")
    if not isinstance(topic, str) or not topic.strip():
        return (400, {"error": "topic is required and must be a non-empty string"})
    try:
        limit = int(body.get("limit", 25))
    except (TypeError, ValueError):
        return (400, {"error": "limit must be an integer"})
    if limit < 1 or limit > 100:
        return (400, {"error": "limit must be in [1, 100]"})
    active_only = bool(body.get("active_only", False))
    agent_id = body.get("agent_id")
    if agent_id is not None and not isinstance(agent_id, str):
        return (400, {"error": "agent_id must be a string"})

    from .walkthrough import compile_walkthrough

    try:
        steps = compile_walkthrough(
            workspace=workspace,
            topic=topic,
            limit=limit,
            active_only=active_only,
            agent_id=agent_id,
        )
    except Exception as exc:
        _log.error("walkthrough_failed", extra={"error": str(exc)})
        return (500, {"error": "internal walkthrough error"})

    return (200, {"topic": topic, "steps": steps, "count": len(steps)})


def _handle_consolidate(workspace: str, body: dict[str, Any], *, actor: str) -> tuple[int, dict[str, Any]]:
    """``POST /consolidate`` — trigger dream cycle.

    Args:
        actor: The door identity ``_dispatch`` derived from the
            credential that passed auth. Keyword-only with **no
            default**, so a caller cannot reach this route's mutating
            work without naming itself.
    """
    dry_run = bool(body.get("dry_run", False))
    auto_repair = bool(body.get("auto_repair", False))

    # Attribution reaches the server log even where it cannot yet reach
    # the chain: ``auto_repair`` rewrites blocks through writers this
    # module does not own.
    # deferred: would pass ``actor`` into ``run_dream_cycle`` so the gate
    # scopes it opens carry the door identity rather than the contextvar
    # fallback; stubbed because ``dream_cycle`` is outside this module's
    # change — upgrade path: an ``actor`` keyword on ``run_dream_cycle``
    # threaded to every ``admit_*`` call it makes.
    _log.info(
        "consolidate_requested",
        extra={"actor": _safe_log(actor), "dry_run": dry_run, "auto_repair": auto_repair},
    )

    from .dream_cycle import run_dream_cycle

    try:
        report = run_dream_cycle(workspace, dry_run=dry_run, auto_repair=auto_repair)
    except Exception as exc:
        _log.error("consolidate_failed", extra={"error": str(exc)})
        return (500, {"error": "internal dream cycle error"})

    summary: dict[str, Any] = {
        "ok": True,
        "dry_run": dry_run,
        "auto_repair": auto_repair,
    }
    for attr in ("entity_proposals", "broken_citations", "stale_blocks", "consolidation_proposals", "errors"):
        value = getattr(report, attr, None)
        if value is not None:
            try:
                summary[attr] = len(value)
            except TypeError:
                summary[attr] = value
    return (200, summary)


_BLOCK_ID_MAX = 256

#: Rationale recorded for a ``DELETE /memories/{id}`` that carried none.
#: The route takes no body, so the record names the door rather than
#: claiming a reason nobody gave. Kept as a constant so the audit record
#: and the tests read the same string.
DEFAULT_DELETE_RATIONALE = "http-delete"

#: Prefix of the actor recorded for a request that passed bearer-token
#: auth; the suffix is ``sha256(token)[:12]``. The credential's identity
#: without the credential — an auditor can group every act under one
#: token, and rotating that token does not rewrite what it already did.
HTTP_TOKEN_ACTOR_PREFIX = "http:tok:"

#: Actor recorded when the operator started the server with
#: ``--allow-unauthenticated-localhost`` on a loopback bind. There is no
#: credential to name, so the record names *that* rather than borrowing a
#: word that reads like a failed lookup.
HTTP_UNAUTHENTICATED_ACTOR = "http:loopback-unauthenticated"

#: Actor recorded when a handler is called in-process rather than served
#: through the dispatcher — a library caller or a test. Every mutating
#: route is handed a door identity by ``_dispatch``, and
#: :meth:`Route.__post_init__` refuses at import to route a mutating
#: handler that cannot take one, so this value can only appear on a
#: direct call. It says that, instead of saying nothing.
DIRECT_CALL_ACTOR = "http:in-process"

#: Strings that name nobody. ``""`` reaches the gate's contextvar
#: fallback, which is ``"anonymous"`` when the REST layer is importable
#: and ``"system"`` when it is not. Measured on 5.0.1: a governed INGEST
#: write recorded ``actor="ingest-door"`` and the ``DELETE`` that removed
#: the same block recorded ``actor="anonymous"`` on both phase rows — the
#: transport authenticated a token and then threw the identity away. A
#: delete attributed to one of these is an unattributed delete, so the
#: doors refuse it rather than record it.
ANONYMOUS_ACTORS: frozenset[str] = frozenset({"", "anonymous", "system"})


def _token_actor(token_value: str) -> str:
    """The audit identity of the bearer token that just passed auth.

    A truncated digest, not the token: an evidence chain is readable by
    everyone who can read the workspace, and a credential that lands in
    it stops being a credential.
    """
    digest = hashlib.sha256(token_value.encode("utf-8")).hexdigest()[:12]
    return f"{HTTP_TOKEN_ACTOR_PREFIX}{digest}"


def _is_named_actor(actor: str) -> bool:
    """True iff *actor* names someone — see :data:`ANONYMOUS_ACTORS`."""
    return actor.strip() not in ANONYMOUS_ACTORS


def _corpus_block_ids(store: Any) -> tuple[list[str], int]:
    """Every block id in the corpus, in a stable order, plus a shortfall count.

    ``list_blocks()`` is **not** this list, and using it as one is the
    defect this function exists to remove. Every store in the tree
    implements ``list_blocks`` exactly as the ``BlockStore`` protocol
    documents it — the set of block-*containing artifacts*: ``.md`` paths
    on the Markdown and encrypted backends, distinct ``file_path`` values
    on Postgres, their union on the sharded store. ``POST /clear`` handed
    those paths to ``delete_block``, which resolves a path to no block at
    all, so the endpoint answered ``{"ok": true, "deleted": 0}`` with the
    corpus fully intact — a wipe that reported success and removed
    nothing (verified by live probe on a real Markdown corpus, and
    present in 5.0.1 before the delete scope existed). Once the scope
    landed it got worse in one specific way: the door minted a DELETE
    authorisation over a set of *filenames*, so the chain carried a
    receipt for a death that never happened — the mirror image of the
    ungated delete this release closed.

    ``get_all`` is on the protocol and implemented by all five stores, so
    this reads ids the same way whatever backend is configured.
    ``active_only=False`` is load-bearing: a clear that walked past
    quarantined and pending blocks would be a *partial* purge reported as
    a whole one, which is the same lie in a smaller box.

    Returns:
        ``(ids, unidentified)`` — ids deduplicated with first-seen order
        preserved, and the number of parsed blocks carrying no ``_id``.
        Those cannot be deleted by id and are counted rather than
        dropped, so a wipe that could not reach everything says so
        instead of reporting a whole purge it did not perform.

    Raises:
        Whatever ``get_all`` raises. Deliberately not swallowed: a store
        that cannot enumerate its corpus must fail the clear, because the
        alternative is the ``deleted: 0`` success this replaces.
    """
    ids: list[str] = []
    seen: set[str] = set()
    unidentified = 0
    for block in store.get_all(active_only=False):
        raw = block.get("_id") if hasattr(block, "get") else None
        bid = str(raw) if raw else ""
        if not bid:
            unidentified += 1
            continue
        if bid in seen:
            continue
        seen.add(bid)
        ids.append(bid)
    return ids, unidentified


def _clear_batch_id(block_ids: list[str]) -> str:
    """A stable subject id for one ``POST /clear`` decision.

    Derived from the frozen id set, not from a clock: the evidence
    record already carries its own timestamp, and a subject id that
    changes between two identical calls tells an auditor nothing while
    making the record impossible to reproduce.
    """
    digest = hashlib.sha256("\n".join(block_ids).encode("utf-8")).hexdigest()
    return f"clear-{digest[:16]}"


def _valid_block_id(block_id: str) -> bool:
    """Audit S-5: shared guard for any user-supplied block_id reaching
    the storage layer. Refuses empty, path-traversal-shaped, or
    excessively-long IDs."""
    if not block_id:
        return False
    if "/" in block_id or "\\" in block_id or ".." in block_id:
        return False
    if len(block_id) > _BLOCK_ID_MAX:
        return False
    return True


def _handle_delete_memory(
    workspace: str,
    block_id: str,
    *,
    actor: str = DIRECT_CALL_ACTOR,
    rationale: str | None = None,
) -> tuple[int, dict[str, Any]]:
    """``DELETE /memories/{id}`` — one governed death, one chain record.

    The route sends no body, so it carries no reason of its own; the
    scope records :data:`DEFAULT_DELETE_RATIONALE` unless a caller passes
    one. Naming the door is the honest floor — an audit record that
    cannot say why content was destroyed is most of the way to no record,
    and "a human typed a reason" is a claim this route cannot make.

    The other half of that record is *who*. It used to be ``"anonymous"``
    on every request this transport served: the handler defaulted
    ``actor`` to ``""``, which the gate resolves through the REST
    contextvar, and this transport is not the REST app so nothing ever
    set it. ``_dispatch`` now derives the identity from the credential
    that passed authentication and passes it here, once, for every
    mutating route — see :data:`HTTP_TOKEN_ACTOR_PREFIX`.

    There is **no existence pre-check**. Resolving the target before
    opening the scope answered "is this id real?" to a caller the gate
    had not yet authorised, and left no record of the question; inside a
    covering scope the store returns ``False`` for an id that is not
    there, which is the same 404 with an authorisation row behind it.
    That is what :func:`~mind_mem.admission.require_delete_admission`
    bought — "authorised" and "present" are told apart by the gate, never
    by a probe.

    Args:
        actor: Identity to attribute the deletion to. The dispatcher
            passes the door identity; a direct in-process caller gets
            :data:`DIRECT_CALL_ACTOR`. An actor in
            :data:`ANONYMOUS_ACTORS` is refused — recording a deletion
            under a word that names nobody is most of the way to no
            record.
        rationale: Overrides :data:`DEFAULT_DELETE_RATIONALE`.

    Returns 403 when governance refuses the scope: the block is still
    there, which is the fail-closed direction — a refused authorisation
    must never be reported as a completed deletion.
    """
    if not _valid_block_id(block_id):
        return (400, {"error": "invalid block id"})
    if not _is_named_actor(actor):
        # Fail closed: nothing is removed, and no record is minted
        # claiming a deletion nobody can be held to.
        _log.error("delete_memory_unnamed_actor", extra={"block_id": _safe_log(block_id)})
        return (500, {"error": "delete requires a named actor"})

    from .block_store import CorpusEncodingError
    from .governance_gate import GovernanceBypassError, get_gate
    from .storage import get_block_store

    try:
        store = get_block_store(workspace)
    except Exception as exc:
        _log.error(
            "delete_memory_store_init_failed",
            extra={"error": _safe_log(exc), "block_id": _safe_log(block_id)},
        )
        return (500, {"error": "internal block store error"})

    admission_id = ""
    try:
        gate = get_gate(workspace)
        with gate.admit_delete(
            block_id,
            rationale=rationale or DEFAULT_DELETE_RATIONALE,
            actor=actor,
        ) as receipt:
            admission_id = receipt.entry_id
            removed = store.delete_block(block_id)
    except (FileNotFoundError, KeyError):
        return (404, {"error": "block not found", "id": block_id})
    except CorpusEncodingError as exc:
        # Anticipated and answered by name. Before this branch existed the
        # decode error fell through to the generic handler below and every
        # Windows CI row answered 500 to `DELETE /memories/<missing-id>`,
        # because the workspace fixture had written an em dash through the
        # locale codepage and `delete_block` scans the corpus.
        _log.error(
            "delete_memory_corpus_encoding",
            extra={"error": _safe_log(exc), "block_id": _safe_log(block_id), "file": _safe_log(exc.path)},
        )
        return _corpus_encoding_response(exc, workspace, id=block_id)
    except GovernanceBypassError as exc:
        # The block is still there. Reporting this as anything but a
        # refusal would tell a caller their content is gone when it is
        # not, which is the one answer a memory product must never give.
        _log.error(
            "delete_memory_refused",
            extra={"error": _safe_log(exc), "block_id": _safe_log(block_id)},
        )
        return (403, {"error": "delete refused by governance", "id": block_id})
    except Exception as exc:
        # ``exc_info`` on purpose: this is the one branch that answers 500 for
        # a reason the code did not anticipate, and the stdlib formatter does
        # not render ``extra``, so a bare event name is what reached the log
        # -- which is all the Windows CI rows had to say about
        # ``DELETE /memories/<missing>`` answering 500 where every other row
        # answers 404. The traceback goes to the server log only; the
        # response body is unchanged and names nothing.
        _log.error(
            "delete_memory_failed",
            exc_info=True,
            extra={"error": _safe_log(exc), "block_id": _safe_log(block_id)},
        )
        return (500, {"error": "internal block store error"})

    if not removed:
        return (404, {"error": "block not found", "id": block_id})
    return (200, {"ok": True, "id": block_id, "admission": admission_id})


def _handle_clear(workspace: str, body: dict[str, Any], *, actor: str = DIRECT_CALL_ACTOR) -> tuple[int, dict[str, Any]]:
    """``POST /clear`` — wipe workspace contents (governance-protected).

    Requires a non-empty ``rationale`` per v3.6.x mandatory rationale
    binding. Refuses unless ``confirm`` is the literal string
    ``"yes-i-really-want-to-clear"``.

    **One decision, one authorisation, one removal record.** The wipe
    runs inside a single
    :meth:`~mind_mem.governance_gate.GovernanceGate.admit_delete_batch`
    scope over the exact set the loop will iterate, frozen before the
    first removal — so a block written *while* the clear runs is outside
    the receipt and cannot be taken by it. Per-block scopes would leave N
    unlinked records in a chain built for low-volume decisions, and
    nothing in them would say the removals were one operation.

    The covered set is whatever :func:`_corpus_block_ids` returns,
    because that is what the loop deletes; the scope covers the iterated
    set by construction rather than by a second enumeration that could
    drift from it. It used to be ``list_blocks()``, which is a list of
    *files* — see :func:`_corpus_block_ids` for what that cost.

    Args:
        actor: See :func:`_handle_delete_memory`. ``_dispatch`` derives
            the door identity from the credential that passed auth and
            passes it here; an actor in :data:`ANONYMOUS_ACTORS` is
            refused before anything is enumerated.

    Returns:
        ``200`` with ``deleted`` and the admission id. ``unreachable`` is
        present only when the corpus held blocks with no id: an
        incomplete wipe must be visibly incomplete, and a key nobody has
        to read is the additive way to say so.

    Scope, stated because a wipe that is narrower than its name is the
    same failure in a smaller box: "the corpus" is whatever
    ``store.get_all`` returns, which is every file
    :data:`~mind_mem.corpus_registry.CORPUS_TABLE` names — the four
    ``CORPUS_DIRS`` (``decisions``, ``tasks``, ``entities``,
    ``intelligence``) **and** the ``memory/`` rows (``MESSAGES.md``,
    ``INBOX.md``, ``IMPORTED.md``, ``INGEST.md``).

    This paragraph used to say the opposite — that a block in
    ``memory/INBOX.md`` was invisible to ``get_all`` and so survived this
    endpoint. It is false against the one corpus definition:
    ``tests/test_one_corpus_definition.py``'s
    ``test_the_clear_door_counts_and_takes_the_memory_corpora`` seeds one
    block per table row, holds the four ``memory/`` ids as its positive
    control, and measures ``deleted == len(before)`` with an empty corpus
    after. What recall can serve is what a clear takes; a ``memory/``
    file the table does *not* name (the daily log) stays outside both.
    """
    rationale = body.get("rationale")
    confirm = body.get("confirm")
    if not _is_named_actor(actor):
        # Fail closed, before the corpus is even enumerated: a wipe is
        # the largest act this surface offers and the least defensible
        # one to record against nobody.
        _log.error("clear_unnamed_actor", extra={"workspace": _safe_log(workspace)})
        return (500, {"error": "clear requires a named actor"})
    if not isinstance(rationale, str) or len(rationale.strip()) < 16:
        return (400, {"error": "rationale is required (min 16 chars per governance policy)"})
    if confirm != "yes-i-really-want-to-clear":
        return (
            400,
            {
                "error": "confirm field must equal 'yes-i-really-want-to-clear' to proceed",
                "rationale_received": rationale[:80],
            },
        )

    from .block_store import CorpusEncodingError
    from .governance_gate import GovernanceBypassError, get_gate
    from .storage import get_block_store

    try:
        store = get_block_store(workspace)
        block_ids, unidentified = _corpus_block_ids(store)
    except Exception as exc:
        _log.error("clear_failed", extra={"error": str(exc)})
        return (500, {"error": "internal block store error"})

    if unidentified:
        _log.warning(
            "clear_blocks_without_id",
            extra={"workspace": _safe_log(workspace), "unreachable": unidentified},
        )

    if not block_ids:
        # No scope: a receipt covering nothing authorises nothing, and
        # minting one would put a decision in the chain that never had a
        # subject. Nothing died, so there is nothing to record.
        empty: dict[str, Any] = {"ok": True, "deleted": 0, "rationale": rationale, "admission": None}
        if unidentified:
            empty["unreachable"] = unidentified
        return (200, empty)

    batch_id = _clear_batch_id(block_ids)
    deleted = 0
    admission_id = ""
    #: block id -> the CorpusEncodingError that stopped its removal.
    undecodable: dict[str, Exception] = {}
    try:
        gate = get_gate(workspace)
        with gate.admit_delete_batch(batch_id, block_ids, rationale=rationale, actor=actor) as receipt:
            admission_id = receipt.entry_id
            # deferred: the loop is O(n²) on a Markdown corpus — each
            # delete_block re-reads and rewrites the whole .md file.
            # Measured on one file: 200 blocks 0.16 s, 400 0.46 s, 800
            # 1.43 s, so ~4 min at 10k. Acceptable for a rare, twice-
            # confirmed destructive call, and the previous shape's cost
            # was zero only because it deleted nothing. Upgrade path: a
            # bulk `delete_blocks(ids)` on the store protocol, splicing
            # each file once, still reporting every removal into the one
            # receipt so the single bulk record is unchanged.
            for bid in block_ids:
                try:
                    if store.delete_block(bid):
                        deleted += 1
                except GovernanceBypassError:
                    # Never swallowed. A store refusing a block this
                    # scope was supposed to cover means the receipt and
                    # the loop disagree about what was authorised, which
                    # is exactly the failure the scope exists to catch.
                    raise
                except CorpusEncodingError as enc_exc:
                    # NOT a debug-level skip. The block may be sitting in
                    # the very file that could not be decoded, so counting
                    # this as "handled" and answering `ok` is a partial
                    # purge reported as a whole one. It is named in the
                    # response instead, beside the ids that resolved to no
                    # file at all.
                    undecodable[bid] = enc_exc
                    _log.warning(
                        "clear_block_corpus_encoding",
                        extra={"block_id": _safe_log(bid), "file": _safe_log(enc_exc.path)},
                    )
                    continue
                except Exception as block_exc:
                    # One bad block must not abort the wipe — record the
                    # failure at debug so the operator can investigate
                    # individual failures without losing the bulk-clear.
                    _log.debug(
                        "clear_block_skip",
                        extra={"block_id": _safe_log(bid), "error": _safe_log(block_exc)},
                    )
                    continue
    except GovernanceBypassError as exc:
        _log.error("clear_refused", extra={"error": _safe_log(exc)})
        return (403, {"error": "clear refused by governance"})
    except Exception as exc:
        _log.error("clear_failed", extra={"error": str(exc)})
        return (500, {"error": "internal block store error"})

    _log.warning(
        "workspace_cleared",
        extra={
            "workspace": _safe_log(workspace),
            "deleted": deleted,
            "rationale": _safe_log(rationale, max_len=120),
        },
    )
    out: dict[str, Any] = {"ok": bool(not undecodable), "deleted": deleted, "rationale": rationale, "admission": admission_id}
    if unidentified:
        out["unreachable"] = unidentified
    if undecodable:
        out["undecodable"] = [
            {"id": bid, "file": os.path.basename(getattr(exc, "path", "")), "detail": getattr(exc, "reason", str(exc))}
            for bid, exc in sorted(undecodable.items())
        ]
    return (200, out)


# ---------------------------------------------------------------------------
# Request dispatcher — single class, all endpoints
# ---------------------------------------------------------------------------


def _handle_fed_vclock(workspace: str, block_id: str) -> tuple[int, dict[str, Any]]:
    """GET /federation/vclock/<block_id> — read per-agent version vector.

    Returns 503 if v4.federation flag is disabled, 200 + dict otherwise.
    Missing block returns an empty version-vector dict (still 200) —
    callers treat that as "no writes yet seen".
    """
    if not _valid_block_id(block_id):
        return (400, {"ok": False, "error": "invalid block_id"})
    try:
        from mind_mem.v4 import federation as fed
    except ImportError:
        return (503, {"ok": False, "error": "federation module unavailable"})
    try:
        vec = fed.get_version_vector(workspace, block_id)
    except Exception as exc:
        # Audit S-3: separate feature-flag-off from internal failure.
        # FeatureDisabledError remains 503 + a stable string; any other
        # exception is logged with detail server-side, surfaced as a
        # generic "federation unavailable" so paths/schema/IO errors
        # don't leak in the wire response.
        return _fed_error_response(exc, "vclock")
    return (200, {"ok": True, "block_id": block_id, "version_vector": vec})


def _fed_error_response(exc: Exception, endpoint: str) -> tuple[int, dict[str, Any]]:
    """Audit S-3: PII-safe federation error mapping.

    Distinguishes the documented feature-disabled path (503 + stable
    message) from internal failures (503 + generic message, full detail
    logged server-side only)."""
    name = type(exc).__name__
    if name in ("FeatureDisabledError", "FeatureFlagDisabled"):
        return (503, {"ok": False, "error": "federation feature disabled"})
    _log.error(
        "federation_internal_error",
        extra={"endpoint": endpoint, "error_type": name},
    )
    return (503, {"ok": False, "error": "federation unavailable"})


def _handle_fed_conflicts(workspace: str, params: dict[str, str]) -> tuple[int, dict[str, Any]]:
    """GET /federation/conflicts?limit=N — list outstanding (unresolved) conflicts."""
    try:
        from mind_mem.v4 import federation as fed
    except ImportError:
        return (503, {"ok": False, "error": "federation module unavailable"})
    try:
        limit_raw = params.get("limit", "100")
        limit = max(1, min(int(limit_raw), 1000))
    except (TypeError, ValueError):
        return (400, {"ok": False, "error": "limit must be a positive integer"})
    try:
        reports = fed.list_conflicts(workspace, limit=limit)
    except Exception as exc:
        return _fed_error_response(exc, "conflicts")
    return (
        200,
        {
            "ok": True,
            "conflicts": [
                {
                    "block_id": r.block_id,
                    "left_agent": r.left_agent,
                    "left_version": r.left_version,
                    "right_agent": r.right_agent,
                    "right_version": r.right_version,
                }
                for r in reports
            ],
        },
    )


def _handle_fed_write(workspace: str, body: dict[str, Any], *, actor: str) -> tuple[int, dict[str, Any]]:
    """POST /federation/write {block_id, agent_id} — record agent write.

    Bumps the (block_id, agent_id) version atomically and reports a
    conflict if the resulting version vector diverges from another
    agent's claim. Auto-detects + logs the conflict to ``tier_conflict_log``.

    Args:
        actor: The door identity, keyword-only with **no default**. It is
            not ``agent_id``: that one is a *claim the body makes* and the
            version vector is keyed on it, while this is the credential
            that passed auth. Both go to the log, because a peer writing
            under someone else's ``agent_id`` is exactly the thing an
            operator would want to be able to see afterwards.
    """
    try:
        from mind_mem.v4 import federation as fed
    except ImportError:
        return (503, {"ok": False, "error": "federation module unavailable"})
    block_id = body.get("block_id")
    agent_id = body.get("agent_id")
    if not isinstance(block_id, str) or not _valid_block_id(block_id):
        return (400, {"ok": False, "error": "block_id (valid string) is required"})
    if not isinstance(agent_id, str) or not _valid_block_id(agent_id):
        return (400, {"ok": False, "error": "agent_id (valid string) is required"})
    _log.info(
        "fed_write_requested",
        extra={"actor": _safe_log(actor), "block_id": _safe_log(block_id), "claimed_agent_id": _safe_log(agent_id)},
    )
    try:
        new_version = fed.record_agent_write(workspace, block_id, agent_id)
        report = fed.detect_conflict(workspace, block_id)
    except Exception as exc:
        return _fed_error_response(exc, "write")
    out: dict[str, Any] = {
        "ok": True,
        "block_id": block_id,
        "agent_id": agent_id,
        "version": new_version,
    }
    if report is not None:
        out["conflict"] = {
            "left_agent": report.left_agent,
            "left_version": report.left_version,
            "right_agent": report.right_agent,
            "right_version": report.right_version,
        }
    return (200, out)


def _handle_fed_resolve(workspace: str, body: dict[str, Any], *, actor: str) -> tuple[int, dict[str, Any]]:
    """POST /federation/resolve {block_id, strategy, merged_payload?}.

    Applies the chosen MergeStrategy to the most-recent open conflict.
    For THREE_WAY_MERGE the caller supplies a merged_payload that is
    treated as the merge result; the function does not invoke a
    server-side merger callable.

    Args:
        actor: The door identity, keyword-only with **no default**.
            Resolving a conflict picks one peer's bytes over another's,
            so the log says who asked for that.
    """
    try:
        from mind_mem.v4 import federation as fed
    except ImportError:
        return (503, {"ok": False, "error": "federation module unavailable"})
    block_id = body.get("block_id")
    strategy = body.get("strategy")
    merged_b64 = body.get("merged_payload")
    if not isinstance(block_id, str) or not _valid_block_id(block_id):
        return (400, {"ok": False, "error": "block_id (valid string) is required"})
    if not isinstance(strategy, str) or strategy not in {s.value for s in fed.MergeStrategy}:
        return (400, {"ok": False, "error": "strategy must be one of MergeStrategy values"})
    _log.info(
        "fed_resolve_requested",
        extra={"actor": _safe_log(actor), "block_id": _safe_log(block_id), "strategy": _safe_log(strategy)},
    )
    merger = None
    if merged_b64 is not None:
        import base64

        try:
            merged_bytes = base64.b64decode(merged_b64)
        except Exception:
            return (400, {"ok": False, "error": "merged_payload must be base64-encoded bytes"})
        merger = lambda _report, _payload=merged_bytes: _payload  # noqa: E731
    try:
        resolution = fed.resolve_conflict(workspace, block_id, strategy, merger=merger)
    except ValueError as exc:
        # FP-4: THREE_WAY_MERGE without merger now raises ValueError. Map
        # caller-facing programming errors to 400 (not 503) so the
        # operator sees a configuration bug, not a transient outage.
        return (400, {"ok": False, "error": str(exc)})
    except Exception as exc:
        return _fed_error_response(exc, "resolve")
    if resolution is None:
        return (404, {"ok": False, "error": "no open conflict for block_id"})
    out: dict[str, Any] = {
        "ok": True,
        "block_id": resolution.block_id,
        "winner_agent": resolution.winner_agent,
        "winner_version": resolution.winner_version,
        "strategy": resolution.strategy.value,
    }
    if resolution.merged_payload is not None:
        import base64

        out["merged_payload"] = base64.b64encode(resolution.merged_payload).decode("ascii")
    return (200, out)


# ---------------------------------------------------------------------------
# The route table — the only thing that makes a handler reachable
# ---------------------------------------------------------------------------

#: The response can carry workspace block content.
CONTENT = "content"
#: It cannot. Swept anyway — a misclassification is what the sweep is for.
NO_CONTENT = "no-content"
_VERDICTS = frozenset({CONTENT, NO_CONTENT})

#: How the dispatcher builds a handler's second argument.
_TAKES = frozenset({"workspace", "params", "body", "tail"})


@dataclass(frozen=True)
class Route:
    """One reachable endpoint, and whether it can serve block content.

    ``verdict`` has **no default**. A new endpoint cannot be routed
    without someone deciding what it serves, and the sweep in
    ``tests/test_http_read_admission.py`` measures that decision against
    a quarantined canary rather than trusting it: the reach set it
    observes must equal the ``content`` set declared here, both ways
    round. A route that starts returning block content joins the reach
    set and fails the build until it is reclassified; a ``content`` route
    that stops reaching fails too, rather than degrading into a canary
    check over an error string.

    This is the HTTP twin of the MCP registry sweep. That sweep covers the
    102 registered tools and nothing else — the whole of this transport,
    every REST client and every library caller were outside it, which is
    how ``GET /memories`` served ``get_all(active_only=False)`` for four
    minor versions with no admission on it at all.
    """

    method: str
    path: str
    handler: Callable[..., tuple[int, dict[str, Any]]]
    takes: str
    verdict: str
    #: Whether serving this route can change workspace state. Like
    #: ``verdict`` it has **no default**, and like ``verdict`` it is not
    #: documentation: ``True`` makes the dispatcher pass the door's actor
    #: identity, and ``__post_init__`` refuses at import to route a
    #: mutating handler that cannot receive one. That is what makes the
    #: actor unforgettable — the next handler cannot be added to this
    #: table without it, rather than being expected to remember.
    mutates: bool
    #: Response for a prefix route whose tail is empty, when the handler
    #: should not be reached at all with a blank id.
    empty_tail_error: str | None = None

    def __post_init__(self) -> None:
        # Import-time, not test-time: a malformed route cannot be loaded,
        # so the module refuses to serve rather than serving something
        # nobody classified.
        if self.verdict not in _VERDICTS:
            raise ValueError(f"route {self.method} {self.path} has verdict {self.verdict!r}; must be one of {sorted(_VERDICTS)}")
        if self.takes not in _TAKES:
            raise ValueError(f"route {self.method} {self.path} takes {self.takes!r}; must be one of {sorted(_TAKES)}")
        actor_param = inspect.signature(self.handler).parameters.get("actor")
        takes_actor = actor_param is not None and actor_param.kind is inspect.Parameter.KEYWORD_ONLY
        if self.mutates and not takes_actor:
            raise ValueError(
                f"route {self.method} {self.path} mutates but {self.handler.__name__} has no keyword-only 'actor' "
                "parameter; a door that changes workspace state must be able to record who opened it"
            )
        if not self.mutates and takes_actor:
            raise ValueError(
                f"route {self.method} {self.path} is declared read-only but {self.handler.__name__} takes a keyword-only "
                "'actor'; the dispatcher passes one only to mutating routes, so one of the two declarations is wrong"
            )

    @property
    def name(self) -> str:
        """``"GET /memories"`` — the id the sweep and the metrics use."""
        return f"{self.method} {self.path}"


#: Every reachable endpoint. ``takes="tail"`` is a prefix match; every
#: other kind is an exact match on the path with the query string split
#: off. Order is the match order.
ROUTES: tuple[Route, ...] = (
    Route("GET", PATH_STATUS, _handle_status, "workspace", NO_CONTENT, mutates=False),
    Route("GET", PATH_MEMORIES, _handle_list_memories, "params", CONTENT, mutates=False),
    Route("GET", PATH_FED_CONFLICTS, _handle_fed_conflicts, "params", NO_CONTENT, mutates=False),
    Route("GET", _FED_VCLOCK_PREFIX, _handle_fed_vclock, "tail", NO_CONTENT, mutates=False, empty_tail_error="block_id required"),
    Route("POST", PATH_QUERY, _handle_query, "body", CONTENT, mutates=False),
    Route("POST", PATH_CONSOLIDATE, _handle_consolidate, "body", NO_CONTENT, mutates=True),
    # MEASURED, not assumed. ``compile_walkthrough`` projects recall rows
    # into ``{step, block_id, role, score, subject}`` and the rows carry
    # ``excerpt`` rather than ``Statement``, so the subject comes out
    # empty and the response is block ids and scores. Same shape, and the
    # same reasoning, as ``compile_truth_walkthrough`` on the MCP side.
    # The reach check is what keeps this honest: the day the projection
    # starts carrying text, the sweep's reach set grows and the build
    # fails until this row says CONTENT.
    Route("POST", PATH_WALKTHROUGH, _handle_walkthrough, "body", NO_CONTENT, mutates=False),
    Route("POST", PATH_CLEAR, _handle_clear, "body", NO_CONTENT, mutates=True),
    Route("POST", PATH_FED_WRITE, _handle_fed_write, "body", NO_CONTENT, mutates=True),
    Route("POST", PATH_FED_RESOLVE, _handle_fed_resolve, "body", NO_CONTENT, mutates=True),
    # The tail is a block id the caller supplied, so an empty one reaches
    # the handler and is refused there by ``_valid_block_id`` — one
    # rejection path for a bad id, not two.
    Route("DELETE", _MEMORY_ID_PREFIX, _handle_delete_memory, "tail", NO_CONTENT, mutates=True),
)


def content_routes() -> frozenset[str]:
    """Names of the routes declared able to serve block content."""
    return frozenset(route.name for route in ROUTES if route.verdict == CONTENT)


def mutating_routes() -> frozenset[str]:
    """Names of the routes the dispatcher hands a door identity to."""
    return frozenset(route.name for route in ROUTES if route.mutates)


def _match_route(method: str, base: str) -> tuple[Route | None, str]:
    """The route serving ``(method, base)``, and the tail it captured."""
    for route in ROUTES:
        if route.method != method:
            continue
        if route.takes == "tail":
            if base.startswith(route.path):
                return (route, base[len(route.path) :])
        elif base == route.path:
            return (route, "")
    return (None, "")


_LOOPBACK_ORIGINS = frozenset(
    {
        "http://127.0.0.1",
        "http://localhost",
        "https://127.0.0.1",
        "https://localhost",
    }
)


def build_handler(
    workspace: str,
    *,
    token: str | None,
    bind_host: str,
    allow_unauthenticated_localhost: bool,
    rate_limiter: "_PerClientRateLimiter | None" = None,
) -> type[BaseHTTPRequestHandler]:
    """Construct a handler class bound to *workspace* + auth settings."""

    auth_required = not (allow_unauthenticated_localhost and _is_loopback(bind_host))

    class Handler(BaseHTTPRequestHandler):
        server_version = "mind-mem-http/0.1"

        def log_message(self, format: str, *args: Any) -> None:  # silence default
            return

        # -- rate-limiting (S-2) ----------------------------------------
        def _rate_limited(self) -> bool:
            """Return True iff the request is denied; emits a 429 response."""
            if rate_limiter is None:
                return False
            client_key = self.client_address[0] if self.client_address else "unknown"
            allowed, retry_after = rate_limiter.allow(client_key)
            if allowed:
                return False
            # Send 429 with Retry-After header so clients back off cleanly.
            body = json.dumps({"status": 429, "error": "rate limit exceeded"}, sort_keys=True).encode("utf-8")
            self.send_response(429)
            self.send_header("Content-Type", "application/json")
            self.send_header("Retry-After", str(int(retry_after) or 1))
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return True

        # -- CORS / CSRF guard (S-7) ------------------------------------
        def _origin_ok(self) -> bool:
            """Refuse cross-origin requests from non-loopback Origins.

            A browser tab loaded from evil.example.com would otherwise be
            able to issue cross-origin POSTs to the loopback transport.
            The custom ``X-MindMem-Token`` header forces a CORS preflight,
            and we reject the preflight + any explicit non-loopback Origin
            on the actual request."""
            origin = self.headers.get("Origin", "").strip()
            if not origin:
                # Same-origin or non-browser caller (curl, MCP client) —
                # no Origin sent. Accept.
                return True
            origin_norm = origin.split("/", 3)
            if len(origin_norm) < 3:
                return False
            host_url = "/".join(origin_norm[:3])
            return host_url in _LOOPBACK_ORIGINS

        # -- auth --------------------------------------------------------
        def _authenticated(self) -> bool:
            if not auth_required:
                return True
            sent = self.headers.get(AUTH_HEADER, "")
            if not sent:
                return False
            # Token rotation (roadmap v4.0.x): accept any of an N-of-K
            # active-tokens set. ``MIND_MEM_TOKENS`` (comma-separated)
            # is preferred when set — supports grace-window rotation
            # without restart. Falls back to the single ``token`` value
            # the handler was built with for backwards compat.
            #
            # Read at request time so ``mm token rotate`` (which appends
            # to MIND_MEM_TOKENS) takes effect for the next request
            # without restarting the server.
            active = _active_tokens(fallback=token)
            if not active:
                return False
            # Constant-time comparison against EVERY active token (audit
            # S-1). The list comprehension is the load-bearing part: with a
            # generator, ``any`` returns on the first True, so the number of
            # compare_digest calls would be 1 for a token at index 0 and
            # len(active) for a miss — leaking the matched token's position
            # through timing. Materialising the list runs exactly one
            # comparison per active token whatever the answer is; each
            # comparison is itself constant-time.
            results = [hmac.compare_digest(sent, t) for t in active]
            return any(results)

        # -- peer allowlist (roadmap v4.0.x federation hardening) -------
        def _peer_allowed(self) -> bool:
            """Return True iff the request source IP passes the operator
            allowlist for federation endpoints.

            Configured via ``MIND_MEM_FED_PEERS`` env var (comma-separated
            list of IPv4 / IPv6 addresses). When the env var is unset
            *and* empty the allowlist is bypassed (backwards compatible
            with the localhost-only default deployment). When set, any
            source IP outside the set is rejected with 403 *before*
            auth — even a valid token doesn't help if the caller isn't
            on the allowlist. Compatible with bearer-token auth, doesn't
            replace it. Always applies to federation endpoints only;
            non-federation paths are unaffected.
            """
            # Only enforce on federation endpoints; status / memories
            # remain governed by token + Origin checks.
            base, _ = _parse_query_params(self.path)
            is_fed = base in {
                PATH_FED_VCLOCK,
                PATH_FED_CONFLICTS,
                PATH_FED_WRITE,
                PATH_FED_RESOLVE,
            } or base.startswith(_FED_VCLOCK_PREFIX)
            if not is_fed:
                return True
            raw = os.environ.get("MIND_MEM_FED_PEERS", "").strip()
            if not raw:
                # No allowlist configured → bypass (existing behaviour).
                return True
            allowed = {p.strip() for p in raw.split(",") if p.strip()}
            source = self.client_address[0] if self.client_address else ""
            return source in allowed

        # -- door identity ----------------------------------------------
        def _door_actor(self) -> str:
            """Who this request is, for the record a mutating route writes.

            Derived from the credential that just passed
            :meth:`_authenticated` — at the dispatcher, once, so no
            handler has to remember. Two shapes and no third:

            ``http:tok:<sha256(token)[:12]>``
                a request that presented a token on the active set. The
                digest is the token's identity without the token, so
                rotation does not rewrite what the old one already did.
            ``http:loopback-unauthenticated``
                the operator started this server with
                ``--allow-unauthenticated-localhost`` on a loopback bind.
                There is no credential to name, and the record says that
                rather than borrowing a word that reads like a failed
                lookup.

            Only ever called after :meth:`_guards_passed`, so the header
            read here is the one that passed the constant-time compare.
            """
            if not auth_required:
                return HTTP_UNAUTHENTICATED_ACTOR
            return _token_actor(self.headers.get(AUTH_HEADER, ""))

        def _reject_auth(self) -> None:
            _write_status(self, 401, "missing or invalid token")

        def _reject_peer(self) -> None:
            _write_status(self, 403, "source IP not on MIND_MEM_FED_PEERS allowlist")

        # -- body parsing -----------------------------------------------
        def _read_json_body(self) -> tuple[dict[str, Any] | None, int]:
            raw, err = _read_body(self)
            if err:
                return (None, err)
            if not raw:
                return ({}, 0)
            try:
                payload = json.loads(raw.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                return (None, 400)
            if not isinstance(payload, dict):
                return (None, 400)
            return (payload, 0)

        # -- request guards, in one place -------------------------------
        def _guards_passed(self) -> bool:
            """Origin, rate limit, peer allowlist, auth — in that order.

            The order is load-bearing and unchanged: a cross-origin
            request is refused before it consumes rate-limit budget, and
            the federation peer allowlist is checked *before* auth, so a
            valid token from an address the operator did not list still
            gets 403.
            """
            if not self._origin_ok():
                _write_status(self, 403, "cross-origin request rejected")
                return False
            if self._rate_limited():
                return False
            if not self._peer_allowed():
                self._reject_peer()
                return False
            if not self._authenticated():
                self._reject_auth()
                return False
            return True

        # -- dispatch ---------------------------------------------------
        def _dispatch(self, method: str, payload: dict[str, Any] | None = None) -> None:
            """Route one request through :data:`ROUTES`, or 404.

            The table is the only source of reachability. An ``if base ==
            ...`` chain here would let a handler be served without a
            classification, which is exactly how this transport ended up
            outside the read-surface sweep.

            The same table decides attribution: a route declaring
            ``mutates`` is handed :meth:`_door_actor` and a route that
            does not, is not. The identity is derived here rather than in
            each handler because "the handler passes an actor" is a thing
            to remember and "the route table says so" is a thing to
            declare — and ``Route.__post_init__`` refuses at import to
            route a mutating handler that cannot take one.
            """
            base, params = _parse_query_params(self.path)
            route, tail = _match_route(method, base)
            if route is None:
                _write_status(self, 404, "not found")
                return
            if route.takes == "tail" and not tail and route.empty_tail_error:
                _write_status(self, 400, route.empty_tail_error)
                return
            attribution: dict[str, Any] = {"actor": self._door_actor()} if route.mutates else {}
            if route.takes == "workspace":
                status, body = route.handler(workspace, **attribution)
            elif route.takes == "params":
                status, body = route.handler(workspace, params, **attribution)
            elif route.takes == "body":
                status, body = route.handler(workspace, payload if payload is not None else {}, **attribution)
            else:
                status, body = route.handler(workspace, tail, **attribution)
            _write_json(self, status, body)

        # -- OPTIONS (CORS preflight reject — S-7) ----------------------
        def do_OPTIONS(self) -> None:
            _write_status(self, 405, "method not allowed")

        # -- GET --------------------------------------------------------
        def do_GET(self) -> None:
            if not self._guards_passed():
                return
            self._dispatch("GET")

        # -- POST -------------------------------------------------------
        def do_POST(self) -> None:
            if not self._guards_passed():
                return
            # Body first, route second — unchanged. An oversized body is
            # a 413 whatever path it was aimed at, so a caller cannot use
            # an unknown path to smuggle one past the cap.
            payload, err = self._read_json_body()
            if err:
                _write_status(self, err, "bad request body")
                return
            if payload is None:
                # err==0 implies non-None payload, but be defensive
                # rather than assert — a stale handler shouldn't 500.
                _write_status(self, 400, "empty body")
                return
            self._dispatch("POST", payload)

        # -- DELETE -----------------------------------------------------
        def do_DELETE(self) -> None:
            if not self._guards_passed():
                return
            self._dispatch("DELETE")

    return Handler


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def serve_http(
    workspace: str,
    *,
    port: int = DEFAULT_PORT,
    host: str = DEFAULT_HOST,
    token: str | None = None,
    allow_unauthenticated_localhost: bool = False,
) -> tuple[threading.Thread, Callable[[], None]]:
    """Start the v3.9 HTTP transport in a background thread.

    Returns ``(server_thread, stop_fn)``. Call ``stop_fn()`` to shut
    the server down cleanly. Token is read from ``MIND_MEM_TOKEN`` if
    not given explicitly. When binding to a loopback address the
    operator may pass ``allow_unauthenticated_localhost=True`` to
    bypass auth (matches the existing MCP HTTP transport posture).
    """
    if not workspace:
        raise ValueError("workspace must be a non-empty path")
    if token is None:
        token = os.environ.get("MIND_MEM_TOKEN", "").strip() or None
    if not allow_unauthenticated_localhost and not token:
        raise ValueError("no MIND_MEM_TOKEN configured and --allow-unauthenticated-localhost not set; refusing to start")

    # Audit S-2: per-IP sliding-window rate limit. Operators can tune via
    # MIND_MEM_HTTP_RATE_MAX_CALLS / MIND_MEM_HTTP_RATE_WINDOW_SECS, or
    # set MAX_CALLS=0 to disable (tests + air-gapped deployments).
    try:
        max_calls = int(os.environ.get("MIND_MEM_HTTP_RATE_MAX_CALLS", DEFAULT_RATE_MAX_CALLS))
        window_seconds = int(os.environ.get("MIND_MEM_HTTP_RATE_WINDOW_SECS", DEFAULT_RATE_WINDOW_SECS))
    except ValueError:
        max_calls = DEFAULT_RATE_MAX_CALLS
        window_seconds = DEFAULT_RATE_WINDOW_SECS
    rate_limiter: _PerClientRateLimiter | None
    if max_calls <= 0:
        rate_limiter = None
    else:
        rate_limiter = _PerClientRateLimiter(
            max_calls=max_calls,
            window_seconds=window_seconds,
            max_clients=MAX_TRACKED_CLIENTS,
        )

    handler_cls = build_handler(
        workspace,
        token=token,
        bind_host=host,
        allow_unauthenticated_localhost=allow_unauthenticated_localhost,
        rate_limiter=rate_limiter,
    )
    httpd = _ThreadingHTTPServer((host, port), handler_cls)
    # serve_forever()'s poll interval bounds shutdown latency; tighten it so
    # _stop() returns promptly instead of waiting up to the 0.5s default.
    thread = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()

    def _stop() -> None:
        httpd.shutdown()
        httpd.server_close()

    # Deterministic boot: do not return until the listener is actually
    # accepting connections (see _wait_until_accepting). If the accept loop
    # never comes up, tear the half-started server down and surface the error.
    try:
        _wait_until_accepting(host, port, deadline=time.monotonic() + _READY_TIMEOUT_SECS)
    except TimeoutError:
        _stop()
        thread.join(timeout=5)
        raise

    _log.info(
        "http_transport_started",
        extra={"host": host, "port": port, "auth": "token" if token else "none"},
    )
    return thread, _stop
