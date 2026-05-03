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

import json
import logging
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from typing import Any, Callable

from .protection import AUTH_HEADER

__all__ = [
    "MAX_BODY_BYTES",
    "serve_http",
    "build_handler",
]

_log = logging.getLogger("mind_mem.http_transport")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_BODY_BYTES = 1_048_576  # 1 MiB cap on POST/DELETE bodies
DEFAULT_PORT = 8765
DEFAULT_HOST = "127.0.0.1"

# Endpoint paths — kept as constants so tests can import them.
PATH_STATUS = "/status"
PATH_QUERY = "/query"
PATH_MEMORIES = "/memories"
PATH_CONSOLIDATE = "/consolidate"
PATH_CLEAR = "/clear"
_MEMORY_ID_PREFIX = "/memories/"

# Loopback addresses that may skip auth when the operator opts in.
_LOOPBACK_ADDRS = frozenset({"127.0.0.1", "::1", "localhost"})


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class _ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


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
# Endpoint handlers — pure functions of (workspace, body, params)
# ---------------------------------------------------------------------------


def _handle_status(workspace: str) -> tuple[int, dict[str, Any]]:
    """``GET /status`` — health + memory count + last-scan timestamp."""
    from .storage import get_block_store

    try:
        store = get_block_store(workspace)
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
            "workspace": os.path.abspath(workspace),
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

    from ._recall_core import recall as _recall
    from .personas import PERSONAS, PersonaError, apply_persona

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
    }
    if persona is not None:
        payload["persona"] = persona
    return (200, payload)


def _handle_list_memories(workspace: str, params: dict[str, str]) -> tuple[int, dict[str, Any]]:
    """``GET /memories?limit=N&active_only=true``."""
    try:
        limit = int(params.get("limit", "100"))
    except ValueError:
        return (400, {"error": "limit must be an integer"})
    if limit < 1 or limit > 1000:
        return (400, {"error": "limit must be in [1, 1000]"})
    active_only_str = params.get("active_only", "false").lower()
    active_only = active_only_str in ("1", "true", "yes")

    from .storage import get_block_store

    try:
        store = get_block_store(workspace)
        blocks = store.get_all(active_only=active_only)
    except Exception as exc:
        _log.error("list_memories_failed", extra={"error": str(exc)})
        return (500, {"error": "internal block store error"})

    summaries = [
        {
            "id": b.get("id"),
            "type": b.get("type") or b.get("block_type"),
            "category": b.get("category"),
            "subject": b.get("Subject") or b.get("subject"),
            "timestamp": b.get("timestamp") or b.get("Timestamp"),
        }
        for b in blocks[:limit]
    ]
    return (200, {"count": len(summaries), "total": len(blocks), "memories": summaries})


def _handle_consolidate(workspace: str, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """``POST /consolidate`` — trigger dream cycle."""
    dry_run = bool(body.get("dry_run", False))
    auto_repair = bool(body.get("auto_repair", False))

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


def _handle_delete_memory(workspace: str, block_id: str) -> tuple[int, dict[str, Any]]:
    """``DELETE /memories/{id}``."""
    if not block_id or "/" in block_id or ".." in block_id:
        return (400, {"error": "invalid block id"})

    from .storage import get_block_store

    try:
        store = get_block_store(workspace)
        removed = store.delete_block(block_id)
    except Exception as exc:
        _log.error("delete_memory_failed", extra={"error": str(exc), "block_id": block_id})
        return (500, {"error": "internal block store error"})

    if not removed:
        return (404, {"error": "block not found", "id": block_id})
    return (200, {"ok": True, "id": block_id})


def _handle_clear(workspace: str, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """``POST /clear`` — wipe workspace contents (governance-protected).

    Requires a non-empty ``rationale`` per v3.6.x mandatory rationale
    binding. Refuses unless ``confirm`` is the literal string
    ``"yes-i-really-want-to-clear"``.
    """
    rationale = body.get("rationale")
    confirm = body.get("confirm")
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

    from .storage import get_block_store

    try:
        store = get_block_store(workspace)
        block_ids = store.list_blocks()
        deleted = 0
        for bid in block_ids:
            try:
                if store.delete_block(bid):
                    deleted += 1
            except Exception:  # one bad block must not abort the wipe
                continue
    except Exception as exc:
        _log.error("clear_failed", extra={"error": str(exc)})
        return (500, {"error": "internal block store error"})

    _log.warning(
        "workspace_cleared",
        extra={"workspace": workspace, "deleted": deleted, "rationale": rationale[:120]},
    )
    return (200, {"ok": True, "deleted": deleted, "rationale": rationale})


# ---------------------------------------------------------------------------
# Request dispatcher — single class, all endpoints
# ---------------------------------------------------------------------------


def build_handler(
    workspace: str,
    *,
    token: str | None,
    bind_host: str,
    allow_unauthenticated_localhost: bool,
) -> type[BaseHTTPRequestHandler]:
    """Construct a handler class bound to *workspace* + auth settings."""

    auth_required = not (allow_unauthenticated_localhost and _is_loopback(bind_host))

    class Handler(BaseHTTPRequestHandler):
        server_version = "mind-mem-http/0.1"

        def log_message(self, format: str, *args: Any) -> None:  # silence default
            return

        # -- auth --------------------------------------------------------
        def _authenticated(self) -> bool:
            if not auth_required:
                return True
            if token is None:
                return False
            sent = self.headers.get(AUTH_HEADER, "")
            return sent == token

        def _reject_auth(self) -> None:
            _write_status(self, 401, "missing or invalid token")

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

        # -- GET --------------------------------------------------------
        def do_GET(self) -> None:
            if not self._authenticated():
                self._reject_auth()
                return
            base, params = _parse_query_params(self.path)
            if base == PATH_STATUS:
                status, body = _handle_status(workspace)
                _write_json(self, status, body)
                return
            if base == PATH_MEMORIES:
                status, body = _handle_list_memories(workspace, params)
                _write_json(self, status, body)
                return
            _write_status(self, 404, "not found")

        # -- POST -------------------------------------------------------
        def do_POST(self) -> None:
            if not self._authenticated():
                self._reject_auth()
                return
            base, _params = _parse_query_params(self.path)
            payload, err = self._read_json_body()
            if err:
                _write_status(self, err, "bad request body")
                return
            assert payload is not None  # mypy: err==0 implies non-None
            if base == PATH_QUERY:
                status, body = _handle_query(workspace, payload)
                _write_json(self, status, body)
                return
            if base == PATH_CONSOLIDATE:
                status, body = _handle_consolidate(workspace, payload)
                _write_json(self, status, body)
                return
            if base == PATH_CLEAR:
                status, body = _handle_clear(workspace, payload)
                _write_json(self, status, body)
                return
            _write_status(self, 404, "not found")

        # -- DELETE -----------------------------------------------------
        def do_DELETE(self) -> None:
            if not self._authenticated():
                self._reject_auth()
                return
            base, _params = _parse_query_params(self.path)
            if base.startswith(_MEMORY_ID_PREFIX):
                block_id = base[len(_MEMORY_ID_PREFIX) :]
                status, body = _handle_delete_memory(workspace, block_id)
                _write_json(self, status, body)
                return
            _write_status(self, 404, "not found")

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

    handler_cls = build_handler(
        workspace,
        token=token,
        bind_host=host,
        allow_unauthenticated_localhost=allow_unauthenticated_localhost,
    )
    httpd = _ThreadingHTTPServer((host, port), handler_cls)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    def _stop() -> None:
        httpd.shutdown()
        httpd.server_close()

    _log.info(
        "http_transport_started",
        extra={"host": host, "port": port, "auth": "token" if token else "none"},
    )
    return thread, _stop
