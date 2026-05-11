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
    except Exception as exc:
        _log.error(
            "delete_memory_store_init_failed",
            extra={"error": _safe_log(exc), "block_id": _safe_log(block_id)},
        )
        return (500, {"error": "internal block store error"})

    # Pre-check existence so a missing-block delete returns 404 even when
    # the underlying store raises (e.g. Windows file-handle quirks under
    # the lock acquisition path) instead of returning False.
    try:
        if hasattr(store, "get_by_id") and store.get_by_id(block_id) is None:
            return (404, {"error": "block not found", "id": block_id})
    except Exception as exc:
        # Treat get_by_id errors as best-effort — fall through to the
        # delete path which handles a missing block via its own return
        # contract. Log at debug so the precheck failure is visible
        # without surfacing as a server error.
        _log.debug(
            "delete_memory_precheck_failed",
            extra={"error": _safe_log(exc), "block_id": _safe_log(block_id)},
        )

    try:
        removed = store.delete_block(block_id)
    except (FileNotFoundError, KeyError):
        return (404, {"error": "block not found", "id": block_id})
    except Exception as exc:
        _log.error(
            "delete_memory_failed",
            extra={"error": _safe_log(exc), "block_id": _safe_log(block_id)},
        )
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
            except Exception as block_exc:
                # One bad block must not abort the wipe — record the
                # failure at debug so the operator can investigate
                # individual failures without losing the bulk-clear.
                _log.debug(
                    "clear_block_skip",
                    extra={"block_id": _safe_log(bid), "error": _safe_log(block_exc)},
                )
                continue
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
    return (200, {"ok": True, "deleted": deleted, "rationale": rationale})


# ---------------------------------------------------------------------------
# Request dispatcher — single class, all endpoints
# ---------------------------------------------------------------------------


def _handle_fed_vclock(workspace: str, block_id: str) -> tuple[int, dict[str, Any]]:
    """GET /federation/vclock/<block_id> — read per-agent version vector.

    Returns 503 if v4.federation flag is disabled, 200 + dict otherwise.
    Missing block returns an empty version-vector dict (still 200) —
    callers treat that as "no writes yet seen".
    """
    try:
        from mind_mem.v4 import federation as fed
    except ImportError:
        return (503, {"ok": False, "error": "federation module unavailable"})
    try:
        vec = fed.get_version_vector(workspace, block_id)
    except Exception as exc:
        # FeatureDisabledError (and any other module-level guard) maps to 503.
        return (503, {"ok": False, "error": f"federation disabled: {exc}"})
    return (200, {"ok": True, "block_id": block_id, "version_vector": vec})


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
        return (503, {"ok": False, "error": f"federation disabled: {exc}"})
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


def _handle_fed_write(workspace: str, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """POST /federation/write {block_id, agent_id} — record agent write.

    Bumps the (block_id, agent_id) version atomically and reports a
    conflict if the resulting version vector diverges from another
    agent's claim. Auto-detects + logs the conflict to ``tier_conflict_log``.
    """
    try:
        from mind_mem.v4 import federation as fed
    except ImportError:
        return (503, {"ok": False, "error": "federation module unavailable"})
    block_id = body.get("block_id")
    agent_id = body.get("agent_id")
    if not isinstance(block_id, str) or not block_id:
        return (400, {"ok": False, "error": "block_id (string) is required"})
    if not isinstance(agent_id, str) or not agent_id:
        return (400, {"ok": False, "error": "agent_id (string) is required"})
    try:
        new_version = fed.record_agent_write(workspace, block_id, agent_id)
        report = fed.detect_conflict(workspace, block_id)
    except Exception as exc:
        return (503, {"ok": False, "error": f"federation disabled: {exc}"})
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


def _handle_fed_resolve(workspace: str, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """POST /federation/resolve {block_id, strategy, merged_payload?}.

    Applies the chosen MergeStrategy to the most-recent open conflict.
    For THREE_WAY_MERGE the caller supplies a merged_payload that is
    treated as the merge result; the function does not invoke a
    server-side merger callable.
    """
    try:
        from mind_mem.v4 import federation as fed
    except ImportError:
        return (503, {"ok": False, "error": "federation module unavailable"})
    block_id = body.get("block_id")
    strategy = body.get("strategy")
    merged_b64 = body.get("merged_payload")
    if not isinstance(block_id, str) or not block_id:
        return (400, {"ok": False, "error": "block_id (string) is required"})
    if not isinstance(strategy, str) or strategy not in {s.value for s in fed.MergeStrategy}:
        return (400, {"ok": False, "error": "strategy must be one of MergeStrategy values"})
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
    except Exception as exc:
        return (503, {"ok": False, "error": f"federation disabled: {exc}"})
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
            if base == PATH_FED_CONFLICTS:
                status, body = _handle_fed_conflicts(workspace, params)
                _write_json(self, status, body)
                return
            if base.startswith(_FED_VCLOCK_PREFIX):
                block_id = base[len(_FED_VCLOCK_PREFIX) :]
                if not block_id:
                    _write_status(self, 400, "block_id required")
                    return
                status, body = _handle_fed_vclock(workspace, block_id)
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
            if payload is None:
                # err==0 implies non-None payload, but be defensive
                # rather than assert — a stale handler shouldn't 500.
                _write_status(self, 400, "empty body")
                return
            if base == PATH_QUERY:
                status, body = _handle_query(workspace, payload)
                _write_json(self, status, body)
                return
            if base == PATH_CONSOLIDATE:
                status, body = _handle_consolidate(workspace, payload)
                _write_json(self, status, body)
                return
            if base == PATH_WALKTHROUGH:
                status, body = _handle_walkthrough(workspace, payload)
                _write_json(self, status, body)
                return
            if base == PATH_CLEAR:
                status, body = _handle_clear(workspace, payload)
                _write_json(self, status, body)
                return
            if base == PATH_FED_WRITE:
                status, body = _handle_fed_write(workspace, payload)
                _write_json(self, status, body)
                return
            if base == PATH_FED_RESOLVE:
                status, body = _handle_fed_resolve(workspace, payload)
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
