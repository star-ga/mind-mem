"""Read-only local memory and graph viewer.

The viewer deliberately has no write route.  It reads through the same
backend-aware admission helpers as recall, and limits every operator supplied
value before it reaches a backend or the graph store.
"""

from __future__ import annotations

import json
import os
import socket
import sqlite3
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

MAX_BLOCKS = 10_000
MAX_QUERY = 256
MAX_ID = 512
MAX_PATH = 2_048
MAX_LIMIT = 50
MAX_DEPTH = 8

_CSP = (
    "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'none'; base-uri 'none'; frame-ancestors 'none'"
)
_LOOPBACK = frozenset({"127.0.0.1", "localhost", "::1"})


def _jsonable_block(block: dict[str, Any]) -> dict[str, Any]:
    """Expose stable block fields, never arbitrary backend internals."""
    out: dict[str, Any] = {}
    for key in ("_id", "Statement", "Summary", "Description", "Status", "Date", "Tags", "Type", "_source_file", "_source_label", "_line"):
        value = block.get(key)
        if value is not None:
            out[key] = value if isinstance(value, (str, int, float, bool, list, dict)) else str(value)
    return out


class ViewerState:
    """Request-time view over the configured, admitted corpus."""

    def __init__(self, workspace: str, agent_id: str | None = None) -> None:
        self.workspace = os.path.realpath(workspace)
        self.agent_id = agent_id

    def blocks(self) -> list[dict[str, Any]]:
        if self.agent_id:
            from .namespace_retrieval import admitted_namespace_blocks

            by_id = admitted_namespace_blocks(self.workspace, self.agent_id) or {}
            result = list(by_id.values())
        else:
            from .storage import iter_blocks

            result = iter_blocks(self.workspace, active_only=True)
        # Limit the searchable/displayed window while preserving source order.
        # Admission still enumerates the configured corpus before this slice;
        # this is not a bound on backend I/O or process memory.
        return result[:MAX_BLOCKS]

    def by_id(self, block_id: str) -> dict[str, Any] | None:
        for block in self.blocks():
            if str(block.get("_id", "")) == block_id:
                return block
        return None

    def search(self, query: str, limit: int) -> list[dict[str, Any]]:
        terms = [term for term in query.casefold().split() if term]
        if not terms:
            return []
        scored: list[tuple[int, int, dict[str, Any]]] = []
        for order, block in enumerate(self.blocks()):
            text = " ".join(str(block.get(key, "")) for key in ("_id", "Statement", "Summary", "Description", "Tags")).casefold()
            score = sum(text.count(term) for term in terms)
            if score:
                scored.append((score, order, block))
        scored.sort(key=lambda row: (-row[0], row[1]))
        return [{**_jsonable_block(block), "score": score} for score, _order, block in scored[:limit]]


class ViewerHandler(BaseHTTPRequestHandler):
    """Strict GET-only handler; assigned ``state`` and ``bound_host`` by factory."""

    server_version = "mind-mem-viewer/1"
    state: ViewerState
    bound_host: str

    def log_message(self, _format: str, *_args: Any) -> None:
        return

    def _security_headers(self) -> None:
        self.send_header("Content-Security-Policy", _CSP)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cache-Control", "no-store")

    def _reject(self, status: HTTPStatus, message: str) -> None:
        body = json.dumps({"error": message}, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self._security_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _origin_allowed(self) -> bool:
        host = self.headers.get("Host", "")
        server_port = int(getattr(self.server, "server_port", 0))
        authority_host = f"[{self.bound_host}]" if self.bound_host == "::1" else self.bound_host
        expected_host = f"{authority_host}:{server_port}"
        valid_hosts = {expected_host}
        if self.bound_host == "127.0.0.1":
            valid_hosts.add(f"localhost:{server_port}")
        if host not in valid_hosts:
            return False
        origin = self.headers.get("Origin")
        if origin is None:
            return True
        return origin in {f"http://{item}" for item in valid_hosts}

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")
        self.send_response(status)
        self._security_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _query(self, query: str, name: str, max_len: int) -> str | None:
        values = parse_qs(query, keep_blank_values=True, strict_parsing=False).get(name, [])
        if len(values) > 1 or (values and len(values[0]) > max_len):
            raise ValueError(f"{name} is too long or repeated")
        return values[0] if values else None

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        if not self._origin_allowed():
            self._reject(HTTPStatus.FORBIDDEN, "host or origin is not allowed")
            return
        if len(self.path) > MAX_PATH or any(ord(ch) < 0x20 for ch in self.path):
            self._reject(HTTPStatus.BAD_REQUEST, "request path is invalid")
            return
        # ``BaseHTTPRequestHandler`` normalizes a request-target beginning
        # with ``//`` before exposing ``self.path``. Inspect the raw line so
        # an authority-form target cannot be mistaken for a local path.
        raw_requestline = getattr(self, "raw_requestline", b"")
        raw_parts = raw_requestline.split(None, 2)
        if len(raw_parts) >= 2:
            try:
                raw_target = raw_parts[1].decode("ascii")
            except UnicodeDecodeError:
                self._reject(HTTPStatus.BAD_REQUEST, "request path is invalid")
                return
            if raw_target.startswith("//"):
                self._reject(HTTPStatus.BAD_REQUEST, "request path is invalid")
                return
        try:
            parsed = urlsplit(self.path)
        except ValueError:
            self._reject(HTTPStatus.BAD_REQUEST, "request path is invalid")
            return
        if parsed.fragment or parsed.scheme or parsed.netloc or not parsed.path.startswith("/"):
            self._reject(HTTPStatus.BAD_REQUEST, "request path is invalid")
            return
        try:
            if parsed.path in {"/", "/index.html"}:
                self._asset(parsed.path)
            elif parsed.path == "/app.js" or parsed.path == "/style.css":
                self._asset(parsed.path)
            elif parsed.path == "/api/blocks":
                limit = int(self._query(parsed.query, "limit", 3) or "50")
                if not 1 <= limit <= MAX_LIMIT:
                    raise ValueError("limit must be in [1, 50]")
                blocks = self.state.blocks()
                self._send_json(
                    {
                        "blocks": [_jsonable_block(block) for block in blocks[:limit]],
                        "count": min(limit, len(blocks)),
                        "truncated": len(blocks) > limit,
                        "principal": self.state.agent_id,
                    }
                )
            elif parsed.path == "/api/search":
                query = self._query(parsed.query, "q", MAX_QUERY)
                if not query:
                    raise ValueError("q is required")
                limit = int(self._query(parsed.query, "limit", 3) or "20")
                if not 1 <= limit <= MAX_LIMIT:
                    raise ValueError("limit must be in [1, 50]")
                hits = self.state.search(query, limit)
                self._send_json({"query": query, "hits": hits, "count": len(hits), "principal": self.state.agent_id})
            elif parsed.path.startswith("/api/block/"):
                raw_id = parsed.path[len("/api/block/") :]
                block_id = unquote(raw_id)
                if not block_id or len(block_id) > MAX_ID or any(ch in block_id for ch in "/\\") or block_id in {".", ".."}:
                    raise LookupError
                block = self.state.by_id(block_id)
                if block is None:
                    self._reject(HTTPStatus.NOT_FOUND, "block not found")
                else:
                    self._send_json({"block": _jsonable_block(block), "principal": self.state.agent_id})
            elif parsed.path == "/api/graph":
                entity = self._query(parsed.query, "entity", MAX_ID)
                if not entity:
                    raise ValueError("entity is required")
                depth = int(self._query(parsed.query, "depth", 2) or "1")
                if not 1 <= depth <= MAX_DEPTH:
                    raise ValueError("depth must be in [1, 8]")
                self._graph(entity, depth)
            else:
                self._reject(HTTPStatus.NOT_FOUND, "route not found")
        except (ValueError, TypeError):
            self._reject(HTTPStatus.BAD_REQUEST, "request parameters are invalid")
        except LookupError:
            self._reject(HTTPStatus.NOT_FOUND, "block not found")
        except (OSError, UnicodeError, sqlite3.Error):
            self._reject(HTTPStatus.SERVICE_UNAVAILABLE, "configured corpus is unavailable")

    def do_POST(self) -> None:  # noqa: N802
        self._reject_method()

    def _reject_method(self) -> None:
        """Reject registered non-GET methods through the guarded JSON path."""
        if not self._origin_allowed():
            self._reject(HTTPStatus.FORBIDDEN, "host or origin is not allowed")
            return
        self._reject(HTTPStatus.METHOD_NOT_ALLOWED, "viewer is read-only")

    # BaseHTTPRequestHandler's default 501 response is HTML and omits the
    # viewer security headers.  Keep the local surface consistently JSON and
    # read-only for the common methods listed below.
    do_PUT = _reject_method
    do_DELETE = _reject_method
    do_PATCH = _reject_method
    do_OPTIONS = _reject_method
    do_HEAD = _reject_method
    do_CONNECT = _reject_method
    do_TRACE = _reject_method

    def _asset(self, path: str) -> None:
        resource_name = "index.html" if path in {"/", "/index.html"} else path[1:]
        resource = files("mind_mem").joinpath("viewer_static").joinpath(resource_name)
        if not resource.is_file():
            self._reject(HTTPStatus.NOT_FOUND, "asset not found")
            return
        body = resource.read_bytes()
        content_type = {
            "index.html": "text/html; charset=utf-8",
            "app.js": "text/javascript; charset=utf-8",
            "style.css": "text/css; charset=utf-8",
        }[resource_name]
        self.send_response(HTTPStatus.OK)
        self._security_headers()
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _graph(self, entity: str, depth: int) -> None:
        if self.state.agent_id:
            # KnowledgeGraph edges currently have no namespace-bound source
            # identity, so exposing them under an agent principal would be an
            # ACL guess.  Keep the limitation explicit until the graph API
            # gains the same source-bound admission contract as recall.
            self._send_json(
                {"error": "namespace-scoped graph viewing is unsupported until graph edges are source-bound"},
                HTTPStatus.NOT_IMPLEMENTED,
            )
            return
        from .knowledge_graph import KnowledgeGraph
        from .mcp.tools._helpers import _kg_path

        path = _kg_path(self.state.workspace)
        if not os.path.isfile(path):
            self._send_json({"entity": entity, "neighbors": [], "available": False, "reason": "knowledge graph is not initialized"})
            return
        kg = None
        try:
            kg = KnowledgeGraph.open_read_only(path)
            canonical = kg.entities.lookup(entity)
            if canonical is None:
                self._send_json({"entity": entity, "neighbors": [], "available": True, "reason": "entity is unknown"})
                return
            neighbors = kg.neighbors(canonical, depth=depth, direction="both", max_results=MAX_LIMIT)
        except sqlite3.Error:
            self._send_json(
                {"entity": entity, "neighbors": [], "available": False, "reason": "knowledge graph is unavailable"},
                HTTPStatus.SERVICE_UNAVAILABLE,
            )
            return
        finally:
            if kg is not None:
                kg.close()
        self._send_json({"entity": entity, "neighbors": neighbors, "available": True, "depth": depth})


class ViewerServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class _IPv6ViewerServer(ViewerServer):
    address_family = socket.AF_INET6


def make_server(workspace: str, host: str = "127.0.0.1", port: int = 8765, agent_id: str | None = None) -> ViewerServer:
    """Build the local viewer server without starting a thread."""
    if host not in _LOOPBACK:
        raise ValueError("mm view only supports loopback hosts; remote binding is unsupported")
    if not 0 <= port <= 65_535:
        raise ValueError("port must be in [0, 65535]")
    if agent_id is not None:
        from .namespaces import _validate_agent_id

        _validate_agent_id(agent_id)
    handler = type(
        "ConfiguredViewerHandler",
        (ViewerHandler,),
        {"state": ViewerState(workspace, agent_id), "bound_host": host},
    )
    server_type = _IPv6ViewerServer if host == "::1" else ViewerServer
    server = server_type((host, port), handler)
    return server


def serve(workspace: str, host: str, port: int, agent_id: str | None = None) -> None:
    """Start the blocking local viewer service."""
    server = make_server(workspace, host, port, agent_id)
    scheme_host = f"[{host}]" if ":" in host else host
    print(f"mind-mem view: http://{scheme_host}:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
