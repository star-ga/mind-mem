"""Tests for the v3.9 HTTP transport adapter.

Spins the server up against a real (tmp_path) workspace and exercises
each endpoint via stdlib ``http.client``. No external dependencies.
"""

from __future__ import annotations

import http.client
import json
import os
import socket
import threading
from pathlib import Path
from typing import Any

import pytest

from mind_mem.http_transport import (
    AUTH_HEADER,
    MAX_BODY_BYTES,
    PATH_CLEAR,
    PATH_CONSOLIDATE,
    PATH_MEMORIES,
    PATH_QUERY,
    PATH_STATUS,
    _parse_query_params,
    serve_http,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _free_port() -> int:
    """Bind to port 0 and release; the kernel hands us an ephemeral one."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _request(
    port: int,
    method: str,
    path: str,
    *,
    body: dict[str, Any] | None = None,
    token: str | None = None,
    raw_body: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, Any]]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    payload: bytes
    if raw_body is not None:
        payload = raw_body
    elif body is not None:
        payload = json.dumps(body).encode("utf-8")
    else:
        payload = b""
    hdrs: dict[str, str] = {"Content-Length": str(len(payload))}
    if payload:
        hdrs["Content-Type"] = "application/json"
    if token:
        hdrs[AUTH_HEADER] = token
    if headers:
        hdrs.update(headers)
    try:
        conn.request(method, path, body=payload, headers=hdrs)
        resp = conn.getresponse()
        raw = resp.read()
        try:
            parsed = json.loads(raw.decode("utf-8")) if raw else {}
        except json.JSONDecodeError:
            parsed = {"_raw": raw.decode("utf-8", "replace")}
        return (resp.status, parsed)
    finally:
        conn.close()


@pytest.fixture
def workspace(tmp_path: Path) -> str:
    """Build a minimal mind-mem workspace and return its path."""
    ws = tmp_path / "ws"
    (ws / "memory").mkdir(parents=True)
    (ws / "intelligence" / "state").mkdir(parents=True)
    (ws / "decisions").mkdir(parents=True)

    config = {
        "version": "3.9.0",
        "workspace_path": str(ws),
        "block_store": {"backend": "markdown"},
    }
    (ws / "mind-mem.json").write_text(json.dumps(config))

    # Drop a single decision block so /memories and /query have data.
    decisions_md = """# Decisions

## D-20260503-001 — example decision
- **Subject:** Test block for HTTP transport
- **Statement:** This is a sample memory used by test_http_transport.
- **Status:** active
- **Timestamp:** 2026-05-03T10:00:00
"""
    (ws / "decisions" / "DECISIONS.md").write_text(decisions_md)
    return str(ws)


@pytest.fixture
def server_localhost_unauth(workspace):
    """Server bound to 127.0.0.1 with auth disabled — for endpoint testing."""
    port = _free_port()
    thread, stop = serve_http(
        workspace=workspace,
        port=port,
        host="127.0.0.1",
        token=None,
        allow_unauthenticated_localhost=True,
    )
    try:
        yield port, thread
    finally:
        stop()


@pytest.fixture
def server_with_token(workspace):
    """Server with token auth required."""
    port = _free_port()
    thread, stop = serve_http(
        workspace=workspace,
        port=port,
        host="127.0.0.1",
        token="test-secret-12345",
        allow_unauthenticated_localhost=False,
    )
    try:
        yield port, "test-secret-12345"
    finally:
        stop()


# ---------------------------------------------------------------------------
# Helper unit tests
# ---------------------------------------------------------------------------


class TestParseQueryParams:
    def test_no_query(self) -> None:
        base, params = _parse_query_params("/foo")
        assert base == "/foo"
        assert params == {}

    def test_empty_query(self) -> None:
        base, params = _parse_query_params("/foo?")
        assert base == "/foo"
        assert params == {}

    def test_single_param(self) -> None:
        base, params = _parse_query_params("/foo?a=1")
        assert base == "/foo"
        assert params == {"a": "1"}

    def test_multi_param(self) -> None:
        base, params = _parse_query_params("/foo?a=1&b=two&c=")
        assert base == "/foo"
        assert params == {"a": "1", "b": "two", "c": ""}


# ---------------------------------------------------------------------------
# Server bootstrap tests
# ---------------------------------------------------------------------------


class TestServeHttpBootstrap:
    def test_refuses_no_token_no_loopback_bypass(self, workspace) -> None:
        os.environ.pop("MIND_MEM_TOKEN", None)
        with pytest.raises(ValueError, match="MIND_MEM_TOKEN"):
            serve_http(workspace=workspace, port=_free_port(), token=None)

    def test_token_from_env(self, workspace) -> None:
        os.environ["MIND_MEM_TOKEN"] = "env-token-9876"
        port = _free_port()
        try:
            thread, stop = serve_http(workspace=workspace, port=port)
            try:
                # bare GET without token must be rejected
                status, body = _request(port, "GET", PATH_STATUS)
                assert status == 401
            finally:
                stop()
        finally:
            os.environ.pop("MIND_MEM_TOKEN", None)

    def test_empty_workspace_rejected(self) -> None:
        with pytest.raises(ValueError, match="workspace"):
            serve_http(workspace="", token="x")


# ---------------------------------------------------------------------------
# Endpoint tests (auth-disabled server)
# ---------------------------------------------------------------------------


class TestStatusEndpoint:
    def test_status_returns_200(self, server_localhost_unauth) -> None:
        port, _ = server_localhost_unauth
        status, body = _request(port, "GET", PATH_STATUS)
        assert status == 200
        assert body["ok"] is True
        assert "workspace" in body
        assert "memory_count" in body
        assert isinstance(body["server_time"], int)

    def test_status_unknown_path_404(self, server_localhost_unauth) -> None:
        port, _ = server_localhost_unauth
        status, _ = _request(port, "GET", "/not-an-endpoint")
        assert status == 404


class TestQueryEndpoint:
    def test_missing_query_400(self, server_localhost_unauth) -> None:
        port, _ = server_localhost_unauth
        status, body = _request(port, "POST", PATH_QUERY, body={})
        assert status == 400
        assert "query" in body["error"]

    def test_empty_query_400(self, server_localhost_unauth) -> None:
        port, _ = server_localhost_unauth
        status, body = _request(port, "POST", PATH_QUERY, body={"query": "   "})
        assert status == 400

    def test_query_returns_results(self, server_localhost_unauth) -> None:
        port, _ = server_localhost_unauth
        status, body = _request(port, "POST", PATH_QUERY, body={"query": "test"})
        assert status == 200
        assert body["query"] == "test"
        assert "results" in body
        assert "count" in body

    def test_limit_out_of_range(self, server_localhost_unauth) -> None:
        port, _ = server_localhost_unauth
        status, _ = _request(port, "POST", PATH_QUERY, body={"query": "x", "limit": 9999})
        assert status == 400
        status, _ = _request(port, "POST", PATH_QUERY, body={"query": "x", "limit": 0})
        assert status == 400

    def test_limit_non_integer(self, server_localhost_unauth) -> None:
        port, _ = server_localhost_unauth
        status, _ = _request(port, "POST", PATH_QUERY, body={"query": "x", "limit": "ten"})
        assert status == 400

    def test_agent_id_must_be_string(self, server_localhost_unauth) -> None:
        port, _ = server_localhost_unauth
        status, _ = _request(port, "POST", PATH_QUERY, body={"query": "x", "agent_id": 123})
        assert status == 400


class TestMemoriesEndpoints:
    def test_list_default(self, server_localhost_unauth) -> None:
        port, _ = server_localhost_unauth
        status, body = _request(port, "GET", PATH_MEMORIES)
        assert status == 200
        assert "memories" in body
        assert isinstance(body["memories"], list)

    def test_list_active_only(self, server_localhost_unauth) -> None:
        port, _ = server_localhost_unauth
        status, body = _request(port, "GET", PATH_MEMORIES + "?active_only=true&limit=10")
        assert status == 200
        assert body["count"] <= 10

    def test_list_limit_invalid(self, server_localhost_unauth) -> None:
        port, _ = server_localhost_unauth
        status, _ = _request(port, "GET", PATH_MEMORIES + "?limit=abc")
        assert status == 400

    def test_list_limit_oversize(self, server_localhost_unauth) -> None:
        port, _ = server_localhost_unauth
        status, _ = _request(port, "GET", PATH_MEMORIES + "?limit=99999")
        assert status == 400

    def test_delete_invalid_id(self, server_localhost_unauth) -> None:
        port, _ = server_localhost_unauth
        # path-traversal attempts must be refused
        status, _ = _request(port, "DELETE", "/memories/../etc/passwd")
        assert status in (400, 404)

    def test_delete_missing_id(self, server_localhost_unauth) -> None:
        port, _ = server_localhost_unauth
        status, body = _request(port, "DELETE", "/memories/D-20990101-999")
        assert status == 404
        assert body["error"] == "block not found"


class TestConsolidateEndpoint:
    def test_dry_run(self, server_localhost_unauth) -> None:
        port, _ = server_localhost_unauth
        status, body = _request(port, "POST", PATH_CONSOLIDATE, body={"dry_run": True})
        assert status == 200
        assert body["ok"] is True
        assert body["dry_run"] is True


class TestClearEndpoint:
    def test_no_rationale_400(self, server_localhost_unauth) -> None:
        port, _ = server_localhost_unauth
        status, body = _request(port, "POST", PATH_CLEAR, body={})
        assert status == 400
        assert "rationale" in body["error"]

    def test_short_rationale_400(self, server_localhost_unauth) -> None:
        port, _ = server_localhost_unauth
        status, _ = _request(port, "POST", PATH_CLEAR, body={"rationale": "tooshort", "confirm": "yes-i-really-want-to-clear"})
        assert status == 400

    def test_missing_confirm_400(self, server_localhost_unauth) -> None:
        port, _ = server_localhost_unauth
        status, _ = _request(
            port,
            "POST",
            PATH_CLEAR,
            body={"rationale": "Cleaning up after test run abc123"},
        )
        assert status == 400

    def test_full_clear_succeeds(self, server_localhost_unauth) -> None:
        port, _ = server_localhost_unauth
        status, body = _request(
            port,
            "POST",
            PATH_CLEAR,
            body={
                "rationale": "End-to-end test wipe of throwaway workspace",
                "confirm": "yes-i-really-want-to-clear",
            },
        )
        assert status == 200
        assert body["ok"] is True
        assert isinstance(body["deleted"], int)


# ---------------------------------------------------------------------------
# Auth + body limit tests
# ---------------------------------------------------------------------------


class TestAuth:
    def test_request_without_token_rejected(self, server_with_token) -> None:
        port, _token = server_with_token
        status, _ = _request(port, "GET", PATH_STATUS)
        assert status == 401

    def test_request_with_wrong_token_rejected(self, server_with_token) -> None:
        port, _token = server_with_token
        status, _ = _request(port, "GET", PATH_STATUS, token="not-the-real-token")
        assert status == 401

    def test_request_with_correct_token_accepted(self, server_with_token) -> None:
        port, token = server_with_token
        status, body = _request(port, "GET", PATH_STATUS, token=token)
        assert status == 200
        assert body["ok"] is True


class TestBodyLimits:
    def test_body_too_large_413(self, server_localhost_unauth) -> None:
        port, _ = server_localhost_unauth
        # The server inspects Content-Length BEFORE reading the body and
        # refuses with 413 if it exceeds MAX_BODY_BYTES — so we can lie
        # about the length to test the gate without sending megabytes
        # (and tripping a broken-pipe race on the client side).
        small = b"{}"
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        try:
            conn.putrequest("POST", PATH_QUERY)
            conn.putheader("Content-Type", "application/json")
            conn.putheader("Content-Length", str(MAX_BODY_BYTES + 1))
            conn.endheaders()
            try:
                conn.send(small)
            except (BrokenPipeError, ConnectionResetError):
                pass  # server may have already closed; the response is what matters
            resp = conn.getresponse()
            assert resp.status == 413
        finally:
            conn.close()

    def test_invalid_json_body_400(self, server_localhost_unauth) -> None:
        port, _ = server_localhost_unauth
        status, _ = _request(port, "POST", PATH_QUERY, raw_body=b"{not valid json}")
        assert status == 400

    def test_non_object_body_400(self, server_localhost_unauth) -> None:
        port, _ = server_localhost_unauth
        status, _ = _request(port, "POST", PATH_QUERY, raw_body=b'["array", "not", "object"]')
        assert status == 400


# ---------------------------------------------------------------------------
# Threading / lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_thread_alive_after_start(self, server_localhost_unauth) -> None:
        port, thread = server_localhost_unauth
        assert isinstance(thread, threading.Thread)
        assert thread.is_alive()

    def test_stop_function_idempotent(self, workspace) -> None:
        port = _free_port()
        thread, stop = serve_http(
            workspace=workspace,
            port=port,
            host="127.0.0.1",
            token=None,
            allow_unauthenticated_localhost=True,
        )
        stop()
        # Calling stop again must not crash
        # (httpd.shutdown is a no-op after server_close; httpd.server_close is too)
        stop()
