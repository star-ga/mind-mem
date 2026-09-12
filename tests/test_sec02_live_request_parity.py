# Copyright 2026 STARGA, Inc.
"""SEC02 over a REAL SOCKET — a declaration control is not evidence of enforcement.

Root's review of the first SEC02 slice found four defects with ONE root cause:
the 11 controls inspected the route table and module source, and not one issued
a request. That single habit produced all of it --

  * they would have survived a dispatcher early bypass;
  * a typo scope="admn" passed every one of them;
  * they implied an import-time parity that only the tests asserted; and
  * they missed that MEASURED, with the documented configuration
    MIND_MEM_TOKEN=user / MIND_MEM_ADMIN_TOKEN=admin, _active_tokens() returned
    ['user'] only -- so the admin passed _caller_is_admin and then FAILED
    AUTHENTICATION before reaching it. The fix locked the admin OUT.

These go over a loopback socket.
"""

from __future__ import annotations

import contextlib
import http.client
import json
import socket
import urllib.error
import urllib.request
from collections.abc import Iterator

import pytest

from mind_mem.http_transport import PATH_CLEAR, PATH_STATUS, serve_http
from mind_mem.protection import AUTH_HEADER

USER = "user-token-value"
ADMIN = "admin-token-value"


def _free_port() -> int:
    with contextlib.closing(socket.socket()) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@contextlib.contextmanager
def _serve(workspace: str, *, token: str | None) -> Iterator[int]:
    port = _free_port()
    _thread, stop = serve_http(
        workspace=workspace,
        port=port,
        host="127.0.0.1",
        token=token,
        allow_unauthenticated_localhost=token is None,
    )
    try:
        yield port
    finally:
        stop()


def _post(port: int, path: str, token: str | None, body: dict) -> int:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", **({AUTH_HEADER: token} if token else {})},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return int(r.status)
    except urllib.error.HTTPError as e:
        return int(e.code)


def _get(port: int, path: str, token: str | None) -> int:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        headers={AUTH_HEADER: token} if token else {},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return int(r.status)
    except urllib.error.HTTPError as e:
        return int(e.code)


def _raw_status(port: int, request: bytes) -> str:
    """Send a deliberately raw request, including bytes urllib rejects."""
    with socket.create_connection(("127.0.0.1", port), timeout=10) as conn:
        conn.sendall(request)
        conn.shutdown(socket.SHUT_WR)
        response = b""
        while True:
            chunk = conn.recv(65536)
            if not chunk:
                break
            response += chunk
    return response.split(b"\r\n", 1)[0].decode("latin-1", "replace")


@pytest.fixture
def separated(tmp_path, monkeypatch):
    """The documented separated configuration."""
    monkeypatch.delenv("MIND_MEM_TOKENS", raising=False)
    monkeypatch.setenv("MIND_MEM_TOKEN", USER)
    monkeypatch.setenv("MIND_MEM_ADMIN_TOKEN", ADMIN)
    return str(tmp_path)


def test_the_admin_can_actually_authenticate(separated):
    """THE DEFECT, over a socket. Before the fix the admin got 401 on a route
    it was authorised for, because authentication never learned the credential."""
    with _serve(separated, token=USER) as port:
        assert _get(port, PATH_STATUS, ADMIN) == 200, (
            "the admin credential failed AUTHENTICATION; authorisation was built on a path that did not know it exists"
        )


def test_a_user_token_cannot_reach_a_mutating_admin_route(separated):
    with _serve(separated, token=USER) as port:
        assert _get(port, PATH_STATUS, USER) == 200, "positive control: the user is authenticated"
        code = _post(port, PATH_CLEAR, USER, {"confirm": "DELETE ALL MEMORIES", "rationale": "sec02 live control"})
        assert code == 404, f"user token reached {PATH_CLEAR} and got {code}"


def test_the_admin_token_does_reach_it(separated):
    """Positive control for the negative above: without this, a broken route
    would pass the previous test for the wrong reason."""
    with _serve(separated, token=USER) as port:
        code = _post(port, PATH_CLEAR, ADMIN, {"confirm": "wrong-confirm-string", "rationale": "sec02 live control"})
        assert code != 404, "the admin was refused the admin route; the guard denies everyone"
        assert code == 400, f"expected the handler's own confirm check to answer, got {code}"


def test_legacy_single_token_keeps_full_access(tmp_path, monkeypatch):
    """Documented compatibility: one credential, no separation, full access."""
    monkeypatch.delenv("MIND_MEM_TOKENS", raising=False)
    monkeypatch.delenv("MIND_MEM_ADMIN_TOKEN", raising=False)
    monkeypatch.setenv("MIND_MEM_TOKEN", USER)
    with _serve(str(tmp_path), token=USER) as port:
        code = _post(port, PATH_CLEAR, USER, {"confirm": "wrong-confirm-string", "rationale": "sec02 live control"})
        assert code == 400, f"legacy full access regressed: {code}"


def test_a_malformed_admin_config_fails_closed_rather_than_becoming_legacy(tmp_path, monkeypatch):
    """An operator who INTENDED separation and mis-typed it must not silently
    get legacy full access -- that failure would look exactly like success."""
    monkeypatch.delenv("MIND_MEM_TOKENS", raising=False)
    monkeypatch.setenv("MIND_MEM_TOKEN", USER)
    monkeypatch.setenv("MIND_MEM_ADMIN_TOKEN", "   ")
    with _serve(str(tmp_path), token=USER) as port:
        assert _get(port, PATH_STATUS, USER) == 200, "positive control: user routes still work"
        code = _post(port, PATH_CLEAR, USER, {"confirm": "DELETE ALL MEMORIES", "rationale": "sec02 live control"})
        assert code == 404, f"malformed admin config degraded to legacy full access: {code}"


def test_malformed_admin_config_never_turns_into_a_raw_header_credential(tmp_path, monkeypatch):
    """A malformed setting is state, not a synthetic bearer token.

    BaseHTTPRequestHandler accepts NUL bytes in a raw header value. The old
    NUL-bearing sentinel therefore was replayable over the network and could
    authenticate as an admin. The malformed configuration must deny it at the
    authentication boundary before route authorization is considered.
    """
    monkeypatch.delenv("MIND_MEM_TOKENS", raising=False)
    monkeypatch.setenv("MIND_MEM_TOKEN", USER)
    monkeypatch.setenv("MIND_MEM_ADMIN_TOKEN", "   ")
    raw_sentinel = b"\x00mind-mem/admin-token-set-but-empty\x00"
    with _serve(str(tmp_path), token=USER) as port:
        request = (
            b"POST /clear HTTP/1.1\r\nHost: 127.0.0.1\r\n"
            + AUTH_HEADER.encode("ascii")
            + b": "
            + raw_sentinel
            + b"\r\nContent-Type: application/json\r\n"
            + b"Content-Length: 46\r\nConnection: close\r\n\r\n"
            + b'{"confirm":"wrong","rationale":"raw sentinel"}'
        )
        assert _raw_status(port, request) == "HTTP/1.0 401 Unauthorized"


def test_auth_and_scope_use_one_admin_snapshot(separated, monkeypatch):
    """Rotation between auth and dispatch cannot change the request's role."""
    from mind_mem import http_transport

    reads: list[int] = []

    def rotating_admin_state():
        reads.append(1)
        if len(reads) == 1:
            return http_transport._AdminAuthState((ADMIN,), True)
        return http_transport._AdminAuthState((USER,), True)

    monkeypatch.setattr(http_transport, "_read_admin_auth_state", rotating_admin_state)
    with _serve(separated, token=USER) as port:
        code = _post(port, PATH_CLEAR, ADMIN, {"confirm": "wrong-confirm-string", "rationale": "snapshot control"})
    assert code == 400, f"the role changed after authentication (got {code})"
    assert reads == [1], f"request read mutable admin configuration {len(reads)} times"


def test_auth_snapshot_rotates_per_keepalive_request(separated, monkeypatch):
    """A persistent HTTP/1.1 connection must not pin the first token forever.

    ``serve_http`` defaults to HTTP/1.0 for compatibility, so this explicitly
    exercises the supported handler mode in which BaseHTTPRequestHandler
    serves multiple requests on one connection.  The old token is accepted by
    the first request, refused after rotation, and the new token is accepted
    immediately on that same socket.
    """
    from mind_mem import http_transport

    original_build_handler = http_transport.build_handler

    def build_http11_handler(*args, **kwargs):
        handler = original_build_handler(*args, **kwargs)
        handler.protocol_version = "HTTP/1.1"
        return handler

    monkeypatch.setattr(http_transport, "build_handler", build_http11_handler)
    with _serve(separated, token=USER) as port:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        try:
            headers = {AUTH_HEADER: USER, "Connection": "keep-alive"}
            conn.request("GET", PATH_STATUS, headers=headers)
            first = conn.getresponse()
            assert first.status == 200
            first.read()

            monkeypatch.setenv("MIND_MEM_TOKEN", "rotated-user-token")
            conn.request("GET", PATH_STATUS, headers=headers)
            old_token = conn.getresponse()
            assert old_token.status == 401, "rotated-out token remained valid on keep-alive connection"
            old_token.read()

            conn.request(
                "GET",
                PATH_STATUS,
                headers={AUTH_HEADER: "rotated-user-token", "Connection": "keep-alive"},
            )
            new_token = conn.getresponse()
            assert new_token.status == 200, "new token was not read on the next keep-alive request"
            new_token.read()
        finally:
            conn.close()


def test_all_expired_plural_set_cannot_fall_back_to_singular(separated, monkeypatch):
    """An expired configured list closes the door rather than reviving fallback."""
    from mind_mem import http_transport

    monkeypatch.setenv("MIND_MEM_TOKENS", "retired|exp=100")
    monkeypatch.setattr(http_transport.time, "time", lambda: 101.0)
    with _serve(separated, token=USER) as port:
        assert _get(port, PATH_STATUS, USER) == 401


def test_all_expired_configuration_cannot_enable_loopback_anonymous_access(tmp_path, monkeypatch):
    """The loopback opt-in must not bypass a configured but expired set."""
    monkeypatch.delenv("MIND_MEM_TOKEN", raising=False)
    monkeypatch.setenv("MIND_MEM_TOKENS", "retired|exp=100")
    monkeypatch.delenv("MIND_MEM_ADMIN_TOKEN", raising=False)
    from mind_mem import http_transport

    monkeypatch.setattr(http_transport.time, "time", lambda: 101.0)
    with _serve(str(tmp_path), token=None) as port:
        assert _get(port, PATH_STATUS, None) == 401


def test_handler_fallback_token_expiry_is_enforced(separated, monkeypatch):
    """An explicit handler fallback follows the same expiry contract as env tokens."""
    from mind_mem import http_transport

    monkeypatch.delenv("MIND_MEM_TOKENS", raising=False)
    monkeypatch.delenv("MIND_MEM_TOKEN", raising=False)
    monkeypatch.delenv("MIND_MEM_ADMIN_TOKEN", raising=False)
    current = [50.0]
    monkeypatch.setattr(http_transport.time, "time", lambda: current[0])
    with _serve(separated, token="fallback|exp=50") as port:
        assert _get(port, PATH_STATUS, "fallback") == 200
        current[0] = 51.0
        assert _get(port, PATH_STATUS, "fallback") == 401


def test_expiry_uses_one_clock_and_preserves_admin_scope(separated, monkeypatch):
    """Auth and admin authorization must use one request timestamp/snapshot."""
    from mind_mem import http_transport

    monkeypatch.setenv("MIND_MEM_ADMIN_TOKEN", f"{ADMIN}|exp=1000000000000")
    captures: list[int] = []
    original_capture = http_transport._capture_auth_snapshot

    def capture(*, fallback):
        captures.append(1)
        return original_capture(fallback=fallback)

    monkeypatch.setattr(http_transport, "_capture_auth_snapshot", capture)
    with _serve(separated, token=USER) as port:
        code = _post(port, PATH_CLEAR, ADMIN, {"confirm": "wrong-confirm-string", "rationale": "expiry snapshot"})
    assert code == 400, "admin token valid at the inclusive deadline must reach the handler"
    assert captures == [1], f"request captured authentication {len(captures)} times"


def test_malformed_admin_expiry_keeps_admin_route_closed(separated, monkeypatch):
    """A malformed admin entry is not a reason to disable privilege separation."""
    monkeypatch.setenv("MIND_MEM_ADMIN_TOKEN", f"{ADMIN}|ttl=1000")
    with _serve(separated, token=USER) as port:
        assert _get(port, PATH_STATUS, USER) == 200
        assert _post(port, PATH_CLEAR, USER, {"confirm": "wrong-confirm-string", "rationale": "malformed admin"}) == 404


def test_expiry_refreshes_on_each_keepalive_request(separated, monkeypatch):
    """Expiry changes must take effect on the next request on a reused socket."""
    from mind_mem import http_transport

    monkeypatch.setenv("MIND_MEM_TOKENS", "old|exp=100,new-token")
    current = [100.0]
    monkeypatch.setattr(http_transport.time, "time", lambda: current[0])
    original_build_handler = http_transport.build_handler

    def build_http11_handler(*args, **kwargs):
        handler = original_build_handler(*args, **kwargs)
        handler.protocol_version = "HTTP/1.1"
        return handler

    monkeypatch.setattr(http_transport, "build_handler", build_http11_handler)
    with _serve(separated, token=USER) as port:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        try:
            headers = {AUTH_HEADER: "old", "Connection": "keep-alive"}
            conn.request("GET", PATH_STATUS, headers=headers)
            first = conn.getresponse()
            assert first.status == 200
            first.read()

            current[0] = 101.0
            conn.request("GET", PATH_STATUS, headers=headers)
            expired = conn.getresponse()
            assert expired.status == 401
            expired.read()

            conn.request("GET", PATH_STATUS, headers={AUTH_HEADER: "new-token", "Connection": "keep-alive"})
            new = conn.getresponse()
            assert new.status == 200
            new.read()
        finally:
            conn.close()


def test_loopback_anonymous_mode_tracks_credentials_per_keepalive_request(tmp_path, monkeypatch):
    """Adding then expiring process-env credentials changes the next request."""
    from mind_mem import http_transport

    for name in ("MIND_MEM_TOKENS", "MIND_MEM_TOKEN", "MIND_MEM_ADMIN_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    current = [100.0]
    monkeypatch.setattr(http_transport.time, "time", lambda: current[0])
    original_build_handler = http_transport.build_handler

    def build_http11_handler(*args, **kwargs):
        handler = original_build_handler(*args, **kwargs)
        handler.protocol_version = "HTTP/1.1"
        return handler

    monkeypatch.setattr(http_transport, "build_handler", build_http11_handler)
    with _serve(str(tmp_path), token=None) as port:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        try:
            headers = {"Connection": "keep-alive"}
            conn.request("GET", PATH_STATUS, headers=headers)
            anonymous = conn.getresponse()
            assert anonymous.status == 200
            anonymous.read()

            monkeypatch.setenv("MIND_MEM_TOKENS", "live-token")
            conn.request("GET", PATH_STATUS, headers=headers)
            configured = conn.getresponse()
            assert configured.status == 401
            configured.read()

            conn.request("GET", PATH_STATUS, headers={**headers, AUTH_HEADER: "live-token"})
            authenticated = conn.getresponse()
            assert authenticated.status == 200
            authenticated.read()

            monkeypatch.setenv("MIND_MEM_TOKENS", "live-token|exp=100")
            current[0] = 101.0
            conn.request("GET", PATH_STATUS, headers={**headers, AUTH_HEADER: "live-token"})
            expired = conn.getresponse()
            assert expired.status == 401
            expired.read()

            conn.request("GET", PATH_STATUS, headers=headers)
            no_longer_anonymous = conn.getresponse()
            assert no_longer_anonymous.status == 401
            no_longer_anonymous.read()
        finally:
            conn.close()


def test_startup_accepts_plural_and_admin_only_credentials(tmp_path, monkeypatch):
    """The startup gate must recognize every transport credential source."""
    from mind_mem import http_transport

    monkeypatch.delenv("MIND_MEM_TOKEN", raising=False)
    monkeypatch.setenv("MIND_MEM_TOKENS", "plural-token")
    monkeypatch.delenv("MIND_MEM_ADMIN_TOKEN", raising=False)
    port = _free_port()
    _thread, stop = http_transport.serve_http(workspace=str(tmp_path), port=port, token=None, allow_unauthenticated_localhost=False)
    try:
        assert _get(port, PATH_STATUS, "plural-token") == 200
    finally:
        stop()

    monkeypatch.delenv("MIND_MEM_TOKENS", raising=False)
    monkeypatch.setenv("MIND_MEM_ADMIN_TOKEN", ADMIN)
    port = _free_port()
    _thread, stop = http_transport.serve_http(workspace=str(tmp_path), port=port, token=None, allow_unauthenticated_localhost=False)
    try:
        assert _get(port, PATH_STATUS, ADMIN) == 200
        assert _get(port, PATH_STATUS, USER) == 401
    finally:
        stop()


# ---------------------------------------------------------------------------
# MUTATION CONTROLS. Root's finding was that source-inspection controls "would
# survive a dispatcher early bypass" -- i.e. they could not tell an enforced
# guard from a decorative one. These two break the guard in different ways and
# require the socket behaviour to change, so a control that cannot fail is
# visible as one that does not go red here.
# ---------------------------------------------------------------------------


def test_removing_the_guard_lets_the_user_through(separated, monkeypatch):
    """MUTATION 1: neuter _caller_is_admin. The user must then REACH the route.

    If this does not change the answer, the guard is not what is refusing the
    user and the negative control above proves nothing.
    """
    from mind_mem import http_transport

    monkeypatch.setattr(http_transport, "_caller_is_admin", lambda presented, active_admin: True)
    with _serve(separated, token=USER) as port:
        code = _post(port, PATH_CLEAR, USER, {"confirm": "wrong-confirm-string", "rationale": "sec02 mutation"})
    assert code == 400, (
        f"with the guard neutered the user still could not reach {PATH_CLEAR} (got {code}); "
        "something other than the scope guard is refusing, so the guard is untested"
    )


def test_emptying_the_admin_set_also_lets_the_user_through(separated, monkeypatch):
    """MUTATION 2: make the admin set empty at request time.

    Enforcement is conditional on an admin credential existing, so emptying it
    must reopen the route -- proving the CONDITION is what gates enforcement,
    not some unrelated refusal.
    """
    from mind_mem import http_transport

    monkeypatch.setattr(
        http_transport,
        "_capture_auth_snapshot",
        lambda *, fallback: http_transport._RequestAuthSnapshot((USER,), (), False),
    )
    with _serve(separated, token=USER) as port:
        code = _post(port, PATH_CLEAR, USER, {"confirm": "wrong-confirm-string", "rationale": "sec02 mutation"})
    assert code == 400, f"emptying the admin set did not reopen the route (got {code})"


def test_a_misspelled_scope_cannot_be_constructed_at_all():
    """MUTATION 3, and the reason it is not a socket test: the typo is now
    refused at IMPORT, so a route with scope='admn' cannot exist to be served.

    Checked independently of the guard, per root: the two failures are
    different (a typo that constructs and silently fails the comparison, versus
    a guard that does not run) and one must not mask the other.
    """
    from mind_mem.http_transport import NO_CONTENT, Route, _handle_status

    with pytest.raises(ValueError, match="scope"):
        Route(
            "GET",
            "/x",
            _handle_status,
            "workspace",
            NO_CONTENT,
            mutates=False,
            scope="admn",
            scope_reason="a reason long enough to satisfy the no-twin rule",
        )
