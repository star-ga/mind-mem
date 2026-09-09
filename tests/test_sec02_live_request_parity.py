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

    monkeypatch.setattr(http_transport, "_active_admin_tokens", lambda: [])
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
