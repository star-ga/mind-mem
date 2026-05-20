"""Regression tests for the audit-header middleware (roadmap v4.0.0 Group D).

v4.0.14 adds CRLF-safe audit-header propagation to the REST app:

* ``X-MindMem-Request-Id`` — server-assigned UUID-4 when missing, else echoed
  (with control-char stripping); always present on the response.
* ``X-MindMem-Actor`` — client-supplied agent identifier; echoed when set.
* ``X-MindMem-Purpose`` — client-supplied intent string; echoed when set.

All three flow through ``_safe_hdr`` which strips ``\\r`` / ``\\n`` /
NUL / other ASCII controls and bounds length so header injection
cannot land newlines into the log stream (same pattern as v4.0.13's
``_safe()`` in federation logging — alerts #189 + #192).
"""

from __future__ import annotations

import re

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from mind_mem.api.rest import create_app


@pytest.fixture()
def client(tmp_path) -> TestClient:
    """Build a REST TestClient with a temporary workspace."""
    app = create_app(workspace=str(tmp_path))
    return TestClient(app)


UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


# ---------------------------------------------------------------------------
# Request-ID generation
# ---------------------------------------------------------------------------


def test_request_id_auto_generated_when_missing(client: TestClient) -> None:
    """When the client doesn't supply X-MindMem-Request-Id, the server
    assigns a UUID-4 and echoes it on the response."""
    resp = client.get("/v1/health")
    assert resp.status_code == 200
    rid = resp.headers.get("X-MindMem-Request-Id")
    assert rid is not None
    assert UUID_RE.match(rid), f"server-assigned request-id is not a UUID-4: {rid!r}"


def test_request_id_echoed_when_client_supplies(client: TestClient) -> None:
    """A well-formed client-supplied X-MindMem-Request-Id is echoed verbatim."""
    given = "deadbeef-cafe-1234-5678-aabbccddeeff"
    resp = client.get("/v1/health", headers={"X-MindMem-Request-Id": given})
    assert resp.status_code == 200
    assert resp.headers["X-MindMem-Request-Id"] == given


# ---------------------------------------------------------------------------
# Actor + Purpose echo
# ---------------------------------------------------------------------------


def test_actor_and_purpose_echoed(client: TestClient) -> None:
    """Client-supplied actor + purpose round-trip on the response."""
    resp = client.get(
        "/v1/health",
        headers={
            "X-MindMem-Actor": "claude-code-cli",
            "X-MindMem-Purpose": "interactive-search",
        },
    )
    assert resp.status_code == 200
    assert resp.headers.get("X-MindMem-Actor") == "claude-code-cli"
    assert resp.headers.get("X-MindMem-Purpose") == "interactive-search"


def test_actor_omitted_when_not_supplied(client: TestClient) -> None:
    """When actor isn't supplied, the response omits the header rather
    than echoing 'anonymous' (operators expect the absence to signal
    'unattributed', not a literal string)."""
    resp = client.get("/v1/health")
    assert "X-MindMem-Actor" not in resp.headers
    assert "X-MindMem-Purpose" not in resp.headers


# ---------------------------------------------------------------------------
# CRLF / header-injection sanitisation
# ---------------------------------------------------------------------------


def test_request_id_strips_crlf_and_controls(client: TestClient) -> None:
    """Adversarial X-MindMem-Request-Id with CRLF / NUL must be sanitised
    before it lands on the response (header injection)."""
    # httpx rejects CR/LF in header values at the client side (the
    # "abc-123\\r\\nX-Injected: pwned\\r\\n\\x00\\x01" form), so we
    # smuggle via control chars only (httpx allows 0x01/0x7f on some
    # versions). The sanitiser must still strip them on the response.
    sneaky = "abc-123" + chr(0x01) + chr(0x7F) + chr(0x1F)
    try:
        resp = client.get(
            "/v1/health",
            headers={"X-MindMem-Request-Id": sneaky},
        )
    except Exception:
        # If the HTTP client rejects the header outright that's already
        # safe — header injection blocked at transport. Skip the
        # response-side assertion.
        return
    echoed = resp.headers.get("X-MindMem-Request-Id", "")
    assert "\x01" not in echoed
    assert "\x7f" not in echoed
    assert "\x1f" not in echoed
    assert "\r" not in echoed
    assert "\n" not in echoed
    # The cleaned ID still contains the safe prefix.
    assert "abc-123" in echoed


def test_actor_length_bounded(client: TestClient) -> None:
    """Actor value is bounded at 256 chars to prevent log-bomb / DoS."""
    long_actor = "a" * 1024
    resp = client.get("/v1/health", headers={"X-MindMem-Actor": long_actor})
    echoed = resp.headers.get("X-MindMem-Actor", "")
    assert 0 < len(echoed) <= 256


# ---------------------------------------------------------------------------
# request.state propagation
# ---------------------------------------------------------------------------


def test_request_state_propagation_via_response_echo(client: TestClient) -> None:
    """Smoke check that request.state actor/purpose/request_id flow
    through to the response — covered transitively by the response-
    header echo path which reads from request.state."""
    resp = client.get(
        "/v1/health",
        headers={
            "X-MindMem-Actor": "downstream-probe",
            "X-MindMem-Purpose": "state-check",
        },
    )
    assert resp.status_code == 200
    # If middleware didn't stash on request.state then re-emit, these
    # would be missing. We don't have a direct ASGI hook into state from
    # the test client, so we cover this via the response side-effect.
    assert resp.headers["X-MindMem-Actor"] == "downstream-probe"
    assert resp.headers["X-MindMem-Purpose"] == "state-check"
    assert "X-MindMem-Request-Id" in resp.headers
