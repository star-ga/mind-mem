"""Wire-transport tests for v4 federation.

Exercises the round-trip between
:mod:`mind_mem.http_transport` (server) and
:mod:`mind_mem.v4.federation_client` (client) so the federation
foundation primitives are reachable over HTTP between two mind-mem
hosts.

The tests run the HTTP server in a background thread bound to
localhost and talk to it via the stdlib-based
:class:`FederationClient`. v4.federation is enabled before each test
via the feature-flag override fixture.
"""

from __future__ import annotations

import socket
import time
from pathlib import Path

import pytest

from mind_mem.http_transport import serve_http
from mind_mem.v4.federation_client import (
    ConflictView,
    FederationAuthError,
    FederationClient,
    FederationFlagDisabled,
    FederationTransportError,
)


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


# Bounded client timeout for the in-process localhost server. The default
# FederationClient read timeout. The production default is 10s; in-process
# tests want headroom for a heavily-loaded Windows CI runner (the in-process
# server thread can be GIL-starved under load, surfacing as a client
# socket.recv_into TimeoutError) while still failing well within the 120s
# pytest-timeout so a genuine hang is still caught by name.
_CLIENT_TIMEOUT = 30.0


def _wait_until_accepting(host: str, port: int, *, timeout: float = 10.0) -> None:
    """Poll-connect to ``(host, port)`` until the listener accepts.

    Defence-in-depth: ``serve_http`` already performs this handshake before
    returning, but the test asserts readiness independently so the suite stays
    deterministic even if the transport's internals change. Raises
    ``TimeoutError`` if the server never comes up.
    """
    deadline = time.monotonic() + timeout
    last_err: OSError | None = None
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return
        except OSError as exc:  # not yet listening
            last_err = exc
            time.sleep(0.01)
    raise TimeoutError(f"server on {host}:{port} never started accepting") from last_err


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    ws.mkdir()
    return ws


@pytest.fixture
def federation_enabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Write a mind-mem.json with ``v4.federation.enabled = true`` and
    point MIND_MEM_CONFIG at it."""
    import json

    cfg = tmp_path / "fed-on.json"
    cfg.write_text(json.dumps({"v4": {"federation": {"enabled": True}}}))
    monkeypatch.setenv("MIND_MEM_CONFIG", str(cfg))


@pytest.fixture
def federation_disabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Write a mind-mem.json with ``v4.federation.enabled = false`` and
    point MIND_MEM_CONFIG at it."""
    import json

    cfg = tmp_path / "fed-off.json"
    cfg.write_text(json.dumps({"v4": {"federation": {"enabled": False}}}))
    monkeypatch.setenv("MIND_MEM_CONFIG", str(cfg))


@pytest.fixture
def server(workspace: Path):
    """Spin up a localhost HTTP transport bound to a free port."""
    port = _free_port()
    thread, stop = serve_http(
        workspace=str(workspace),
        port=port,
        host="127.0.0.1",
        token="test-token",
        allow_unauthenticated_localhost=False,
    )
    # serve_http blocks until the listener accepts; assert it independently.
    _wait_until_accepting("127.0.0.1", port)
    base_url = f"http://127.0.0.1:{port}"
    try:
        yield base_url
    finally:
        stop()
        thread.join(timeout=5)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_vclock_empty_block_returns_empty_vector(server: str, federation_enabled: None) -> None:
    client = FederationClient(server, token="test-token", timeout=_CLIENT_TIMEOUT)
    vec = client.get_vclock("unknown-block")
    assert vec == {}


def test_push_write_increments_version(server: str, federation_enabled: None) -> None:
    client = FederationClient(server, token="test-token", timeout=_CLIENT_TIMEOUT)
    r1 = client.push_write("block-1", agent_id="alice")
    assert r1.version == 1
    assert r1.conflict is None
    r2 = client.push_write("block-1", agent_id="alice")
    assert r2.version == 2
    assert r2.conflict is None


def test_two_agent_divergence_surfaces_conflict(server: str, federation_enabled: None) -> None:
    client = FederationClient(server, token="test-token", timeout=_CLIENT_TIMEOUT)
    # Alice writes once, Bob writes twice → divergence.
    client.push_write("block-2", agent_id="alice")
    client.push_write("block-2", agent_id="bob")
    r = client.push_write("block-2", agent_id="bob")
    assert r.version == 2
    assert r.conflict is not None
    assert isinstance(r.conflict, ConflictView)
    # Bob has 2, Alice has 1 → bob is left, alice is right.
    assert r.conflict.left_agent == "bob"
    assert r.conflict.left_version == 2
    assert r.conflict.right_agent == "alice"
    assert r.conflict.right_version == 1


def test_list_conflicts_returns_open_pairs(server: str, federation_enabled: None) -> None:
    client = FederationClient(server, token="test-token", timeout=_CLIENT_TIMEOUT)
    client.push_write("block-3", agent_id="alice")
    client.push_write("block-3", agent_id="bob")
    client.push_write("block-3", agent_id="bob")
    conflicts = client.list_conflicts()
    assert len(conflicts) >= 1
    open_block_ids = {c.block_id for c in conflicts}
    assert "block-3" in open_block_ids


def test_resolve_higher_version_picks_left_agent(server: str, federation_enabled: None) -> None:
    client = FederationClient(server, token="test-token", timeout=_CLIENT_TIMEOUT)
    client.push_write("block-4", agent_id="alice")
    client.push_write("block-4", agent_id="bob")
    client.push_write("block-4", agent_id="bob")
    resolution = client.resolve_conflict("block-4", strategy="higher_version")
    assert resolution.block_id == "block-4"
    assert resolution.winner_agent == "bob"
    assert resolution.winner_version == 2
    assert resolution.strategy == "higher_version"
    assert resolution.merged_payload is None


def test_resolve_three_way_merge_round_trips_payload(server: str, federation_enabled: None) -> None:
    client = FederationClient(server, token="test-token", timeout=_CLIENT_TIMEOUT)
    client.push_write("block-5", agent_id="alice")
    client.push_write("block-5", agent_id="bob")
    client.push_write("block-5", agent_id="bob")
    payload = b"merged content with binary \xff\xfe\x00 bytes"
    resolution = client.resolve_conflict(
        "block-5",
        strategy="three_way_merge",
        merged_payload=payload,
    )
    assert resolution.merged_payload == payload
    assert resolution.strategy == "three_way_merge"
    # winner_agent is the synthetic merge label.
    assert resolution.winner_agent.startswith("merge:")


def test_resolve_missing_conflict_returns_404(server: str, federation_enabled: None) -> None:
    client = FederationClient(server, token="test-token", timeout=_CLIENT_TIMEOUT)
    with pytest.raises(FederationTransportError) as exc_info:
        client.resolve_conflict("never-written", strategy="higher_version")
    assert "404" in str(exc_info.value)


def test_unauthenticated_request_rejected(server: str, federation_enabled: None) -> None:
    client = FederationClient(server, token="WRONG-token", timeout=_CLIENT_TIMEOUT)
    with pytest.raises(FederationAuthError):
        client.push_write("block-x", agent_id="alice")


def test_flag_disabled_returns_503(server: str, federation_disabled: None) -> None:
    client = FederationClient(server, token="test-token", timeout=_CLIENT_TIMEOUT)
    with pytest.raises(FederationFlagDisabled):
        client.push_write("block-y", agent_id="alice")


def test_malformed_write_body_returns_400(server: str, federation_enabled: None) -> None:
    import urllib.error
    import urllib.request

    req = urllib.request.Request(
        f"{server}/federation/write",
        method="POST",
        data=b"{}",
        headers={"Content-Type": "application/json", "X-MindMem-Token": "test-token"},
    )
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req, timeout=5)
    assert exc_info.value.code == 400


def test_full_two_host_handshake_via_vclock_then_resolve(workspace: Path, federation_enabled: None) -> None:
    """End-to-end: writer agent on local server publishes writes, second
    agent on the same server records its writes, conflict is detected
    by the wire surface, then resolved via the same endpoint."""
    port = _free_port()
    thread, stop = serve_http(
        workspace=str(workspace),
        port=port,
        host="127.0.0.1",
        token="test-token",
        allow_unauthenticated_localhost=False,
    )
    try:
        client = FederationClient(f"http://127.0.0.1:{port}", token="test-token", timeout=_CLIENT_TIMEOUT)
        for _ in range(3):
            client.push_write("hand-1", agent_id="alice")
        client.push_write("hand-1", agent_id="bob")
        vec = client.get_vclock("hand-1")
        assert vec == {"alice": 3, "bob": 1}
        result = client.push_write("hand-1", agent_id="bob")
        # alice=3, bob=2 after this — left=alice(3), right=bob(2). The
        # left_agent is still alice (sorted by version desc), but bob
        # wrote LAST in wall-clock order.
        assert result.conflict is not None
        assert result.conflict.left_agent == "alice"
        # Audit FP-1: LAST_WRITER_WINS now resolves by wall-clock
        # last_seen_at — bob wins because bob's write was the most
        # recent in wall-clock order, even though alice has version 3.
        resolution = client.resolve_conflict("hand-1", strategy="last_writer_wins")
        assert resolution.winner_agent == "bob"
    finally:
        stop()
        thread.join(timeout=5)
