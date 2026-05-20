"""Regression tests for MIND_MEM_FED_PEERS operator-side peer allowlist
(roadmap v4.0.x federation transport hardening).

The federation HTTP listener accepts a comma-separated list of source
IPs via ``MIND_MEM_FED_PEERS``. When the env var is unset, the allowlist
is bypassed (backwards compatible with the localhost default). When set,
federation endpoints reject any source IP not on the list with 403,
*before* the auth check — a valid token doesn't help.

Non-federation endpoints (``/status``, ``/memories``) are unaffected.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Iterator

import pytest


def _free_port() -> int:
    """Bind a temporary socket to find a free TCP port."""
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture()
def fed_workspace(tmp_path) -> str:
    """Build a minimal mind-mem workspace skeleton."""
    cfg = {"v4": {"federation": {"enabled": True}}}
    (tmp_path / "mind-mem.json").write_text(json.dumps(cfg))
    (tmp_path / "decisions").mkdir()
    return str(tmp_path)


@pytest.fixture()
def serve_loopback(fed_workspace: str, monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[str, int]]:
    """Spin up the mind-mem HTTP transport on a loopback port."""
    from mind_mem import http_transport

    port = _free_port()
    base = f"http://127.0.0.1:{port}"

    monkeypatch.delenv("MIND_MEM_TOKEN", raising=False)
    monkeypatch.setenv("MIND_MEM_FEDERATION_ENABLED", "1")
    # Disable the rate limiter for the test (we hit /status repeatedly
    # to warm up + each endpoint multiple times across tests).
    monkeypatch.setenv("MIND_MEM_HTTP_RATE_MAX_CALLS", "0")

    thread, stop = http_transport.serve_http(
        workspace=fed_workspace,
        host="127.0.0.1",
        port=port,
        allow_unauthenticated_localhost=True,
    )
    # Wait for the listener to be ready.
    for _ in range(50):
        try:
            urllib.request.urlopen(f"{base}/status", timeout=1).read()
            break
        except Exception:
            time.sleep(0.05)
    try:
        yield base, port
    finally:
        stop()
        thread.join(timeout=5)


def _http_status(url: str, timeout: float = 2.0) -> int:
    """Return the HTTP status code from a GET; 0 on transport error."""
    try:
        resp = urllib.request.urlopen(url, timeout=timeout)
        return resp.status
    except urllib.error.HTTPError as exc:
        return exc.code
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Allowlist unset (default) — backwards compatible
# ---------------------------------------------------------------------------


def test_allowlist_unset_passes_through(
    serve_loopback: tuple[str, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When MIND_MEM_FED_PEERS is unset, federation calls go through
    normally (the existing localhost default is preserved)."""
    monkeypatch.delenv("MIND_MEM_FED_PEERS", raising=False)
    base, _ = serve_loopback
    # /federation/conflicts is a federation endpoint that's safe to GET.
    status = _http_status(f"{base}/federation/conflicts")
    # 200 (data) or 404 (no conflicts row yet) — both indicate the
    # allowlist didn't reject. 403 = allowlist blocked, 0 = transport
    # failure. We assert neither block reason fired.
    assert status not in (0, 403), f"unexpected status {status}"


# ---------------------------------------------------------------------------
# Allowlist set, source not on it → 403
# ---------------------------------------------------------------------------


def test_allowlist_excludes_loopback_blocks_federation(
    serve_loopback: tuple[str, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With MIND_MEM_FED_PEERS=10.0.0.5 (not the loopback we're calling
    from), every federation endpoint must reject with 403."""
    monkeypatch.setenv("MIND_MEM_FED_PEERS", "10.0.0.5,10.0.0.6")
    base, _ = serve_loopback
    status = _http_status(f"{base}/federation/conflicts")
    assert status == 403, f"expected 403, got {status}"


def test_allowlist_blocks_federation_write_post(
    serve_loopback: tuple[str, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """POST to /federation/write is also gated by the allowlist."""
    monkeypatch.setenv("MIND_MEM_FED_PEERS", "10.0.0.5")
    base, _ = serve_loopback
    body = json.dumps({"block_id": "BLOCK-1", "agent_id": "agent-A"}).encode()
    req = urllib.request.Request(
        f"{base}/federation/write",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        urllib.request.urlopen(req, timeout=2)
        pytest.fail("expected 403 from peer-allowlist rejection")
    except urllib.error.HTTPError as exc:
        assert exc.code == 403, f"expected 403, got {exc.code}"


# ---------------------------------------------------------------------------
# Allowlist includes 127.0.0.1 → loopback federation calls succeed
# ---------------------------------------------------------------------------


def test_allowlist_includes_loopback_allows_federation(
    serve_loopback: tuple[str, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With MIND_MEM_FED_PEERS=127.0.0.1, loopback federation calls
    pass the allowlist check."""
    monkeypatch.setenv("MIND_MEM_FED_PEERS", "127.0.0.1")
    base, _ = serve_loopback
    status = _http_status(f"{base}/federation/conflicts")
    assert status != 403, "loopback explicitly allowed but got 403"


# ---------------------------------------------------------------------------
# Non-federation endpoints unaffected by the allowlist
# ---------------------------------------------------------------------------


def test_allowlist_does_not_block_status_endpoint(
    serve_loopback: tuple[str, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Even with MIND_MEM_FED_PEERS set to exclude the caller,
    non-federation endpoints (/status) remain reachable."""
    monkeypatch.setenv("MIND_MEM_FED_PEERS", "10.0.0.5")
    base, _ = serve_loopback
    status = _http_status(f"{base}/status")
    assert status == 200, f"non-federation endpoint blocked unexpectedly: {status}"
