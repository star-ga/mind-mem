"""MCP resources go through the same ACL and rate limiting as tools.

Architecture review F14: ``resources.register`` bound every resource straight
onto FastMCP, so a resource read was outside the per-client rate limiter (a
client could run ``mind-mem://recall/{query}`` -- a full recall -- in a loop
with no budget) and outside the ACL (an auth-introspection failure, which
fails every TOOL closed, still served every resource; an unclassified resource
was served too). These tests read through the LIVE FastMCP server with an
in-memory client, so they exercise the registration, not the bare functions.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile

import pytest

from mind_mem.mcp.infra import rate_limit as _rate_limit
from mind_mem.mcp.infra.acl import USER_RESOURCES


@pytest.fixture
def workspace(monkeypatch: pytest.MonkeyPatch):
    from mind_mem.init_workspace import init

    ws = tempfile.mkdtemp(prefix="mm_res_acl_")
    init(ws)
    monkeypatch.setenv("MIND_MEM_WORKSPACE", ws)
    monkeypatch.setenv("MIND_MEM_CONFIG", os.path.join(ws, "mind-mem.json"))
    monkeypatch.setenv("MIND_MEM_SCOPE", "user")
    monkeypatch.delenv("MIND_MEM_ACL_DISABLED", raising=False)
    with _rate_limit._rate_limiters_lock:
        _rate_limit._rate_limiters.clear()
    try:
        yield ws
    finally:
        with _rate_limit._rate_limiters_lock:
            _rate_limit._rate_limiters.clear()
        shutil.rmtree(ws, ignore_errors=True)


def _read(uri: str) -> str:
    from fastmcp import Client

    from mind_mem.mcp.server import mcp

    async def go() -> str:
        async with Client(mcp) as client:
            contents = await client.read_resource(uri)
        return "".join(getattr(c, "text", "") for c in contents)

    return asyncio.run(go())


def _registered_resource_names() -> set[str]:
    from mind_mem.mcp import resources

    found: set[str] = set()

    class _Probe:
        def resource(self, uri):  # noqa: ANN001, ANN202 - FastMCP's shape
            def bind(fn):  # noqa: ANN001, ANN202
                found.add(fn.__name__)
                return fn

            return bind

    resources.register(_Probe())
    return found


def test_positive_control_resource_serves_when_allowed(workspace: str) -> None:
    payload = json.loads(_read("mind-mem://health"))
    assert "files" in payload, payload


def test_every_registered_resource_is_classified() -> None:
    assert _registered_resource_names() == set(USER_RESOURCES)


@pytest.mark.parametrize("uri", ["mind-mem://health", "mind-mem://recall/frost", "mind-mem://entities/projects"])
def test_resource_read_is_charged_to_the_rate_limiter(workspace: str, uri: str) -> None:
    exhausted = _rate_limit.SlidingWindowRateLimiter(max_calls=1, window_seconds=60)
    assert exhausted.allow()[0]  # spend the only call in the window
    with _rate_limit._rate_limiters_lock:
        _rate_limit._rate_limiters[f"pid-{os.getpid()}"] = exhausted
    payload = json.loads(_read(uri))
    assert payload.get("error") == "Rate limit exceeded. Try again later.", payload


def test_resource_read_consumes_budget(workspace: str) -> None:
    """Positive control for the limiter: a served read appends a timestamp."""
    limiter = _rate_limit.SlidingWindowRateLimiter(max_calls=50, window_seconds=60)
    with _rate_limit._rate_limiters_lock:
        _rate_limit._rate_limiters[f"pid-{os.getpid()}"] = limiter
    _read("mind-mem://health")
    assert len(limiter._timestamps) == 1


@pytest.mark.parametrize("uri", ["mind-mem://health", "mind-mem://tasks", "mind-mem://recall/frost"])
def test_resource_fails_closed_when_auth_introspection_fails(workspace: str, uri: str, monkeypatch: pytest.MonkeyPatch) -> None:
    from mind_mem.mcp.infra import acl

    def _boom():
        raise RuntimeError("token provider unavailable")

    monkeypatch.setattr(acl, "get_access_token", _boom)
    payload = json.loads(_read(uri))
    assert payload.get("error") == "Permission denied: authentication context unavailable", payload


def test_unclassified_resource_is_refused(workspace: str, monkeypatch: pytest.MonkeyPatch) -> None:
    from mind_mem.mcp import resources

    monkeypatch.setattr(resources, "USER_RESOURCES", frozenset(USER_RESOURCES - {"get_health"}))
    payload = json.loads(_read("mind-mem://health"))
    assert payload.get("error") == "Resource 'get_health' is not in ACL policy", payload
