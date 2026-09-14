"""Request authentication is resolved once at the observed MCP boundary."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


def _token(*, subject: str | None = "alice", client_id: str = "alice-client", scopes: list[str] | None = None):
    claims = {} if subject is None else {"sub": subject}
    return SimpleNamespace(claims=claims, client_id=client_id, scopes=scopes or ["user"])


def _tool(body):
    from mind_mem.mcp.infra.observability import mcp_tool_observe

    def recall():
        return body()

    return mcp_tool_observe(recall)


def test_observed_dispatch_resolves_token_once_and_binds_principal(monkeypatch: pytest.MonkeyPatch) -> None:
    from mind_mem.mcp.infra import acl, rate_limit

    token_reads = Mock(side_effect=[_token(), None])
    monkeypatch.setattr(acl, "get_access_token", token_reads)
    late_rate_read = Mock(side_effect=AssertionError("rate limiter must use the auth snapshot"))
    monkeypatch.setattr(rate_limit, "get_access_token", late_rate_read)

    def body():
        return json.dumps(
            {
                "agent": acl.authenticated_agent_id(),
                "scope": acl._get_request_scope(),
                "client": rate_limit._get_client_id(),
            }
        )

    result = json.loads(_tool(body)())
    assert result == {"agent": "alice", "scope": "user", "client": "alice-client"}
    assert token_reads.call_count == 1
    late_rate_read.assert_not_called()


def test_registered_recall_uses_the_bound_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise the actual registered public recall callable, not a stub ACL name."""
    import mind_mem.mcp.tools.recall as recall_tools
    from mind_mem.mcp.infra import acl

    monkeypatch.setattr(acl, "get_access_token", lambda: _token(subject="alice"))
    monkeypatch.setattr(
        recall_tools,
        "_recall_impl",
        lambda *args, **kwargs: json.dumps({"agent": acl.authenticated_agent_id()}),
    )
    assert json.loads(recall_tools.recall("aurora")) == {"agent": "alice"}


def test_missing_verified_subject_is_refused_before_body_and_client_id_is_not_principal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mind_mem.mcp.infra import acl

    body = Mock(return_value='{"unexpected":true}')
    monkeypatch.setattr(acl, "get_access_token", lambda: _token(subject=None, client_id="generic-client"))

    result = json.loads(_tool(body)())
    assert result["scope"] == "deny"
    assert "generic-client" not in json.dumps(result)
    body.assert_not_called()


def test_provider_failure_is_generic_and_does_not_leak_exception_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    from mind_mem.mcp.infra import acl

    body = Mock(return_value='{"unexpected":true}')
    monkeypatch.setattr(acl, "get_access_token", Mock(side_effect=RuntimeError("secret-token-value")))

    result = json.loads(_tool(body)())
    assert result == {
        "error": "Permission denied: authentication context unavailable",
        "scope": "deny",
    }
    assert "secret-token-value" not in json.dumps(result)
    body.assert_not_called()


def test_unbound_stdio_remains_allowed_and_context_is_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    from mind_mem.audit_context import UNATTRIBUTED, current_agent_id
    from mind_mem.mcp.infra import acl

    monkeypatch.setattr(acl, "get_access_token", lambda: None)
    tool = _tool(lambda: json.dumps({"agent": acl.authenticated_agent_id()}))
    assert json.loads(tool()) == {"agent": None}
    assert current_agent_id.get() == UNATTRIBUTED


def test_prebound_transport_identity_survives_no_token_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    from mind_mem.audit_context import bind_current_agent
    from mind_mem.mcp.infra import acl

    monkeypatch.setattr(acl, "get_access_token", lambda: None)
    tool = _tool(lambda: json.dumps({"agent": acl.authenticated_agent_id()}))
    with bind_current_agent("alice"):
        assert json.loads(tool()) == {"agent": "alice"}


def test_nested_decorated_calls_reuse_frozen_outer_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    from mind_mem.mcp.infra import acl, rate_limit

    token_reads = Mock(side_effect=[_token(subject="alice"), None])
    monkeypatch.setattr(acl, "get_access_token", token_reads)
    monkeypatch.setattr(rate_limit, "get_access_token", Mock(side_effect=AssertionError("late lookup")))
    inner = _tool(lambda: acl.authenticated_agent_id())
    outer = _tool(lambda: json.dumps({"outer": acl.authenticated_agent_id(), "inner": inner()}))

    assert json.loads(outer()) == {"outer": "alice", "inner": "alice"}
    assert token_reads.call_count == 1


def test_prebound_identity_mismatch_fails_closed_and_actor_header_cannot_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mind_mem.audit_context import bind_audit_context, bind_current_agent, context_from_headers
    from mind_mem.mcp.infra import acl

    monkeypatch.setattr(acl, "get_access_token", lambda: _token(subject="alice"))
    body = Mock(return_value=json.dumps({"agent": acl.authenticated_agent_id()}))
    tool = _tool(body)
    ctx = context_from_headers(lambda name: "bob" if name == "x-mindmem-actor" else None, transport="test")
    with bind_audit_context(ctx):
        assert json.loads(tool()) == {"agent": "alice"}
    with bind_current_agent("bob"):
        result = json.loads(tool())
        assert result["scope"] == "deny"
    assert body.call_count == 1


def test_contextvars_isolate_mixed_principals_on_concurrent_registered_dispatches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mind_mem.audit_context import UNATTRIBUTED, bind_current_agent, current_agent_id
    from mind_mem.mcp.infra import acl

    monkeypatch.setattr(acl, "get_access_token", lambda: None)
    tool = _tool(lambda: json.dumps({"agent": acl.authenticated_agent_id()}))

    def invoke(agent: str) -> str:
        with bind_current_agent(agent):
            result = json.loads(tool())
            assert current_agent_id.get() == agent
            return result["agent"]

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = sorted(pool.map(invoke, ("alice", "bob")))
    assert results == ["alice", "bob"]
    assert current_agent_id.get() == UNATTRIBUTED
