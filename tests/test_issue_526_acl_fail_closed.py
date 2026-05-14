"""Regression for issue #526: ACL `_get_request_scope` must fail-closed.

Before the fix, an exception in `get_access_token()` silently returned
``None``, which fell through to ``"user"`` at the call site — so a
transient introspection error degraded an admin-scoped session to user
scope without any operator signal.

After the fix:
  • Exceptions return the sentinel ``"deny"``.
  • ``check_tool_acl(tool, "deny")`` rejects EVERY tool (user + admin)
    with HTTP-401-shaped JSON.
  • The decorator's first-line check short-circuits to that rejection
    before any other gate.
  • An ``acl_introspection_failed`` warning is logged.
"""

from __future__ import annotations

import json
from unittest.mock import patch


def test_get_request_scope_returns_deny_on_introspection_exception():
    from mind_mem.mcp.infra import acl as _acl

    with patch.object(_acl, "get_access_token", side_effect=RuntimeError("boom")):
        assert _acl._get_request_scope() == "deny"


def test_get_request_scope_returns_none_when_no_token():
    from mind_mem.mcp.infra import acl as _acl

    with patch.object(_acl, "get_access_token", return_value=None):
        assert _acl._get_request_scope() is None


def test_check_tool_acl_rejects_deny_scope_on_user_tool():
    """Deny sentinel must reject even user-scoped tools (fail-closed)."""
    from mind_mem.mcp.infra import acl as _acl

    # `recall` is a user-tool — without the deny gate, a deny-scope
    # caller would still be allowed.
    result = _acl.check_tool_acl("recall", "deny")
    assert result is not None
    parsed = json.loads(result)
    assert "Permission denied" in parsed["error"]
    assert parsed["scope"] == "deny"


def test_check_tool_acl_rejects_deny_scope_on_admin_tool():
    from mind_mem.mcp.infra import acl as _acl

    result = _acl.check_tool_acl("delete_memory_item", "deny")
    assert result is not None
    parsed = json.loads(result)
    assert "Permission denied" in parsed["error"]


def test_decorator_short_circuits_on_introspection_failure():
    """End-to-end: introspection raises -> decorator rejects without
    even calling the wrapped function body."""
    from mind_mem.mcp.infra import acl as _acl
    from mind_mem.mcp.infra import observability as _obs

    called = {"body": False}

    @_obs.mcp_tool_observe
    def recall():
        called["body"] = True
        return "should-not-reach"

    with patch.object(_acl, "get_access_token", side_effect=RuntimeError("boom")):
        result = recall()

    assert called["body"] is False, "function body must not run when introspection fails"
    parsed = json.loads(result)
    assert "Permission denied" in parsed["error"]
    assert parsed["scope"] == "deny"
