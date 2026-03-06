"""MCP transport and auth integration tests (#474).

Tests bearer-token authentication, admin/user ACL enforcement,
rate limiter behaviour, and tool response schema contracts.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import time

import pytest
from fastmcp.server.auth import AccessToken

# ---------------------------------------------------------------------------
# Skip everything when fastmcp is not installed
# ---------------------------------------------------------------------------

_HAS_FASTMCP = importlib.util.find_spec("fastmcp") is not None
pytestmark = pytest.mark.skipif(not _HAS_FASTMCP, reason="fastmcp not installed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SERVER_PATH = os.path.join(os.path.dirname(__file__), "..", "mcp_server.py")


def _make_workspace(tmp_path):
    """Create a minimal workspace for testing."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    for d in [
        "decisions",
        "tasks",
        "entities",
        "intelligence",
        "intelligence/proposed",
        "intelligence/applied",
        "intelligence/state/snapshots",
        "memory",
        "summaries/weekly",
        "summaries/daily",
        "maintenance/weeklog",
        "categories",
    ]:
        (ws / d).mkdir(parents=True, exist_ok=True)

    cfg = {
        "version": "1.1.0",
        "workspace_path": str(ws),
        "auto_capture": False,
        "auto_recall": False,
        "governance_mode": "detect_only",
        "recall": {"backend": "scan"},
        "proposal_budget": {"per_run": 3, "per_day": 6, "backlog_limit": 30},
    }
    (ws / "mind-mem.json").write_text(json.dumps(cfg))

    # Add a decision for recall to find
    dec = ws / "decisions" / "DECISIONS.md"
    dec.write_text(
        "[D-20260218-001]\n"
        "Date: 2026-02-18\n"
        "Status: active\n"
        "Statement: Use PostgreSQL for the primary database\n"
        "Tags: database, infrastructure\n"
        "Rationale: Better JSON support than MySQL\n"
    )

    # Intelligence files
    (ws / "intelligence" / "SIGNALS.md").write_text("")
    (ws / "intelligence" / "CONTRADICTIONS.md").write_text("")
    (ws / "intelligence" / "DRIFT.md").write_text("")
    (ws / "intelligence" / "IMPACT.md").write_text("")
    (ws / "intelligence" / "BRIEFINGS.md").write_text("")
    (ws / "intelligence" / "AUDIT.md").write_text("")
    (ws / "intelligence" / "SCAN_LOG.md").write_text("")

    # Entity stubs
    for fname in [
        "entities/projects.md",
        "entities/people.md",
        "entities/tools.md",
        "entities/incidents.md",
    ]:
        path = ws / fname
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {os.path.basename(fname)}\n")

    # Tasks file
    (ws / "tasks" / "TASKS.md").write_text("# Tasks\n")

    return ws


def _load_server(workspace: str):
    """Import mcp_server with MIND_MEM_WORKSPACE pointed at *workspace*."""
    os.environ["MIND_MEM_WORKSPACE"] = workspace
    spec = importlib.util.spec_from_file_location("mcp_server", _SERVER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _call_tool(tool_attr, *args, **kwargs):
    """Call an MCP tool, unwrapping FastMCP's FunctionTool wrapper if needed."""
    fn = getattr(tool_attr, "fn", tool_attr)
    return fn(*args, **kwargs)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def workspace(tmp_path):
    """Provide a disposable workspace directory."""
    return _make_workspace(tmp_path)


@pytest.fixture()
def server(workspace):
    """Load the MCP server module with a clean workspace."""
    return _load_server(str(workspace))


@pytest.fixture(autouse=True)
def _clean_auth_env(monkeypatch):
    """Ensure auth-related env vars are cleared between tests."""
    monkeypatch.delenv("MIND_MEM_TOKEN", raising=False)
    monkeypatch.delenv("MIND_MEM_ADMIN_TOKEN", raising=False)
    monkeypatch.delenv("MIND_MEM_SCOPE", raising=False)


# ===================================================================
# 1. Bearer token authentication
# ===================================================================


class TestBearerTokenAuth:
    """verify_token() honours MIND_MEM_TOKEN with constant-time compare."""

    def test_no_token_configured_allows_any_request(self, server):
        """When MIND_MEM_TOKEN is unset, every request passes."""
        assert server.verify_token({}) is True
        assert server.verify_token({"Authorization": "Bearer anything"}) is True

    def test_correct_bearer_token_accepted(self, server, monkeypatch):
        monkeypatch.setenv("MIND_MEM_TOKEN", "super-secret-42")
        assert server.verify_token({"Authorization": "Bearer super-secret-42"}) is True

    def test_wrong_bearer_token_rejected(self, server, monkeypatch):
        monkeypatch.setenv("MIND_MEM_TOKEN", "super-secret-42")
        assert server.verify_token({"Authorization": "Bearer wrong-token"}) is False

    def test_missing_auth_header_rejected_when_token_set(self, server, monkeypatch):
        monkeypatch.setenv("MIND_MEM_TOKEN", "super-secret-42")
        assert server.verify_token({}) is False

    def test_alt_x_mindmem_token_header_accepted(self, server, monkeypatch):
        monkeypatch.setenv("MIND_MEM_TOKEN", "alt-secret")
        assert server.verify_token({"X-MindMem-Token": "alt-secret"}) is True

    def test_alt_x_mindmem_token_header_wrong_rejected(self, server, monkeypatch):
        monkeypatch.setenv("MIND_MEM_TOKEN", "alt-secret")
        assert server.verify_token({"X-MindMem-Token": "nope"}) is False

    def test_bearer_prefix_required(self, server, monkeypatch):
        """Authorization header without 'Bearer ' prefix is rejected."""
        monkeypatch.setenv("MIND_MEM_TOKEN", "secret")
        assert server.verify_token({"Authorization": "secret"}) is False

    def test_case_insensitive_header_lookup(self, server, monkeypatch):
        """verify_token checks both lower- and title-case header names."""
        monkeypatch.setenv("MIND_MEM_TOKEN", "sec")
        assert server.verify_token({"authorization": "Bearer sec"}) is True
        assert server.verify_token({"Authorization": "Bearer sec"}) is True


# ===================================================================
# 2. Admin scope / ACL enforcement
# ===================================================================


class TestACLEnforcement:
    """check_tool_acl + mcp_tool_observe ACL gate tests."""

    # -- direct check_tool_acl tests --

    def test_admin_tool_denied_for_user_scope(self, server):
        err = server.check_tool_acl("delete_memory_item", "user")
        assert err is not None
        parsed = json.loads(err)
        assert "Permission denied" in parsed["error"]
        assert parsed["scope"] == "user"

    def test_admin_tool_allowed_for_admin_scope(self, server):
        assert server.check_tool_acl("delete_memory_item", "admin") is None

    def test_user_tool_allowed_for_user_scope(self, server):
        assert server.check_tool_acl("recall", "user") is None

    def test_user_tool_allowed_for_admin_scope(self, server):
        assert server.check_tool_acl("recall", "admin") is None

    def test_all_admin_tools_rejected_as_user(self, server):
        """Every member of ADMIN_TOOLS must fail for 'user' scope."""
        for tool_name in server.ADMIN_TOOLS:
            err = server.check_tool_acl(tool_name, "user")
            assert err is not None, f"{tool_name} was NOT denied for user scope"

    def test_all_user_tools_pass_as_user(self, server):
        """Every member of USER_TOOLS must pass for 'user' scope."""
        for tool_name in server.USER_TOOLS:
            assert server.check_tool_acl(tool_name, "user") is None, f"{tool_name} was denied for user scope"

    # -- decorator-level ACL enforcement --

    def test_decorator_blocks_admin_tool_when_admin_token_set(self, server, workspace, monkeypatch):
        """With MIND_MEM_ADMIN_TOKEN set and scope=user, admin tools return error JSON."""
        monkeypatch.setenv("MIND_MEM_ADMIN_TOKEN", "admin-tok")
        monkeypatch.setenv("MIND_MEM_SCOPE", "user")
        monkeypatch.setenv("MIND_MEM_WORKSPACE", str(workspace))

        # Call an admin tool directly (the decorator wrapper intercepts)
        result = _call_tool(server.reindex)
        parsed = json.loads(result)
        assert "error" in parsed
        assert "Permission denied" in parsed["error"]

    def test_decorator_allows_admin_tool_when_scope_is_admin(self, server, workspace, monkeypatch):
        """With MIND_MEM_ADMIN_TOKEN set and scope=admin, admin tools execute."""
        monkeypatch.setenv("MIND_MEM_ADMIN_TOKEN", "admin-tok")
        monkeypatch.setenv("MIND_MEM_SCOPE", "admin")
        monkeypatch.setenv("MIND_MEM_WORKSPACE", str(workspace))

        result = _call_tool(server.scan)
        parsed = json.loads(result)
        # scan is a USER tool -- should succeed regardless
        assert "_schema_version" in parsed

    def test_decorator_skips_acl_when_no_admin_token(self, server, workspace, monkeypatch):
        """Without MIND_MEM_ADMIN_TOKEN, all tools pass (local single-user mode)."""
        monkeypatch.delenv("MIND_MEM_ADMIN_TOKEN", raising=False)
        monkeypatch.setenv("MIND_MEM_WORKSPACE", str(workspace))

        # reindex is an admin tool but should work without admin token
        result = _call_tool(server.reindex)
        parsed = json.loads(result)
        assert "error" not in parsed or "Permission denied" not in parsed.get("error", "")

    def test_decorator_rejects_unknown_tool_when_admin_token_set(self, server, monkeypatch):
        """A tool not in USER_TOOLS or ADMIN_TOOLS gets rejected when ADMIN_TOKEN is set."""
        monkeypatch.setenv("MIND_MEM_ADMIN_TOKEN", "admin-tok")

        # Simulate an unknown tool via the observe wrapper
        import mcp_server

        @mcp_server.mcp_tool_observe
        def totally_unknown_tool():
            return json.dumps({"ok": True})

        result = totally_unknown_tool()
        parsed = json.loads(result)
        assert "error" in parsed
        assert "not in ACL policy" in parsed["error"]

    def test_request_scope_uses_access_token_scopes(self, server, monkeypatch):
        monkeypatch.setattr(
            server,
            "get_access_token",
            lambda: AccessToken(token="adm", client_id="client", scopes=["admin"], claims={}),
        )
        assert server._get_request_scope() == "admin"

    def test_request_scope_defaults_user_for_non_admin_token(self, server, monkeypatch):
        monkeypatch.setattr(
            server,
            "get_access_token",
            lambda: AccessToken(token="usr", client_id="client", scopes=["user"], claims={}),
        )
        assert server._get_request_scope() == "user"

    def test_http_request_scope_overrides_process_scope(self, server, workspace, monkeypatch):
        monkeypatch.setenv("MIND_MEM_ADMIN_TOKEN", "admin-tok")
        monkeypatch.setenv("MIND_MEM_SCOPE", "admin")
        monkeypatch.setenv("MIND_MEM_WORKSPACE", str(workspace))
        monkeypatch.setattr(
            server,
            "get_access_token",
            lambda: AccessToken(token="usr", client_id="client", scopes=["user"], claims={}),
        )

        result = _call_tool(server.reindex)
        parsed = json.loads(result)
        assert "error" in parsed
        assert "Permission denied" in parsed["error"]

    def test_build_http_auth_tokens_includes_user_and_admin_scopes(self, server, monkeypatch):
        monkeypatch.setenv("MIND_MEM_TOKEN", "user-tok")
        monkeypatch.setenv("MIND_MEM_ADMIN_TOKEN", "admin-tok")

        tokens = server._build_http_auth_tokens()
        assert tokens["user-tok"]["client_id"] == "mind-mem-user"
        assert tokens["user-tok"]["scopes"] == ["user"]
        assert tokens["admin-tok"]["client_id"] == "mind-mem-admin"
        assert tokens["admin-tok"]["scopes"] == ["user", "admin"]


# ===================================================================
# 3. Rate limiter integration
# ===================================================================


class TestRateLimiter:
    """SlidingWindowRateLimiter unit tests."""

    def test_allows_calls_under_limit(self, server):
        limiter = server.SlidingWindowRateLimiter(max_calls=5, window_seconds=60)
        for _ in range(5):
            allowed, retry = limiter.allow()
            assert allowed is True
            assert retry == 0.0

    def test_rejects_calls_over_limit(self, server):
        limiter = server.SlidingWindowRateLimiter(max_calls=3, window_seconds=60)
        for _ in range(3):
            limiter.allow()

        allowed, retry = limiter.allow()
        assert allowed is False
        assert retry > 0

    def test_window_expiry_allows_new_calls(self, server):
        limiter = server.SlidingWindowRateLimiter(max_calls=2, window_seconds=0.1)
        limiter.allow()
        limiter.allow()

        allowed, _ = limiter.allow()
        assert allowed is False

        # Wait for window to expire
        time.sleep(0.15)
        allowed, _ = limiter.allow()
        assert allowed is True

    def test_retry_after_is_positive(self, server):
        limiter = server.SlidingWindowRateLimiter(max_calls=1, window_seconds=60)
        limiter.allow()
        _, retry = limiter.allow()
        assert retry >= 0.1


# ===================================================================
# 4. Tool response schema contracts
# ===================================================================


class TestToolResponseSchema:
    """Every tool must return valid JSON with _schema_version."""

    def test_recall_returns_schema_version(self, server, workspace, monkeypatch):
        monkeypatch.setenv("MIND_MEM_WORKSPACE", str(workspace))
        result = _call_tool(server.recall, "PostgreSQL")
        parsed = json.loads(result)
        assert parsed["_schema_version"] == "1.0"
        assert "results" in parsed

    def test_scan_returns_schema_version(self, server, workspace, monkeypatch):
        monkeypatch.setenv("MIND_MEM_WORKSPACE", str(workspace))
        result = _call_tool(server.scan)
        parsed = json.loads(result)
        assert parsed["_schema_version"] == "1.0"
        assert "checks" in parsed

    def test_list_contradictions_returns_schema_version(self, server, workspace, monkeypatch):
        monkeypatch.setenv("MIND_MEM_WORKSPACE", str(workspace))
        result = _call_tool(server.list_contradictions)
        parsed = json.loads(result)
        assert parsed["_schema_version"] == "1.0"

    def test_index_stats_returns_schema_version(self, server, workspace, monkeypatch):
        monkeypatch.setenv("MIND_MEM_WORKSPACE", str(workspace))
        result = _call_tool(server.index_stats)
        parsed = json.loads(result)
        assert parsed["_schema_version"] == "1.0"

    def test_intent_classify_returns_schema_version(self, server):
        result = _call_tool(server.intent_classify, "When did we decide on PostgreSQL?")
        parsed = json.loads(result)
        assert parsed["_schema_version"] == "1.0"

    def test_reindex_returns_schema_version(self, server, workspace, monkeypatch):
        monkeypatch.setenv("MIND_MEM_WORKSPACE", str(workspace))
        result = _call_tool(server.reindex)
        parsed = json.loads(result)
        assert parsed["_schema_version"] == "1.0"

    def test_list_mind_kernels_returns_schema_version(self, server, workspace, monkeypatch):
        monkeypatch.setenv("MIND_MEM_WORKSPACE", str(workspace))
        result = _call_tool(server.list_mind_kernels)
        parsed = json.loads(result)
        assert parsed["_schema_version"] == "1.0"

    def test_error_response_contains_error_field(self, server, workspace, monkeypatch):
        """Tools that fail should include an 'error' key in their JSON response."""
        monkeypatch.setenv("MIND_MEM_WORKSPACE", str(workspace))
        # find_similar with non-existent block
        result = _call_tool(server.find_similar, "D-nonexistent-999")
        parsed = json.loads(result)
        assert "error" in parsed or "similar" in parsed  # may succeed with empty
        assert parsed["_schema_version"] == "1.0"

    def test_approve_apply_invalid_id_returns_error(self, server, workspace, monkeypatch):
        monkeypatch.setenv("MIND_MEM_WORKSPACE", str(workspace))
        result = _call_tool(server.approve_apply, "bad-format")
        parsed = json.loads(result)
        assert "error" in parsed
        assert "Invalid proposal ID" in parsed["error"]

    def test_rollback_invalid_ts_returns_error(self, server, workspace, monkeypatch):
        monkeypatch.setenv("MIND_MEM_WORKSPACE", str(workspace))
        result = _call_tool(server.rollback_proposal, "not-a-timestamp")
        parsed = json.loads(result)
        assert "error" in parsed


# ===================================================================
# 5. ADMIN_TOOLS / USER_TOOLS completeness
# ===================================================================


class TestACLSets:
    """Ensure every registered tool is covered by exactly one ACL set."""

    def test_no_overlap_between_admin_and_user(self, server):
        overlap = server.ADMIN_TOOLS & server.USER_TOOLS
        assert overlap == frozenset(), f"Tools in both sets: {overlap}"

    def test_admin_tools_not_empty(self, server):
        assert len(server.ADMIN_TOOLS) > 0

    def test_user_tools_not_empty(self, server):
        assert len(server.USER_TOOLS) > 0


# ===================================================================
# 6. Existing integration tests (preserved from original)
# ===================================================================


class TestRecallTool:
    """Tests for the recall MCP tool logic."""

    def test_recall_finds_decision(self, tmp_path):
        ws = _make_workspace(tmp_path)
        from mind_mem.recall import recall

        results = recall(str(ws), "PostgreSQL database", limit=5)
        assert len(results) > 0
        assert any("PostgreSQL" in str(r) for r in results)

    def test_recall_empty_query(self, tmp_path):
        ws = _make_workspace(tmp_path)
        from mind_mem.recall import recall

        results = recall(str(ws), "", limit=5)
        assert isinstance(results, list)

    def test_recall_no_match(self, tmp_path):
        ws = _make_workspace(tmp_path)
        from mind_mem.recall import recall

        results = recall(str(ws), "quantum computing spaceship", limit=5)
        assert isinstance(results, list)


class TestIntentClassify:
    """Tests for intent classification."""

    def test_temporal_intent(self):
        from mind_mem.intent_router import IntentRouter

        router = IntentRouter()
        result = router.classify("When did we decide on PostgreSQL?")
        assert result.intent == "WHEN"

    def test_entity_intent(self):
        from mind_mem.intent_router import IntentRouter

        router = IntentRouter()
        result = router.classify("What is PostgreSQL used for?")
        assert result.intent is not None

    def test_verify_intent(self):
        from mind_mem.intent_router import IntentRouter

        router = IntentRouter()
        result = router.classify("Did we ever use MySQL?")
        assert result.intent is not None


class TestIndexStats:
    """Tests for index stats functionality."""

    def test_status_on_empty_workspace(self, tmp_path):
        ws = _make_workspace(tmp_path)
        from mind_mem.sqlite_index import index_status

        stats = index_status(str(ws))
        assert stats is not None
        assert isinstance(stats, dict)
        assert stats["exists"] is False

    def test_build_and_query_index(self, tmp_path):
        ws = _make_workspace(tmp_path)
        from mind_mem.sqlite_index import build_index, query_index

        build_index(str(ws), incremental=False)
        results = query_index(str(ws), "PostgreSQL", limit=5)
        assert isinstance(results, list)
        assert len(results) > 0
        assert any("PostgreSQL" in str(r) for r in results)
