"""ACL coverage invariant for the MCP tool surface.

Regression tests for the classification gap in
``mind_mem.mcp.infra.acl``: 13 registered tools (the 6 v3.2.0
consolidated dispatchers other than ``recall``, plus all 7 arch-mind
wrappers) were in neither ``ADMIN_TOOLS`` nor ``USER_TOOLS``.

That is not "unprivileged" — it is *unreachable*. ``mcp_tool_observe``
ends its ACL chain with ``elif tool_name not in USER_TOOLS: return
{"error": "Tool '<name>' is not in ACL policy"}``, which fires before
the tool body runs, and the ``MIND_MEM_ACL_DISABLED`` escape re-applies
the same unknown-tool rejection, so no configuration could reach them.
The mirror-image gap was also present: 5 ACL entries named no
registered tool, which would have silently pre-authorised a future tool
of that name at that scope without review.

The tests below pin both directions of the invariant, and probe the
enforcement mechanism itself rather than only set membership — a name
being in a frozenset proves nothing about whether the decorator lets
the call through.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from mind_mem.mcp.infra import rate_limit as _rate_limit
from mind_mem.mcp.infra.acl import ADMIN_TOOLS, USER_TOOLS
from mind_mem.mcp.infra.observability import mcp_tool_observe

_GATE_PASSED = "__gate_passed__"

# The 13 tools that were registered but unclassified, split by the scope
# each one must carry. Admin-scoped because the dispatcher reaches an
# admin capability through ``__wrapped__`` (which strips the decorator,
# and with it the only per-tool ACL check), or because the arch-mind
# wrapper writes to the arch-mind evidence store.
NEWLY_ADMIN = (
    "staged_change",  # -> propose_update / approve_apply / rollback_proposal
    "memory_verify",  # -> verify_chain (admin) + verify_merkle / mind_mem_verify
    "arch_baseline",  # writes <repo>/.arch-mind/baseline.json
    "arch_session_start",  # writes a session_start evidence node
    "arch_session_end",  # writes a session_end evidence node
    # The four READ-ONLY arch-mind wrappers are admin too, and deliberately so.
    # Classifying the 13 unclassified tools was the fix; granting user scope a
    # NEW capability was not. These shell out to the arch-mind binary against
    # any ABSOLUTE path on the host -- _validate_arch_path does not realpath or
    # stat -- and arch_check_rules reports path:line findings and distinguishes
    # a missing rules file, i.e. a directory-listing and file-existence oracle
    # outside the workspace. They were unreachable at EVERY scope before, so
    # admin-scoping completes the ACL without widening anything.
    "arch_history",
    "arch_delta",
    "arch_check_rules",
    "arch_metric_explain",
)
NEWLY_USER = (
    "graph",  # admin branch self-enforces enforce_capability_acl(graph_add_edge)
    "core",
    "kernels",
    "compiled_truth",
)


@pytest.fixture(autouse=True)
def _isolate_acl_env(monkeypatch, tmp_path):
    """Deterministic ACL inputs: user scope, no override, fresh budget.

    The per-client rate limiter is process-global and shared with every
    other test in the session; a full suite can leave the pid bucket
    near its 120-calls-per-minute cap, which would surface here as a
    rate-limit envelope instead of the ACL verdict under test.
    """
    monkeypatch.setenv("MIND_MEM_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("MIND_MEM_SCOPE", "user")
    monkeypatch.delenv("MIND_MEM_ACL_DISABLED", raising=False)
    with _rate_limit._rate_limiters_lock:
        _rate_limit._rate_limiters.clear()
    yield
    with _rate_limit._rate_limiters_lock:
        _rate_limit._rate_limiters.clear()


def _registered_tool_names() -> set[str]:
    """Names of every tool registered on the live FastMCP instance."""
    from mind_mem.mcp.server import mcp

    return {tool.name for tool in asyncio.run(mcp.list_tools())}


def _probe(tool_name: str) -> str:
    """Run *tool_name* through the real decorator gate with a stub body.

    ``mcp_tool_observe`` keys its ACL decision on ``fn.__name__``, so a
    stub carrying the tool's name exercises the identical gate the real
    tool does, without executing the tool body (arch-mind wrappers shell
    out to an external binary, and the admin dispatchers would write).
    Returns the sentinel when the gate let the call through, or the
    gate's JSON error string when it did not.
    """

    def _stub() -> str:
        return _GATE_PASSED

    _stub.__name__ = tool_name
    return mcp_tool_observe(_stub)()


class TestACLCoversTheRegisteredSurface:
    def test_every_registered_tool_is_classified(self):
        """No registered tool may be missing from both ACL sets.

        An unclassified tool is dead on arrival: the decorator's
        terminal ``not in USER_TOOLS`` branch rejects every call.
        """
        unclassified = sorted(_registered_tool_names() - (ADMIN_TOOLS | USER_TOOLS))
        assert unclassified == [], f"registered but unreachable (every call gets 'not in ACL policy'): {unclassified}"

    def test_no_acl_entry_without_a_registered_tool(self):
        """No ACL entry may name a tool that does not exist.

        A stale entry is a standing grant: register a tool with that
        name later and it inherits the scope with no review.
        """
        stale = sorted((ADMIN_TOOLS | USER_TOOLS) - _registered_tool_names())
        assert stale == [], f"ACL grants naming no registered tool: {stale}"

    def test_sets_stay_disjoint(self):
        assert ADMIN_TOOLS & USER_TOOLS == frozenset()


class TestPreviouslyUnreachableToolsAreReachable:
    """The 13 tools must now get a real scope verdict, not 'unknown'.

    Nine are admin, four are user. The split is a security decision, not a
    transcription: see the arch-mind note on NEWLY_ADMIN.
    """

    @pytest.mark.parametrize("tool_name", NEWLY_USER)
    def test_user_scoped_tool_passes_the_gate(self, tool_name):
        assert _probe(tool_name) == _GATE_PASSED

    @pytest.mark.parametrize("tool_name", NEWLY_ADMIN)
    def test_admin_scoped_tool_is_denied_for_scope_reason(self, tool_name):
        """Denied — but as an admin tool, not as an unknown one.

        Both classifications deny a user-scope call, so only the reason
        distinguishes a correct admin classification from the old
        unclassified state. Assert on the reason.
        """
        parsed = json.loads(_probe(tool_name))
        assert parsed["error"] == f"Permission denied: '{tool_name}' requires admin scope"
        assert parsed["scope"] == "user"

    @pytest.mark.parametrize("tool_name", NEWLY_ADMIN)
    def test_admin_scoped_tool_passes_the_gate_for_admin(self, tool_name, monkeypatch):
        monkeypatch.setenv("MIND_MEM_SCOPE", "admin")
        assert _probe(tool_name) == _GATE_PASSED


class TestConsolidatedDispatchersExecute:
    """End-to-end: the real dispatcher body runs, not just a stub."""

    def test_kernels_dispatcher_returns_its_payload(self):
        from mind_mem.mcp.tools import public

        parsed = json.loads(public.kernels("list"))
        assert "kernels" in parsed, parsed

    def test_graph_dispatcher_reaches_its_body(self):
        from mind_mem.mcp.tools import public

        parsed = json.loads(public.graph("stats"))
        # The tmp workspace is uninitialised, so a workspace error here
        # is expected and fine — what matters is that the call got past
        # the ACL gate into the tool body.
        assert "ACL policy" not in json.dumps(parsed)


class TestGraphDispatcherStillGuardsItsAdminBranch:
    """User-scoping ``graph`` must not open the graph_add_edge write.

    ``graph`` is user-scope only because its ``add_edge`` branch calls
    ``enforce_capability_acl("graph_add_edge")`` before dispatching. If
    that guard is ever removed, classifying the dispatcher as user-scope
    becomes a privilege escalation — this test fails first.
    """

    def test_add_edge_branch_denied_for_user_scope(self):
        from mind_mem.mcp.tools import public

        parsed = json.loads(
            public.graph(
                "add_edge",
                subject="s",
                predicate="p",
                object="o",
                source_block_id="b",
            )
        )
        assert parsed["error"] == "Permission denied: 'graph_add_edge' requires admin scope"

    def test_read_branch_allowed_for_user_scope(self):
        from mind_mem.mcp.tools import public

        assert "ACL policy" not in public.graph("stats")


class TestFailClosedBehaviourIsUnchanged:
    """The fix classifies names; it must not soften the gate itself."""

    def test_unknown_name_is_still_rejected(self):
        parsed = json.loads(_probe("a_tool_that_does_not_exist"))
        assert parsed["error"] == "Tool 'a_tool_that_does_not_exist' is not in ACL policy"

    @pytest.mark.parametrize(
        "stale_name",
        ["write_memory", "apply_proposal", "reindex_vectors", "search_memory", "list_memory"],
    )
    def test_removed_stale_grants_now_fail_closed(self, stale_name):
        """Dropping a stale entry tightens, never loosens.

        ``search_memory`` / ``list_memory`` were user-scope grants for
        tools that do not exist; with the entry gone, a tool later
        registered under that name is rejected until someone classifies
        it deliberately.
        """
        parsed = json.loads(_probe(stale_name))
        assert parsed["error"] == f"Tool '{stale_name}' is not in ACL policy"
