"""Per-tool ACL — scope enforcement for the MCP surface.

Extracted from ``mcp_server.py`` in the v3.2.0 §1.2 decomposition
(see docs/v3.2.0-mcp-decomposition-plan.md PR-1). Two frozensets
(``ADMIN_TOOLS``, ``USER_TOOLS``) classify every ``@mcp.tool`` by
required scope; ``check_tool_acl`` is the gate consulted before
each tool body runs; ``_get_request_scope`` resolves the scope of
the active FastMCP access token. Behavior is bit-for-bit identical
to the pre-move version — the metric name ``mcp_acl_denied`` and
the log category ``mcp_server`` are preserved so dashboards and
log-based assertions keep working.
"""

from __future__ import annotations

import json

from fastmcp.server.dependencies import get_access_token

from mind_mem.observability import get_logger, metrics

_log = get_logger("mcp_server")


ADMIN_TOOLS = frozenset(
    {
        "write_memory",
        "apply_proposal",
        "approve_apply",
        "reject_proposal",
        "rollback_proposal",
        "delete_memory_item",
        "reindex_vectors",
        "propose_update",
        "reindex",
        "export_memory",
        "verify_chain",
        "compact",
        "encrypt_file",
        "decrypt_file",
    }
)

USER_TOOLS = frozenset(
    {
        "recall",
        "recall_with_axis",
        "verify_merkle",
        "mind_mem_verify",
        "observe_signal",
        "signal_stats",
        "graph_query",
        "graph_stats",
        "graph_add_edge",
        "build_core",
        "load_core",
        "unload_core",
        "list_cores",
        "plan_consolidation",
        "pack_recall_budget",
        "ontology_load",
        "ontology_validate",
        "stream_status",
        "propagate_staleness",
        "project_profile",
        "vault_sync",
        "vault_scan",
        "agent_inject",
        "search_memory",
        "list_memory",
        "list_contradictions",
        "scan",
        "hybrid_search",
        "find_similar",
        "intent_classify",
        "index_stats",
        "retrieval_diagnostics",
        "memory_evolution",
        "category_summary",
        "prefetch",
        "list_mind_kernels",
        "get_mind_kernel",
        "calibration_feedback",
        "calibration_stats",
        "list_evidence",
        "get_block",
        "memory_health",
        "traverse_graph",
        "stale_blocks",
        "dream_cycle",
        "compiled_truth_load",
        "compiled_truth_add_evidence",
        "compiled_truth_contradictions",
        "governance_health_bench",
        # v3.11.0 — quality gate + typed lineage edges.
        "validate_block",
        "block_lineage",
        "add_block_edge",
        # v3.11.1 — backfill ACL gaps surfaced by the v3.11.0 audit.
        # These tools were registered in v3.8.x/v3.9.0 but never added
        # to the whitelist; the security-hardening commit that enforced
        # ACL didn't catch them. Tests that exercise them have been
        # silently failing on `acl_unknown_tool` since v3.8.4.
        "audit_model_tool",
        "sign_model_tool",
        "verify_model_tool",
        "compile_truth_walkthrough",
        "recall_with_persona",
        "mic_convert_tool",
        "mic_inspect_tool",
        "pipeline_status",
        "reindex_dirty",
    }
)

_ADMIN_SCOPES = frozenset({"admin", "full", "mind-mem:admin"})


def check_tool_acl(tool_name: str, scope: str) -> str | None:
    """Check whether *scope* is allowed to call *tool_name*.

    Returns None if allowed, or a JSON error string if denied.

    Issue #526: scope == "deny" is the fail-closed sentinel returned by
    ``_get_request_scope`` when token introspection raises. Reject
    every tool — admin or user — when we see it.
    """
    if scope == "deny":
        metrics.inc("mcp_acl_denied")
        _log.warning("acl_denied", tool=tool_name, scope=scope, reason="introspection_failed")
        return json.dumps(
            {
                "error": "Permission denied: authentication context unavailable",
                "scope": scope,
            }
        )
    if tool_name in ADMIN_TOOLS and scope != "admin":
        metrics.inc("mcp_acl_denied")
        _log.warning("acl_denied", tool=tool_name, scope=scope)
        return json.dumps(
            {
                "error": f"Permission denied: '{tool_name}' requires admin scope",
                "scope": scope,
                "hint": "Admin scope is controlled via MIND_MEM_SCOPE=admin env var.",
            }
        )
    return None


def _get_request_scope() -> str | None:
    """Return ACL scope from the active FastMCP access token, if any.

    Issue #526 (Critical, fail-closed): any exception from
    ``get_access_token()`` previously degraded silently to ``None``,
    which then fell through to ``"user"`` at the call site — turning a
    transient introspection error into an authn-context drop. Now:

      • Exceptions return the sentinel ``"deny"`` so ``enforce_acl``
        rejects the call (admin tools become inaccessible, user tools
        also become inaccessible — fail-closed).
      • The exception type + token prefix (first 4 chars only) are
        logged so operators have signal.
      • A counter is bumped so dashboards can alert on the rate.

    ``access_token is None`` is the legitimate "no auth context"
    branch (stdio, unauthenticated HTTP) and still returns ``None`` so
    the caller's default-scope policy applies.
    """
    try:
        access_token = get_access_token()
    except Exception as exc:
        # First-4-char token prefix is safe to log (entropy < 24 bits)
        # and lets operators correlate failures without exposing the
        # full credential.
        try:
            from .observability import metrics

            metrics.inc("mcp_acl_introspection_failed_total")
        except Exception:
            pass
        _log.warning(
            "acl_introspection_failed",
            error_type=type(exc).__name__,
            scope="deny",
        )
        return "deny"

    if access_token is None:
        return None

    token_scopes = set(access_token.scopes or [])
    return "admin" if token_scopes & _ADMIN_SCOPES else "user"
