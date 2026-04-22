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
    }
)

_ADMIN_SCOPES = frozenset({"admin", "full", "mind-mem:admin"})


def check_tool_acl(tool_name: str, scope: str) -> str | None:
    """Check whether *scope* is allowed to call *tool_name*.

    Returns None if allowed, or a JSON error string if denied.
    """
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
    """Return ACL scope from the active FastMCP access token, if any."""
    try:
        access_token = get_access_token()
    except Exception:
        return None

    if access_token is None:
        return None

    token_scopes = set(access_token.scopes or [])
    return "admin" if token_scopes & _ADMIN_SCOPES else "user"
