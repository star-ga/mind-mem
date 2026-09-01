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
import os

from fastmcp.server.dependencies import get_access_token

from mind_mem.observability import get_logger, metrics
from mind_mem.scopes import ADMIN_SCOPES  # the single definition

_log = get_logger("mcp_server")


# ACL COVERAGE INVARIANT (pinned by tests/test_acl_tool_coverage.py):
# ``ADMIN_TOOLS | USER_TOOLS`` must equal exactly the set of tool names
# registered on the FastMCP instance in ``mind_mem.mcp.server``.
#
#   • A registered tool in NEITHER set is unreachable, not merely
#     unprivileged: ``mcp_tool_observe``'s terminal ``not in USER_TOOLS``
#     branch rejects the call with "is not in ACL policy" before the body
#     runs, and the ``MIND_MEM_ACL_DISABLED`` escape re-applies the same
#     unknown-tool rejection, so no configuration can reach it.
#   • A name in either set with no registered tool is a stale grant that
#     would silently pre-authorise a future tool of that name at that
#     scope, with no review.
#
# So: classify every tool into exactly one set in the same change that
# registers it. (The stale entries ``write_memory``, ``apply_proposal``,
# ``reindex_vectors``, ``search_memory`` and ``list_memory`` were dropped
# for the second reason — no tool of those names is registered.)
# The four read-only arch-mind wrappers are ADMIN, not USER. Classifying the
# 13 unclassified tools was the fix; granting user scope a NEW capability was
# not. These shell out to the arch-mind binary against any ABSOLUTE path on the
# host, and arch_check_rules reports path:line findings and distinguishes a
# missing rules file — a directory-listing and file-existence oracle outside the
# workspace. They were unreachable at every scope before, so admin-scoping keeps
# the ACL complete without widening anything.
ADMIN_TOOLS = frozenset(
    {
        "approve_apply",
        "reject_proposal",
        "rollback_proposal",
        "delete_memory_item",
        "propose_update",
        "reindex",
        "export_memory",
        "verify_chain",
        # anchor_root APPENDS to the workspace anchor trail, so it is a write
        # even though it reads the Merkle root to do it. Admin, like the rest
        # of the chain-mutating surface.
        "anchor_root",
        "compact",
        "encrypt_file",
        "decrypt_file",
        # Direct typed-knowledge-graph edge writes bypass the HITL
        # proposal gate, so they require the admin scope. User-scope
        # ingestion routes through graph_ingest signal staging +
        # operator approval instead.
        "graph_add_edge",
        # HITL typed-edge proposal flow: approving / rejecting a proposal
        # is an operator governance decision. approve_edge is the SOLE
        # committer of the propose→approve path (it mutates the
        # source-of-truth edges table); reject_edge changes proposal state.
        # Both are admin-scoped. (propose_edge / list_edge_proposals are
        # user-scope — staging + read never touch the graph.)
        "approve_edge",
        "reject_edge",
        # Entity observation writes mutate the entity registry.
        "entity_add_observation",
        # Consolidated dispatchers that can reach an admin capability.
        # Both invoke their callee through ``__wrapped__``, which strips
        # ``@mcp_tool_observe`` — the only place per-tool ACL runs — so
        # the DISPATCHER name is the sole remaining gate and must carry
        # the scope of its most privileged branch:
        #   staged_change → propose_update / approve_apply /
        #                   rollback_proposal (all admin).
        #   memory_verify → verify_chain (admin), plus verify_merkle and
        #                   mind_mem_verify (user); admin wins.
        # ``graph`` is user-scope by contrast because its one admin
        # branch calls ``enforce_capability_acl("graph_add_edge")``
        # itself before dispatching.
        "staged_change",
        "memory_verify",
        # arch-mind wrappers that MUTATE the arch-mind evidence store at
        # a caller-supplied repository path — baseline.json for
        # arch_baseline, chained session_start / session_end evidence
        # nodes for the session pair. The read-only arch_* wrappers are
        # user-scope; see USER_TOOLS.
        "arch_baseline",
        "arch_history",
        "arch_delta",
        "arch_check_rules",
        "arch_metric_explain",
        "arch_session_start",
        "arch_session_end",
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
        # HITL typed-edge proposal flow — staging + read only, never a
        # source-of-truth write (approve_edge/reject_edge are admin-scoped).
        "propose_edge",
        "list_edge_proposals",
        # Entity observations — read-only view of accreted per-entity facts.
        "entity_observations",
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
        "report_outcome",
        "outcome_stats",
        "list_evidence",
        # Read-only view of the anchor trail (and its integrity problems).
        "anchor_history",
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
        # v4.9.2 — conversational chat layer. Read-only: recalls
        # evidence and returns a cited answer; never writes.
        "chat_with_memory",
        # GUARDRAIL blocks — read-only trigger evaluation + guardrail-first
        # recall. Never writes: guardrails are authored through
        # propose_update -> HITL like every other block kind.
        "check_guardrails",
        "recall_with_guardrails",
        # TASK-FRAME / DEAD-END blocks — read-only session continuity and
        # negative action-space memory. Never writes: frames and dead ends
        # are authored through propose_update -> HITL like every other
        # block kind, and a dead end warns without ever blocking.
        "resume_brief",
        "check_dead_ends",
        # v3.2.0 consolidated dispatchers. These are registered in
        # ``mcp.tools.public`` but were never classified, so the
        # unknown-tool branch rejected every call to them — the
        # consolidated surface was advertised and unreachable. Each of
        # these routes only to user-scope callees, except ``graph``,
        # whose sole admin branch (``add_edge``) enforces the admin
        # capability itself. ``staged_change`` and ``memory_verify``
        # reach admin capabilities with no such guard and are in
        # ADMIN_TOOLS instead.
        "graph",
        "core",
        "kernels",
        "compiled_truth",
        # arch-mind wrappers — read-only analysis only: list the evidence
        # store, diff two baselines, apply rules to a fixture, explain one
        # metric. Nothing here writes an evidence node; the three that do
        # are admin-scoped above.
    }
)

# The ONE admin-scope vocabulary. src/mind_mem/api/rest.py imports this rather
# than carrying its own literal: the two disagreed (REST accepted only "admin"),
# and while the REST admin gate was being skipped entirely that difference was
# invisible. Fixing the gate made it reachable and would have locked every
# "full"-scoped key out of the REST admin endpoints. Two layers asking different
# questions about the same word is the defect class this codebase keeps hitting.
#: Re-exported from :mod:`mind_mem.scopes` (imported at the top of this file).
_ADMIN_SCOPES = ADMIN_SCOPES  # back-compat for existing in-module references


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


def enforce_capability_acl(capability: str) -> str | None:
    """Enforce the ACL for *capability* exactly as ``@mcp_tool_observe``
    would for a tool of that name. Returns None if allowed, or the same
    JSON error string the decorator returns when denied.

    Confused-deputy guard for consolidated dispatchers: they invoke the
    underlying tool via ``__wrapped__`` (to avoid double-charging the
    rate limiter), which also strips the decorator's ACL gate — the
    only enforcement point. A dispatcher branch that maps to an
    admin-scope capability must therefore call this BEFORE the
    ``__wrapped__`` call, so the check binds to the CAPABILITY and
    cannot regress no matter how the dispatcher name itself is later
    classified in ``ADMIN_TOOLS`` / ``USER_TOOLS``.

    Scope resolution mirrors the decorator: the ``deny`` fail-closed
    sentinel wins over everything; otherwise the token scope, falling
    back to ``MIND_MEM_SCOPE`` (default ``user``). The documented
    ``MIND_MEM_ACL_DISABLED`` dev/test override is honoured with the
    same audited ``acl_bypassed_via_env`` warning the decorator emits.
    """
    scope = _get_request_scope()
    if scope == "deny":
        return check_tool_acl(capability, "deny")
    acl_scope = scope or os.environ.get("MIND_MEM_SCOPE", "user")
    if os.environ.get("MIND_MEM_ACL_DISABLED", "").lower() in ("1", "true", "yes"):
        if capability in ADMIN_TOOLS:
            _log.warning(
                "acl_bypassed_via_env",
                extra={
                    "tool": capability,
                    "reason": "MIND_MEM_ACL_DISABLED",
                    "scope": acl_scope,
                },
            )
        return None
    return check_tool_acl(capability, acl_scope)


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
        except Exception:  # nosec B110 — metric counter increment; outer except already handles the real auth failure
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
