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
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Iterator

from fastmcp.server.dependencies import get_access_token

from mind_mem.observability import get_logger, metrics
from mind_mem.scopes import ADMIN_SCOPES  # the single definition

_log = get_logger("mcp_server")


@dataclass(frozen=True)
class AuthSnapshot:
    """The one authentication lookup used by an observed tool call.

    ``status`` is intentionally small and internal: ``authenticated`` has a
    validated namespace principal, ``unbound`` is the legitimate no-token
    stdio case, and ``denied`` covers an unavailable or malformed authority.
    Keeping this in a task-local context prevents a later token lookup from
    changing the identity after rate/scope admission has already happened.
    """

    status: str
    principal: str | None = None
    scope: str | None = None
    client_id: str | None = None


_AUTH_SNAPSHOT: ContextVar[AuthSnapshot | None] = ContextVar("mcp_auth_snapshot", default=None)


def current_auth_snapshot() -> AuthSnapshot | None:
    """Return the request snapshot, if the MCP observer has bound one."""

    return _AUTH_SNAPSHOT.get()


def _snapshot_from_token(*, require_principal: bool = True) -> AuthSnapshot:
    """Resolve token metadata once, without exposing provider failures.

    The built-in HTTP static map supplies verified ``sub`` claims for
    ``mind-mem-user`` and ``mind-mem-admin``. Custom providers must provide a
    verified subject (or ``agent_id`` claim); an arbitrary client id is only a
    rate-limit key. Namespace access still follows each workspace's explicit
    ``mind-mem-acl.json`` grants.
    """

    try:
        access_token = get_access_token()
    except Exception as exc:
        _record_auth_failure(exc)
        return AuthSnapshot("denied")

    if access_token is None:
        from mind_mem.audit_context import UNATTRIBUTED, current_agent_id

        bound = current_agent_id.get()
        if bound != UNATTRIBUTED:
            return AuthSnapshot("authenticated", bound)
        return AuthSnapshot("unbound")

    try:
        claims: Any = getattr(access_token, "claims", None)
        if not isinstance(claims, dict):
            claims = {}
        # Namespace identity must come from a verified subject claim.  The
        # token client_id remains a rate-limit identity and is never promoted
        # into a namespace principal by itself.
        candidate = claims.get("sub") or getattr(access_token, "subject", None)
        if not candidate:
            candidate = claims.get("agent_id")
        if not isinstance(candidate, str) or not candidate:
            if not require_principal:
                raw_scopes = getattr(access_token, "scopes", ())
                token_scopes = set(raw_scopes or ())
                scope = "admin" if token_scopes & _ADMIN_SCOPES else "user"
                client_id = getattr(access_token, "client_id", None)
                if not isinstance(client_id, str) or not client_id:
                    client_id = None
                return AuthSnapshot("scoped", None, scope, client_id)
            raise ValueError("authenticated token has no namespace principal")
        from mind_mem.namespaces import _validate_agent_id

        principal = _validate_agent_id(candidate)
        from mind_mem.audit_context import UNATTRIBUTED, current_agent_id

        bound = current_agent_id.get()
        if bound != UNATTRIBUTED and bound != principal:
            raise ValueError("authenticated identity conflicts with bound transport identity")
        raw_scopes = getattr(access_token, "scopes", ())
        token_scopes = set(raw_scopes or ())
        scope = "admin" if token_scopes & _ADMIN_SCOPES else "user"
        client_id = getattr(access_token, "client_id", None)
        if not isinstance(client_id, str) or not client_id:
            client_id = None
        return AuthSnapshot("authenticated", principal, scope, client_id)
    except Exception as exc:
        _record_auth_failure(exc)
        return AuthSnapshot("denied")


def _record_auth_failure(exc: BaseException) -> None:
    """Record an auth failure without including credentials or payloads."""

    try:
        metrics.inc("mcp_acl_introspection_failed_total")
    except Exception:  # pragma: no cover - metrics are best effort
        pass
    _log.warning("acl_introspection_failed", error_type=type(exc).__name__, scope="deny")


@contextmanager
def bind_auth_snapshot() -> Iterator[AuthSnapshot]:
    """Bind one auth snapshot and its principal for a complete tool call."""

    existing = current_auth_snapshot()
    if existing is not None:
        # Consolidated tools can invoke another decorated callable. Reusing
        # the outer frame is essential: a second provider lookup could return
        # different metadata and split one logical request across principals.
        yield existing
        return

    snapshot = _snapshot_from_token()
    snapshot_token = _AUTH_SNAPSHOT.set(snapshot)
    try:
        from mind_mem.audit_context import UNATTRIBUTED, bind_current_agent, current_agent_id

        bound = current_agent_id.get()
        if snapshot.status == "authenticated" and snapshot.principal:
            if bound != UNATTRIBUTED and bound != snapshot.principal:
                # Normally caught during resolution; retain a defensive
                # check for a provider that mutates token metadata mid-call.
                yield AuthSnapshot("denied")
            elif bound == UNATTRIBUTED:
                with bind_current_agent(snapshot.principal):
                    yield snapshot
            else:
                yield snapshot
        else:
            yield snapshot
    finally:
        _AUTH_SNAPSHOT.reset(snapshot_token)


def authenticated_agent_id() -> str | None:
    """Return the transport-authenticated namespace principal, if present.

    ``X-MindMem-Actor`` is a provenance claim and is deliberately not read
    here.  FastMCP exposes the already-verified access token to tool code;
    static and JWT providers carry the subject in ``claims['sub']`` (or the
    token subject). A token's client id is retained for rate limiting and is
    never treated as a namespace principal.
    Direct stdio calls have no access token and retain the existing
    workspace-level behavior.  A pre-bound internal transport context wins so
    REST/gRPC adapters and source-bound tests use the same principal seam.
    """
    snapshot = current_auth_snapshot()
    from mind_mem.audit_context import UNATTRIBUTED, current_agent_id

    bound = current_agent_id.get()
    if snapshot is None and bound and bound != UNATTRIBUTED:
        return bound

    # ``None`` is the SDK's explicit no-request result (stdio and legacy
    # operator calls). An exception means an authenticated transport could
    # not be inspected; propagating it keeps the public recall body from
    # silently becoming workspace-wide after an authn failure.
    if snapshot is None:
        snapshot = _snapshot_from_token()
    if snapshot.status == "denied":
        raise ValueError("authentication context unavailable")
    if snapshot.status == "authenticated":
        return snapshot.principal
    if snapshot.status == "unbound":
        return None
    raise ValueError("authenticated token has no namespace principal")


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
        "propose_slot_update",
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
        # lint_autofix STAGES a repair proposal in intelligence/proposed/ —
        # it never writes the corpus, but putting an item in front of the
        # operator's approval gate is the same class of act as
        # propose_update, which is admin here. Its read-only twin ``lint``
        # is user-scope; see USER_TOOLS.
        "lint_autofix",
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
        # Deterministic corpus lint — reads the corpus, writes nothing and
        # proposes nothing. The repair half (lint_autofix) is admin-scoped.
        "lint",
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
        # export_core RENDERS a loaded bundle into a static interchange
        # format under memory/cores/exports/. It reads the corpus snapshot
        # a user-scope build_core already produced and writes only an
        # export artifact — never the corpus, never a proposal — so it
        # carries build_core's scope, not admin.
        "export_core",
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
        # similar_trajectories READS the trajectories/ sidecar that
        # report_outcome (a user-scope tool) writes, through admit_corpus.
        # It writes nothing, proposes nothing, and never reaches the corpus,
        # so it carries report_outcome's scope rather than admin.
        "similar_trajectories",
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
    snapshot = current_auth_snapshot()
    if snapshot is None:
        snapshot = _snapshot_from_token(require_principal=False)
    if snapshot.status == "denied":
        return "deny"
    return snapshot.scope
