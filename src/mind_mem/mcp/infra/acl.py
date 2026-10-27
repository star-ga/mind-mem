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
from typing import Any, Callable, Iterator

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
    except Exception as metric_exc:  # pragma: no cover - metrics are best effort
        _log.warning("acl_failure_metric_unavailable", error_type=type(metric_exc).__name__)
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


@contextmanager
def bind_transport_auth(*, principal: str, scope: str) -> Iterator[AuthSnapshot]:
    """Bind a transport's already-verified auth decision for one call.

    REST authentication runs in a FastAPI dependency threadpool, while the
    synchronous endpoint and the observed MCP callable run in another
    context.  This seam carries the *decision* made by the REST dependency;
    it never trusts a caller header or re-reads ambient ``MIND_MEM_SCOPE``.
    A nested transport may reuse the same decision only when both principal
    and scope match; a mismatch is a fail-closed ``denied`` snapshot.
    """
    if scope not in {"admin", "user"}:
        raise ValueError("transport scope must be 'admin' or 'user'")
    existing = current_auth_snapshot()
    if existing is not None:
        if existing.status == "authenticated" and existing.principal == principal and existing.scope == scope:
            yield existing
            return
        conflict = AuthSnapshot("denied")
        token = _AUTH_SNAPSHOT.set(conflict)
        try:
            yield conflict
        finally:
            _AUTH_SNAPSHOT.reset(token)
        return

    snapshot = AuthSnapshot("authenticated", principal, scope)
    snapshot_token = _AUTH_SNAPSHOT.set(snapshot)
    try:
        from mind_mem.audit_context import bind_current_agent

        with bind_current_agent(principal):
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
        "approve_entity_merge",
        "reverse_entity_merge",
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
        # F14 (architecture review): a tool that writes in ANY argument
        # combination is admin. Each of these was user-scope and was measured
        # writing at user scope (tests/test_acl_user_scope_read_only.py):
        #   add_block_edge             lineage edge + evidence-chain entry
        #   compiled_truth_add_evidence  entities/compiled/*.md + evidence chain
        #   ontology_load              the PROCESS-GLOBAL ontology registry, so
        #                              on HTTP one client changed what every
        #                              other client's ontology_validate used
        #                              (make_active=False still replaces a
        #                              loaded version of the same name)
        #   vault_sync                 writes a file into an allow-listed vault
        #   sign_model_tool            writes key material + manifest sidecars
        #                              at a caller-chosen host path
        #   build_core / export_core   .mmcore bundle / export artifact files
        #   load_core / unload_core    the process-global core registry
        #   governance_health_bench    its drift probe writes signals into the
        #                              workspace drift store
        # Their read-only neighbours (ontology_validate, vault_scan,
        # verify_model_tool, audit_model_tool, list_cores, block_lineage,
        # compiled_truth_load) stay user-scope.
        "add_block_edge",
        "compiled_truth_add_evidence",
        "ontology_load",
        "vault_sync",
        # calibration_feedback writes the calibration_feedback table, which
        # recall reads on EVERY query (CalibrationManager in _recall_core)
        # to reweight served scores -- there is no admin step between the
        # row and every client's ranking, so it is not a staging write.
        "calibration_feedback",
        # report_outcome appends to recall_outcome, which the validity gate
        # serves to every client under the supported opt-in
        # recall.validity_gate.outcome_attribution. Every row counts (no actor
        # or scope filter), so a row written while the opt-in is off is served
        # the moment it turns on: no config read at write time can make the
        # write staging. Derived from STAGING_STORE_READERS and held there by
        # tests/test_acl_staging_readers.py.
        "report_outcome",
        "sign_model_tool",
        "build_core",
        "export_core",
        "load_core",
        "unload_core",
        "governance_health_bench",
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
        "propose_entity_merge",
        "list_entity_merge_proposals",
        # Entity observations — read-only view of accreted per-entity facts.
        "entity_observations",
        "list_cores",
        "plan_consolidation",
        "pack_recall_budget",
        "ontology_validate",
        "stream_status",
        "propagate_staleness",
        "project_profile",
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
        "calibration_stats",
        "outcome_stats",
        # similar_trajectories READS the trajectories/ sidecar that the admin
        # tool report_outcome writes, through admit_corpus. It writes nothing,
        # proposes nothing, and never reaches the corpus.
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
        "compiled_truth_contradictions",
        # v3.11.0 — quality gate + typed lineage edges.
        "validate_block",
        "block_lineage",
        # v3.11.1 — backfill ACL gaps surfaced by the v3.11.0 audit.
        # These tools were registered in v3.8.x/v3.9.0 but never added
        # to the whitelist; the security-hardening commit that enforced
        # ACL didn't catch them. Tests that exercise them have been
        # silently failing on `acl_unknown_tool` since v3.8.4.
        "audit_model_tool",
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
        # consolidated surface was advertised and unreachable. Their READ
        # branches are user-scope; every branch that writes is denied at
        # user scope by WRITE_MODE_GATES below (graph add_edge, core
        # build/load/unload, compiled_truth add_evidence) — ``graph`` also
        # enforces its admin branch itself. ``staged_change`` and
        # ``memory_verify`` have no read branch worth splitting out and are
        # in ADMIN_TOOLS instead.
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

#: User-scope tools that WRITE, pinned. Each one appends to its own HITL
#: staging or feedback store and nothing else: a proposal waits for an admin
#: approve_* call before it reaches the source of truth, a signal row is an
#: input the operator-side tuning reads. Staging holds only while NO reader of
#: the store changes what other clients are served -- on OR off: a row written
#: while a serving reader is off is served the moment the operator turns it
#: on. So a store with any ``serving`` reader in ``STAGING_STORE_READERS`` is
#: never staging (``recall_outcome``; its writer report_outcome is admin), and
#: ``_gate_staging_readers`` denies such a pin at runtime whatever the config
#: says. A store that recall reads by default (calibration_feedback) is not
#: staging either: its writers are admin. This is the only write a
#: user-scope tool may make; tests/test_acl_user_scope_read_only.py measures
#: every user tool against it and holds the set as an exact ratchet, so a new
#: entry is a reviewed decision and not a side effect. Values are the logical
#: snapshot keys (``<workspace-relative path>`` or ``<db path>::<table>``).
USER_STAGING_WRITES: dict[str, tuple[str, ...]] = {
    "observe_signal": ("memory/interaction_signals.jsonl",),
    "propose_edge": ("memory/knowledge_graph.db::edge_proposals",),
    "propose_entity_merge": ("memory/knowledge_graph.db::entity_merge_proposals",),
}


def staging_store_token(store: str) -> str:
    """The name a staged store goes by in code: its table, else its file stem."""
    if "::" in store:
        return store.rsplit("::", 1)[1]
    return os.path.splitext(os.path.basename(store))[0]


@dataclass(frozen=True)
class StoreReader:
    """How one function touches a staged store.

    ``role`` is ``plumbing`` (passes rows on; whatever calls it is classified
    in turn), ``writer`` (the staging append itself), ``report`` (shows rows to
    the caller and moves nothing), ``schema`` (creates the store and reads no
    row of it), ``derived`` (feeds an artifact that reaches
    other clients only through a separate admin step), ``approval`` (applies a
    staged row to shared state; an admin tool) or ``serving`` (changes what
    OTHER clients are served). A serving reader names the config keys that turn
    it on -- all of them must be on for it to read -- so the all-readers-on test
    template can enable it; whether it is on today does not matter to the ACL:
    its store is never user-writable.
    """

    role: str
    enabled_by: tuple[str, ...] = ()
    why: str = ""


_P = StoreReader("plumbing")
_W = StoreReader("writer")
_R = StoreReader("report")
_S = StoreReader("schema")

#: Every function that touches a staged store -- one a user-scope tool writes,
#: or one taken out of user scope because a reader serves it -- directly,
#: through plumbing, or through a module/class constant that names it, keyed
#: by ``staging_store_token``. tests/test_acl_staging_readers.py derives the
#: touching set from the source (AST, transitively) and fails on any function
#: missing here, so a new reader -- and the config key that turns it on -- is
#: classified in the change that adds it; a serving reader makes every MCP
#: writer of its store admin, and the all-readers-on template in
#: tests/test_acl_user_scope_read_only.py enables every such key.
STAGING_STORE_READERS: dict[str, dict[str, StoreReader]] = {
    "recall_outcome": {
        "mind_mem.calibration._init_calibration_schema": _S,
        "mind_mem.outcome_store.record_outcome": _W,
        "mind_mem.calibration.CalibrationManager.record_outcome": _W,
        "mind_mem.outcome_attribution.report_outcome": _W,
        "mind_mem.mcp.tools.calibration.report_outcome": _W,
        "mind_mem.outcome_store.get_outcome_signals": _P,
        "mind_mem.calibration.CalibrationManager.get_outcome_signals": _P,
        "mind_mem.outcome_attribution.load_outcome_signals": _P,
        "mind_mem.validity_gate._load_outcome_signals": StoreReader(
            "serving",
            ("recall.validity_gate", "recall.validity_gate.outcome_attribution"),
            "every client's recall reweights its served scores by these outcomes",
        ),
        "mind_mem.outcome_store.get_outcome_stats": _R,
        "mind_mem.outcome_store.list_outcomes": _R,
        "mind_mem.calibration.CalibrationManager.get_outcome_stats": _R,
        "mind_mem.calibration.CalibrationManager.list_outcomes": _R,
        "mind_mem.outcome_attribution.outcome_stats": _R,
        "mind_mem.accountability_views.credit_rows": _R,
        # Computed abstention is scored by joining its recorded predictions to
        # these outcomes. Every function below hands the caller AGGREGATE counts
        # (per band / use / basis: success, failure, neutral) -- no block id, no
        # text -- and moves nothing; recall never reads either side of the join.
        "mind_mem.abstention_store.stats_from_manager": _R,
        "mind_mem.abstention_store.abstention_stats": _R,
        "mind_mem.calibration.CalibrationManager.get_abstention_stats": _R,
        "mind_mem.calibration.CalibrationManager.get_calibration_stats": _R,
        "mind_mem.mcp.tools.calibration.calibration_stats": _R,
        "mind_mem.outcome_store._abstention_section": _P,
    },
    "interaction_signals": {
        "mind_mem.mcp.tools.signal.observe_signal": _W,
        "mind_mem.mcp.tools._helpers._signal_store_path": _P,
        "mind_mem.online_trainer._signal_ledger_path": _P,
        "mind_mem.online_trainer.run_harvest_job": StoreReader(
            "derived",
            why="appends training tuples; weights reach recall only through the governed promotion step",
        ),
        "mind_mem.online_trainer.harvest_stats": _R,
        "mind_mem.mcp.tools.signal.signal_stats": _R,
        "mind_mem.mcp.tools.memory_ops.index_stats": _R,
        "mind_mem.mcp.tools.memory_ops._mrr_drift": _R,
    },
    "edge_proposals": {
        "mind_mem.knowledge_graph.KnowledgeGraph.__init__": _S,
        "mind_mem.knowledge_graph.KnowledgeGraph.propose_edge": _W,
        "mind_mem.mcp.tools.graph.propose_edge": _W,
        "mind_mem.knowledge_graph.KnowledgeGraph.get_edge_proposal": _P,
        "mind_mem.knowledge_graph.KnowledgeGraph.list_edge_proposals": _R,
        "mind_mem.knowledge_graph.KnowledgeGraph.approve_edge": StoreReader("approval"),
        "mind_mem.knowledge_graph.KnowledgeGraph.reject_edge": StoreReader("approval"),
        "mind_mem.mcp.tools.graph.approve_edge": StoreReader("approval"),
    },
    "entity_merge_proposals": {
        "mind_mem.knowledge_graph.KnowledgeGraph.__init__": _S,
        "mind_mem.knowledge_graph.KnowledgeGraph.propose_entity_merge": _W,
        "mind_mem.knowledge_graph.KnowledgeGraph.get_entity_merge_proposal": _P,
        "mind_mem.knowledge_graph.KnowledgeGraph.list_entity_merge_proposals": _R,
        "mind_mem.knowledge_graph.KnowledgeGraph.approve_entity_merge": StoreReader("approval"),
        "mind_mem.knowledge_graph.KnowledgeGraph.reverse_entity_merge": StoreReader("approval"),
        "mind_mem.mcp.tools.graph.approve_entity_merge": StoreReader("approval"),
        "mind_mem.mcp.tools.graph.reverse_entity_merge": StoreReader("approval"),
    },
}


def cross_client_reader_config_keys() -> tuple[str, ...]:
    """Every config key that turns on a reader serving a staged store to others."""
    keys = {k for readers in STAGING_STORE_READERS.values() for r in readers.values() if r.role == "serving" for k in r.enabled_by}
    return tuple(sorted(keys))


def _gate_staging_readers(tool_name: str) -> WriteMode | None:
    """Deny a staging write to a store any reader serves to other clients.

    Config-independent on purpose: a row written while the reader is off is
    served once it is turned on, so "off right now" is no reason to stage.
    Reads no file and no clock -- the answer is a property of the tables.
    """
    for store in USER_STAGING_WRITES.get(tool_name, ()):
        for func, reader in STAGING_STORE_READERS.get(staging_store_token(store), {}).items():
            if reader.role == "serving":
                return WriteMode(f"{tool_name}({reader.enabled_by[-1]})", f"{reader.why} ({func})")
    return None


@dataclass(frozen=True)
class WriteMode:
    """An argument combination of a user-scope tool that writes.

    ``capability`` names what the call reaches -- an admin tool for a
    consolidated dispatcher branch, otherwise ``tool(argument)`` -- and is
    what the denial reports, so a client learns which admin tool (or mode)
    does what it asked.
    """

    capability: str
    detail: str


def _gate_stale_blocks(args: dict[str, Any]) -> WriteMode | None:
    if args.get("clear_block_id"):
        return WriteMode("stale_blocks(clear_block_id)", "clears a staleness flag in the causal graph")
    return None


def _gate_memory_evolution(args: dict[str, Any]) -> WriteMode | None:
    # Fail closed on anything but the read action: an unknown action gets a
    # scope error at user scope instead of reaching the body.
    if args.get("action", "get") != "get":
        return WriteMode("memory_evolution(action=update)", "rewrites stored block importance")
    return None


def _gate_reindex_dirty(args: dict[str, Any]) -> WriteMode | None:
    if not args.get("dry_run", False):
        return WriteMode("reindex_dirty(dry_run=False)", "rewrites dirty corpus blocks and appends evidence")
    return None


def _gate_dream_cycle(args: dict[str, Any]) -> WriteMode | None:
    # The default mode is served at user scope as a dry run (the body asks
    # request_has_admin_scope); auto_repair is the mode that mutates.
    if args.get("auto_repair", False):
        return WriteMode("dream_cycle(auto_repair=True)", "applies repairs to the workspace")
    return None


_CORE_WRITE_ACTIONS = {"build": "build_core", "load": "load_core", "unload": "unload_core"}


def _gate_core(args: dict[str, Any]) -> WriteMode | None:
    capability = _CORE_WRITE_ACTIONS.get(args.get("action", ""))
    if capability is not None:
        return WriteMode(capability, "writes a core bundle or the process-global core registry")
    return None


def _gate_compiled_truth(args: dict[str, Any]) -> WriteMode | None:
    if args.get("action") == "add_evidence":
        return WriteMode("compiled_truth_add_evidence", "writes compiled truth and the evidence chain")
    return None


def _gate_graph(args: dict[str, Any]) -> WriteMode | None:
    # The dispatcher also enforces this itself (enforce_capability_acl);
    # gating it here as well keeps the one table complete.
    if args.get("action") == "add_edge":
        return WriteMode("graph_add_edge", "writes a knowledge-graph edge")
    return None


#: Argument-level ACL for user-scope tools whose READ mode is user-scope and
#: whose WRITE mode is not. ``mcp_tool_observe`` binds the call's arguments
#: to the tool signature (defaults applied, so positional and keyword calls
#: are judged alike) and consults this table before the body runs; a write
#: mode at non-admin scope gets the same denial envelope as an admin tool.
#: Kept here, beside the scope sets, so classification lives in one file.
WRITE_MODE_GATES: dict[str, Callable[[dict[str, Any]], WriteMode | None]] = {
    "stale_blocks": _gate_stale_blocks,
    "memory_evolution": _gate_memory_evolution,
    "reindex_dirty": _gate_reindex_dirty,
    "dream_cycle": _gate_dream_cycle,
    "core": _gate_core,
    "compiled_truth": _gate_compiled_truth,
    "graph": _gate_graph,
}

#: MCP resources (``@mcp.resource``) carry a scope too. Every one is a
#: read-only, admission-filtered view, so all are user-scope; the set exists
#: so a resource reaches the same gate a tool does (fail-closed ``deny``
#: sentinel, per-client rate limit, unknown-name rejection) and so a new
#: resource is classified in the change that registers it. Pinned against the
#: live registration by tests/test_resource_acl_rate_limit.py.
USER_RESOURCES = frozenset(
    {
        "get_decisions",
        "get_tasks",
        "get_entities",
        "get_signals",
        "get_contradictions",
        "get_health",
        "get_recall",
        "get_ledger",
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


#: Every tool whose call is judged by ``write_mode_of``: the argument gates
#: plus the staging writers, whose write is admin if a reader serves their
#: store to other clients.
WRITE_GATED_TOOLS = frozenset(WRITE_MODE_GATES) | frozenset(USER_STAGING_WRITES)


def write_mode_of(tool_name: str, arguments: dict[str, Any]) -> WriteMode | None:
    """The cross-client write this call would make, or None for a read/staging."""
    gate = WRITE_MODE_GATES.get(tool_name)
    mode = gate(arguments) if gate is not None else None
    return mode if mode is not None else _gate_staging_readers(tool_name)


def check_write_mode_acl(tool_name: str, arguments: dict[str, Any], scope: str) -> str | None:
    """Deny a user-scope tool's WRITE mode at non-admin *scope*.

    Returns None when the call is allowed (admin scope, an ungated tool, or a
    read mode), else the JSON denial. *arguments* must already be bound to
    the tool signature with defaults applied.
    """
    if scope == "admin":
        return None
    mode = write_mode_of(tool_name, arguments)
    if mode is None:
        return None
    metrics.inc("mcp_acl_denied")
    _log.warning("acl_denied", tool=tool_name, scope=scope, write_mode=mode.capability)
    return json.dumps(
        {
            "error": f"Permission denied: '{mode.capability}' requires admin scope",
            "tool": tool_name,
            "write_mode": mode.detail,
            "scope": scope,
            "hint": "Admin scope is controlled via MIND_MEM_SCOPE=admin env var.",
        }
    )


def request_has_admin_scope() -> bool:
    """True when the active call may perform an admin-scope write.

    Same resolution as :func:`enforce_capability_acl`: the ``deny``
    sentinel is never admin, otherwise the token scope, falling back to
    ``MIND_MEM_SCOPE`` (default ``user``). The documented
    ``MIND_MEM_ACL_DISABLED`` dev/test override counts as admin, as it does
    for every admin tool. Never logs: a body that only asks "may I persist?"
    must not change what it emits when the answer is no.
    """
    scope = _get_request_scope()
    if scope == "deny":
        return False
    if os.environ.get("MIND_MEM_ACL_DISABLED", "").lower() in ("1", "true", "yes"):
        return True
    return (scope or os.environ.get("MIND_MEM_SCOPE", "user")) == "admin"


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
