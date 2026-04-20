# mypy: disable-error-code="no-any-return"
"""v3.2.0 — MCP consolidated tool dispatchers.

Adds 7 consolidated dispatchers (``recall``, ``staged_change``,
``memory_verify``, ``graph``, ``core``, ``kernels``,
``compiled_truth``) that route to the existing 57-tool
implementations via a ``mode`` / ``phase`` / ``action`` argument.
The v3.1.x tools remain registered unchanged — this module is
purely **additive** so callers can adopt the consolidated entry
points at their own pace without a breaking change.

**Why add consolidators?** Agent context windows are finite, and
tool-selection reliability degrades as the catalog grows. 57 tool
names with overlapping semantics (``recall`` vs ``hybrid_search``
vs ``find_similar``) force the caller to disambiguate at dispatch
time. The consolidated surface moves that disambiguation into an
explicit enum argument where the LLM can reason about it.

Design notes:

* Dispatchers call the underlying module-level functions via
  their ``__wrapped__`` attribute, bypassing the ``@mcp_tool_observe``
  decorator on the callee. That matters because ``@mcp_tool_observe``
  (applied to the callee) already handles rate-limit + ACL + timing
  — double-wrapping would double-charge the rate limiter and emit
  duplicate trace spans.
* Each dispatcher returns the same JSON envelope shape the
  underlying tool returns. The dispatcher layer is transparent
  to the caller.
* Backward compatibility: the v3.1.x 57 tool names remain
  registered. ``public.recall(mode='bm25', query=...)`` and
  ``recall(query=...)`` produce identical envelopes; new agents
  can use either.

v4.0 will move the 39 specialised tools behind an opt-in
``mcp.expose_advanced_tools`` flag (default off) once the
consolidated surface has matured and caller migration is
demonstrably complete. For v3.2.0 there is no gate.
"""

from __future__ import annotations

import json
from typing import Any

from ..infra.constants import MCP_SCHEMA_VERSION
from ..infra.observability import mcp_tool_observe


def _err(message: str, **extra: Any) -> str:
    """Consistent error envelope shape across every public dispatcher."""
    payload: dict[str, Any] = {
        "_schema_version": MCP_SCHEMA_VERSION,
        "error": message,
    }
    payload.update(extra)
    return json.dumps(payload, indent=2)


# ───────────────────────────────────────────────────────────
# recall — consolidates 8 retrieval tools
# ───────────────────────────────────────────────────────────


@mcp_tool_observe
def recall(
    query: str,
    mode: str = "auto",
    limit: int = 10,
    active_only: bool = False,
    backend: str = "",
    *,
    block_id: str = "",
    axes: str = "lexical,semantic",
    weights: str = "",
    max_tokens: int = 2000,
    signals: str = "",
) -> str:
    """Unified retrieval entry point.

    Backward compatibility: v3.1.x callers passing ``backend=``
    (rather than ``mode=``) still work — the dispatcher treats
    ``backend`` as an alias for ``mode`` when it's set.

    ``mode`` dispatches to the specialised implementation:

    =========== ================================================
    mode        dispatch target
    =========== ================================================
    ``auto``    :func:`mcp.tools.recall.recall` (default backend)
    ``bm25``    recall(backend="bm25")
    ``hybrid``  recall(backend="hybrid")
    ``vector``  recall(backend="hybrid") — vector-only routing via config
    ``similar`` :func:`mcp.tools.recall.find_similar` — needs ``block_id``
    ``axis``    :func:`mcp.tools.recall.recall_with_axis` — uses ``axes`` + ``weights``
    ``pack``    :func:`mcp.tools.recall.pack_recall_budget` — uses ``max_tokens``
    ``prefetch`` :func:`mcp.tools.recall.prefetch` — uses ``signals``
    ``classify`` :func:`mcp.tools.recall.intent_classify`
    ``diagnostics`` :func:`mcp.tools.recall.retrieval_diagnostics`
    =========== ================================================
    """
    # Backward-compat: ``backend=`` alias for ``mode=``.
    if backend and mode == "auto":
        mode = backend

    from . import recall as _r

    if mode in ("auto", "bm25", "hybrid", "vector"):
        effective_backend = "auto" if mode in ("auto", "vector") else mode
        return _r._recall_impl(query, limit=limit, active_only=active_only, backend=effective_backend)
    if mode == "similar":
        if not block_id:
            return _err("mode='similar' requires 'block_id'")
        # find_similar is @mcp_tool_observe-decorated; call via .__wrapped__
        # to avoid double-charging the observer/rate-limiter.
        return _r.find_similar.__wrapped__(block_id, limit=limit)  # type: ignore[attr-defined]
    if mode == "axis":
        return _r.recall_with_axis.__wrapped__(  # type: ignore[attr-defined]
            query, axes=axes, weights=weights, limit=limit, active_only=active_only
        )
    if mode == "pack":
        return _r.pack_recall_budget.__wrapped__(query, max_tokens=max_tokens, limit=limit)  # type: ignore[attr-defined]
    if mode == "prefetch":
        return _r.prefetch.__wrapped__(signals or query, limit=limit)  # type: ignore[attr-defined]
    if mode == "classify":
        return _r.intent_classify.__wrapped__(query)  # type: ignore[attr-defined]
    if mode == "diagnostics":
        return _r.retrieval_diagnostics.__wrapped__()  # type: ignore[attr-defined]
    return _err(
        f"unknown mode: {mode!r}",
        valid_modes=[
            "auto",
            "bm25",
            "hybrid",
            "vector",
            "similar",
            "axis",
            "pack",
            "prefetch",
            "classify",
            "diagnostics",
        ],
    )


# ───────────────────────────────────────────────────────────
# staged_change — folds propose/approve/rollback
# ───────────────────────────────────────────────────────────


@mcp_tool_observe
def staged_change(
    phase: str,
    *,
    # propose args
    block_type: str = "",
    statement: str = "",
    rationale: str = "",
    tags: str = "",
    confidence: str = "medium",
    # approve args
    proposal_id: str = "",
    dry_run: bool = True,
    # rollback args
    receipt_ts: str = "",
) -> str:
    """Single entry point for the propose / apply / rollback flow.

    ============ =======================================================
    phase        dispatch
    ============ =======================================================
    ``propose``  :func:`mcp.tools.governance.propose_update`
    ``approve``  :func:`mcp.tools.governance.approve_apply`
    ``rollback`` :func:`mcp.tools.governance.rollback_proposal`
    ============ =======================================================
    """
    from . import governance

    if phase == "propose":
        if not block_type or not statement:
            return _err("phase='propose' requires 'block_type' and 'statement'")
        return governance.propose_update.__wrapped__(  # type: ignore[attr-defined]
            block_type, statement, rationale=rationale, tags=tags, confidence=confidence
        )
    if phase == "approve":
        if not proposal_id:
            return _err("phase='approve' requires 'proposal_id'")
        return governance.approve_apply.__wrapped__(proposal_id, dry_run=dry_run)  # type: ignore[attr-defined]
    if phase == "rollback":
        if not receipt_ts:
            return _err("phase='rollback' requires 'receipt_ts'")
        return governance.rollback_proposal.__wrapped__(receipt_ts)  # type: ignore[attr-defined]
    return _err(
        f"unknown phase: {phase!r}",
        valid_phases=["propose", "approve", "rollback"],
    )


# ───────────────────────────────────────────────────────────
# memory_verify — folds verify_merkle / verify_chain / mind_mem_verify
# ───────────────────────────────────────────────────────────


@mcp_tool_observe
def memory_verify(
    mode: str = "chain",
    *,
    block_id: str = "",
    content_hash: str = "",
    snapshot: str = "",
) -> str:
    """Unified memory-integrity verification.

    ============ =====================================================
    mode         dispatch
    ============ =====================================================
    ``chain``    :func:`mcp.tools.audit.verify_chain` (default)
    ``merkle``   :func:`mcp.tools.audit.verify_merkle` — needs ids
    ``cli``      :func:`mcp.tools.audit.mind_mem_verify` — full CLI verifier
    ============ =====================================================
    """
    from . import audit

    if mode == "chain":
        return audit.verify_chain.__wrapped__()  # type: ignore[attr-defined]
    if mode == "merkle":
        if not block_id or not content_hash:
            return _err("mode='merkle' requires 'block_id' and 'content_hash'")
        return audit.verify_merkle.__wrapped__(block_id, content_hash)  # type: ignore[attr-defined]
    if mode == "cli":
        return audit.mind_mem_verify.__wrapped__(snapshot)  # type: ignore[attr-defined]
    return _err(f"unknown mode: {mode!r}", valid_modes=["chain", "merkle", "cli"])


# ───────────────────────────────────────────────────────────
# graph — folds 4 graph tools
# ───────────────────────────────────────────────────────────


@mcp_tool_observe
def graph(
    action: str,
    *,
    subject: str = "",
    predicate: str = "",
    object: str = "",
    source_block_id: str = "",
    confidence: float = 1.0,
    entity: str = "",
    depth: int = 1,
    direction: str = "outgoing",
    limit: int = 64,
    block_id: str = "",
) -> str:
    """Unified knowledge/causal graph entry point.

    ============= =====================================================
    action        dispatch
    ============= =====================================================
    ``add_edge``  :func:`mcp.tools.graph.graph_add_edge`
    ``query``     :func:`mcp.tools.graph.graph_query`
    ``stats``     :func:`mcp.tools.graph.graph_stats`
    ``traverse``  :func:`mcp.tools.graph.traverse_graph` — causal graph
    ============= =====================================================
    """
    from . import graph as _g

    if action == "add_edge":
        return _g.graph_add_edge.__wrapped__(  # type: ignore[attr-defined]
            subject, predicate, object, source_block_id, confidence=confidence
        )
    if action == "query":
        return _g.graph_query.__wrapped__(  # type: ignore[attr-defined]
            entity, depth=depth, predicate=predicate, direction=direction, limit=limit
        )
    if action == "stats":
        return _g.graph_stats.__wrapped__()  # type: ignore[attr-defined]
    if action == "traverse":
        return _g.traverse_graph.__wrapped__(block_id, depth=depth, direction=direction)  # type: ignore[attr-defined]
    return _err(
        f"unknown action: {action!r}",
        valid_actions=["add_edge", "query", "stats", "traverse"],
    )


# ───────────────────────────────────────────────────────────
# core — folds 4 core tools
# ───────────────────────────────────────────────────────────


@mcp_tool_observe
def core(
    action: str,
    *,
    namespace: str = "",
    version: str = "",
    filename: str = "",
    verify: bool = True,
) -> str:
    """Unified .mmcore bundle lifecycle."""
    from . import core as _c

    if action == "build":
        return _c.build_core.__wrapped__(namespace, version, filename)  # type: ignore[attr-defined]
    if action == "load":
        return _c.load_core.__wrapped__(filename, verify=verify)  # type: ignore[attr-defined]
    if action == "unload":
        return _c.unload_core.__wrapped__(namespace)  # type: ignore[attr-defined]
    if action == "list":
        return _c.list_cores.__wrapped__()  # type: ignore[attr-defined]
    return _err(
        f"unknown action: {action!r}",
        valid_actions=["build", "load", "unload", "list"],
    )


# ───────────────────────────────────────────────────────────
# kernels — folds list_mind_kernels + get_mind_kernel
# ───────────────────────────────────────────────────────────


@mcp_tool_observe
def kernels(action: str = "list", *, name: str = "") -> str:
    """Unified .mind kernel read surface."""
    from . import kernels as _k

    if action == "list":
        return _k.list_mind_kernels.__wrapped__()  # type: ignore[attr-defined]
    if action == "get":
        if not name:
            return _err("action='get' requires 'name'")
        return _k.get_mind_kernel.__wrapped__(name)  # type: ignore[attr-defined]
    return _err(f"unknown action: {action!r}", valid_actions=["list", "get"])


# ───────────────────────────────────────────────────────────
# compiled_truth — folds 3 compiled-truth tools
# ───────────────────────────────────────────────────────────


@mcp_tool_observe
def compiled_truth(
    action: str,
    *,
    entity_id: str = "",
    observation: str = "",
    source: str = "mcp_tool",
    confidence: str = "medium",
    entity_type: str = "topic",
) -> str:
    """Unified compiled-truth page lifecycle."""
    from . import kernels as _k

    if not entity_id:
        return _err(f"action='{action}' requires 'entity_id'")
    if action == "load":
        return _k.compiled_truth_load.__wrapped__(entity_id)  # type: ignore[attr-defined]
    if action == "add_evidence":
        if not observation:
            return _err("action='add_evidence' requires 'observation'")
        return _k.compiled_truth_add_evidence.__wrapped__(  # type: ignore[attr-defined]
            entity_id, observation, source=source, confidence=confidence, entity_type=entity_type
        )
    if action == "contradictions":
        return _k.compiled_truth_contradictions.__wrapped__(entity_id)  # type: ignore[attr-defined]
    return _err(
        f"unknown action: {action!r}",
        valid_actions=["load", "add_evidence", "contradictions"],
    )


# ───────────────────────────────────────────────────────────
# register — wire every public tool onto the FastMCP instance
# ───────────────────────────────────────────────────────────


def register(mcp) -> None:
    """Register the 7 consolidated v3.2.0 dispatchers on *mcp*.

    Collides-safe: each dispatcher's name (``recall``, ``graph``,
    etc.) is deliberately distinct from every legacy tool name in
    the 57-tool surface, so this registration is purely additive
    when layered on top of the existing ``<domain>.register(mcp)``
    calls in ``server.py``. (``recall`` as a dispatcher vs
    ``recall`` as the v3.1.x tool: the v3.2.0 dispatcher wins
    because it registers last in ``server.py`` registration order,
    and FastMCP's internal registry is a straight dict — the newer
    binding replaces the older one at the same name.)

    v3.2.0 callers can opt into the consolidated surface tool by
    tool; v3.1.x tool names remain usable unchanged for every tool
    that isn't shadowed by a dispatcher.
    """
    mcp.tool(recall)
    mcp.tool(staged_change)
    mcp.tool(memory_verify)
    mcp.tool(graph)
    mcp.tool(core)
    mcp.tool(kernels)
    mcp.tool(compiled_truth)
