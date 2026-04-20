#!/usr/bin/env python3
"""Mind-Mem MCP Server — persistent memory for paranoid/safety-first coding agents.

Exposes Mind-Mem as a Model Context Protocol server, making structured memory
accessible to any MCP-compatible client (Claude Code, Claude Desktop, Cursor,
Windsurf, OpenClaw).

Resources (read-only):
    mind-mem://decisions         — All active decisions
    mind-mem://tasks             — All tasks
    mind-mem://entities/{type}   — Entity files (projects, people, tools, incidents)
    mind-mem://signals           — Auto-captured signals
    mind-mem://contradictions    — Detected contradictions
    mind-mem://health            — Workspace health summary
    mind-mem://recall/{query}    — BM25 recall search
    mind-mem://ledger            — Shared fact ledger (multi-agent)

Tools (32):
    recall               — Search memory (auto/bm25/hybrid backend)
    propose_update       — Propose a new decision/task (writes to SIGNALS.md, never source of truth)
    approve_apply        — Apply a staged proposal (dry-run by default)
    rollback_proposal    — Rollback an applied proposal by receipt timestamp
    scan                 — Run integrity scan
    list_contradictions  — List detected contradictions with resolution status
    hybrid_search        — (Deprecated) Alias for recall(backend="hybrid")
    find_similar         — Find blocks similar to a given block
    intent_classify      — Show routing strategy for a query
    index_stats          — Block counts, index status, kernel info
    retrieval_diagnostics — Pipeline rejection rates, intent histogram, hard negatives
    reindex              — Trigger FTS index rebuild
    memory_evolution     — A-MEM metadata for a block
    list_mind_kernels    — List available .mind kernel configs
    get_mind_kernel      — Read a specific .mind kernel
    category_summary     — Category summaries for a topic
    prefetch             — Pre-assemble context from conversation signals
    delete_memory_item   — Delete a block by ID (admin)
    export_memory        — Export all blocks as JSONL (user)
    calibration_feedback — Record retrieval quality feedback for calibration loop
    calibration_stats    — Per-block calibration scores, per-query-type accuracy
    verify_chain         — Verify SHA3-512 governance hash chain integrity
    list_evidence        — List governance evidence objects with filters
    get_block            — Direct block lookup by ID (returns full block content)
    memory_health        — Deep health dashboard (stale blocks, orphans, drift, coverage)
    traverse_graph       — Navigate causal graph from a block (deps + dependents)
    compact              — Run compaction: archive old blocks, clean snapshots/signals
    stale_blocks         — List blocks needing review due to upstream changes
    dream_cycle          — Run autonomous memory enrichment with optional auto-repair
    compiled_truth_load  — Load a compiled truth page for an entity
    compiled_truth_add_evidence — Add evidence and auto-recompile truth page
    compiled_truth_contradictions — Detect contradictions in a truth page

Transport:
    stdio (default, for Claude Code / Claude Desktop)
    http  (for remote / multi-client)

Usage:
    # stdio (Claude Code / Claude Desktop)
    python3 mcp_server.py

    # http
    python3 mcp_server.py --transport http --port 8765

    # http with token auth
    MIND_MEM_TOKEN=secret python3 mcp_server.py --transport http --port 8765

    # with custom workspace
    MIND_MEM_WORKSPACE=/path/to/workspace python3 mcp_server.py

Claude Desktop config (~/.claude/claude_desktop_config.json):
    {
      "mcpServers": {
        "mind-mem": {
          "command": "python3",
          "args": ["/path/to/mind-mem/mcp_server.py"],
          "env": {"MIND_MEM_WORKSPACE": "/path/to/workspace"}
        }
      }
    }
"""

from __future__ import annotations

import json
import os
import re as _re_mod
import sqlite3
import sys
import tempfile
import time
from typing import Any

# Allow running `python3 mcp_server.py` directly from a source checkout.
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_REPO_ROOT, "src")
if os.path.isdir(_SRC_DIR) and _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# mind-mem imports (package mapped to scripts/ via pyproject.toml)
from mind_mem.block_parser import BlockCorruptedError, get_active, parse_file  # noqa: E402, F401
from mind_mem.corpus_registry import CORPUS_DIRS  # noqa: E402
from mind_mem.mind_filelock import FileLock  # noqa: E402
from fastmcp import FastMCP  # noqa: E402
from mind_mem.mind_ffi import (  # noqa: E402
    get_mind_dir,
    is_available as mind_kernel_available,
    is_protected as mind_kernel_protected,
    list_kernels as ffi_list_kernels,
)
from mind_mem.observability import get_logger, metrics  # noqa: E402
from mind_mem.recall import recall as recall_engine  # noqa: E402
from mind_mem.retrieval_graph import retrieval_diagnostics as _retrieval_diag  # noqa: E402
from mind_mem.sqlite_index import (  # noqa: E402
    _db_path as fts_db_path,
    query_index as fts_query,
)

_log = get_logger("mcp_server")

# v3.2.0 §1.2 PR-1: MCP_SCHEMA_VERSION re-exported from mcp.infra.constants
# so every infra submodule and the remaining mcp_server body share a single
# source of truth (avoids the two-definition drift risk).
from mind_mem.mcp.infra.constants import MCP_SCHEMA_VERSION  # noqa: E402,F401


# ---------------------------------------------------------------------------
# ACL — per-tool scope enforcement (#20)
# ---------------------------------------------------------------------------
#
# v3.2.0 §1.2 PR-1: ACL helpers moved to mind_mem.mcp.infra.acl. Re-exported
# here so every existing call site in this module + every test that patches
# ``mcp_server.check_tool_acl`` / ``mcp_server.ADMIN_TOOLS`` keeps working.
#
# Absolute import (not relative) because the top-level developer-checkout
# shim ``/home/n/mind-mem/mcp_server.py`` runs this file via
# ``exec(compile(...))`` with no parent package, and the test harness loads
# it via ``spec_from_file_location`` which also strips the package.
# ``mind_mem.mcp.infra.acl`` resolves in both paths because the shim
# inserts ``src/`` onto ``sys.path`` before the exec.
from mind_mem.mcp.infra.acl import (  # noqa: E402,F401 — public re-export shim
    ADMIN_TOOLS,
    USER_TOOLS,
    _ADMIN_SCOPES,
    _get_request_scope,
    check_tool_acl,
)


# v3.2.0 §1.2 PR-1: HTTP auth helpers moved to
# mind_mem.mcp.infra.http_auth. Re-exported here so every tool body and
# every test that references ``server.verify_token`` /
# ``server._build_http_auth_tokens`` / ``server._check_token`` keeps
# working.
from mind_mem.mcp.infra.http_auth import (  # noqa: E402,F401 — public re-export shim
    _build_http_auth_tokens,
    _check_token,
    verify_token,
)


# ---------------------------------------------------------------------------
# Rate limiter — sliding window (#21)
# ---------------------------------------------------------------------------
#
# v3.2.0 §1.2 PR-1: rate-limit primitives moved to
# mind_mem.mcp.infra.rate_limit. Re-exported here so every existing call
# site (``mcp_tool_observe``, ``tests/test_mcp_integration.py``) keeps
# working. ``_get_limits`` / ``_DEFAULT_LIMITS`` stay here for now and
# move alongside the rest of config handling in a later step of PR-1;
# ``_init_rate_limiter`` in the new module late-imports ``_get_limits``
# from this module to avoid the import cycle.
from mind_mem.mcp.infra.rate_limit import (  # noqa: E402,F401 — public re-export shim
    _RATE_LIMITER_MAX,
    SlidingWindowRateLimiter,
    _get_client_id,
    _get_client_rate_limiter,
    _init_rate_limiter,
    _rate_limiters,
    _rate_limiters_lock,
)


# ---------------------------------------------------------------------------
# Configurable limits (#37) — loaded from mind-mem.json "limits" section
# ---------------------------------------------------------------------------
#
# v3.2.0 §1.2 PR-1: config helpers moved to mind_mem.mcp.infra.config.
# Re-exported here so every existing call site + every test that reads
# ``server._get_limits`` / ``server._DEFAULT_LIMITS`` keeps working.
from mind_mem.mcp.infra.config import (  # noqa: E402,F401 — public re-export shim
    _DEFAULT_LIMITS,
    QUERY_TIMEOUT_SECONDS,
    _get_limits,
    _load_config,
    _load_extra_categories,
)


# v3.2.0 §1.2 PR-1: observability helpers moved to
# mind_mem.mcp.infra.observability. Re-exported here so every
# ``@mcp_tool_observe`` decorator application + every test that patches
# ``server._sqlite_busy_error`` keeps working.
from mind_mem.mcp.infra.observability import (  # noqa: E402,F401 — public re-export shim
    _is_db_locked,
    _sqlite_busy_error,
    mcp_tool_observe,
)


# ---------------------------------------------------------------------------
# Server setup
# ---------------------------------------------------------------------------

mcp = FastMCP(
    name="mind-mem",
    instructions=(
        "Mind-Mem: persistent, auditable, contradiction-safe memory for coding agents. "
        "Use recall to search memory. Use propose_update to suggest changes (never writes "
        "directly to source of truth). All proposals go through human review."
    ),
)


# v3.2.0 §1.2 PR-1: workspace helpers moved to mind_mem.mcp.infra.workspace.
# Re-exported here so every existing call site inside this module + every
# test that patches `mcp_server._workspace` keeps working.
#
# Absolute import (not relative) because the top-level developer-checkout
# shim ``/home/n/mind-mem/mcp_server.py`` runs this file via
# ``exec(compile(...))`` with no parent package, and the test harness
# loads it via ``spec_from_file_location`` which also strips the package.
# ``mind_mem.mcp.infra.workspace`` resolves in both paths because the
# shim inserts ``src/`` onto ``sys.path`` before the exec.
from mind_mem.mcp.infra.workspace import (  # noqa: E402,F401 — public re-export shim
    _check_workspace,
    _read_file,
    _validate_path,
    _workspace,
)


# ---------------------------------------------------------------------------
# Resources (read-only)
# ---------------------------------------------------------------------------
#
# v3.2.0 §1.2 PR-2: @mcp.resource bodies moved to mind_mem.mcp.resources.
# The module defines every resource function at module level and exposes
# ``register(mcp)`` which wires them onto the FastMCP instance. We
# re-export the function names here so ``server.get_decisions`` etc.
# keep resolving for tests + callers.
from mind_mem.mcp import resources as _mcp_resources  # noqa: E402,F401
from mind_mem.mcp.resources import (  # noqa: E402,F401 — public re-export shim
    _blocks_to_json,
    get_contradictions,
    get_decisions,
    get_entities,
    get_health,
    get_ledger,
    get_recall,
    get_signals,
    get_tasks,
)

_mcp_resources.register(mcp)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


def _recall_impl(query: str, limit: int = 10, active_only: bool = False, backend: str = "auto") -> str:
    """Core recall implementation shared by recall() and hybrid_search()."""
    ws = _workspace()
    ws_err = _check_workspace(ws)
    if ws_err:
        return ws_err
    limits = _get_limits(ws)
    limit = max(1, min(limit, limits["max_recall_results"]))
    timeout_seconds = limits.get("query_timeout_seconds", QUERY_TIMEOUT_SECONDS)
    recall_start = time.monotonic()
    if backend not in ("auto", "bm25", "hybrid"):
        backend = "auto"
    warnings: list[str] = []
    config_warnings: list[str] = []
    used_backend = "scan"
    results: list = []

    # --- Hybrid path (backend="hybrid" or "auto") ---
    if backend in ("hybrid", "auto"):
        try:
            from mind_mem.hybrid_recall import HybridBackend, validate_recall_config

            config = _load_config(ws)
            recall_cfg = config.get("recall", {})
            if not isinstance(recall_cfg, dict):
                recall_cfg = {}
            schema_errors = validate_recall_config(recall_cfg)
            if schema_errors:
                config_warnings = schema_errors
                _log.warning("recall_config_errors", errors=schema_errors)
            hb = HybridBackend.from_config(config)
            results = hb.search(query, ws, limit=limit, active_only=active_only)
            used_backend = "hybrid"
        except ImportError:
            if backend == "hybrid":
                warnings.append("Hybrid backend unavailable — falling back to BM25.")
            # Fall through to BM25 path
        except sqlite3.OperationalError as exc:
            if _is_db_locked(exc):
                return _sqlite_busy_error()
            raise
        except (OSError, ValueError, KeyError) as exc:
            _log.warning("recall_hybrid_failed", query=query, error=str(exc))
            if backend == "hybrid":
                warnings.append(f"Hybrid search failed — falling back to BM25: {exc}")

    # --- BM25 path (backend="bm25", or fallback from auto/hybrid) ---
    if used_backend != "hybrid":
        try:
            if os.path.isfile(fts_db_path(ws)):
                results = fts_query(ws, query, limit=limit, active_only=active_only)
                used_backend = "sqlite"
            else:
                results = recall_engine(ws, query, limit=limit, active_only=active_only)
                used_backend = "scan"
                warnings.append("FTS5 index not found — using full scan. Run 'reindex' tool for faster queries.")
        except sqlite3.OperationalError as exc:
            if _is_db_locked(exc):
                return _sqlite_busy_error()
            raise

    # Query timeout enforcement (#476)
    recall_elapsed = time.monotonic() - recall_start
    if recall_elapsed > timeout_seconds:
        _log.warning(
            "query_timeout_exceeded",
            elapsed=round(recall_elapsed, 2),
            limit=timeout_seconds,
            backend=used_backend,
        )
        warnings.append(f"Query exceeded timeout ({round(recall_elapsed, 1)}s > {timeout_seconds}s). Results may be incomplete.")

    # Generate query_id for calibration feedback loop
    try:
        from mind_mem.calibration import make_query_id

        query_id = make_query_id(query)
    except ImportError:
        query_id = ""

    metrics.inc("mcp_recall_queries")
    _log.info("mcp_recall", query=query, backend=used_backend, results=len(results))
    envelope: dict = {
        "_schema_version": MCP_SCHEMA_VERSION,
        "backend": used_backend,
        "query": query,
        "query_id": query_id,
        "count": len(results),
        "results": results,
    }
    if warnings:
        envelope["warnings"] = warnings
    if config_warnings:
        envelope["config_warnings"] = config_warnings
    if not results:
        envelope["message"] = "No matching blocks found. Try broader terms or check workspace."
    return json.dumps(envelope, indent=2, default=str)


@mcp.tool
@mcp_tool_observe
def recall(
    query: str,
    limit: int = 10,
    active_only: bool = False,
    backend: str = "auto",
) -> str:
    """Search across all memory files with ranked retrieval.

    Supports BM25 (FTS5), hybrid BM25+Vector RRF fusion, or auto mode
    which tries hybrid first and falls back to BM25.

    Args:
        query: Search query (supports stemming and domain-aware expansion).
        limit: Maximum number of results (default: 10).
        active_only: Only return blocks with Status: active.
        backend: Retrieval backend — "auto" (default, hybrid→BM25 fallback),
                 "bm25" (keyword only), or "hybrid" (BM25+Vector RRF fusion).

    Returns:
        JSON array of ranked results with scores, IDs, and matched content.
    """
    return _recall_impl(query, limit=limit, active_only=active_only, backend=backend)


# v3.2.0 §1.2 PR-3: audit tools moved to mcp.tools.audit (see end of file
# for mind_mem_verify, verify_chain, list_evidence — also moved).
from mind_mem.mcp.tools import audit as _tools_audit  # noqa: E402,F401
from mind_mem.mcp.tools.audit import (  # noqa: E402,F401 — public re-export shim
    list_evidence,
    mind_mem_verify,
    verify_chain,
    verify_merkle,
)

_tools_audit.register(mcp)


# v3.2.0 §1.2 PR-3: workspace-path helpers + lazy singletons moved to
# mind_mem.mcp.tools._helpers so every tool module shares a single
# definition. Re-exported here because later tools in this file still
# reference them at call time (pre-extraction).
from mind_mem.mcp.tools._helpers import (  # noqa: E402,F401 — public re-export shim
    _change_stream,
    _core_dir,
    _core_registry,
    _kg_path,
    _ontology_registry,
    _signal_store_path,
)


# v3.2.0 §1.2 PR-3: core tools moved to mcp.tools.core
from mind_mem.mcp.tools import core as _tools_core  # noqa: E402,F401
from mind_mem.mcp.tools.core import (  # noqa: E402,F401 — public re-export shim
    build_core,
    list_cores,
    load_core,
    unload_core,
)

_tools_core.register(mcp)


# v3.2.0 §1.2 PR-3: consolidation tools moved to mcp.tools.consolidation
from mind_mem.mcp.tools import consolidation as _tools_consolidation  # noqa: E402,F401
from mind_mem.mcp.tools.consolidation import (  # noqa: E402,F401 — public re-export shim
    dream_cycle,
    plan_consolidation,
    project_profile,
    propagate_staleness,
)

_tools_consolidation.register(mcp)


# v3.2.0 §1.2 PR-3: ontology tools moved to mcp.tools.ontology
from mind_mem.mcp.tools import ontology as _tools_ontology  # noqa: E402,F401
from mind_mem.mcp.tools.ontology import (  # noqa: E402,F401 — public re-export shim
    ontology_load,
    ontology_validate,
)

_tools_ontology.register(mcp)


# v3.2.0 §1.2 PR-3: agent tools moved to mcp.tools.agent
from mind_mem.mcp.tools import agent as _tools_agent  # noqa: E402,F401
from mind_mem.mcp.tools.agent import (  # noqa: E402,F401 — public re-export shim
    _vault_allowlist,
    _vault_root_allowed,
    agent_inject,
    stream_status,
    vault_scan,
    vault_sync,
)

_tools_agent.register(mcp)


@mcp.tool
@mcp_tool_observe
def pack_recall_budget(query: str, max_tokens: int = 2000, limit: int = 20) -> str:
    """Run a recall, then pack the result list under a token budget.

    Returns the subset of results that fits and the tail that was
    dropped, plus the reserved token budget for graph / provenance
    metadata. Use when wiring mind-mem into an agent whose prompt
    already approaches its model's context window.
    """
    from .cognitive_forget import pack_to_budget

    ws = _workspace()
    ws_err = _check_workspace(ws)
    if ws_err:
        return ws_err

    if not isinstance(query, str) or not query.strip():
        return json.dumps({"error": "query must be a non-empty string"})
    if max_tokens <= 0 or max_tokens > 1_000_000:
        return json.dumps({"error": "max_tokens must be in [1, 1_000_000]"})
    if limit < 1 or limit > 500:
        return json.dumps({"error": "limit must be in [1, 500]"})

    raw = json.loads(_recall_impl(query, limit=limit))
    # _recall_impl returns an envelope or a list depending on configuration;
    # normalise to a flat list of result dicts.
    if isinstance(raw, dict):
        results = raw.get("results", []) or []
    elif isinstance(raw, list):
        results = raw
    else:
        results = []

    try:
        packed = pack_to_budget(results, max_tokens=int(max_tokens))
    except ValueError as exc:
        return json.dumps({"error": str(exc)})

    return json.dumps(
        {
            "query": query,
            "included": packed.included,
            "dropped": packed.dropped,
            **packed.as_dict(),
            "_schema_version": "1.0",
        },
        indent=2,
        default=str,
    )


# v3.2.0 §1.2 PR-3: graph tools moved to mcp.tools.graph (traverse_graph
# moves in the same module — see below where it used to live).
from mind_mem.mcp.tools import graph as _tools_graph  # noqa: E402,F401
from mind_mem.mcp.tools.graph import (  # noqa: E402,F401 — public re-export shim
    graph_add_edge,
    graph_query,
    graph_stats,
    traverse_graph,
)

_tools_graph.register(mcp)


# v3.2.0 §1.2 PR-3: signal tools moved to mcp.tools.signal
from mind_mem.mcp.tools import signal as _tools_signal  # noqa: E402,F401
from mind_mem.mcp.tools.signal import (  # noqa: E402,F401 — public re-export shim
    observe_signal,
    signal_stats,
)

_tools_signal.register(mcp)


@mcp.tool
@mcp_tool_observe
def recall_with_axis(
    query: str,
    axes: str = "lexical,semantic",
    weights: str = "",
    limit: int = 10,
    active_only: bool = False,
    adversarial: bool = False,
    allow_rotation: bool = True,
) -> str:
    """Axis-aware recall under the Observer-Dependent Cognition model.

    Declares the observation basis explicitly. Each result carries per-axis
    confidence scores, and the system rotates to orthogonal axes when
    initial confidence is low.

    Args:
        query: Search query.
        axes: Comma-separated axes (``lexical,semantic,temporal,entity_graph,
              contradiction,adversarial``). Defaults to ``lexical,semantic``
              (matches the v1.x behaviour).
        weights: Optional ``axis=weight,axis=weight`` override. Non-zero
              weights implicitly activate their axis; ``axes`` acts as a
              safety allowlist when both are supplied.
        limit: Maximum results (default 10).
        active_only: Exclude superseded blocks (contradiction / adversarial
              axes ignore this flag so dissent stays visible).
        adversarial: When true, run each active axis's adversarial pair in
              parallel and fuse the results.
        allow_rotation: When true (default), rotate to orthogonal axes if
              top-confidence falls below the rotation threshold.

    Returns:
        JSON envelope with ``results`` (each tagged with per-axis metadata),
        ``weights``, ``rotated``, ``diversity``, and ``attempts``.
    """
    from .axis_recall import recall_with_axis as _axis_recall
    from .observation_axis import AxisWeights, ObservationAxis

    ws = _workspace()
    ws_err = _check_workspace(ws)
    if ws_err:
        return ws_err

    # Bound the inputs up front so a misbehaving caller can't burn
    # unbounded CPU on enum lookups or a huge result fetch.
    _MAX_ARG_LEN = 1024  # axes / weights string length
    _MAX_TOKENS = 16  # per enum, number of axes / weight pairs
    _MAX_LIMIT = 500

    if len(axes) > _MAX_ARG_LEN or len(weights) > _MAX_ARG_LEN:
        return json.dumps({"error": f"axes/weights args must be ≤{_MAX_ARG_LEN} chars"})
    if limit < 1 or limit > _MAX_LIMIT:
        return json.dumps({"error": f"limit must be in [1, {_MAX_LIMIT}]"})

    # Parse axis list
    axis_tokens = [tok.strip() for tok in axes.split(",") if tok.strip()]
    if not axis_tokens:
        return json.dumps({"error": "axes must include at least one axis name"})
    if len(axis_tokens) > _MAX_TOKENS:
        return json.dumps({"error": f"axes list must contain ≤{_MAX_TOKENS} entries"})
    try:
        allowed = {ObservationAxis.from_str(tok) for tok in axis_tokens}
    except ValueError as exc:
        return json.dumps({"error": str(exc)})

    # Parse optional explicit weights
    if weights.strip():
        weight_entries = [kv for kv in weights.split(",") if kv.strip()]
        if len(weight_entries) > _MAX_TOKENS:
            return json.dumps({"error": f"weights list must contain ≤{_MAX_TOKENS} entries"})
        weight_map: dict[str, float] = {}
        for kv in weight_entries:
            kv = kv.strip()
            if "=" not in kv:
                return json.dumps({"error": f"weight entry must be axis=value, got {kv!r}"})
            axis_name, value = kv.split("=", 1)
            try:
                weight_map[axis_name.strip()] = float(value.strip())
            except ValueError:
                return json.dumps({"error": f"weight for {axis_name!r} is not numeric: {value!r}"})
        try:
            parsed_weights = AxisWeights.from_mapping(weight_map)
        except ValueError as exc:
            return json.dumps({"error": str(exc)})
        # Zero out any axis not in the allowlist so the ``axes`` arg wins.
        effective: dict[str, float] = {}
        for axis in allowed:
            effective[axis.value] = parsed_weights.as_dict().get(axis.value, 0.0)
        weight_obj = AxisWeights.from_mapping(effective)
    else:
        weight_obj = AxisWeights.uniform(allowed)

    try:
        result = _axis_recall(
            ws,
            query,
            weights=weight_obj,
            limit=limit,
            active_only=active_only,
            adversarial=adversarial,
            allow_rotation=allow_rotation,
        )
    except ValueError as exc:
        return json.dumps({"error": str(exc)})

    envelope = {
        "query": query,
        "results": result["results"],
        "weights": result["weights"],
        "rotated": result["rotated"],
        "diversity": result["diversity"],
        "attempts": result["attempts"],
        "_schema_version": "1.0",
    }
    return json.dumps(envelope, indent=2, default=str)


# v3.2.0 §1.2 PR-3: governance tools moved to mcp.tools.governance
# (approve_apply, rollback_proposal, memory_evolution are also part of this
# set — re-exports for all six land below once the later blocks are deleted.)
from mind_mem.mcp.tools import governance as _tools_governance  # noqa: E402,F401
from mind_mem.mcp.tools.governance import (  # noqa: E402,F401 — public re-export shim
    approve_apply,
    list_contradictions,
    memory_evolution,
    propose_update,
    rollback_proposal,
    scan,
)

_tools_governance.register(mcp)


# v3.2.0 §1.2 PR-3: encryption tools moved to
# mind_mem.mcp.tools.encryption. Re-exported here so
# ``server.encrypt_file(...)`` still routes through the observe
# wrapper and tests that reference ``server._safe_vault_path`` /
# ``server._encryption_passphrase`` still resolve.
from mind_mem.mcp.tools import encryption as _tools_encryption  # noqa: E402,F401
from mind_mem.mcp.tools.encryption import (  # noqa: E402,F401 — public re-export shim
    _encryption_passphrase,
    _safe_vault_path,
    decrypt_file,
    encrypt_file,
)

_tools_encryption.register(mcp)


# v3.2.0 §1.2 PR-3: benchmark tools moved to mcp.tools.benchmark
from mind_mem.mcp.tools import benchmark as _tools_benchmark  # noqa: E402,F401
from mind_mem.mcp.tools.benchmark import (  # noqa: E402,F401 — public re-export shim
    category_summary,
    governance_health_bench,
)

_tools_benchmark.register(mcp)


# ---------------------------------------------------------------------------
# New Tools (7-12) — Hybrid, similarity, intent, stats, reindex, evolution
# ---------------------------------------------------------------------------


@mcp.tool
@mcp_tool_observe
def hybrid_search(query: str, limit: int = 10, active_only: bool = False) -> str:
    """Hybrid BM25+Vector recall with RRF fusion.

    .. deprecated::
        Use ``recall(backend="hybrid")`` instead. This tool will be removed in a
        future release.

    Args:
        query: Search query.
        limit: Maximum results (default: 10).
        active_only: Only return active blocks.

    Returns:
        JSON array of ranked results from fused retrieval.
    """
    import warnings

    warnings.warn(
        "hybrid_search is deprecated. Use recall(backend='hybrid') instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    raw = _recall_impl(query, limit=limit, active_only=active_only, backend="hybrid")
    try:
        envelope = json.loads(raw)
        envelope["_deprecation_notice"] = "hybrid_search is deprecated. Use recall with backend='hybrid' instead."
        return json.dumps(envelope, indent=2)
    except (json.JSONDecodeError, TypeError):
        return raw


@mcp.tool
@mcp_tool_observe
def find_similar(block_id: str, limit: int = 5) -> str:
    """Find blocks similar to a given block using vector similarity.

    Requires vector backend (sentence-transformers). Falls back gracefully.

    Args:
        block_id: Source block ID to find similar blocks for.
        limit: Maximum similar blocks to return (default: 5).

    Returns:
        JSON array of similar blocks with similarity scores.
    """
    if not _re_mod.match(r"^[A-Z]+-[a-zA-Z0-9_.-]+$", block_id):
        return json.dumps({"error": f"Invalid block_id format: {block_id}"})
    ws = _workspace()
    limits = _get_limits(ws)
    limit = max(1, min(limit, limits["max_similar_results"]))
    try:
        from mind_mem.block_metadata import BlockMetadataManager

        db_path = os.path.join(ws, "memory", "block_meta.db")
        mgr = BlockMetadataManager(db_path)
        co_blocks = mgr.get_co_occurring_blocks(block_id, limit=limit)
        metrics.inc("mcp_find_similar_queries")
        return json.dumps(
            {
                "_schema_version": MCP_SCHEMA_VERSION,
                "source": block_id,
                "similar": co_blocks,
                "method": "co-occurrence",
            },
            indent=2,
        )
    except ImportError:
        return json.dumps(
            {
                "_schema_version": MCP_SCHEMA_VERSION,
                "error": "find_similar requires block_metadata module",
                "block_id": block_id,
            },
            indent=2,
        )
    except sqlite3.OperationalError as exc:
        if _is_db_locked(exc):
            return _sqlite_busy_error()
        raise
    except (OSError, ValueError, KeyError) as exc:
        _log.warning("find_similar_failed", block_id=block_id, error=str(exc))
        return json.dumps(
            {
                "_schema_version": MCP_SCHEMA_VERSION,
                "error": "Failed to find similar blocks. The co-occurrence index may not be initialized.",
                "block_id": block_id,
            },
            indent=2,
        )


@mcp.tool
@mcp_tool_observe
def intent_classify(query: str) -> str:
    """Show the routing strategy for a query.

    Classifies query intent into one of 9 types (WHY, WHEN, ENTITY, WHAT,
    HOW, LIST, VERIFY, COMPARE, TRACE) and returns retrieval parameters.

    Args:
        query: The query to classify.

    Returns:
        JSON with intent type, confidence, sub-intents, and parameters.
    """
    try:
        from mind_mem.intent_router import IntentRouter

        router = IntentRouter()
        result = router.classify(query)
        metrics.inc("mcp_intent_classify")
        return json.dumps(
            {
                "_schema_version": MCP_SCHEMA_VERSION,
                "query": query,
                "intent": result.intent,
                "confidence": result.confidence,
                "sub_intents": result.sub_intents,
                "params": result.params,
            },
            indent=2,
        )
    except ImportError:
        return json.dumps(
            {
                "_schema_version": MCP_SCHEMA_VERSION,
                "error": "intent_router module not available",
                "query": query,
            },
            indent=2,
        )
    except (ValueError, KeyError, AttributeError) as exc:
        _log.warning("intent_classify_failed", query=query, error=str(exc))
        return json.dumps(
            {
                "_schema_version": MCP_SCHEMA_VERSION,
                "error": "Intent classification failed",
                "query": query,
            },
            indent=2,
        )


@mcp.tool
@mcp_tool_observe
def index_stats() -> str:
    """Block counts, index staleness, vector coverage, and MIND kernel status.

    Returns:
        JSON with workspace statistics.
    """
    ws = _workspace()
    stats: dict = {"_schema_version": MCP_SCHEMA_VERSION}

    # Use FTS index for block counts when available (O(1) vs O(N) file parsing)
    db = fts_db_path(ws)
    fts_exists = os.path.isfile(db) if db else False
    stats["fts_index_exists"] = fts_exists

    if fts_exists:
        try:
            from mind_mem.sqlite_index import index_status as fts_status

            fts_info = fts_status(ws)
            stats["total_blocks"] = fts_info.get("blocks", 0)
            stats["last_build"] = fts_info.get("last_build")
            stats["stale_files"] = fts_info.get("stale_files", 0)
            stats["db_size_bytes"] = fts_info.get("db_size_bytes", 0)
        except sqlite3.OperationalError as exc:
            if _is_db_locked(exc):
                return _sqlite_busy_error()
            raise
        except (OSError, ValueError, KeyError) as e:
            _log.debug("fts_status_failed", error=str(e))
            fts_exists = False  # fall through to file scan

    if not fts_exists:
        # Fallback: count blocks by parsing files (O(N))
        for kind in CORPUS_DIRS:
            d = os.path.join(ws, kind)
            if os.path.isdir(d):
                count = 0
                for fn in os.listdir(d):
                    if fn.endswith(".md"):
                        try:
                            blocks = parse_file(os.path.join(d, fn))
                            count += len(blocks)
                        except (OSError, ValueError) as e:
                            _log.debug("index_stats_parse_failed", file=fn, error=str(e))
                stats[f"{kind}_blocks"] = count

    # MIND kernel status
    mind_dir = get_mind_dir(ws)
    kernels = ffi_list_kernels(mind_dir)
    stats["mind_kernels"] = kernels
    stats["mind_kernel_compiled"] = mind_kernel_available()
    stats["mind_kernel_protected"] = mind_kernel_protected()

    # v2.0.0b1 — LLM prefix cache + speculative prefetch stats.
    # Narrow except: only swallow import failures (module not present) and
    # attribute failures (symbol renamed). Anything else propagates so
    # real regressions surface in the MCP error envelope.
    try:
        from mind_mem.prefix_cache import all_stats as _prefix_all_stats

        stats["prefix_caches"] = [s.as_dict() for s in _prefix_all_stats()]
    except (ImportError, AttributeError) as exc:
        _log.debug("prefix_cache_stats_unavailable", error=str(exc))
        stats["prefix_caches"] = []

    try:
        from mind_mem.speculative_prefetch import get_default_predictor

        stats["speculative_prefetch"] = get_default_predictor().stats().as_dict()
    except (ImportError, AttributeError) as exc:
        _log.debug("speculative_prefetch_stats_unavailable", error=str(exc))
        stats["speculative_prefetch"] = {}

    # v2.1.0 — interaction signal store stats
    try:
        from mind_mem.interaction_signals import SignalStore

        sig_store = SignalStore(_signal_store_path(ws))
        stats["interaction_signals"] = sig_store.stats().as_dict()
    except (ImportError, AttributeError, OSError) as exc:
        _log.debug("interaction_signal_stats_unavailable", error=str(exc))
        stats["interaction_signals"] = {}

    metrics.inc("mcp_index_stats")
    _log.info("mcp_index_stats", stats=stats)
    return json.dumps(stats, indent=2)


@mcp.tool
@mcp_tool_observe
def retrieval_diagnostics(last_n: int = 50, max_age_days: int = 7) -> str:
    """Pipeline diagnostics: per-stage rejection rates, intent distribution, and hard negative summary.

    Surfaces the internal veto histogram — candidates rejected at each gate
    (BM25, dedup, rerank, knee cutoff) plus confidence distributions.

    Args:
        last_n: Number of recent queries to analyze (default 50).
        max_age_days: Only consider queries within this window (default 7).

    Returns:
        JSON with stage_stats, rejection_rates, intent_distribution,
        score_distribution, and hard_negatives summary.
    """
    ws = _workspace()
    try:
        result = _retrieval_diag(ws, last_n=last_n, max_age_days=max_age_days)
    except sqlite3.OperationalError as exc:
        if _is_db_locked(exc):
            return _sqlite_busy_error()
        raise
    result["_schema_version"] = MCP_SCHEMA_VERSION
    metrics.inc("mcp_retrieval_diagnostics")
    return json.dumps(result, indent=2)


@mcp.tool
@mcp_tool_observe
def reindex(include_vectors: bool = False) -> str:
    """Trigger FTS index rebuild, optionally with vector indexing.

    Args:
        include_vectors: Also rebuild vector index (requires sentence-transformers).

    Returns:
        JSON with reindex results.
    """
    ws = _workspace()
    ws_err = _check_workspace(ws)
    if ws_err:
        return ws_err
    results: dict = {"_schema_version": MCP_SCHEMA_VERSION, "fts": False, "vectors": False}

    try:
        from mind_mem.sqlite_index import build_index

        build_index(ws)
        results["fts"] = True
    except sqlite3.OperationalError as exc:
        if _is_db_locked(exc):
            return _sqlite_busy_error()
        raise
    except (OSError, ValueError) as e:
        _log.warning("reindex_fts_failed", error=str(e))
        results["fts_error"] = "FTS index rebuild failed. Run: mind-mem-scan --reindex"

    if include_vectors:
        try:
            from mind_mem.recall_vector import rebuild_index

            rebuild_index(ws)
            results["vectors"] = True
        except ImportError:
            results["vectors_error"] = "sentence-transformers not installed"
        except (OSError, ValueError) as exc:
            _log.warning("reindex_vectors_failed", error=str(exc))
            results["vectors_error"] = "Vector index rebuild failed"

    # Regenerate category summaries
    try:
        from mind_mem.category_distiller import CategoryDistiller

        extra_cats = _load_extra_categories(ws)
        distiller = CategoryDistiller(extra_categories=extra_cats if extra_cats else None)
        written = distiller.distill(ws)
        results["categories"] = len(written)
    except ImportError:
        _log.debug("reindex_category_distiller_unavailable")
    except (OSError, ValueError) as exc:
        _log.warning("reindex_categories_failed", error=str(exc))
        results["categories_error"] = "Category distillation failed"

    metrics.inc("mcp_reindex")
    _log.info("mcp_reindex", results=results)
    return json.dumps(results, indent=2)


# ---------------------------------------------------------------------------
# Category & Prefetch tools (13-14)
# ---------------------------------------------------------------------------


@mcp.tool
@mcp_tool_observe
def prefetch(signals: str, limit: int = 5) -> str:
    """Pre-assembles likely-needed context from recent conversation signals.

    Given entity mentions, topic keywords, or short phrases from the current
    conversation, anticipates what memory blocks will be needed next.

    Args:
        signals: Comma-separated list of recent signals (entity names, topics,
                 keywords from the conversation).
        limit: Maximum blocks to return (default: 5).

    Returns:
        JSON array of pre-ranked blocks ready for context injection.
    """
    ws = _workspace()
    signal_list = [s.strip() for s in signals.split(",") if s.strip()]
    if not signal_list:
        return json.dumps(
            {
                "_schema_version": MCP_SCHEMA_VERSION,
                "error": "No signals provided. Pass comma-separated keywords.",
            }
        )

    limits = _get_limits(ws)
    limit = max(1, min(limit, limits["max_prefetch_results"]))
    try:
        from mind_mem.recall import prefetch_context

        results = prefetch_context(ws, signal_list, limit=limit)
        metrics.inc("mcp_prefetch_queries")
        _log.info("mcp_prefetch", signals=signal_list, results=len(results))
        return json.dumps(
            {
                "_schema_version": MCP_SCHEMA_VERSION,
                "signals": signal_list,
                "count": len(results),
                "results": results,
            },
            indent=2,
            default=str,
        )
    except Exception:
        import traceback

        _log.warning("prefetch_failed", signals=signal_list, traceback=traceback.format_exc())
        return json.dumps(
            {
                "_schema_version": MCP_SCHEMA_VERSION,
                "error": "Prefetch failed",
                "signals": signal_list,
            },
            indent=2,
        )


# ---------------------------------------------------------------------------
# Kernel config tools
# ---------------------------------------------------------------------------


# v3.2.0 §1.2 PR-3: kernel + compiled_truth tools moved to mcp.tools.kernels
from mind_mem.mcp.tools import kernels as _tools_kernels  # noqa: E402,F401
from mind_mem.mcp.tools.kernels import (  # noqa: E402,F401 — public re-export shim
    compiled_truth_add_evidence,
    compiled_truth_contradictions,
    compiled_truth_load,
    get_mind_kernel,
    list_mind_kernels,
)

_tools_kernels.register(mcp)


# ---------------------------------------------------------------------------
# New tools (#35) — delete_memory_item, export_memory
# ---------------------------------------------------------------------------

# Block ID prefix to directory mapping
_BLOCK_PREFIX_MAP = {
    "D": ("decisions", "DECISIONS.md"),
    "T": ("tasks", "TASKS.md"),
    "C": ("intelligence", "CONTRADICTIONS.md"),
    "INC": ("entities", "incidents.md"),
    "PRJ": ("entities", "projects.md"),
    "PER": ("entities", "people.md"),
    "TOOL": ("entities", "tools.md"),
}


def _find_block_file(ws: str, block_id: str) -> str | None:
    """Resolve a block ID to its source .md file path.

    Returns absolute path or None if the prefix is unrecognized.
    """
    for prefix, (subdir, filename) in _BLOCK_PREFIX_MAP.items():
        if block_id.startswith(prefix + "-"):
            return os.path.join(ws, subdir, filename)
    return None


@mcp.tool
@mcp_tool_observe
def delete_memory_item(block_id: str) -> str:
    """Delete a block by ID from its source .md file.

    Admin-scope tool. Removes the block atomically (write to temp, rename).

    Args:
        block_id: The block ID to delete (e.g., "D-20260213-001", "T-20260215-003").

    Returns:
        JSON confirmation with deleted block ID and file path.
    """
    ws = _workspace()
    ws_err = _check_workspace(ws)
    if ws_err:
        return ws_err

    # Validate block_id format
    if not _re_mod.match(r"^[A-Z]+-[a-zA-Z0-9-]+$", block_id):
        return json.dumps(
            {
                "_schema_version": MCP_SCHEMA_VERSION,
                "error": f"Invalid block ID format: {block_id}",
            }
        )

    filepath = _find_block_file(ws, block_id)
    if filepath is None:
        return json.dumps(
            {
                "_schema_version": MCP_SCHEMA_VERSION,
                "error": f"Unrecognized block ID prefix: {block_id}",
                "hint": "Supported prefixes: " + ", ".join(sorted(_BLOCK_PREFIX_MAP.keys())),
            }
        )

    if not os.path.isfile(filepath):
        return json.dumps(
            {
                "_schema_version": MCP_SCHEMA_VERSION,
                "error": f"Source file not found: {filepath}",
                "block_id": block_id,
            }
        )

    # Read file, find and remove the block (under lock to prevent races)
    with FileLock(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        lines = content.split("\n")
        block_start = None
        block_end = None
        block_header = f"[{block_id}]"

        for i, line in enumerate(lines):
            if line.strip() == block_header:
                block_start = i
            elif block_start is not None and block_end is None:
                # Block ends at next block header, separator, or EOF
                if line.startswith("[") and line.strip().endswith("]") and _re_mod.match(r"^\[[A-Z]+-", line.strip()):
                    block_end = i
                elif line.strip() == "---":
                    # Only treat "---" as a block boundary when it appears at a
                    # block boundary position: either at the start of the file or
                    # preceded by a blank line (not inside block content).
                    preceding_blank = (i == 0) or (lines[i - 1].strip() == "")
                    if preceding_blank:
                        block_end = i + 1  # include the separator

        if block_start is None:
            return json.dumps(
                {
                    "_schema_version": MCP_SCHEMA_VERSION,
                    "error": f"Block {block_id} not found in {os.path.basename(filepath)}",
                    "block_id": block_id,
                }
            )

        if block_end is None:
            block_end = len(lines)

        # Remove the block lines
        deleted_content = "\n".join(lines[block_start:block_end])
        new_lines = lines[:block_start] + lines[block_end:]
        new_content = "\n".join(new_lines)

        # Log deleted block for recovery
        from datetime import datetime, timezone

        deleted_log = os.path.join(ws, "memory", "deleted_blocks.jsonl")
        os.makedirs(os.path.dirname(deleted_log), exist_ok=True)
        with open(deleted_log, "a", encoding="utf-8") as dl:
            entry = {
                "block_id": block_id,
                "deleted_at": datetime.now(timezone.utc).isoformat(),
                "content": deleted_content,
            }
            dl.write(json.dumps(entry, default=str) + "\n")

        # Atomic write: temp file + rename
        dir_name = os.path.dirname(filepath)
        fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".md.tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as tmp_f:
                tmp_f.write(new_content)
            os.replace(tmp_path, filepath)
        except Exception:
            # Clean up temp file on failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    metrics.inc("mcp_delete_memory_item")
    _log.info("mcp_delete_memory_item", block_id=block_id, file=os.path.basename(filepath))

    return json.dumps(
        {
            "_schema_version": MCP_SCHEMA_VERSION,
            "status": "deleted",
            "block_id": block_id,
            "file": os.path.basename(filepath),
            "lines_removed": block_end - block_start,
        },
        indent=2,
    )


@mcp.tool
@mcp_tool_observe
def export_memory(format: str = "jsonl", include_metadata: bool = False, max_blocks: int = 10000) -> str:
    """Export all workspace blocks as JSONL.

    Parses all .md files in decisions/, tasks/, entities/, intelligence/
    and returns one JSON object per line.

    Args:
        format: Output format — currently only "jsonl" is supported.
        include_metadata: Include A-MEM metadata fields (_entities, _dates, etc.).
        max_blocks: Maximum number of blocks to export (default 10000). Prevents
            unbounded memory usage on large workspaces.

    Returns:
        JSONL string with all blocks, or JSON error on failure.
    """
    ws = _workspace()
    ws_err = _check_workspace(ws)
    if ws_err:
        return ws_err

    if format != "jsonl":
        return json.dumps(
            {
                "_schema_version": MCP_SCHEMA_VERSION,
                "error": f"Unsupported format: {format}. Use 'jsonl'.",
            }
        )

    all_blocks = []

    for subdir in CORPUS_DIRS:
        dir_path = os.path.join(ws, subdir)
        if not os.path.isdir(dir_path):
            continue
        for fn in sorted(os.listdir(dir_path)):
            if not fn.endswith(".md"):
                continue
            filepath = os.path.join(dir_path, fn)
            try:
                blocks = parse_file(filepath)
            except (OSError, ValueError) as exc:
                _log.warning("export_parse_failed", file=fn, error=str(exc))
                continue
            for block in blocks:
                block["_source_file"] = f"{subdir}/{fn}"
                if not include_metadata:
                    # Strip internal metadata fields
                    for key in list(block.keys()):
                        if key.startswith("_") and key not in ("_id", "_source_file"):
                            del block[key]
                all_blocks.append(block)

    # Cap output to prevent unbounded memory usage (#447)
    truncated = False
    if len(all_blocks) > max_blocks:
        total = len(all_blocks)
        all_blocks = all_blocks[:max_blocks]
        truncated = True
        _log.warning("export_memory_truncated", total=total, max_blocks=max_blocks)

    # Build JSONL output
    jsonl_lines = [json.dumps(b, default=str) for b in all_blocks]
    jsonl_output = "\n".join(jsonl_lines)

    metrics.inc("mcp_export_memory")
    _log.info("mcp_export_memory", format=format, blocks=len(all_blocks))

    envelope = {
        "_schema_version": MCP_SCHEMA_VERSION,
        "format": format,
        "block_count": len(all_blocks),
        "data": jsonl_output,
    }
    if truncated:
        envelope["warning"] = f"Output truncated to {max_blocks} blocks (total: {total}). Increase max_blocks to export more."

    return json.dumps(
        envelope,
        indent=2,
    )


# ---------------------------------------------------------------------------
# v2.0.0a1 governance tools — verify_chain, list_evidence
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Direct block access, health dashboard, graph, compaction, staleness tools
# ---------------------------------------------------------------------------


@mcp.tool
@mcp_tool_observe
def get_block(block_id: str) -> str:
    """Retrieve a single block by its ID with full content.

    Direct lookup equivalent — returns the complete block including all fields
    (Statement, Status, Tags, Date, etc.) without requiring a search query.
    Useful when you already know the block ID from a previous recall or reference.

    Args:
        block_id: The block ID (e.g., "D-20260213-001", "T-20260215-003").

    Returns:
        JSON with the full block content, source file, and metadata.
    """
    if not _re_mod.match(r"^[A-Z]+-[a-zA-Z0-9_.-]+$", block_id):
        return json.dumps(
            {
                "_schema_version": MCP_SCHEMA_VERSION,
                "error": f"Invalid block_id format: {block_id}",
            }
        )

    ws = _workspace()
    ws_err = _check_workspace(ws)
    if ws_err:
        return ws_err

    # Try known prefix mapping first for O(1) lookup
    filepath = _find_block_file(ws, block_id)
    if filepath and os.path.isfile(filepath):
        try:
            blocks = parse_file(filepath)
            for block in blocks:
                if block.get("_id") == block_id:
                    rel_path = os.path.relpath(filepath, ws)
                    block["_source_file"] = rel_path.replace(os.sep, "/")
                    metrics.inc("mcp_get_block")
                    return json.dumps(
                        {
                            "_schema_version": MCP_SCHEMA_VERSION,
                            "block_id": block_id,
                            "found": True,
                            "block": block,
                        },
                        indent=2,
                        default=str,
                    )
        except (OSError, ValueError, BlockCorruptedError) as exc:
            _log.debug("get_block_parse_failed", file=filepath, error=str(exc))

    # Fallback: scan all corpus directories
    for subdir in CORPUS_DIRS:
        dir_path = os.path.join(ws, subdir)
        if not os.path.isdir(dir_path):
            continue
        for fn in os.listdir(dir_path):
            if not fn.endswith(".md"):
                continue
            fpath = os.path.join(dir_path, fn)
            if fpath == filepath:
                continue  # Already checked above
            try:
                blocks = parse_file(fpath)
                for block in blocks:
                    if block.get("_id") == block_id:
                        block["_source_file"] = f"{subdir}/{fn}"
                        metrics.inc("mcp_get_block")
                        return json.dumps(
                            {
                                "_schema_version": MCP_SCHEMA_VERSION,
                                "block_id": block_id,
                                "found": True,
                                "block": block,
                            },
                            indent=2,
                            default=str,
                        )
            except (OSError, ValueError, BlockCorruptedError):
                continue

    metrics.inc("mcp_get_block_miss")
    return json.dumps(
        {
            "_schema_version": MCP_SCHEMA_VERSION,
            "block_id": block_id,
            "found": False,
            "error": f"Block {block_id} not found in any corpus file.",
            "hint": "Check the block ID and ensure the workspace is initialized.",
        },
        indent=2,
    )


@mcp.tool
@mcp_tool_observe
def memory_health() -> str:
    """Deep health dashboard for the memory workspace.

    Analyzes workspace quality across multiple dimensions:
    - Block counts and active/total ratios per corpus directory
    - Stale blocks needing review (from causal graph staleness flags)
    - Drift items detected (semantic belief drift)
    - Embedding/vector coverage (what percentage of blocks are embedded)
    - Pending signals and unresolved contradictions
    - Compaction candidates (archivable blocks and expired snapshots)
    - FTS index freshness

    Returns:
        JSON health report with per-dimension scores and actionable recommendations.
    """
    ws = _workspace()
    ws_err = _check_workspace(ws)
    if ws_err:
        return ws_err

    health: dict[str, Any] = {"_schema_version": MCP_SCHEMA_VERSION}
    recommendations: list[str] = []

    # 1. Block counts per corpus directory
    corpus_stats: dict[str, dict[str, int]] = {}
    total_blocks = 0
    total_active = 0
    for subdir in CORPUS_DIRS:
        dir_path = os.path.join(ws, subdir)
        if not os.path.isdir(dir_path):
            corpus_stats[subdir] = {"total": 0, "active": 0}
            continue
        sub_total = 0
        sub_active = 0
        for fn in os.listdir(dir_path):
            if not fn.endswith(".md") or fn.endswith("_ARCHIVE.md"):
                continue
            try:
                blocks = parse_file(os.path.join(dir_path, fn))
                sub_total += len(blocks)
                sub_active += len(get_active(blocks))
            except (OSError, ValueError):
                pass
        corpus_stats[subdir] = {"total": sub_total, "active": sub_active}
        total_blocks += sub_total
        total_active += sub_active
    health["corpus"] = corpus_stats
    health["total_blocks"] = total_blocks
    health["total_active"] = total_active

    # 2. Stale blocks from causal graph
    stale_count = 0
    try:
        from mind_mem.causal_graph import CausalGraph

        cg = CausalGraph(ws)
        stale = cg.get_stale_blocks()
        stale_count = len(stale)
        health["stale_blocks"] = stale_count
        if stale_count > 0:
            health["stale_block_ids"] = [s["block_id"] for s in stale[:10]]
            recommendations.append(
                f"{stale_count} stale block(s) need review. Use stale_blocks tool for details, then update or clear staleness."
            )
    except (ImportError, sqlite3.OperationalError, OSError, ValueError) as exc:
        health["stale_blocks"] = 0
        _log.debug("health_stale_check_skipped", error=str(exc))

    # 3. Drift items
    drift_path = os.path.join(ws, "intelligence", "DRIFT.md")
    drift_count = 0
    if os.path.isfile(drift_path):
        try:
            drift_count = len(parse_file(drift_path))
        except (OSError, ValueError):
            pass
    health["drift_items"] = drift_count
    if drift_count > 0:
        recommendations.append(f"{drift_count} drift item(s) detected. Review intelligence/DRIFT.md for belief shifts.")

    # 4. Embedding/vector coverage
    import struct as _struct_mod

    try:
        from mind_mem import recall_vector as _rv

        vec_path = _rv._index_path(ws)  # type: ignore[attr-defined]
        if os.path.isfile(vec_path):
            with open(vec_path, "rb") as f:
                header = f.read(8)
                if len(header) >= 4:
                    embedded_count = _struct_mod.unpack("<I", header[:4])[0]
                    health["embedded_blocks"] = embedded_count
                    if total_blocks > 0:
                        coverage = round(embedded_count / total_blocks * 100, 1)
                        health["embedding_coverage_pct"] = coverage
                        if coverage < 80:
                            recommendations.append(f"Embedding coverage is {coverage}%. Run reindex(include_vectors=True).")
                else:
                    health["embedded_blocks"] = 0
                    health["embedding_coverage_pct"] = 0.0
        else:
            health["embedded_blocks"] = 0
            health["embedding_coverage_pct"] = 0.0
            if total_blocks > 10:
                recommendations.append("No vector index found. Run reindex(include_vectors=True) for hybrid search.")
    except (ImportError, OSError, _struct_mod.error):
        health["embedded_blocks"] = "unknown"
        health["embedding_coverage_pct"] = "unknown"

    # 5. Pending signals and unresolved contradictions
    signals_path = os.path.join(ws, "intelligence", "SIGNALS.md")
    pending_signals = 0
    if os.path.isfile(signals_path):
        try:
            sigs = parse_file(signals_path)
            pending_signals = len([s for s in sigs if s.get("Status", "pending") == "pending"])
        except (OSError, ValueError):
            pass
    health["pending_signals"] = pending_signals
    if pending_signals > 5:
        recommendations.append(f"{pending_signals} pending signals. Review and apply or reject them.")

    contra_path = os.path.join(ws, "intelligence", "CONTRADICTIONS.md")
    contra_count = 0
    if os.path.isfile(contra_path):
        try:
            contra_count = len(parse_file(contra_path))
        except (OSError, ValueError):
            pass
    health["unresolved_contradictions"] = contra_count
    if contra_count > 0:
        recommendations.append(f"{contra_count} unresolved contradiction(s). Use list_contradictions for details.")

    # 6. FTS index freshness
    db = fts_db_path(ws)
    if db and os.path.isfile(db):
        try:
            from mind_mem.sqlite_index import index_status as fts_status

            info = fts_status(ws)
            health["fts_index"] = {
                "exists": True,
                "blocks_indexed": info.get("blocks", 0),
                "stale_files": info.get("stale_files", 0),
                "last_build": info.get("last_build"),
                "db_size_bytes": info.get("db_size_bytes", 0),
            }
            stale_files = info.get("stale_files", 0)
            if stale_files > 0:
                recommendations.append(f"FTS index has {stale_files} stale file(s). Run reindex tool.")
        except (sqlite3.OperationalError, OSError, ValueError):
            health["fts_index"] = {"exists": True, "error": "Could not read index status"}
    else:
        health["fts_index"] = {"exists": False}
        recommendations.append("No FTS index. Run reindex tool for fast keyword search.")

    # 7. Compaction candidates
    try:
        from mind_mem.compaction import archive_completed_blocks, compact_signals

        archivable = archive_completed_blocks(ws, days=90, dry_run=True)
        compactable_signals = compact_signals(ws, days=60, dry_run=True)
        health["compaction"] = {
            "archivable_blocks": len(archivable),
            "compactable_signals": len(compactable_signals),
        }
        total_compactable = len(archivable) + len(compactable_signals)
        if total_compactable > 0:
            recommendations.append(f"{total_compactable} item(s) ready for compaction. Run compact tool.")
    except (ImportError, OSError, ValueError) as exc:
        health["compaction"] = {"error": str(exc)}

    health["recommendations"] = recommendations
    health["score"] = "healthy" if not recommendations else "needs_attention"

    metrics.inc("mcp_memory_health")
    _log.info("mcp_memory_health", total_blocks=total_blocks, recommendations=len(recommendations))
    return json.dumps(health, indent=2, default=str)


@mcp.tool
@mcp_tool_observe
def compact(dry_run: bool = True, archive_days: int = 90, signal_days: int = 60, snapshot_days: int = 30) -> str:
    """Run workspace compaction — archive old blocks, clean snapshots, remove resolved signals.

    Maintenance tool that keeps the workspace lean without losing data:
    - Archives completed/canceled tasks and superseded/revoked decisions
    - Removes expired apply snapshots
    - Compacts resolved/rejected signals from SIGNALS.md
    - Archives old daily log files

    SAFETY: Defaults to dry_run=True. Set dry_run=False to execute.
    Archived blocks are moved to *_ARCHIVE.md files, never deleted.

    Args:
        dry_run: Preview what would be done without changing files (default: True).
        archive_days: Archive completed blocks older than N days (default: 90).
        signal_days: Remove resolved signals older than N days (default: 60).
        snapshot_days: Remove apply snapshots older than N days (default: 30).

    Returns:
        JSON report with actions taken (or previewed) per category.
    """
    ws = _workspace()
    ws_err = _check_workspace(ws)
    if ws_err:
        return ws_err

    from mind_mem.compaction import (
        archive_completed_blocks,
        cleanup_daily_logs,
        cleanup_snapshots,
        compact_signals,
    )

    actions: dict[str, list[str]] = {}

    # 1. Archive completed blocks
    try:
        block_actions = archive_completed_blocks(ws, days=archive_days, dry_run=dry_run)
        actions["archived_blocks"] = block_actions
    except (OSError, ValueError) as exc:
        actions["archived_blocks_error"] = [str(exc)]
        _log.warning("compact_archive_failed", error=str(exc))

    # 2. Cleanup snapshots
    try:
        snap_actions = cleanup_snapshots(ws, days=snapshot_days, dry_run=dry_run)
        actions["cleaned_snapshots"] = snap_actions
    except (OSError, ValueError) as exc:
        actions["cleaned_snapshots_error"] = [str(exc)]
        _log.warning("compact_snapshots_failed", error=str(exc))

    # 3. Compact signals
    try:
        signal_actions = compact_signals(ws, days=signal_days, dry_run=dry_run)
        actions["compacted_signals"] = signal_actions
    except (OSError, ValueError) as exc:
        actions["compacted_signals_error"] = [str(exc)]
        _log.warning("compact_signals_failed", error=str(exc))

    # 4. Archive daily logs
    try:
        log_actions = cleanup_daily_logs(ws, days=180, dry_run=dry_run)
        actions["archived_logs"] = log_actions
    except (OSError, ValueError) as exc:
        actions["archived_logs_error"] = [str(exc)]
        _log.warning("compact_logs_failed", error=str(exc))

    total_actions = sum(len(v) for v in actions.values() if isinstance(v, list))

    metrics.inc("mcp_compact")
    _log.info("mcp_compact", dry_run=dry_run, total_actions=total_actions)

    return json.dumps(
        {
            "_schema_version": MCP_SCHEMA_VERSION,
            "status": "dry_run" if dry_run else "executed",
            "dry_run": dry_run,
            "total_actions": total_actions,
            "actions": actions,
            "next_step": (
                "Call again with dry_run=False to execute."
                if dry_run and total_actions > 0
                else "Workspace is clean — nothing to compact."
                if total_actions == 0
                else None
            ),
        },
        indent=2,
    )


@mcp.tool
@mcp_tool_observe
def stale_blocks(limit: int = 20, clear_block_id: str = "") -> str:
    """List blocks flagged as stale due to upstream changes, or clear a staleness flag.

    When a block is modified, all downstream dependents in the causal graph are
    automatically flagged as stale (needing review). This tool surfaces those
    flags so agents can prioritize which blocks to re-evaluate.

    Args:
        limit: Maximum number of stale blocks to return (default: 20).
        clear_block_id: If provided, clear the staleness flag for this block
                        (after reviewing/updating it). Leave empty to list.

    Returns:
        JSON with stale block list (with reasons and timestamps), or
        confirmation of flag clearance.
    """
    ws = _workspace()

    try:
        from mind_mem.causal_graph import CausalGraph

        cg = CausalGraph(ws)

        # Clear mode
        if clear_block_id:
            if not _re_mod.match(r"^[A-Z]+-[a-zA-Z0-9_.-]+$", clear_block_id):
                return json.dumps(
                    {
                        "_schema_version": MCP_SCHEMA_VERSION,
                        "error": f"Invalid block_id format: {clear_block_id}",
                    }
                )
            cleared = cg.clear_staleness(clear_block_id)
            metrics.inc("mcp_stale_cleared")
            return json.dumps(
                {
                    "_schema_version": MCP_SCHEMA_VERSION,
                    "action": "cleared",
                    "block_id": clear_block_id,
                    "was_stale": cleared,
                },
                indent=2,
            )

        # List mode
        stale = cg.get_stale_blocks()
        stale = stale[: max(1, min(limit, 100))]

        metrics.inc("mcp_stale_blocks")
        _log.info("mcp_stale_blocks", count=len(stale))

        if not stale:
            return json.dumps(
                {
                    "_schema_version": MCP_SCHEMA_VERSION,
                    "status": "clean",
                    "stale_count": 0,
                    "message": "No stale blocks. All blocks are up to date.",
                },
                indent=2,
            )

        return json.dumps(
            {
                "_schema_version": MCP_SCHEMA_VERSION,
                "status": "stale_found",
                "stale_count": len(stale),
                "blocks": stale,
                "hint": "Review each stale block and update or call stale_blocks(clear_block_id='...') to clear.",
            },
            indent=2,
            default=str,
        )

    except ImportError:
        return json.dumps(
            {
                "_schema_version": MCP_SCHEMA_VERSION,
                "error": "causal_graph module not available",
            },
            indent=2,
        )
    except sqlite3.OperationalError as exc:
        if _is_db_locked(exc):
            return _sqlite_busy_error()
        raise
    except (OSError, ValueError) as exc:
        _log.warning("stale_blocks_failed", error=str(exc))
        return json.dumps(
            {
                "_schema_version": MCP_SCHEMA_VERSION,
                "error": f"Stale block lookup failed: {exc}",
            },
            indent=2,
        )


# ---------------------------------------------------------------------------
# Calibration feedback tools (v3.2.0 §1.2 PR-3 — moved to mcp.tools.calibration)
# ---------------------------------------------------------------------------
from mind_mem.mcp.tools import calibration as _tools_calibration  # noqa: E402,F401
from mind_mem.mcp.tools.calibration import (  # noqa: E402,F401 — public re-export shim
    calibration_feedback,
    calibration_stats,
)

_tools_calibration.register(mcp)


# ---------------------------------------------------------------------------
# Dream Cycle + Compiled Truth tools
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    """Entry point for the MCP server (used by console_scripts and __main__)."""
    import argparse

    parser = argparse.ArgumentParser(description="Mind-Mem MCP Server")
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio", help="Transport protocol (default: stdio)")
    parser.add_argument("--port", type=int, default=8765, help="HTTP port (only used with --transport http)")
    parser.add_argument("--token", default=None, help="Bearer token for HTTP auth (or set MIND_MEM_TOKEN env var)")
    parser.add_argument("--watch", action="store_true", help="Auto-reindex when workspace .md files change")
    parser.add_argument("--watch-interval", type=float, default=5.0, help="File watch polling interval in seconds (default: 5.0)")
    args = parser.parse_args()

    # Set token from CLI arg if provided (env var takes precedence if both set)
    if args.token and not os.environ.get("MIND_MEM_TOKEN"):
        import warnings

        warnings.warn(
            "Passing --token on the command line exposes it in /proc/cmdline. Use MIND_MEM_TOKEN environment variable instead.",
            stacklevel=2,
        )
        os.environ["MIND_MEM_TOKEN"] = args.token

    # Warn if HTTP transport is used without an admin token.
    if args.transport == "http" and _check_token() and not os.environ.get("MIND_MEM_ADMIN_TOKEN"):
        import warnings as _w

        _w.warn(
            "HTTP transport without MIND_MEM_ADMIN_TOKEN: authenticated clients only receive user scope. "
            "Set MIND_MEM_ADMIN_TOKEN to enable admin operations over HTTP.",
            stacklevel=2,
        )

    token = _check_token()
    _log.info("mcp_server_start", transport=args.transport, workspace=_workspace(), auth="token" if token else "none")

    # File watcher: auto-reindex on .md changes
    if args.watch:
        from mind_mem.watcher import FileWatcher
        from mind_mem.sqlite_index import build_index

        ws = _workspace()

        def _on_changes(changed_files: set[str]) -> None:
            try:
                result = build_index(ws, incremental=True)
                _log.info(
                    "watch_reindex_complete",
                    blocks_new=result.get("blocks_new", 0),
                    blocks_modified=result.get("blocks_modified", 0),
                )
            except Exception as e:
                _log.warning("watch_reindex_failed", error=str(e))

        watcher = FileWatcher(ws, callback=_on_changes, interval=args.watch_interval)
        watcher.start()
        _log.info("file_watcher_enabled", interval=args.watch_interval)

    if args.transport == "http":
        auth_tokens = _build_http_auth_tokens()
        if not auth_tokens:
            _log.warning(
                "mcp_http_no_auth",
                hint="HTTP transport running without token auth. Set MIND_MEM_TOKEN or MIND_MEM_ADMIN_TOKEN for security.",
            )
        else:
            # Enforce Bearer token auth on HTTP transport.
            from fastmcp.server.auth import StaticTokenVerifier

            mcp.auth = StaticTokenVerifier(tokens=auth_tokens)
            _log.info("mcp_auth_enforced", mode="static_token", token_count=len(auth_tokens))
        mcp.run(transport="sse", port=args.port)
    else:
        mcp.run()


if __name__ == "__main__":
    main()
