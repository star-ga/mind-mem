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

import os
import sys

# Allow running `python3 mcp_server.py` directly from a source checkout.
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_REPO_ROOT, "src")
if os.path.isdir(_SRC_DIR) and _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# mind-mem imports (package mapped to scripts/ via pyproject.toml)
from mind_mem.block_parser import BlockCorruptedError, get_active, parse_file  # noqa: E402, F401
from fastmcp import FastMCP  # noqa: E402
from mind_mem.observability import get_logger, metrics  # noqa: E402,F401
# ``fts_db_path`` + ``metrics`` re-exported so ``tests/test_mcp_v140.py``
# keeps resolving ``mcp_server.fts_db_path`` + ``mcp_server.metrics``
# after the memory_ops + infra extractions.
from mind_mem.sqlite_index import _db_path as fts_db_path  # noqa: E402,F401

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
#
# v3.2.0 §1.2 PR-3: recall tools moved to mcp.tools.recall.
# ``_recall_impl`` is re-exported because ``agent_inject`` still
# late-imports it via ``mind_mem.mcp_server._recall_impl``.
from mind_mem.mcp.tools import recall as _tools_recall  # noqa: E402,F401
from mind_mem.mcp.tools.recall import (  # noqa: E402,F401 — public re-export shim
    _recall_impl,
    find_similar,
    hybrid_search,
    intent_classify,
    pack_recall_budget,
    prefetch,
    recall,
    recall_with_axis,
    retrieval_diagnostics,
)

_tools_recall.register(mcp)


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


# v3.2.0 §1.2 PR-3: memory_ops tools moved to mcp.tools.memory_ops
# (index_stats, reindex, delete_memory_item, export_memory, get_block,
#  memory_health, compact, stale_blocks). Re-exports land together with
#  the ``_BLOCK_PREFIX_MAP`` + ``_find_block_file`` helpers they share.
from mind_mem.mcp.tools import memory_ops as _tools_memory_ops  # noqa: E402,F401
from mind_mem.mcp.tools.memory_ops import (  # noqa: E402,F401 — public re-export shim
    _BLOCK_PREFIX_MAP,
    _find_block_file,
    compact,
    delete_memory_item,
    export_memory,
    get_block,
    index_stats,
    memory_health,
    reindex,
    stale_blocks,
)

_tools_memory_ops.register(mcp)


# ---------------------------------------------------------------------------
# Category & Prefetch tools (13-14)
# ---------------------------------------------------------------------------


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
