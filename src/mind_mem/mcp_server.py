#!/usr/bin/env python3
"""Mind-Mem MCP Server — public facade (v3.2.0 §1.2 PR-final shim).

The historical ``mind_mem.mcp_server`` module has been decomposed
into ``mind_mem.mcp.*`` submodules per
docs/v3.2.0-mcp-decomposition-plan.md. This file is now a thin
re-export shim that keeps every documented import path working:

* ``from mind_mem.mcp_server import mcp, main`` (entry point)
* ``from mind_mem.mcp_server import recall, propose_update, ...`` (every tool)
* ``from mind_mem.mcp_server import get_decisions, ...`` (every resource)
* ``from mind_mem.mcp_server import check_tool_acl, ADMIN_TOOLS, ...`` (infra)
* ``mcp_server.verify_token`` / ``mcp_server.metrics`` / etc.
  (attribute access from tests)

Nothing is removed — the shim stays for at least one major
version of overlap (until v4.0). The split lets new code import
from the precise module that owns each symbol instead of the
4.6KLOC god-object this module used to be.

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

# ---------------------------------------------------------------------------
# Public re-exports — every symbol the pre-v3.2.0 monolith exposed.
# ---------------------------------------------------------------------------
#
# Absolute imports (not relative) because the top-level developer-checkout
# shim at ``/home/n/mind-mem/mcp_server.py`` runs this file via
# ``exec(compile(...))`` with no parent package, and the test harness
# loads it via ``spec_from_file_location`` which also strips the
# package. ``mind_mem.mcp.*`` resolves in both paths because the shim
# inserts ``src/`` onto ``sys.path`` before the exec.

# Symbols used by tests that patch attributes on this module.
from mind_mem.block_parser import BlockCorruptedError, get_active, parse_file  # noqa: E402, F401
from fastmcp import FastMCP  # noqa: E402, F401
from mind_mem.observability import get_logger, metrics  # noqa: E402, F401
from mind_mem.sqlite_index import _db_path as fts_db_path  # noqa: E402, F401

_log = get_logger("mcp_server")

# Infra — cross-cutting helpers used across every domain module.
from mind_mem.mcp.infra.acl import (  # noqa: E402, F401
    ADMIN_TOOLS,
    USER_TOOLS,
    _ADMIN_SCOPES,
    _get_request_scope,
    check_tool_acl,
)
from mind_mem.mcp.infra.config import (  # noqa: E402, F401
    _DEFAULT_LIMITS,
    QUERY_TIMEOUT_SECONDS,
    _get_limits,
    _load_config,
    _load_extra_categories,
)
from mind_mem.mcp.infra.constants import MCP_SCHEMA_VERSION  # noqa: E402, F401
from mind_mem.mcp.infra.http_auth import (  # noqa: E402, F401
    _build_http_auth_tokens,
    _check_token,
    verify_token,
)
from mind_mem.mcp.infra.observability import (  # noqa: E402, F401
    _is_db_locked,
    _sqlite_busy_error,
    mcp_tool_observe,
)
from mind_mem.mcp.infra.rate_limit import (  # noqa: E402, F401
    _RATE_LIMITER_MAX,
    SlidingWindowRateLimiter,
    _get_client_id,
    _get_client_rate_limiter,
    _init_rate_limiter,
    _rate_limiters,
    _rate_limiters_lock,
)
from mind_mem.mcp.infra.workspace import (  # noqa: E402, F401
    _check_workspace,
    _read_file,
    _validate_path,
    _workspace,
)

# Server — FastMCP instance + entry point (tool + resource registrations
# happen at import time inside ``mind_mem.mcp.server``).
from mind_mem.mcp.server import main, mcp  # noqa: E402, F401

# Resources (read-only).
from mind_mem.mcp.resources import (  # noqa: E402, F401
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

# Shared tool-private helpers (workspace paths + lazy singletons).
from mind_mem.mcp.tools._helpers import (  # noqa: E402, F401
    _change_stream,
    _core_dir,
    _core_registry,
    _kg_path,
    _ontology_registry,
    _signal_store_path,
)

# Tools — every @mcp.tool name the monolith exposed.
from mind_mem.mcp.tools.agent import (  # noqa: E402, F401
    _vault_allowlist,
    _vault_root_allowed,
    agent_inject,
    stream_status,
    vault_scan,
    vault_sync,
)
from mind_mem.mcp.tools.audit import (  # noqa: E402, F401
    list_evidence,
    mind_mem_verify,
    verify_chain,
    verify_merkle,
)
from mind_mem.mcp.tools.benchmark import (  # noqa: E402, F401
    category_summary,
    governance_health_bench,
)
from mind_mem.mcp.tools.calibration import (  # noqa: E402, F401
    calibration_feedback,
    calibration_stats,
)
from mind_mem.mcp.tools.consolidation import (  # noqa: E402, F401
    dream_cycle,
    plan_consolidation,
    project_profile,
    propagate_staleness,
)
from mind_mem.mcp.tools.core import (  # noqa: E402, F401
    build_core,
    list_cores,
    load_core,
    unload_core,
)
from mind_mem.mcp.tools.encryption import (  # noqa: E402, F401
    _encryption_passphrase,
    _safe_vault_path,
    decrypt_file,
    encrypt_file,
)
from mind_mem.mcp.tools.governance import (  # noqa: E402, F401
    approve_apply,
    list_contradictions,
    memory_evolution,
    propose_update,
    rollback_proposal,
    scan,
)
from mind_mem.mcp.tools.graph import (  # noqa: E402, F401
    graph_add_edge,
    graph_query,
    graph_stats,
    traverse_graph,
)
from mind_mem.mcp.tools.kernels import (  # noqa: E402, F401
    compiled_truth_add_evidence,
    compiled_truth_contradictions,
    compiled_truth_load,
    get_mind_kernel,
    list_mind_kernels,
)
from mind_mem.mcp.tools.memory_ops import (  # noqa: E402, F401
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
from mind_mem.mcp.tools.ontology import (  # noqa: E402, F401
    ontology_load,
    ontology_validate,
)
from mind_mem.mcp.tools.recall import (  # noqa: E402, F401
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
from mind_mem.mcp.tools.signal import (  # noqa: E402, F401
    observe_signal,
    signal_stats,
)


if __name__ == "__main__":
    main()
