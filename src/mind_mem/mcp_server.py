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

import functools
import hmac
import json
import os
import re as _re_mod
import sqlite3
import sys
import tempfile
import threading
import time
from collections import OrderedDict
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
from fastmcp.server.dependencies import get_access_token  # noqa: E402
from mind_mem.mind_ffi import (  # noqa: E402
    get_mind_dir,
    is_available as mind_kernel_available,
    is_protected as mind_kernel_protected,
    list_kernels as ffi_list_kernels,
    load_all_kernel_configs,
    load_kernel_config,
)
from mind_mem.observability import get_logger, metrics  # noqa: E402
from mind_mem.recall import recall as recall_engine  # noqa: E402
from mind_mem.retrieval_graph import retrieval_diagnostics as _retrieval_diag  # noqa: E402
from mind_mem.sqlite_index import (  # noqa: E402
    _db_path as fts_db_path,
    query_index as fts_query,
)

_log = get_logger("mcp_server")

MCP_SCHEMA_VERSION = "1.0"


# ---------------------------------------------------------------------------
# ACL — per-tool scope enforcement (#20)
# ---------------------------------------------------------------------------

ADMIN_TOOLS = frozenset(
    {
        "write_memory",
        "apply_proposal",
        "approve_apply",
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


def _build_http_auth_tokens() -> dict[str, dict[str, Any]]:
    """Build StaticTokenVerifier token metadata from environment variables."""
    tokens: dict[str, dict[str, Any]] = {}

    user_token = _check_token()
    if user_token:
        tokens[user_token] = {
            "client_id": "mind-mem-user",
            "scopes": ["user"],
            "sub": "mind-mem-user",
        }

    admin_token = os.environ.get("MIND_MEM_ADMIN_TOKEN")
    if admin_token:
        tokens[admin_token] = {
            "client_id": "mind-mem-admin",
            "scopes": ["user", "admin"],
            "sub": "mind-mem-admin",
        }

    return tokens


# ---------------------------------------------------------------------------
# Rate limiter — sliding window (#21)
# ---------------------------------------------------------------------------


class SlidingWindowRateLimiter:
    """In-memory sliding-window rate limiter."""

    def __init__(self, max_calls: int = 120, window_seconds: int = 60):
        self.max_calls = max_calls
        self.window = window_seconds
        self._timestamps: list[float] = []
        self._lock = threading.Lock()

    def allow(self) -> tuple[bool, float]:
        """Check if a call is allowed.

        Returns (allowed, retry_after_seconds).  retry_after is 0.0 when allowed.
        """
        now = time.monotonic()
        with self._lock:
            cutoff = now - self.window
            self._timestamps = [t for t in self._timestamps if t > cutoff]
            if len(self._timestamps) >= self.max_calls:
                retry_after = self._timestamps[0] - cutoff
                return False, max(retry_after, 0.1)
            self._timestamps.append(now)
            return True, 0.0


# ---------------------------------------------------------------------------
# Configurable limits (#37) — loaded from mind-mem.json "limits" section
# ---------------------------------------------------------------------------

_DEFAULT_LIMITS = {
    "max_recall_results": 100,
    "max_similar_results": 50,
    "max_prefetch_results": 20,
    "max_category_results": 10,
    "query_timeout_seconds": 30,
    "rate_limit_calls_per_minute": 120,
}


def _get_limits(ws: str | None = None) -> dict:
    """Return the limits dict from config, falling back to defaults."""
    if ws is None:
        try:
            ws = _workspace()
        except Exception:
            return dict(_DEFAULT_LIMITS)
    cfg = _load_config(ws)
    limits = cfg.get("limits", {})
    result = dict(_DEFAULT_LIMITS)
    for key in _DEFAULT_LIMITS:
        if key in limits:
            try:
                result[key] = int(limits[key])
            except (TypeError, ValueError):
                pass  # keep default
    return result


def _init_rate_limiter() -> SlidingWindowRateLimiter:
    """Create a rate limiter from config limits (used as per-client factory)."""
    try:
        limits = _get_limits()
        max_calls = limits["rate_limit_calls_per_minute"]
    except Exception:
        max_calls = 120
    return SlidingWindowRateLimiter(max_calls=max_calls, window_seconds=60)


# Per-client rate limiters keyed by client_id — prevents one client from
# exhausting the global budget and blocking all other clients. An
# attacker rotating Authorization tokens on every request would
# otherwise grow this dict unbounded, so we LRU-evict the least-
# recently-used entry when the cap is reached.
_RATE_LIMITER_MAX: int = 1024
_rate_limiters: "OrderedDict[str, SlidingWindowRateLimiter]" = OrderedDict()
_rate_limiters_lock = threading.Lock()


def _get_client_rate_limiter(client_id: str) -> SlidingWindowRateLimiter:
    """Return (creating if needed) the per-client SlidingWindowRateLimiter."""
    with _rate_limiters_lock:
        existing = _rate_limiters.get(client_id)
        if existing is not None:
            _rate_limiters.move_to_end(client_id, last=True)
            return existing
        limiter = _init_rate_limiter()
        _rate_limiters[client_id] = limiter
        while len(_rate_limiters) > _RATE_LIMITER_MAX:
            _rate_limiters.popitem(last=False)
        return limiter


def _get_client_id() -> str:
    """Return a stable client identifier for the current request."""
    try:
        token = get_access_token()
        if token is not None and token.client_id:
            return token.client_id
    except Exception:
        pass
    return "default"

# Per-query timeout in seconds (read from config at call time via _get_limits)
QUERY_TIMEOUT_SECONDS = _DEFAULT_LIMITS["query_timeout_seconds"]


def _sqlite_busy_error() -> str:
    """Return structured JSON error for SQLite database locked (#29)."""
    return json.dumps(
        {
            "_schema_version": MCP_SCHEMA_VERSION,
            "error": "database_busy",
            "message": "Database is temporarily locked by another process",
            "retry_after_seconds": 1,
        }
    )


def _is_db_locked(exc: sqlite3.OperationalError) -> bool:
    """Check if a sqlite3.OperationalError is a database-locked error."""
    return "database is locked" in str(exc).lower()


def mcp_tool_observe(fn):
    """Decorator that wraps MCP tool calls with observability logging (#31).

    Logs structured JSON for every call: tool_name, duration_ms, success,
    error_type, result_size.  Also increments success/failure counters.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        tool_name = fn.__name__

        # Rate limit enforcement (#475): per-client sliding window
        client_id = _get_client_id()
        limiter = _get_client_rate_limiter(client_id)
        allowed, retry_after = limiter.allow()
        if not allowed:
            _log.warning("rate_limit_exceeded", tool=tool_name, client=client_id, retry_after=retry_after)
            return json.dumps(
                {
                    "error": "Rate limit exceeded. Try again later.",
                    "retry_after_seconds": round(retry_after, 1),
                    "_schema_version": MCP_SCHEMA_VERSION,
                }
            )

        # ACL enforcement: when MIND_MEM_ADMIN_TOKEN is configured,
        # admin tools require MIND_MEM_SCOPE=admin (stdio) or a valid
        # Authorization header (http).  When no admin token is set the
        # check is skipped so single-user local setups are unaffected.
        admin_token = os.environ.get("MIND_MEM_ADMIN_TOKEN")
        request_scope = _get_request_scope()
        acl_scope = request_scope or os.environ.get("MIND_MEM_SCOPE", "user")
        acl_active = admin_token is not None or request_scope is not None
        if acl_active and tool_name in ADMIN_TOOLS:
            scope = acl_scope
            acl_error = check_tool_acl(tool_name, scope)
            if acl_error:
                _log.warning("acl_blocked", tool=tool_name, scope=scope)
                return acl_error
        elif acl_active and tool_name not in USER_TOOLS:
            _log.warning("acl_unknown_tool", tool=tool_name)
            return json.dumps({"error": f"Tool '{tool_name}' is not in ACL policy", "_schema_version": "1.0"})

        start = time.monotonic()
        error_type = None
        success = True
        result = ""
        try:
            result = fn(*args, **kwargs)
            return result
        except Exception as exc:
            success = False
            error_type = type(exc).__name__
            raise
        finally:
            duration_ms = round((time.monotonic() - start) * 1000, 2)
            result_size = len(result) if isinstance(result, str) else 0
            _log.info(
                "mcp_tool_call",
                tool_name=tool_name,
                duration_ms=duration_ms,
                success=success,
                error_type=error_type,
                result_size=result_size,
            )
            metrics.observe("mcp_tool_duration_ms", duration_ms)
            if success:
                metrics.inc("mcp_tool_success")
            else:
                metrics.inc("mcp_tool_failure")

    return wrapper


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


def _workspace() -> str:
    """Resolve workspace path from environment."""
    ws = os.environ.get("MIND_MEM_WORKSPACE", ".")
    return os.path.abspath(ws)


def _check_workspace(ws: str) -> str | None:
    """Validate workspace exists and has expected structure.

    Returns None if valid, or an error JSON string if invalid.
    """
    if not os.path.isdir(ws):
        return json.dumps({"error": "Workspace not found. Run: mind-mem-init <path>"})
    decisions_dir = os.path.join(ws, "decisions")
    if not os.path.isdir(decisions_dir):
        return json.dumps({"error": "Workspace is missing the 'decisions/' directory. Run: mind-mem-init <path>"})
    return None


def _validate_path(ws: str, rel_path: str) -> str:
    """Validate that rel_path resolves inside workspace. Returns resolved path.

    Raises ValueError if the path escapes the workspace boundary.
    """
    ws_real = os.path.realpath(ws)
    path = os.path.realpath(os.path.join(ws_real, rel_path))
    if path != ws_real and not path.startswith(ws_real + os.sep):
        raise ValueError("Invalid path: escapes workspace")
    return path


def _read_file(rel_path: str) -> str:
    """Read a file from workspace, return contents or error message."""
    ws = _workspace()
    try:
        path = _validate_path(ws, rel_path)
    except ValueError:
        return "Error: path escapes workspace"
    if not os.path.isfile(path):
        return f"File not found: {rel_path}"
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _blocks_to_json(blocks: list[dict]) -> str:
    """Convert parsed blocks to JSON string."""
    return json.dumps(blocks, indent=2, default=str)


def _load_config(ws: str) -> dict:
    """Load mind-mem.json config with graceful fallback (#26).

    On JSONDecodeError, logs line/column and returns DEFAULT_CONFIG.
    """
    config_path = os.path.join(ws, "mind-mem.json")
    if not os.path.isfile(config_path):
        return {}
    try:
        with open(config_path, encoding="utf-8") as f:
            return dict(json.load(f))
    except json.JSONDecodeError as exc:
        _log.warning(
            "config_json_decode_error",
            path=config_path,
            line=exc.lineno,
            column=exc.colno,
            msg=str(exc),
        )
        # Fall back to built-in defaults
        from mind_mem.init_workspace import DEFAULT_CONFIG

        return dict(DEFAULT_CONFIG)
    except (OSError, UnicodeDecodeError) as exc:
        _log.warning("config_read_error", path=config_path, error=str(exc))
        return {}


def _load_extra_categories(ws: str) -> dict:
    """Load extra_categories from mind-mem.json config."""
    cfg = _load_config(ws)
    return dict(cfg.get("categories", {}).get("extra_categories", {}))


# ---------------------------------------------------------------------------
# Resources (read-only)
# ---------------------------------------------------------------------------


@mcp.resource("mind-mem://decisions")
def get_decisions() -> str:
    """Active decisions from the workspace. Structured blocks with IDs, statements, dates, and status."""
    ws = _workspace()
    path = os.path.join(ws, "decisions", "DECISIONS.md")
    if not os.path.isfile(path):
        return json.dumps([])
    blocks = parse_file(path)
    active = get_active(blocks)
    return _blocks_to_json(active)


@mcp.resource("mind-mem://tasks")
def get_tasks() -> str:
    """All tasks from the workspace."""
    ws = _workspace()
    path = os.path.join(ws, "tasks", "TASKS.md")
    if not os.path.isfile(path):
        return json.dumps([])
    blocks = parse_file(path)
    return _blocks_to_json(blocks)


@mcp.resource("mind-mem://entities/{entity_type}")
def get_entities(entity_type: str) -> str:
    """Entity files: projects, people, tools, or incidents."""
    allowed = {"projects", "people", "tools", "incidents"}
    if entity_type not in allowed:
        return json.dumps({"error": f"Unknown entity type: {entity_type}. Use: {', '.join(sorted(allowed))}"})
    return _read_file(f"entities/{entity_type}.md")


@mcp.resource("mind-mem://signals")
def get_signals() -> str:
    """Auto-captured signals pending review."""
    return _read_file("intelligence/SIGNALS.md")


@mcp.resource("mind-mem://contradictions")
def get_contradictions() -> str:
    """Detected contradictions between decisions."""
    return _read_file("intelligence/CONTRADICTIONS.md")


@mcp.resource("mind-mem://health")
def get_health() -> str:
    """Workspace health summary: block counts, coverage, and metrics."""
    ws = _workspace()
    result: dict[str, Any] = {"files": {}, "metrics": {}}

    corpus = {
        "decisions": "decisions/DECISIONS.md",
        "tasks": "tasks/TASKS.md",
        "contradictions": "intelligence/CONTRADICTIONS.md",
        "signals": "intelligence/SIGNALS.md",
    }

    for label, rel_path in corpus.items():
        path = os.path.join(ws, rel_path)
        if os.path.isfile(path):
            blocks = parse_file(path)
            result["files"][label] = {
                "total": len(blocks),
                "active": len(get_active(blocks)),
            }
        else:
            result["files"][label] = {"total": 0, "active": 0}

    # State snapshot metrics
    state_path = os.path.join(ws, "memory", "intel-state.json")
    if os.path.isfile(state_path):
        with open(state_path, "r", encoding="utf-8") as f:
            try:
                state = json.load(f)
                result["metrics"] = state.get("metrics", {})
            except json.JSONDecodeError:
                pass

    return json.dumps(result, indent=2)


@mcp.resource("mind-mem://recall/{query}")
def get_recall(query: str) -> str:
    """Search memory using ranked recall (FTS5 or BM25 scan)."""
    ws = _workspace()
    if os.path.isfile(fts_db_path(ws)):
        results = fts_query(ws, query, limit=10)
    else:
        results = recall_engine(ws, query, limit=10)
    return json.dumps(results, indent=2, default=str)


@mcp.resource("mind-mem://ledger")
def get_ledger() -> str:
    """Shared fact ledger for multi-agent memory propagation."""
    return _read_file("shared/intelligence/LEDGER.md")


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
        warnings.append(
            f"Query exceeded timeout ({round(recall_elapsed, 1)}s > {timeout_seconds}s). Results may be incomplete."
        )

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


@mcp.tool
@mcp_tool_observe
def verify_merkle(block_id: str, content_hash: str) -> str:
    """Verify a block's Merkle inclusion against the live tree.

    Builds the Merkle tree from the current block index and returns a
    JSON envelope with the proof and an ``ok`` flag indicating whether
    the caller-supplied content hash reproduces the stored root.

    Args:
        block_id: Identifier of the block to prove.
        content_hash: Claimed SHA-256 (or SHA3-512) of the block's
            canonical content. The exact digest algorithm is irrelevant
            to the tree — the caller must match whatever went in.

    Returns:
        JSON with ``ok`` (bool), ``root`` (hex), ``proof`` (list of
        sibling/direction pairs), and ``error`` when verification fails.
    """
    from .merkle_tree import MerkleTree

    ws = _workspace()
    ws_err = _check_workspace(ws)
    if ws_err:
        return ws_err

    if not isinstance(block_id, str) or not block_id.strip():
        return json.dumps({"ok": False, "error": "block_id must be a non-empty string"})
    if not isinstance(content_hash, str) or not content_hash.strip():
        return json.dumps({"ok": False, "error": "content_hash must be a non-empty string"})

    # Collect (block_id, content_hash) tuples from the FTS index so the
    # Merkle tree mirrors the live retrieval set. If FTS isn't built the
    # tool returns a deterministic "no blocks" failure rather than
    # silently claiming success.
    try:
        from .sqlite_index import merkle_leaves

        leaves = merkle_leaves(ws)
    except (ImportError, AttributeError):
        leaves = []

    if not leaves:
        return json.dumps(
            {
                "ok": False,
                "error": "no block index available — run 'mind-mem-scan' first",
            }
        )

    tree = MerkleTree()
    tree.build(leaves)
    try:
        proof = tree.get_proof(block_id)
    except KeyError:
        return json.dumps(
            {
                "ok": False,
                "error": f"block_id not in tree: {block_id!r}",
                "root": tree.root_hash,
            }
        )

    ok = tree.verify_proof(block_id, content_hash, proof, tree.root_hash)
    # Proof format: each item is a [sibling_hash, direction] pair where
    # direction is "left" or "right" indicating the side of the sibling
    # relative to the current node. Third-party verifiers should use
    # proof_format_version=1 to opt into breaking changes later.
    return json.dumps(
        {
            "ok": bool(ok),
            "root": tree.root_hash,
            "proof": proof,
            "proof_format_version": 1,
            "block_id": block_id,
            "_schema_version": "1.0",
        },
        indent=2,
    )


def _signal_store_path(ws: str) -> str:
    return os.path.join(ws, "memory", "interaction_signals.jsonl")


def _kg_path(ws: str) -> str:
    return os.path.join(ws, "memory", "knowledge_graph.db")


_CORE_REGISTRY: Any = None


def _core_registry() -> Any:
    global _CORE_REGISTRY
    if _CORE_REGISTRY is None:
        from .context_core import CoreRegistry

        _CORE_REGISTRY = CoreRegistry()
    return _CORE_REGISTRY


def _core_dir(ws: str) -> str:
    path = os.path.join(ws, "memory", "cores")
    os.makedirs(path, exist_ok=True)
    return path


@mcp.tool
@mcp_tool_observe
def build_core(namespace: str, version: str, filename: str = "") -> str:
    """Build a .mmcore bundle from the active workspace's blocks.

    Snapshots the current block index + knowledge graph into a portable
    `.mmcore` archive. Downstream callers can load it into another
    mind-mem instance via `load_core`.

    Args:
        namespace: Identifier used to prefix blocks when loaded.
        version: Caller-facing semver recorded in the manifest.
        filename: Optional output filename (defaults to
            ``<namespace>-<version>.mmcore`` under ``memory/cores/``).

    Returns:
        JSON envelope with the bundle path and manifest summary.
    """
    from .context_core import build_core as _build_core
    from .knowledge_graph import KnowledgeGraph

    ws = _workspace()
    ws_err = _check_workspace(ws)
    if ws_err:
        return ws_err

    if not isinstance(namespace, str) or not namespace.strip():
        return json.dumps({"error": "namespace must be a non-empty string"})
    if not isinstance(version, str) or not version.strip():
        return json.dumps({"error": "version must be a non-empty string"})

    # Assemble blocks from the FTS index (best-effort — empty on fresh ws).
    blocks: list[dict] = []
    try:
        from .sqlite_index import merkle_leaves as _leaves

        for bid, content_hash in _leaves(ws):
            blocks.append({"_id": bid, "content_hash": content_hash})
    except (ImportError, AttributeError):
        pass

    edges: list[dict] = []
    kg_file = _kg_path(ws)
    if os.path.isfile(kg_file):
        kg = KnowledgeGraph(kg_file)
        try:
            for e in kg.edges_from("__all__") if False else []:
                edges.append(e.as_dict())
            # Pull all edges regardless of subject: walk the DB directly.
            rows = kg._conn.execute(
                "SELECT subject, predicate, object, source_block_id, confidence, "
                "valid_from, valid_until, metadata FROM edges"
            ).fetchall()
            for row in rows:
                edges.append(
                    {
                        "subject": row["subject"],
                        "predicate": row["predicate"],
                        "object": row["object"],
                        "source_block_id": row["source_block_id"],
                        "confidence": row["confidence"],
                        "valid_from": row["valid_from"],
                        "valid_until": row["valid_until"],
                        "metadata": row["metadata"],
                    }
                )
        finally:
            kg.close()

    out_name = filename.strip() or f"{namespace.strip()}-{version.strip()}.mmcore"
    if any(ch in out_name for ch in "/\\"):
        return json.dumps({"error": "filename must not contain path separators"})
    out_path = os.path.join(_core_dir(ws), out_name)

    try:
        manifest = _build_core(
            out_path,
            namespace=namespace,
            version=version,
            blocks=blocks,
            edges=edges,
        )
    except ValueError as exc:
        return json.dumps({"error": str(exc)})

    return json.dumps(
        {"path": out_path, "manifest": manifest.as_dict(), "_schema_version": "1.0"},
        indent=2,
    )


@mcp.tool
@mcp_tool_observe
def load_core(filename: str, verify: bool = True) -> str:
    """Load a .mmcore bundle from the workspace's cores/ directory.

    Args:
        filename: Core filename relative to ``memory/cores/``.
        verify: Recompute and compare the content hash (default True).
    """
    from .context_core import CoreLoadError

    ws = _workspace()
    ws_err = _check_workspace(ws)
    if ws_err:
        return ws_err

    if not isinstance(filename, str) or not filename.strip():
        return json.dumps({"error": "filename must be a non-empty string"})
    if any(ch in filename for ch in "/\\"):
        return json.dumps({"error": "filename must not contain path separators"})

    path = os.path.join(_core_dir(ws), filename.strip())
    try:
        loaded = _core_registry().load(path, verify=verify)
    except CoreLoadError as exc:
        return json.dumps({"error": str(exc)})
    except RuntimeError as exc:
        return json.dumps({"error": str(exc)})
    return json.dumps(
        {
            "loaded": True,
            "namespace": loaded.manifest.namespace,
            "blocks": loaded.block_count(),
            "edges": loaded.edge_count(),
            "content_hash": loaded.manifest.content_hash,
            "_schema_version": "1.0",
        },
        indent=2,
    )


@mcp.tool
@mcp_tool_observe
def unload_core(namespace: str) -> str:
    """Unload a previously-loaded core by namespace."""
    ws = _workspace()
    ws_err = _check_workspace(ws)
    if ws_err:
        return ws_err
    if not isinstance(namespace, str) or not namespace.strip():
        return json.dumps({"error": "namespace must be a non-empty string"})
    ok = _core_registry().unload(namespace.strip())
    return json.dumps({"unloaded": bool(ok)})


@mcp.tool
@mcp_tool_observe
def list_cores() -> str:
    """List every currently-loaded .mmcore bundle (namespace + stats)."""
    ws = _workspace()
    ws_err = _check_workspace(ws)
    if ws_err:
        return ws_err
    return json.dumps(
        {"cores": _core_registry().stats(), "_schema_version": "1.0"}, indent=2
    )


@mcp.tool
@mcp_tool_observe
def plan_consolidation(
    importance_threshold: float = 0.25,
    stale_days: int = 14,
    archive_after_days: int = 60,
    grace_days: int = 30,
) -> str:
    """Dry-run the cognitive forgetting cycle.

    Reports which blocks would be MARKED, ARCHIVED, or FORGOTTEN based
    on the current block_meta telemetry. No state is mutated — this is
    purely a preview so callers can inspect the plan before approving.

    Args:
        importance_threshold: Blocks below this importance (combined
            with age) are candidates for marking.
        stale_days: Days of inactivity before a low-importance block
            is flagged.
        archive_after_days: Days after being MERGED before a block
            moves to cold storage.
        grace_days: Days in ARCHIVED state before a block is eligible
            for permanent forget. Reversible until this window closes.
    """
    from .cognitive_forget import (
        BlockCognition,
        BlockLifecycle,
        ConsolidationConfig,
        plan_consolidation as _plan,
    )

    ws = _workspace()
    ws_err = _check_workspace(ws)
    if ws_err:
        return ws_err

    try:
        cfg = ConsolidationConfig(
            importance_threshold=float(importance_threshold),
            stale_days=int(stale_days),
            archive_after_days=int(archive_after_days),
            grace_days=int(grace_days),
        )
    except ValueError as exc:
        return json.dumps({"error": str(exc)})

    # Pull block telemetry from block_meta if present.
    import sqlite3 as _sqlite3

    db_path = os.path.join(ws, ".sqlite_index", "index.db")
    blocks: list[BlockCognition] = []
    if os.path.isfile(db_path):
        conn = _sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30.0)
        conn.row_factory = _sqlite3.Row
        try:
            rows = conn.execute(
                """
                SELECT b.id AS block_id,
                       COALESCE(bm.importance, 0.5) AS importance,
                       bm.last_accessed AS last_accessed,
                       COALESCE(bm.access_count, 0) AS access_count
                FROM blocks b
                LEFT JOIN block_meta bm ON bm.id = b.id
                """
            ).fetchall()
            for r in rows:
                try:
                    blocks.append(
                        BlockCognition(
                            block_id=r["block_id"],
                            importance=float(r["importance"]),
                            last_accessed=r["last_accessed"],
                            access_count=int(r["access_count"]),
                            created_at=None,
                            size_bytes=0,
                            lifecycle=BlockLifecycle.ACTIVE,
                        )
                    )
                except ValueError:
                    continue
        finally:
            conn.close()

    plan = _plan(blocks, config=cfg)
    return json.dumps(
        {
            "config": {
                "importance_threshold": cfg.importance_threshold,
                "stale_days": cfg.stale_days,
                "archive_after_days": cfg.archive_after_days,
                "grace_days": cfg.grace_days,
            },
            "plan": plan.as_dict(),
            "_schema_version": "1.0",
        },
        indent=2,
    )


@mcp.tool
@mcp_tool_observe
def ontology_load(spec: str, make_active: bool = False) -> str:
    """Load an ontology from an inline JSON spec.

    The spec must be a JSON object with ``version`` and ``types``
    fields (see ``Ontology.from_dict``). Pass ``make_active=True`` to
    promote the loaded ontology to the default used by
    ``ontology_validate``.
    """
    from .ontology import Ontology

    ws = _workspace()
    ws_err = _check_workspace(ws)
    if ws_err:
        return ws_err

    if not isinstance(spec, str) or not spec.strip():
        return json.dumps({"error": "spec must be a non-empty JSON string"})
    if len(spec) > 1_048_576:
        return json.dumps({"error": "spec must be ≤1 MiB"})
    try:
        data = json.loads(spec)
    except json.JSONDecodeError as exc:
        return json.dumps({"error": f"spec is not valid JSON: {exc}"})
    if not isinstance(data, dict):
        return json.dumps({"error": "spec must decode to a JSON object"})
    try:
        ont = Ontology.from_dict(data)
    except (ValueError, KeyError, TypeError) as exc:
        return json.dumps({"error": f"invalid ontology: {exc}"})

    _ontology_registry().load(ont, make_active=bool(make_active))
    return json.dumps(
        {
            "loaded": True,
            "version": ont.version,
            "types": ont.type_names(),
            "active": bool(make_active)
            or _ontology_registry().active().version == ont.version,
            "_schema_version": "1.0",
        },
        indent=2,
    )


@mcp.tool
@mcp_tool_observe
def ontology_validate(block: str, type_name: str, strict: bool = True) -> str:
    """Validate *block* (JSON object) against the active ontology.

    Returns ``{valid: bool, errors: [str]}`` — an empty ``errors``
    list means the block satisfies the type's effective schema
    (including inherited parent properties).
    """
    ws = _workspace()
    ws_err = _check_workspace(ws)
    if ws_err:
        return ws_err

    if not isinstance(block, str) or not block.strip():
        return json.dumps({"error": "block must be a non-empty JSON string"})
    if len(block) > 1_048_576:
        return json.dumps({"error": "block must be ≤1 MiB"})
    try:
        block_obj = json.loads(block)
    except json.JSONDecodeError as exc:
        return json.dumps({"error": f"block is not valid JSON: {exc}"})
    if not isinstance(block_obj, dict):
        return json.dumps({"error": "block must decode to a JSON object"})
    if not isinstance(type_name, str) or not type_name.strip():
        return json.dumps({"error": "type_name must be a non-empty string"})

    ont = _ontology_registry().active()
    if ont is None:
        return json.dumps({"error": "no active ontology; call ontology_load first"})
    errors = ont.validate(type_name, block_obj, strict=bool(strict))
    return json.dumps(
        {
            "valid": len(errors) == 0,
            "errors": errors,
            "type": type_name,
            "ontology_version": ont.version,
            "_schema_version": "1.0",
        },
        indent=2,
    )


@mcp.tool
@mcp_tool_observe
def propagate_staleness(seed_block_ids: str, max_hops: int = 3) -> str:
    """Diffuse staleness outward from seed blocks over the xref graph.

    Args:
        seed_block_ids: Comma-separated block ids that are known to be
            stale (just superseded / contradicted).
        max_hops: Cap on traversal depth. Default 3 matches the
            roadmap's ``0.9 → 0.5 → 0.2`` decay schedule.

    Returns:
        JSON map of block_id → staleness score in [0, 1].
    """
    import sqlite3 as _sqlite3

    from .staleness import propagate_staleness as _propagate

    ws = _workspace()
    ws_err = _check_workspace(ws)
    if ws_err:
        return ws_err

    if not isinstance(seed_block_ids, str) or not seed_block_ids.strip():
        return json.dumps({"error": "seed_block_ids must be a non-empty string"})
    seeds = [
        bid.strip() for bid in seed_block_ids.split(",") if bid.strip()
    ][:64]
    if not seeds:
        return json.dumps({"error": "no seed block ids supplied"})
    if not (0 <= max_hops <= 8):
        return json.dumps({"error": "max_hops must be in [0, 8]"})

    # Pull xref graph from the FTS index when available.
    adjacency: dict[str, list[str]] = {}
    db_path = os.path.join(ws, ".sqlite_index", "index.db")
    if os.path.isfile(db_path):
        conn = _sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30.0)
        conn.row_factory = _sqlite3.Row
        try:
            rows = conn.execute("SELECT src, dst FROM xref_edges").fetchall()
            for r in rows:
                adjacency.setdefault(r["src"], []).append(r["dst"])
                adjacency.setdefault(r["dst"], []).append(r["src"])
        finally:
            conn.close()

    plan = _propagate(seeds, adjacency, max_hops=max_hops)
    return json.dumps(
        {
            "seed": list(plan.seed),
            "max_hops": plan.max_hops,
            "scores": plan.scores,
            "_schema_version": "1.0",
        },
        indent=2,
    )


@mcp.tool
@mcp_tool_observe
def project_profile(name: str = "", top_k: int = 10) -> str:
    """Auto-generate a structured project intelligence profile.

    Aggregates block types, top concepts, top files, entities, and
    recent-activity counts from the active workspace. Intended for
    session-start injection so an agent begins with project context.
    """
    import sqlite3 as _sqlite3

    from .project_profile import build_profile

    ws = _workspace()
    ws_err = _check_workspace(ws)
    if ws_err:
        return ws_err

    if not (0 <= top_k <= 100):
        return json.dumps({"error": "top_k must be in [0, 100]"})
    project_name = name.strip() or os.path.basename(os.path.realpath(ws))

    blocks: list[dict] = []
    db_path = os.path.join(ws, ".sqlite_index", "index.db")
    if os.path.isfile(db_path):
        conn = _sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30.0)
        conn.row_factory = _sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT id, type, file, date, json_blob FROM blocks LIMIT 50000"
            ).fetchall()
            for r in rows:
                entry: dict[str, Any] = {
                    "_id": r["id"],
                    "type": r["type"],
                    "file": r["file"],
                    "date": r["date"],
                }
                try:
                    raw = json.loads(r["json_blob"] or "{}")
                except (json.JSONDecodeError, TypeError, ValueError):
                    raw = {}
                if isinstance(raw, dict):
                    for key in ("text", "statement", "excerpt", "content"):
                        if key in raw:
                            entry[key] = raw[key]
                    for key in ("entities", "mentions"):
                        if key in raw:
                            entry[key] = raw[key]
                blocks.append(entry)
        finally:
            conn.close()

    profile = build_profile(blocks, name=project_name, top_k=top_k)
    return json.dumps({**profile.as_dict(), "_schema_version": "1.0"}, indent=2)


@mcp.tool
@mcp_tool_observe
def agent_inject(query: str, agent: str = "generic", limit: int = 10) -> str:
    """Render a context snippet in the target agent's expected format.

    Wraps :func:`recall` and :class:`AgentFormatter` so MCP-capable
    callers can pre-render context for non-MCP siblings (codex, gemini
    CLI, Cursor, Windsurf, Aider).

    Args:
        query: Search query.
        agent: One of claude-code / codex / gemini / cursor / windsurf
            / aider / generic.
        limit: Maximum result count fed into the formatter.
    """
    from .agent_bridge import AgentFormatter, KNOWN_AGENTS, UnknownAgentError

    ws = _workspace()
    ws_err = _check_workspace(ws)
    if ws_err:
        return ws_err

    if not isinstance(query, str) or not query.strip():
        return json.dumps({"error": "query must be a non-empty string"})
    if agent not in KNOWN_AGENTS:
        return json.dumps(
            {
                "error": f"unknown agent: {agent!r}",
                "valid": list(KNOWN_AGENTS),
            }
        )
    if not (1 <= limit <= 100):
        return json.dumps({"error": "limit must be in [1, 100]"})

    raw = json.loads(_recall_impl(query, limit=limit))
    if isinstance(raw, dict):
        results = raw.get("results", []) or []
    elif isinstance(raw, list):
        results = raw
    else:
        results = []

    fmt = AgentFormatter(max_blocks=limit)
    try:
        text = fmt.inject(agent, query, results)
    except UnknownAgentError as exc:
        return json.dumps({"error": str(exc)})
    return json.dumps(
        {"agent": agent, "query": query, "snippet": text, "_schema_version": "1.0"},
        indent=2,
    )


def _vault_allowlist() -> list[str]:
    """Return the configured vault-root allowlist.

    Set ``MIND_MEM_VAULT_ALLOWLIST`` to a ``:``-separated list of
    absolute directories. When set, every vault MCP tool refuses
    requests targeting paths outside the list. Empty/unset = allow
    any path (legacy behaviour; recommended only for local dev).
    """
    raw = os.environ.get("MIND_MEM_VAULT_ALLOWLIST", "").strip()
    if not raw:
        return []
    sep = ";" if ";" in raw else ":"
    return [
        os.path.realpath(p.strip())
        for p in raw.split(sep)
        if p.strip()
    ]


def _vault_root_allowed(vault_root: str) -> tuple[bool, str]:
    """Check vault_root against the allowlist. (ok, reason)."""
    allow = _vault_allowlist()
    if not allow:
        return True, ""
    target = os.path.realpath(vault_root.strip())
    for root in allow:
        try:
            common = os.path.commonpath([target, root])
        except ValueError:
            continue
        if common == root:
            return True, ""
    return False, (
        f"vault_root {vault_root!r} is outside MIND_MEM_VAULT_ALLOWLIST"
    )


@mcp.tool
@mcp_tool_observe
def vault_scan(vault_root: str, sync_dirs: str = "") -> str:
    """Walk an Obsidian-style vault and return parsed VaultBlocks (JSON).

    Args:
        vault_root: Absolute path to the vault root.
        sync_dirs: Optional comma-separated list of subdirectories to
            scan. Empty = full vault (minus default excludes).
    """
    from .agent_bridge import VaultBridge

    if not isinstance(vault_root, str) or not vault_root.strip():
        return json.dumps({"error": "vault_root must be a non-empty string"})
    ok, reason = _vault_root_allowed(vault_root)
    if not ok:
        return json.dumps({"error": reason})
    dirs = [d.strip() for d in sync_dirs.split(",") if d.strip()] or None
    try:
        bridge = VaultBridge(vault_root=vault_root.strip())
        blocks = bridge.scan(sync_dirs=dirs)
    except (FileNotFoundError, ValueError) as exc:
        return json.dumps({"error": str(exc)})
    return json.dumps(
        {
            "vault_root": vault_root,
            "blocks": [b.as_dict() for b in blocks],
            "_schema_version": "1.0",
        },
        indent=2,
    )


@mcp.tool
@mcp_tool_observe
def vault_sync(
    vault_root: str,
    block_id: str,
    relative_path: str,
    body: str,
    block_type: str = "note",
    title: str = "",
    overwrite: bool = False,
) -> str:
    """Write a single block back into a vault at a relative path.

    Refuses to write outside the vault root or to overwrite existing
    files unless ``overwrite`` is true.
    """
    from .agent_bridge import VaultBlock, VaultBridge

    for arg, label in (
        (vault_root, "vault_root"),
        (block_id, "block_id"),
        (relative_path, "relative_path"),
    ):
        if not isinstance(arg, str) or not arg.strip():
            return json.dumps({"error": f"{label} must be a non-empty string"})
    ok, reason = _vault_root_allowed(vault_root)
    if not ok:
        return json.dumps({"error": reason})
    try:
        bridge = VaultBridge(vault_root=vault_root.strip())
        target = bridge.write(
            VaultBlock(
                relative_path=relative_path.strip(),
                block_id=block_id.strip(),
                block_type=block_type.strip() or "note",
                title=title.strip() or block_id.strip(),
                body=body,
            ),
            overwrite=bool(overwrite),
        )
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        return json.dumps({"error": str(exc)})
    return json.dumps(
        {"written": target, "_schema_version": "1.0"}, indent=2
    )


@mcp.tool
@mcp_tool_observe
def stream_status() -> str:
    """Current change-stream publish / delivery / drop counters."""
    ws = _workspace()
    ws_err = _check_workspace(ws)
    if ws_err:
        return ws_err
    return json.dumps(
        {**_change_stream().stats().as_dict(), "_schema_version": "1.0"},
        indent=2,
    )


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


@mcp.tool
@mcp_tool_observe
def graph_add_edge(
    subject: str,
    predicate: str,
    object: str,
    source_block_id: str,
    confidence: float = 1.0,
) -> str:
    """Record a typed relationship in the knowledge graph.

    Args:
        subject / object: Entity names (aliases resolved automatically).
        predicate: One of authored_by, depends_on, contradicts,
            supersedes, part_of, mentioned_in, related_to.
        source_block_id: The block this relationship was extracted from.
        confidence: Extractor confidence in [0, 1].

    Returns:
        JSON with the canonicalised edge or an error envelope.
    """
    from .knowledge_graph import KnowledgeGraph, Predicate

    ws = _workspace()
    ws_err = _check_workspace(ws)
    if ws_err:
        return ws_err

    if not isinstance(subject, str) or not subject.strip():
        return json.dumps({"error": "subject must be a non-empty string"})
    if not isinstance(object, str) or not object.strip():
        return json.dumps({"error": "object must be a non-empty string"})
    if len(subject) > 512 or len(object) > 512:
        return json.dumps({"error": "entity names must be ≤512 chars"})
    try:
        pred = Predicate.from_str(predicate)
    except ValueError as exc:
        return json.dumps({"error": str(exc)})

    kg = KnowledgeGraph(_kg_path(ws))
    try:
        edge = kg.add_edge(
            subject, pred, object,
            source_block_id=source_block_id,
            confidence=float(confidence),
        )
    except ValueError as exc:
        return json.dumps({"error": str(exc)})
    finally:
        kg.close()
    return json.dumps({**edge.as_dict(), "_schema_version": "1.0"}, indent=2)


@mcp.tool
@mcp_tool_observe
def graph_query(
    entity: str,
    depth: int = 1,
    predicate: str = "",
    direction: str = "outgoing",
    limit: int = 64,
) -> str:
    """N-hop traversal from *entity*.

    Args:
        entity: Starting entity (any alias — resolved automatically).
        depth: Maximum hops (1-8).
        predicate: Optional predicate filter (blank = all).
        direction: ``outgoing`` / ``incoming`` / ``both``.
        limit: Maximum neighbours returned (1-256).

    Returns:
        JSON list of neighbours with hop count + predicate path.
    """
    from .knowledge_graph import KnowledgeGraph, Predicate

    ws = _workspace()
    ws_err = _check_workspace(ws)
    if ws_err:
        return ws_err

    if not isinstance(entity, str) or not entity.strip():
        return json.dumps({"error": "entity must be a non-empty string"})
    if len(entity) > 512:
        return json.dumps({"error": "entity name must be ≤512 chars"})
    if not (1 <= depth <= 8):
        return json.dumps({"error": "depth must be in [1, 8]"})
    if not (1 <= limit <= 256):
        return json.dumps({"error": "limit must be in [1, 256]"})
    pred_obj = None
    if predicate.strip():
        try:
            pred_obj = Predicate.from_str(predicate)
        except ValueError as exc:
            return json.dumps({"error": str(exc)})
    if direction not in {"outgoing", "incoming", "both"}:
        return json.dumps(
            {"error": "direction must be 'outgoing' / 'incoming' / 'both'"}
        )

    kg = KnowledgeGraph(_kg_path(ws))
    try:
        neighbours = kg.neighbors(
            entity,
            depth=depth,
            predicate=pred_obj,
            direction=direction,
            max_results=limit,
        )
    except ValueError as exc:
        return json.dumps({"error": str(exc)})
    finally:
        kg.close()
    return json.dumps(
        {
            "entity": entity,
            "depth": depth,
            "direction": direction,
            "neighbors": neighbours,
            "_schema_version": "1.0",
        },
        indent=2,
    )


@mcp.tool
@mcp_tool_observe
def graph_stats() -> str:
    """Aggregated knowledge-graph stats for the active workspace."""
    from .knowledge_graph import KnowledgeGraph

    ws = _workspace()
    ws_err = _check_workspace(ws)
    if ws_err:
        return ws_err

    kg_file = _kg_path(ws)
    if not os.path.isfile(kg_file):
        return json.dumps(
            {"entities": 0, "edges": 0, "predicates": {}, "orphan_entities": 0,
             "_schema_version": "1.0"},
            indent=2,
        )
    kg = KnowledgeGraph(kg_file)
    try:
        stats = kg.stats().as_dict()
    finally:
        kg.close()
    return json.dumps({**stats, "_schema_version": "1.0"}, indent=2)


@mcp.tool
@mcp_tool_observe
def observe_signal(
    session_id: str,
    previous_query: str,
    new_query: str,
    previous_results: str = "",
) -> str:
    """Capture a re-query / refinement / correction feedback signal.

    Called by the recall pipeline (or an external agent) when the user
    rephrases or retries a query within the same session. The classifier
    decides whether the pair is a genuine feedback event and, if so,
    appends a structured :class:`Signal` to the append-only store.

    Args:
        session_id: Conversation / agent session identifier.
        previous_query: The earlier query the user issued.
        new_query: The follow-up query.
        previous_results: Comma-separated block ids returned by the
            earlier recall. Used by the A/B eval harness to score future
            retrieval changes against what the user already saw.

    Returns:
        JSON with either a captured signal (``signal_id``, ``type``,
        ``similarity``) or ``{"captured": false}`` when the two queries
        are unrelated and no feedback was recorded.
    """
    from .interaction_signals import SignalStore

    ws = _workspace()
    ws_err = _check_workspace(ws)
    if ws_err:
        return ws_err

    if not isinstance(session_id, str) or not session_id.strip():
        return json.dumps({"error": "session_id must be a non-empty string"})
    if not isinstance(previous_query, str) or not isinstance(new_query, str):
        return json.dumps({"error": "queries must be strings"})
    if len(previous_query) > 8192 or len(new_query) > 8192:
        return json.dumps({"error": "queries must be ≤8192 chars"})

    prev_ids: list[str] = []
    if previous_results.strip():
        prev_ids = [
            bid.strip()
            for bid in previous_results.split(",")
            if bid.strip()
        ][:64]

    store = SignalStore(_signal_store_path(ws))
    result = store.observe_pair(
        session_id=session_id,
        previous_query=previous_query,
        new_query=new_query,
        previous_results=prev_ids,
    )
    if result is None:
        return json.dumps(
            {"captured": False, "reason": "unrelated or duplicate"}
        )
    return json.dumps(
        {
            "captured": True,
            "signal_id": result.signal_id,
            "signal_type": result.signal_type.value,
            "similarity": round(result.similarity, 4),
            "_schema_version": "1.0",
        }
    )


@mcp.tool
@mcp_tool_observe
def signal_stats() -> str:
    """Return aggregated interaction-signal counts for the workspace.

    Useful for operators monitoring how often users re-query / correct /
    refine — a spike in corrections usually flags a regression in the
    underlying recall pipeline.
    """
    from .interaction_signals import SignalStore

    ws = _workspace()
    ws_err = _check_workspace(ws)
    if ws_err:
        return ws_err

    store = SignalStore(_signal_store_path(ws))
    return json.dumps(
        {**store.stats().as_dict(), "_schema_version": "1.0"},
        indent=2,
    )


@mcp.tool
@mcp_tool_observe
def mind_mem_verify(snapshot: str = "") -> str:
    """Run the standalone `mind-mem-verify` CLI against the current workspace.

    Exposes the external verifier via MCP so agents can run it without
    shelling out. ``snapshot`` is optional; when set it points to a
    snapshot directory **relative to the workspace** whose manifest
    will be checked against the live chain + Merkle tree. Absolute
    paths or `..` traversal are rejected so an MCP caller cannot ask
    the verifier to read outside the workspace.
    """
    from .verify_cli import verify_workspace

    ws = _workspace()
    ws_err = _check_workspace(ws)
    if ws_err:
        return ws_err

    snap = snapshot.strip() or None
    if snap is not None:
        if len(snap) > 512:
            return json.dumps({"error": "snapshot path too long"})
        if os.path.isabs(snap) or snap.startswith(("/", "\\")):
            return json.dumps(
                {"error": "snapshot must be a workspace-relative path"}
            )
        # Normalise and double-check: after joining with ws, the result
        # must still live under ws. verify_workspace also enforces this
        # but failing early in the MCP layer yields a cleaner error.
        resolved = os.path.realpath(os.path.join(ws, snap))
        if not resolved.startswith(os.path.realpath(ws) + os.sep):
            return json.dumps(
                {"error": f"snapshot path escapes workspace: {snap!r}"}
            )
    report = verify_workspace(ws, snapshot=snap)
    envelope = report.as_dict()
    envelope["_schema_version"] = "1.0"
    return json.dumps(envelope, indent=2)


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
    _MAX_TOKENS = 16     # per enum, number of axes / weight pairs
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


@mcp.tool
@mcp_tool_observe
def propose_update(
    block_type: str,
    statement: str,
    rationale: str = "",
    tags: str = "",
    confidence: str = "medium",
) -> str:
    """Propose a new decision or task. Writes to SIGNALS.md for human review.

    SAFETY: This never writes directly to DECISIONS.md or TASKS.md.
    All proposals must go through the apply engine (/apply) for review.

    Args:
        block_type: Type of block — "decision" or "task".
        statement: The decision statement or task description.
        rationale: Why this decision/task is needed.
        tags: Comma-separated tags (e.g., "database, infrastructure").
        confidence: Signal confidence — "high", "medium", or "low".

    Returns:
        Confirmation with signal ID and status.
    """
    ws = _workspace()

    if block_type not in ("decision", "task"):
        return json.dumps({"error": f"block_type must be 'decision' or 'task', got '{block_type}'"})

    from datetime import datetime

    from mind_mem.capture import CONFIDENCE_TO_PRIORITY, append_signals

    today = datetime.now().strftime("%Y-%m-%d")
    priority = CONFIDENCE_TO_PRIORITY.get(confidence, "P2")

    # Cap statement length to prevent oversized signals
    statement = statement[:500]

    signal = {
        "line": 0,
        "type": block_type,
        "text": statement,
        "pattern": "mcp_propose_update",
        "confidence": confidence,
        "priority": priority,
        "structure": {
            "subject": " ".join(statement.split()[:3]) if statement else "",
            "tags": [t.strip() for t in tags.split(",") if t.strip()],
        },
    }
    if rationale:
        signal["structure"]["rationale"] = rationale  # type: ignore[index]

    written = append_signals(ws, [signal], today)

    metrics.inc("mcp_proposals")
    _log.info("mcp_propose", block_type=block_type, confidence=confidence, written=written)

    return json.dumps(
        {
            "_schema_version": MCP_SCHEMA_VERSION,
            "status": "proposed",
            "written": written,
            "location": "intelligence/SIGNALS.md",
            "next_step": (
                "Run /apply or `python3 maintenance/apply_engine.py` to review and promote to source of truth."
            ),
            "safety": "This signal is in SIGNALS.md only. It has NOT been written to DECISIONS.md or TASKS.md.",
        },
        indent=2,
    )


@mcp.tool
@mcp_tool_observe
def scan() -> str:
    """Run integrity scan — contradictions, drift, dead decisions, impact graph.

    Returns:
        JSON summary of scan results.
    """
    ws = _workspace()
    ws_err = _check_workspace(ws)
    if ws_err:
        return ws_err

    checks: dict[str, Any] = {}

    # Parse decisions
    decisions_path = os.path.join(ws, "decisions", "DECISIONS.md")
    if os.path.isfile(decisions_path):
        blocks = parse_file(decisions_path)
        active = get_active(blocks)
        checks["decisions"] = {
            "total": len(blocks),
            "active": len(active),
        }
    else:
        checks["decisions"] = {"total": 0, "active": 0}

    # Check contradictions — report both raw entries and resolvable ones
    contra_path = os.path.join(ws, "intelligence", "CONTRADICTIONS.md")
    raw_count = 0
    if os.path.isfile(contra_path):
        raw_count = len(parse_file(contra_path))
    try:
        from mind_mem.conflict_resolver import resolve_contradictions

        resolutions = resolve_contradictions(ws)
        checks["contradictions"] = {
            "raw": raw_count,
            "resolvable": len(resolutions),
        }
    except (ImportError, OSError, ValueError) as exc:
        _log.warning("scan_contradiction_check_failed", error=str(exc))
        checks["contradictions"] = {"raw": raw_count, "resolvable": 0}

    # Check drift
    drift_path = os.path.join(ws, "intelligence", "DRIFT.md")
    if os.path.isfile(drift_path):
        drifts = parse_file(drift_path)
        checks["drift_items"] = len(drifts)
    else:
        checks["drift_items"] = 0

    # Check signals
    signals_path = os.path.join(ws, "intelligence", "SIGNALS.md")
    if os.path.isfile(signals_path):
        signals = parse_file(signals_path)
        checks["pending_signals"] = len(signals)
    else:
        checks["pending_signals"] = 0

    result: dict[str, Any] = {"_schema_version": MCP_SCHEMA_VERSION, "checks": checks}
    metrics.inc("mcp_scans")
    _log.info("mcp_scan", checks=checks)

    return json.dumps(result, indent=2)


@mcp.tool
@mcp_tool_observe
def list_contradictions() -> str:
    """List detected contradictions with resolution analysis.

    Enriches the raw contradiction list with AutoResolver's
    preference-boosted + side-effect-aware suggestions when the
    workspace has an existing resolver state; falls back to the
    plain list otherwise.

    Returns:
        JSON array of contradictions with strategy recommendations.
    """
    ws = _workspace()

    from mind_mem.conflict_resolver import resolve_contradictions

    resolutions = resolve_contradictions(ws)
    if not resolutions:
        return json.dumps(
            {
                "_schema_version": MCP_SCHEMA_VERSION,
                "status": "clean",
                "contradictions": 0,
                "message": "No contradictions found.",
            }
        )

    # Best-effort AutoResolver enrichment — side effects + preference
    # learning. Never masks the raw resolutions on failure.
    enriched: list[dict] = []
    try:
        from mind_mem.auto_resolver import AutoResolver

        suggestions = AutoResolver(ws).suggest_resolutions()
        by_id = {s.contradiction_id: s for s in suggestions}
        for res in resolutions:
            sug = by_id.get(res.get("contradiction_id"))
            merged = dict(res)
            if sug is not None:
                merged["confidence_score"] = sug.confidence_score
                merged["side_effects"] = list(sug.side_effects)
                merged["preference_boost_applied"] = True
            enriched.append(merged)
    except Exception as exc:  # pragma: no cover — best-effort
        _log.warning("auto_resolver_enrich_failed", error=str(exc))
        enriched = list(resolutions)

    return json.dumps(
        {
            "_schema_version": MCP_SCHEMA_VERSION,
            "status": "contradictions_found",
            "contradictions": len(enriched),
            "resolutions": enriched,
        },
        indent=2,
        default=str,
    )


def _encryption_passphrase() -> str | None:
    """Fetch the at-rest encryption passphrase from env. None = disabled."""
    raw = os.environ.get("MIND_MEM_ENCRYPTION_PASSPHRASE", "").strip()
    return raw or None


@mcp.tool
@mcp_tool_observe
def encrypt_file(file_path: str) -> str:
    """Encrypt a single workspace file at rest.

    Requires ``MIND_MEM_ENCRYPTION_PASSPHRASE`` to be set in the
    server environment. Files already encrypted are no-ops.
    Admin-scope tool — never exposed to user tokens.

    Args:
        file_path: Absolute path to the plaintext file.

    Returns:
        JSON status envelope.
    """
    passphrase = _encryption_passphrase()
    if not passphrase:
        return json.dumps(
            {
                "_schema_version": MCP_SCHEMA_VERSION,
                "error": "MIND_MEM_ENCRYPTION_PASSPHRASE is not set",
            }
        )
    if not isinstance(file_path, str) or not file_path.strip():
        return json.dumps({"error": "file_path must be a non-empty string"})
    ws = _workspace()
    try:
        safe_path = _safe_vault_path(ws, file_path)
    except Exception as exc:
        return json.dumps({"error": f"path rejected: {exc}"})
    try:
        from mind_mem.encryption import EncryptionManager

        EncryptionManager(ws, passphrase).encrypt_file(safe_path)
    except Exception as exc:
        return json.dumps({"error": f"encrypt failed: {exc}"})
    return json.dumps(
        {
            "_schema_version": MCP_SCHEMA_VERSION,
            "encrypted": True,
            "path": safe_path,
        }
    )


@mcp.tool
@mcp_tool_observe
def decrypt_file(file_path: str) -> str:
    """Return plaintext bytes (base64-encoded) for an encrypted file.

    Does not modify the on-disk ciphertext. Admin-scope tool.

    Args:
        file_path: Absolute path to the encrypted file.

    Returns:
        JSON with ``plaintext_b64`` field on success.
    """
    import base64

    passphrase = _encryption_passphrase()
    if not passphrase:
        return json.dumps(
            {
                "_schema_version": MCP_SCHEMA_VERSION,
                "error": "MIND_MEM_ENCRYPTION_PASSPHRASE is not set",
            }
        )
    if not isinstance(file_path, str) or not file_path.strip():
        return json.dumps({"error": "file_path must be a non-empty string"})
    ws = _workspace()
    try:
        safe_path = _safe_vault_path(ws, file_path)
    except Exception as exc:
        return json.dumps({"error": f"path rejected: {exc}"})
    try:
        from mind_mem.encryption import EncryptionManager

        plaintext = EncryptionManager(ws, passphrase).decrypt_file(safe_path)
    except Exception as exc:
        return json.dumps({"error": f"decrypt failed: {exc}"})
    return json.dumps(
        {
            "_schema_version": MCP_SCHEMA_VERSION,
            "path": safe_path,
            "plaintext_b64": base64.b64encode(plaintext).decode("ascii"),
        }
    )


def _safe_vault_path(ws: str, candidate: str) -> str:
    """Resolve *candidate* against *ws* and reject path-escapes."""
    resolved = os.path.realpath(candidate)
    ws_abs = os.path.realpath(ws)
    try:
        common = os.path.commonpath([resolved, ws_abs])
    except ValueError as exc:
        raise ValueError(f"path escapes workspace: {candidate}") from exc
    if common != ws_abs:
        raise ValueError(f"path escapes workspace: {candidate}")
    if not os.path.isfile(resolved):
        raise FileNotFoundError(resolved)
    return resolved


@mcp.tool
@mcp_tool_observe
def governance_health_bench() -> str:
    """Run the governance health benchmark suite.

    Exercises contradiction detection, audit completeness, drift
    detection, and scalability probes against the current workspace.

    Returns:
        JSON report covering all bench sub-suites and aggregated
        pass/fail counts.
    """
    ws = _workspace()
    try:
        from mind_mem.governance_bench import GovernanceBench

        report = GovernanceBench(ws).run_all()
    except Exception as exc:
        return json.dumps(
            {
                "_schema_version": MCP_SCHEMA_VERSION,
                "error": f"bench run failed: {exc}",
            }
        )
    out = {"_schema_version": MCP_SCHEMA_VERSION, **report}
    return json.dumps(out, indent=2, default=str)


@mcp.tool
@mcp_tool_observe
def approve_apply(proposal_id: str, dry_run: bool = True) -> str:
    """Apply a staged proposal from intelligence/proposed/.

    SAFETY: Defaults to dry_run=True. Set dry_run=False to actually apply.
    Creates a snapshot before applying for rollback support.

    Args:
        proposal_id: The proposal ID (e.g., "P-20260213-002").
        dry_run: If True (default), validate without executing. Set False to apply.

    Returns:
        JSON with apply result, receipt timestamp (for rollback), and status.
    """
    ws = _workspace()

    import re

    if not re.match(r"^P-\d{8}-\d{3}$", proposal_id):
        return json.dumps({"error": f"Invalid proposal ID format: {proposal_id}. Expected P-YYYYMMDD-NNN."})

    import contextlib
    import io

    from mind_mem.apply_engine import apply_proposal, find_proposal
    from mind_mem.contradiction_detector import check_proposal_contradictions

    # Run contradiction check before apply (surfaces conflicts to reviewer)
    contra_report = None
    try:
        proposal, _source = find_proposal(ws, proposal_id)
        if proposal:
            contra_report = check_proposal_contradictions(ws, proposal)
    except Exception as e:
        _log.warning("contradiction_check_failed", error=str(e))

    # Capture stdout from apply_engine (it prints progress)
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        success, message = apply_proposal(ws, proposal_id, dry_run=dry_run)

    log_output = capture.getvalue()

    metrics.inc("mcp_apply_calls")
    _log.info("mcp_approve_apply", proposal_id=proposal_id, dry_run=dry_run, success=success)

    # If apply_engine returned False due to contradictions, reflect that
    blocked_by_contradictions = not success and message == "Blocked: contradictions detected"

    result = {
        "_schema_version": MCP_SCHEMA_VERSION,
        "status": (
            "blocked_contradictions"
            if blocked_by_contradictions
            else "applied"
            if success and not dry_run
            else "dry_run_passed"
            if success
            else "failed"
        ),
        "proposal_id": proposal_id,
        "dry_run": dry_run,
        "success": success,
        "message": message,
        "log": log_output[-2000:] if len(log_output) > 2000 else log_output,
        "next_step": (
            "Resolve contradictions or set contradiction.block_on_detect=false in mind-mem.json."
            if blocked_by_contradictions
            else "Call again with dry_run=False to apply."
            if success and dry_run
            else None
        ),
    }

    # Include contradiction report in output
    if contra_report:
        result["contradictions"] = {
            "summary": contra_report["summary"],
            "has_contradictions": contra_report["has_contradictions"],
            "contradiction_count": contra_report["contradiction_count"],
            "total_conflicts": contra_report["total_conflicts"],
            "conflicts": contra_report["conflicts"],
        }

    return json.dumps(result, indent=2)


@mcp.tool
@mcp_tool_observe
def rollback_proposal(receipt_ts: str) -> str:
    """Rollback an applied proposal using its receipt timestamp.

    Restores workspace from the pre-apply snapshot.

    Args:
        receipt_ts: Receipt timestamp from apply (format: YYYYMMDD-HHMMSS).

    Returns:
        JSON with rollback result and post-rollback check status.
    """
    ws = _workspace()

    import re

    if not re.match(r"^\d{8}-\d{6}$", receipt_ts):
        return json.dumps({"error": f"Invalid receipt timestamp format: {receipt_ts}. Expected YYYYMMDD-HHMMSS."})

    import contextlib
    import io

    from mind_mem.apply_engine import rollback as engine_rollback

    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        success = engine_rollback(ws, receipt_ts)

    log_output = capture.getvalue()

    metrics.inc("mcp_rollbacks")
    _log.info("mcp_rollback", receipt_ts=receipt_ts, success=success)

    return json.dumps(
        {
            "_schema_version": MCP_SCHEMA_VERSION,
            "status": "rolled_back" if success else "rollback_failed",
            "receipt_ts": receipt_ts,
            "success": success,
            "log": log_output[-2000:] if len(log_output) > 2000 else log_output,
        },
        indent=2,
    )


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


@mcp.tool
@mcp_tool_observe
def memory_evolution(block_id: str, action: str = "get") -> str:
    """A-MEM metadata for a block — importance, access patterns, keywords.

    Args:
        block_id: The block ID (e.g., "D-20260213-001").
        action: "get" to read metadata, "update" to recompute importance.

    Returns:
        JSON with block importance, access_count, keywords, connections.
    """
    if not _re_mod.match(r"^[A-Z]+-[a-zA-Z0-9_.-]+$", block_id):
        return json.dumps({"error": f"Invalid block_id format: {block_id}"})
    ws = _workspace()
    db_path = os.path.join(ws, "memory", "block_meta.db")

    try:
        from mind_mem.block_metadata import BlockMetadataManager

        mgr = BlockMetadataManager(db_path)

        if action == "update":
            importance = mgr.update_importance(block_id)
            metrics.inc("mcp_evolution_updates")
            return json.dumps(
                {
                    "_schema_version": MCP_SCHEMA_VERSION,
                    "block_id": block_id,
                    "action": "updated",
                    "importance": round(importance, 4),
                },
                indent=2,
            )
        else:
            importance = mgr.get_importance_boost(block_id)
            co_blocks = mgr.get_co_occurring_blocks(block_id)
            metrics.inc("mcp_evolution_reads")
            return json.dumps(
                {
                    "_schema_version": MCP_SCHEMA_VERSION,
                    "block_id": block_id,
                    "importance": round(importance, 4),
                    "co_occurring_blocks": co_blocks,
                },
                indent=2,
            )

    except ImportError:
        return json.dumps(
            {
                "_schema_version": MCP_SCHEMA_VERSION,
                "error": "memory_evolution requires block_metadata module",
                "block_id": block_id,
            },
            indent=2,
        )
    except sqlite3.OperationalError as exc:
        if _is_db_locked(exc):
            return _sqlite_busy_error()
        raise
    except (OSError, ValueError, KeyError) as exc:
        _log.warning("memory_evolution_failed", block_id=block_id, error=str(exc))
        return json.dumps(
            {
                "_schema_version": MCP_SCHEMA_VERSION,
                "error": "Memory evolution lookup failed. Access history may not be initialized.",
                "block_id": block_id,
            },
            indent=2,
        )


# ---------------------------------------------------------------------------
# Category & Prefetch tools (13-14)
# ---------------------------------------------------------------------------


@mcp.tool
@mcp_tool_observe
def category_summary(topic: str, limit: int = 3) -> str:
    """Returns category summaries relevant to a given topic.

    Uses the category distiller to find and return thematic summary files
    matching the topic. Categories are auto-generated from memory blocks.

    Args:
        topic: Topic or query to find relevant categories for.
        limit: Maximum number of category summaries to return (default: 3).

    Returns:
        Concatenated category summaries with block references.
    """
    ws = _workspace()
    limits = _get_limits(ws)
    try:
        from mind_mem.category_distiller import CategoryDistiller

        extra_cats = _load_extra_categories(ws)
        distiller = CategoryDistiller(extra_categories=extra_cats if extra_cats else None)
        context = distiller.get_category_context(topic, ws, limit=max(1, min(limit, limits["max_category_results"])))
        cats = distiller.get_categories_for_query(topic)
        metrics.inc("mcp_category_summary")
        _log.info("mcp_category_summary", topic=topic, matched_categories=cats[:limit])
        if not context:
            return json.dumps(
                {
                    "_schema_version": MCP_SCHEMA_VERSION,
                    "topic": topic,
                    "status": "no_categories",
                    "hint": "Run reindex to generate category files, or add blocks with matching tags.",
                },
                indent=2,
            )
        return json.dumps(
            {
                "_schema_version": MCP_SCHEMA_VERSION,
                "topic": topic,
                "matched_categories": cats[:limit],
                "content": context,
            },
            indent=2,
        )
    except ImportError:
        return json.dumps(
            {
                "_schema_version": MCP_SCHEMA_VERSION,
                "error": "category_distiller module not available",
            }
        )
    except (OSError, ValueError, KeyError) as exc:
        _log.warning("category_summary_failed", topic=topic, error=str(exc))
        return json.dumps(
            {
                "_schema_version": MCP_SCHEMA_VERSION,
                "error": "Category summary lookup failed",
                "topic": topic,
            },
            indent=2,
        )


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


@mcp.tool
@mcp_tool_observe
def list_mind_kernels() -> str:
    """List available .mind kernel configuration files.

    Kernels define tuning parameters for recall, reranking, RM3 expansion,
    and other pipeline components.

    Returns:
        JSON array of kernel names with their sections and parameters.
    """
    ws = _workspace()
    mind_dir = get_mind_dir(ws)
    all_cfgs = load_all_kernel_configs(mind_dir)

    result = []
    for name, cfg in sorted(all_cfgs.items()):
        result.append(
            {
                "name": name,
                "sections": list(cfg.keys()),
            }
        )

    metrics.inc("mcp_kernel_list")
    _log.info("mcp_list_kernels", count=len(result))
    return json.dumps(
        {
            "_schema_version": MCP_SCHEMA_VERSION,
            "kernels": result,
        },
        indent=2,
    )


@mcp.tool
@mcp_tool_observe
def get_mind_kernel(name: str) -> str:
    """Read a specific .mind kernel configuration as structured JSON.

    Parses the INI-style [section] / key = value format.

    Args:
        name: Kernel name (e.g., "recall", "rm3", "rerank", "temporal",
              "adversarial", "hybrid").

    Returns:
        JSON with the full kernel configuration, or error if not found.
    """
    if not _re_mod.match(r"^[a-zA-Z0-9_-]{1,64}$", name):
        return json.dumps(
            {
                "_schema_version": MCP_SCHEMA_VERSION,
                "error": f"Invalid kernel name: {name}",
            }
        )

    ws = _workspace()
    mind_dir = get_mind_dir(ws)
    path = os.path.join(mind_dir, f"{name}.mind")

    cfg = load_kernel_config(path)
    if not cfg:
        return json.dumps(
            {
                "_schema_version": MCP_SCHEMA_VERSION,
                "error": f"Kernel '{name}' not found",
            }
        )

    metrics.inc("mcp_kernel_reads")
    _log.info("mcp_get_kernel", name=name, sections=list(cfg.keys()))
    return json.dumps(
        {
            "_schema_version": MCP_SCHEMA_VERSION,
            "name": name,
            "config": cfg,
        },
        indent=2,
    )


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
        envelope["warning"] = (
            f"Output truncated to {max_blocks} blocks (total: {total}). Increase max_blocks to export more."
        )

    return json.dumps(
        envelope,
        indent=2,
    )


# ---------------------------------------------------------------------------
# v2.0.0a1 governance tools — verify_chain, list_evidence
# ---------------------------------------------------------------------------


@mcp.tool
@mcp_tool_observe
def verify_chain() -> str:
    """Verify the integrity of the SHA3-512 governance hash chain.

    Walks every entry in the chain and checks that each entry_hash matches
    its recomputed value and that chain linkage is unbroken.

    Returns:
        JSON with valid (bool), length (int), and broken_at (int, -1 if valid).
    """
    ws = _workspace()
    try:
        from mind_mem.governance_gate import get_gate

        gate = get_gate(ws)
        chain = gate.chain
        hc_valid, broken_at = chain.verify_chain()
        length = chain.length

        evidence = gate.evidence
        ev_valid, broken_ids = evidence.verify_chain()
    except Exception as exc:
        _log.warning("verify_chain_failed", error=str(exc))
        return json.dumps(
            {
                "_schema_version": MCP_SCHEMA_VERSION,
                "error": f"Chain verification failed: {exc}",
            },
            indent=2,
        )

    overall_valid = hc_valid and ev_valid
    metrics.inc("mcp_verify_chain")
    _log.info(
        "mcp_verify_chain",
        valid=overall_valid,
        hash_chain_valid=hc_valid,
        length=length,
        broken_at=broken_at,
        evidence_valid=ev_valid,
        evidence_broken_ids=broken_ids,
    )
    return json.dumps(
        {
            "_schema_version": MCP_SCHEMA_VERSION,
            "valid": overall_valid,
            "hash_chain": {
                "valid": hc_valid,
                "length": length,
                "broken_at": broken_at,
            },
            "evidence_chain": {
                "valid": ev_valid,
                "broken_ids": broken_ids,
            },
        },
        indent=2,
    )


@mcp.tool
@mcp_tool_observe
def list_evidence(
    block_id: str = "",
    action: str = "",
    limit: int = 20,
) -> str:
    """List governance evidence objects, optionally filtered by block_id or action.

    Args:
        block_id: Filter to evidence records for this block ID (optional).
        action: Filter by evidence action type — PROPOSE, APPLY, ROLLBACK,
                CONTRADICT, DRIFT, RESOLVE, VERIFY (optional).
        limit: Maximum number of records to return (default 20).

    Returns:
        JSON array of evidence objects as dicts.
    """
    ws = _workspace()
    try:
        from mind_mem.evidence_objects import EvidenceAction
        from mind_mem.governance_gate import get_gate

        gate = get_gate(ws)
        evidence = gate.evidence

        if block_id:
            records = evidence.get_evidence_for_block(block_id)
        elif action:
            try:
                ev_action = EvidenceAction(action.upper())
            except ValueError:
                return json.dumps(
                    {
                        "_schema_version": MCP_SCHEMA_VERSION,
                        "error": (
                            f"Unknown action: {action!r}. "
                            "Valid values: PROPOSE, APPLY, ROLLBACK, CONTRADICT, DRIFT, RESOLVE, VERIFY"
                        ),
                    },
                    indent=2,
                )
            records = evidence.get_evidence_by_action(ev_action)
        else:
            records = evidence.get_latest(limit)

        # Apply limit
        records = records[-limit:] if len(records) > limit else records

    except Exception as exc:
        _log.warning("list_evidence_failed", error=str(exc))
        return json.dumps(
            {
                "_schema_version": MCP_SCHEMA_VERSION,
                "error": f"Evidence listing failed: {exc}",
            },
            indent=2,
        )

    metrics.inc("mcp_list_evidence")
    _log.info("mcp_list_evidence", block_id=block_id, action=action, count=len(records))
    return json.dumps(
        {
            "_schema_version": MCP_SCHEMA_VERSION,
            "count": len(records),
            "evidence": [r.to_dict() for r in records],
        },
        indent=2,
        default=str,
    )


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
        return json.dumps({
            "_schema_version": MCP_SCHEMA_VERSION,
            "error": f"Invalid block_id format: {block_id}",
        })

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
                    return json.dumps({
                        "_schema_version": MCP_SCHEMA_VERSION,
                        "block_id": block_id,
                        "found": True,
                        "block": block,
                    }, indent=2, default=str)
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
                        return json.dumps({
                            "_schema_version": MCP_SCHEMA_VERSION,
                            "block_id": block_id,
                            "found": True,
                            "block": block,
                        }, indent=2, default=str)
            except (OSError, ValueError, BlockCorruptedError):
                continue

    metrics.inc("mcp_get_block_miss")
    return json.dumps({
        "_schema_version": MCP_SCHEMA_VERSION,
        "block_id": block_id,
        "found": False,
        "error": f"Block {block_id} not found in any corpus file.",
        "hint": "Check the block ID and ensure the workspace is initialized.",
    }, indent=2)


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
                f"{stale_count} stale block(s) need review. "
                "Use stale_blocks tool for details, then update or clear staleness."
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
        recommendations.append(
            f"{drift_count} drift item(s) detected. Review intelligence/DRIFT.md for belief shifts."
        )

    # 4. Embedding/vector coverage
    import struct as _struct_mod
    try:
        from mind_mem.recall_vector import _index_path
        vec_path = _index_path(ws)
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
                            recommendations.append(
                                f"Embedding coverage is {coverage}%. Run reindex(include_vectors=True)."
                            )
                else:
                    health["embedded_blocks"] = 0
                    health["embedding_coverage_pct"] = 0.0
        else:
            health["embedded_blocks"] = 0
            health["embedding_coverage_pct"] = 0.0
            if total_blocks > 10:
                recommendations.append(
                    "No vector index found. Run reindex(include_vectors=True) for hybrid search."
                )
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
        recommendations.append(
            f"{pending_signals} pending signals. Review and apply or reject them."
        )

    contra_path = os.path.join(ws, "intelligence", "CONTRADICTIONS.md")
    contra_count = 0
    if os.path.isfile(contra_path):
        try:
            contra_count = len(parse_file(contra_path))
        except (OSError, ValueError):
            pass
    health["unresolved_contradictions"] = contra_count
    if contra_count > 0:
        recommendations.append(
            f"{contra_count} unresolved contradiction(s). Use list_contradictions for details."
        )

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
                recommendations.append(
                    f"FTS index has {stale_files} stale file(s). Run reindex tool."
                )
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
            recommendations.append(
                f"{total_compactable} item(s) ready for compaction. Run compact tool."
            )
    except (ImportError, OSError, ValueError) as exc:
        health["compaction"] = {"error": str(exc)}

    health["recommendations"] = recommendations
    health["score"] = "healthy" if not recommendations else "needs_attention"

    metrics.inc("mcp_memory_health")
    _log.info("mcp_memory_health", total_blocks=total_blocks, recommendations=len(recommendations))
    return json.dumps(health, indent=2, default=str)


@mcp.tool
@mcp_tool_observe
def traverse_graph(block_id: str, depth: int = 2, direction: str = "both") -> str:
    """Navigate the causal dependency graph from a block.

    Traverses the directed graph of block relationships (depends_on, supersedes,
    informs, contradicts) to show how blocks are connected. Useful for impact
    analysis before modifying a block, or understanding why a block exists.

    Args:
        block_id: Starting block ID (e.g., "D-20260213-001").
        depth: Maximum traversal depth (default: 2, max: 5).
        direction: "upstream" (what this block depends on), "downstream"
                   (what depends on this block), or "both" (default).

    Returns:
        JSON with graph nodes, edges, and causal chains from the starting block.
    """
    if not _re_mod.match(r"^[A-Z]+-[a-zA-Z0-9_.-]+$", block_id):
        return json.dumps({
            "_schema_version": MCP_SCHEMA_VERSION,
            "error": f"Invalid block_id format: {block_id}",
        })

    depth = max(1, min(depth, 5))

    if direction not in ("upstream", "downstream", "both"):
        return json.dumps({
            "_schema_version": MCP_SCHEMA_VERSION,
            "error": f"Invalid direction: {direction}. Use 'upstream', 'downstream', or 'both'.",
        })

    ws = _workspace()
    try:
        from mind_mem.causal_graph import CausalGraph

        cg = CausalGraph(ws)

        result: dict[str, Any] = {
            "_schema_version": MCP_SCHEMA_VERSION,
            "block_id": block_id,
            "direction": direction,
            "depth": depth,
        }

        if direction in ("upstream", "both"):
            chains = cg.causal_chain(block_id, max_depth=depth)
            deps = cg.dependencies(block_id)
            result["upstream"] = {
                "direct_dependencies": [
                    {
                        "target": e.target_id,
                        "edge_type": e.edge_type,
                        "weight": e.weight,
                    }
                    for e in deps
                ],
                "causal_chains": chains,
            }

        if direction in ("downstream", "both"):
            dependents = cg.dependents(block_id)
            # BFS for downstream subgraph
            downstream_nodes: list[dict] = []
            visited: set[str] = {block_id}
            current_layer = [block_id]
            for d in range(depth):
                next_layer: list[str] = []
                for node_id in current_layer:
                    node_deps = cg.dependents(node_id)
                    for e in node_deps:
                        if e.source_id not in visited:
                            visited.add(e.source_id)
                            next_layer.append(e.source_id)
                            downstream_nodes.append({
                                "block_id": e.source_id,
                                "depends_on": node_id,
                                "edge_type": e.edge_type,
                                "depth": d + 1,
                            })
                current_layer = next_layer
                if not current_layer:
                    break

            result["downstream"] = {
                "direct_dependents": [
                    {
                        "source": e.source_id,
                        "edge_type": e.edge_type,
                        "weight": e.weight,
                    }
                    for e in dependents
                ],
                "reachable_nodes": downstream_nodes,
            }

        # Graph summary
        summary = cg.summary()
        result["graph_summary"] = summary

        metrics.inc("mcp_traverse_graph")
        _log.info("mcp_traverse_graph", block_id=block_id, direction=direction, depth=depth)
        return json.dumps(result, indent=2, default=str)

    except ImportError:
        return json.dumps({
            "_schema_version": MCP_SCHEMA_VERSION,
            "error": "causal_graph module not available",
            "block_id": block_id,
        }, indent=2)
    except sqlite3.OperationalError as exc:
        if _is_db_locked(exc):
            return _sqlite_busy_error()
        raise
    except (OSError, ValueError) as exc:
        _log.warning("traverse_graph_failed", block_id=block_id, error=str(exc))
        return json.dumps({
            "_schema_version": MCP_SCHEMA_VERSION,
            "error": f"Graph traversal failed: {exc}",
            "block_id": block_id,
        }, indent=2)


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

    return json.dumps({
        "_schema_version": MCP_SCHEMA_VERSION,
        "status": "dry_run" if dry_run else "executed",
        "dry_run": dry_run,
        "total_actions": total_actions,
        "actions": actions,
        "next_step": (
            "Call again with dry_run=False to execute." if dry_run and total_actions > 0
            else "Workspace is clean — nothing to compact." if total_actions == 0
            else None
        ),
    }, indent=2)


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
                return json.dumps({
                    "_schema_version": MCP_SCHEMA_VERSION,
                    "error": f"Invalid block_id format: {clear_block_id}",
                })
            cleared = cg.clear_staleness(clear_block_id)
            metrics.inc("mcp_stale_cleared")
            return json.dumps({
                "_schema_version": MCP_SCHEMA_VERSION,
                "action": "cleared",
                "block_id": clear_block_id,
                "was_stale": cleared,
            }, indent=2)

        # List mode
        stale = cg.get_stale_blocks()
        stale = stale[:max(1, min(limit, 100))]

        metrics.inc("mcp_stale_blocks")
        _log.info("mcp_stale_blocks", count=len(stale))

        if not stale:
            return json.dumps({
                "_schema_version": MCP_SCHEMA_VERSION,
                "status": "clean",
                "stale_count": 0,
                "message": "No stale blocks. All blocks are up to date.",
            }, indent=2)

        return json.dumps({
            "_schema_version": MCP_SCHEMA_VERSION,
            "status": "stale_found",
            "stale_count": len(stale),
            "blocks": stale,
            "hint": "Review each stale block and update or call stale_blocks(clear_block_id='...') to clear.",
        }, indent=2, default=str)

    except ImportError:
        return json.dumps({
            "_schema_version": MCP_SCHEMA_VERSION,
            "error": "causal_graph module not available",
        }, indent=2)
    except sqlite3.OperationalError as exc:
        if _is_db_locked(exc):
            return _sqlite_busy_error()
        raise
    except (OSError, ValueError) as exc:
        _log.warning("stale_blocks_failed", error=str(exc))
        return json.dumps({
            "_schema_version": MCP_SCHEMA_VERSION,
            "error": f"Stale block lookup failed: {exc}",
        }, indent=2)


# ---------------------------------------------------------------------------
# Calibration feedback tools
# ---------------------------------------------------------------------------


@mcp.tool
@mcp_tool_observe
def calibration_feedback(
    query_id: str,
    block_ids_useful: list[str] | None = None,
    block_ids_not_useful: list[str] | None = None,
    feedback_type: str = "accepted",
) -> str:
    """Record retrieval quality feedback for calibration.

    After a recall query, report which blocks were useful and which were not.
    This feeds a calibration loop that adjusts block ranking over time:
    consistently useful blocks get boosted, consistently unhelpful blocks
    get demoted.

    Args:
        query_id: The query_id from a previous recall result envelope.
        block_ids_useful: Block IDs that were useful/relevant.
        block_ids_not_useful: Block IDs that were not useful/irrelevant.
        feedback_type: Feedback kind — "accepted" (user used results),
            "rejected" (results were wrong), or "ignored" (user skipped).

    Returns:
        JSON confirmation with recorded feedback counts.
    """
    ws = _workspace()

    if feedback_type not in ("accepted", "rejected", "ignored"):
        return json.dumps({
            "_schema_version": MCP_SCHEMA_VERSION,
            "error": f"Invalid feedback_type: {feedback_type}. Must be accepted/rejected/ignored.",
        })

    useful = block_ids_useful or []
    not_useful = block_ids_not_useful or []

    if not useful and not not_useful:
        return json.dumps({
            "_schema_version": MCP_SCHEMA_VERSION,
            "error": "At least one of block_ids_useful or block_ids_not_useful must be provided.",
        })

    try:
        from mind_mem.calibration import CalibrationManager

        cal = CalibrationManager(ws)
        result = cal.record_feedback(
            query_id=query_id,
            block_ids_useful=useful,
            block_ids_not_useful=not_useful,
            feedback_type=feedback_type,
        )
    except ImportError:
        return json.dumps({
            "_schema_version": MCP_SCHEMA_VERSION,
            "error": "Calibration module not available.",
        })
    except Exception as exc:
        _log.warning("calibration_feedback_failed", error=str(exc))
        return json.dumps({
            "_schema_version": MCP_SCHEMA_VERSION,
            "error": f"Failed to record feedback: {exc}",
        })

    metrics.inc("mcp_calibration_feedback")
    return json.dumps(
        {
            "_schema_version": MCP_SCHEMA_VERSION,
            "status": "recorded",
            **result,
        },
        indent=2,
    )


@mcp.tool
@mcp_tool_observe
def calibration_stats() -> str:
    """Report calibration health — per-block scores, per-query-type accuracy.

    Shows which blocks are being boosted or demoted by the calibration
    feedback loop, and how accurate retrieval is for different query types
    (WHAT, WHEN, WHO, HOW, etc.).

    Returns:
        JSON report with calibration health metrics.
    """
    ws = _workspace()

    try:
        from mind_mem.calibration import CalibrationManager

        cal = CalibrationManager(ws)
        stats = cal.get_calibration_stats()
    except ImportError:
        return json.dumps({
            "_schema_version": MCP_SCHEMA_VERSION,
            "error": "Calibration module not available.",
        })
    except Exception as exc:
        _log.warning("calibration_stats_failed", error=str(exc))
        return json.dumps({
            "_schema_version": MCP_SCHEMA_VERSION,
            "error": f"Failed to retrieve calibration stats: {exc}",
        })

    metrics.inc("mcp_calibration_stats")
    return json.dumps(
        {
            "_schema_version": MCP_SCHEMA_VERSION,
            **stats,
        },
        indent=2,
    )


# ---------------------------------------------------------------------------
# Dream Cycle + Compiled Truth tools
# ---------------------------------------------------------------------------


@mcp.tool
@mcp_tool_observe
def dream_cycle(
    auto_repair: bool = False,
    lookback_days: int = 7,
    stale_days: int = 30,
) -> str:
    """Run the dream cycle — autonomous memory enrichment.

    Scans workspace for untracked entities, broken citations, stale blocks,
    and repeated facts. Optionally auto-repairs findings.

    Args:
        auto_repair: If True, auto-create entity files, suggest citation
            fixes, and promote repeated facts to compiled truth pages.
        lookback_days: Days to look back for entity discovery (default: 7).
        stale_days: Days before a block is considered stale (default: 30).

    Returns:
        JSON report with findings and any repair actions taken.
    """
    ws = _workspace()

    try:
        from mind_mem.dream_cycle import run_dream_cycle

        report = run_dream_cycle(
            ws,
            dry_run=False,
            auto_repair=auto_repair,
            lookback_days=lookback_days,
            stale_days=stale_days,
        )
    except Exception as exc:
        _log.warning("dream_cycle_failed", error=str(exc))
        return json.dumps({
            "_schema_version": MCP_SCHEMA_VERSION,
            "error": f"Dream cycle failed: {exc}",
        })

    result = {
        "_schema_version": MCP_SCHEMA_VERSION,
        "timestamp": report.timestamp,
        "entity_proposals": len(report.entity_proposals),
        "broken_citations": len(report.broken_citations),
        "stale_blocks": len(report.stale_blocks),
        "consolidation_candidates": len(report.consolidation_candidates),
        "total_findings": report.total_findings,
    }

    if report.entity_proposals:
        result["entities"] = [
            {"type": e.entity_type, "slug": e.slug, "source": e.source_file}
            for e in report.entity_proposals[:20]
        ]
    if report.broken_citations:
        result["citations"] = [
            {"id": c.cited_id, "file": c.source_file, "line": c.line_number}
            for c in report.broken_citations[:20]
        ]
    if report.stale_blocks:
        result["stale"] = [
            {"id": s.block_id, "days": s.days_stale}
            for s in report.stale_blocks[:20]
        ]
    if report.consolidation_candidates:
        result["consolidation"] = [
            {"fact": c.fact_text[:80], "count": c.occurrences}
            for c in report.consolidation_candidates[:10]
        ]
    if report.repair_actions:
        result["repairs"] = [
            {"type": a.action_type, "target": a.target, "detail": a.detail}
            for a in report.repair_actions
        ]
        result["total_repairs"] = len(report.repair_actions)
    if report.errors:
        result["errors"] = list(report.errors)

    metrics.inc("mcp_dream_cycle")
    return json.dumps(result, indent=2)


@mcp.tool
@mcp_tool_observe
def compiled_truth_load(entity_id: str) -> str:
    """Load a compiled truth page for an entity.

    Returns the current understanding and evidence trail for the entity.

    Args:
        entity_id: Entity identifier (e.g. "PRJ-mind-mem", "PER-nikolai").

    Returns:
        JSON with compiled section, evidence entries, and metadata.
    """
    ws = _workspace()

    try:
        from mind_mem.compiled_truth import load_truth_page
        page = load_truth_page(ws, entity_id)
    except Exception as exc:
        return json.dumps({
            "_schema_version": MCP_SCHEMA_VERSION,
            "error": f"Failed to load truth page: {exc}",
        })

    if page is None:
        return json.dumps({
            "_schema_version": MCP_SCHEMA_VERSION,
            "error": f"No compiled truth page found for '{entity_id}'.",
            "hint": "Create one with compiled_truth_add_evidence.",
        })

    result = {
        "_schema_version": MCP_SCHEMA_VERSION,
        "entity_id": page.entity_id,
        "entity_type": page.entity_type,
        "version": page.version,
        "last_compiled": page.last_compiled,
        "compiled_section": page.compiled_section,
        "evidence_count": len(page.evidence_entries),
        "evidence": [
            {
                "timestamp": e.timestamp,
                "source": e.source,
                "observation": e.observation,
                "confidence": e.confidence,
                "superseded": e.superseded,
            }
            for e in page.evidence_entries
        ],
    }

    metrics.inc("mcp_compiled_truth_load")
    return json.dumps(result, indent=2)


@mcp.tool
@mcp_tool_observe
def compiled_truth_add_evidence(
    entity_id: str,
    observation: str,
    source: str = "mcp_tool",
    confidence: str = "medium",
    entity_type: str = "topic",
) -> str:
    """Add evidence to a compiled truth page and auto-recompile.

    Creates the page if it doesn't exist. Automatically recompiles the
    compiled section after adding evidence.

    Args:
        entity_id: Entity identifier (e.g. "PRJ-mind-mem").
        observation: The evidence text to record.
        source: Where the evidence came from (default: "mcp_tool").
        confidence: Confidence level — "high", "medium", or "low".
        entity_type: Entity type if creating new page — "project", "person",
            "tool", or "topic" (default: "topic").

    Returns:
        JSON with the updated page metadata.
    """
    ws = _workspace()

    try:
        from mind_mem.compiled_truth import (
            CompiledTruthPage,
            EvidenceEntry,
            add_evidence,
            load_truth_page,
            recompile_truth,
            save_truth_page,
        )
        from datetime import datetime, timezone

        page = load_truth_page(ws, entity_id)
        now_iso = datetime.now(timezone.utc).isoformat()

        if page is None:
            page = CompiledTruthPage(
                entity_id=entity_id,
                entity_type=entity_type,
                compiled_section="",
                evidence_entries=[],
                last_compiled=now_iso,
                version=0,
            )

        entry = EvidenceEntry(
            timestamp=now_iso,
            source=source,
            observation=observation,
            confidence=confidence,
            superseded=False,
        )

        page = add_evidence(page, entry)
        page = recompile_truth(page)
        path = save_truth_page(ws, page)

    except Exception as exc:
        return json.dumps({
            "_schema_version": MCP_SCHEMA_VERSION,
            "error": f"Failed to add evidence: {exc}",
        })

    result = {
        "_schema_version": MCP_SCHEMA_VERSION,
        "entity_id": page.entity_id,
        "version": page.version,
        "evidence_count": len(page.evidence_entries),
        "path": path,
        "message": f"Evidence added and page recompiled (v{page.version}).",
    }

    metrics.inc("mcp_compiled_truth_add_evidence")
    return json.dumps(result, indent=2)


@mcp.tool
@mcp_tool_observe
def compiled_truth_contradictions(entity_id: str) -> str:
    """Detect contradictions in a compiled truth page.

    Compares evidence entries for negation asymmetry and antonym pairs.

    Args:
        entity_id: Entity identifier to analyse.

    Returns:
        JSON with detected contradictions.
    """
    ws = _workspace()

    try:
        from mind_mem.compiled_truth import detect_contradictions, load_truth_page

        page = load_truth_page(ws, entity_id)
        if page is None:
            return json.dumps({
                "_schema_version": MCP_SCHEMA_VERSION,
                "error": f"No compiled truth page found for '{entity_id}'.",
            })

        conflicts = detect_contradictions(page)
    except Exception as exc:
        return json.dumps({
            "_schema_version": MCP_SCHEMA_VERSION,
            "error": f"Failed to detect contradictions: {exc}",
        })

    result = {
        "_schema_version": MCP_SCHEMA_VERSION,
        "entity_id": entity_id,
        "contradiction_count": len(conflicts),
        "contradictions": [
            {
                "entry_a": {"timestamp": a.timestamp, "observation": a.observation[:100]},
                "entry_b": {"timestamp": b.timestamp, "observation": b.observation[:100]},
                "reason": reason,
            }
            for a, b, reason in conflicts
        ],
    }

    metrics.inc("mcp_compiled_truth_contradictions")
    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _check_token() -> str | None:
    """Get token from environment. Returns None if no auth configured."""
    return os.environ.get("MIND_MEM_TOKEN")


def verify_token(headers: dict) -> bool:
    """Verify Bearer token from request headers. Constant-time compare.

    Returns True if:
      - No token is configured (open access), or
      - Token matches via Authorization: Bearer <token> or X-MindMem-Token header.

    Returns False if token is configured but missing/invalid.
    """
    expected = _check_token()
    if expected is None:
        return True  # No auth configured — allow

    # Try Authorization: Bearer <token>
    auth = headers.get("authorization", headers.get("Authorization", ""))
    if auth.startswith("Bearer "):
        provided = auth[7:]
        if hmac.compare_digest(provided, expected):
            return True

    # Try X-MindMem-Token header
    alt = headers.get("x-mindmem-token", headers.get("X-MindMem-Token", ""))
    if alt and hmac.compare_digest(alt, expected):
        return True

    metrics.inc("mcp_http_auth_failures")
    return False


def main():
    """Entry point for the MCP server (used by console_scripts and __main__)."""
    import argparse

    parser = argparse.ArgumentParser(description="Mind-Mem MCP Server")
    parser.add_argument(
        "--transport", choices=["stdio", "http"], default="stdio", help="Transport protocol (default: stdio)"
    )
    parser.add_argument("--port", type=int, default=8765, help="HTTP port (only used with --transport http)")
    parser.add_argument("--token", default=None, help="Bearer token for HTTP auth (or set MIND_MEM_TOKEN env var)")
    parser.add_argument("--watch", action="store_true", help="Auto-reindex when workspace .md files change")
    parser.add_argument(
        "--watch-interval", type=float, default=5.0, help="File watch polling interval in seconds (default: 5.0)"
    )
    args = parser.parse_args()

    # Set token from CLI arg if provided (env var takes precedence if both set)
    if args.token and not os.environ.get("MIND_MEM_TOKEN"):
        import warnings

        warnings.warn(
            "Passing --token on the command line exposes it in /proc/cmdline. "
            "Use MIND_MEM_TOKEN environment variable instead.",
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
                hint="HTTP transport running without token auth. "
                "Set MIND_MEM_TOKEN or MIND_MEM_ADMIN_TOKEN for security.",
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
