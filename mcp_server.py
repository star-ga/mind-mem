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

Tools (19):
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
import tempfile
import threading
import time

# mind-mem imports (package mapped to scripts/ via pyproject.toml)
from mind_mem.block_parser import BlockCorruptedError, get_active, parse_file  # noqa: E402, F401
from mind_mem.mind_filelock import FileLock  # noqa: E402
from fastmcp import FastMCP  # noqa: E402
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
    }
)

USER_TOOLS = frozenset(
    {
        "recall",
        "search_memory",
        "list_memory",
        "list_contradictions",
        "scan",
        "export_memory",
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
    }
)


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
    """Create rate limiter from config limits."""
    try:
        limits = _get_limits()
        max_calls = limits["rate_limit_calls_per_minute"]
    except Exception:
        max_calls = 120
    return SlidingWindowRateLimiter(max_calls=max_calls, window_seconds=60)


# Global rate limiter instance — initialized from config
_rate_limiter = _init_rate_limiter()

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

        # ACL enforcement: when MIND_MEM_ADMIN_TOKEN is configured,
        # admin tools require MIND_MEM_SCOPE=admin (stdio) or a valid
        # Authorization header (http).  When no admin token is set the
        # check is skipped so single-user local setups are unaffected.
        admin_token = os.environ.get("MIND_MEM_ADMIN_TOKEN")
        if admin_token and tool_name in ADMIN_TOOLS:
            scope = os.environ.get("MIND_MEM_SCOPE", "user")
            acl_error = check_tool_acl(tool_name, scope)
            if acl_error:
                _log.warning("acl_blocked", tool=tool_name, scope=scope)
                return acl_error
        elif admin_token and tool_name not in USER_TOOLS:
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
        return json.dumps({"error": "Workspace not found. Run: python3 scripts/init_workspace.py <path>"})
    decisions_dir = os.path.join(ws, "decisions")
    if not os.path.isdir(decisions_dir):
        return json.dumps(
            {"error": "Workspace is missing the 'decisions/' directory. Run: python3 scripts/init_workspace.py <path>"}
        )
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
            return json.load(f)
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
    return cfg.get("categories", {}).get("extra_categories", {})


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
    result = {"files": {}, "metrics": {}}

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

    metrics.inc("mcp_recall_queries")
    _log.info("mcp_recall", query=query, backend=used_backend, results=len(results))
    envelope: dict = {
        "_schema_version": MCP_SCHEMA_VERSION,
        "backend": used_backend,
        "query": query,
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
        signal["structure"]["rationale"] = rationale

    written = append_signals(ws, [signal], today)

    metrics.inc("mcp_proposals")
    _log.info("mcp_propose", block_type=block_type, confidence=confidence, written=written)

    return json.dumps(
        {
            "_schema_version": MCP_SCHEMA_VERSION,
            "status": "proposed",
            "written": written,
            "location": "intelligence/SIGNALS.md",
            "next_step": "Run /apply or `python3 maintenance/apply_engine.py` to review and promote to source of truth.",
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

    result = {"_schema_version": MCP_SCHEMA_VERSION, "checks": {}}

    # Parse decisions
    decisions_path = os.path.join(ws, "decisions", "DECISIONS.md")
    if os.path.isfile(decisions_path):
        blocks = parse_file(decisions_path)
        active = get_active(blocks)
        result["checks"]["decisions"] = {
            "total": len(blocks),
            "active": len(active),
        }
    else:
        result["checks"]["decisions"] = {"total": 0, "active": 0}

    # Check contradictions — report both raw entries and resolvable ones
    contra_path = os.path.join(ws, "intelligence", "CONTRADICTIONS.md")
    raw_count = 0
    if os.path.isfile(contra_path):
        raw_count = len(parse_file(contra_path))
    try:
        from mind_mem.conflict_resolver import resolve_contradictions

        resolutions = resolve_contradictions(ws)
        result["checks"]["contradictions"] = {
            "raw": raw_count,
            "resolvable": len(resolutions),
        }
    except (ImportError, OSError, ValueError) as exc:
        _log.warning("scan_contradiction_check_failed", error=str(exc))
        result["checks"]["contradictions"] = {"raw": raw_count, "resolvable": 0}

    # Check drift
    drift_path = os.path.join(ws, "intelligence", "DRIFT.md")
    if os.path.isfile(drift_path):
        drifts = parse_file(drift_path)
        result["checks"]["drift_items"] = len(drifts)
    else:
        result["checks"]["drift_items"] = 0

    # Check signals
    signals_path = os.path.join(ws, "intelligence", "SIGNALS.md")
    if os.path.isfile(signals_path):
        signals = parse_file(signals_path)
        result["checks"]["pending_signals"] = len(signals)
    else:
        result["checks"]["pending_signals"] = 0

    metrics.inc("mcp_scans")
    _log.info("mcp_scan", checks=result["checks"])

    return json.dumps(result, indent=2)


@mcp.tool
@mcp_tool_observe
def list_contradictions() -> str:
    """List detected contradictions with resolution analysis.

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

    return json.dumps(
        {
            "_schema_version": MCP_SCHEMA_VERSION,
            "status": "contradictions_found",
            "contradictions": len(resolutions),
            "resolutions": resolutions,
        },
        indent=2,
        default=str,
    )


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

    from mind_mem.apply_engine import apply_proposal

    # Capture stdout from apply_engine (it prints progress)
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        success, message = apply_proposal(ws, proposal_id, dry_run=dry_run)

    log_output = capture.getvalue()

    metrics.inc("mcp_apply_calls")
    _log.info("mcp_approve_apply", proposal_id=proposal_id, dry_run=dry_run, success=success)

    return json.dumps(
        {
            "_schema_version": MCP_SCHEMA_VERSION,
            "status": "applied" if success and not dry_run else "dry_run_passed" if success else "failed",
            "proposal_id": proposal_id,
            "dry_run": dry_run,
            "success": success,
            "message": message,
            "log": log_output[-2000:] if len(log_output) > 2000 else log_output,
            "next_step": "Call again with dry_run=False to apply." if success and dry_run else None,
        },
        indent=2,
    )


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

    Deprecated: use recall(backend="hybrid") instead. This is a thin wrapper.

    Args:
        query: Search query.
        limit: Maximum results (default: 10).
        active_only: Only return active blocks.

    Returns:
        JSON array of ranked results from fused retrieval.
    """
    return _recall_impl(query, limit=limit, active_only=active_only, backend="hybrid")


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
        for kind in ["decisions", "tasks", "entities"]:
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
        results["fts_error"] = "FTS index rebuild failed. Run: python3 scripts/sqlite_index.py build --workspace ."

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
        new_lines = lines[:block_start] + lines[block_end:]
        new_content = "\n".join(new_lines)

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
def export_memory(format: str = "jsonl", include_metadata: bool = False) -> str:
    """Export all workspace blocks as JSONL.

    Parses all .md files in decisions/, tasks/, entities/, intelligence/
    and returns one JSON object per line.

    Args:
        format: Output format — currently only "jsonl" is supported.
        include_metadata: Include A-MEM metadata fields (_entities, _dates, etc.).

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
    scan_dirs = ["decisions", "tasks", "entities", "intelligence"]

    for subdir in scan_dirs:
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

    # Build JSONL output
    jsonl_lines = [json.dumps(b, default=str) for b in all_blocks]
    jsonl_output = "\n".join(jsonl_lines)

    metrics.inc("mcp_export_memory")
    _log.info("mcp_export_memory", format=format, blocks=len(all_blocks))

    return json.dumps(
        {
            "_schema_version": MCP_SCHEMA_VERSION,
            "format": format,
            "block_count": len(all_blocks),
            "data": jsonl_output,
        },
        indent=2,
    )


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
        os.environ["MIND_MEM_TOKEN"] = args.token

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
        if not token:
            _log.warning(
                "mcp_http_no_auth",
                hint="HTTP transport running without token auth. "
                "Set MIND_MEM_TOKEN env var or use --token for security.",
            )
        if token:
            # Enforce Bearer token auth on HTTP transport.
            # FastMCP's StaticTokenVerifier gates all requests.
            from fastmcp.server.auth import OAuthProvider, StaticTokenVerifier

            verifier = StaticTokenVerifier(
                tokens={token: {"sub": "mind-mem-client", "scope": "full"}},
            )
            auth_provider = OAuthProvider(token_verifier=verifier)
            mcp._auth = auth_provider
            _log.info("mcp_auth_enforced", mode="static_token")
        mcp.run(transport="sse", port=args.port)
    else:
        mcp.run()


if __name__ == "__main__":
    main()
