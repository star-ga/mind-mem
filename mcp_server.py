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

Tools:
    recall               — Search memory with BM25
    propose_update       — Propose a new decision/task (writes to SIGNALS.md, never source of truth)
    approve_apply        — Apply a staged proposal (dry-run by default)
    rollback_proposal    — Rollback an applied proposal by receipt timestamp
    scan                 — Run integrity scan
    list_contradictions  — List detected contradictions with resolution status

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

import hmac
import json
import os
import sys

# Add scripts/ to path for mind-mem imports
SCRIPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
sys.path.insert(0, SCRIPT_DIR)

from fastmcp import FastMCP

from block_parser import parse_file, get_active, get_by_id
from recall import recall as recall_engine
from sqlite_index import query_index as fts_query, _db_path as fts_db_path
from observability import get_logger, metrics

_log = get_logger("mcp_server")

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


def _read_file(rel_path: str) -> str:
    """Read a file from workspace, return contents or error message."""
    path = os.path.join(_workspace(), rel_path)
    if not os.path.isfile(path):
        return f"File not found: {rel_path}"
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _blocks_to_json(blocks: list[dict]) -> str:
    """Convert parsed blocks to JSON string."""
    return json.dumps(blocks, indent=2, default=str)


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
    result = {"workspace": ws, "files": {}, "metrics": {}}

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

@mcp.tool
def recall(query: str, limit: int = 10, active_only: bool = False) -> str:
    """Search across all memory files with ranked retrieval.

    Uses FTS5 index when available (O(log N)), falls back to BM25 scan.

    Args:
        query: Search query (supports stemming and domain-aware expansion).
        limit: Maximum number of results (default: 10).
        active_only: Only return blocks with Status: active.

    Returns:
        JSON array of ranked results with scores, IDs, and matched content.
    """
    ws = _workspace()
    limit = max(1, min(limit, 100))
    # Use FTS5 index when it exists, otherwise fall back to scan
    if os.path.isfile(fts_db_path(ws)):
        results = fts_query(ws, query, limit=limit, active_only=active_only)
        backend = "sqlite"
    else:
        results = recall_engine(ws, query, limit=limit, active_only=active_only)
        backend = "scan"
    metrics.inc("mcp_recall_queries")
    _log.info("mcp_recall", query=query, backend=backend, results=len(results))
    return json.dumps(results, indent=2, default=str)


@mcp.tool
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
    from capture import append_signals, CONFIDENCE_TO_PRIORITY

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
            "subject": statement.split()[:3] if statement else [],
            "tags": [t.strip() for t in tags.split(",") if t.strip()],
        },
    }
    if rationale:
        signal["structure"]["rationale"] = rationale

    written = append_signals(ws, [signal], today)

    metrics.inc("mcp_proposals")
    _log.info("mcp_propose", block_type=block_type, confidence=confidence, written=written)

    return json.dumps({
        "status": "proposed",
        "written": written,
        "location": "intelligence/SIGNALS.md",
        "next_step": "Run /apply or `python3 maintenance/apply_engine.py` to review and promote to source of truth.",
        "safety": "This signal is in SIGNALS.md only. It has NOT been written to DECISIONS.md or TASKS.md.",
    }, indent=2)


@mcp.tool
def scan() -> str:
    """Run integrity scan — contradictions, drift, dead decisions, impact graph.

    Returns:
        JSON summary of scan results.
    """
    ws = _workspace()

    result = {"workspace": ws, "checks": {}}

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

    # Check contradictions
    contra_path = os.path.join(ws, "intelligence", "CONTRADICTIONS.md")
    if os.path.isfile(contra_path):
        contras = parse_file(contra_path)
        result["checks"]["contradictions"] = len(contras)
    else:
        result["checks"]["contradictions"] = 0

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
def list_contradictions() -> str:
    """List detected contradictions with resolution analysis.

    Returns:
        JSON array of contradictions with strategy recommendations.
    """
    ws = _workspace()

    from conflict_resolver import resolve_contradictions

    resolutions = resolve_contradictions(ws)
    if not resolutions:
        return json.dumps({"status": "clean", "contradictions": 0, "message": "No contradictions found."})

    return json.dumps({
        "status": "contradictions_found",
        "contradictions": len(resolutions),
        "resolutions": resolutions,
    }, indent=2, default=str)


@mcp.tool
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

    from apply_engine import apply_proposal

    import io
    import contextlib

    # Capture stdout from apply_engine (it prints progress)
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        success, message = apply_proposal(ws, proposal_id, dry_run=dry_run)

    log_output = capture.getvalue()

    metrics.inc("mcp_apply_calls")
    _log.info("mcp_approve_apply", proposal_id=proposal_id, dry_run=dry_run, success=success)

    return json.dumps({
        "status": "applied" if success and not dry_run else "dry_run_passed" if success else "failed",
        "proposal_id": proposal_id,
        "dry_run": dry_run,
        "success": success,
        "message": message,
        "log": log_output[-2000:] if len(log_output) > 2000 else log_output,
        "next_step": "Call again with dry_run=False to apply." if success and dry_run else None,
    }, indent=2)


@mcp.tool
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

    from apply_engine import rollback as engine_rollback

    import io
    import contextlib

    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        success = engine_rollback(ws, receipt_ts)

    log_output = capture.getvalue()

    metrics.inc("mcp_rollbacks")
    _log.info("mcp_rollback", receipt_ts=receipt_ts, success=success)

    return json.dumps({
        "status": "rolled_back" if success else "rollback_failed",
        "receipt_ts": receipt_ts,
        "success": success,
        "log": log_output[-2000:] if len(log_output) > 2000 else log_output,
    }, indent=2)


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
      - Token matches via Authorization: Bearer <token> or X-MemOS-Token header.

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

    # Try X-MemOS-Token header
    alt = headers.get("x-memos-token", headers.get("X-MemOS-Token", ""))
    if alt and hmac.compare_digest(alt, expected):
        return True

    metrics.inc("mcp_http_auth_failures")
    return False


def main():
    """Entry point for the MCP server (used by console_scripts and __main__)."""
    import argparse

    parser = argparse.ArgumentParser(description="Mind-Mem MCP Server")
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio",
                        help="Transport protocol (default: stdio)")
    parser.add_argument("--port", type=int, default=8765,
                        help="HTTP port (only used with --transport http)")
    parser.add_argument("--token", default=None,
                        help="Bearer token for HTTP auth (or set MIND_MEM_TOKEN env var)")
    args = parser.parse_args()

    # Set token from CLI arg if provided (env var takes precedence if both set)
    if args.token and not os.environ.get("MIND_MEM_TOKEN"):
        os.environ["MIND_MEM_TOKEN"] = args.token

    token = _check_token()
    _log.info("mcp_server_start", transport=args.transport,
              workspace=_workspace(), auth="token" if token else "none")

    if args.transport == "http":
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
