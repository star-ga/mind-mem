"""MCP ``@mcp.resource`` declarations.

Extracted from ``mcp_server.py`` in the v3.2.0 §1.2 decomposition
(see docs/v3.2.0-mcp-decomposition-plan.md PR-2). The plan's target
topology lists one file per resource under ``mcp/resources/``; we
collapse all eight into a single module because (a) every resource
body is 5–15 lines, (b) they share the same imports, and (c) they
are a cohesive "read-only view over the workspace" surface — one
module keeps the diff reviewable and the docstrings co-located.

Resources exposed:
    mind-mem://decisions       — active decisions (DECISIONS.md)
    mind-mem://tasks           — all tasks (TASKS.md)
    mind-mem://entities/{type} — projects, people, tools, incidents
    mind-mem://signals         — auto-captured signals
    mind-mem://contradictions  — detected contradictions
    mind-mem://health          — workspace health summary
    mind-mem://recall/{query}  — BM25 recall search
    mind-mem://ledger          — shared multi-agent fact ledger

Registration pattern: the functions are defined at module level so
tests that reference ``server.get_decisions`` etc. keep working;
``register(mcp)`` wires them onto a FastMCP instance after the
server has been constructed. This avoids the circular import that
a top-level ``@mcp.resource`` decorator would create between
``mcp_server`` and this module.
"""

from __future__ import annotations

import json
import os
from typing import Any

from mind_mem.block_parser import get_active, parse_file
from mind_mem.recall import recall as recall_engine
from mind_mem.sqlite_index import _db_path as fts_db_path
from mind_mem.sqlite_index import query_index as fts_query

from .infra.workspace import _read_file, _workspace


def _blocks_to_json(blocks: list[dict]) -> str:
    """Convert parsed blocks to JSON string."""
    return json.dumps(blocks, indent=2, default=str)


def get_decisions() -> str:
    """Active decisions from the workspace. Structured blocks with IDs, statements, dates, and status."""
    ws = _workspace()
    path = os.path.join(ws, "decisions", "DECISIONS.md")
    if not os.path.isfile(path):
        return json.dumps([])
    blocks = parse_file(path)
    active = get_active(blocks)
    return _blocks_to_json(active)


def get_tasks() -> str:
    """All tasks from the workspace."""
    ws = _workspace()
    path = os.path.join(ws, "tasks", "TASKS.md")
    if not os.path.isfile(path):
        return json.dumps([])
    blocks = parse_file(path)
    return _blocks_to_json(blocks)


def get_entities(entity_type: str) -> str:
    """Entity files: projects, people, tools, or incidents."""
    allowed = {"projects", "people", "tools", "incidents"}
    if entity_type not in allowed:
        return json.dumps({"error": f"Unknown entity type: {entity_type}. Use: {', '.join(sorted(allowed))}"})
    return _read_file(f"entities/{entity_type}.md")


def get_signals() -> str:
    """Auto-captured signals pending review."""
    return _read_file("intelligence/SIGNALS.md")


def get_contradictions() -> str:
    """Detected contradictions between decisions."""
    return _read_file("intelligence/CONTRADICTIONS.md")


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


def get_recall(query: str) -> str:
    """Search memory using ranked recall (FTS5 or BM25 scan)."""
    ws = _workspace()
    if os.path.isfile(fts_db_path(ws)):
        results = fts_query(ws, query, limit=10)
    else:
        results = recall_engine(ws, query, limit=10)
    return json.dumps(results, indent=2, default=str)


def get_ledger() -> str:
    """Shared fact ledger for multi-agent memory propagation."""
    return _read_file("shared/intelligence/LEDGER.md")


def register(mcp) -> None:
    """Wire every resource body onto *mcp*. Called once from mcp_server.py."""
    mcp.resource("mind-mem://decisions")(get_decisions)
    mcp.resource("mind-mem://tasks")(get_tasks)
    mcp.resource("mind-mem://entities/{entity_type}")(get_entities)
    mcp.resource("mind-mem://signals")(get_signals)
    mcp.resource("mind-mem://contradictions")(get_contradictions)
    mcp.resource("mind-mem://health")(get_health)
    mcp.resource("mind-mem://recall/{query}")(get_recall)
    mcp.resource("mind-mem://ledger")(get_ledger)
