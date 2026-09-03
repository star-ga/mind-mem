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

**Every resource that returns corpus content is admitted.** Measured
2026-09-02 against a workspace seeded with a quarantined canary per
backing file, five of the eight served it verbatim: ``get_tasks``
returned every parsed block, and ``get_entities`` / ``get_signals`` /
``get_contradictions`` / ``get_ledger`` returned the raw Markdown, so a
quarantined block was readable through a registered ``mind-mem://`` URI
by any client that could read a resource at all. That is the ``get_block``
defect on the surface next door, and it had the same cause: the
enumeration that checks read surfaces (``count_mcp_tools._tool_names``)
collects ``mcp.tool`` arguments, and a resource is not a tool, so nothing
in the tripwire could see one.

The fix is structural rather than five remembered filters:
:func:`_admitted_corpus` is the **only** way a body in this module gets a
corpus row, and it ends in :func:`~mind_mem.admission.admit_read` — the
same egress decision recall's legs and the HTTP transport make. The raw
reader is gone from this module's imports, so a new resource cannot
return corpus bytes by reaching for the convenient helper; it has to go
through the seam or write its own file I/O in plain sight of review.
``tests/test_read_surface_resources.py`` sweeps every *registered*
resource with the canary and holds an exact-equality ratchet on the set
that leaks, which is empty.

A resource cannot ask who is calling. ``register`` binds these functions
onto FastMCP directly, with none of the ACL wrapping ``mcp_tool_observe``
gives a tool, so ``MIND_MEM_SCOPE`` / the admin token never reach here.
Anything that needs an operator behind it — reviewing what admission
withheld, above all — belongs on the tool side, which is classified in
``ADMIN_TOOLS`` / ``USER_TOOLS`` and enforced per call.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from typing import Any

from mind_mem.admission import admit_read
from mind_mem.block_parser import get_active, parse_file
from mind_mem.observability import get_logger
from mind_mem.recall import recall as recall_engine
from mind_mem.sqlite_index import _db_path as fts_db_path
from mind_mem.sqlite_index import query_index as fts_query

from .infra.workspace import _validate_path, _workspace

_log = get_logger("mcp_resources")


def _blocks_to_json(blocks: list[dict]) -> str:
    """Convert parsed blocks to JSON string."""
    return json.dumps(blocks, indent=2, default=str)


def _admitted_corpus(
    rel_path: str,
    *,
    surface: str,
    refine: Callable[[list[dict]], list[dict]] | None = None,
) -> tuple[list[dict], int]:
    """The blocks of one corpus file this surface may serve, and how many it may not.

    The single seam. ``parse_file`` answers with everything on disk —
    quarantined and pending blocks included — which is correct for a parser
    and wrong for a surface that hands the result to a client, so no body in
    this module calls it directly and no body reads a corpus file as bytes.

    ``workspace`` is passed to :func:`~mind_mem.admission.admit_read`
    deliberately. The caller here parsed the blocks itself, so the status
    refresh is buying little, but the release set is the leg that matters:
    an operator-approved readmission is resolved from the corpus and a
    local ``if status != "active"`` cannot see one. Omitting the workspace
    is the *strict* direction and would still be safe; it would also make
    this surface disagree with recall about a block an operator released,
    and the cost of agreeing is one corpus parse cached on file identity
    (``admissibility._LIVE_STATUS_CACHE``), resolved lazily.

    Args:
        rel_path: Workspace-relative corpus file.
        surface: Recorded on the withheld metric, for the same reason
            ``admit_leg`` takes ``leg``.
        refine: Optional caller filter applied to the parsed blocks
            *before* admission — a caller's narrowing (``get_active`` on
            decisions), never the governance decision. Admission runs on
            whatever it returns, so a refinement can only ever remove.

    Returns:
        ``(admitted, withheld)`` — the blocks a caller may be shown, order
        preserved, and the number the gate dropped. The count is returned
        rather than only logged because a listing that is silently short
        reads as a complete corpus to whoever gets it.

    Raises:
        Whatever the parse or the status refresh raises. Deliberately not
        swallowed: a surface that cannot confirm a status must fail rather
        than serve the copy it could not check. The one caught error is a
        path that leaves the workspace, which is a refusal and not a read.
    """
    ws = _workspace()
    try:
        path = _validate_path(ws, rel_path)
    except ValueError:
        # Unreachable with the constants below (every rel_path is a
        # literal and ``get_entities`` allow-lists its segment), kept
        # because the guard has to hold for the NEXT resource too, and
        # because it is what rejects a symlinked corpus file — the TOCTOU
        # hole audit S-10 closed for the three resources that already
        # read through the validating helper.
        _log.warning("resource_path_rejected", surface=surface, rel_path=rel_path)
        return [], 0
    if not os.path.isfile(path):
        return [], 0
    blocks = parse_file(path)
    if refine is not None:
        blocks = refine(blocks)
    decision = admit_read(blocks, workspace=ws, surface=surface)
    if decision.withheld:
        _log.info("resource_withheld", surface=surface, withheld=decision.withheld)
    return decision.admitted, decision.withheld


def get_decisions() -> str:
    """Active decisions from the workspace. Structured blocks with IDs, statements, dates, and status."""
    admitted, _withheld = _admitted_corpus(
        "decisions/DECISIONS.md",
        surface="resource:decisions",
        refine=get_active,
    )
    return _blocks_to_json(admitted)


def get_tasks() -> str:
    """All admitted tasks from the workspace."""
    admitted, _withheld = _admitted_corpus("tasks/TASKS.md", surface="resource:tasks")
    return _blocks_to_json(admitted)


def get_entities(entity_type: str) -> str:
    """Entity files: projects, people, tools, or incidents."""
    allowed = {"projects", "people", "tools", "incidents"}
    if entity_type not in allowed:
        return json.dumps({"error": f"Unknown entity type: {entity_type}. Use: {', '.join(sorted(allowed))}"})
    admitted, _withheld = _admitted_corpus(f"entities/{entity_type}.md", surface=f"resource:entities:{entity_type}")
    return _blocks_to_json(admitted)


def get_signals() -> str:
    """Reviewed signals, and a count of what review still owes.

    The one resource whose corpus is withheld *by design*: ``capture``
    writes ``Status: pending`` blocks nobody has looked at, and ``pending``
    is in ``admissibility.UNADMITTED`` because
    ``enums.INITIAL_STATUS[AUTO_CAPTURE]`` mints it. So admission empties
    this file in the normal case, and a bare ``[]`` would be the wrong
    answer here and only here — it reads as "no signals" when the truth is
    "N signals exist and none has passed review". The envelope says which.

    The count is all a resource may give: nothing here can ask who is
    calling (see the module docstring), so the pending blocks are not
    servable through this URI at any scope. Reviewing them is the
    ACL-classified tool side — ``signal_stats`` for the shape of the
    backlog, ``propose_update`` / ``approve_apply`` to admit any of it.
    """
    admitted, withheld = _admitted_corpus("intelligence/SIGNALS.md", surface="resource:signals")
    return json.dumps(
        {
            "resource": "mind-mem://signals",
            "signals": admitted,
            "withheld_count": withheld,
            "note": (
                "Signals are auto-captured pending review, and admission withholds a pending block. "
                "This resource serves only signals that have passed the gate; "
                "review the backlog through the ACL-scoped tools (signal_stats, propose_update/approve_apply)."
            ),
        },
        indent=2,
        default=str,
    )


def get_contradictions() -> str:
    """Detected contradictions between decisions."""
    admitted, _withheld = _admitted_corpus("intelligence/CONTRADICTIONS.md", surface="resource:contradictions")
    return _blocks_to_json(admitted)


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
    """Shared fact ledger for multi-agent memory propagation.

    Blocks, not bytes. ``namespaces.SharedLedger.append_fact`` writes
    ``[FACT-...]`` blocks carrying a free-text ``Text:`` field and a
    ``Status:`` line, so this file is corpus content like any other and was
    being served verbatim — the sweep in
    ``tests/test_read_surface_resources.py`` had classified it
    ``no-content`` on the strength of the *decisions* canary never
    appearing in it, which is true of every file except one and proves
    nothing about this one. Seeded with a quarantined ``FACT`` block,
    measured 2026-09-02: served. It goes through the same seam now.
    """
    admitted, _withheld = _admitted_corpus("shared/intelligence/LEDGER.md", surface="resource:ledger")
    return _blocks_to_json(admitted)


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
