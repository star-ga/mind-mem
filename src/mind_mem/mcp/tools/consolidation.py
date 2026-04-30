"""Memory-consolidation MCP tools.

Extracted from ``mcp_server.py`` per docs/v3.2.0-mcp-decomposition-plan.md
(PR-3 slice, consolidation domain). Four tools cover the "memory
settles over time" surface:

* ``plan_consolidation`` — dry-run of the cognitive-forgetting cycle.
* ``propagate_staleness`` — diffusion of staleness scores over xrefs.
* ``project_profile`` — structured session-start intelligence
  summary.
* ``dream_cycle`` — autonomous memory enrichment (entity discovery,
  broken-citation scan, consolidation candidates, optional auto-repair).
"""

from __future__ import annotations

import json
import os
from typing import Any

from ..infra.constants import MCP_SCHEMA_VERSION
from ..infra.observability import mcp_tool_observe
from ..infra.workspace import _check_workspace, _workspace
from ._helpers import get_logger, metrics

_log = get_logger("mcp_server")


@mcp_tool_observe
def plan_consolidation(
    importance_threshold: float = 0.25,
    stale_days: int = 14,
    archive_after_days: int = 60,
    grace_days: int = 30,
) -> str:
    """Dry-run the cognitive forgetting cycle."""
    from mind_mem.cognitive_forget import (
        BlockCognition,
        BlockLifecycle,
        ConsolidationConfig,
    )
    from mind_mem.cognitive_forget import (
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


@mcp_tool_observe
def propagate_staleness(seed_block_ids: str, max_hops: int = 3) -> str:
    """Diffuse staleness outward from seed blocks over the xref graph."""
    import sqlite3 as _sqlite3

    from mind_mem.staleness import propagate_staleness as _propagate

    ws = _workspace()
    ws_err = _check_workspace(ws)
    if ws_err:
        return ws_err

    if not isinstance(seed_block_ids, str) or not seed_block_ids.strip():
        return json.dumps({"error": "seed_block_ids must be a non-empty string"})
    seeds = [bid.strip() for bid in seed_block_ids.split(",") if bid.strip()][:64]
    if not seeds:
        return json.dumps({"error": "no seed block ids supplied"})
    if not (0 <= max_hops <= 8):
        return json.dumps({"error": "max_hops must be in [0, 8]"})

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


@mcp_tool_observe
def project_profile(name: str = "", top_k: int = 10) -> str:
    """Auto-generate a structured project intelligence profile."""
    import sqlite3 as _sqlite3

    from mind_mem.project_profile import build_profile

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
            rows = conn.execute("SELECT id, type, file, date, json_blob FROM blocks LIMIT 50000").fetchall()
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


@mcp_tool_observe
def dream_cycle(
    auto_repair: bool = False,
    lookback_days: int = 7,
    stale_days: int = 30,
) -> str:
    """Run the dream cycle — autonomous memory enrichment."""
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
        return json.dumps(
            {
                "_schema_version": MCP_SCHEMA_VERSION,
                "error": f"Dream cycle failed: {exc}",
            }
        )

    result: dict[str, Any] = {
        "_schema_version": MCP_SCHEMA_VERSION,
        "timestamp": report.timestamp,
        "entity_proposals": len(report.entity_proposals),
        "broken_citations": len(report.broken_citations),
        "stale_blocks": len(report.stale_blocks),
        "consolidation_candidates": len(report.consolidation_candidates),
        "total_findings": report.total_findings,
    }

    if report.entity_proposals:
        result["entities"] = [{"type": e.entity_type, "slug": e.slug, "source": e.source_file} for e in report.entity_proposals[:20]]
    if report.broken_citations:
        result["citations"] = [{"id": c.cited_id, "file": c.source_file, "line": c.line_number} for c in report.broken_citations[:20]]
    if report.stale_blocks:
        result["stale"] = [{"id": s.block_id, "days": s.days_stale} for s in report.stale_blocks[:20]]
    if report.consolidation_candidates:
        result["consolidation"] = [{"fact": c.fact_text[:80], "count": c.occurrences} for c in report.consolidation_candidates[:10]]
    if report.repair_actions:
        result["repairs"] = [{"type": a.action_type, "target": a.target, "detail": a.detail} for a in report.repair_actions]
        result["total_repairs"] = len(report.repair_actions)
    if report.errors:
        result["errors"] = list(report.errors)

    metrics.inc("mcp_dream_cycle")
    return json.dumps(result, indent=2)


def register(mcp) -> None:
    """Wire the consolidation tools onto *mcp*."""
    mcp.tool(plan_consolidation)
    mcp.tool(propagate_staleness)
    mcp.tool(project_profile)
    mcp.tool(dream_cycle)
