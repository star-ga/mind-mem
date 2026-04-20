"""Governance MCP tools — propose / apply / rollback / scan / contradictions / memory_evolution.

Extracted from ``mcp_server.py`` per docs/v3.2.0-mcp-decomposition-plan.md
(PR-3 slice, governance domain). Six tools that cover the
"memory is never modified except by governance" invariant:

* ``propose_update`` — stage a new decision/task as a SIGNAL.
* ``approve_apply`` — apply a staged proposal (dry-run by default).
* ``rollback_proposal`` — restore workspace from pre-apply snapshot.
* ``scan`` — integrity scan (contradictions / drift / pending).
* ``list_contradictions`` — enriched contradiction listing.
* ``memory_evolution`` — A-MEM metadata for a block.
"""

from __future__ import annotations

import json
import os
import re as _re_mod
import sqlite3
from typing import Any

from mind_mem.block_parser import get_active, parse_file
from mind_mem.observability import get_logger, metrics
from mind_mem.telemetry import traced as _traced

from ..infra.constants import MCP_SCHEMA_VERSION
from ..infra.observability import _is_db_locked, _sqlite_busy_error, mcp_tool_observe
from ..infra.workspace import _check_workspace, _workspace

_log = get_logger("mcp_server")


@mcp_tool_observe
@_traced("propose_update")
def propose_update(
    block_type: str,
    statement: str,
    rationale: str = "",
    tags: str = "",
    confidence: str = "medium",
) -> str:
    """Propose a new decision or task. Writes to SIGNALS.md for human review."""
    ws = _workspace()

    if block_type not in ("decision", "task"):
        return json.dumps({"error": f"block_type must be 'decision' or 'task', got '{block_type}'"})

    from datetime import datetime

    from mind_mem.capture import CONFIDENCE_TO_PRIORITY, append_signals

    today = datetime.now().strftime("%Y-%m-%d")
    priority = CONFIDENCE_TO_PRIORITY.get(confidence, "P2")

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

    # v3.2.1: invalidate the recall cache so the next query doesn't
    # serve a pre-proposal envelope that omits the new signal.
    _invalidate_recall_cache()

    return json.dumps(
        {
            "_schema_version": MCP_SCHEMA_VERSION,
            "status": "proposed",
            "written": written,
            "location": "intelligence/SIGNALS.md",
            "next_step": ("Run /apply or `python3 maintenance/apply_engine.py` to review and promote to source of truth."),
            "safety": "This signal is in SIGNALS.md only. It has NOT been written to DECISIONS.md or TASKS.md.",
        },
        indent=2,
    )


def _invalidate_recall_cache() -> None:
    """Flush the recall cache after a governance event.

    Namespace-wide invalidation — targeted per-block invalidation
    would require tracking which queries touched which blocks, which
    is more complexity than the typical workspace needs. Best-effort
    (swallows errors so governance operations never fail because the
    cache backend is unavailable).
    """
    try:
        from mind_mem.recall_cache import invalidate

        invalidate()
    except Exception as exc:  # pragma: no cover — best-effort
        _log.debug("recall_cache_invalidate_failed", error=str(exc))


@mcp_tool_observe
@_traced("scan")
def scan() -> str:
    """Run integrity scan — contradictions, drift, dead decisions, impact graph."""
    ws = _workspace()
    ws_err = _check_workspace(ws)
    if ws_err:
        return ws_err

    checks: dict[str, Any] = {}

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

    drift_path = os.path.join(ws, "intelligence", "DRIFT.md")
    if os.path.isfile(drift_path):
        drifts = parse_file(drift_path)
        checks["drift_items"] = len(drifts)
    else:
        checks["drift_items"] = 0

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


@mcp_tool_observe
def list_contradictions() -> str:
    """List detected contradictions with resolution analysis."""
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

    enriched: list[dict] = []
    try:
        from mind_mem.auto_resolver import AutoResolver

        suggestions = AutoResolver(ws).suggest_resolutions()
        by_id = {s.contradiction_id: s for s in suggestions}
        for res in resolutions:
            sug = by_id.get(str(res.get("contradiction_id", "")))
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


@mcp_tool_observe
def approve_apply(proposal_id: str, dry_run: bool = True) -> str:
    """Apply a staged proposal from intelligence/proposed/."""
    ws = _workspace()

    import re

    if not re.match(r"^P-\d{8}-\d{3}$", proposal_id):
        return json.dumps({"error": f"Invalid proposal ID format: {proposal_id}. Expected P-YYYYMMDD-NNN."})

    import contextlib
    import io

    from mind_mem.apply_engine import apply_proposal, find_proposal
    from mind_mem.contradiction_detector import check_proposal_contradictions

    contra_report = None
    try:
        proposal, _source = find_proposal(ws, proposal_id)
        if proposal:
            contra_report = check_proposal_contradictions(ws, proposal)
    except Exception as e:
        _log.warning("contradiction_check_failed", error=str(e))

    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        success, message = apply_proposal(ws, proposal_id, dry_run=dry_run)

    log_output = capture.getvalue()

    metrics.inc("mcp_apply_calls")
    _log.info("mcp_approve_apply", proposal_id=proposal_id, dry_run=dry_run, success=success)

    # v3.2.1: invalidate recall cache only on a real (non-dry-run) apply.
    if success and not dry_run:
        _invalidate_recall_cache()

    blocked_by_contradictions = not success and message == "Blocked: contradictions detected"

    result: dict[str, Any] = {
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

    if contra_report:
        result["contradictions"] = {
            "summary": contra_report["summary"],
            "has_contradictions": contra_report["has_contradictions"],
            "contradiction_count": contra_report["contradiction_count"],
            "total_conflicts": contra_report["total_conflicts"],
            "conflicts": contra_report["conflicts"],
        }

    return json.dumps(result, indent=2)


@mcp_tool_observe
def rollback_proposal(receipt_ts: str) -> str:
    """Rollback an applied proposal using its receipt timestamp."""
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

    # v3.2.1: post-rollback cache flush so recall sees the restored state.
    if success:
        _invalidate_recall_cache()

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


@mcp_tool_observe
def memory_evolution(block_id: str, action: str = "get") -> str:
    """A-MEM metadata for a block — importance, access patterns, keywords."""
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


def register(mcp) -> None:
    """Wire the governance tools onto *mcp*."""
    mcp.tool(propose_update)
    mcp.tool(scan)
    mcp.tool(list_contradictions)
    mcp.tool(approve_apply)
    mcp.tool(rollback_proposal)
    mcp.tool(memory_evolution)
