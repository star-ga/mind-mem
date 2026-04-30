"""Memory operations MCP tools — index / lifecycle / health / export.

Extracted from ``mcp_server.py`` per docs/v3.2.0-mcp-decomposition-plan.md
(PR-3 slice, memory_ops domain). Eight tools cover the workspace's
lifecycle + introspection surface:

* ``index_stats`` / ``reindex`` — FTS5 index state + rebuild.
* ``delete_memory_item`` — atomic admin-scope block removal with
  an append-only recovery log.
* ``export_memory`` — JSONL dump of every block with configurable
  metadata + size cap.
* ``get_block`` / ``memory_health`` / ``compact`` / ``stale_blocks`` —
  block lookup, health dashboard, compaction, and staleness-flag
  management built on the causal graph.

Also hosts the ``_BLOCK_PREFIX_MAP`` + ``_find_block_file``
resolver shared by ``delete_memory_item`` and ``get_block``.
"""

from __future__ import annotations

import json
import os
import re as _re_mod
import sqlite3
import tempfile
from typing import Any

from mind_mem.block_parser import BlockCorruptedError, get_active, parse_file
from mind_mem.corpus_registry import CORPUS_DIRS
from mind_mem.mind_ffi import get_mind_dir
from mind_mem.mind_ffi import is_available as mind_kernel_available
from mind_mem.mind_ffi import is_protected as mind_kernel_protected
from mind_mem.mind_ffi import list_kernels as ffi_list_kernels
from mind_mem.mind_filelock import FileLock
from mind_mem.sqlite_index import _db_path as fts_db_path

from ..infra.config import _load_extra_categories
from ..infra.constants import MCP_SCHEMA_VERSION
from ..infra.observability import _is_db_locked, _sqlite_busy_error, mcp_tool_observe
from ..infra.workspace import _check_workspace, _workspace
from ._helpers import _signal_store_path, get_logger, metrics

_log = get_logger("mcp_server")


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
    """Resolve a block ID to its source .md file path."""
    for prefix, (subdir, filename) in _BLOCK_PREFIX_MAP.items():
        if block_id.startswith(prefix + "-"):
            return os.path.join(ws, subdir, filename)
    return None


@mcp_tool_observe
def index_stats() -> str:
    """Block counts, index staleness, vector coverage, and MIND kernel status."""
    ws = _workspace()
    stats: dict[str, Any] = {"_schema_version": MCP_SCHEMA_VERSION}

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
            fts_exists = False

    if not fts_exists:
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

    mind_dir = get_mind_dir(ws)
    kernels = ffi_list_kernels(mind_dir)
    stats["mind_kernels"] = kernels
    stats["mind_kernel_compiled"] = mind_kernel_available()
    stats["mind_kernel_protected"] = mind_kernel_protected()

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


@mcp_tool_observe
def reindex(include_vectors: bool = False) -> str:
    """Trigger FTS index rebuild, optionally with vector indexing."""
    ws = _workspace()
    ws_err = _check_workspace(ws)
    if ws_err:
        return ws_err
    results: dict[str, Any] = {"_schema_version": MCP_SCHEMA_VERSION, "fts": False, "vectors": False}

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


@mcp_tool_observe
def delete_memory_item(block_id: str) -> str:
    """Delete a block by ID from its source .md file (admin-scope)."""
    ws = _workspace()
    ws_err = _check_workspace(ws)
    if ws_err:
        return ws_err

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

    with FileLock(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        lines = content.split("\n")
        block_start: int | None = None
        block_end: int | None = None
        block_header = f"[{block_id}]"

        for i, line in enumerate(lines):
            if line.strip() == block_header:
                block_start = i
            elif block_start is not None and block_end is None:
                if line.startswith("[") and line.strip().endswith("]") and _re_mod.match(r"^\[[A-Z]+-", line.strip()):
                    block_end = i
                elif line.strip() == "---":
                    preceding_blank = (i == 0) or (lines[i - 1].strip() == "")
                    if preceding_blank:
                        block_end = i + 1

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

        deleted_content = "\n".join(lines[block_start:block_end])
        new_lines = lines[:block_start] + lines[block_end:]
        new_content = "\n".join(new_lines)

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

        dir_name = os.path.dirname(filepath)
        fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".md.tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as tmp_f:
                tmp_f.write(new_content)
            os.replace(tmp_path, filepath)
        except Exception:
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


@mcp_tool_observe
def export_memory(format: str = "jsonl", include_metadata: bool = False, max_blocks: int = 10000) -> str:
    """Export all workspace blocks as JSONL."""
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

    all_blocks: list[dict] = []

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
                    for key in list(block.keys()):
                        if key.startswith("_") and key not in ("_id", "_source_file"):
                            del block[key]
                all_blocks.append(block)

    truncated = False
    total = len(all_blocks)
    if len(all_blocks) > max_blocks:
        all_blocks = all_blocks[:max_blocks]
        truncated = True
        _log.warning("export_memory_truncated", total=total, max_blocks=max_blocks)

    jsonl_lines = [json.dumps(b, default=str) for b in all_blocks]
    jsonl_output = "\n".join(jsonl_lines)

    metrics.inc("mcp_export_memory")
    _log.info("mcp_export_memory", format=format, blocks=len(all_blocks))

    envelope: dict[str, Any] = {
        "_schema_version": MCP_SCHEMA_VERSION,
        "format": format,
        "block_count": len(all_blocks),
        "data": jsonl_output,
    }
    if truncated:
        envelope["warning"] = f"Output truncated to {max_blocks} blocks (total: {total}). Increase max_blocks to export more."

    return json.dumps(envelope, indent=2)


@mcp_tool_observe
def get_block(block_id: str) -> str:
    """Retrieve a single block by its ID with full content."""
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

    for subdir in CORPUS_DIRS:
        dir_path = os.path.join(ws, subdir)
        if not os.path.isdir(dir_path):
            continue
        for fn in os.listdir(dir_path):
            if not fn.endswith(".md"):
                continue
            fpath = os.path.join(dir_path, fn)
            if fpath == filepath:
                continue
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


@mcp_tool_observe
def memory_health() -> str:
    """Deep health dashboard for the memory workspace."""
    ws = _workspace()
    ws_err = _check_workspace(ws)
    if ws_err:
        return ws_err

    health: dict[str, Any] = {"_schema_version": MCP_SCHEMA_VERSION}
    recommendations: list[str] = []

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


@mcp_tool_observe
def compact(dry_run: bool = True, archive_days: int = 90, signal_days: int = 60, snapshot_days: int = 30) -> str:
    """Run workspace compaction — archive old blocks, clean snapshots, remove resolved signals."""
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

    try:
        block_actions = archive_completed_blocks(ws, days=archive_days, dry_run=dry_run)
        actions["archived_blocks"] = block_actions
    except (OSError, ValueError) as exc:
        actions["archived_blocks_error"] = [str(exc)]
        _log.warning("compact_archive_failed", error=str(exc))

    try:
        snap_actions = cleanup_snapshots(ws, days=snapshot_days, dry_run=dry_run)
        actions["cleaned_snapshots"] = snap_actions
    except (OSError, ValueError) as exc:
        actions["cleaned_snapshots_error"] = [str(exc)]
        _log.warning("compact_snapshots_failed", error=str(exc))

    try:
        signal_actions = compact_signals(ws, days=signal_days, dry_run=dry_run)
        actions["compacted_signals"] = signal_actions
    except (OSError, ValueError) as exc:
        actions["compacted_signals_error"] = [str(exc)]
        _log.warning("compact_signals_failed", error=str(exc))

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


@mcp_tool_observe
def stale_blocks(limit: int = 20, clear_block_id: str = "") -> str:
    """List blocks flagged as stale, or clear a staleness flag."""
    ws = _workspace()

    try:
        from mind_mem.causal_graph import CausalGraph

        cg = CausalGraph(ws)

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


def register(mcp) -> None:
    """Wire the memory_ops tools onto *mcp*."""
    mcp.tool(index_stats)
    mcp.tool(reindex)
    mcp.tool(delete_memory_item)
    mcp.tool(export_memory)
    mcp.tool(get_block)
    mcp.tool(memory_health)
    mcp.tool(compact)
    mcp.tool(stale_blocks)
