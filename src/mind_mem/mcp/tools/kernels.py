"""MIND kernel + compiled-truth MCP tools.

Extracted from ``mcp_server.py`` per docs/v3.2.0-mcp-decomposition-plan.md
(PR-3 slice, kernels domain). Five tools per the plan's mapping:

* ``list_mind_kernels`` / ``get_mind_kernel`` — read-only access
  to the ``.mind`` kernel files that tune recall, reranking,
  RM3 expansion, and related pipeline knobs.
* ``compiled_truth_load`` / ``compiled_truth_add_evidence`` /
  ``compiled_truth_contradictions`` — per-entity truth pages that
  aggregate evidence and recompile their canonical understanding.
"""

from __future__ import annotations

import json
import os
import re as _re_mod

from mind_mem.mind_ffi import get_mind_dir, load_all_kernel_configs, load_kernel_config
from ._helpers import get_logger, metrics

from ..infra.constants import MCP_SCHEMA_VERSION
from ..infra.observability import mcp_tool_observe
from ..infra.workspace import _workspace

_log = get_logger("mcp_server")


@mcp_tool_observe
def list_mind_kernels() -> str:
    """List available .mind kernel configuration files."""
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


@mcp_tool_observe
def get_mind_kernel(name: str) -> str:
    """Read a specific .mind kernel configuration as structured JSON."""
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


@mcp_tool_observe
def compiled_truth_load(entity_id: str) -> str:
    """Load a compiled truth page for an entity."""
    ws = _workspace()

    try:
        from mind_mem.compiled_truth import load_truth_page

        page = load_truth_page(ws, entity_id)
    except Exception as exc:
        return json.dumps(
            {
                "_schema_version": MCP_SCHEMA_VERSION,
                "error": f"Failed to load truth page: {exc}",
            }
        )

    if page is None:
        return json.dumps(
            {
                "_schema_version": MCP_SCHEMA_VERSION,
                "error": f"No compiled truth page found for '{entity_id}'.",
                "hint": "Create one with compiled_truth_add_evidence.",
            }
        )

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


@mcp_tool_observe
def compiled_truth_add_evidence(
    entity_id: str,
    observation: str,
    source: str = "mcp_tool",
    confidence: str = "medium",
    entity_type: str = "topic",
) -> str:
    """Add evidence to a compiled truth page and auto-recompile."""
    ws = _workspace()

    try:
        from datetime import datetime, timezone

        from mind_mem.compiled_truth import (
            CompiledTruthPage,
            EvidenceEntry,
            add_evidence,
            load_truth_page,
            recompile_truth,
            save_truth_page,
        )

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
        return json.dumps(
            {
                "_schema_version": MCP_SCHEMA_VERSION,
                "error": f"Failed to add evidence: {exc}",
            }
        )

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


@mcp_tool_observe
def compiled_truth_contradictions(entity_id: str) -> str:
    """Detect contradictions in a compiled truth page."""
    ws = _workspace()

    try:
        from mind_mem.compiled_truth import detect_contradictions, load_truth_page

        page = load_truth_page(ws, entity_id)
        if page is None:
            return json.dumps(
                {
                    "_schema_version": MCP_SCHEMA_VERSION,
                    "error": f"No compiled truth page found for '{entity_id}'.",
                }
            )

        conflicts = detect_contradictions(page)
    except Exception as exc:
        return json.dumps(
            {
                "_schema_version": MCP_SCHEMA_VERSION,
                "error": f"Failed to detect contradictions: {exc}",
            }
        )

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


def register(mcp) -> None:
    """Wire the kernel + compiled-truth tools onto *mcp*."""
    mcp.tool(list_mind_kernels)
    mcp.tool(get_mind_kernel)
    mcp.tool(compiled_truth_load)
    mcp.tool(compiled_truth_add_evidence)
    mcp.tool(compiled_truth_contradictions)
