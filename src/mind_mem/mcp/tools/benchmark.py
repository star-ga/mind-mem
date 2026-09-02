"""Benchmark + category-summary MCP tools.

Extracted from ``mcp_server.py`` per docs/v3.2.0-mcp-decomposition-plan.md
(PR-3 slice, benchmark domain). Two tools live here:

* ``governance_health_bench`` — exercises contradiction detection,
  audit completeness, drift, and scalability probes.
* ``category_summary`` — category distiller lookup driven by the
  configurable ``max_category_results`` limit and the
  ``categories.extra_categories`` config block.
"""

from __future__ import annotations

import json
import re
from typing import Any

from mind_mem.admission import admit_read

from ..infra.config import _get_limits, _load_extra_categories
from ..infra.constants import MCP_SCHEMA_VERSION
from ..infra.observability import mcp_tool_observe
from ..infra.workspace import _workspace
from ._helpers import get_logger, metrics

_log = get_logger("mcp_server")

#: A category file's per-block section header, written by
#: ``CategoryDistiller._write_category_file`` as ``### <block_id>``.
_SECTION_HEADER = re.compile(r"^### (\S+)[ \t]*$", re.MULTILINE)

#: Status stamped on a section whose block id resolves to nothing in the
#: corpus. It is deliberately not a status anybody recognises, so
#: ``is_admissible_status`` (an allow-list) withholds it. That is what makes
#: a FORGED section fail closed: a quarantined block whose Statement text
#: contains a ``### D-FAKE`` line would otherwise split into a section of its
#: own carrying whatever status it cared to print.
_UNRESOLVED_STATUS = "__unresolved__"


def _category_sections(context: str) -> tuple[str, list[tuple[str, str]]]:
    """Split a rendered category context into a preamble and block sections.

    Returns ``(preamble, [(block_id, section_text), ...])``. The preamble is
    everything before the first ``### `` header — the category titles and the
    distiller's own banner, which carry no block content.
    """
    matches = list(_SECTION_HEADER.finditer(context))
    if not matches:
        return context, []
    sections: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(context)
        sections.append((match.group(1), context[match.start() : end]))
    return context[: matches[0].start()], sections


def _admit_category_context(workspace: str, context: str) -> tuple[str, int]:
    """Drop the sections of *context* whose blocks are not servable.

    Category files are a DERIVED artifact: the distiller reads the corpus and
    writes each block's ``Statement`` into ``categories/<name>.md``. It does
    not filter on admission, so a quarantined inbox drop's statement is copied
    into a file this tool then serves verbatim to a USER-scope caller — the
    same leak as ``get_block``, one artifact removed. (The root fix belongs in
    the distiller, which should categorise only the admitted corpus; this is
    the egress half, and it holds even for category files written before that
    lands or by an older release.)

    The status printed in the file is NOT trusted — it is a copy that went
    stale the moment a block was quarantined, and it sits inside attacker-
    supplied text. Every section id is re-resolved against the block store and
    the answer comes from :func:`admit_read`, the same seam recall uses.
    """
    preamble, sections = _category_sections(context)
    if not sections:
        return context, 0

    from mind_mem.storage import get_block_store

    live = {str(block.get("_id")): block.get("Status") for block in get_block_store(workspace).get_all(active_only=False)}
    rows = [{"_id": block_id, "Status": live.get(block_id, _UNRESOLVED_STATUS)} for block_id, _ in sections]
    decision = admit_read(rows, workspace=workspace, surface="category_summary")
    servable = {str(row["_id"]) for row in decision.admitted}
    kept = "".join(text for block_id, text in sections if block_id in servable)
    return preamble + kept, decision.withheld


def _kind_summaries_section(ws: str) -> list[dict] | None:
    """The ``v4.kind_summaries`` table of contents, or ``None`` when OFF.

    ``category_summary`` answers "what does this workspace know about X" from
    the category distiller's thematic files. ``v4.kind_summaries`` answers the
    same question along the other axis — one summary per BLOCK KIND, the
    GraphRAG-style per-domain table of contents — so this is where it belongs.

    ONE flag read per tool call, and a QUIET one: ``is_enabled_quiet`` never
    logs, so with the flag off this tool is byte-for-byte the 5.0.0 tool, with
    no config parse inside any loop and no line in the log that the unwired
    build did not emit. Read-only — ``list_summaries`` never writes; the
    refresh side is ``mm kinds backfill``.
    """
    try:
        from mind_mem.v4.feature_flags import is_enabled_quiet

        if not is_enabled_quiet("kind_summaries"):
            return None
        from mind_mem.v4.kind_summaries import list_summaries

        return [
            {
                "kind": s.kind,
                "block_count": s.block_count,
                "updated_at": s.updated_at,
                "summary": s.summary,
            }
            for s in list_summaries(ws)
        ]
    except Exception as exc:  # noqa: BLE001 - a sidecar section never breaks the tool
        _log.debug("category_summary_kind_summaries_skipped", error=str(exc))
        return None


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


@mcp_tool_observe
def category_summary(topic: str, limit: int = 3) -> str:
    """Returns category summaries for the ADMITTED blocks matching a topic.

    Uses the category distiller to find thematic summary files matching the
    topic. Those files are derived from the corpus and the distiller copies
    every block's statement into them, quarantined ones included, so the
    per-block sections are re-checked against admission here before anything
    is served and ``withheld_count`` reports how many were dropped.

    Args:
        topic: Topic or query to find relevant categories for.
        limit: Maximum number of category summaries to return (default: 3).

    Returns:
        Concatenated category summaries with block references, plus
        ``withheld_count``.
    """
    ws = _workspace()
    limits = _get_limits(ws)
    try:
        from mind_mem.category_distiller import CategoryDistiller

        extra_cats = _load_extra_categories(ws)
        distiller = CategoryDistiller(extra_categories=extra_cats if extra_cats else None)
        context = distiller.get_category_context(topic, ws, limit=max(1, min(limit, limits["max_category_results"])))
        context, withheld = _admit_category_context(ws, context)
        cats = distiller.get_categories_for_query(topic)
        metrics.inc("mcp_category_summary")
        _log.info("mcp_category_summary", topic=topic, matched_categories=cats[:limit])
        kind_sections = _kind_summaries_section(ws)
        if not context:
            empty: dict = {
                "_schema_version": MCP_SCHEMA_VERSION,
                "topic": topic,
                "status": "no_categories",
                "hint": "Run reindex to generate category files, or add blocks with matching tags.",
            }
            empty["withheld_count"] = withheld
            if kind_sections is not None:
                empty["kind_summaries"] = kind_sections
            return json.dumps(empty, indent=2)
        payload: dict[str, Any] = {
            "_schema_version": MCP_SCHEMA_VERSION,
            "topic": topic,
            "matched_categories": cats[:limit],
            "withheld_count": withheld,
            "content": context,
        }
        if kind_sections is not None:
            payload["kind_summaries"] = kind_sections
        return json.dumps(payload, indent=2)
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


def register(mcp) -> None:
    """Wire the benchmark tools onto *mcp*."""
    mcp.tool(governance_health_bench)
    mcp.tool(category_summary)
