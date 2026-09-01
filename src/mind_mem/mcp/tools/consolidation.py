"""Memory-consolidation MCP tools.

Extracted from ``mcp_server.py`` per docs/v3.2.0-mcp-decomposition-plan.md
(PR-3 slice, consolidation domain). Four tools cover the "memory
settles over time" surface:

* ``plan_consolidation`` — dry-run of the cognitive-forgetting cycle,
  optionally carrying a proposal-only ``granularity_align`` section
  (``v4.granularity_align``, default OFF).
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


def _index_db_path(ws: str) -> str:
    """Absolute path of the recall index DB for workspace *ws*.

    Routes through :mod:`mind_mem.sqlite_index`, which owns ``DB_REL_PATH``
    and is the only writer of this database. These tools used to join their
    own directory/filename literal, a layout nothing in the product ever
    writes, so every read below found no file and the tools reported an
    empty corpus while still answering ``success``. Deriving the path from
    the writer keeps reader and writer from drifting apart again.

    Imported lazily to keep this module import-cheap, matching the rest of
    the file.
    """
    from mind_mem.sqlite_index import _db_path

    return _db_path(ws)


@mcp_tool_observe
def plan_consolidation(
    importance_threshold: float = 0.25,
    stale_days: int = 14,
    archive_after_days: int = 60,
    grace_days: int = 30,
    maturity_gate: bool = False,
    min_maturity: float = 0.5,
) -> str:
    """Dry-run the cognitive forgetting cycle.

    ``maturity_gate`` is **off by default**; with it off this tool's JSON is
    byte-for-byte what it was before the gate existed. Turning it on holds
    back blocks below ``min_maturity`` and any block on a live contradiction,
    and adds a ``maturity_gate`` report section to the response.

    A second, independently gated section — ``granularity_align`` — appears
    only when ``v4.granularity_align`` is on in the workspace's (or the
    ambient) ``mind-mem.json``. It lists pairs of blocks that say the same
    thing at different levels of abstraction, together with the merged block
    each pair would collapse to. It is **proposal-only**: the section is data,
    nothing is written, and a merge reaches the corpus only by being routed
    through ``propose_update`` and executed by ``approve_apply``. With the
    flag off the section is absent and no extra byte is read or emitted.
    """
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

    db_path = _index_db_path(ws)
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
    else:
        _log.warning("consolidation_index_missing", tool="plan_consolidation", path=db_path)

    payload: dict[str, Any] = {
        "config": {
            "importance_threshold": cfg.importance_threshold,
            "stale_days": cfg.stale_days,
            "archive_after_days": cfg.archive_after_days,
            "grace_days": cfg.grace_days,
        },
        "plan": None,
        "_schema_version": "1.0",
    }

    # v4.granularity_align plug point. The probe below is quiet and
    # fail-closed: with the flag off nothing is read from the corpus, no key
    # is added, and the JSON stays byte-for-byte what it was.
    granularity = _granularity_settings(ws)
    granularity_on = isinstance(granularity, dict) and granularity.get("enabled") is True

    if not maturity_gate:
        # Default path — no gate object is ever built, so the response is
        # identical to the pre-gate implementation.
        payload["plan"] = _plan(blocks, config=cfg).as_dict()
        if granularity_on:
            payload["granularity_align"] = _granularity_section(db_path, granularity)
        return json.dumps(payload, indent=2)

    from mind_mem.consolidation_maturity_gate import (
        MaturityGate,
        MaturityGateConfig,
        collect_contradicted_block_ids,
    )

    try:
        gate_cfg = MaturityGateConfig(enabled=True, min_maturity=float(min_maturity))
    except (TypeError, ValueError) as exc:
        return json.dumps({"error": str(exc)})

    gate = MaturityGate(
        gate_cfg,
        block_meta=_load_block_meta(db_path),
        contradicted_ids=collect_contradicted_block_ids(ws),
    )
    decision = gate.evaluate(blocks)
    payload["plan"] = _plan(blocks, config=cfg, gate=gate).as_dict()
    payload["maturity_gate"] = {"min_maturity": gate_cfg.min_maturity, **decision.as_dict()}
    if granularity_on:
        payload["granularity_align"] = _granularity_section(db_path, granularity)
    return json.dumps(payload, indent=2)


def _load_block_meta(db_path: str) -> dict[str, dict[str, Any]]:
    """Read the maturity-relevant frontmatter fields for every indexed block.

    Only runs when the maturity gate is enabled. Never raises: an unreadable
    index degrades to an empty mapping, which scores every block low and
    therefore *holds* it — the conservative direction for a protective gate.
    """
    import sqlite3 as _sqlite3

    meta: dict[str, dict[str, Any]] = {}
    if not os.path.isfile(db_path):
        return meta
    try:
        conn = _sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30.0)
        conn.row_factory = _sqlite3.Row
        try:
            rows = conn.execute("SELECT id, status, json_blob FROM blocks").fetchall()
        finally:
            conn.close()
    except _sqlite3.Error as exc:
        _log.warning("maturity_gate_meta_read_failed", error=str(exc))
        return meta

    for r in rows:
        try:
            raw = json.loads(r["json_blob"] or "{}")
        except (json.JSONDecodeError, TypeError, ValueError):
            raw = {}
        entry: dict[str, Any] = dict(raw) if isinstance(raw, dict) else {}
        entry["Status"] = r["status"] or entry.get("Status", "")
        meta[r["id"]] = entry
    return meta


# ---------------------------------------------------------------------------
# v4.granularity_align — proposal-only merge candidates (default OFF)
# ---------------------------------------------------------------------------

#: v4 feature flag gating the ``granularity_align`` section of the plan.
_GRANULARITY_FLAG = "granularity_align"

#: Frontmatter fields, in a fixed order, whose text carries the claim a block
#: makes. Fixed order because the concatenation feeds a similarity score, and a
#: score that depended on dict iteration order would not be reproducible.
_GRANULARITY_TEXT_FIELDS = ("Statement", "Title", "Summary", "Description", "Context", "Name")

#: Default ceiling on blocks compared per call. Candidate detection is O(n^2)
#: in the number of blocks, so an unbounded scan would turn one MCP call into a
#: multi-minute pin of the server on a large corpus. Raise it deliberately via
#: the flag's ``max_blocks`` key, with that cost in mind.
_GRANULARITY_MAX_BLOCKS = 400

#: Default ceiling on returned candidates (the module's own default is 50).
_GRANULARITY_MAX_CANDIDATES = 20


def _granularity_settings(ws: str) -> dict[str, Any]:
    """Read ``v4.granularity_align`` for *ws* — fail-closed and QUIET.

    Workspace config first, ambient config second, exactly as
    :func:`mind_mem.maintenance_migrate.flag_enabled` resolves its own flag:
    this tool's whole subject is one explicit workspace directory, so that
    directory's ``mind-mem.json`` outranks whatever the process environment
    happens to point at — while ``MIND_MEM_CONFIG`` still works for a caller
    that sets it.

    Deliberately does NOT call ``feature_flags.is_enabled``. That helper logs
    ``v4_config_unreadable`` when the config will not parse, and this probe
    runs on the DEFAULT path of the tool — so with the flag off and a
    malformed ``mind-mem.json`` the wired build would emit a line the unwired
    build does not. A probe deciding whether a feature is on must not itself
    be observable when the answer is no; ``flag_config(..., quiet=True)`` is
    the same lookup with the logging and the warning-dedup bookkeeping
    skipped.

    Returns the raw sub-config (possibly empty). The caller applies the
    canonical ``{"enabled": true}`` test, so a bare ``true`` — or any other
    truthy shape — still cannot switch the surface on.
    """
    try:
        with open(os.path.join(ws, "mind-mem.json"), encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        data = None
    if isinstance(data, dict):
        block = data.get("v4")
        if isinstance(block, dict) and isinstance(block.get(_GRANULARITY_FLAG), dict):
            return dict(block[_GRANULARITY_FLAG])

    try:
        from mind_mem.v4.feature_flags import flag_config

        ambient = flag_config(_GRANULARITY_FLAG, quiet=True)
    except Exception:
        # Unimportable registry, unreadable config, a non-dict v4 block: every
        # one of them means "off", and none of them may be announced here.
        return {}
    return dict(ambient) if isinstance(ambient, dict) else {}


def _bounded_float(raw: Any, default: float, low: float, high: float) -> float:
    """Coerce a config value into ``[low, high]``, falling back to *default*."""
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    if value != value:  # NaN — comparisons below would silently pass it through
        return default
    return max(low, min(high, value))


def _bounded_int(raw: Any, default: int, low: int, high: int) -> int:
    """Coerce a config value into ``[low, high]``, falling back to *default*."""
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, value))


def _load_granularity_blocks(db_path: str, limit: int) -> list[dict[str, Any]]:
    """Read the block text granularity alignment compares. Flag-gated caller only.

    Top-level blocks only (``parent_id = ''``): the sub-blocks are extracted
    fact cards, and proposing a merge of two fact cards would propose editing
    something no file holds.

    ``ORDER BY id LIMIT ?`` — a deterministic prefix, so the same corpus and
    the same cap always compare the same blocks. No clock is read here or
    anywhere below it.

    Never raises: an index that cannot be read degrades to an empty list, and
    an empty list yields no candidates, which is the safe direction for a
    surface whose output is a proposal.

    **Admission-filtered.** The rows go through ``admit_corpus`` before any
    text is read out of them. Without that this leg surfaced QUARANTINED and
    PENDING block text verbatim through ``plan_consolidation`` -- a USER-scope
    MCP tool -- which is a quarantine bypass: untrusted content became readable
    without ever passing admission. Selecting ``status`` is not filtering on
    it. Every other block-reading leg in the package calls ``admit_corpus``
    (entity_prefetch, graph_recall, kg_fusion, _recall_core, hybrid_recall);
    this one is now one of them.
    """
    import sqlite3 as _sqlite3

    blocks: list[dict[str, Any]] = []
    if not os.path.isfile(db_path):
        return blocks
    try:
        conn = _sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30.0)
        conn.row_factory = _sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT id, status, tags, json_blob FROM blocks WHERE parent_id = '' ORDER BY id LIMIT ?",
                (limit,),
            ).fetchall()
        finally:
            conn.close()
    except _sqlite3.Error as exc:
        _log.warning("granularity_align_read_failed", error=str(exc))
        return blocks

    # ADMISSION GATE. Selecting `status` is not filtering on it: without this
    # call the leg surfaced QUARANTINED and PENDING block text verbatim through
    # plan_consolidation, a USER-scope MCP tool -- untrusted content readable
    # without ever passing admission. Every other block-reading leg in the
    # package filters here (entity_prefetch, graph_recall, kg_fusion,
    # _recall_core, hybrid_recall); this one is now one of them.
    from mind_mem.admissibility import admit_corpus

    admitted = admit_corpus([{"_id": r["id"], "Status": r["status"], "_row": r} for r in rows])

    for _entry in admitted:
        r = _entry["_row"]
        try:
            raw = json.loads(r["json_blob"] or "{}")
        except (json.JSONDecodeError, TypeError, ValueError):
            raw = {}
        if not isinstance(raw, dict):
            raw = {}
        text = " ".join(str(raw[field]) for field in _GRANULARITY_TEXT_FIELDS if isinstance(raw.get(field), str) and raw[field])
        entry: dict[str, Any] = {
            "_id": r["id"],
            "content": text,
            "tags": r["tags"] or str(raw.get("Tags") or ""),
            "Status": r["status"] or str(raw.get("Status") or ""),
        }
        if "Maturity" in raw:
            entry["Maturity"] = raw["Maturity"]
        blocks.append(entry)
    return blocks


def _granularity_section(db_path: str, settings: dict[str, Any]) -> dict[str, Any]:
    """Build the proposal-only merge-candidate section of the plan.

    Runs only with ``v4.granularity_align`` on. Every value here is a pure
    function of (indexed blocks, settings): :mod:`mind_mem.granularity_align`
    reads no clock and draws no randomness, and the candidate order is the
    module's own deterministic sort.

    **This function never writes.** It reads the index read-only and returns
    data. ``applied`` is a constant ``false`` and ``route`` names the only
    path a merge may take to the corpus — ``propose_update`` then
    ``approve_apply`` — because the merged block in each entry is a
    *suggestion*, and the HITL gate is what turns a suggestion into a write.
    """
    from mind_mem.granularity_align import (
        DEFAULT_MIN_SIMILARITY,
        find_merge_candidates,
        merge_blocks,
    )

    min_similarity = _bounded_float(settings.get("min_similarity", DEFAULT_MIN_SIMILARITY), DEFAULT_MIN_SIMILARITY, 0.0, 1.0)
    max_candidates = _bounded_int(settings.get("max_candidates", _GRANULARITY_MAX_CANDIDATES), _GRANULARITY_MAX_CANDIDATES, 0, 500)
    max_blocks = _bounded_int(settings.get("max_blocks", _GRANULARITY_MAX_BLOCKS), _GRANULARITY_MAX_BLOCKS, 1, 5000)

    blocks = _load_granularity_blocks(db_path, max_blocks)
    candidates = find_merge_candidates(blocks, min_similarity=min_similarity, max_candidates=max_candidates)

    entries: list[dict[str, Any]] = []
    for cand in candidates:
        merged = merge_blocks(cand.block_a, cand.block_b, strategy=cand.suggested_strategy)
        entry = cand.to_dict()
        entry["merged"] = {
            "block_id": str(merged.get("_id", "")),
            "merged_from": [str(bid) for bid in merged.get("_merged_from", [])],
            "tags": str(merged.get("tags", "")),
            "statement": str(merged.get("content") or merged.get("excerpt") or "")[:500],
        }
        entries.append(entry)

    return {
        "min_similarity": min_similarity,
        "max_candidates": max_candidates,
        "max_blocks": max_blocks,
        "scanned_blocks": len(blocks),
        "truncated": len(blocks) >= max_blocks,
        "candidates": entries,
        "applied": False,
        "route": "propose_update -> approve_apply",
    }


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
    db_path = _index_db_path(ws)
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
    else:
        _log.warning("consolidation_index_missing", tool="propagate_staleness", path=db_path)

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
    db_path = _index_db_path(ws)
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
    else:
        _log.warning("consolidation_index_missing", tool="project_profile", path=db_path)

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
