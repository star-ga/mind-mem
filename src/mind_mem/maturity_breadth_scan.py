# Copyright 2026 STARGA, Inc.
"""What would the breadth rebalance do to this corpus? (Group S)

:func:`mind_mem.block_maturity.maturity_score` gained a corroboration
**breadth** component whose weight is taken out of the edge component's
rather than added on top.  Which of the two weight profiles applies is
decided per call by whether the caller supplies
``distinct_project_count``, so nothing about the live default moved when
the component landed.

Before anyone proposes moving it, this scan answers the question that has
to be answered first: *what would scoring every block under the
breadth-aware profile actually change?*  It reads a workspace, scores
every block both ways, and reports the delta distribution.

It is **strictly read-only**.  It opens the recall index with
``mode=ro``, writes nothing, and changes no default — running it is
always safe, including against a live store.

That is also why it issues its own SQL instead of calling
:func:`mind_mem.block_lineage.distinct_project_counts`, which would be
the obvious reuse: every reader in that module first calls
``ensure_lineage_schema``, so asking it a question *migrates the store*.
Measured on a fresh index — the scan leaves the database bytes
untouched, the reader changes them.  Do not "simplify" this back.

    from mind_mem.maturity_breadth_scan import scan_maturity_breadth
    report = scan_maturity_breadth("/path/to/workspace")

    $ python -m mind_mem.maturity_breadth_scan --workspace /path/to/workspace
    $ mind-mem-breadth-scan -w /path/to/workspace --limit 20

Reading the report
------------------
``blocks_scored`` counts blocks that carry at least one incoming lineage
edge; blocks with none score identically under both profiles (both
corroboration terms are zero), so they are counted in ``blocks_total``
and excluded from the delta statistics.

A large negative ``mean_delta`` with ``edges_with_provenance`` at zero is
the expected reading on a store whose edges all predate the provenance
column — it says the rebalance would deflate every corroborated block
purely for lack of provenance, which is an argument for backfilling
provenance, not for moving the default.

Stdlib only (``sqlite3``, ``json``, ``argparse``).
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from dataclasses import asdict, dataclass, field
from typing import Any

from .block_maturity import (
    MATURITY_EDGE_WEIGHT,
    MATURITY_EDGE_WEIGHT_WITH_BREADTH,
    MATURITY_PROJECT_SATURATION,
    MATURITY_PROJECT_WEIGHT,
    maturity_score,
)
from .observability import get_logger

__all__ = [
    "BlockDelta",
    "BreadthScanReport",
    "main",
    "scan_maturity_breadth",
]

_log = get_logger("maturity_breadth_scan")


@dataclass(frozen=True)
class BlockDelta:
    """One block's score under both profiles."""

    block_id: str
    incoming_edges: int
    distinct_projects: int
    current_score: float
    rebalanced_score: float

    @property
    def delta(self) -> float:
        return self.rebalanced_score - self.current_score


@dataclass(frozen=True)
class BreadthScanReport:
    """Corpus-wide summary of the hypothetical rebalance."""

    workspace: str
    blocks_total: int = 0
    blocks_scored: int = 0
    blocks_changed: int = 0
    blocks_improved: int = 0
    blocks_reduced: int = 0
    edges_total: int = 0
    edges_with_provenance: int = 0
    distinct_projects_seen: int = 0
    mean_delta: float = 0.0
    min_delta: float = 0.0
    max_delta: float = 0.0
    weights: dict[str, float] = field(default_factory=dict)
    samples: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _db_path(workspace: str) -> str:
    return os.path.join(os.path.abspath(workspace), ".mind-mem-index", "recall.db")


def _open_readonly(path: str) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=10.0)


def _block_metadata(conn: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    """Read ``block_id -> scoring fields`` from the index, best effort."""
    meta: dict[str, dict[str, Any]] = {}
    try:
        rows = conn.execute("SELECT id, status, json_blob FROM blocks").fetchall()
    except sqlite3.Error as exc:
        _log.warning("blocks_table_unavailable", error=str(exc))
        return meta

    for block_id, status, blob in rows:
        fields: dict[str, Any] = {"Status": status or ""}
        if blob:
            try:
                parsed = json.loads(blob)
            except (ValueError, TypeError):
                parsed = None
            if isinstance(parsed, dict):
                inner = parsed.get("metadata")
                if isinstance(inner, dict):
                    for key in ("Status", "Lifecycle", "Maturity"):
                        if key in inner:
                            fields[key] = inner[key]
        meta[str(block_id)] = fields
    return meta


@dataclass(frozen=True)
class _EdgeFacts:
    """What the lineage graph says about incoming edges."""

    graph_present: bool = False
    edge_counts: dict[str, int] = field(default_factory=dict)
    project_counts: dict[str, int] = field(default_factory=dict)
    edges_total: int = 0
    edges_with_provenance: int = 0
    distinct_projects_seen: int = 0


def _incoming_edges(conn: sqlite3.Connection) -> _EdgeFacts:
    """Per-block incoming edge counts and distinct-project counts.

    A store whose graph table or provenance column does not exist yet
    degrades to empty counts rather than raising — the scan must be able
    to run anywhere, including against an index that has never had a
    lineage edge written to it.
    """
    edge_counts: dict[str, int] = {}
    project_counts: dict[str, int] = {}
    try:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(co_retrieval)").fetchall()}
    except sqlite3.Error as exc:
        _log.warning("co_retrieval_unavailable", error=str(exc))
        return _EdgeFacts()
    if not columns:
        return _EdgeFacts()

    has_provenance = "origin_project" in columns
    select = "SELECT mem2_id, origin_project FROM co_retrieval" if has_provenance else "SELECT mem2_id, '' FROM co_retrieval"
    try:
        rows = conn.execute(select).fetchall()
    except sqlite3.Error as exc:
        _log.warning("co_retrieval_read_failed", error=str(exc))
        return _EdgeFacts(graph_present=True)

    projects_by_block: dict[str, set[str]] = {}
    all_projects: set[str] = set()
    with_provenance = 0
    for dst, project in rows:
        block_id = str(dst)
        edge_counts[block_id] = edge_counts.get(block_id, 0) + 1
        key = str(project or "")
        if key:
            with_provenance += 1
            projects_by_block.setdefault(block_id, set()).add(key)
            all_projects.add(key)

    project_counts = {block_id: len(keys) for block_id, keys in projects_by_block.items()}
    return _EdgeFacts(
        graph_present=True,
        edge_counts=edge_counts,
        project_counts=project_counts,
        edges_total=len(rows),
        edges_with_provenance=with_provenance,
        distinct_projects_seen=len(all_projects),
    )


def scan_maturity_breadth(
    workspace: str,
    *,
    sample_limit: int = 10,
) -> BreadthScanReport:
    """Report the score delta the breadth rebalance would produce.

    Args:
        workspace: Workspace root (the directory holding
            ``.mind-mem-index/recall.db``).
        sample_limit: How many of the largest-magnitude per-block deltas
            to include in ``samples``.  ``0`` omits them.

    Returns:
        A :class:`BreadthScanReport`.  Nothing is written and no default
        is changed; a missing or unreadable index yields an empty report
        carrying a note, never an exception.
    """
    if not isinstance(workspace, str) or not workspace.strip():
        raise ValueError("workspace must be a non-empty string")

    ws = os.path.abspath(workspace)
    weights = {
        "current_edge_weight": MATURITY_EDGE_WEIGHT,
        "rebalanced_edge_weight": MATURITY_EDGE_WEIGHT_WITH_BREADTH,
        "rebalanced_project_weight": MATURITY_PROJECT_WEIGHT,
        "project_saturation": float(MATURITY_PROJECT_SATURATION),
    }
    notes: list[str] = []

    path = _db_path(ws)
    if not os.path.isfile(path):
        notes.append("no recall index at .mind-mem-index/recall.db — nothing to scan")
        return BreadthScanReport(workspace=ws, weights=weights, notes=notes)

    try:
        conn = _open_readonly(path)
    except sqlite3.Error as exc:
        notes.append(f"recall index unreadable: {exc}")
        return BreadthScanReport(workspace=ws, weights=weights, notes=notes)

    try:
        meta = _block_metadata(conn)
        facts = _incoming_edges(conn)
    finally:
        conn.close()

    if not meta:
        notes.append("no blocks table rows — index may need a rebuild")
    if not facts.graph_present:
        notes.append("no co_retrieval table in this index — the lineage graph has never been written here")
    elif facts.edges_total == 0:
        notes.append("no lineage edges recorded — every block scores identically under both profiles")
    elif facts.edges_with_provenance == 0:
        notes.append("no edge carries provenance yet — breadth counts as zero, so the rebalance can only deflate scores")

    deltas: list[BlockDelta] = []
    for block_id, edges in sorted(facts.edge_counts.items()):
        block = meta.get(block_id)
        if block is None:
            # An edge naming a block the index does not hold. Counted in
            # edges_total, excluded here: scoring a block we cannot read
            # would report the delta of an empty dict, not of the block.
            continue
        projects = facts.project_counts.get(block_id, 0)
        deltas.append(
            BlockDelta(
                block_id=block_id,
                incoming_edges=edges,
                distinct_projects=projects,
                current_score=maturity_score(dict(block), incoming_edge_count=edges),
                rebalanced_score=maturity_score(dict(block), incoming_edge_count=edges, distinct_project_count=projects),
            )
        )

    changed = [d for d in deltas if d.delta != 0.0]
    values = [d.delta for d in deltas]
    samples: list[dict[str, Any]] = []
    if sample_limit > 0 and changed:
        ranked = sorted(changed, key=lambda d: (-abs(d.delta), d.block_id))[:sample_limit]
        samples = [{**asdict(d), "delta": d.delta} for d in ranked]

    return BreadthScanReport(
        workspace=ws,
        blocks_total=len(meta),
        blocks_scored=len(deltas),
        blocks_changed=len(changed),
        blocks_improved=sum(1 for d in changed if d.delta > 0),
        blocks_reduced=sum(1 for d in changed if d.delta < 0),
        edges_total=facts.edges_total,
        edges_with_provenance=facts.edges_with_provenance,
        distinct_projects_seen=facts.distinct_projects_seen,
        mean_delta=(sum(values) / len(values)) if values else 0.0,
        min_delta=min(values) if values else 0.0,
        max_delta=max(values) if values else 0.0,
        weights=weights,
        samples=samples,
        notes=notes,
    )


def main() -> int:
    """CLI entry point — print the scan report as JSON."""
    parser = argparse.ArgumentParser(
        prog="mind-mem-breadth-scan",
        description="Report the maturity-score delta the corroboration-breadth rebalance would produce. Read-only; changes nothing.",
    )
    parser.add_argument("--workspace", "-w", default=".", help="workspace root (default: current directory)")
    parser.add_argument("--limit", type=int, default=10, help="how many largest-magnitude per-block deltas to sample (default: 10)")
    args = parser.parse_args()

    report = scan_maturity_breadth(args.workspace, sample_limit=max(0, args.limit))
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
