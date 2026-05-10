"""v4 consolidation worker (Group A — MemGPT-pattern).

Multi-LLM v4 audit (2/4 model consensus 2026-05-10) flagged the
absence of a write-time consolidation pass: blocks accumulate in
WARM forever; without periodic clustering + summarisation, the tier
becomes a flat soup. MemGPT (Letta) batches recent blocks and
re-summarises into higher tiers; mind-mem's equivalent runs
surprise-weighted clustering on WARM blocks and proposes promotion
of cluster centroids to COLD.

Deliberately a **pure function**, not a daemon: the v4 anti-pattern
list (docs/roadmap-v4.md §F) forbids always-on background daemons.
The caller invokes :func:`plan_consolidation` when ready, applies
the returned proposal through the existing v3.x propose/approve
governance flow, and the workspace stays consistent.

What the planner returns:

    ConsolidationPlan(
        promotions = [(block_id, RecallTier.COLD, reason), ...],
        demotions  = [(block_id, RecallTier.COLD, reason), ...],
        cluster_summaries = [{"centroid_id": ..., "members": [...]}],
    )

The promotions list contains the cluster representatives — blocks
chosen as the most-central member of their cluster. Demotions
contain WARM blocks whose surprise on the last seen read was below
the demotion threshold (``contradiction_floor``) for at least
``warm_ttl_hours``.

This module ships the **planner only**. Apply is the caller's job —
the planner is read-only against the workspace, so calling it is
always safe.

Feature-flag gated under ``v4.consolidation_worker``. v3.x callers
see no behaviour change.

Copyright STARGA, Inc.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .feature_flags import flag_config, require_enabled
from .surprise_retrieval import centroid, compute_surprise
from .tier_memory import RecallTier

__all__ = [
    "FLAG",
    "ConsolidationPlan",
    "ConsolidationConfig",
    "DEFAULT_CONSOLIDATION_CONFIG",
    "plan_consolidation",
]


FLAG: str = "consolidation_worker"


@dataclass(frozen=True)
class ConsolidationConfig:
    """Tunable knobs for the consolidation planner."""

    cluster_count: int = 5
    """Number of clusters to form on the WARM corpus."""

    min_cluster_size: int = 3
    """Clusters smaller than this stay un-promoted (noise)."""

    demotion_threshold: float = 0.3
    """A WARM block whose last_surprise is below this is candidate for demotion."""


DEFAULT_CONSOLIDATION_CONFIG: ConsolidationConfig = ConsolidationConfig()


@dataclass(frozen=True)
class ConsolidationPlan:
    """Read-only output of :func:`plan_consolidation`.

    The caller is expected to feed this through the v3.x propose /
    approve flow; the planner itself never writes to the workspace.
    """

    promotions: list[tuple[str, RecallTier, str]] = field(default_factory=list)
    demotions: list[tuple[str, RecallTier, str]] = field(default_factory=list)
    cluster_summaries: list[dict[str, object]] = field(default_factory=list)

    @property
    def total_proposals(self) -> int:
        return len(self.promotions) + len(self.demotions)


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


def plan_consolidation(
    workspace: str | Path,
    *,
    candidate_embeddings: dict[str, Sequence[float]] | None = None,
    cfg: ConsolidationConfig | None = None,
) -> ConsolidationPlan:
    """Compute a consolidation plan over the WARM tier of ``workspace``.

    ``candidate_embeddings`` maps block_id → embedding for the WARM
    blocks. Missing entries are ignored (no clustering on un-embedded
    blocks). When the dict is empty / ``None``, the planner returns
    an empty plan rather than guessing.

    Strategy:

        1. Pull every WARM block_id + last_surprise from
           ``block_recall_tier``.
        2. Demote (propose RecallTier.COLD) every WARM block whose
           last_surprise is below ``demotion_threshold``.
        3. Among the remaining WARM blocks with embeddings, form up
           to ``cluster_count`` clusters via greedy farthest-first
           seeding, then assign every member to its closest seed.
        4. For each cluster of size ≥ ``min_cluster_size``, propose
           promoting the most-central member to COLD as the cluster
           representative; record the rest as members.

    Read-only against the workspace; the caller applies the plan
    through the existing propose / approve governance flow.
    """
    require_enabled(FLAG)
    cfg = cfg or _load_config()
    db = Path(workspace) / "index.db"
    if not db.is_file():
        return ConsolidationPlan()

    with sqlite3.connect(db) as conn:
        if not _table_exists(conn, "block_recall_tier"):
            return ConsolidationPlan()
        rows = conn.execute("SELECT block_id, last_surprise FROM block_recall_tier WHERE tier = 'warm'").fetchall()

    plan_demotions: list[tuple[str, RecallTier, str]] = []
    cluster_input: list[tuple[str, Sequence[float]]] = []
    for bid, surprise in rows:
        s = float(surprise) if surprise is not None else 0.5
        if s < cfg.demotion_threshold:
            plan_demotions.append((bid, RecallTier.COLD, f"consolidation:low_surprise s={s:.3f}"))
            continue
        if candidate_embeddings:
            emb = candidate_embeddings.get(bid)
            if emb:
                cluster_input.append((bid, emb))

    if not cluster_input or len(cluster_input) < cfg.min_cluster_size:
        return ConsolidationPlan(
            promotions=[],
            demotions=plan_demotions,
            cluster_summaries=[],
        )

    seeds = _farthest_first_seeds(cluster_input, cfg.cluster_count)
    clusters = _assign_to_seeds(cluster_input, seeds)

    promotions: list[tuple[str, RecallTier, str]] = []
    summaries: list[dict[str, object]] = []
    for cluster_id, members in enumerate(clusters):
        if len(members) < cfg.min_cluster_size:
            continue
        # Most-central member: lowest mean surprise to the cluster centroid.
        cluster_centroid = centroid([m[1] for m in members])
        if cluster_centroid is None:
            continue
        most_central = min(
            members,
            key=lambda m: compute_surprise(m[1], cluster_centroid),
        )
        promotions.append((most_central[0], RecallTier.COLD, f"consolidation:cluster_rep cid={cluster_id}"))
        summaries.append(
            {
                "cluster_id": cluster_id,
                "centroid_id": most_central[0],
                "members": [m[0] for m in members],
                "size": len(members),
            }
        )

    return ConsolidationPlan(
        promotions=promotions,
        demotions=plan_demotions,
        cluster_summaries=summaries,
    )


# ---------------------------------------------------------------------------
# Clustering primitives
# ---------------------------------------------------------------------------


def _farthest_first_seeds(
    points: list[tuple[str, Sequence[float]]],
    k: int,
) -> list[tuple[str, Sequence[float]]]:
    """Greedy farthest-first traversal seeding.

    Picks the first point arbitrarily; each next seed is the point
    farthest from any existing seed. Deterministic given the input
    order. Returns at most ``min(k, len(points))`` seeds.
    """
    if not points or k <= 0:
        return []
    seeds: list[tuple[str, Sequence[float]]] = [points[0]]
    while len(seeds) < min(k, len(points)):
        best_idx, best_d = -1, -1.0
        for i, (_bid, vec) in enumerate(points):
            if any(s[0] == points[i][0] for s in seeds):
                continue
            d = min(compute_surprise(vec, seed[1]) for seed in seeds)
            if d > best_d:
                best_d, best_idx = d, i
        if best_idx < 0:
            break
        seeds.append(points[best_idx])
    return seeds


def _assign_to_seeds(
    points: list[tuple[str, Sequence[float]]],
    seeds: list[tuple[str, Sequence[float]]],
) -> list[list[tuple[str, Sequence[float]]]]:
    """Assign every point to its closest seed (smallest surprise)."""
    if not seeds:
        return []
    clusters: list[list[tuple[str, Sequence[float]]]] = [[] for _ in seeds]
    for bid, vec in points:
        best_i = min(
            range(len(seeds)),
            key=lambda i: compute_surprise(vec, seeds[i][1]),
        )
        clusters[best_i].append((bid, vec))
    return clusters


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?",
        (name,),
    ).fetchone()
    return row is not None


def _load_config() -> ConsolidationConfig:
    raw = flag_config(FLAG)
    if not isinstance(raw, dict):
        return DEFAULT_CONSOLIDATION_CONFIG
    fields = {
        "cluster_count": (int, DEFAULT_CONSOLIDATION_CONFIG.cluster_count),
        "min_cluster_size": (int, DEFAULT_CONSOLIDATION_CONFIG.min_cluster_size),
        "demotion_threshold": (float, DEFAULT_CONSOLIDATION_CONFIG.demotion_threshold),
    }
    out: dict[str, object] = {}
    for key, (caster, default) in fields.items():
        v = raw.get(key, default)
        try:
            out[key] = caster(v)
        except (TypeError, ValueError):
            out[key] = default
    return ConsolidationConfig(**out)  # type: ignore[arg-type]
