"""Tier-aware recall score boosting (v3.2.0 hot/cold tier wire-up).

Links the TierManager memory_tiers.py to the recall scoring path so
blocks in trusted tiers (SHARED / LONG_TERM / VERIFIED) get a score
multiplier that surfaces them above noisy WORKING-tier blocks.

This module is the single integration point — the recall pipeline
calls :func:`apply_tier_boosts` after scoring + reranking; every
other tier-aware code path in the codebase should route through
this function rather than reaching into :mod:`memory_tiers`
directly. That keeps the coupling between recall and tier policy
explicit in one file.

Score multipliers (matching the design note in
``tiered_memory.py::Tier.retrieval_boost``):

    WORKING    → 0.7x (demoted; still surfaces but ranks below trusted)
    SHARED     → 1.0x (neutral)
    LONG_TERM  → 1.5x
    VERIFIED   → 2.0x

Opt-in via ``mind-mem.json`` ``retrieval.tier_boost: true``
(defaults to ``false`` so v3.1.x behaviour is preserved for users
who haven't reviewed the tier policy). A future release will
flip the default once enough workspaces have confirmed the
ranking behaves sensibly.
"""

from __future__ import annotations

import os
from typing import Any

from .observability import get_logger

_log = get_logger("tier_recall")

# Keep in lockstep with ``memory_tiers.MemoryTier`` ordinal values —
# duplicating here avoids pulling the TierManager SQLite state into
# every recall call. The boost only depends on the integer value,
# not the live DB state.
#
# v3.3.0 Tier 4 #10 — ``_TIER_BOOST`` is the baseline. Callers that
# configure ``retrieval.tier_boost_weights`` in ``mind-mem.json`` get
# per-tier override multipliers instead; see
# :func:`resolve_tier_weights` below.
_TIER_BOOST: dict[int, float] = {
    1: 0.7,  # WORKING
    2: 1.0,  # SHARED
    3: 1.5,  # LONG_TERM
    4: 2.0,  # VERIFIED
}


# Canonical tier name → integer mapping so operators can write
# human-readable config without memorising the integer codes.
_TIER_NAME_TO_INT: dict[str, int] = {
    "WORKING": 1,
    "SHARED": 2,
    "LONG_TERM": 3,
    "VERIFIED": 4,
}


def resolve_tier_weights(config: dict[str, Any] | None) -> dict[int, float]:
    """Resolve per-tier boost multipliers from config.

    v3.3.0 Tier 4 #10 — ``retrieval.tier_boost_weights`` lets operators
    hand-tune or learn per-tier multipliers without patching the
    baseline ``_TIER_BOOST`` constant. Config accepts either
    human-readable tier names or integer keys:

    .. code-block:: json

        {
          "retrieval": {
            "tier_boost_weights": {
              "WORKING": 0.6,
              "SHARED": 1.0,
              "LONG_TERM": 1.8,
              "VERIFIED": 2.4
            }
          }
        }

    Missing tier keys fall back to the baseline value. Non-positive or
    non-numeric values are rejected and fall back to the baseline.

    An offline training loop (see ``benchmarks/tier_weight_search.py``
    in the v3.3.0 bench kit) reads ``locomo_judge_results_*.json`` and
    grid-searches the four multipliers against the LoCoMo mean score
    to produce an optimised weight dict the operator pastes into their
    ``mind-mem.json``.
    """
    baseline = dict(_TIER_BOOST)
    if not config or not isinstance(config, dict):
        return baseline
    retrieval = config.get("retrieval", {})
    if not isinstance(retrieval, dict):
        return baseline
    weights = retrieval.get("tier_boost_weights", {})
    if not isinstance(weights, dict):
        return baseline
    out = dict(baseline)
    for key, val in weights.items():
        if isinstance(val, (int, float)) and val > 0:
            tier_int: int | None = None
            if isinstance(key, int) and key in out:
                tier_int = key
            elif isinstance(key, str):
                if key.upper() in _TIER_NAME_TO_INT:
                    tier_int = _TIER_NAME_TO_INT[key.upper()]
                elif key.isdigit() and int(key) in out:
                    tier_int = int(key)
            if tier_int is not None:
                out[tier_int] = float(val)
    return out


def _tier_db_path(workspace: str) -> str:
    """Path to the tier-assignment SQLite db.

    Matches the location used by ``TierManager`` when the workspace
    follows the default layout. Callers that override ``tier_manager.db_path``
    must pass their value directly to :func:`apply_tier_boosts`.
    """
    return os.path.join(workspace, ".sqlite_index", "index.db")


def _load_tier_map(workspace: str) -> dict[str, int]:
    """Return ``{block_id: tier_int}`` for every block with an explicit tier.

    Missing rows default to WORKING (tier 1) at the caller's boost
    lookup — we don't materialize them here so the map stays tight.
    Returns an empty dict when the tier table doesn't exist yet
    (fresh workspace) or the read fails for any reason.
    """
    db_path = _tier_db_path(workspace)
    if not os.path.isfile(db_path):
        return {}
    try:
        import sqlite3

        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5.0)
        try:
            rows = conn.execute("SELECT id, tier FROM block_tiers").fetchall()
        finally:
            conn.close()
        return {r[0]: int(r[1]) for r in rows}
    except Exception as exc:
        _log.debug("tier_map_load_failed", error=str(exc))
        return {}


def apply_tier_boosts(
    results: list[dict[str, Any]],
    workspace: str,
    *,
    score_field: str = "score",
    annotate: bool = True,
) -> list[dict[str, Any]]:
    """Multiply every result's ``score_field`` by its tier boost.

    Args:
        results: Recall result dicts. Must each carry ``_id`` and
            the configured ``score_field`` (default ``"score"``).
            Results without a score are left untouched.
        workspace: Workspace path — used to locate the tier SQLite.
        score_field: Name of the score key to rewrite in place.
        annotate: When True (default) each result grows a
            ``_tier`` and ``_tier_boost`` field so downstream
            consumers can explain the ranking. When False only
            the score is modified.

    Returns:
        The same list (mutated in place). Sorted by score descending
        after the boosts are applied so callers can drop straight
        into their presentation layer without an extra ``sort``
        step.

    This function is *cheap* — one SQLite read, one O(N) pass — so
    it's safe to call on every recall. The caller is responsible
    for the opt-in config check.
    """
    if not results:
        return results
    tier_map = _load_tier_map(workspace)
    for block in results:
        bid = block.get("_id") or block.get("id")
        if not bid:
            continue
        tier = tier_map.get(str(bid), 1)  # WORKING by default
        boost = _TIER_BOOST.get(tier, 1.0)
        current = block.get(score_field)
        if isinstance(current, (int, float)):
            block[score_field] = float(current) * boost
        if annotate:
            block["_tier"] = tier
            block["_tier_boost"] = boost
    results.sort(key=lambda b: b.get(score_field, 0.0), reverse=True)
    return results


def is_tier_boost_enabled(config: dict[str, Any] | None) -> bool:
    """Read the opt-in flag from mind-mem.json config."""
    if not config:
        return False
    retrieval = config.get("retrieval", {})
    if not isinstance(retrieval, dict):
        return False
    return bool(retrieval.get("tier_boost", False))


def tier_boost_summary(tier_map: dict[str, int]) -> dict[str, Any]:
    """Count blocks per tier — useful for diagnostics endpoints.

    Callers typically get ``tier_map`` from :func:`_load_tier_map`
    directly. Exported because ``retrieval_diagnostics`` renders
    this in the per-query trace.
    """
    counts: dict[str, int] = {name: 0 for name in ("WORKING", "SHARED", "LONG_TERM", "VERIFIED")}
    name_by_int = {1: "WORKING", 2: "SHARED", 3: "LONG_TERM", 4: "VERIFIED"}
    for tier_int in tier_map.values():
        name = name_by_int.get(tier_int)
        if name is not None:
            counts[name] += 1
    counts["total"] = sum(counts[k] for k in ("WORKING", "SHARED", "LONG_TERM", "VERIFIED"))
    return counts


__all__: list[str] = [
    "apply_tier_boosts",
    "is_tier_boost_enabled",
    "tier_boost_summary",
]
