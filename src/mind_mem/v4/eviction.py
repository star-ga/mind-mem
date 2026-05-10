"""v4 eviction policies (Group A — long-term decay).

Round 3 multi-LLM audit (2/4 model agreement 2026-05-10) flagged the
absence of a cold-tier eviction surface: blocks accumulate forever
once they hit COLD; without a policy to prune low-impact memories,
the workspace bloats and recall performance degrades.

This module ships **pluggable eviction policies** as a pure function
that returns a *plan*. The caller applies the plan through the v3
governance flow (same shape as :mod:`consolidation_worker`). No
side effects from the planner; safe to call any time.

Built-in policies:

    LRU                     Evict the K oldest blocks by
                            ``last_seen_at`` from the ``COLD`` tier.
    LOW_SURPRISE            Evict COLD blocks whose
                            ``last_surprise`` < threshold.
    AGE                     Evict COLD blocks whose ``last_seen_at``
                            is older than ``cutoff_iso``.
    COMPOSITE               Run two policies and union the IDs.
    custom                  Caller registers via
                            :func:`register_policy`.

Eviction does not mean *delete* by default — it means propose
``RecallTier.COLD → archive``. The downstream apply path (caller's
choice) decides whether that's a soft tombstone, a sealed-evidence
mirror, or hard delete.

Feature-flag gated under ``v4.eviction``.

Copyright STARGA, Inc.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from .feature_flags import flag_config, require_enabled

__all__ = [
    "FLAG",
    "EvictionPolicy",
    "EvictionPlan",
    "register_policy",
    "available_policies",
    "plan_eviction",
    "DEFAULT_EVICTION_LIMIT",
    "DEFAULT_LOW_SURPRISE_THRESHOLD",
]


FLAG: str = "eviction"

DEFAULT_EVICTION_LIMIT: int = 100
DEFAULT_LOW_SURPRISE_THRESHOLD: float = 0.1


class EvictionPolicy(str, Enum):
    LRU = "lru"
    LOW_SURPRISE = "low_surprise"
    AGE = "age"
    COMPOSITE = "composite"


@dataclass(frozen=True)
class EvictionPlan:
    """Read-only output of :func:`plan_eviction`.

    The caller applies the plan through propose/approve. Each entry
    is ``(block_id, reason)``; the reason carries a short tag so
    downstream observers can audit the eviction by policy.
    """

    policy: EvictionPolicy | str
    candidates: list[tuple[str, str]] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.candidates)


# ---------------------------------------------------------------------------
# Pluggable policy registry
# ---------------------------------------------------------------------------


PolicyFn = Callable[..., list[tuple[str, str]]]

_registry: dict[EvictionPolicy | str, PolicyFn] = {}
_lock_obj = __import__("threading").Lock()


def register_policy(name: EvictionPolicy | str, fn: PolicyFn) -> None:
    """Register a custom eviction policy under ``name``.

    Custom policies receive ``(workspace, **kwargs)`` and return a
    list of ``(block_id, reason)`` tuples. Built-in names can be
    overridden, useful for testing.
    """
    require_enabled(FLAG)
    with _lock_obj:
        _registry[name] = fn


def available_policies() -> list[EvictionPolicy | str]:
    """Return every registered policy name."""
    require_enabled(FLAG)
    with _lock_obj:
        return list(_registry.keys())


# ---------------------------------------------------------------------------
# Built-in policies
# ---------------------------------------------------------------------------


def _lru_policy(workspace: str | Path, *, limit: int = DEFAULT_EVICTION_LIMIT, **_: object) -> list[tuple[str, str]]:
    db = Path(workspace) / "index.db"
    if not db.is_file():
        return []
    with sqlite3.connect(db) as conn:
        if not _table_exists(conn, "block_recall_tier"):
            return []
        rows = conn.execute(
            "SELECT block_id, last_seen_at FROM block_recall_tier WHERE tier = 'cold' ORDER BY last_seen_at ASC LIMIT ?",
            (max(0, int(limit)),),
        ).fetchall()
    return [(bid, f"lru:last_seen={ts}") for bid, ts in rows]


def _low_surprise_policy(
    workspace: str | Path,
    *,
    threshold: float = DEFAULT_LOW_SURPRISE_THRESHOLD,
    limit: int = DEFAULT_EVICTION_LIMIT,
    **_: object,
) -> list[tuple[str, str]]:
    db = Path(workspace) / "index.db"
    if not db.is_file():
        return []
    with sqlite3.connect(db) as conn:
        if not _table_exists(conn, "block_recall_tier"):
            return []
        rows = conn.execute(
            "SELECT block_id, last_surprise FROM block_recall_tier "
            "WHERE tier = 'cold' AND last_surprise IS NOT NULL "
            "AND last_surprise < ? ORDER BY last_surprise ASC LIMIT ?",
            (float(threshold), max(0, int(limit))),
        ).fetchall()
    return [(bid, f"low_surprise:s={float(s):.3f}") for bid, s in rows]


def _age_policy(
    workspace: str | Path,
    *,
    cutoff_iso: str = "1970-01-01T00:00:00Z",
    limit: int = DEFAULT_EVICTION_LIMIT,
    **_: object,
) -> list[tuple[str, str]]:
    db = Path(workspace) / "index.db"
    if not db.is_file():
        return []
    with sqlite3.connect(db) as conn:
        if not _table_exists(conn, "block_recall_tier"):
            return []
        rows = conn.execute(
            "SELECT block_id, last_seen_at FROM block_recall_tier "
            "WHERE tier = 'cold' AND last_seen_at < ? "
            "ORDER BY last_seen_at ASC LIMIT ?",
            (cutoff_iso, max(0, int(limit))),
        ).fetchall()
    return [(bid, f"age:last_seen={ts}") for bid, ts in rows]


def _composite_policy(
    workspace: str | Path,
    *,
    policies: list[str] | None = None,
    **kwargs: object,
) -> list[tuple[str, str]]:
    """Union of multiple policies' candidates, deduped on block_id.

    Order preserved — first policy's pick wins on dedupe.
    """
    if not policies:
        return []
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for p_name in policies:
        with _lock_obj:
            fn = _registry.get(p_name) or _registry.get(EvictionPolicy(p_name))
        if fn is None:
            continue
        for bid, reason in fn(workspace, **kwargs):
            if bid in seen:
                continue
            seen.add(bid)
            out.append((bid, f"composite:{p_name}|{reason}"))
    return out


# Bootstrap registry at import time (does not require the flag — registry
# population is independent of use).
_registry[EvictionPolicy.LRU] = _lru_policy
_registry[EvictionPolicy.LOW_SURPRISE] = _low_surprise_policy
_registry[EvictionPolicy.AGE] = _age_policy
_registry[EvictionPolicy.COMPOSITE] = _composite_policy


# ---------------------------------------------------------------------------
# Public planner
# ---------------------------------------------------------------------------


def plan_eviction(
    workspace: str | Path,
    policy: EvictionPolicy | str = EvictionPolicy.LRU,
    **kwargs: object,
) -> EvictionPlan:
    """Compute an eviction plan against the workspace's COLD tier.

    Pure function; reads the tier table, returns a plan. No side
    effects. Returns an empty plan when the policy is unknown / DB is
    missing / table doesn't exist.
    """
    require_enabled(FLAG)
    if isinstance(policy, str):
        try:
            policy_key: EvictionPolicy | str = EvictionPolicy(policy)
        except ValueError:
            policy_key = policy
    else:
        policy_key = policy
    with _lock_obj:
        fn = _registry.get(policy_key)
    if fn is None:
        return EvictionPlan(policy=policy_key, candidates=[])
    candidates = fn(workspace, **kwargs)
    return EvictionPlan(policy=policy_key, candidates=candidates)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?",
        (name,),
    ).fetchone()
    return row is not None


def _load_threshold() -> float:
    raw = flag_config(FLAG)
    if not isinstance(raw, dict):
        return DEFAULT_LOW_SURPRISE_THRESHOLD
    v = raw.get("threshold", DEFAULT_LOW_SURPRISE_THRESHOLD)
    try:
        out = float(v)
    except (TypeError, ValueError):
        return DEFAULT_LOW_SURPRISE_THRESHOLD
    return max(0.0, min(1.0, out))
