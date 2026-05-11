"""v4 recall-tier memory (Group A: cognition / model layer).

Adds an *orthogonal* tier axis to the v3.x memory store. v3 already has
a four-tier lifecycle system in ``memory_tiers.py`` (WORKING → SHARED
→ LONG_TERM → VERIFIED) that tracks where a block sits in the
sharing/confirmation lifecycle. This module sits on a different axis:
**recall recency + surprise**. A block is HOT/WARM/COLD based on how
recently it surfaced in a recall and how high-surprise the surfacing
was — not on how widely it has been shared.

The two axes are independent. A block can be SHARED+HOT (recently
relevant shared context) or LONG_TERM+COLD (always-known facts that
haven't been queried lately).

Decay semantics by recall tier:

    HOT
        LRU + TTL. The freshest 100 blocks (by config). Aging out:
        whichever happens first — beyond the TTL window or pushed out
        of the LRU cap by newer hot writes. Aged-out blocks fall to
        WARM.

    WARM
        TTL with surprise-weighted re-promotion. Default WARM block
        ages to COLD after ``warm_ttl_hours``. A high-surprise read
        (semantic distance from the rolling recall context above
        ``promote_threshold``) bumps the block back to HOT instead of
        toward COLD.

    COLD
        Indefinite, gated by contradiction-density. Cold blocks stay
        cold forever unless a contradiction lands on or near them
        (lineage-staleness BFS density above ``contradiction_floor``),
        at which point they re-enter WARM for re-evaluation.

The feature is **off by default**. Calling any public function in
this module without the ``v4.tier_memory`` flag enabled raises
:class:`FeatureDisabledError`. v3.x callers see no behaviour change.

Schema additions (lazily created on first write when the flag is on):

    CREATE TABLE IF NOT EXISTS block_recall_tier (
        block_id        TEXT PRIMARY KEY,
        tier            TEXT NOT NULL,         -- 'hot' | 'warm' | 'cold'
        last_seen_at    TEXT NOT NULL,         -- ISO 8601 UTC
        promoted_count  INTEGER NOT NULL DEFAULT 0,
        last_surprise   REAL                   -- last computed surprise
    )

This file ships the **type + read surface** only: the enum, the
config dataclass, the schema bootstrap, the per-block reader, and the
tier-filtered list. Promotion / demotion / surprise-driven re-promo
land in subsequent v4 iterations.

Copyright STARGA, Inc.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable

from .feature_flags import flag_config, require_enabled

__all__ = [
    "FLAG",
    "RecallTier",
    "TierConfig",
    "DEFAULT_TIER_CONFIG",
    "StaleVersionError",
    "ensure_recall_tier_schema",
    "get_recall_tier",
    "get_tier_version",
    "list_blocks_in_recall_tier",
]


class StaleVersionError(RuntimeError):
    """Raised when a tier write is rejected because ``expected_version``
    no longer matches the row's current ``block_version``.

    Closes the v4-audit-2026-05-10 unanimous blind spot: without a CAS
    contract, two agents that both promote the same block can silently
    clobber each other's writes. Callers catch this, re-read the
    current version with :func:`get_tier_version`, and retry.
    """


#: The feature-flag key in ``mind-mem.json: v4: {...}``.
FLAG: str = "tier_memory"


class RecallTier(str, Enum):
    """Three-value recall-tier axis.

    Values are the lowercase strings used in the SQLite ``tier``
    column so the enum round-trips through plain text storage.
    """

    HOT = "hot"
    WARM = "warm"
    COLD = "cold"


@dataclass(frozen=True)
class TierConfig:
    """Tunable knobs for the recall-tier decay schedules.

    Defaults sit in :data:`DEFAULT_TIER_CONFIG`. Override via
    ``mind-mem.json``:

    ::

        "v4": {
            "tier_memory": {
                "enabled": true,
                "hot_capacity":         100,
                "hot_ttl_hours":         24,
                "warm_ttl_hours":      720,
                "promote_threshold":    0.65,
                "contradiction_floor":  0.5
            }
        }

    All thresholds are bare floats — Q16.16 conversion happens at the
    scoring boundary, not at config load.
    """

    #: Maximum block count in HOT before LRU eviction kicks in.
    hot_capacity: int = 100

    #: Hours a HOT block can sit untouched before aging to WARM.
    hot_ttl_hours: float = 24.0

    #: Hours a WARM block can sit untouched before aging to COLD.
    warm_ttl_hours: float = 720.0

    #: Surprise score (semantic distance from rolling recall context)
    #: above which a WARM read promotes back to HOT instead of aging
    #: toward COLD.
    promote_threshold: float = 0.65

    #: Lineage-staleness density above which a COLD block is re-armed
    #: into WARM for re-evaluation. Threshold operates on the same
    #: 0..1 scale as :data:`block_lineage.KIND_DECAY`.
    contradiction_floor: float = 0.5


#: Default :class:`TierConfig`. Construction is cheap; callers can
#: instantiate ``TierConfig(...)`` to override individual knobs.
DEFAULT_TIER_CONFIG: TierConfig = TierConfig()


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA_SQL: str = """
CREATE TABLE IF NOT EXISTS block_recall_tier (
    block_id        TEXT PRIMARY KEY,
    tier            TEXT NOT NULL,
    last_seen_at    TEXT NOT NULL,
    promoted_count  INTEGER NOT NULL DEFAULT 0,
    last_surprise   REAL,
    block_version   INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_block_recall_tier_tier
    ON block_recall_tier (tier, last_seen_at);
"""

#: Idempotent migration that brings pre-CAS deployments forward without
#: data movement.  ``ALTER TABLE … ADD COLUMN`` defaults all existing
#: rows to version 0; new tier writes increment from there.
_MIGRATE_ADD_VERSION_SQL: str = "ALTER TABLE block_recall_tier ADD COLUMN block_version INTEGER NOT NULL DEFAULT 0"


def ensure_recall_tier_schema(workspace: str | Path) -> None:
    """Create the ``block_recall_tier`` table on first call.

    Idempotent — safe to call on every write path. The table is created
    in the workspace's main ``index.db`` next to the existing
    ``block_meta`` / ``block_tiers`` tables. v3.x callers don't see
    the new table because they never query it.
    """
    require_enabled(FLAG)
    db = Path(workspace) / "index.db"
    if not db.parent.is_dir():
        db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db, timeout=30) as conn:
        conn.executescript(_SCHEMA_SQL)
        # Pre-CAS deployments may already have the table without
        # block_version. Add it on the fly; idempotent.
        cols = {row[1] for row in conn.execute("PRAGMA table_info(block_recall_tier)")}
        if "block_version" not in cols:
            conn.execute(_MIGRATE_ADD_VERSION_SQL)
        conn.commit()


# ---------------------------------------------------------------------------
# Reader API
# ---------------------------------------------------------------------------


def get_recall_tier(workspace: str | Path, block_id: str) -> RecallTier:
    """Return the recall tier for a single block.

    Blocks with no row in ``block_recall_tier`` default to
    :attr:`RecallTier.WARM` — same default the schema would emit on a
    first write.
    """
    require_enabled(FLAG)
    db = Path(workspace) / "index.db"
    if not db.is_file():
        return RecallTier.WARM
    with sqlite3.connect(db, timeout=30) as conn:
        # Don't auto-create the table here — read-only path.
        row = (
            conn.execute(
                "SELECT tier FROM block_recall_tier WHERE block_id = ?",
                (block_id,),
            ).fetchone()
            if _table_exists(conn, "block_recall_tier")
            else None
        )
    if row is None:
        return RecallTier.WARM
    try:
        return RecallTier(row[0])
    except ValueError:
        # Unknown stored value — treat as WARM rather than crashing.
        return RecallTier.WARM


def get_tier_version(workspace: str | Path, block_id: str) -> int:
    """Return the current ``block_version`` for a block's tier row.

    Returns ``0`` for blocks with no row (the same default a fresh
    insert would carry). Callers compute their next CAS write as
    ``expected_version = current``; the write succeeds iff the value
    is still ``current`` at apply time and increments to ``current +
    1``. Two concurrent writers see the same ``current``, only one's
    CAS succeeds; the loser raises :class:`StaleVersionError` and can
    re-read.

    No flag check on the read path itself — surfacing a version is
    pure metadata. The *write* function that consumes this version
    (lands when the promotion API is wired) is the gated surface.
    """
    db = Path(workspace) / "index.db"
    if not db.is_file():
        return 0
    with sqlite3.connect(db, timeout=30) as conn:
        if not _table_exists(conn, "block_recall_tier"):
            return 0
        cols = {row[1] for row in conn.execute("PRAGMA table_info(block_recall_tier)")}
        if "block_version" not in cols:
            return 0
        row = conn.execute(
            "SELECT block_version FROM block_recall_tier WHERE block_id = ?",
            (block_id,),
        ).fetchone()
    if row is None:
        return 0
    try:
        return int(row[0])
    except (TypeError, ValueError):
        return 0


def list_blocks_in_recall_tier(
    workspace: str | Path,
    tier: RecallTier | str,
    *,
    limit: int = 100,
) -> list[str]:
    """Return up to ``limit`` block IDs sitting in the given tier.

    Ordered by ``last_seen_at`` ascending (oldest first) — useful for
    decay sweeps that want to demote stalest first. Empty list when no
    rows match (or the schema doesn't exist yet).
    """
    require_enabled(FLAG)
    if isinstance(tier, str):
        tier = RecallTier(tier)
    db = Path(workspace) / "index.db"
    if not db.is_file():
        return []
    with sqlite3.connect(db, timeout=30) as conn:
        if not _table_exists(conn, "block_recall_tier"):
            return []
        rows: Iterable[tuple[str]] = conn.execute(
            "SELECT block_id FROM block_recall_tier WHERE tier = ? ORDER BY last_seen_at ASC LIMIT ?",
            (tier.value, int(limit)),
        ).fetchall()
    return [r[0] for r in rows]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?",
        (name,),
    ).fetchone()
    return row is not None


def _load_config() -> TierConfig:
    """Read the active config; merge with :data:`DEFAULT_TIER_CONFIG`.

    Unknown keys are ignored. Type-mismatched keys fall back to the
    default value for that field — fail-soft so an operator typo can't
    crash the recall path.
    """
    raw = flag_config(FLAG)
    if not isinstance(raw, dict):
        return DEFAULT_TIER_CONFIG
    fields = {
        "hot_capacity": (int, DEFAULT_TIER_CONFIG.hot_capacity),
        "hot_ttl_hours": (float, DEFAULT_TIER_CONFIG.hot_ttl_hours),
        "warm_ttl_hours": (float, DEFAULT_TIER_CONFIG.warm_ttl_hours),
        "promote_threshold": (float, DEFAULT_TIER_CONFIG.promote_threshold),
        "contradiction_floor": (float, DEFAULT_TIER_CONFIG.contradiction_floor),
    }
    out: dict[str, object] = {}
    for key, (caster, default) in fields.items():
        v = raw.get(key, default)
        try:
            out[key] = caster(v)
        except (TypeError, ValueError):
            out[key] = default
    return TierConfig(**out)  # type: ignore[arg-type]
