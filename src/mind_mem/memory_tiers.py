# Copyright 2026 STARGA, Inc.
"""Tiered memory with auto-promotion.

Blocks move through four tiers based on access frequency, age,
confirmation count, and confidence score:

    WORKING  ->  SHARED  ->  LONG_TERM  ->  VERIFIED

Tier assignments are persisted in a SQLite ``block_tiers`` table.
Block metadata (access count, age, confirmations, confidence) is read
from ``block_meta`` and ``block_tier_meta`` tables — the latter is
managed by this module and extended by callers (e.g. contradiction
detectors) that write confidence/confirmation updates.

Usage::

    mgr = TierManager("/path/to/index.db")
    mgr.get_tier("B-001")               # -> MemoryTier.WORKING (default)
    mgr.check_promotion("B-001")        # -> MemoryTier.SHARED | None
    mgr.promote("B-001", MemoryTier.SHARED)
    promotions = mgr.run_promotion_cycle()   # [(id, old, new), ...]
"""

from __future__ import annotations

import enum
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from .connection_manager import ConnectionManager
from .observability import get_logger, metrics


def _hours_since(iso_timestamp: str | None, now: datetime) -> float:
    """Hours elapsed between an ISO timestamp and *now*.

    Returns 0 for unparseable or absent inputs so the caller treats the
    block as "just seen" rather than accidentally demoting/evicting it.
    """
    if not iso_timestamp:
        return 0.0
    try:
        text = iso_timestamp
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
    except (ValueError, TypeError):
        return 0.0
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return max(0.0, (now - dt).total_seconds() / 3600.0)


_log = get_logger("memory_tiers")


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class MemoryTier(enum.IntEnum):
    """Ordered tier levels — higher value = more stable/trusted."""

    WORKING = 1
    SHARED = 2
    LONG_TERM = 3
    VERIFIED = 4


class DemotionReason(str, enum.Enum):
    """Reason recorded when a block is demoted."""

    LOW_CONFIDENCE = "low_confidence"
    CONTRADICTION = "contradiction"
    MANUAL = "manual"
    STALE = "stale"


@dataclass
class TierPolicy:
    """Criteria a block must satisfy to enter this tier.

    Fields:
        min_access_count:  Minimum cumulative accesses.
        min_age_hours:     Minimum hours since block creation.
        min_confirmations: Minimum explicit confirmations received.
        min_confidence:    Minimum confidence score (0.0–1.0).
        max_idle_hours:    If >0, a block whose last access is older than
                           this many hours is demoted back to the previous
                           tier (TTL/LRU decay — #502, v3.0.0+).
        ttl_hours:         If >0, a block older than this many hours and
                           at this tier is evicted outright (returns to
                           WORKING). Only applies to WORKING tier by
                           default so top tiers never auto-delete.
    """

    min_access_count: int = 0
    min_age_hours: float = 0.0
    min_confirmations: int = 0
    min_confidence: float = 0.0
    max_idle_hours: float = 0.0
    ttl_hours: float = 0.0


def default_policies() -> dict[MemoryTier, TierPolicy]:
    """Return the default promotion + decay policies for all tiers.

    WORKING is the entry tier; its ttl_hours caps how long a never-
    re-accessed block sticks around. Higher tiers have max_idle_hours
    (demote to previous tier after inactivity) but no ttl (never
    outright evicted — once trusted, only explicit demotion removes).
    """
    return {
        MemoryTier.WORKING: TierPolicy(
            # WORKING is the only tier with outright eviction (ttl). After
            # a week of no access we assume the block was a one-off
            # observation that never graduated and can be purged.
            ttl_hours=24.0 * 7,
        ),
        MemoryTier.SHARED: TierPolicy(
            min_access_count=3,
            min_age_hours=1.0,
            max_idle_hours=24.0 * 14,  # 2 weeks idle → demote to WORKING
        ),
        MemoryTier.LONG_TERM: TierPolicy(
            min_access_count=10,
            min_age_hours=24.0,
            min_confirmations=2,
            max_idle_hours=24.0 * 60,  # 2 months idle → demote to SHARED
        ),
        MemoryTier.VERIFIED: TierPolicy(
            min_confirmations=5,
            min_confidence=0.9,
            max_idle_hours=24.0 * 180,  # 6 months idle → demote to LONG_TERM
        ),
    }


# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

_TIERS_SCHEMA = """
CREATE TABLE IF NOT EXISTS block_tiers (
    id          TEXT PRIMARY KEY,
    tier        INTEGER NOT NULL DEFAULT 1,
    updated_at  TEXT NOT NULL,
    demotion_reason TEXT
);
"""

_TIER_META_SCHEMA = """
CREATE TABLE IF NOT EXISTS block_tier_meta (
    id              TEXT PRIMARY KEY,
    created_at      TEXT NOT NULL,
    confirmations   INTEGER NOT NULL DEFAULT 0,
    confidence      REAL NOT NULL DEFAULT 1.0,
    contradicted    INTEGER NOT NULL DEFAULT 0
);
"""

_BLOCK_META_READ_SQL = """
SELECT
    bm.access_count,
    btm.created_at,
    btm.confirmations,
    btm.confidence
FROM block_meta bm
JOIN block_tier_meta btm ON bm.id = btm.id
WHERE bm.id = ?
"""


# ---------------------------------------------------------------------------
# TierManager
# ---------------------------------------------------------------------------


class TierManager:
    """Manage memory tier assignments for blocks.

    All reads use per-thread connections (WAL); all writes are
    serialized through a single write connection, matching the
    ConnectionManager pattern used elsewhere in mind-mem.
    """

    def __init__(
        self,
        db_path: str,
        policies: dict[MemoryTier, TierPolicy] | None = None,
    ) -> None:
        self._db_path = db_path
        self._policies: dict[MemoryTier, TierPolicy] = policies or default_policies()
        self._conn_mgr = ConnectionManager(db_path)
        self._lock = threading.RLock()
        self._ensure_schema()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_tier(self, block_id: str) -> MemoryTier:
        """Return the current tier for *block_id*.

        Blocks that have never had a tier assignment default to WORKING.
        """
        try:
            conn = self._conn_mgr.get_read_connection()
            row = conn.execute("SELECT tier FROM block_tiers WHERE id = ?", (block_id,)).fetchone()
            if row is None:
                return MemoryTier.WORKING
            return MemoryTier(row[0])
        except (sqlite3.Error, ValueError):
            return MemoryTier.WORKING

    def check_promotion(self, block_id: str) -> Optional[MemoryTier]:
        """Return the next tier if *block_id* is eligible for promotion, else None.

        Does NOT perform the promotion — call :meth:`promote` to apply it.
        """
        current = self.get_tier(block_id)
        next_tier = self._next_tier(current)
        if next_tier is None:
            return None
        meta = self._read_meta(block_id)
        if meta is None:
            return None
        if self._satisfies_policy(meta, self._policies[next_tier]):
            return next_tier
        return None

    def promote(self, block_id: str, to_tier: MemoryTier) -> bool:
        """Promote *block_id* to *to_tier*.

        Promotion must advance by exactly one tier.  Returns True on
        success, False if the move is invalid (skipping tiers, or
        demoting).
        """
        current = self.get_tier(block_id)
        if to_tier.value != current.value + 1:
            _log.debug(
                "promote_rejected",
                block_id=block_id,
                current=current.name,
                requested=to_tier.name,
            )
            return False
        if not self._write_tier(block_id, to_tier, demotion_reason=None):
            return False
        _log.info("promoted", block_id=block_id, from_tier=current.name, to_tier=to_tier.name)
        return True

    def demote(self, block_id: str, to_tier: MemoryTier, reason: DemotionReason) -> bool:
        """Demote *block_id* to *to_tier* with a recorded *reason*.

        Demotion must move to a strictly lower tier than the current
        assignment.  Returns True on success, False if the move is
        invalid.
        """
        current = self.get_tier(block_id)
        if to_tier.value >= current.value:
            _log.debug(
                "demote_rejected",
                block_id=block_id,
                current=current.name,
                requested=to_tier.name,
            )
            return False
        if not self._write_tier(block_id, to_tier, demotion_reason=reason.value):
            return False
        _log.info(
            "demoted",
            block_id=block_id,
            from_tier=current.name,
            to_tier=to_tier.name,
            reason=reason.value,
        )
        return True

    def get_blocks_by_tier(self, tier: MemoryTier) -> list[str]:
        """Return all block IDs currently assigned to *tier*."""
        try:
            conn = self._conn_mgr.get_read_connection()
            rows = conn.execute(
                "SELECT id FROM block_tiers WHERE tier = ? ORDER BY id",
                (tier.value,),
            ).fetchall()
            return [r[0] for r in rows]
        except sqlite3.Error:
            return []

    def run_promotion_cycle(self) -> list[tuple[str, MemoryTier, MemoryTier]]:
        """Check every tracked block and apply eligible promotions.

        Returns a list of ``(block_id, old_tier, new_tier)`` tuples for
        every promotion that was applied during this cycle.
        """
        all_ids = self._all_tracked_ids()
        promotions: list[tuple[str, MemoryTier, MemoryTier]] = []
        for block_id in all_ids:
            new_tier = self.check_promotion(block_id)
            if new_tier is not None:
                old_tier = self.get_tier(block_id)
                if self.promote(block_id, new_tier):
                    promotions.append((block_id, old_tier, new_tier))
        _log.info("promotion_cycle_complete", promotions=len(promotions))
        return promotions

    def run_decay_cycle(self, *, now: datetime | None = None) -> tuple[list[tuple[str, MemoryTier, MemoryTier]], list[str]]:
        """Apply TTL + max-idle decay across every tracked block.

        Two outcomes per block:

        1. ``max_idle_hours`` exceeded → demoted one tier (STALE reason).
        2. ``ttl_hours`` exceeded (WORKING only by default) → evicted
           from tier tracking entirely (row deleted from block_tiers).

        Returns ``(demotions, evicted_block_ids)``.
        """
        current = now or datetime.now(timezone.utc)
        all_ids = self._all_tracked_ids()
        demotions: list[tuple[str, MemoryTier, MemoryTier]] = []
        evicted: list[str] = []
        for block_id in all_ids:
            tier = self.get_tier(block_id)
            policy = self._policies.get(tier)
            if policy is None:
                continue
            meta = self._read_meta(block_id)
            # When no block_meta row exists (common for blocks registered
            # directly via _register_block without touching block_meta),
            # fall back to the tier row's own updated_at timestamp so
            # decay still applies. Idle time counts from "first tracked".
            if meta is None:
                last_access = self._tier_updated_at(block_id)
            else:
                last_access = meta.get("last_access_at") or meta.get("created_at")
            idle_hours = _hours_since(last_access, current)
            # Eviction (WORKING tier default).
            if policy.ttl_hours > 0 and idle_hours >= policy.ttl_hours:
                if self._evict(block_id):
                    evicted.append(block_id)
                continue
            # Demotion to previous tier.
            if policy.max_idle_hours > 0 and idle_hours >= policy.max_idle_hours:
                prev = MemoryTier(max(1, int(tier) - 1))
                if prev != tier and self.demote(block_id, prev, DemotionReason.STALE):
                    demotions.append((block_id, tier, prev))
        _log.info(
            "decay_cycle_complete",
            demotions=len(demotions),
            evicted=len(evicted),
        )
        metrics.inc("tier_demotions", len(demotions))
        metrics.inc("tier_evictions", len(evicted))
        return demotions, evicted

    def _tier_updated_at(self, block_id: str) -> str | None:
        """Return the block_tiers.updated_at ISO string, or None."""
        with self._lock:
            try:
                conn = self._conn_mgr.get_read_connection()
                row = conn.execute(
                    "SELECT updated_at FROM block_tiers WHERE id = ?",
                    (block_id,),
                ).fetchone()
                if row is None:
                    return None
                # row may be sqlite3.Row or tuple — pick the single column
                try:
                    return str(row["updated_at"]) if row["updated_at"] is not None else None
                except (IndexError, KeyError, TypeError):
                    return str(row[0]) if row[0] is not None else None
            except sqlite3.Error:
                return None

    def _evict(self, block_id: str) -> bool:
        """Remove the block from tier tracking (used by run_decay_cycle)."""
        with self._lock:
            try:
                with self._conn_mgr.write_lock:
                    conn = self._conn_mgr.get_write_connection()
                    cur = conn.execute("DELETE FROM block_tiers WHERE id = ?", (block_id,))
                    conn.commit()
                    return cur.rowcount > 0
            except sqlite3.Error as exc:
                _log.warning("evict_failed", block_id=block_id, error=str(exc))
                return False

    # ------------------------------------------------------------------
    # Semi-private helper used by tests to pre-register blocks
    # ------------------------------------------------------------------

    def _register_block(self, block_id: str, tier: MemoryTier = MemoryTier.WORKING) -> None:
        """Ensure *block_id* has an explicit tier record (used by tests and ingestion)."""
        with self._lock:
            try:
                with self._conn_mgr.write_lock:
                    conn = self._conn_mgr.get_write_connection()
                    now = datetime.now(timezone.utc).isoformat()
                    conn.execute(
                        """INSERT INTO block_tiers (id, tier, updated_at)
                           VALUES (?, ?, ?)
                           ON CONFLICT(id) DO NOTHING""",
                        (block_id, tier.value, now),
                    )
                    conn.commit()
            except sqlite3.Error as exc:
                _log.error("register_block_failed", block_id=block_id, error=str(exc))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_schema(self) -> None:
        with self._lock:
            try:
                with self._conn_mgr.write_lock:
                    conn = self._conn_mgr.get_write_connection()
                    conn.execute(_TIERS_SCHEMA)
                    conn.execute(_TIER_META_SCHEMA)
                    conn.commit()
            except sqlite3.Error as exc:
                _log.error("schema_init_failed", error=str(exc))

    def _write_tier(self, block_id: str, tier: MemoryTier, *, demotion_reason: str | None) -> bool:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            try:
                with self._conn_mgr.write_lock:
                    conn = self._conn_mgr.get_write_connection()
                    conn.execute(
                        """INSERT INTO block_tiers (id, tier, updated_at, demotion_reason)
                           VALUES (?, ?, ?, ?)
                           ON CONFLICT(id) DO UPDATE SET
                               tier = ?,
                               updated_at = ?,
                               demotion_reason = ?""",
                        (
                            block_id,
                            tier.value,
                            now,
                            demotion_reason,
                            tier.value,
                            now,
                            demotion_reason,
                        ),
                    )
                    conn.commit()
                    return True
            except sqlite3.Error as exc:
                _log.error("write_tier_failed", block_id=block_id, error=str(exc))
                return False

    def _read_meta(self, block_id: str) -> Optional[dict]:
        """Return merged metadata for promotion eligibility checks.

        Reads from ``block_meta`` JOIN ``block_tier_meta``.  If either
        row is missing the block has no metadata and is not eligible.
        """
        try:
            conn = self._conn_mgr.get_read_connection()
            row = conn.execute(_BLOCK_META_READ_SQL, (block_id,)).fetchone()
            if row is None:
                return None
            access_count, created_at, confirmations, confidence = row
            if created_at is None:
                return None
            try:
                created_dt = datetime.fromisoformat(created_at)
            except (ValueError, TypeError):
                return None
            now = datetime.now(timezone.utc)
            if created_dt.tzinfo is None:
                created_dt = created_dt.replace(tzinfo=timezone.utc)
            age_hours = max((now - created_dt).total_seconds() / 3600.0, 0.0)
            return {
                "access_count": int(access_count or 0),
                "age_hours": age_hours,
                "confirmations": int(confirmations or 0),
                "confidence": float(confidence if confidence is not None else 1.0),
            }
        except (sqlite3.Error, TypeError, ValueError):
            return None

    def _satisfies_policy(self, meta: dict, policy: TierPolicy) -> bool:
        return bool(
            meta["access_count"] >= policy.min_access_count
            and meta["age_hours"] >= policy.min_age_hours
            and meta["confirmations"] >= policy.min_confirmations
            and meta["confidence"] >= policy.min_confidence
        )

    def _next_tier(self, current: MemoryTier) -> Optional[MemoryTier]:
        """Return the tier one step above *current*, or None at VERIFIED."""
        try:
            return MemoryTier(current.value + 1)
        except ValueError:
            return None

    def _all_tracked_ids(self) -> list[str]:
        """Return all block IDs that have an explicit tier record."""
        try:
            conn = self._conn_mgr.get_read_connection()
            rows = conn.execute("SELECT id FROM block_tiers ORDER BY id").fetchall()
            return [r[0] for r in rows]
        except sqlite3.Error:
            return []
