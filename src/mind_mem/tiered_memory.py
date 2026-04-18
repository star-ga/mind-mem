# Copyright 2026 STARGA, Inc.
"""4-tier memory consolidation with Ebbinghaus decay (v2.6.0).

Blocks inhabit one of four tiers:

- **Tier 0 — Working** — raw daily log entries, TTL 30 days
- **Tier 1 — Episodic** — compressed session summaries
- **Tier 2 — Semantic** — verified facts / entities
- **Tier 3 — Procedural** — learned strategies, highest durability

Each block carries a ``strength`` field in [0, 1] that decays
exponentially with a 30-day half-life and resets on access. Blocks
that are observed N or more times across sessions are flagged for
auto-promotion to the next tier (governance-gated for Tier 2 → 3).

Tier-aware retrieval multiplies the BM25F score by:
    Tier 3 → 2.0x, Tier 2 → 1.5x, Tier 1 → 1.0x, Tier 0 → 0.7x.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import IntEnum
from typing import Iterable, Optional


class Tier(IntEnum):
    WORKING = 0
    EPISODIC = 1
    SEMANTIC = 2
    PROCEDURAL = 3

    @property
    def retrieval_boost(self) -> float:
        return {
            Tier.WORKING: 0.7,
            Tier.EPISODIC: 1.0,
            Tier.SEMANTIC: 1.5,
            Tier.PROCEDURAL: 2.0,
        }[self]


_HALF_LIFE_DAYS: float = 30.0


@dataclass
class TieredBlock:
    """Per-tier telemetry for consolidation decisions."""

    block_id: str
    tier: Tier
    strength: float  # [0, 1]
    last_accessed: Optional[str] = None
    access_count: int = 0
    session_count: int = 0

    def __post_init__(self) -> None:
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("strength must be in [0, 1]")
        if self.access_count < 0 or self.session_count < 0:
            raise ValueError("access/session counts must be >= 0")


def _parse(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    text = ts.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def decay(
    strength: float,
    last_accessed: Optional[str],
    *,
    now: Optional[datetime] = None,
    half_life_days: float = _HALF_LIFE_DAYS,
) -> float:
    """Apply Ebbinghaus-style exponential decay to a strength value."""
    current = now or datetime.now(timezone.utc)
    last = _parse(last_accessed)
    if last is None or strength <= 0.0:
        return max(0.0, min(1.0, strength))
    age_days = max(0.0, (current - last).total_seconds() / 86400.0)
    if half_life_days <= 0:
        return strength
    return max(0.0, min(1.0, strength * math.pow(0.5, age_days / half_life_days)))


def reset_on_access(strength: float) -> float:
    """Bump strength back toward 1.0 when the block is accessed."""
    return max(strength, 1.0)


@dataclass(frozen=True)
class PromotionCandidate:
    """A block the auto-promotion engine wants to move up a tier."""

    block_id: str
    from_tier: Tier
    to_tier: Tier
    reason: str


def promote_candidates(
    blocks: Iterable[TieredBlock],
    *,
    min_sessions_per_tier: int = 3,
    now: Optional[datetime] = None,
) -> list[PromotionCandidate]:
    """Flag blocks that should be promoted under the roadmap rule.

    A block observed in 3+ sessions qualifies for promotion to the
    next tier. Tier 2 → 3 emits the candidate but the caller is
    expected to route it through the governance proposal path.
    """
    out: list[PromotionCandidate] = []
    for b in blocks:
        if b.tier == Tier.PROCEDURAL:
            continue
        if b.session_count < min_sessions_per_tier:
            continue
        target = Tier(int(b.tier) + 1)
        reason = f"observed in {b.session_count} sessions (threshold {min_sessions_per_tier})"
        out.append(
            PromotionCandidate(
                block_id=b.block_id,
                from_tier=b.tier,
                to_tier=target,
                reason=reason,
            )
        )
    return out


def tier_boost(tier: Tier, score: float) -> float:
    """Multiply a retrieval score by the tier's configured boost."""
    return score * tier.retrieval_boost


__all__ = [
    "Tier",
    "TieredBlock",
    "PromotionCandidate",
    "decay",
    "reset_on_access",
    "promote_candidates",
    "tier_boost",
]
