# Copyright 2026 STARGA, Inc.
"""Cognitive forgetting + token budget (v2.4.0).

Two related concerns:

1. **Active forgetting** — a four-stage state machine (mark → merge →
   archive → forget) that curates a block store so stale, low-value,
   and rarely-accessed blocks don't dominate retrieval forever.
   Every forget decision is reversible within a grace window so a
   wrong decision can be recovered.

2. **Token budget packing** — given a token budget, select a subset of
   recall results that fits while preserving the highest-value blocks
   first. The selection respects a configurable reserve for graph
   context and provenance metadata.

Both features are opt-in config knobs layered on top of the existing
retrieval pipeline. Zero new dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Iterable, Mapping, Optional

# ---------------------------------------------------------------------------
# Forgetting state machine
# ---------------------------------------------------------------------------


class BlockLifecycle(str, Enum):
    """States a block can inhabit in the forgetting cycle."""

    ACTIVE = "active"
    MARKED = "marked"  # flagged for review
    MERGED = "merged"  # combined into a summary block
    ARCHIVED = "archived"  # cold storage — still queryable, not in hot index
    FORGOTTEN = "forgotten"  # permanently removed; grace window expired


@dataclass(frozen=True)
class BlockCognition:
    """Per-block telemetry used by the forgetting decision functions."""

    block_id: str
    importance: float  # [0, 1]; higher = keep
    last_accessed: Optional[str]  # ISO8601 or None
    access_count: int
    created_at: Optional[str]
    size_bytes: int = 0
    lifecycle: BlockLifecycle = BlockLifecycle.ACTIVE

    def __post_init__(self) -> None:
        if not 0.0 <= self.importance <= 1.0:
            raise ValueError(f"importance must be in [0, 1], got {self.importance!r}")
        if self.access_count < 0:
            raise ValueError("access_count must be >= 0")
        if self.size_bytes < 0:
            raise ValueError("size_bytes must be >= 0")


# ---------------------------------------------------------------------------
# Decision functions — pure, so they're easy to test
# ---------------------------------------------------------------------------


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def should_mark(
    block: BlockCognition,
    *,
    now: datetime,
    importance_threshold: float,
    stale_days: int,
) -> bool:
    """True if *block* deserves a mark (flagged for later review).

    A block is marked when it's both LOW IMPORTANCE and hasn't been
    accessed for at least ``stale_days`` days. Either condition alone
    is insufficient — a low-importance block that was just read is
    still being used, and a high-importance block that's rarely read
    is often still load-bearing (e.g., an ADR).
    """
    if block.lifecycle is not BlockLifecycle.ACTIVE:
        return False
    if block.importance >= importance_threshold:
        return False
    last = _parse_iso(block.last_accessed) or _parse_iso(block.created_at)
    if last is None:
        return True
    return (now - last) >= timedelta(days=stale_days)


def should_archive(
    block: BlockCognition,
    *,
    now: datetime,
    archive_after_days: int,
) -> bool:
    """True if a MERGED block is old enough to move to cold storage."""
    if block.lifecycle is not BlockLifecycle.MERGED:
        return False
    last = _parse_iso(block.last_accessed) or _parse_iso(block.created_at)
    if last is None:
        return True
    return (now - last) >= timedelta(days=archive_after_days)


def should_forget(
    block: BlockCognition,
    *,
    now: datetime,
    grace_days: int,
) -> bool:
    """True if an ARCHIVED block is past its grace window.

    Blocks only ever reach :attr:`BlockLifecycle.FORGOTTEN` through this
    check, which is itself governance-gated at the caller. Direct
    active-→ forgotten transitions are intentionally impossible.
    """
    if block.lifecycle is not BlockLifecycle.ARCHIVED:
        return False
    last = _parse_iso(block.last_accessed) or _parse_iso(block.created_at)
    if last is None:
        return True
    return (now - last) >= timedelta(days=grace_days)


# ---------------------------------------------------------------------------
# ConsolidationPlan — pure output, easy to preview before applying
# ---------------------------------------------------------------------------


@dataclass
class ConsolidationPlan:
    """Proposed transitions generated by a dry-run of the consolidator."""

    mark: list[str] = field(default_factory=list)
    archive: list[str] = field(default_factory=list)
    forget: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.mark) + len(self.archive) + len(self.forget)

    def as_dict(self) -> dict[str, Any]:
        return {
            "mark": list(self.mark),
            "archive": list(self.archive),
            "forget": list(self.forget),
            "total": self.total,
        }


@dataclass(frozen=True)
class ConsolidationConfig:
    """Tunable thresholds for :func:`plan_consolidation`.

    Defaults match the roadmap wording (importance threshold 0.25,
    stale window 14 days, archive after 60 days, 30-day grace before
    permanent forget).
    """

    importance_threshold: float = 0.25
    stale_days: int = 14
    archive_after_days: int = 60
    grace_days: int = 30

    def __post_init__(self) -> None:
        if not 0.0 <= self.importance_threshold <= 1.0:
            raise ValueError("importance_threshold must be in [0, 1]")
        for label, val in (
            ("stale_days", self.stale_days),
            ("archive_after_days", self.archive_after_days),
            ("grace_days", self.grace_days),
        ):
            if val < 0:
                raise ValueError(f"{label} must be >= 0")


def plan_consolidation(
    blocks: Iterable[BlockCognition],
    *,
    config: Optional[ConsolidationConfig] = None,
    now: Optional[datetime] = None,
) -> ConsolidationPlan:
    """Generate a :class:`ConsolidationPlan` from block telemetry."""
    cfg = config or ConsolidationConfig()
    current = now or datetime.now(timezone.utc)
    plan = ConsolidationPlan()
    for b in blocks:
        if should_mark(
            b,
            now=current,
            importance_threshold=cfg.importance_threshold,
            stale_days=cfg.stale_days,
        ):
            plan.mark.append(b.block_id)
            continue
        if should_archive(b, now=current, archive_after_days=cfg.archive_after_days):
            plan.archive.append(b.block_id)
            continue
        if should_forget(b, now=current, grace_days=cfg.grace_days):
            plan.forget.append(b.block_id)
    return plan


# ---------------------------------------------------------------------------
# Token budget packing
# ---------------------------------------------------------------------------


def estimate_tokens(text: str) -> int:
    """Rough token estimator: ~4 chars per token (OpenAI BPE heuristic).

    Dependency-free approximation — good enough for packing decisions.
    A model-aware tokenizer can replace this later without API change.
    """
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


@dataclass(frozen=True)
class PackedBudget:
    """Result of packing a list of results under a token ceiling."""

    included: list[dict]
    dropped: list[dict]
    budget: int
    tokens_used: int
    reserved_graph: int
    reserved_provenance: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "included_count": len(self.included),
            "dropped_count": len(self.dropped),
            "budget": self.budget,
            "tokens_used": self.tokens_used,
            "reserved_graph": self.reserved_graph,
            "reserved_provenance": self.reserved_provenance,
        }


# Defaults drawn from the roadmap: 15% graph context, 10% provenance.
_GRAPH_RESERVE_FRAC: float = 0.15
_PROVENANCE_RESERVE_FRAC: float = 0.10


def pack_to_budget(
    results: Iterable[Mapping[str, Any]],
    *,
    max_tokens: int,
    text_field: str = "excerpt",
    graph_reserve_frac: float = _GRAPH_RESERVE_FRAC,
    provenance_reserve_frac: float = _PROVENANCE_RESERVE_FRAC,
) -> PackedBudget:
    """Pack results into ``max_tokens`` respecting graph + provenance reserves.

    Results that don't fit are *dropped* rather than truncated —
    truncation would silently mangle fact cards / decision statements.
    The input order is treated as priority order (highest first).
    """
    if max_tokens <= 0:
        raise ValueError("max_tokens must be > 0")
    if not 0.0 <= graph_reserve_frac < 1.0:
        raise ValueError("graph_reserve_frac must be in [0, 1)")
    if not 0.0 <= provenance_reserve_frac < 1.0:
        raise ValueError("provenance_reserve_frac must be in [0, 1)")
    if graph_reserve_frac + provenance_reserve_frac >= 1.0:
        raise ValueError("graph + provenance reserves must leave room for block content")

    reserved_graph = int(max_tokens * graph_reserve_frac)
    reserved_prov = int(max_tokens * provenance_reserve_frac)
    block_budget = max_tokens - reserved_graph - reserved_prov

    included: list[dict] = []
    dropped: list[dict] = []
    used = 0
    for res in results:
        text = str(res.get(text_field, ""))
        cost = estimate_tokens(text)
        if used + cost <= block_budget:
            included.append(dict(res) | {"_token_cost": cost})
            used += cost
        else:
            dropped.append(dict(res) | {"_token_cost": cost})
    return PackedBudget(
        included=included,
        dropped=dropped,
        budget=max_tokens,
        tokens_used=used,
        reserved_graph=reserved_graph,
        reserved_provenance=reserved_prov,
    )


__all__ = [
    "BlockLifecycle",
    "BlockCognition",
    "ConsolidationConfig",
    "ConsolidationPlan",
    "plan_consolidation",
    "should_mark",
    "should_archive",
    "should_forget",
    "estimate_tokens",
    "pack_to_budget",
    "PackedBudget",
]
