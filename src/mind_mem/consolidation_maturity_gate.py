# Copyright 2026 STARGA, Inc.
"""Maturity gate for the consolidation path (Group H — safe half).

The cognitive-forgetting cycle in :mod:`mind_mem.cognitive_forget` decides
which blocks get marked, archived, and eventually forgotten purely from
*telemetry* (importance, recency, access count).  It has no notion of how
**settled** a block is.  A brand-new, still-churning block and a
long-corroborated one are treated identically.

This module adds an admission gate in front of that flow:

* a block whose :func:`mind_mem.block_maturity.maturity_score` is below
  ``min_maturity`` is **young** and is held back — consolidation never
  touches it;
* a block that sits on either end of a **live contradiction** is held
  back unconditionally, so consolidation can never merge, archive, or
  forget across an unresolved conflict.

**Default-OFF.**  :class:`MaturityGateConfig` ships with ``enabled=False``
and :func:`mind_mem.cognitive_forget.plan_consolidation` takes the gate as
an optional keyword that defaults to ``None``.  With the flag off, not a
single byte of the existing consolidation output changes — the gate object
is never even constructed on the default path.

The gate only ever *removes* candidate transitions; it can never introduce
one.  That makes it safe to enable incrementally.

Scope
-----
This is the **safe half** of Group H.  The gate guards the existing
mark/archive/forget flow only.

# deferred: the destructive granularity/merge operation (rewriting two
# blocks into one) is intentionally NOT implemented here - upgrade path:
# build it on top of `MaturityGate.evaluate`, which already refuses to
# admit either end of a live contradiction, so a future merge can only
# ever see conflict-free, matured candidates.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Protocol, runtime_checkable

from .block_maturity import maturity_score
from .observability import get_logger

__all__ = [
    "HOLD_CONTRADICTED",
    "HOLD_YOUNG",
    "BlockLike",
    "GateDecision",
    "MaturityGate",
    "MaturityGateConfig",
    "collect_contradicted_block_ids",
]

_log = get_logger("consolidation_maturity_gate")

#: Hold reason — block maturity is below the configured threshold.
HOLD_YOUNG = "young"

#: Hold reason — block is an endpoint of an unresolved contradiction.
HOLD_CONTRADICTED = "live_contradiction"


@runtime_checkable
class BlockLike(Protocol):
    """Anything the gate can identify — duck-typed on ``block_id``.

    Declared structurally so this module never imports
    :mod:`mind_mem.cognitive_forget` (which imports the gate protocol in
    the other direction).  ``block_id`` is a read-only member so frozen
    dataclasses (``BlockCognition``) satisfy the protocol.
    """

    @property
    def block_id(self) -> str: ...  # pragma: no cover - protocol


@dataclass(frozen=True)
class MaturityGateConfig:
    """Tunables for :class:`MaturityGate`.  Disabled by default."""

    enabled: bool = False
    min_maturity: float = 0.5
    protect_contradicted: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be a bool")
        if not isinstance(self.protect_contradicted, bool):
            raise TypeError("protect_contradicted must be a bool")
        try:
            threshold = float(self.min_maturity)
        except (TypeError, ValueError) as exc:
            raise ValueError("min_maturity must be a float in [0, 1]") from exc
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"min_maturity must be in [0, 1], got {self.min_maturity!r}")


@dataclass(frozen=True)
class GateDecision:
    """Immutable record of what a gate pass admitted and what it held."""

    admitted: tuple[str, ...]
    held: tuple[str, ...]
    reasons: Mapping[str, str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "admitted": list(self.admitted),
            "held": list(self.held),
            "reasons": dict(self.reasons),
            "admitted_count": len(self.admitted),
            "held_count": len(self.held),
        }


class MaturityGate:
    """Admission gate: ``admits(block)`` is False for young/contradicted blocks.

    Args:
        config: Gate tunables.  When ``config.enabled`` is False the gate
            admits everything (identity behaviour).
        block_meta: Mapping of ``block_id`` → block dict carrying the
            frontmatter fields :func:`maturity_score` reads (``Maturity``,
            ``Status``, ``Lifecycle``).  Blocks absent from the mapping are
            scored from an empty dict, which is deliberately conservative:
            an unknown block scores low and is therefore held.
        edge_counts: Optional ``block_id`` → incoming corroborating-edge
            count, forwarded to :func:`maturity_score`.
        contradicted_ids: Block ids sitting on a live contradiction.
    """

    __slots__ = ("_config", "_block_meta", "_edge_counts", "_contradicted")

    def __init__(
        self,
        config: MaturityGateConfig,
        *,
        block_meta: Mapping[str, Mapping[str, Any]] | None = None,
        edge_counts: Mapping[str, int] | None = None,
        contradicted_ids: Iterable[str] | None = None,
    ) -> None:
        if not isinstance(config, MaturityGateConfig):
            raise TypeError("config must be a MaturityGateConfig")
        self._config = config
        self._block_meta: Mapping[str, Mapping[str, Any]] = MappingProxyType(dict(block_meta or {}))
        self._edge_counts: Mapping[str, int] = MappingProxyType(dict(edge_counts or {}))
        self._contradicted: frozenset[str] = frozenset(str(b) for b in (contradicted_ids or ()))

    @property
    def config(self) -> MaturityGateConfig:
        return self._config

    @staticmethod
    def _block_id(block: BlockLike | str) -> str:
        return block if isinstance(block, str) else str(getattr(block, "block_id", ""))

    def score(self, block: BlockLike | str) -> float:
        """Maturity score in [0, 1] for *block* (0.0 for an unknown id)."""
        block_id = self._block_id(block)
        if not block_id:
            return 0.0
        meta = self._block_meta.get(block_id) or {}
        edges = self._edge_counts.get(block_id)
        return maturity_score(dict(meta), incoming_edge_count=edges)

    def hold_reason(self, block: BlockLike | str) -> str | None:
        """Return why *block* is held back, or ``None`` when it is admitted."""
        if not self._config.enabled:
            return None
        block_id = self._block_id(block)
        if self._config.protect_contradicted and block_id in self._contradicted:
            return HOLD_CONTRADICTED
        if self.score(block_id) < float(self._config.min_maturity):
            return HOLD_YOUNG
        return None

    def admits(self, block: BlockLike | str) -> bool:
        """True when consolidation is allowed to touch *block*."""
        return self.hold_reason(block) is None

    def evaluate(self, blocks: Iterable[BlockLike | str]) -> GateDecision:
        """Partition *blocks* into admitted/held, preserving input order."""
        admitted: list[str] = []
        held: list[str] = []
        reasons: dict[str, str] = {}
        for block in blocks:
            block_id = self._block_id(block)
            reason = self.hold_reason(block)
            if reason is None:
                admitted.append(block_id)
            else:
                held.append(block_id)
                reasons[block_id] = reason
        return GateDecision(
            admitted=tuple(admitted),
            held=tuple(held),
            reasons=MappingProxyType(reasons),
        )


# ---------------------------------------------------------------------------
# Workspace adapter — best effort, never raises
# ---------------------------------------------------------------------------


def collect_contradicted_block_ids(workspace: str) -> frozenset[str]:
    """Collect block ids on a *live* contradiction in *workspace*.

    Two independent sources are unioned:

    1. ``contradicts`` lineage edges in the recall graph
       (``.mind-mem-index/recall.db`` → ``co_retrieval``).
    2. Detected contradictions surfaced by
       :func:`mind_mem.conflict_resolver.resolve_contradictions`
       (``intelligence/CONTRADICTIONS.md``).  An entry that carries a
       truthy ``resolved`` marker is skipped — it is no longer live.

    Both sources are read best-effort: a missing DB, an absent corpus
    file, or a read error yields fewer ids, never an exception.  The gate
    is a *protective* filter, so degrading to "fewer protected blocks" is
    the honest failure mode and is surfaced by the caller's report rather
    than crashing the consolidation dry-run.
    """
    if not isinstance(workspace, str) or not workspace.strip():
        raise ValueError("workspace must be a non-empty string")
    ws = os.path.abspath(workspace)
    found: set[str] = set()

    db_path = os.path.join(ws, ".mind-mem-index", "recall.db")
    if os.path.isfile(db_path):
        import sqlite3

        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10.0)
            try:
                rows = conn.execute("SELECT mem1_id, mem2_id FROM co_retrieval WHERE kind = 'contradicts'").fetchall()
            finally:
                conn.close()
            for left, right in rows:
                found.add(str(left))
                found.add(str(right))
        except sqlite3.Error as exc:
            # No ``kind`` column / no table / locked DB → degrade, never crash.
            _log.warning("contradiction_edges_unavailable", error=str(exc))

    try:
        from .conflict_resolver import resolve_contradictions

        for entry in resolve_contradictions(ws):
            if entry.get("resolved"):
                continue
            for key in ("block_a", "block_b"):
                value = entry.get(key)
                if isinstance(value, str) and value:
                    found.add(value)
    except Exception as exc:  # pragma: no cover - defensive: corpus/parse failure
        _log.warning("contradiction_corpus_unavailable", error=str(exc))

    return frozenset(found)
