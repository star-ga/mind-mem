# Copyright 2026 STARGA, Inc.
"""Cascading staleness propagation (v2.6.0).

When a block is superseded or contradicted, its neighbours in the
knowledge graph are likely to be stale too — not for certain, but
probabilistically. This module computes a staleness score in [0, 1]
for every affected block by diffusing outward from a seed set with a
distance-based decay.

The score is **advisory**: it's meant as a multiplicative penalty in
retrieval scoring, not a hard filter. A block flagged `stale=0.9` can
still be retrieved if nothing better is available.

Pure Python stdlib.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


# ---------------------------------------------------------------------------
# Decay schedule
# ---------------------------------------------------------------------------


# Default staleness by hop distance from a seed. Entry 0 is the seed
# itself; entries beyond len(_DEFAULT_DECAY)-1 are treated as clean.
_DEFAULT_DECAY: tuple[float, ...] = (1.0, 0.9, 0.5, 0.2)


@dataclass(frozen=True)
class StalenessPlan:
    """The computed staleness score per block after propagation."""

    scores: dict[str, float]  # block_id → score in [0, 1]
    seed: tuple[str, ...]
    max_hops: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "seed": list(self.seed),
            "max_hops": self.max_hops,
            "scores": dict(self.scores),
        }

    def flagged(self, threshold: float) -> list[str]:
        """Block ids whose score is >= *threshold* (sorted desc by score)."""
        above = [(bid, s) for bid, s in self.scores.items() if s >= threshold]
        above.sort(key=lambda kv: (-kv[1], kv[0]))
        return [bid for bid, _ in above]


def _effective_decay(decay: Iterable[float]) -> tuple[float, ...]:
    vals = tuple(float(x) for x in decay)
    if not vals:
        return _DEFAULT_DECAY
    for v in vals:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"decay values must be in [0, 1], got {v!r}")
    return vals


def propagate_staleness(
    seed_blocks: Iterable[str],
    adjacency: Mapping[str, Iterable[str]],
    *,
    decay: Iterable[float] = _DEFAULT_DECAY,
    max_hops: int | None = None,
) -> StalenessPlan:
    """Diffuse staleness outward from *seed_blocks* over *adjacency*.

    Args:
        seed_blocks: Block ids that are known-stale (e.g., just
            superseded or contradicted).
        adjacency: Undirected neighbour map. The caller builds this
            however they want — cross-reference graph, co-retrieval
            graph, entity graph — and the propagator stays agnostic.
        decay: Per-hop staleness. Index 0 is applied to the seeds,
            index 1 to 1-hop neighbours, and so on. Anything beyond
            the provided list is considered clean.
        max_hops: Hard cap on traversal depth. Defaults to
            ``len(decay) - 1`` which is the deepest hop that can
            receive a non-zero score.

    Returns:
        A :class:`StalenessPlan` mapping block ids to the highest
        staleness score they received. The same block may be reached
        via multiple paths; we keep the maximum so a closer seed
        always wins over a farther one.
    """
    decay_vals = _effective_decay(decay)
    hop_cap = max_hops if max_hops is not None else len(decay_vals) - 1
    if hop_cap < 0:
        return StalenessPlan(scores={}, seed=tuple(seed_blocks), max_hops=0)

    seeds: list[str] = []
    for s in seed_blocks:
        if s and s not in seeds:
            seeds.append(str(s))

    scores: dict[str, float] = {}
    visited: dict[str, int] = {}
    queue: deque[tuple[str, int]] = deque()
    for s in seeds:
        scores[s] = max(scores.get(s, 0.0), decay_vals[0])
        visited[s] = 0
        queue.append((s, 0))

    while queue:
        node, hop = queue.popleft()
        if hop >= hop_cap:
            continue
        next_hop = hop + 1
        if next_hop >= len(decay_vals):
            continue
        contribution = decay_vals[next_hop]
        for neighbour in adjacency.get(node, ()):  # type: ignore[arg-type]
            n = str(neighbour)
            prior = visited.get(n)
            if prior is not None and prior <= next_hop:
                # Already visited at equal-or-closer hop; skip.
                continue
            visited[n] = next_hop
            scores[n] = max(scores.get(n, 0.0), contribution)
            queue.append((n, next_hop))

    return StalenessPlan(
        scores=scores,
        seed=tuple(seeds),
        max_hops=hop_cap,
    )


__all__ = ["StalenessPlan", "propagate_staleness"]
