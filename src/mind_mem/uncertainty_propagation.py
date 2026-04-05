# Copyright 2026 STARGA, Inc.
"""Multi-hop uncertainty propagation for mind-mem retrieval.

When a query requires multiple hops (A -> B -> C), confidence earned at hop N
should reduce trust in all downstream hops.  The propagated confidence for a
hop is:

    adjusted_confidence = raw_confidence * parent_adjusted_confidence * decay

Chain confidence is the product of all adjusted confidences, giving the overall
reliability of the full multi-hop retrieval path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace


@dataclass
class HopResult:
    """A single retrieval hop in a multi-hop query chain.

    Attributes:
        block_id: Unique identifier of the retrieved memory block.
        content: Text content of the retrieved block.
        confidence: Confidence score in [0.0, 1.0].  For propagated results
            this reflects accumulated uncertainty from ancestor hops.
        hop_depth: Distance from the root query (root = 0).
        parent_hop_id: block_id of the parent hop, or None for root hops.
    """

    block_id: str
    content: str
    confidence: float
    hop_depth: int
    parent_hop_id: str | None


class UncertaintyPropagator:
    """Propagates retrieval confidence across multi-hop query chains.

    Each hop's confidence is discounted by its parent's adjusted confidence
    and a configurable decay factor, modelling the compounding uncertainty
    inherent in chained retrieval.

    Args:
        decay_factor: Multiplicative decay applied at each hop (default 0.85).
            Values closer to 1.0 preserve confidence; values closer to 0.0
            cause rapid degradation.
    """

    def __init__(self, decay_factor: float = 0.85) -> None:
        self.decay_factor = decay_factor

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def propagate(self, hops: list[HopResult]) -> list[HopResult]:
        """Return new HopResult objects with confidence adjusted for chain depth.

        The input list is never mutated.  Each returned HopResult is a new
        object whose confidence reflects the accumulated uncertainty from all
        ancestor hops in the chain.

        Root hops (parent_hop_id is None) keep their original confidence.
        Child hops are discounted:

            adjusted = raw * parent_adjusted * decay_factor

        Result confidence is clamped to [0.0, 1.0].

        Args:
            hops: Raw retrieval results, in any order.

        Returns:
            New list of HopResult objects with adjusted confidences.
        """
        if not hops:
            return []

        # Index originals by block_id for parent lookup.
        by_id: dict[str, HopResult] = {h.block_id: h for h in hops}

        # Memoize adjusted confidences to avoid repeated traversal.
        adjusted_conf: dict[str, float] = {}

        def resolve(block_id: str, visiting: frozenset[str] = frozenset()) -> float:
            if block_id in adjusted_conf:
                return adjusted_conf[block_id]
            hop = by_id[block_id]
            if (
                hop.parent_hop_id is None
                or hop.parent_hop_id not in by_id
                or hop.parent_hop_id in visiting
            ):
                # Root, unknown parent, or cycle detected — treat as root.
                value = max(0.0, min(1.0, hop.confidence))
            else:
                parent_conf = resolve(hop.parent_hop_id, visiting | {block_id})
                value = max(0.0, min(1.0, hop.confidence * parent_conf * self.decay_factor))
            adjusted_conf[block_id] = value
            return value

        for hop in hops:
            resolve(hop.block_id)

        return [replace(h, confidence=adjusted_conf[h.block_id]) for h in hops]

    def chain_confidence(self, hops: list[HopResult]) -> float:
        """Overall confidence for the complete multi-hop chain.

        Computed as the product of all adjusted hop confidences.  An empty
        chain returns 1.0 (vacuous truth — no hops, no uncertainty).

        Args:
            hops: Raw retrieval results (propagation is applied internally).

        Returns:
            Scalar in [0.0, 1.0] representing end-to-end retrieval confidence.
        """
        if not hops:
            return 1.0
        adjusted = self.propagate(hops)
        return math.prod(h.confidence for h in adjusted)

    def should_truncate(self, hop: HopResult, min_confidence: float = 0.1) -> bool:
        """Return True when a hop's confidence is too low to continue traversal.

        Callers should check this after propagation and stop expanding branches
        whose confidence has dropped below the acceptable threshold.

        Args:
            hop: A HopResult (typically already propagated).
            min_confidence: Minimum acceptable confidence (default 0.1).

        Returns:
            True if hop.confidence < min_confidence, False otherwise.
        """
        return hop.confidence < min_confidence
