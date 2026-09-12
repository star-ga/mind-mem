"""Degree-gated hub-node selection.

ROADMAP ("Hub-node profile synthesis (degree-gated)"): "for high-degree nodes only
(degree >= 3), pool every mention + graph neighborhood into a synthesized profile".

This module is the GATE, not the synthesis. It answers exactly one question -- which
nodes earn a synthesised profile -- and it answers it without a model, a clock, or a
file. The synthesis step needs a compressor; the gate does not, and shipping the gate
separately is what gives "synthesise profiles" a defensible scope. Without it the
step either runs over every node (most of which have nothing to pool) or over a list
somebody wrote by hand.

Degree here is the number of DISTINCT neighbours. Ten edges to one neighbour is not
breadth, and counting them as breadth would nominate a node whose whole
"neighbourhood" is a single repeated mention. Direction is ignored: a node referenced
by three others has as much to pool as one referencing three. Self-edges do not count
-- a node referencing itself has learned nothing about its surroundings.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence

__all__ = ["HUB_DEGREE_THRESHOLD", "degree_of", "neighbours_of", "hub_nodes"]

#: The roadmap says "degree >= 3", and the comparison is inclusive. An exclusive
#: comparison would exclude every 3-neighbour node, which is the largest and most
#: common class of real hub -- the off-by-one is not a rounding detail, it is most of
#: the population.
HUB_DEGREE_THRESHOLD = 3

Edge = Sequence[str]


def _adjacency(edges: Iterable[Edge]) -> dict[str, set[str]]:
    """Undirected adjacency, self-edges dropped, malformed rows skipped.

    A graph read from disk can carry a short or empty row; one must not kill a sweep
    over thousands of nodes, and a skipped row is visibly absent from the counts
    rather than silently mis-counted as a neighbour.
    """
    adj: dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        if len(edge) < 2:
            continue
        src, dst = edge[0], edge[1]
        if not src or not dst:
            continue
        adj[src]  # noqa: B018 -- a node with only self-edges still exists, at degree 0
        adj[dst]
        if src == dst:
            continue
        adj[src].add(dst)
        adj[dst].add(src)
    return adj


def neighbours_of(edges: Iterable[Edge], node: str) -> list[str]:
    """The distinct neighbours of `node`, sorted. Empty for an unknown node."""
    return sorted(_adjacency(edges).get(node, set()))


def degree_of(edges: Iterable[Edge], node: str) -> int:
    """Count of DISTINCT neighbours, ignoring direction and self-edges."""
    return len(_adjacency(edges).get(node, set()))


def hub_nodes(edges: Iterable[Edge]) -> list[str]:
    """Every node whose distinct-neighbour count reaches HUB_DEGREE_THRESHOLD.

    Sorted, so two runs over the same graph are comparable: the result decides what a
    compressor is asked to summarise, and a reviewer diffing two runs must see a real
    change rather than a reshuffle.
    """
    adj = _adjacency(edges)
    return sorted(n for n, near in adj.items() if len(near) >= HUB_DEGREE_THRESHOLD)
