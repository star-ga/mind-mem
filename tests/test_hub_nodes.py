"""Degree-gated hub-node selection — which nodes earn a synthesised profile.

ROADMAP ("Hub-node profile synthesis (degree-gated)"): "for high-degree nodes only
(degree >= 3), pool every mention + graph neighborhood into a synthesized profile
(summary + 3-5 traceable atomic facts + structured time range), 'resolve
contradictions by preferring the most specific claim, invent nothing.' Reuses the
Group H recompaction fixed-point + injected-compressor machinery; emits as a
propose_update, never a direct write."

The SYNTHESIS needs a compressor (a model). The GATE does not, and the gate is what
decides which nodes get one -- so it ships first and independently. Without it,
"synthesise profiles" has no defensible scope and would either run over every node
(expensive, and most nodes have nothing to pool) or over a hand-picked list.

Three properties pinned here, each a way the gate could be quietly wrong:
  * the threshold is >= 3, not > 3 -- an off-by-one silently excludes the smallest
    real hubs, which are the majority of them;
  * degree counts DISTINCT neighbours, so ten edges to one neighbour is degree 1
    and does not qualify. A repeated edge is not breadth, and counting it as such
    would nominate nodes with nothing to synthesise;
  * direction is ignored. A node referenced by three others is as much a hub as one
    referencing three.

Pure: no model, no clock, no I/O.
"""

from __future__ import annotations

from mind_mem.hub_nodes import HUB_DEGREE_THRESHOLD, degree_of, hub_nodes


def _edges():
    # A-hub has three DISTINCT neighbours; B-multi has ten edges to one neighbour.
    return [
        ("A-hub", "n1"), ("A-hub", "n2"), ("A-hub", "n3"),
        *[("B-multi", "n9") for _ in range(10)],
        ("C-pair", "n1"), ("C-pair", "n2"),
    ]


def test_the_threshold_is_three_inclusive():
    """>= 3, not > 3. An off-by-one excludes the smallest real hubs."""
    assert HUB_DEGREE_THRESHOLD == 3


def test_degree_counts_distinct_neighbours():
    assert degree_of(_edges(), "A-hub") == 3


def test_repeated_edges_to_one_neighbour_are_degree_one():
    """A repeated edge is not breadth; counting it as such nominates a node with
    nothing to pool."""
    assert degree_of(_edges(), "B-multi") == 1


def test_direction_is_ignored():
    """Referenced-by-three is as much a hub as references-three."""
    edges = [("x", "T"), ("y", "T"), ("z", "T")]
    assert degree_of(edges, "T") == 3
    assert "T" in hub_nodes(edges)


def test_a_node_at_the_threshold_qualifies():
    assert "A-hub" in hub_nodes(_edges())


def test_a_node_below_the_threshold_does_not():
    """POSITIVE CONTROL: a gate admitting everything would pass the test above."""
    got = hub_nodes(_edges())
    assert "C-pair" not in got, got
    assert "B-multi" not in got, got


def test_self_edges_do_not_inflate_degree():
    """A node referencing itself has learned nothing about its neighbourhood."""
    edges = [("S", "S"), ("S", "n1"), ("S", "n2")]
    assert degree_of(edges, "S") == 2
    assert "S" not in hub_nodes(edges)


def test_the_result_is_sorted_and_deterministic():
    edges = [("b", "1"), ("b", "2"), ("b", "3"), ("a", "1"), ("a", "2"), ("a", "3")]
    assert hub_nodes(edges) == ["a", "b"]
    assert hub_nodes(edges) == hub_nodes(list(edges))


def test_an_empty_graph_yields_no_hubs_and_does_not_raise():
    assert hub_nodes([]) == []
    assert degree_of([], "missing") == 0


def test_a_malformed_edge_is_skipped_not_fatal():
    """A graph read from disk can carry a short row; one must not kill the sweep."""
    edges = [("A", "n1"), ("A",), (), ("A", "n2"), ("A", "n3")]  # type: ignore[list-item]
    assert degree_of(edges, "A") == 3
