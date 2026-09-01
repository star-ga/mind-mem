"""``uncertainty_propagation`` wired into the two multi-hop walks that need it.

The module shipped with a full unit suite (``test_uncertainty_propagation.py``)
and no caller, so the arithmetic was proven and never used. This file pins the
wiring instead of the arithmetic: that the walks actually route through
``propagate()`` / ``should_truncate()`` / ``chain_confidence()``, and that
deleting those calls fails a test rather than quietly restoring the bug.

The bug being closed is an over-confidence one. Before this wiring a block
three hops out was reported exactly like a direct hit — ``graph_expand``
appended it with a ``_graph_hop`` label and no confidence at all, and
``traverse_graph`` reported every id in a causal chain with the same standing
regardless of depth. A consumer reading either surface treats triple-indirect
evidence as first-hand.

Two properties are asserted throughout:

* **Flag ON** — a 3-hop chain at 0.9 per hop yields *strictly decreasing*
  adjusted confidence, and a branch that falls under ``min_confidence`` is
  pruned mid-walk (not appended, not walked past).
* **Flag OFF** — every response is exactly what it was before the flag
  existed: same blocks, same order, same scores, same key sets, no new field.
  ``retrieval.multi_hop.uncertainty`` defaults to off, so this is the path
  every existing deployment stays on.
"""

from __future__ import annotations

import json
import os
from typing import Any

import pytest
from _recall_clock_sentinel import clock_census

from mind_mem.graph_recall import (
    graph_expand,
    is_uncertainty_enabled,
    resolve_chain_decay,
    resolve_graph_config,
)

# Canonical ``D-YYYYMMDD-NNN`` ids so ``_recall_constants._BLOCK_ID_RE``
# recognises the cross-references in block text.
H0 = "D-20260420-100"
H1 = "D-20260420-101"
H2 = "D-20260420-102"
H3 = "D-20260420-103"
H4 = "D-20260420-104"

#: 0.9 raw confidence per hop, and a rank decay of 1.0 so the confidence
#: sequence is the propagator's alone and not entangled with score decay.
HOP_CONF = 0.9
NO_RANK_DECAY = 1.0


def _block(bid: str, text: str = "") -> dict:
    return {"_id": bid, "content": text, "type": "decision", "Statement": text}


def _chain_corpus() -> list[dict]:
    """H0 -> H1 -> H2 -> H3 -> H4, a single unbranched xref chain."""
    return [
        _block(H0, f"See {H1}"),
        _block(H1, f"See {H2}"),
        _block(H2, f"See {H3}"),
        _block(H3, f"See {H4}"),
        _block(H4, "leaf"),
    ]


def _seeds() -> list[dict]:
    return [{"_id": H0, "score": 10.0}]


def _by_hop(out: list[dict]) -> dict[int, dict]:
    return {r["_graph_hop"]: r for r in out if "_graph_hop" in r}


# ---------------------------------------------------------------------------
# graph_expand — propagate() + should_truncate() on the recall walk
# ---------------------------------------------------------------------------


class TestGraphExpandUncertainty:
    def test_three_hop_chain_confidence_strictly_decreases(self) -> None:
        """The headline property: hop 3 must not read like hop 1.

        At 0.9 per hop the propagator chains each hop through its parent's
        *already adjusted* confidence, so the sequence is 0.9, 0.81, 0.729 —
        strictly decreasing. Remove the ``propagate()`` call and the field is
        gone; replace it with the raw per-edge value and the sequence goes
        flat at 0.9. Both are caught here.
        """
        out = graph_expand(
            _seeds(),
            _chain_corpus(),
            max_hops=3,
            decay=NO_RANK_DECAY,
            uncertainty=True,
            hop_confidence=HOP_CONF,
            min_confidence=0.0,
        )
        by_hop = _by_hop(out)
        assert sorted(by_hop) == [1, 2, 3], "the walk must reach all three hops"

        confidences = [by_hop[h]["_hop_confidence"] for h in (1, 2, 3)]
        assert confidences == [
            pytest.approx(0.9),
            pytest.approx(0.81),
            pytest.approx(0.729),
        ]
        # Stated as the property, not just the numbers: strictly decreasing.
        assert all(a > b for a, b in zip(confidences, confidences[1:]))
        # And the specific over-confidence claim: hop 3 is not hop 1.
        assert by_hop[3]["_hop_confidence"] < by_hop[1]["_hop_confidence"]

    def test_confidence_is_chained_through_the_parent_not_the_seed(self) -> None:
        """Each hop discounts against its immediate parent, not the seed.

        Seed-relative would give 0.9 at every hop (or 0.9 ** 1 each time);
        parent-chained compounds. The distinction is the whole point of
        ``propagate()`` and is invisible at hop 1.
        """
        out = graph_expand(
            _seeds(),
            _chain_corpus(),
            max_hops=3,
            decay=NO_RANK_DECAY,
            uncertainty=True,
            hop_confidence=HOP_CONF,
            min_confidence=0.0,
        )
        by_hop = _by_hop(out)
        assert by_hop[2]["_hop_confidence"] == pytest.approx(by_hop[1]["_hop_confidence"] * HOP_CONF)
        assert by_hop[3]["_hop_confidence"] == pytest.approx(by_hop[2]["_hop_confidence"] * HOP_CONF)

    def test_the_rank_decay_also_discounts_trust(self) -> None:
        """``decay`` is the propagator's ``decay_factor`` on this path.

        One knob, so a hop cannot lose rank while keeping trust.
        """
        out = graph_expand(
            _seeds(),
            _chain_corpus(),
            max_hops=2,
            decay=0.5,
            uncertainty=True,
            hop_confidence=HOP_CONF,
            min_confidence=0.0,
        )
        by_hop = _by_hop(out)
        assert by_hop[1]["_hop_confidence"] == pytest.approx(0.9 * 0.5)
        assert by_hop[2]["_hop_confidence"] == pytest.approx(0.9 * 0.45 * 0.5)

    def test_a_sub_threshold_branch_is_truncated(self) -> None:
        """The second half of "working": a branch under 0.1 is dropped.

        With rank decay 0.5 the chain reads 0.45, 0.2025, 0.0911 — the third
        hop falls under the 0.1 default and must not be appended at all.
        Dropping ``should_truncate()`` returns it and fails here.
        """
        out = graph_expand(
            _seeds(),
            _chain_corpus(),
            max_hops=3,
            decay=0.5,
            uncertainty=True,
            hop_confidence=HOP_CONF,
            min_confidence=0.1,
        )
        by_hop = _by_hop(out)
        assert sorted(by_hop) == [1, 2]
        assert H3 not in {r["_id"] for r in out}
        assert by_hop[2]["_hop_confidence"] == pytest.approx(0.2025)

    def test_truncation_stops_the_branch_rather_than_skipping_one_node(self) -> None:
        """A pruned node is not walked past either — the whole tail goes.

        Skipping only the offending node while still enqueuing it would leave
        hop 4 reachable, which is the failure this asserts against.
        """
        out = graph_expand(
            _seeds(),
            _chain_corpus(),
            max_hops=4,
            decay=0.5,
            uncertainty=True,
            hop_confidence=HOP_CONF,
            min_confidence=0.1,
        )
        ids = {r["_id"] for r in out}
        assert H3 not in ids
        assert H4 not in ids

    def test_a_generous_threshold_keeps_the_whole_chain(self) -> None:
        """Truncation is threshold-driven, not a blanket depth cut."""
        out = graph_expand(
            _seeds(),
            _chain_corpus(),
            max_hops=3,
            decay=0.5,
            uncertainty=True,
            hop_confidence=HOP_CONF,
            min_confidence=0.05,
        )
        assert sorted(_by_hop(out)) == [1, 2, 3]

    def test_the_walk_is_deterministic_and_reads_no_clock(self) -> None:
        """Recall is a pure function of (corpus, config, scoring_instant).

        The census observes every ``datetime.now`` / ``date.today`` executed
        anywhere inside ``mind_mem``; an ``except`` clause cannot hide a read
        from it.
        """
        with clock_census() as census:
            first = graph_expand(
                _seeds(),
                _chain_corpus(),
                max_hops=3,
                decay=0.5,
                uncertainty=True,
                hop_confidence=HOP_CONF,
                min_confidence=0.1,
            )
            second = graph_expand(
                _seeds(),
                _chain_corpus(),
                max_hops=3,
                decay=0.5,
                uncertainty=True,
                hop_confidence=HOP_CONF,
                min_confidence=0.1,
            )
        census.assert_clock_free()
        assert first == second


class TestGraphExpandFlagOff:
    """Default-off must be byte-identical to the pre-flag behaviour."""

    def test_flag_off_output_is_identical_to_the_pre_flag_walk(self) -> None:
        off = graph_expand(_seeds(), _chain_corpus(), max_hops=3, decay=0.5)
        # The literal pre-flag contract: three appended blocks, seed-relative
        # score decay, ``_graph_parent`` naming the seed, nothing else added.
        assert [r["_id"] for r in off] == [H0, H1, H2, H3]
        assert [r.get("_graph_hop") for r in off] == [None, 1, 2, 3]
        assert [r["score"] for r in off] == [
            pytest.approx(10.0),
            pytest.approx(5.0),
            pytest.approx(2.5),
            pytest.approx(1.25),
        ]
        assert {r.get("_graph_parent") for r in off[1:]} == {H0}

    def test_flag_off_adds_no_confidence_field(self) -> None:
        off = graph_expand(_seeds(), _chain_corpus(), max_hops=3, decay=0.5)
        assert all("_hop_confidence" not in r for r in off)

    def test_flag_off_never_truncates(self) -> None:
        """The pruning that the flag buys must not leak into the default path.

        Same corpus and decay as ``test_a_sub_threshold_branch_is_truncated``,
        where the flag-on walk stops at hop 2.
        """
        off = graph_expand(_seeds(), _chain_corpus(), max_hops=3, decay=0.5)
        assert sorted(_by_hop(off)) == [1, 2, 3]

    def test_the_default_is_off(self) -> None:
        """Not passing the argument at all is the same as passing False."""
        implicit = graph_expand(_seeds(), _chain_corpus(), max_hops=3, decay=0.5)
        explicit = graph_expand(_seeds(), _chain_corpus(), max_hops=3, decay=0.5, uncertainty=False)
        assert implicit == explicit


# ---------------------------------------------------------------------------
# Config gate — the path hybrid_recall._maybe_graph_expand actually takes
# ---------------------------------------------------------------------------


def _cfg(**uncertainty: Any) -> dict:
    return {"retrieval": {"multi_hop": {"uncertainty": uncertainty}}}


class TestConfigGate:
    def test_absent_block_is_off(self) -> None:
        assert is_uncertainty_enabled(None) is False
        assert is_uncertainty_enabled({}) is False
        assert is_uncertainty_enabled({"retrieval": {"multi_hop": {}}}) is False

    @pytest.mark.parametrize("value", [False, "true", 1, None, {}])
    def test_anything_but_literal_true_is_off(self, value: Any) -> None:
        """Fail-closed: a typo cannot switch the surface on."""
        assert is_uncertainty_enabled(_cfg(enabled=value)) is False

    def test_enabled_true_is_on(self) -> None:
        assert is_uncertainty_enabled(_cfg(enabled=True)) is True

    def test_flag_off_config_yields_the_historical_three_key_params(self) -> None:
        """``resolve_graph_config`` must not grow keys on the default path.

        Its result is splatted straight into ``graph_expand``; an unconditional
        new key would change every existing call.
        """
        assert resolve_graph_config(_cfg()) == {"max_hops": 2, "decay": 0.5, "max_neighbors_per_hop": 5}
        assert resolve_graph_config(None) == {"max_hops": 2, "decay": 0.5, "max_neighbors_per_hop": 5}

    def test_enabled_config_carries_the_knobs(self) -> None:
        params = resolve_graph_config(_cfg(enabled=True, hop_confidence=0.7, min_confidence=0.2))
        assert params["uncertainty"] is True
        assert params["hop_confidence"] == pytest.approx(0.7)
        assert params["min_confidence"] == pytest.approx(0.2)

    @pytest.mark.parametrize("bad", [-0.5, 1.5, "0.7", True, None])
    def test_out_of_range_knobs_fall_back_to_defaults(self, bad: Any) -> None:
        params = resolve_graph_config(_cfg(enabled=True, hop_confidence=bad, min_confidence=bad))
        assert params["hop_confidence"] == pytest.approx(0.9)
        assert params["min_confidence"] == pytest.approx(0.1)

    def test_resolved_params_are_exactly_the_call_hybrid_recall_makes(self) -> None:
        """``graph_expand(results, all_blocks, **resolve_graph_config(cfg))``.

        That is the literal call site in ``hybrid_recall._maybe_graph_expand``,
        so this proves the config→walk path end to end rather than asserting
        on a dict in isolation. A knob the walk does not accept would raise
        ``TypeError`` here.
        """
        cfg = _cfg(enabled=True, hop_confidence=HOP_CONF, min_confidence=0.0)
        cfg["retrieval"]["multi_hop"]["max_hops"] = 3
        cfg["retrieval"]["multi_hop"]["decay"] = NO_RANK_DECAY

        params = resolve_graph_config(cfg)
        out = graph_expand(_seeds(), _chain_corpus(), **params)
        by_hop = _by_hop(out)
        assert [by_hop[h]["_hop_confidence"] for h in (1, 2, 3)] == [
            pytest.approx(0.9),
            pytest.approx(0.81),
            pytest.approx(0.729),
        ]

        off_params = resolve_graph_config({"retrieval": {"multi_hop": {"max_hops": 3, "decay": NO_RANK_DECAY}}})
        assert all("_hop_confidence" not in r for r in graph_expand(_seeds(), _chain_corpus(), **off_params))

    def test_chain_decay_is_separate_from_the_rank_decay(self) -> None:
        assert resolve_chain_decay(None) == pytest.approx(0.85)
        assert resolve_chain_decay(_cfg(enabled=True)) == pytest.approx(0.85)
        assert resolve_chain_decay(_cfg(enabled=True, chain_decay=0.5)) == pytest.approx(0.5)
        assert resolve_chain_decay(_cfg(enabled=True, chain_decay=7)) == pytest.approx(0.85)


# ---------------------------------------------------------------------------
# traverse_graph — chain_confidence() on the causal-graph envelope
# ---------------------------------------------------------------------------

CHAIN = (H3, H2, H1, H0)  # H3 depends_on H2 depends_on H1 depends_on H0
EDGE_WEIGHT = 0.9


@pytest.fixture
def causal_ws(tmp_path):
    """A 3-hop causal chain, every edge weighted 0.9."""
    from mind_mem.causal_graph import CausalGraph

    root = tmp_path / "ws"
    (root / ".mind-mem-index").mkdir(parents=True)
    cg = CausalGraph(str(root))
    for source, target in zip(CHAIN, CHAIN[1:]):
        cg.add_edge(source, target, "depends_on", weight=EDGE_WEIGHT)
    return root


def _traverse(ws, *, block_id: str, direction: str, depth: int = 3, uncertainty: dict | None = None) -> dict:
    from mind_mem.mcp.infra.workspace import use_workspace
    from mind_mem.mcp.tools.graph import traverse_graph

    config = os.path.join(str(ws), "mind-mem.json")
    if uncertainty is None:
        if os.path.exists(config):
            os.remove(config)
    else:
        with open(config, "w", encoding="utf-8") as fh:
            json.dump({"retrieval": {"multi_hop": {"uncertainty": uncertainty}}}, fh)
    with use_workspace(str(ws)):
        # ``__wrapped__`` skips the ACL/observability decorator, matching the
        # existing traverse_graph tests.
        return json.loads(traverse_graph.__wrapped__(block_id, depth=depth, direction=direction))


class TestTraverseGraphChainConfidence:
    def test_a_three_hop_chain_decreases_and_carries_an_end_to_end_number(self, causal_ws) -> None:
        out = _traverse(
            causal_ws,
            block_id=H3,
            direction="upstream",
            uncertainty={"enabled": True, "chain_decay": 1.0},
        )
        annotated = out["upstream"]["chain_confidence"]
        assert len(annotated) == 1
        entry = annotated[0]
        assert entry["chain"] == list(CHAIN)

        depths = {h["depth"]: h["confidence"] for h in entry["hops"]}
        assert depths[0] == pytest.approx(1.0), "the queried block is hop 0, fully trusted"
        assert [depths[d] for d in (1, 2, 3)] == [
            pytest.approx(0.9),
            pytest.approx(0.81),
            pytest.approx(0.729),
        ]
        assert depths[3] < depths[2] < depths[1]

        # chain_confidence() is the product of every adjusted hop.
        assert entry["confidence"] == pytest.approx(1.0 * 0.9 * 0.81 * 0.729)

    def test_chain_decay_further_discounts_each_hop(self, causal_ws) -> None:
        out = _traverse(
            causal_ws,
            block_id=H3,
            direction="upstream",
            uncertainty={"enabled": True, "chain_decay": 0.5},
        )
        depths = {h["depth"]: h["confidence"] for h in out["upstream"]["chain_confidence"][0]["hops"]}
        assert depths[1] == pytest.approx(0.9 * 1.0 * 0.5)
        assert depths[2] == pytest.approx(0.9 * depths[1] * 0.5)

    def test_a_downstream_dependent_three_hops_out_is_not_a_direct_dependent(self, causal_ws) -> None:
        """The reported bug, from the other direction.

        Before the wiring every ``reachable_nodes`` entry carried a depth and
        no confidence, so a depth-3 dependent read exactly like a depth-1 one.
        """
        out = _traverse(
            causal_ws,
            block_id=H0,
            direction="downstream",
            uncertainty={"enabled": True, "chain_decay": 1.0},
        )
        by_depth = {n["depth"]: n["confidence"] for n in out["downstream"]["reachable_nodes"]}
        assert [by_depth[d] for d in (1, 2, 3)] == [
            pytest.approx(0.9),
            pytest.approx(0.81),
            pytest.approx(0.729),
        ]
        assert by_depth[3] < by_depth[1]

    def test_a_weak_edge_starts_the_chain_weak(self, causal_ws, tmp_path) -> None:
        """Raw hop confidence is the stored edge weight, not a constant."""
        from mind_mem.causal_graph import CausalGraph

        cg = CausalGraph(str(causal_ws))
        cg.add_edge(H3, H2, "depends_on", weight=0.2)
        out = _traverse(
            causal_ws,
            block_id=H3,
            direction="upstream",
            uncertainty={"enabled": True, "chain_decay": 1.0},
        )
        depths = {h["depth"]: h["confidence"] for h in out["upstream"]["chain_confidence"][0]["hops"]}
        assert depths[1] == pytest.approx(0.2)
        assert depths[2] == pytest.approx(0.2 * 0.9)


class TestTraverseGraphFlagOff:
    def test_flag_off_envelope_has_exactly_the_historical_keys(self, causal_ws) -> None:
        out = _traverse(causal_ws, block_id=H3, direction="both")
        assert set(out["upstream"]) == {"direct_dependencies", "causal_chains"}
        assert set(out["downstream"]) == {"direct_dependents", "reachable_nodes"}

    def test_flag_off_reachable_nodes_carry_no_confidence(self, causal_ws) -> None:
        out = _traverse(causal_ws, block_id=H0, direction="downstream")
        nodes = out["downstream"]["reachable_nodes"]
        assert nodes, "the fixture chain must produce downstream nodes"
        assert all("confidence" not in n for n in nodes)
        assert all(set(n) == {"block_id", "depends_on", "edge_type", "depth"} for n in nodes)

    def test_flag_off_and_flag_on_agree_on_everything_but_the_new_keys(self, causal_ws) -> None:
        off = _traverse(causal_ws, block_id=H3, direction="upstream")
        on = _traverse(
            causal_ws,
            block_id=H3,
            direction="upstream",
            uncertainty={"enabled": True},
        )
        assert on["upstream"]["causal_chains"] == off["upstream"]["causal_chains"]
        assert on["upstream"]["direct_dependencies"] == off["upstream"]["direct_dependencies"]
        assert set(on["upstream"]) - set(off["upstream"]) == {"chain_confidence"}
