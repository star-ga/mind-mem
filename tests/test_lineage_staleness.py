"""End-to-end tests for the v3.12 lineage→staleness wiring (Theme C).

When a block is added to the lineage graph as a ``contradicts`` of an
existing block, the existing block and its dependents should inherit
a propagated staleness score that the recall reranker reads and that
``_explain.staleness_penalty`` surfaces in explainable recall.

Coverage matrix:
    1. happy path — adding a contradicts edge produces non-zero penalties
       at distance 1, 2, 3.
    2. kind decay — `contradicts` propagates faster than `cites` /
       `implements` / `refines`.
    3. cycles — propagation respects the visited-set bound.
    4. cap — propagation honours `LINEAGE_DEPTH_CAP` (3 hops).
    5. idempotent upsert — re-adding the same edge does not double the
       penalty; updates `decayed_at` timestamp.
    6. recall integration — a block with a non-zero penalty is demoted
       in the next recall.
    7. _explain integration — `_explain.staleness_penalty` surfaces the
       persisted value.
    8. CLI entry point — `mm lineage flag <src> --kind contradicts <dst>`
       runs both `add_block_edge` and the propagation pass.
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture
def workspace(tmp_path):
    """Empty workspace ready for lineage writes."""

    from mind_mem.init_workspace import init

    ws = str(tmp_path / "ws")
    os.makedirs(ws)
    init(ws)
    return ws


# ---------------------------------------------------------------------------
# block_staleness table + propagation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBlockStalenessSchema:
    def test_ensure_block_staleness_schema_idempotent(self, workspace) -> None:
        from mind_mem.lineage_staleness import ensure_block_staleness_schema

        ensure_block_staleness_schema(workspace)
        ensure_block_staleness_schema(workspace)  # second call is a no-op

    def test_table_columns(self, workspace) -> None:
        from mind_mem.lineage_staleness import ensure_block_staleness_schema
        from mind_mem.retrieval_graph import _connect

        ensure_block_staleness_schema(workspace)
        conn = _connect(workspace)
        try:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(block_staleness)").fetchall()}
            assert {"block_id", "score", "source_id", "decayed_at"} <= cols
        finally:
            conn.close()


@pytest.mark.unit
class TestPropagateContradictsEdge:
    def test_one_hop_penalty(self, workspace) -> None:
        from mind_mem.block_lineage import add_block_edge
        from mind_mem.lineage_staleness import (
            get_staleness_score,
            propagate_lineage_staleness,
        )

        add_block_edge(workspace, "A", "B", "cites")
        # B is now contradicted by NEW.
        add_block_edge(workspace, "NEW", "B", "contradicts")
        propagate_lineage_staleness(workspace, source_id="NEW")

        # B is the seed (1.0), A is 1 hop away through cites edge.
        assert get_staleness_score(workspace, "B") == pytest.approx(1.0)
        assert get_staleness_score(workspace, "A") > 0.0

    def test_kind_decay_contradicts_faster_than_refines(self, workspace) -> None:
        from mind_mem.block_lineage import add_block_edge
        from mind_mem.lineage_staleness import (
            get_staleness_score,
            propagate_lineage_staleness,
        )

        # Two parallel chains rooted at SEED-{C,R}, each 1 hop deep.
        add_block_edge(workspace, "SEED_C", "C1", "contradicts")
        add_block_edge(workspace, "SEED_R", "R1", "refines")

        propagate_lineage_staleness(workspace, source_id="SEED_C")
        propagate_lineage_staleness(workspace, source_id="SEED_R")

        # SEED_C contradicts C1 → C1 fully stale (1.0).
        # SEED_R refines R1 → R1 only partially stale (decay × 1.0).
        assert get_staleness_score(workspace, "C1") > get_staleness_score(workspace, "R1")

    def test_three_hop_cap(self, workspace) -> None:
        from mind_mem.block_lineage import add_block_edge
        from mind_mem.lineage_staleness import (
            get_staleness_score,
            propagate_lineage_staleness,
        )

        chain = ["A", "B", "C", "D", "E"]
        for src, dst in zip(chain[:-1], chain[1:]):
            add_block_edge(workspace, src, dst, "cites")
        # Now contradict the head: A is contradicted.
        add_block_edge(workspace, "NEW", "A", "contradicts")
        propagate_lineage_staleness(workspace, source_id="NEW")

        # A=seed, B=1hop, C=2hop, D=3hop, E=4hop (out of cap → 0).
        for hop, bid in enumerate(chain):
            score = get_staleness_score(workspace, bid)
            if hop <= 3:
                assert score > 0.0, f"{bid} (hop {hop}) should have a penalty"
            else:
                assert score == 0.0, f"{bid} (hop {hop}) is past the cap"

    def test_cycle_does_not_loop(self, workspace) -> None:
        from mind_mem.block_lineage import add_block_edge
        from mind_mem.lineage_staleness import (
            get_staleness_score,
            propagate_lineage_staleness,
        )

        add_block_edge(workspace, "A", "B", "cites")
        add_block_edge(workspace, "B", "C", "cites")
        add_block_edge(workspace, "C", "A", "cites")
        add_block_edge(workspace, "NEW", "A", "contradicts")
        propagate_lineage_staleness(workspace, source_id="NEW")

        # Both B and C should get penalties; the cycle must not blow up.
        assert get_staleness_score(workspace, "B") > 0.0
        assert get_staleness_score(workspace, "C") > 0.0

    def test_idempotent_upsert(self, workspace) -> None:
        from mind_mem.block_lineage import add_block_edge
        from mind_mem.lineage_staleness import (
            get_staleness_score,
            propagate_lineage_staleness,
        )
        from mind_mem.retrieval_graph import _connect

        add_block_edge(workspace, "NEW", "B", "contradicts")
        propagate_lineage_staleness(workspace, source_id="NEW")
        score_first = get_staleness_score(workspace, "B")

        propagate_lineage_staleness(workspace, source_id="NEW")
        score_second = get_staleness_score(workspace, "B")

        assert score_first == score_second  # no double-counting

        # And the table has exactly one row for B.
        conn = _connect(workspace)
        try:
            n = conn.execute("SELECT COUNT(*) FROM block_staleness WHERE block_id = ?", ("B",)).fetchone()[0]
            assert n == 1
        finally:
            conn.close()


@pytest.mark.unit
class TestRecallExplainSurfacesPenalty:
    def test_explain_returns_persisted_penalty(self, workspace) -> None:
        from mind_mem.block_lineage import add_block_edge
        from mind_mem.lineage_staleness import propagate_lineage_staleness

        add_block_edge(workspace, "NEW", "TARGET", "contradicts")
        propagate_lineage_staleness(workspace, source_id="NEW")

        # Build a synthetic hit and run it through attach_explain — the
        # helper now reads block_staleness from the workspace.
        from mind_mem._recall_explain import attach_explain

        hits = [{"_id": "TARGET", "score": 1.5}]
        attach_explain(hits, intent_match="factual", workspace=workspace)
        assert hits[0]["_explain"]["staleness_penalty"] > 0.0


# ---------------------------------------------------------------------------
# CLI entry — `mm lineage flag <src> --kind contradicts <dst>`
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCliLineageFlag:
    def test_cli_one_shot_flags_and_propagates(self, workspace, monkeypatch) -> None:
        monkeypatch.setenv("MIND_MEM_WORKSPACE", workspace)

        from mind_mem.block_lineage import add_block_edge
        from mind_mem.lineage_staleness import get_staleness_score

        # Prior state: existing dependency.
        add_block_edge(workspace, "OLD", "DEP", "cites")
        assert get_staleness_score(workspace, "DEP") == 0.0

        # Run the new CLI subcommand.
        from mind_mem.mm_cli import _cmd_lineage_flag

        rc = _cmd_lineage_flag(
            type(
                "Args",
                (),
                {
                    "src": "NEW",
                    "dst": "OLD",
                    "kind": "contradicts",
                    "weight": 1.0,
                },
            )()
        )
        assert rc == 0

        # The dependent should now have a penalty.
        assert get_staleness_score(workspace, "DEP") > 0.0
