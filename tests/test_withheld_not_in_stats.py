# Copyright 2026 STARGA, Inc.
"""Withheld content must not be inferable from the index's AGGREGATES.

The gate withholds a quarantined block's bytes. It did not withhold the
block's effect on the numbers a caller can read, and those numbers are
served to **user** scope: ``index_stats`` and ``memory_health`` return the
index's block count, and every recall score is a ``bm25()`` value computed
over the whole FTS5 table. Measured on a 12-block corpus before the fix
(``build_index`` on two workspaces differing only by one quarantined
block):

===========================================  ==========  ==========
reading                                      admitted    +1 withheld
===========================================  ==========  ==========
``index_status(ws)["blocks"]``                       12          13
``bm25`` of an admitted doc, term SHARED        -1.9688     -1.6026
``bm25`` of an admitted doc, term NOT shared    -2.8718     -3.1645
``query_index`` score, shared term               2.3266      1.8489
``query_index`` score, unrelated term            3.3580      3.6175
===========================================  ==========  ==========

The third row is the one that settles it: the quarantined block does not
contain that term at all. It moved the score anyway, through the document
count and the average document length that ``bm25`` divides by. So the
existence of withheld content was readable off the ranking of a query that
has nothing to do with it, by a caller the gate refuses to show it to.

Every negative assertion here is paired with a positive control, because
"the canary is not in the statistics" passes trivially if the canary was
never written. The controls prove three separate things: the fixture holds
both blocks, the gate distinguishes them, and the *method used to look*
can see the withheld block when it is allowed to (``blocks_fts_withheld``,
``index_status(include_withheld=True)``).
"""

from __future__ import annotations

import inspect
import os
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from mind_mem import sqlite_index
from mind_mem.admissibility import is_admissible_status
from mind_mem.block_parser import parse_file
from mind_mem.block_store import _render_block
from mind_mem.init_workspace import init
from mind_mem.sqlite_index import (
    _aggregate_facts_to_parents,
    _db_path,
    build_index,
    index_status,
    is_stale,
    merkle_leaves,
    query_index,
)

#: A term two ADMITTED blocks share with the quarantined one — the classic
#: term-frequency channel.
SHARED = "zeppelinography"
#: A term ONLY an admitted block carries. The quarantined block still moves
#: its score, through the corpus-wide document count and length average.
LONELY = "kestrelwatching"
#: Text that exists only in the withheld block.
CANARY = "QUARANTINEDCANARY9d41b7"

_FILLER = " ".join(f"filler{i}" for i in range(40))


def _decisions(with_withheld: bool, *, withheld_status: str = "quarantined") -> list[dict]:
    """Twelve admitted decisions, optionally plus one withheld block.

    Twelve rather than two on purpose: FTS5's IDF is degenerate on a
    two-document corpus (it clamps to a floor), so a small fixture can
    make a real statistics shift look like no shift at all.
    """
    blocks = []
    for i in range(1, 13):
        text = f"Decision number {i} about databases and pipelines and governance."
        if i in (1, 2):
            text = f"Decision number {i} about {SHARED} plumage and migration routes."
        if i == 3:
            text = f"Decision number {i} about the {LONELY} burrow survey."
        blocks.append(
            {
                "_id": f"D-2026010{i // 10}-{i:03d}",
                "Statement": text,
                "Date": "2026-01-01",
                "Status": "active",
                "Type": "decision",
            }
        )
    if with_withheld:
        blocks.append(
            {
                "_id": "D-20260101-099",
                "Statement": f"{CANARY} {SHARED} {SHARED} {SHARED} {SHARED} {_FILLER}",
                "Date": "2026-01-01",
                "Status": withheld_status,
                "Type": "decision",
            }
        )
    return blocks


def _make_ws(root: Path, name: str, *, with_withheld: bool, withheld_status: str = "quarantined") -> str:
    ws = str(root / name)
    os.makedirs(ws)
    init(ws)
    _write_decisions(ws, _decisions(with_withheld, withheld_status=withheld_status))
    build_index(ws, incremental=False)
    return ws


def _write_decisions(ws: str, blocks: list[dict]) -> None:
    """Render with the store's own writer so the fixture cannot drift."""
    path = Path(ws) / "decisions" / "DECISIONS.md"
    path.write_text("\n".join(_render_block(b) for b in blocks), encoding="utf-8")


def _conn(ws: str) -> sqlite3.Connection:
    conn = sqlite3.connect(_db_path(ws))
    conn.row_factory = sqlite3.Row
    return conn


def _bm25(ws: str, term: str, table: str = "blocks_fts") -> dict[str, float]:
    """``block_id -> bm25`` for *term*, straight out of SQLite."""
    conn = _conn(ws)
    try:
        rows = conn.execute(
            f"SELECT block_id, bm25({table}) AS score FROM {table} WHERE {table} MATCH ? ORDER BY block_id",  # nosec B608 — `table` is a test-local literal
            (f'"{term}"',),
        ).fetchall()
        return {r["block_id"]: round(r["score"], 8) for r in rows}
    finally:
        conn.close()


def _fts_ids(ws: str, table: str = "blocks_fts") -> set[str]:
    conn = _conn(ws)
    try:
        return {r[0] for r in conn.execute(f"SELECT block_id FROM {table}")}  # nosec B608 — test-local literal
    finally:
        conn.close()


def _scores(ws: str, term: str, **kw: Any) -> list[tuple[str, float]]:
    return [(h["_id"], h["score"]) for h in query_index(ws, term, limit=20, rerank=False, **kw)]


@pytest.fixture()
def clean(tmp_path: Path) -> str:
    return _make_ws(tmp_path, "clean", with_withheld=False)


@pytest.fixture()
def dirty(tmp_path: Path) -> str:
    return _make_ws(tmp_path, "dirty", with_withheld=True)


# ---------------------------------------------------------------------------
# Positive controls. Without these three, every assertion below is vacuous.
# ---------------------------------------------------------------------------


class TestPositiveControls:
    def test_the_fixture_really_holds_the_withheld_block(self, dirty: str) -> None:
        parsed = parse_file(os.path.join(dirty, "decisions", "DECISIONS.md"))
        statements = " ".join(str(b.get("Statement", "")) for b in parsed)
        assert CANARY in statements, "fixture lost the withheld block"
        assert SHARED in statements
        assert len(parsed) == 13, f"expected 12 admitted + 1 withheld, got {len(parsed)}"

    def test_the_gate_really_separates_the_two(self) -> None:
        assert is_admissible_status("active") is True
        assert is_admissible_status("quarantined") is False

    def test_the_withheld_block_really_reached_the_index(self, dirty: str) -> None:
        """The block IS indexed — it is just not in the statistics table.

        This is what makes "not in blocks_fts" a finding rather than an
        accident: the same build put it in ``blocks`` and in the withheld
        shadow, so the method used to look can see it.
        """
        conn = _conn(dirty)
        try:
            row = conn.execute("SELECT status FROM blocks WHERE id = 'D-20260101-099'").fetchone()
        finally:
            conn.close()
        assert row is not None, "the withheld block never reached the index at all"
        assert row["status"] == "quarantined"
        assert "D-20260101-099" in _fts_ids(dirty, "blocks_fts_withheld")


# ---------------------------------------------------------------------------
# The demonstration: the aggregates must not move.
# ---------------------------------------------------------------------------


class TestStatisticsDoNotMove:
    def test_the_withheld_block_is_not_a_document_of_the_admitted_corpus(self, dirty: str) -> None:
        assert "D-20260101-099" not in _fts_ids(dirty, "blocks_fts")

    def test_bm25_of_a_shared_term_does_not_move(self, clean: str, dirty: str) -> None:
        """The term-frequency channel: the withheld block uses this term."""
        before, after = _bm25(clean, SHARED), _bm25(dirty, SHARED)
        assert before, "positive control: the term must match something in the clean corpus"
        assert after == before, f"withheld content moved bm25: {before} -> {after}"

    def test_bm25_of_an_unrelated_term_does_not_move(self, clean: str, dirty: str) -> None:
        """The document-count / length-average channel.

        The withheld block does not contain this term. Before the fix its
        mere presence still moved the score, from -2.8718 to -3.1645.
        """
        before, after = _bm25(clean, LONELY), _bm25(dirty, LONELY)
        assert before, "positive control: the term must match something in the clean corpus"
        assert after == before, f"withheld content moved bm25 for a term it does not contain: {before} -> {after}"

    def test_the_withheld_text_is_still_matchable_where_it_lives(self, dirty: str) -> None:
        """Positive control for the two assertions above.

        They compare bm25 over ``blocks_fts``. If the withheld block had
        simply not been indexed at all, they would pass while proving
        nothing about admission. It IS indexed, and matchable, in the
        withheld shadow.
        """
        assert _bm25(dirty, CANARY, "blocks_fts_withheld"), "the withheld block is not matchable anywhere"
        assert not _bm25(dirty, CANARY, "blocks_fts"), "the withheld canary reached the admitted table"

    def test_query_index_scores_do_not_move(self, clean: str, dirty: str) -> None:
        for term in (SHARED, LONELY):
            before, after = _scores(clean, term), _scores(dirty, term)
            assert before, f"positive control: {term} must retrieve something"
            assert after == before, f"withheld content moved the ranking for {term!r}: {before} -> {after}"

    def test_the_withheld_block_is_not_a_candidate(self, dirty: str) -> None:
        ids = [bid for bid, _ in _scores(dirty, SHARED)]
        assert ids, "positive control: the query must retrieve the admitted blocks"
        assert "D-20260101-099" not in ids


class TestBlockCountDoesNotMove:
    def test_index_status_block_count_does_not_move(self, clean: str, dirty: str) -> None:
        """The count ``index_stats`` and ``memory_health`` serve to user scope."""
        assert index_status(clean)["blocks"] == 12
        assert index_status(dirty)["blocks"] == 12

    def test_include_withheld_is_the_governed_way_to_see_it(self, dirty: str) -> None:
        """Positive control for the count: the method CAN see the block.

        A count that simply lost the block would satisfy the assertion
        above while telling us nothing. Asked explicitly, it reports both
        the full total and the withheld figure.
        """
        st = index_status(dirty, include_withheld=True)
        assert st["blocks"] == 13
        assert st["withheld"] == 1
        assert "withheld" not in index_status(dirty), "the default envelope must carry no figure that moves"

    def test_an_unrecognised_status_is_withheld_from_the_count_too(self, tmp_path: Path) -> None:
        """Fail-closed: the count follows the allow-list, not a deny-list.

        A status nobody has named is withheld by ``is_admissible_status``,
        and the count must agree — otherwise a new ingest door minting an
        unnamed status would put its blocks back into a user-visible
        aggregate.
        """
        ws = _make_ws(tmp_path, "unknown", with_withheld=True, withheld_status="teleported")
        assert index_status(ws)["blocks"] == 12
        assert index_status(ws, include_withheld=True)["withheld"] == 1


class TestGraphBoostDoesNotCarryWithheldNeighbours:
    """A withheld neighbour must not lend its score to an admitted block.

    ``xref_edges`` spans the whole corpus by design, so an unfiltered
    traversal hands a boost across the admission boundary in both
    directions. The half that survives every downstream content filter is
    the one that moves an ADMITTED block's rank.
    """

    @staticmethod
    def _ws(root: Path, name: str, neighbour_status: str | None) -> str:
        ws = str(root / name)
        os.makedirs(ws)
        init(ws)
        blocks = [
            {
                "_id": "D-20260101-001",
                "Statement": f"The {LONELY} decision, which follows D-20260101-099 closely.",
                "Date": "2026-01-01",
                "Status": "active",
                "Type": "decision",
            }
        ]
        if neighbour_status is not None:
            blocks.append(
                {
                    "_id": "D-20260101-099",
                    "Statement": f"{CANARY} neighbour of D-20260101-001 about {LONELY} surveys.",
                    "Date": "2026-01-01",
                    "Status": neighbour_status,
                    "Type": "decision",
                }
            )
        _write_decisions(ws, blocks)
        build_index(ws, incremental=False)
        return ws

    def test_an_admitted_neighbour_does_move_the_score(self, tmp_path: Path) -> None:
        """Positive control: the channel exists and this test can see it."""
        alone = self._ws(tmp_path, "alone", None)
        withneighbour = self._ws(tmp_path, "admitted-neighbour", "active")
        base = dict(_scores(alone, LONELY, graph_boost=True))
        boosted = dict(_scores(withneighbour, LONELY, graph_boost=True))
        assert base and boosted, "positive control: the query must retrieve the seed"
        assert "D-20260101-099" in boosted, "an admitted neighbour must be reachable through the graph"

    def test_a_withheld_neighbour_neither_boosts_nor_appears(self, tmp_path: Path) -> None:
        alone = self._ws(tmp_path, "alone", None)
        quarantined = self._ws(tmp_path, "withheld-neighbour", "quarantined")
        base = dict(_scores(alone, LONELY, graph_boost=True))
        withq = dict(_scores(quarantined, LONELY, graph_boost=True))
        assert base, "positive control: the query must retrieve the seed"
        assert "D-20260101-099" not in withq, "a withheld neighbour was injected into the results"
        assert withq["D-20260101-001"] == base["D-20260101-001"], f"a withheld neighbour moved an admitted block's score: {base} -> {withq}"

    def test_the_edge_to_the_withheld_block_really_exists(self, tmp_path: Path) -> None:
        """Positive control: the traversal had something to refuse.

        Without this, "no boost" could simply mean the fixture never
        produced a cross-reference.
        """
        ws = self._ws(tmp_path, "edge-present", "quarantined")
        conn = _conn(ws)
        try:
            edges = conn.execute("SELECT src, dst FROM xref_edges WHERE src = 'D-20260101-001' AND dst = 'D-20260101-099'").fetchall()
        finally:
            conn.close()
        assert edges, "the fixture produced no cross-reference edge to the withheld block"


class TestFactParentInjectionIsGated:
    """``_aggregate_facts_to_parents`` reads ``blocks``, which holds everything."""

    @staticmethod
    def _seed(tmp_path: Path, parent_status: str) -> str:
        ws = str(tmp_path / f"facts-{parent_status}")
        os.makedirs(ws)
        init(ws)
        _write_decisions(
            ws,
            [
                {
                    "_id": "D-20260101-050",
                    "Statement": f"A parent decision about {LONELY} which is long enough to yield fact cards.",
                    "Date": "2026-01-01",
                    "Status": parent_status,
                    "Type": "decision",
                }
            ],
        )
        build_index(ws, incremental=False)
        return ws

    def _inject(self, ws: str) -> list[dict]:
        conn = _conn(ws)
        try:
            return _aggregate_facts_to_parents(
                conn,
                [{"_id": "D-20260101-050::F1", "score": 5.0, "status": "active"}],
                ws,
            )
        finally:
            conn.close()

    def test_an_admitted_parent_is_injected(self, tmp_path: Path) -> None:
        """Positive control: the injection path works and this test reaches it."""
        out = self._inject(self._seed(tmp_path, "active"))
        assert [r["_id"] for r in out] == ["D-20260101-050"]

    def test_a_withheld_parent_is_not_injected(self, tmp_path: Path) -> None:
        ws = self._seed(tmp_path, "quarantined")
        conn = _conn(ws)
        try:
            present = conn.execute("SELECT 1 FROM blocks WHERE id = 'D-20260101-050'").fetchone()
        finally:
            conn.close()
        assert present, "positive control: the parent row must exist for the refusal to mean anything"
        assert self._inject(ws) == []


class TestReleaseRoundTrip:
    """The withheld set is a door, not a trap: prove it opens AND closes.

    Withholding a block from the statistics table would be a regression if
    it also cost the release path, which must take effect with no reindex
    (the index anchor is attested and a release must not churn it).
    """

    @staticmethod
    def _ws(tmp_path: Path) -> tuple[str, Path, Path]:
        ws = str(tmp_path / "release")
        os.makedirs(ws)
        init(ws)
        decisions = Path(ws) / "decisions" / "DECISIONS.md"
        inbox = Path(ws) / "memory" / "INBOX.md"
        decisions.write_text(
            _render_block(
                {
                    "_id": "D-20260101-001",
                    "Statement": "Baseline admitted decision about pipelines.",
                    "Date": "2026-01-01",
                    "Status": "active",
                    "Type": "decision",
                }
            ),
            encoding="utf-8",
        )
        inbox.write_text(
            _render_block(
                {
                    "_id": "INBOX-20260101-001",
                    "Statement": f"Unreviewed note about {LONELY} sightings.",
                    "Date": "2026-01-01",
                    "Status": "pending",
                    "Type": "inbox",
                }
            ),
            encoding="utf-8",
        )
        build_index(ws, incremental=False)
        return ws, decisions, inbox

    @staticmethod
    def _release(decisions: Path) -> None:
        with decisions.open("a", encoding="utf-8") as fh:
            fh.write(
                "\n"
                + _render_block(
                    {
                        "_id": "D-20260102-001",
                        "Statement": "Release approved.",
                        "Date": "2026-01-02",
                        "Status": "active",
                        "Type": "decision",
                        "Releases": "INBOX-20260101-001",
                    }
                )
            )

    def test_withheld_then_released_then_withheld_again(self, tmp_path: Path) -> None:
        ws, decisions, inbox = self._ws(tmp_path)

        # 1. Withheld: absent from results and from the block count.
        assert "INBOX-20260101-001" not in [bid for bid, _ in _scores(ws, LONELY)]
        assert index_status(ws)["blocks"] == 1
        assert index_status(ws, include_withheld=True)["withheld"] == 1

        # 2. Released, with NO reindex — the capability the shadow table exists to keep.
        self._release(decisions)
        assert "INBOX-20260101-001" in [bid for bid, _ in _scores(ws, LONELY)], "a governance release must take effect without a reindex"

        # 3. The next incremental build moves it into the admitted table.
        build_index(ws, incremental=True)
        assert "INBOX-20260101-001" in _fts_ids(ws, "blocks_fts")
        assert "INBOX-20260101-001" not in _fts_ids(ws, "blocks_fts_withheld")
        assert index_status(ws)["blocks"] == 3

        # 4. And back: quarantining it returns it to the withheld set. A state
        #    machine with no reporter for the return trip is a one-way door.
        inbox.write_text(inbox.read_text(encoding="utf-8").replace("Status: pending", "Status: quarantined"), encoding="utf-8")
        decisions.write_text(decisions.read_text(encoding="utf-8").replace("Releases: INBOX-20260101-001\n", ""), encoding="utf-8")
        build_index(ws, incremental=True)
        assert "INBOX-20260101-001" not in _fts_ids(ws, "blocks_fts")
        assert "INBOX-20260101-001" in _fts_ids(ws, "blocks_fts_withheld")
        assert "INBOX-20260101-001" not in [bid for bid, _ in _scores(ws, LONELY)]


class TestFullRebuildDoesNotDuplicateDocuments:
    """A second way the same statistics get distorted, from the other side.

    ``blocks_fts`` has no primary key, and a full rebuild cleared only
    ``file_state`` and ``index_meta`` — so every block was "new" again
    while its old row survived. Three full rebuilds of a one-block corpus
    left ``blocks``=1 and ``blocks_fts``=3 (measured). A duplicated
    document inflates the bm25 document count and the length average
    exactly as a withheld one did, and doubles the block in the results.
    """

    def test_repeated_full_rebuilds_leave_one_row_per_block(self, tmp_path: Path) -> None:
        ws = _make_ws(tmp_path, "rebuilt", with_withheld=True)
        conn = _conn(ws)
        try:

            def counts() -> tuple[int, int, int]:
                return (
                    conn.execute("SELECT COUNT(*) FROM blocks").fetchone()[0],
                    conn.execute("SELECT COUNT(*) FROM blocks_fts").fetchone()[0],
                    conn.execute("SELECT COUNT(*) FROM blocks_fts_withheld").fetchone()[0],
                )

            first = counts()
            assert first == (13, 12, 1), f"positive control: one build must index every block exactly once, got {first}"
            for _ in range(2):
                build_index(ws, incremental=False)
            assert counts() == first, "a full rebuild duplicated index rows"
        finally:
            conn.close()

    def test_scores_survive_a_rebuild_unchanged(self, tmp_path: Path) -> None:
        """Determinism: the same corpus must score the same after a rebuild."""
        ws = _make_ws(tmp_path, "stable", with_withheld=True)
        before = _scores(ws, SHARED)
        build_index(ws, incremental=False)
        assert before, "positive control: the query must retrieve something"
        assert _scores(ws, SHARED) == before


class TestAttestedAnchorStillCoversEverything:
    """The Merkle anchor is deliberately NOT narrowed to the admitted set.

    It is an integrity attestation over what the index stores, and
    ``mind-mem-verify`` compares its root against a recorded anchor.
    Dropping withheld rows from it would both weaken the attestation and
    invalidate every anchor recorded before 5.0.2. Pinned here so the
    scope of this change stays a decision rather than a drift.
    """

    def test_merkle_leaves_cover_the_withheld_block(self, dirty: str) -> None:
        assert "D-20260101-099" in {bid for bid, _ in merkle_leaves(dirty)}


# ---------------------------------------------------------------------------
# The sweep. One fixed function per aggregate was how the block count got
# fixed and the build summary did not — a tripwire that enumerates the one
# reading we already knew about cannot find the next one. So this walks the
# module's whole public surface, and a new public callable fails the suite
# until somebody says which side of the gate its numbers are on.
# ---------------------------------------------------------------------------


#: Every public callable ``mind_mem.sqlite_index`` exports, and the claim
#: this file makes about the numbers it returns.
#:
#: ``egress``
#:     Served, or derivable by a caller the gate refuses the content to.
#:     Must be IDENTICAL between a corpus with a withheld block and one
#:     without. This is the side-channel invariant.
#: ``indexer-truth``
#:     The builder reporting the work it did, to an operator. Allowed —
#:     required — to count the whole index, because a builder that hid
#:     rows from its own report would be lying about what it indexed.
#:     ``build_index`` also returns the egress-safe ``blocks_admitted``,
#:     which IS checked as egress below.
#: ``attestation``
#:     The Merkle anchor. Covers everything on purpose (see
#:     ``TestAttestedAnchorStillCoversEverything``); narrowing it would
#:     void every anchor recorded before 5.0.2.
#: ``cli``
#:     ``argparse`` entry point; returns no aggregate of its own.
_PUBLIC_SURFACE: dict[str, str] = {
    "build_index": "indexer-truth",
    # Returns ``build_index``'s summary verbatim, or ``None``. Same numbers,
    # same operator audience, same side of the gate — and it serves no
    # content of its own, so there is nothing here for a refused caller to
    # derive that ``build_index`` did not already report.
    "ensure_index": "indexer-truth",
    "query_index": "egress",
    "is_stale": "egress",
    "merkle_leaves": "attestation",
    "index_status": "egress",
    "main": "cli",
}

#: Readings that must not move, keyed for the failure message.
_MUST_NOT_MOVE = (
    "index_status.blocks",
    "index_status.stale_files",
    "is_stale",
    "query_index.scores.shared",
    "query_index.scores.lonely",
    "query_index.hit_count",
    "build.blocks_admitted",
)

#: Readings that MUST move. These are the positive control for the whole
#: sweep: they prove the snapshot-and-compare method can see a difference
#: between the two workspaces at all. Without them, seven "did not move"
#: assertions would pass just as happily against a harness that was
#: comparing a workspace with itself.
_MUST_MOVE = (
    "build.total_blocks",
    "build.blocks_withheld",
    "merkle.leaf_count",
)


def _aggregates(ws: str) -> dict[str, Any]:
    """Every number the public surface of the index module will hand out."""
    summary = build_index(ws, incremental=False)
    return {
        "build.total_blocks": summary["total_blocks"],
        "build.blocks_admitted": summary["blocks_admitted"],
        "build.blocks_withheld": summary["blocks_withheld"],
        "index_status.blocks": index_status(ws)["blocks"],
        "index_status.stale_files": index_status(ws)["stale_files"],
        "is_stale": is_stale(ws),
        "query_index.scores.shared": _scores(ws, SHARED),
        "query_index.scores.lonely": _scores(ws, LONELY),
        "query_index.hit_count": len(query_index(ws, SHARED, limit=20, rerank=False)),
        "merkle.leaf_count": len(merkle_leaves(ws)),
    }


class TestTheAggregateSweep:
    """Clean vs. one-quarantined-block, across the whole public surface."""

    def test_every_public_callable_is_classified(self) -> None:
        """A new public aggregate fails here until its side is declared.

        The read-surface tripwire that missed ``get_block`` enumerated one
        module instead of the registry. Same shape, same fix: enumerate
        what the module actually exports, not what we remembered writing.
        """
        exported = {
            name
            for name, obj in inspect.getmembers(sqlite_index, inspect.isfunction)
            if not name.startswith("_") and obj.__module__ == sqlite_index.__name__
        }
        assert exported, "positive control: the module must export something to classify"
        assert exported == set(_PUBLIC_SURFACE), (
            f"unclassified public callables: {sorted(exported - set(_PUBLIC_SURFACE))}; "
            f"classified but gone: {sorted(set(_PUBLIC_SURFACE) - exported)}"
        )

    def test_the_two_workspaces_really_differ(self, clean: str, dirty: str) -> None:
        """Positive control: the fixture pair is not the same corpus twice."""
        assert _fts_ids(clean, "blocks_fts_withheld") == set()
        assert _fts_ids(dirty, "blocks_fts_withheld") == {"D-20260101-099"}

    def test_the_sweep_can_detect_a_difference(self, clean: str, dirty: str) -> None:
        """Positive control for the METHOD, not for the code under test.

        Every "did not move" below is only evidence if this passes: it
        shows the same snapshot-and-compare finds a real difference where
        one is supposed to exist.
        """
        before, after = _aggregates(clean), _aggregates(dirty)
        moved = [k for k in _MUST_MOVE if before[k] != after[k]]
        assert moved == list(_MUST_MOVE), (
            f"the sweep saw no difference in {sorted(set(_MUST_MOVE) - set(moved))} — "
            "every invariance assertion in this class is vacuous until it does"
        )

    def test_no_egress_aggregate_moves(self, clean: str, dirty: str) -> None:
        before, after = _aggregates(clean), _aggregates(dirty)
        drift = {k: (before[k], after[k]) for k in _MUST_NOT_MOVE if before[k] != after[k]}
        assert drift == {}, f"withheld content is inferable from: {drift}"

    def test_the_build_summary_decomposes_exactly(self, clean: str, dirty: str) -> None:
        """``total`` is admitted + withheld, and the withheld count is real."""
        for ws, expected in ((clean, 0), (dirty, 1)):
            agg = _aggregates(ws)
            assert agg["build.blocks_withheld"] == expected
            assert agg["build.total_blocks"] == agg["build.blocks_admitted"] + agg["build.blocks_withheld"]

    def test_the_admitted_count_agrees_with_the_served_count(self, clean: str, dirty: str) -> None:
        """One counting authority: the builder and the status agree.

        They disagreed before 5.0.2 — ``index_status`` ran the admission
        allow-list and the build summary ran a bare ``COUNT(*)`` — which
        is exactly the drift a second counter always produces.
        """
        for ws in (clean, dirty):
            agg = _aggregates(ws)
            assert agg["build.blocks_admitted"] == agg["index_status.blocks"]


class TestMutationTwin:
    """Disable the gate; every invariance assertion above must go red.

    A test that cannot fail is not a test. These break the ONE thing that
    holds the invariant — the admission verdict that decides which FTS
    table a block is written to, and which rows the counters admit — and
    assert the leak comes back. If a future refactor makes the withheld
    block harmless by accident (say, by never indexing it at all), these
    fail and say the suite has stopped proving anything.
    """

    @staticmethod
    def _admit_everything(monkeypatch: pytest.MonkeyPatch) -> None:
        """The gate, disabled: every block is admissible, releases ignored."""
        monkeypatch.setattr(
            sqlite_index,
            "_admit_ids",
            lambda pairs, **_kw: {bid for bid, _status in pairs},
        )
        monkeypatch.setattr(sqlite_index, "is_admissible_status", lambda _status: True)

    def test_the_withheld_block_rejoins_the_admitted_corpus(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self._admit_everything(monkeypatch)
        ws = _make_ws(tmp_path, "mutant", with_withheld=True)
        assert "D-20260101-099" in _fts_ids(ws), "the mutation did not disable the gate"

    def test_the_bm25_channel_reopens(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        clean_ws = _make_ws(tmp_path, "control", with_withheld=False)
        control = _bm25(clean_ws, LONELY)
        self._admit_everything(monkeypatch)
        mutant = _make_ws(tmp_path, "mutant", with_withheld=True)
        assert _bm25(mutant, LONELY) != control, (
            "with the gate disabled the unrelated term's score must move again — "
            "if it does not, the bm25 assertions in this file are proving nothing"
        )

    def test_the_egress_sweep_goes_red(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        clean_ws = _make_ws(tmp_path, "control", with_withheld=False)
        before = _aggregates(clean_ws)
        self._admit_everything(monkeypatch)
        mutant = _make_ws(tmp_path, "mutant", with_withheld=True)
        after = _aggregates(mutant)
        moved = [k for k in _MUST_NOT_MOVE if before[k] != after[k]]
        assert moved, "the ungated build leaked nothing — the sweep cannot be measuring the gate"
        assert "index_status.blocks" in moved
        assert "build.blocks_admitted" in moved
