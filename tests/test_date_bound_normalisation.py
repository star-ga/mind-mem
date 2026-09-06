"""Date bounds must mean the same thing whatever a Date field looks like.

Both the Python recall funnel and the SQLite push-down compare dates
LEXICOGRAPHICALLY, which is only valid on a canonical form. "/" is 0x2F and
"-" is 0x2D, so a slash-stamped day sorts as though it were the last instant of
its year. Measured on the shipped path for a block really dated 2026-05-20 and
written ``2026/05/20 (Wed) 02:21``: an ``until=2026-12-31`` bound DROPPED it
and a ``since=2026-08-01`` bound SERVED it -- silent loss in one direction and
silent over-serving in the other, on the same block.

Only bounds inside the block's own calendar year were affected, because the
year digits compare first. That is why it survived casual testing.

There are TWO independent sites and each drops the row on its own, so both are
covered here: repairing one alone leaves date-bounded recall wrong.

A previous fix for this defect edited the same-named ``_block_date`` in
``memory_index``, which only feeds markdown index generation. These tests
import from the modules actually on the recall path, so that mistake cannot
repeat silently.
"""

import os
import sqlite3
import tempfile

import pytest

from mind_mem._recall_core import _block_date, _in_date_range
from mind_mem.block_parser import canonical_day

SLASH = "2026/05/20 (Wed) 02:21"
ISO = "2026-05-20"


class TestCanonicalDay:
    def test_both_separators_reach_the_same_day(self) -> None:
        assert canonical_day(SLASH) == canonical_day(ISO) == "2026-05-20"

    def test_unparseable_stays_undated_rather_than_sorting_somewhere(self) -> None:
        for bad in ("no date here", "", None, 12345, [], {}):
            assert canonical_day(bad) == ""

    def test_it_normalises_the_separator_not_just_matches_it(self) -> None:
        """Matching '/' without rewriting it inverts bounds instead of fixing them."""
        assert "/" not in canonical_day(SLASH)


class TestPythonFunnelBounds:
    """``_recall_core._block_date`` -> ``_in_date_range`` (recall filter path)."""

    @pytest.mark.parametrize(
        "raw",
        [SLASH, ISO],
        ids=["slash", "iso"],
    )
    def test_in_range_block_is_kept(self, raw: str) -> None:
        assert _in_date_range(_block_date({"Date": raw}), "2026-01-01", "2026-12-31") is True

    @pytest.mark.parametrize("raw", [SLASH, ISO], ids=["slash", "iso"])
    def test_block_before_since_is_excluded(self, raw: str) -> None:
        assert _in_date_range(_block_date({"Date": raw}), "2026-08-01", None) is False

    @pytest.mark.parametrize("raw", [SLASH, ISO], ids=["slash", "iso"])
    def test_block_after_until_is_excluded(self, raw: str) -> None:
        assert _in_date_range(_block_date({"Date": raw}), None, "2026-01-31") is False

    def test_the_two_spellings_agree_on_every_bound(self) -> None:
        bounds = [
            ("2026-01-01", "2026-12-31"),
            ("2026-08-01", None),
            (None, "2026-01-31"),
            ("2027-01-01", None),
            (None, "2025-12-31"),
            (None, None),
        ]
        for since, until in bounds:
            a = _in_date_range(_block_date({"Date": SLASH}), since, until)
            b = _in_date_range(_block_date({"Date": ISO}), since, until)
            assert a == b, f"spellings disagree on since={since} until={until}: {a} vs {b}"

    def test_lowercase_date_key_is_normalised_too(self) -> None:
        assert _block_date({"date": SLASH}) == "2026-05-20"


class TestSqlPushDownBounds:
    """The stored ``blocks.date`` column, compared in SQL under BINARY collation."""

    @staticmethod
    def _build(tmp: str) -> str:
        d = os.path.join(tmp, "decisions")
        os.makedirs(d, exist_ok=True)
        rows = [
            f"[D-20260520-000001]\nStatement: the slash stamped entry\nDate: {SLASH}\nStatus: active\n",
            f"[D-20260520-000002]\nStatement: the iso stamped entry\nDate: {ISO}\nStatus: active\n",
        ]
        with open(os.path.join(d, "DECISIONS.md"), "w", encoding="utf-8") as fh:
            fh.write("\n---\n\n".join(rows))
        with open(os.path.join(tmp, "mind-mem.json"), "w", encoding="utf-8") as fh:
            fh.write("{}")
        from mind_mem.sqlite_index import build_index

        build_index(tmp, incremental=False)
        return os.path.join(tmp, ".mind-mem-index", "recall.db")

    def test_stored_dates_are_canonical(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = self._build(tmp)
            con = sqlite3.connect(db)
            try:
                stored = dict(con.execute("SELECT id, date FROM blocks WHERE parent_id = ''").fetchall())
            finally:
                con.close()
            assert stored, "no blocks indexed; the test proves nothing"
            for bid, value in stored.items():
                assert "/" not in value, f"{bid} stored a slash date: {value!r}"

    def test_a_bounded_sql_query_returns_both_spellings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = self._build(tmp)
            con = sqlite3.connect(db)
            try:
                got = [
                    r[0]
                    for r in con.execute(
                        "SELECT id FROM blocks WHERE parent_id = '' AND date <> '' AND date >= ? AND date <= ? ORDER BY id",
                        ("2026-01-01", "2026-12-31"),
                    ).fetchall()
                ]
            finally:
                con.close()
            assert len(got) == 2, f"an in-range block was dropped by the SQL bound: {got}"

    def test_an_out_of_range_bound_excludes_both_spellings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = self._build(tmp)
            con = sqlite3.connect(db)
            try:
                got = [
                    r[0]
                    for r in con.execute(
                        "SELECT id FROM blocks WHERE parent_id = '' AND date <> '' AND date >= ?",
                        ("2026-08-01",),
                    ).fetchall()
                ]
            finally:
                con.close()
            assert got == [], f"an out-of-range block was served by the SQL bound: {got}"
