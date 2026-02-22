#!/usr/bin/env python3
"""Tests for _recall_temporal.py — time-aware hard filters for temporal queries."""

import os
import sys
import unittest
from datetime import date

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from _recall_temporal import apply_temporal_filter, resolve_time_reference


class TestResolveTimeReference(unittest.TestCase):
    """Test relative time expression parsing."""

    def setUp(self):
        # Fixed reference date: Wednesday, 2026-02-18
        self.ref = date(2026, 2, 18)

    # -- yesterday / today ------------------------------------------------

    def test_yesterday(self):
        start, end = resolve_time_reference("What happened yesterday?", self.ref)
        self.assertEqual(start, date(2026, 2, 17))
        self.assertEqual(end, date(2026, 2, 17))

    def test_today(self):
        start, end = resolve_time_reference("What is happening today?", self.ref)
        self.assertEqual(start, date(2026, 2, 18))
        self.assertEqual(end, date(2026, 2, 18))

    # -- last week --------------------------------------------------------

    def test_last_week(self):
        start, end = resolve_time_reference("What did she do last week?", self.ref)
        # ref is Wed 2026-02-18 → this Monday = 2026-02-16
        # last Monday = 2026-02-09, last Sunday = 2026-02-15
        self.assertEqual(start, date(2026, 2, 9))
        self.assertEqual(end, date(2026, 2, 15))

    # -- this week --------------------------------------------------------

    def test_this_week(self):
        start, end = resolve_time_reference("What happened this week?", self.ref)
        # ref is Wed 2026-02-18 → this Monday = 2026-02-16
        self.assertEqual(start, date(2026, 2, 16))
        self.assertEqual(end, date(2026, 2, 18))

    # -- last month -------------------------------------------------------

    def test_last_month(self):
        start, end = resolve_time_reference("What happened last month?", self.ref)
        self.assertEqual(start, date(2026, 1, 1))
        self.assertEqual(end, date(2026, 1, 31))

    # -- this month -------------------------------------------------------

    def test_this_month(self):
        start, end = resolve_time_reference("Events this month", self.ref)
        self.assertEqual(start, date(2026, 2, 1))
        self.assertEqual(end, date(2026, 2, 18))

    # -- last year --------------------------------------------------------

    def test_last_year(self):
        start, end = resolve_time_reference("What happened last year?", self.ref)
        self.assertEqual(start, date(2025, 1, 1))
        self.assertEqual(end, date(2025, 12, 31))

    # -- this year --------------------------------------------------------

    def test_this_year(self):
        start, end = resolve_time_reference("Events this year", self.ref)
        self.assertEqual(start, date(2026, 1, 1))
        self.assertEqual(end, date(2026, 2, 18))

    # -- last N days ------------------------------------------------------

    def test_last_7_days(self):
        start, end = resolve_time_reference("Show me the last 7 days", self.ref)
        self.assertEqual(start, date(2026, 2, 11))
        self.assertEqual(end, date(2026, 2, 18))

    def test_last_30_days(self):
        start, end = resolve_time_reference("Activity in the last 30 days", self.ref)
        self.assertEqual(start, date(2026, 1, 19))
        self.assertEqual(end, date(2026, 2, 18))

    # -- N days ago -------------------------------------------------------

    def test_3_days_ago(self):
        start, end = resolve_time_reference("What happened 3 days ago?", self.ref)
        self.assertEqual(start, date(2026, 2, 15))
        self.assertEqual(end, date(2026, 2, 15))

    # -- N weeks ago ------------------------------------------------------

    def test_2_weeks_ago(self):
        start, end = resolve_time_reference("Events from 2 weeks ago", self.ref)
        # target ~2026-02-04, window +-3 days
        self.assertEqual(start, date(2026, 2, 1))
        self.assertEqual(end, date(2026, 2, 7))

    # -- N months ago -----------------------------------------------------

    def test_3_months_ago(self):
        start, end = resolve_time_reference("What happened 3 months ago?", self.ref)
        # ~90 days back from 2026-02-18 = ~2025-11-20 → November range
        self.assertEqual(start, date(2025, 11, 1))
        self.assertEqual(end, date(2025, 11, 30))

    # -- in Month / in Month Year -----------------------------------------

    def test_in_january(self):
        start, end = resolve_time_reference("What happened in January?", self.ref)
        self.assertEqual(start, date(2026, 1, 1))
        self.assertEqual(end, date(2026, 1, 31))

    def test_in_january_2025(self):
        start, end = resolve_time_reference("What happened in January 2025?", self.ref)
        self.assertEqual(start, date(2025, 1, 1))
        self.assertEqual(end, date(2025, 1, 31))

    def test_in_february(self):
        start, end = resolve_time_reference("Events in February", self.ref)
        self.assertEqual(start, date(2026, 2, 1))
        self.assertEqual(end, date(2026, 2, 28))

    def test_in_abbreviated_month(self):
        start, end = resolve_time_reference("What happened in Sep 2025?", self.ref)
        self.assertEqual(start, date(2025, 9, 1))
        self.assertEqual(end, date(2025, 9, 30))

    # -- in YYYY ----------------------------------------------------------

    def test_in_2025(self):
        start, end = resolve_time_reference("Events in 2025", self.ref)
        self.assertEqual(start, date(2025, 1, 1))
        self.assertEqual(end, date(2025, 12, 31))

    # -- before/after ISO date --------------------------------------------

    def test_before_iso_date(self):
        start, end = resolve_time_reference("Events before 2025-06-15", self.ref)
        self.assertIsNone(start)
        self.assertEqual(end, date(2025, 6, 15))

    def test_after_iso_date(self):
        start, end = resolve_time_reference("Events after 2025-03-01", self.ref)
        self.assertEqual(start, date(2025, 3, 1))
        self.assertIsNone(end)

    # -- before/after Month DD, YYYY --------------------------------------

    def test_before_month_date(self):
        start, end = resolve_time_reference("Events before January 15, 2025", self.ref)
        self.assertIsNone(start)
        self.assertEqual(end, date(2025, 1, 15))

    def test_after_month_date(self):
        start, end = resolve_time_reference("Events after March 1, 2025", self.ref)
        self.assertEqual(start, date(2025, 3, 1))
        self.assertIsNone(end)

    # -- no match ---------------------------------------------------------

    def test_no_temporal_reference(self):
        start, end = resolve_time_reference("What is Caroline's favorite color?", self.ref)
        self.assertIsNone(start)
        self.assertIsNone(end)

    def test_empty_query(self):
        start, end = resolve_time_reference("", self.ref)
        self.assertIsNone(start)
        self.assertIsNone(end)

    # -- case insensitive -------------------------------------------------

    def test_case_insensitive_yesterday(self):
        start, end = resolve_time_reference("YESTERDAY events", self.ref)
        self.assertEqual(start, date(2026, 2, 17))

    def test_case_insensitive_month(self):
        start, end = resolve_time_reference("in MARCH 2025", self.ref)
        self.assertEqual(start, date(2025, 3, 1))


class TestApplyTemporalFilter(unittest.TestCase):
    """Test block filtering by date range."""

    def _make_block(self, block_id, date_str=None, score=1.0):
        block = {"_id": block_id, "score": score}
        if date_str:
            block["Date"] = date_str
        return block

    def test_filter_within_range(self):
        blocks = [
            self._make_block("B1", "2025-01-10"),
            self._make_block("B2", "2025-01-20"),
            self._make_block("B3", "2025-02-05"),
        ]
        result = apply_temporal_filter(blocks, date(2025, 1, 1), date(2025, 1, 31))
        ids = [b["_id"] for b in result]
        self.assertEqual(ids, ["B1", "B2"])

    def test_blocks_without_date_pass_through(self):
        blocks = [
            self._make_block("B1", "2025-01-10"),
            self._make_block("B2"),  # no date
            self._make_block("B3", "2025-03-01"),
        ]
        result = apply_temporal_filter(blocks, date(2025, 1, 1), date(2025, 1, 31))
        ids = [b["_id"] for b in result]
        self.assertIn("B2", ids)  # undated passes through
        self.assertIn("B1", ids)
        self.assertNotIn("B3", ids)

    def test_only_start_date(self):
        blocks = [
            self._make_block("B1", "2025-01-10"),
            self._make_block("B2", "2025-03-20"),
        ]
        result = apply_temporal_filter(blocks, date(2025, 2, 1), None)
        ids = [b["_id"] for b in result]
        self.assertEqual(ids, ["B2"])

    def test_only_end_date(self):
        blocks = [
            self._make_block("B1", "2025-01-10"),
            self._make_block("B2", "2025-03-20"),
        ]
        result = apply_temporal_filter(blocks, None, date(2025, 2, 1))
        ids = [b["_id"] for b in result]
        self.assertEqual(ids, ["B1"])

    def test_no_range_returns_all(self):
        blocks = [
            self._make_block("B1", "2025-01-10"),
            self._make_block("B2"),
        ]
        result = apply_temporal_filter(blocks, None, None)
        self.assertEqual(len(result), 2)

    def test_empty_blocks_list(self):
        result = apply_temporal_filter([], date(2025, 1, 1), date(2025, 12, 31))
        self.assertEqual(result, [])

    def test_unparseable_date_passes_through(self):
        blocks = [
            self._make_block("B1", "not-a-date"),
            self._make_block("B2", "2025-01-15"),
        ]
        result = apply_temporal_filter(blocks, date(2025, 1, 1), date(2025, 1, 31))
        ids = [b["_id"] for b in result]
        self.assertIn("B1", ids)  # unparseable passes through
        self.assertIn("B2", ids)

    def test_boundary_inclusive(self):
        blocks = [
            self._make_block("B1", "2025-01-01"),
            self._make_block("B2", "2025-01-31"),
            self._make_block("B3", "2025-02-01"),
        ]
        result = apply_temporal_filter(blocks, date(2025, 1, 1), date(2025, 1, 31))
        ids = [b["_id"] for b in result]
        self.assertIn("B1", ids)
        self.assertIn("B2", ids)
        self.assertNotIn("B3", ids)

    def test_date_with_time_suffix_handled(self):
        """Date fields may have time appended: '2025-01-15T10:30:00'."""
        blocks = [
            self._make_block("B1", "2025-01-15T10:30:00"),
        ]
        result = apply_temporal_filter(blocks, date(2025, 1, 1), date(2025, 1, 31))
        self.assertEqual(len(result), 1)


class TestDefaultReferenceDate(unittest.TestCase):
    """Test that resolve_time_reference uses today when no reference_date given."""

    def test_yesterday_defaults_to_today(self):
        start, end = resolve_time_reference("yesterday")
        self.assertIsNotNone(start)
        self.assertIsNotNone(end)
        self.assertEqual(start, end)


if __name__ == "__main__":
    unittest.main()
