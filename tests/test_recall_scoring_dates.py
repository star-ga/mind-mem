"""A slash-dated corpus must not silently lose its temporal signal.

An unparseable date scores 0.5 -- the identical score for having no date at
all -- so a parser that rejects a real-world format fails invisibly: every
layer above sees a plausible number and nothing logs. LongMemEval-S ships
``2023/05/20 (Sat) 02:21`` on every session, which is 26.6% of that benchmark
in temporal-reasoning questions alone.
"""

from datetime import datetime, timezone

from mind_mem._recall_scoring import _parse_utc_day, date_score

NOW = datetime(2023, 6, 1, tzinfo=timezone.utc)
DAY = datetime(2023, 5, 20, tzinfo=timezone.utc)


class TestSlashDatesParse:
    def test_the_real_longmemeval_stamp_parses(self) -> None:
        assert _parse_utc_day("2023/05/20 (Sat) 02:21") == DAY

    def test_slash_and_dash_are_the_same_day(self) -> None:
        assert _parse_utc_day("2023/05/20") == _parse_utc_day("2023-05-20")

    def test_it_scores_as_a_date_not_as_a_missing_one(self) -> None:
        scored = date_score({"Date": "2023/05/20 (Sat) 02:21"}, now=NOW)
        assert scored != date_score({}, now=NOW), (
            "a parseable date must not score the same as no date at all -- that equality is what made this failure invisible"
        )
        assert scored == date_score({"Date": "2023-05-20"}, now=NOW)


class TestNoRegression:
    def test_iso_is_unchanged(self) -> None:
        assert _parse_utc_day("2023-05-20") == DAY
        assert date_score({"Date": "2023-05-20"}, now=NOW) > 0.9

    def test_unparseable_still_falls_back(self) -> None:
        for bad in ("garbage", "", "20/05/2023", "not a date"):
            assert _parse_utc_day(bad) is None
            assert date_score({"Date": bad}, now=NOW) == 0.5

    def test_non_string_is_rejected_not_raised(self) -> None:
        for bad in (None, 12345, [], {}):
            assert _parse_utc_day(bad) is None


class TestDateProximityReadsSlashDates:
    """The penalty branch made this failure worse than neutral.

    ``_date_proximity_score`` answers an empty block-date list with 0.8 -- a
    penalty for "no date when the query has one". A block stamped
    ``2023/05/20`` extracted to [], so the block whose date EXACTLY matched the
    query scored 0.8 while an ISO-stamped twin scored 1.5: a 1.875x swing
    against the correct answer, not a miss.
    """

    QUERY = "What did I say on 2023-05-20?"
    SLASH = "Statement: I moved house.\nDate: 2023/05/20 (Sat) 02:21\n"
    ISO = "Statement: I moved house.\nDate: 2023-05-20\n"

    def test_slash_dates_are_extracted(self) -> None:
        from mind_mem._recall_scoring import _extract_dates

        assert _extract_dates(self.SLASH) == _extract_dates(self.ISO) != []

    def test_matching_slash_date_boosts_like_iso(self) -> None:
        from mind_mem._recall_scoring import _date_proximity_score

        assert _date_proximity_score(self.QUERY, self.SLASH) == _date_proximity_score(self.QUERY, self.ISO)

    def test_a_distant_date_still_does_not_boost(self) -> None:
        from mind_mem._recall_scoring import _date_proximity_score

        far = self.SLASH.replace("2023/05/20", "2023/01/02")
        assert _date_proximity_score(self.QUERY, far) < 1.0

    def test_a_genuinely_undated_block_keeps_its_penalty(self) -> None:
        from mind_mem._recall_scoring import _date_proximity_score

        assert _date_proximity_score(self.QUERY, "Statement: no date here") == 0.8


class TestDateRangeFilterKeepsDatedBlocks:
    """The severe one: a hard filter, not a ranking nudge.

    ``_recall_core`` filters recall hits with
    ``_in_date_range(_block_date(h), since, until)``, and ``_in_date_range``
    rejects an empty date because "block has no date -> cannot satisfy a
    date-bound query" -- correct logic fed by a reader that could not see the
    date. A slash-dated block was therefore DROPPED from every date-bounded
    recall, not merely ranked lower.
    """

    SINCE, UNTIL = "2023-05-01", "2023-06-01"

    def _kept(self, date_value: str) -> bool:
        from mind_mem._recall_core import _in_date_range
        from mind_mem.memory_index import _block_date

        return _in_date_range(_block_date({"Date": date_value, "_id": "S-1"}), self.SINCE, self.UNTIL)

    def test_a_slash_dated_block_is_not_dropped(self) -> None:
        assert self._kept("2023/05/20 (Sat) 02:21") is True

    def test_slash_and_iso_agree(self) -> None:
        assert self._kept("2023/05/20") == self._kept("2023-05-20") is True

    def test_normalised_to_iso_because_comparison_is_lexicographic(self) -> None:
        from mind_mem.memory_index import _block_date

        got = _block_date({"Date": "2023/05/20 (Sat) 02:21", "_id": "S-1"})
        assert got == "2023-05-20", f"a surviving '/' inverts every bound: {got!r}"

    def test_out_of_range_is_still_excluded(self) -> None:
        assert self._kept("2023/01/02") is False

    def test_genuinely_undated_is_still_excluded(self) -> None:
        assert self._kept("") is False

    def test_block_kind_is_unaffected(self) -> None:
        from mind_mem.memory_index import _block_kind

        assert _block_kind({"_id": "D-20230520-001"}) == "D"
        assert _block_kind({"_id": "nope"}) == "OTHER"
