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
            "a parseable date must not score the same as no date at all -- "
            "that equality is what made this failure invisible"
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
