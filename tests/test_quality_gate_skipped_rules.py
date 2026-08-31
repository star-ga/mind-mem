"""A rule that did not run must not look like a rule that passed.

Rule 6, ``near_duplicate``, is the only one of the eight documented rules that
needs input beyond the candidate text — a ``recent`` window of ``(text,
timestamp)`` pairs to be a duplicate *of*. ``recent`` is keyword-only and
defaults to ``None``, and neither product call site supplies it, so the rule
never executes in the product.

That is a wiring gap, not a bug in itself. The bug is that both ``_pass`` and
``_fail`` live inside ``if recent:``, so on the default path rule 6 simply
vanished: ``checked_rules`` listed seven entries and nothing anywhere said the
eighth had been skipped. A verdict that reports seven rules as if it had run
eight overstates what was checked.
"""

from __future__ import annotations

import datetime as dt

from mind_mem.quality_gate import validate_block

GOOD = "A perfectly ordinary decision statement with enough content to clear the length rule."


class TestNearDuplicateIsReportedWhenItCannotRun:
    def test_no_recent_window_marks_the_rule_skipped(self) -> None:
        """Before the fix the rule left no trace at all in the verdict."""
        verdict = validate_block(GOOD)
        assert verdict.accept is True
        assert "near_duplicate" in verdict.skipped_rules
        assert "near_duplicate" not in verdict.checked_rules

    def test_every_documented_rule_is_accounted_for(self) -> None:
        """checked + skipped covers the whole rule set, with no overlap."""
        verdict = validate_block(GOOD)
        accounted = set(verdict.checked_rules) | set(verdict.skipped_rules)
        assert not set(verdict.checked_rules) & set(verdict.skipped_rules)
        assert accounted == {
            "empty",
            "too_short",
            "oversize",
            "malformed_utf8",
            "stopwords_only",
            "near_duplicate",
            "injection_marker",
        }

    def test_skip_is_visible_in_the_serialized_verdict(self) -> None:
        """The MCP preview tool returns to_dict() verbatim."""
        payload = validate_block(GOOD).to_dict()
        assert payload["skipped_rules"] == ["near_duplicate"]

    def test_a_supplied_window_runs_the_rule_and_skips_nothing(self) -> None:
        recent = [(GOOD, dt.datetime.now(dt.timezone.utc))]
        verdict = validate_block(GOOD, recent=recent, strict=True)
        assert verdict.skipped_rules == []
        assert "near_duplicate" in verdict.checked_rules
        assert verdict.accept is False
        assert any("near_duplicate" in r for r in verdict.reasons)

    def test_a_supplied_window_with_no_duplicate_still_checks_the_rule(self) -> None:
        recent = [("something else entirely, unrelated text", dt.datetime.now(dt.timezone.utc))]
        verdict = validate_block(GOOD, recent=recent, strict=True)
        assert verdict.skipped_rules == []
        assert "near_duplicate" in verdict.checked_rules
        assert verdict.accept is True

    def test_forced_verdict_still_reports_the_skip(self) -> None:
        verdict = validate_block(GOOD, force=True)
        assert verdict.forced is True
        assert verdict.skipped_rules == ["near_duplicate"]
