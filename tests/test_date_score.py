"""Tests for date scoring function."""

from __future__ import annotations

from mind_mem._recall_scoring import date_score


def test_date_score_no_date():
    assert date_score({}) == 0.5


def test_date_score_empty_date():
    assert date_score({"Date": ""}) == 0.5


def test_date_score_valid():
    result = date_score({"Date": "2026-01-15"})
    assert isinstance(result, float)


def test_date_score_recent_higher():
    recent = date_score({"Date": "2026-02-20"})
    older = date_score({"Date": "2025-01-01"})
    assert recent >= older


def test_date_score_invalid_format():
    result = date_score({"Date": "not-a-date"})
    assert isinstance(result, float)
    assert result == 0.5
