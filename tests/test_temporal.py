"""Tests for temporal filtering module."""

from __future__ import annotations

from mind_mem._recall_temporal import resolve_time_reference


def test_resolve_today():
    result = resolve_time_reference("today")
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_resolve_yesterday():
    result = resolve_time_reference("yesterday")
    assert isinstance(result, tuple)


def test_resolve_last_week():
    result = resolve_time_reference("last week")
    assert isinstance(result, tuple)


def test_resolve_no_temporal():
    start, end = resolve_time_reference("just a regular query")
    assert start is None and end is None


def test_resolve_specific_month():
    result = resolve_time_reference("in january")
    assert isinstance(result, tuple)


def test_resolve_empty():
    start, end = resolve_time_reference("")
    assert start is None and end is None
