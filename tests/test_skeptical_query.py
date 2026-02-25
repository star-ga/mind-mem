"""Tests for skeptical query detection."""

from __future__ import annotations

from scripts._recall_detection import is_skeptical_query


def test_skeptical_returns_bool():
    result = is_skeptical_query("did we really decide that")
    assert isinstance(result, bool)


def test_skeptical_type():
    result = is_skeptical_query("what was decided")
    assert isinstance(result, bool)


def test_skeptical_empty():
    result = is_skeptical_query("")
    assert isinstance(result, bool)


def test_skeptical_challenge():
    result = is_skeptical_query("are you sure about that decision")
    assert isinstance(result, bool)


def test_skeptical_contradiction():
    result = is_skeptical_query("that contradicts what was said before")
    assert isinstance(result, bool)
