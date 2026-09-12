"""Pure token-expiry grammar and boundary tests."""

from __future__ import annotations

import pytest

from mind_mem.token_expiry import active_token_values, is_expired, parse_token_entry


def test_bare_token_has_no_expiry() -> None:
    assert parse_token_entry("plain-token") == ("plain-token", None)
    assert is_expired("plain-token", now=10**12) is False


def test_expiry_is_absolute_and_inclusive() -> None:
    assert parse_token_entry("rotating|exp=1000") == ("rotating", 1000)
    assert is_expired("rotating|exp=1000", now=999) is False
    assert is_expired("rotating|exp=1000", now=1000) is False
    assert is_expired("rotating|exp=1000", now=1001) is True


@pytest.mark.parametrize("entry", ["t|exp=not-a-number", "t|exp=", "t|exp=-1", "t|ttl=100"])
def test_malformed_or_unknown_expiry_fails_closed(entry: str) -> None:
    assert is_expired(entry, now=0) is True


def test_active_values_drop_expired_and_blank_entries() -> None:
    raw = "keep,drop|exp=100,keep-too|exp=9999,,"
    assert active_token_values(raw, now=500) == ["keep", "keep-too"]


def test_all_expired_values_stay_empty() -> None:
    assert active_token_values("a|exp=1,b|exp=2", now=500) == []


def test_huge_expiry_does_not_require_float_conversion() -> None:
    huge = 10**1000
    assert active_token_values(f"long-lived|exp={huge}", now=huge - 1) == ["long-lived"]
