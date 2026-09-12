"""Token rotation: the grace window must EXPIRE server-side, not advisorily.

ROADMAP residual, recorded against the shipped token-rotation primitive: "the grace
window is ADVISORY. `grace_seconds` is printed and the operator must run the emitted
`shell_final` export; the server never timestamps or auto-expires an old token, so
the 'then expires' half of the original line is not implemented."

MEASURED before building: `MIND_MEM_TOKENS="old,new"` yields both tokens active,
neither carrying an expiry, so a retired credential stays valid until a human edits
an environment variable. Rotation without expiry is not rotation -- it is
accumulation, and the security property the feature advertises ("then expires")
does not exist.

THE COMPATIBILITY CONSTRAINT is why this is additive rather than a redesign: every
existing deployment sets bare tokens with no timestamp, and those must keep working
forever. An expiry is therefore an OPTIONAL suffix on an entry, and a bare entry
means "no expiry" exactly as it does today.

FAIL-CLOSED CHOICE, stated because it is the one judgement call here: an entry whose
expiry cannot be parsed is treated as EXPIRED, not as never-expiring. A typo in a
timestamp must not silently grant a permanent credential -- that is the direction a
mistake should fail in when the subject is authentication.
"""

from __future__ import annotations

import pytest

from mind_mem.token_expiry import (
    active_token_values,
    is_expired,
    parse_token_entry,
)


# --------------------------------------------------------------------------
# Backward compatibility: a bare token still works, forever
# --------------------------------------------------------------------------

def test_a_bare_token_has_no_expiry():
    value, expires = parse_token_entry("plain-token-value")
    assert value == "plain-token-value"
    assert expires is None


def test_a_bare_token_is_never_expired():
    """Every deployment today looks like this. It must keep working."""
    assert is_expired("plain-token-value", now=10**12) is False


def test_whitespace_around_an_entry_is_tolerated():
    value, _ = parse_token_entry("  spaced-token  ")
    assert value == "spaced-token"


# --------------------------------------------------------------------------
# The new half: an entry may carry an expiry
# --------------------------------------------------------------------------

def test_an_entry_can_declare_an_expiry():
    value, expires = parse_token_entry("rotating-token|exp=1000")
    assert value == "rotating-token"
    assert expires == 1000


def test_a_token_past_its_expiry_is_expired():
    assert is_expired("rotating-token|exp=1000", now=1001) is True


def test_a_token_before_its_expiry_is_live():
    """POSITIVE CONTROL: an is_expired that always returned True would lock
    every operator out while passing the test above."""
    assert is_expired("rotating-token|exp=1000", now=999) is False


def test_the_boundary_is_inclusive_of_the_live_side():
    """At exactly the expiry instant the token is still valid.

    Chosen so a clock equal to the deadline cannot produce a spurious
    authentication failure mid-request; the ambiguity is resolved toward the
    in-flight client the grace window exists to protect.
    """
    assert is_expired("t|exp=1000", now=1000) is False


# --------------------------------------------------------------------------
# Fail closed on a malformed expiry
# --------------------------------------------------------------------------

@pytest.mark.parametrize("entry", ["t|exp=notanumber", "t|exp=", "t|exp=-1"])
def test_a_malformed_expiry_is_treated_as_EXPIRED(entry):
    """A typo must not grant a permanent credential."""
    assert is_expired(entry, now=0) is True


def test_an_unknown_suffix_key_fails_closed_too():
    """`|ttl=` is not `|exp=`; guessing what an operator meant is not safe here."""
    assert is_expired("t|ttl=1000", now=0) is True


# --------------------------------------------------------------------------
# The set a server should accept
# --------------------------------------------------------------------------

def test_expired_entries_are_dropped_from_the_active_set():
    raw = "keep-me,drop-me|exp=100,keep-me-too|exp=9999"
    assert active_token_values(raw, now=500) == ["keep-me", "keep-me-too"]


def test_an_all_expired_set_is_EMPTY_not_silently_permissive():
    """The dangerous failure: expiring every token must lock the door, not open it.

    A caller seeing [] must refuse the request. Returning the raw list on an
    empty result would turn a fully-rotated deployment into an open one.
    """
    assert active_token_values("a|exp=1,b|exp=2", now=500) == []


def test_a_legacy_set_with_no_expiries_is_returned_whole():
    assert active_token_values("a,b,c", now=10**12) == ["a", "b", "c"]


def test_blank_entries_are_ignored_rather_than_becoming_an_empty_token():
    """An empty token value would match an empty Authorization header."""
    assert active_token_values("a,,b, ,c", now=0) == ["a", "b", "c"]
