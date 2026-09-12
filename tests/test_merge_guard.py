"""Description-grounded over-merge guard.

ROADMAP ("Description-grounded entity resolution"): "...using a one-line per-entity
description as disambiguation context rather than edit distance. ... Two enforced
failure modes: unmatched name -> single-element cluster (never silently dropped);
over-merge guarded by description mismatch + HITL review."

The first failure mode landed with `entity_blocking`. This is the second, and it is
the one with teeth: merging two entities that are NOT the same destroys information
with no signal -- the two descriptions become one, and nothing records that a choice
was made. Splitting a wrongly-split entity later is cheap; unmerging is not.

So the guard's whole job is to never hand back "yes, merge". The vocabulary is closed
and every arm withholds the automatic merge:

  * REFUSED_KIND_CONFLICT     -- the descriptions assert different kinds of thing
  * REFUSED_NO_SHARED_CONTEXT -- nothing in either description corroborates the other
  * REVIEW_REQUIRED           -- a human decides

`REVIEW_REQUIRED` is emphatically NOT a pass, and the tests pin that no input
whatsoever yields `auto_merge=True`. That assertion is the point of the module: a
later edit that adds a fast path for "obviously the same person" fails this file and
has to argue with the roadmap's HITL requirement instead of quietly bypassing it.

A missing description is REVIEW_REQUIRED, not silence: a check that could not run
must never return what a passing check returns.

Pure: no model, no clock, no I/O.
"""

from __future__ import annotations

from mind_mem.merge_guard import KINDS, MergeVerdict, assess_merge

PERSON = "Ada Lovelace, mathematician on the analytical-engine notes"
COMPANY = "Lovelace Ltd, a company selling analytical tooling"
SAME_PERSON = "A. Lovelace, the mathematician behind the analytical notes"
UNRELATED_PERSON = "Grace Hopper, an admiral in naval computing"


def test_no_input_whatsoever_authorises_an_automatic_merge():
    """THE LOAD-BEARING TEST. The roadmap requires HITL review for every merge, so
    there must be no input at all for which this guard says "go ahead"."""
    inputs = [
        (PERSON, SAME_PERSON), (PERSON, COMPANY), (PERSON, UNRELATED_PERSON),
        (PERSON, PERSON), ("", ""), (PERSON, ""), ("", PERSON),
        (PERSON, "   "), ("a", "a"), (COMPANY, COMPANY),
    ]
    for left, right in inputs:
        got = assess_merge(left, right)
        assert got.auto_merge is False, (left, right, got)


def test_a_kind_conflict_is_refused_outright():
    got = assess_merge(PERSON, COMPANY)
    assert got.verdict is MergeVerdict.REFUSED_KIND_CONFLICT, got
    assert "person" in got.reason and "organisation" in got.reason, got.reason


def test_the_same_kind_with_corroborating_context_goes_to_review():
    """POSITIVE CONTROL for the test above: the strongest possible same-entity
    signal still only reaches REVIEW_REQUIRED, so the no-auto-merge assertion is
    not passing merely because every input happened to be refused."""
    got = assess_merge(PERSON, SAME_PERSON)
    assert got.verdict is MergeVerdict.REVIEW_REQUIRED, got


def test_the_same_kind_without_shared_context_is_refused():
    got = assess_merge(PERSON, UNRELATED_PERSON)
    assert got.verdict is MergeVerdict.REFUSED_NO_SHARED_CONTEXT, got


def test_a_missing_description_goes_to_review_not_silence():
    """A check that could not run must not return what a passing check returns —
    and must say WHY, since a human is the next reader."""
    for left, right in ((PERSON, ""), ("", PERSON), ("", ""), (PERSON, "   ")):
        got = assess_merge(left, right)
        assert got.verdict is MergeVerdict.REVIEW_REQUIRED, (left, right, got)
        assert "description" in got.reason, got.reason


def test_every_verdict_carries_a_reason():
    """The verdict is queued for a person. A verdict with no reason is a decision
    they cannot review, which defeats the HITL guard it exists to serve."""
    for left, right in ((PERSON, COMPANY), (PERSON, SAME_PERSON),
                        (PERSON, UNRELATED_PERSON), ("", PERSON)):
        assert assess_merge(left, right).reason.strip()


def test_the_assessment_is_symmetric():
    """Merge is symmetric, so argument order must not change the answer — a guard
    that refuses (a,b) and reviews (b,a) can be defeated by swapping."""
    for left, right in ((PERSON, COMPANY), (PERSON, SAME_PERSON),
                        (PERSON, UNRELATED_PERSON), ("", PERSON)):
        assert assess_merge(left, right).verdict is assess_merge(right, left).verdict


def test_an_undeterminable_kind_does_not_invent_one():
    """Neither description names a kind, so there is no kind conflict to find; the
    guard must not guess a kind and then refuse on the guess."""
    got = assess_merge("the widget on the shelf", "the widget in the crate")
    assert got.verdict is not MergeVerdict.REFUSED_KIND_CONFLICT, got
    assert got.auto_merge is False


def test_an_ambiguous_kind_fails_toward_review_not_toward_refusal():
    """A description matching TWO kinds is genuinely ambiguous — "the engineer at
    the agency" names a person and an organisation. Guessing one would drive a
    refusal the reviewer cannot check, so ambiguity must not reach the conflict
    arm. This is the safe direction: the guard loses a catch, never invents one."""
    got = assess_merge("the engineer at the agency, our main contact",
                       "the agency contact we engineer releases with")
    assert got.verdict is not MergeVerdict.REFUSED_KIND_CONFLICT, got
    assert got.auto_merge is False


def test_the_kind_set_is_closed():
    """An open kind set means an unseen word becomes a new kind, and two entities
    of genuinely the same kind then read as a conflict."""
    assert set(KINDS) == {"person", "organisation", "place", "product", "event"}


def test_it_is_deterministic():
    first = assess_merge(PERSON, SAME_PERSON)
    second = assess_merge(PERSON, SAME_PERSON)
    assert (first.verdict, first.reason, first.auto_merge) == (
        second.verdict, second.reason, second.auto_merge)
