"""Canary blocks — detect corpus poisoning by planting what must NOT move.

ROADMAP (Adversarial / poisoning defense): "per-actor anomaly detection + canary
blocks not yet shipped."

Of those two, canaries are the half this product can have WITHOUT breaking its own
wedge. Per-actor anomaly detection means learned per-actor scoring, which the
roadmap already rules out for trust scores: "No per-actor learned or anomaly
scoring (determinism wedge)." A canary is the opposite -- a fixed, known-good block
whose exact recall behaviour is recorded once, so any later deviation is a
deterministic signal rather than a statistical one.

WHAT A CANARY DETECTS, stated precisely, because "poisoning defense" oversells it:

  * a canary that STOPS being recalled for its own query -> something displaced,
    demoted or quarantined it. That is the ranking-suppression attack: bury the
    real answer under injected blocks.
  * a canary whose CONTENT changed -> an unauthorised edit reached the corpus.

WHAT IT DOES NOT DETECT: injected blocks that add a false claim without touching
the canary. A canary is a tripwire, not a filter. Saying so is the point -- a
defense that is believed to do more than it does is worse than none.

Pure: the fingerprint is a hash of recorded fields, no clock, so a canary check
run tomorrow compares like with like.
"""

from __future__ import annotations

import pytest

from mind_mem.canary import (
    CanaryVerdict,
    canary_fingerprint,
    check_canaries,
    is_canary,
)


def _canary(bid="CANARY-001", statement="The canary statement never changes", **kw):
    return {"_id": bid, "Statement": statement, "Status": "Active",
            "Tags": "canary", **kw}


# --------------------------------------------------------------------------
# Recognising a canary
# --------------------------------------------------------------------------

def test_a_tagged_block_is_a_canary():
    assert is_canary(_canary()) is True


def test_an_ordinary_block_is_not():
    """POSITIVE CONTROL: if everything were a canary the check is meaningless."""
    assert is_canary({"_id": "D-1", "Statement": "ordinary", "Tags": "decision"}) is False


def test_the_id_prefix_alone_is_not_enough():
    """A block must DECLARE itself a canary, not merely be named like one.

    Otherwise an attacker plants CANARY-999 and its 'clean' verdict launders the
    corpus -- the tripwire would be under the attacker's control.
    """
    assert is_canary({"_id": "CANARY-999", "Statement": "planted", "Tags": "decision"}) is False


# --------------------------------------------------------------------------
# The fingerprint is what makes a deviation detectable
# --------------------------------------------------------------------------

def test_the_fingerprint_is_stable_across_calls():
    c = _canary()
    assert canary_fingerprint(c) == canary_fingerprint(dict(c))


def test_changing_the_statement_changes_the_fingerprint():
    a, b = _canary(), _canary(statement="The canary statement was edited")
    assert canary_fingerprint(a) != canary_fingerprint(b)


def test_changing_the_status_changes_the_fingerprint():
    """A quarantined canary is a deviation even with identical text."""
    a, b = _canary(), _canary(Status="Quarantined")
    assert canary_fingerprint(a) != canary_fingerprint(b)


def test_an_unrelated_field_does_not_change_it():
    """POSITIVE CONTROL for the two above: a fingerprint over EVERYTHING would
    flip on any metadata touch and drown the signal in noise."""
    a, b = _canary(), _canary(AccessCount="17")
    assert canary_fingerprint(a) == canary_fingerprint(b)


# --------------------------------------------------------------------------
# check_canaries: the verdict
# --------------------------------------------------------------------------

def test_an_unchanged_canary_is_clean():
    canaries = [_canary()]
    baseline = {c["_id"]: canary_fingerprint(c) for c in canaries}
    v = check_canaries(canaries, baseline)
    assert isinstance(v, CanaryVerdict)
    assert v.ok is True and v.changed == () and v.missing == ()


def test_an_edited_canary_is_reported_by_id():
    original = _canary()
    baseline = {original["_id"]: canary_fingerprint(original)}
    v = check_canaries([_canary(statement="edited by someone")], baseline)
    assert v.ok is False
    assert v.changed == ("CANARY-001",), v


def test_a_VANISHED_canary_is_reported_and_not_silently_clean():
    """Deletion is the cheapest attack on a tripwire."""
    baseline = {"CANARY-001": canary_fingerprint(_canary())}
    v = check_canaries([], baseline)
    assert v.ok is False
    assert v.missing == ("CANARY-001",), v


def test_a_new_canary_absent_from_the_baseline_is_not_a_failure():
    """Planting a canary must not require re-baselining before it is legal."""
    v = check_canaries([_canary(bid="CANARY-NEW")], {})
    assert v.ok is True, v


def test_an_empty_baseline_and_empty_corpus_is_not_a_pass_worth_trusting():
    """A check over nothing must SAY so rather than report health.

    This is the vacuous-pass failure: 'no canaries deviated' reads identically
    whether the corpus is clean or the canaries were never planted.
    """
    v = check_canaries([], {})
    assert v.ok is True
    assert v.vacuous is True, "a check with no canaries must mark itself vacuous"


def test_a_real_check_is_not_marked_vacuous():
    c = _canary()
    v = check_canaries([c], {c["_id"]: canary_fingerprint(c)})
    assert v.vacuous is False
