"""Status-reversal detection missed every INFLECTED form.

Found 2026-09-11 while building the L2 prescriptive-blocks evaluation, when a positive
control asserting that "enabled" vs "disabled" is a contradiction FAILED.

`_classify_conflict` carries a `status_reversal_pairs` list of bare verb forms —
`("enable", "disable")`, `("allow", "deny")`, `("accept", "reject")`, `("add", "remove")`,
`("increase", "decrease")`, `("start", "stop")` — and matches each with a word boundary
(`\\benable\\b`). So the forms that actually appear in written decisions never match:

    "The federation endpoint is ENABLED"  vs  "... is DISABLED"   -> refinement, not
                                                                     contradiction

A decision block says "tier decay is enabled", not "tier decay enable". The inflected
forms are the common case and the bare imperative is the rare one, so the check was
missing most of what it was written to catch.

This is under-detection in a CONTRADICTION-DETECTION product, which is the direction that
matters: a missed contradiction is a corpus that quietly holds both answers, and the
reviewer is never asked. It is also why the fix is conservative — the pairs are extended,
never the thresholds, so nothing that was previously a contradiction stops being one.
"""

from __future__ import annotations

import pytest

from mind_mem.contradiction_detector import _classify_conflict, _tfidf_cosine_similarity


def _verdict(a: str, b: str) -> str:
    return _classify_conflict(a, b, _tfidf_cosine_similarity(a, b))


@pytest.mark.parametrize(
    ("word_a", "word_b"),
    [
        ("enabled", "disabled"),
        ("allowed", "denied"),
        ("accepted", "rejected"),
        ("added", "removed"),
        ("increased", "decreased"),
        ("started", "stopped"),
        ("enabling", "disabling"),
    ],
)
def test_an_inflected_status_reversal_is_a_contradiction(word_a: str, word_b: str):
    a = f"The tier decay setting is {word_a} for this workspace."
    b = f"The tier decay setting is {word_b} for this workspace."
    assert _verdict(a, b) == "contradiction", (word_a, word_b, _tfidf_cosine_similarity(a, b))


@pytest.mark.parametrize(
    ("word_a", "word_b"),
    [("enable", "disable"), ("allow", "deny"), ("accept", "reject"), ("true", "false")],
)
def test_the_BARE_forms_still_work(word_a: str, word_b: str):
    """REGRESSION GUARD. The fix extends the pair list; it must not replace the forms
    that already matched, or it trades one blind spot for another."""
    a = f"The policy is set to {word_a} for every peer."
    b = f"The policy is set to {word_b} for every peer."
    assert _verdict(a, b) == "contradiction", (word_a, word_b)


def test_unrelated_text_is_STILL_not_a_contradiction():
    """POSITIVE CONTROL against over-detection. Extending a word list is exactly the kind
    of change that starts flagging everything, and a detector that cries contradiction on
    every pair trains an operator to ignore it — which is worse than under-detecting,
    because it disables the surface rather than narrowing it."""
    a = "The release ships on Tuesday and the build is green."
    b = "Ada Lovelace wrote the Analytical Engine notes."
    assert _verdict(a, b) != "contradiction"


def test_one_word_of_a_pair_alone_is_not_a_contradiction():
    """Both halves must be present on opposite sides. A block that merely says "enabled"
    does not contradict another that says "enabled"."""
    a = "The tier decay setting is enabled for this workspace."
    b = "The tier decay setting is enabled for every other workspace."
    assert _verdict(a, b) != "contradiction"
