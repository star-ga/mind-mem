"""L2 evaluation: should a PRESCRIPTIVE block kind exist? Measured answer: no.

ROADMAP ("L2 — Prescriptive blocks (EVALUATE, not committed)"): "Assess whether a
procedural/prescriptive block kind — recall returns *a strategy*, not *a fact* — earns a
place in the governed store. ... a strategy is a claim about *what worked*, which is
exactly the kind of assertion that goes stale silently and that the contradiction-detection
surface was built to catch for facts. Adding a block kind whose staleness is harder to
detect than a fact's would trade recall breadth for governance strength — the wrong
direction for this project."

The deliverable is a recommendation with evidence, and this file IS the evidence: the
measurements are pinned as tests so the recommendation cannot quietly rot. **If the
detector ever becomes order-sensitive, the last test here goes red and the evaluation must
be revisited** — which is the only honest way to record a "no" that depends on a current
limitation.

RECOMMENDATION: **do not add a prescriptive block kind.** Two measurements, and the second
is stronger than the item supposed.

1. DEMAND IS ~0.1%. Over the live 2,726-block corpus: 297 blocks (10.9%) carry imperative
   language ("always", "never", "prefer"), but only 2 (0.1%) have conditional-strategy
   shape ("if X then use Y"), and the single block matching both signals is a false
   positive (a research-findings summary). The 10.9% are standing RULES — propositional
   claims about policy — and decision blocks already hold them well.

2. THE GOVERNANCE SURFACE IS NOT JUST WEAKER FOR STRATEGIES, IT IS MISLEADING. "Always run
   the byte-identity gate before the benchmark" and "Always run the benchmark before the
   byte-identity gate" are token-identical: order is the only difference, and a
   bag-of-words comparator cannot see order. Measured cosine similarity **1.000**, verdict
   **"duplicate"** — reported to a reviewer as "🔄 near-duplicate found". A reviewer told
   "you already have this" would reasonably reject the new block as REDUNDANT when it is
   the OPPOSITE of the stored one.

   For a FACT the same surface says "refinement", which invites comparison. For a strategy
   it says "duplicate", which invites dismissal. The failure is an actively wrong label,
   not a missed detection — worse than the item's stated worry.

The prerequisite, if it is ever wanted: an ORDER-SENSITIVE comparator, because order is
most of what a procedure asserts. Until that exists, adding the kind trades governance
strength for recall breadth in precisely the direction the item warns against.
"""

from __future__ import annotations

from mind_mem.contradiction_detector import _classify_conflict, _tfidf_cosine_similarity

GATE_FIRST = "Always run the byte-identity gate before the benchmark."
BENCH_FIRST = "Always run the benchmark before the byte-identity gate."


def test_a_reversed_procedure_is_token_identical_to_the_comparator():
    """The mechanism behind the whole recommendation. Order is the only difference, and
    a bag-of-words similarity cannot see order at all."""
    assert _tfidf_cosine_similarity(GATE_FIRST, BENCH_FIRST) == 1.0


def test_a_reversed_procedure_is_MISLABELLED_as_a_duplicate():
    """Not "missed" — mislabelled, in the most misleading direction available. A reviewer
    told "near-duplicate" would reject the new block as redundant when it is the opposite
    of the stored one."""
    sim = _tfidf_cosine_similarity(GATE_FIRST, BENCH_FIRST)
    assert _classify_conflict(GATE_FIRST, BENCH_FIRST, sim) == "duplicate"


def test_a_conflicting_FACT_gets_a_label_that_invites_comparison():
    """The contrast that makes the recommendation specific rather than general pessimism.
    A fact conflict is labelled "refinement" — still not "contradiction", but a label that
    sends a reviewer to compare the two rather than to discard one."""
    a, b = "The release ships on Tuesday.", "The release ships on Friday."
    sim = _tfidf_cosine_similarity(a, b)
    assert _classify_conflict(a, b, sim) == "refinement"


def test_the_detector_CAN_say_contradiction_so_the_finding_is_not_vacuous():
    """POSITIVE CONTROL. If `_classify_conflict` never returned "contradiction" at all,
    every result above would be uninformative about strategies specifically."""
    # "true"/"false" is used because it VERIFIABLY works. An earlier version used
    # "enabled"/"disabled" and failed — the status-reversal list holds bare verb forms
    # matched with word boundaries (`\benable\b`), so the INFLECTED forms never match.
    # That is a separate, real defect in the fact path, fixed in its own commit rather
    # than folded into this evaluation.
    a = "The strict-FP flag is true for this build."
    b = "The strict-FP flag is false for this build."
    sim = _tfidf_cosine_similarity(a, b)
    assert _classify_conflict(a, b, sim) == "contradiction", sim


def test_THE_EVALUATION_DEPENDS_ON_THIS_AND_MUST_BE_REVISITED_IF_IT_CHANGES():
    """The recommendation is a "no" that rests on a CURRENT limitation, so it is pinned.

    If the comparator ever becomes order-sensitive — a token-order term, an n-gram
    feature, anything — this assertion goes red, and the L2 recommendation has to be
    re-derived rather than inherited. Recording a conditional "no" without a trigger that
    fires when its condition changes is how a stale decision survives.
    """
    forward = "first compile then link"
    backward = "first link then compile"
    assert _tfidf_cosine_similarity(forward, backward) == 1.0, (
        "the comparator now distinguishes token ORDER; the L2 prescriptive-blocks "
        "evaluation rested on it being unable to, and must be redone"
    )
