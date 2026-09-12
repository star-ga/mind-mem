"""Description-grounded over-merge guard for entity resolution.

ROADMAP ("Description-grounded entity resolution"): "over-merge guarded by
description mismatch + HITL review."

Merging two entities that are not the same destroys information with no signal left
behind: the two descriptions become one and nothing records that a choice was made.
A wrongly-SPLIT entity is cheap to fix later; a wrongly-MERGED one usually is not.
That asymmetry is why this module exists and why it is shaped the way it is.

The guard never authorises a merge. Its verdict vocabulary is closed and every arm
withholds automatic action:

  REFUSED_KIND_CONFLICT      the two descriptions assert different kinds of thing
  REFUSED_NO_SHARED_CONTEXT  nothing in either description corroborates the other
  REVIEW_REQUIRED            a person decides

`REVIEW_REQUIRED` is not a pass. `MergeAssessment.auto_merge` is `False` on every
path, and a test asserts that across a matrix of inputs -- including the strongest
possible same-entity signal -- so a later "obviously the same person" fast path has
to argue with the HITL requirement rather than quietly bypass it.

A missing or blank description yields REVIEW_REQUIRED with a reason that says so. A
check that could not run must never return what a passing check returns.

The kind set is CLOSED. With an open set, an unrecognised noun becomes a new kind and
two entities of genuinely the same kind read as a conflict -- the guard would then
refuse real merges for a reason no reviewer can act on.

Pure: no model, no clock, no I/O. The description-similarity judgement the roadmap
describes needs a model; this is the deterministic part that runs first and decides
what the model (and the human) is asked about at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

__all__ = ["KINDS", "MergeVerdict", "MergeAssessment", "assess_merge"]

#: Mutually exclusive kinds, each with the surface words that assert it. Closed on
#: purpose -- see the module docstring.
#:
#: Every marker here is a word that names WHAT THE SUBJECT IS. Words that commonly
#: name an OBJECT inside a description are deliberately absent: "engine" would make
#: "mathematician who worked on the Analytical Engine" match both `person` and
#: `product`, and a bag of words cannot tell a subject from an object. Their absence
#: costs the guard some conflicts it might have caught; including them would cost it
#: refusals of real merges, which is the expensive direction.
_KIND_WORDS: dict[str, frozenset[str]] = {
    "person": frozenset({
        "person", "mathematician", "engineer", "author", "admiral", "scientist",
        "developer", "researcher", "founder", "artist", "musician", "teacher",
        "physician", "colleague", "contractor", "analyst",
    }),
    "organisation": frozenset({
        "organisation", "organization", "company", "corporation", "ltd", "inc",
        "llc", "gmbh", "startup", "agency", "institute", "university",
        "foundation", "nonprofit", "consultancy",
    }),
    "place": frozenset({
        "place", "city", "town", "country", "region", "province", "village",
        "island", "district", "neighbourhood", "neighborhood",
    }),
    "product": frozenset({
        "library", "package", "device", "app", "application",
        "platform", "codebase", "gadget", "appliance",
    }),
    "event": frozenset({
        "event", "conference", "meeting", "release", "launch", "summit",
        "workshop", "outage", "incident", "reorg",
    }),
}

KINDS: tuple[str, ...] = tuple(_KIND_WORDS)

#: Words too common to corroborate anything. A shared "the" is not shared context,
#: and treating it as such would turn REFUSED_NO_SHARED_CONTEXT into a verdict no
#: pair ever receives -- a guard arm that cannot fire is not a guard arm.
_STOP_WORDS = frozenset({
    "a", "an", "and", "the", "of", "on", "in", "at", "to", "for", "with", "by",
    "is", "was", "are", "were", "be", "been", "as", "that", "which", "who",
    "this", "these", "those", "it", "its", "from", "or", "but", "not", "no",
    "he", "she", "they", "them", "his", "her", "their", "worked", "built",
    "thing", "other",
})


class MergeVerdict(Enum):
    """Closed verdict set. Every arm withholds the automatic merge."""

    REFUSED_KIND_CONFLICT = "refused-kind-conflict"
    REFUSED_NO_SHARED_CONTEXT = "refused-no-shared-context"
    REVIEW_REQUIRED = "review-required"


@dataclass(frozen=True)
class MergeAssessment:
    """A verdict, why, and the standing refusal to act on it automatically.

    `auto_merge` is a field rather than an absent concept so that the "no path
    authorises a merge" property is something a test can assert about every result,
    instead of a convention a reader has to trust.
    """

    verdict: MergeVerdict
    reason: str
    auto_merge: bool = False


def _words(description: str) -> frozenset[str]:
    cleaned = "".join(c.lower() if c.isalnum() else " " for c in description)
    return frozenset(w for w in cleaned.split() if w)


def _kind_of(words: frozenset[str]) -> str | None:
    """The single kind this description asserts, or None.

    A description matching two kinds returns None rather than picking one: "the
    engineer at the company" is genuinely ambiguous, and guessing would let the
    guess drive a refusal the reviewer cannot check.
    """
    hits = [kind for kind, markers in _KIND_WORDS.items() if words & markers]
    return hits[0] if len(hits) == 1 else None


def assess_merge(left_description: str, right_description: str) -> MergeAssessment:
    """Judge a candidate entity merge from the two one-line descriptions.

    Symmetric in its arguments, because merge is: a guard that refuses `(a, b)` and
    reviews `(b, a)` is defeated by swapping.
    """
    left, right = _words(left_description), _words(right_description)

    if not left or not right:
        which = "both descriptions" if not (left or right) else "one description"
        return MergeAssessment(
            MergeVerdict.REVIEW_REQUIRED,
            f"{which} is missing or blank, so the description check could not run",
        )

    left_kind, right_kind = _kind_of(left), _kind_of(right)
    if left_kind and right_kind and left_kind != right_kind:
        kinds = " and ".join(sorted((left_kind, right_kind)))
        return MergeAssessment(
            MergeVerdict.REFUSED_KIND_CONFLICT,
            f"the descriptions assert different kinds: {kinds}",
        )

    shared = (left & right) - _STOP_WORDS
    if not shared:
        return MergeAssessment(
            MergeVerdict.REFUSED_NO_SHARED_CONTEXT,
            "no content word appears in both descriptions, so neither corroborates "
            "the other",
        )

    corroboration = ", ".join(sorted(shared)[:5])
    return MergeAssessment(
        MergeVerdict.REVIEW_REQUIRED,
        f"descriptions are compatible and share: {corroboration} — a reviewer "
        f"decides; this guard never merges on its own",
    )
