"""Enum-keyed upsert slots: make contradiction structurally impossible (M4).

WHAT THIS IS FOR. `contradiction_detector`, `conflict_resolver` and
`compiled_truth_contradictions` are all DETECTIVE: they run after two conflicting
facts already coexist, and they only help if detection finds them. For a BOUNDED
fact space that is unnecessary work with a false-negative rate. If each fact is
filed under a topic slug drawn from a CLOSED set, a second statement on the same
topic collides by construction -- an exact key match, at zero detection cost.

THE CLOSED SET IS THE LOAD-BEARING PART. An open-ended topic string lets an
extractor file ``plan`` on Monday and ``plan_tier`` on Friday, and two
contradicting facts then never collide at all. :data:`SLOTS` is therefore closed
and :func:`normalise_slot` REFUSES anything outside it, rather than accepting an
invented member and silently losing the collision.

WHAT WE DELIBERATELY DO NOT COPY. The pattern this borrows from upserts by silent
overwrite -- no proposal, no lineage, no rollback. Nothing here mutates a block.
:func:`supersession_target` returns the id a caller should PROPOSE to supersede,
so the replacement travels the governed path (``propose_update`` ->
``approve_apply``) and stays a recorded, reversible event. Their exactness with
our audit trail is strictly better than either alone, and it is the honest reason
to build this rather than adopt theirs.

RESIDUAL RISK, stated rather than hidden. The model still chooses the slug. This
moves the failure from "forgets what it wrote" to "picks the WRONG slot" -- a
closed enum rejects an invented member, but it cannot tell that a fact about
ownership was filed under ``plan``. That is narrower and validatable, NOT
eliminated, and contradiction is not solved by this module.

Its sibling cost, also deliberate: a free-form fact with no natural identity has
no slot, therefore nothing to collide on, and such facts accumulate. That trade is
accepted -- what is refused is content-hash keying at demo width, since 32 bits is
collision-prone as a durable identity.

PRECEDENT is internal and one layer up: ``512-mind/src/drift.mind`` applies the
same closed-set discipline to MEANING rather than to keys -- ``no_semantic_drift``
enumerates the mutation classes that corrupt a contract ("must not" -> "should
not" weakens an obligation; "fail open" -> "fail safe" inverts a default) and
asserts against that fixed list. Enumerate the space so the violation is
structural instead of detected. 121 enums across that repo make it house style.

Pure by construction: no I/O, no clock, no randomness. A slot decision gates a
governed write, so it must replay identically.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Iterable, Mapping, Optional, Sequence

#: Field a block carries its slot in. Absent for free-form facts.
SLOT_FIELD = "Slot"


class Slot(str, Enum):
    """The CLOSED set of topic slugs a bounded fact may be filed under.

    Deliberately small. Every member is a topic where a second statement
    genuinely REPLACES the first rather than adding to it -- which is the
    property that makes an exact-key collision the right semantics. A topic
    where two facts can legitimately coexist does not belong here; it belongs in
    the free-form space, and M4's escape-hatch cost applies to it.

    Adding a member is a product decision, not a refactor: it declares that a
    second fact on that topic must supersede rather than accumulate.
    """

    #: The current plan of record for the thing the block is about.
    PLAN = "plan"
    #: Who owns it. One owner at a time by definition.
    OWNER = "owner"
    #: Its current status/standing, where a later reading replaces an earlier.
    STATE = "state"
    #: The single deadline or target date in force.
    DEADLINE = "deadline"
    #: The decision of record on a settled question.
    VERDICT = "verdict"
    #: The version or revision currently in force.
    VERSION = "version"


#: The legal slugs, as plain strings, for callers that never touch the enum.
SLOTS: frozenset[str] = frozenset(s.value for s in Slot)


class UnknownSlot(ValueError):
    """A slug outside :data:`SLOTS`.

    Fatal rather than tolerated. Accepting an invented slug is precisely the
    failure this module exists to prevent: the fact would be filed, the
    collision would never happen, and nothing would report a problem.
    """


def normalise_slot(raw: object) -> str:
    """The canonical slug for *raw*, or raise :class:`UnknownSlot`.

    Case and surrounding whitespace normalise, because ``Plan`` and ``plan`` must
    be ONE key -- two spellings of one topic is the open-string failure wearing a
    disguise. Anything else is refused.
    """
    text = str(raw or "").strip().lower()
    if text not in SLOTS:
        raise UnknownSlot(
            f"{raw!r} is not a legal upsert slot. Legal slots: {sorted(SLOTS)}. "
            f"An invented slug would file the fact where nothing can collide with "
            f"it, which is the contradiction this module exists to prevent -- so "
            f"pick an existing member, or add one deliberately (that is a product "
            f"decision: it declares a second fact on the topic must SUPERSEDE)."
        )
    return text


def slot_of(block: Optional[Mapping[str, Any]]) -> Optional[str]:
    """The block's canonical slot, or ``None`` for a free-form fact.

    Returns ``None`` rather than raising on an ABSENT slot: having no slot is a
    legitimate state (M4's escape hatch). An absent slot and an INVALID one are
    different facts, so a present-but-illegal value still raises.
    """
    if not block:
        return None
    raw = block.get(SLOT_FIELD)
    if raw is None or str(raw).strip() == "":
        return None
    return normalise_slot(raw)


def collides(a: Optional[Mapping[str, Any]], b: Optional[Mapping[str, Any]]) -> bool:
    """True when *a* and *b* occupy the same slot and are different blocks.

    Exact, symmetric, and free of similarity scoring. Two blocks on one slot
    collide however differently they are worded; two blocks on different slots do
    not collide however alike they read -- the key is the authority, which is
    what makes the guarantee structural instead of probabilistic.

    A block never collides with itself: an upsert must not propose superseding
    the very block it is updating.
    """
    sa, sb = slot_of(a), slot_of(b)
    if sa is None or sb is None or sa != sb:
        return False
    ida = str((a or {}).get("_id") or (a or {}).get("id") or "")
    idb = str((b or {}).get("_id") or (b or {}).get("id") or "")
    if ida and idb and ida == idb:
        return False
    return True


def _order_key(block: Mapping[str, Any]) -> tuple[str, str]:
    """Sort key for choosing among incumbents: newest Date, then id.

    ``Date`` is the block's own recorded field, not a clock read here -- so the
    choice replays identically. The id tiebreak exists so two same-dated
    incumbents resolve deterministically rather than by dict order: an arbitrary
    pick inside a governed path turns a reversible event into an unexplainable
    one.
    """
    return (str(block.get("Date") or ""), str(block.get("_id") or block.get("id") or ""))


def supersession_target(
    existing: Sequence[Mapping[str, Any]] | Iterable[Mapping[str, Any]],
    *,
    slot: object,
) -> Optional[str]:
    """The block id a new fact on *slot* should PROPOSE to supersede.

    Returns ``None`` when the slot is unoccupied -- then the write is an ordinary
    new block with nothing to replace.

    This returns an id and nothing else. It performs no write, marks nothing
    superseded and touches no file, because the replacement must travel
    ``propose_update`` -> ``approve_apply`` to stay a recorded, reversible event.
    A function here that mutated would reproduce the silent overwrite this design
    explicitly refuses.

    Raises:
        UnknownSlot: *slot* is not a legal member. Fail closed -- an invented
            slug must not degrade into "no incumbent, write freely".
    """
    canonical = normalise_slot(slot)
    occupants = [b for b in (existing or ()) if slot_of(b) == canonical]
    if not occupants:
        return None
    newest = max(occupants, key=_order_key)
    return str(newest.get("_id") or newest.get("id") or "") or None


def upsert_plan(
    existing: Sequence[Mapping[str, Any]] | Iterable[Mapping[str, Any]],
    *,
    slot: object,
) -> dict[str, Optional[str]]:
    """What a governed writer should DO about a new fact on *slot*.

    The one function a caller needs, and it returns a PLAN rather than performing
    anything:

    * ``{"action": "create", "supersedes": None}`` -- the slot is free.
    * ``{"action": "supersede", "supersedes": "<block id>"}`` -- the slot is
      occupied, and this id is the incumbent the caller must route through
      ``propose_update`` -> ``approve_apply``.

    Returning a plan is the whole discipline. A function that wrote the
    supersession itself would be the silent overwrite this module refuses, and the
    audit trail is the only reason to build this instead of adopting the pattern
    it borrows from. The caller stays the one governed door; this only tells it
    what the closed-set key implies.

    Raises:
        UnknownSlot: fail closed on an invented slug rather than degrading to
            "create", which would file a fact where nothing can collide with it.
    """
    target = supersession_target(existing, slot=slot)
    if target is None:
        return {"action": "create", "supersedes": None}
    return {"action": "supersede", "supersedes": target}


__all__ = [
    "SLOT_FIELD",
    "SLOTS",
    "Slot",
    "UnknownSlot",
    "collides",
    "normalise_slot",
    "slot_of",
    "supersession_target",
    "upsert_plan",
]
