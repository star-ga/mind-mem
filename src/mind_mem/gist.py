"""The gist/slot split: what gets EMBEDDED versus what FILTERS (item 1b).

WHY THIS EXISTS, concretely. Two records that differ only in a slot VALUE --
"ship date is Tuesday" and "ship date is Friday" -- are nearly identical as text.
An embedding of the full statement therefore scores them as the same fact, and a
similarity search cannot separate them. That is the "one flat string per record"
failure: the contradiction is in the value, and the value is the part a semantic
match throws away.

Strip the slot value and the two become the SAME gist. That identity is the
signal: same gist + different slot value is a SLOT-DELTA (the second fact
replaces the first), while a different gist is a genuinely new fact.

    gist  -> embedded, for "is this about the same thing?"
    slots -> exact-match filters, for "which value does it assert?"

``upsert_slots`` (M4) is the write half of this and answers "must this collide?".
This is the read half, and ROADMAP 1b is explicit that one without the other is
broken: "building those without the embed split wires slots into writes while
leaving reads broken."

THE NON-NEGOTIABLE. The canonical ``Statement`` must never move. The evidence
chain and MIC preimages hash it, so a function that rewrote it would silently
invalidate every receipt already issued. Everything here DERIVES a value and
returns it; nothing in this module mutates a block.

Pure: no clock, no I/O, no randomness. A gist decides whether a governed write
supersedes an existing fact, so it must replay identically -- a gist that drifted
with today's stopword list would make yesterday's supersession unexplainable.

RESIDUAL LIMIT, stated rather than implied: the gist is a lexical derivation, not
an understanding. Two facts phrased with entirely different vocabulary about the
same topic will not share a gist, and this module will call them distinct. It
narrows the "same flat string" failure; it does not solve paraphrase.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Optional

#: Field holding the canonical text. Read-only here, always.
STATEMENT_FIELD = "Statement"

#: Template words that carry no fact and would otherwise fork one gist into two.
#: Deliberately SMALL and closed: every addition changes what counts as "the same
#: thing", and a long list starts collapsing facts that differ meaningfully.
_BOILERPLATE = frozenset({"the", "a", "an", "is", "are", "was", "were", "be", "to", "of", "on", "at"})

#: A slot's value is the tail after its key phrase. Kept as one pattern per slot
#: rather than a general parser: a general one would guess, and a wrong guess here
#: silently mis-files a fact under the right key with the wrong value.
_VALUE_PATTERNS: dict[str, re.Pattern[str]] = {
    "deadline": re.compile(r"\b(?:ship\s+)?date\s+(?:is\s+)?(?P<v>.+)$", re.I),
    "owner": re.compile(r"\b(?:owner\s+is|owns?)\s+(?P<v>.+)$", re.I),
    "plan": re.compile(r"\bplan\s+(?:is\s+)?(?P<v>.+)$", re.I),
    "state": re.compile(r"\b(?:state|status)\s+(?:is\s+)?(?P<v>.+)$", re.I),
    "verdict": re.compile(r"\bverdict\s+(?:is\s+)?(?P<v>.+)$", re.I),
    "version": re.compile(r"\bversion\s+(?:is\s+)?(?P<v>.+)$", re.I),
}


def _slot(block: Optional[Mapping[str, Any]]) -> Optional[str]:
    """The block's canonical slot, or None. Delegates so there is ONE authority."""
    from .upsert_slots import UnknownSlot, slot_of

    try:
        return slot_of(block)
    except UnknownSlot:
        # An illegal slug is not this module's error to raise -- upsert_slots owns
        # that refusal on the write path. Here it simply means "no usable slot",
        # so the block is treated as free-form rather than crashing a read.
        return None


def extract_slots(block: Optional[Mapping[str, Any]]) -> dict[str, str]:
    """``{slot: value}`` for *block*, empty when it is free-form.

    Typed key -> value, as 1b specifies. One entry at most today, because a block
    carries one ``Slot``; the dict shape is the contract so a block carrying
    several later needs no signature change.
    """
    slot = _slot(block)
    if slot is None:
        return {}
    text = str((block or {}).get(STATEMENT_FIELD) or "").strip()
    pattern = _VALUE_PATTERNS.get(slot)
    if pattern is not None:
        m = pattern.search(text)
        if m:
            return {slot: m.group("v").strip().rstrip(".").strip()}
    # A slot with no recognisable value still records the slot: knowing WHICH
    # topic a fact is about is useful even when the value could not be parsed,
    # and returning {} here would make the block look free-form.
    return {slot: text}


def gist_of(block: Optional[Mapping[str, Any]]) -> str:
    """The statement minus its slot value and template boilerplate.

    What gets EMBEDDED. Two facts differing only in a slot value share this
    string, which is exactly the collision a full-statement embedding misses.

    Never mutates *block* and never writes back: the canonical statement is
    hashed by the evidence chain and must not move.
    """
    text = str((block or {}).get(STATEMENT_FIELD) or "")
    slots = extract_slots(block)
    for value in slots.values():
        if value and value != text:
            # Remove the VALUE only. Removing the key phrase too would collapse
            # "date is X" and "owner is X" into one gist -- facts about different
            # topics that merely share a value.
            text = text.replace(value, " ")
    words = [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _BOILERPLATE]
    return " ".join(words)


def is_slot_delta(
    incoming: Optional[Mapping[str, Any]], existing: Optional[Mapping[str, Any]]
) -> bool:
    """True when *incoming* asserts a NEW VALUE for the same fact as *existing*.

    The governed answer to "one flat string per record": rather than writing a
    second block, a caller seeing True should propose a slot-delta against the
    incumbent -- which routes through ``propose_update`` -> ``approve_apply`` and
    stays reversible.

    False for an identical restatement (nothing to supersede), for a different
    gist (a genuinely new fact), and for a free-form block (1b's escape hatch,
    same accepted cost as M4's: such facts accumulate).
    """
    a_slots, b_slots = extract_slots(incoming), extract_slots(existing)
    if not a_slots or not b_slots:
        return False
    if set(a_slots) != set(b_slots):
        return False
    if gist_of(incoming) != gist_of(existing):
        return False
    return a_slots != b_slots


__all__ = ["STATEMENT_FIELD", "extract_slots", "gist_of", "is_slot_delta"]
