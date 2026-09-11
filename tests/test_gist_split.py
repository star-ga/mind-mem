"""Item 1b — the gist/slot split: what gets EMBEDDED vs what FILTERS.

ROADMAP 1b: "Keep the canonical `statement` untouched (evidence chain and MIC
preimages must not move); add versioned `slots` (typed key->value) and `gist`
(statement minus slot values and template boilerplate). **The gist is embedded;
the slots are exact-match filters.** A proposal whose gist matches an existing
block's offers a slot-delta instead of a new block ... Absorbs Group M's
enum-keyed upsert slots: building those without the embed split wires slots into
writes while leaving READS broken."

M4 landed the write half (upsert_slots.py). This is the read half, and the reason
it is needed is concrete: two records that differ only in a slot VALUE --
"ship date is Tuesday" and "ship date is Friday" -- are near-identical as text,
so an embedding of the full statement scores them as the same fact and a
similarity search cannot separate them. Strip the slot value and they become the
SAME gist, which is the signal: identical gist plus differing slot value is a
slot-delta, not a new fact.

THE NON-NEGOTIABLE: the canonical statement must not move. The evidence chain and
MIC preimages hash it, so a function that rewrote it would silently invalidate
every existing receipt. Everything here DERIVES; nothing mutates.

Pure: no clock, no I/O. A gist decides whether a write supersedes, so it must
replay identically.
"""

from __future__ import annotations

import pytest

from mind_mem.gist import extract_slots, gist_of, is_slot_delta


# --------------------------------------------------------------------------
# The statement is never touched
# --------------------------------------------------------------------------

def test_the_canonical_statement_is_never_modified():
    """Non-negotiable: the evidence chain and MIC preimages hash it."""
    block = {"_id": "D-1", "Statement": "Ship date is Tuesday", "Slot": "deadline"}
    before = dict(block)
    gist_of(block)
    extract_slots(block)
    assert block == before, "a derivation mutated the block it read"


def test_the_gist_is_derived_not_stored_over_the_statement():
    block = {"_id": "D-1", "Statement": "Ship date is Tuesday", "Slot": "deadline"}
    g = gist_of(block)
    assert g != block["Statement"], "the gist must differ from the raw statement"
    assert block["Statement"] == "Ship date is Tuesday"


# --------------------------------------------------------------------------
# The read half: two records differing only in a slot value share one gist
# --------------------------------------------------------------------------

def test_two_facts_differing_only_in_a_slot_value_share_a_gist():
    """THE point. An embedding of the full statement cannot separate these."""
    a = {"_id": "D-1", "Statement": "Ship date is Tuesday", "Slot": "deadline"}
    b = {"_id": "D-2", "Statement": "Ship date is Friday", "Slot": "deadline"}
    assert gist_of(a) == gist_of(b), (gist_of(a), gist_of(b))


def test_facts_about_genuinely_different_things_do_not_share_a_gist():
    """POSITIVE CONTROL. A gist that collapsed everything would be useless.

    Without this, a gist_of() returning "" would pass the test above.
    """
    a = {"_id": "D-1", "Statement": "Ship date is Tuesday", "Slot": "deadline"}
    b = {"_id": "D-2", "Statement": "Ada owns the compiler", "Slot": "owner"}
    assert gist_of(a) != gist_of(b)


def test_the_gist_is_not_empty_for_a_real_statement():
    """An empty gist matches every other empty gist — a silent collapse."""
    assert gist_of({"Statement": "Ship date is Tuesday", "Slot": "deadline"}).strip()


def test_boilerplate_does_not_fork_the_gist():
    """Template wording differences must not make one fact look like two."""
    a = {"_id": "D-1", "Statement": "The ship date is Tuesday.", "Slot": "deadline"}
    b = {"_id": "D-2", "Statement": "ship date is  friday", "Slot": "deadline"}
    assert gist_of(a) == gist_of(b), (gist_of(a), gist_of(b))


# --------------------------------------------------------------------------
# slot-delta: the governed answer to "one flat string per record"
# --------------------------------------------------------------------------

def test_same_gist_plus_different_slot_value_is_a_slot_delta():
    a = {"_id": "D-1", "Statement": "Ship date is Tuesday", "Slot": "deadline"}
    b = {"_id": "D-2", "Statement": "Ship date is Friday", "Slot": "deadline"}
    assert is_slot_delta(a, b) is True


def test_an_identical_restatement_is_NOT_a_delta():
    """Nothing changed, so there is nothing to supersede."""
    a = {"_id": "D-1", "Statement": "Ship date is Tuesday", "Slot": "deadline"}
    b = {"_id": "D-2", "Statement": "Ship date is Tuesday", "Slot": "deadline"}
    assert is_slot_delta(a, b) is False


def test_a_different_gist_is_a_new_fact_not_a_delta():
    a = {"_id": "D-1", "Statement": "Ship date is Tuesday", "Slot": "deadline"}
    b = {"_id": "D-2", "Statement": "Ada owns the compiler", "Slot": "owner"}
    assert is_slot_delta(a, b) is False


def test_a_slotless_block_is_never_a_delta():
    """1b's escape hatch, same as M4's: free-form facts accumulate."""
    a = {"_id": "D-1", "Statement": "Ship date is Tuesday"}
    b = {"_id": "D-2", "Statement": "Ship date is Friday"}
    assert is_slot_delta(a, b) is False


def test_slot_extraction_is_typed_key_to_value():
    got = extract_slots({"_id": "D-1", "Statement": "Ship date is Tuesday", "Slot": "deadline"})
    assert isinstance(got, dict)
    assert got.get("deadline"), got
    for k, v in got.items():
        assert isinstance(k, str) and isinstance(v, str), (k, v)


def test_the_module_is_pure():
    import ast

    import mind_mem.gist as m

    tree = ast.parse(open(m.__file__, encoding="utf-8").read())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    impure = imported & {"os", "time", "random", "datetime", "subprocess", "pathlib", "json"}
    assert not impure, f"gist must stay pure; it imports {sorted(impure)}"


def test_the_read_half_and_the_write_half_agree():
    """1b and M4 must not disagree, or a caller gets contradictory advice.

    upsert_slots.collides answers "must this collide?" on the write path;
    gist.is_slot_delta answers "is this a new value for the same fact?" on the
    read path. A pair that collides and shares a gist must be a delta, and a pair
    on different slots must be neither. A read half that disagreed with the write
    half would be worse than having neither.
    """
    from mind_mem.upsert_slots import collides, upsert_plan

    a = {"_id": "D-1", "Statement": "Ship date is Tuesday", "Slot": "deadline"}
    b = {"_id": "D-2", "Statement": "Ship date is Friday", "Slot": "deadline"}
    c = {"_id": "D-3", "Statement": "Ada owns the compiler", "Slot": "owner"}

    assert collides(a, b) is True and is_slot_delta(a, b) is True
    assert collides(a, c) is False and is_slot_delta(a, c) is False

    # And the governed plan names the incumbent the delta would supersede.
    assert upsert_plan([a], slot="deadline") == {"action": "supersede", "supersedes": "D-1"}
