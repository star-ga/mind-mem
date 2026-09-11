"""M4 — enum-keyed upsert slots: prevent contradiction STRUCTURALLY.

ROADMAP M4, and the design constraints it is explicit about:

* File each fact under a topic slug drawn from a **closed set**, so a second
  statement on the same topic collides BY CONSTRUCTION -- an exact key collision,
  not a fuzzy similarity match that can miss.
* "The closed set is the load-bearing part: an open-ended topic string lets an
  extractor file `plan` on Monday and `plan_tier` on Friday, and the two
  contradicting facts never collide at all."
* "Where we must not copy them: their upsert is a silent overwrite with no
  proposal, no lineage, no rollback. Ours must route the supersession through
  propose_update -> approve_apply so the replacement is a RECORDED, REVERSIBLE
  event."
* Residual risk M4 insists be stated rather than hidden: the model still chooses
  the slug, so this moves the failure from "forgets what it wrote" to "picks the
  wrong enum member" -- narrower and VALIDATABLE, not eliminated.

Precedent is internal: 512-mind/src/drift.mind applies the same closed-set move
to meaning rather than keys. This is the house style, not a one-off.

Everything here is pure: no I/O, no clock. A slot decision must be replayable,
because it decides whether a governed write supersedes an existing fact.
"""

from __future__ import annotations

import pytest

from mind_mem.upsert_slots import (
    SLOTS,
    UnknownSlot,
    collides,
    normalise_slot,
    slot_of,
    supersession_target,
)


# --------------------------------------------------------------------------
# The closed set is the load-bearing part
# --------------------------------------------------------------------------

def test_the_slot_set_is_closed_and_non_empty():
    assert SLOTS, "a closed set with no members cannot key anything"
    assert all(s == s.lower() and " " not in s for s in SLOTS), SLOTS


def test_an_invented_slug_is_REFUSED_not_accepted():
    """The whole point. An open string lets `plan` and `plan_tier` coexist."""
    with pytest.raises(UnknownSlot):
        normalise_slot("plan_tier_v2_final")


def test_the_refusal_names_the_legal_members():
    """A caller that guessed wrong must be able to pick correctly."""
    with pytest.raises(UnknownSlot) as e:
        normalise_slot("nonsense")
    msg = str(e.value)
    assert any(s in msg for s in sorted(SLOTS)[:3]), msg


@pytest.mark.parametrize("raw", ["PLAN", " plan ", "Plan"])
def test_case_and_whitespace_normalise_rather_than_forking_the_key(raw):
    """`Plan` and `plan` must be ONE slot or the collision silently fails."""
    assert normalise_slot(raw) == normalise_slot("plan")


# --------------------------------------------------------------------------
# Collision by construction, not by similarity
# --------------------------------------------------------------------------

def test_two_facts_on_one_slot_collide_exactly():
    a = {"_id": "D-1", "Slot": "plan", "Statement": "we ship on Tuesday"}
    b = {"_id": "D-2", "Slot": "plan", "Statement": "we ship on Friday"}
    assert collides(a, b) is True


def test_different_slots_do_not_collide_however_similar_the_text():
    """Exactness cuts both ways, and that is the design.

    A fuzzy matcher would call these a contradiction; the slot key says they are
    facts about different topics, and the key is the authority.
    """
    a = {"_id": "D-1", "Slot": "plan", "Statement": "we ship on Tuesday"}
    b = {"_id": "D-2", "Slot": "owner", "Statement": "we ship on Tuesday"}
    assert collides(a, b) is False


def test_a_block_with_no_slot_never_collides():
    """M4's stated escape-hatch cost: free-form facts accumulate.

    Recorded as a test rather than a comment so nobody later reads the
    accumulation as a bug and 'fixes' it by inventing a slug.
    """
    a = {"_id": "D-1", "Statement": "a free-form observation"}
    b = {"_id": "D-2", "Statement": "a contradicting free-form observation"}
    assert collides(a, b) is False
    assert slot_of(a) is None


def test_collision_is_symmetric():
    a = {"_id": "D-1", "Slot": "plan"}
    b = {"_id": "D-2", "Slot": "plan"}
    assert collides(a, b) == collides(b, a)


def test_a_block_does_not_collide_with_itself():
    """An upsert must not propose superseding the very block it is updating."""
    a = {"_id": "D-1", "Slot": "plan"}
    assert collides(a, dict(a)) is False


# --------------------------------------------------------------------------
# The supersession is a GOVERNED event, never a silent overwrite
# --------------------------------------------------------------------------

def test_a_collision_yields_a_supersession_TARGET_not_a_mutation():
    """We must not copy the silent overwrite. This returns what to PROPOSE."""
    existing = [
        {"_id": "D-OLD", "Slot": "plan", "Statement": "we ship on Tuesday"},
        {"_id": "D-OTHER", "Slot": "owner", "Statement": "Ada owns it"},
    ]
    target = supersession_target(existing, slot="plan")
    assert target == "D-OLD", target


def test_no_incumbent_means_no_supersession_and_therefore_a_plain_write():
    assert supersession_target([], slot="plan") is None
    assert supersession_target([{"_id": "D-1", "Slot": "owner"}], slot="plan") is None


def test_the_newest_incumbent_is_the_one_superseded():
    """With two rows on one slot, the supersession must be unambiguous.

    Ambiguity here would silently pick one, and an arbitrary choice in a
    governed path is how a reversible event becomes an unexplainable one.
    """
    existing = [
        {"_id": "D-A", "Slot": "plan", "Date": "2026-01-01"},
        {"_id": "D-B", "Slot": "plan", "Date": "2026-06-01"},
    ]
    assert supersession_target(existing, slot="plan") == "D-B"


def test_an_unknown_slot_cannot_reach_the_supersession_path_at_all():
    """Fail closed: a bad slug must not silently behave like 'no collision'."""
    with pytest.raises(UnknownSlot):
        supersession_target([], slot="invented_slug")


# --------------------------------------------------------------------------
# The residual risk M4 requires be stated
# --------------------------------------------------------------------------

def test_the_module_states_its_residual_risk():
    """M4: 'Say so rather than claiming contradiction is solved.'

    A closed enum rejects an invented member; it cannot detect a fact filed
    under the WRONG legal member. That limit must be written down where a reader
    will find it, not left to this test.
    """
    import mind_mem.upsert_slots as m

    doc = (m.__doc__ or "").lower()
    assert "wrong" in doc and "slot" in doc, "the residual risk is not documented"
    assert "not eliminat" in doc or "narrower" in doc, doc[:200]


def test_the_module_is_pure():
    """A slot decision must replay identically; it gates a governed write."""
    import mind_mem.upsert_slots as m

    import ast

    # Assert on the IMPORT GRAPH, not on substrings. My first version searched the
    # raw text for "random" and matched the module's own docstring word
    # "randomness" -- the same prose-collision trap that has bitten source-
    # inspection tests in this repo four times. An AST walk cannot match prose.
    tree = ast.parse(open(m.__file__, encoding="utf-8").read())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    impure = imported & {"os", "time", "random", "datetime", "subprocess", "pathlib", "json"}
    assert not impure, f"upsert_slots must stay pure; it imports {sorted(impure)}"

    # And no call to a builtin that touches the world.
    called = {
        n.func.id for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    assert "open" not in called, "upsert_slots must not read or write files"


# --------------------------------------------------------------------------
# upsert_plan — the one call a governed writer makes
# --------------------------------------------------------------------------

def test_a_free_slot_plans_a_plain_create():
    from mind_mem.upsert_slots import upsert_plan

    assert upsert_plan([], slot="plan") == {"action": "create", "supersedes": None}


def test_an_occupied_slot_plans_a_governed_supersession():
    from mind_mem.upsert_slots import upsert_plan

    existing = [{"_id": "D-OLD", "Slot": "plan", "Statement": "ship Tuesday"}]
    assert upsert_plan(existing, slot="plan") == {
        "action": "supersede",
        "supersedes": "D-OLD",
    }


def test_the_plan_never_mutates_the_blocks_it_was_given():
    """It returns a PLAN. A mutating version would be the silent overwrite."""
    from mind_mem.upsert_slots import upsert_plan

    existing = [{"_id": "D-OLD", "Slot": "plan", "Statement": "ship Tuesday"}]
    before = [dict(b) for b in existing]
    upsert_plan(existing, slot="plan")
    assert existing == before, "upsert_plan modified its input"


def test_the_plan_fails_closed_on_an_invented_slug():
    """Degrading to 'create' would file a fact nothing can ever collide with."""
    from mind_mem.upsert_slots import UnknownSlot, upsert_plan

    with pytest.raises(UnknownSlot):
        upsert_plan([], slot="plan_v2")


def test_only_two_actions_exist_so_a_caller_can_be_exhaustive():
    """A third action added later must force every call site to decide."""
    from mind_mem.upsert_slots import upsert_plan

    seen = {
        upsert_plan([], slot="plan")["action"],
        upsert_plan([{"_id": "D-1", "Slot": "plan"}], slot="plan")["action"],
    }
    assert seen == {"create", "supersede"}
