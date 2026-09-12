"""Deterministic blocking for entity resolution — keep the model off easy cases.

ROADMAP ("Blocking + LLM-arbitration hybrid"): "cheap deterministic blocking
(inverted index on name tokens / embedding neighbors) narrows candidates to
50-100-item blocks; the LLM only arbitrates *within* a block. Keeps resolution
sublinear and keeps the expensive model call off the easy cases (typos, casing) that
deterministic logic already handles."

This builds the DETERMINISTIC half only, and that is the point of the split: an
arbitration layer is a model call, but blocking is an inverted index, and shipping
the cheap layer first means the expensive one is never asked about a casing
difference.

The companion item names two enforced failure modes, both pinned below:
  * "unmatched name -> single-element cluster (never silently dropped)"
  * "over-merge guarded by description mismatch + HITL review"

The first is the load-bearing one here: a name that matches nothing must come back
as a cluster of one, not vanish. A resolver that silently drops what it cannot match
loses entities with no signal at all -- and losing one quietly is worse than
refusing to resolve it.

Pure: no model, no clock, no I/O.
"""

from __future__ import annotations

import pytest

from mind_mem.entity_blocking import (
    MAX_BLOCK_SIZE,
    block_for,
    build_blocks,
    name_tokens,
)


# --------------------------------------------------------------------------
# Tokenisation: what makes two surface forms land together
# --------------------------------------------------------------------------

def test_casing_does_not_separate_two_forms():
    assert name_tokens("Ada Lovelace") == name_tokens("ada lovelace")


def test_punctuation_and_spacing_do_not_separate_them():
    assert name_tokens("O'Brien, Ada") == name_tokens("ada o brien")


def test_distinct_names_produce_distinct_tokens():
    """POSITIVE CONTROL: a tokeniser returning a constant would pass the above."""
    assert name_tokens("Ada Lovelace") != name_tokens("Alan Turing")


def test_an_empty_name_yields_no_tokens():
    assert name_tokens("") == frozenset()


# --------------------------------------------------------------------------
# Blocking: candidates that share a token
# --------------------------------------------------------------------------

def test_surface_variants_land_in_one_block():
    names = ["Ada Lovelace", "ada lovelace", "Lovelace, Ada", "Alan Turing"]
    blocks = build_blocks(names)
    ada = block_for(blocks, "Ada Lovelace")
    assert set(ada) == {"Ada Lovelace", "ada lovelace", "Lovelace, Ada"}, ada


def test_an_unrelated_name_is_not_dragged_in():
    blocks = build_blocks(["Ada Lovelace", "Alan Turing"])
    assert block_for(blocks, "Ada Lovelace") == ["Ada Lovelace"]


def test_an_UNMATCHED_name_is_a_single_element_cluster_never_dropped():
    """The item's first enforced failure mode.

    A resolver that silently drops what it cannot match loses entities with no
    signal at all, which is worse than refusing to resolve them.
    """
    blocks = build_blocks(["Ada Lovelace", "Zzyzx Nobody"])
    got = block_for(blocks, "Zzyzx Nobody")
    assert got == ["Zzyzx Nobody"], got


def test_every_input_appears_in_some_block():
    """The same property, asserted over the whole set rather than one case."""
    names = ["Ada Lovelace", "ada lovelace", "Alan Turing", "Zzyzx Nobody", ""]
    blocks = build_blocks(names)
    covered = {n for b in blocks.values() for n in b}
    assert covered == {n for n in names if n.strip()}, covered


def test_blocks_are_deterministic_and_ordered():
    """Two runs must agree, or a reviewer cannot tell a change from a reshuffle."""
    names = ["Ada Lovelace", "ada lovelace", "Alan Turing"]
    assert build_blocks(names) == build_blocks(list(names))
    assert block_for(build_blocks(names), "Ada Lovelace") == sorted(
        block_for(build_blocks(names), "Ada Lovelace")
    )


# --------------------------------------------------------------------------
# The bound that keeps the model call cheap
# --------------------------------------------------------------------------

def test_a_block_is_capped():
    """"narrows candidates to 50-100-item blocks" — an unbounded block hands the
    arbitration layer the whole corpus, which is the cost this exists to avoid."""
    names = [f"Ada Variant{i}" for i in range(400)]
    blocks = build_blocks(names)
    for members in blocks.values():
        assert len(members) <= MAX_BLOCK_SIZE, len(members)


def test_the_cap_does_not_drop_names_from_coverage():
    """Truncating a block must not lose an entity — it must place it elsewhere.

    Without this, the cap would silently reintroduce the dropped-entity failure the
    single-element-cluster rule exists to prevent.
    """
    names = [f"Ada Variant{i}" for i in range(400)]
    blocks = build_blocks(names)
    covered = {n for b in blocks.values() for n in b}
    assert covered == set(names), len(set(names) - covered)
