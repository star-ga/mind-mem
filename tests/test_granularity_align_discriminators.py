"""A merge is never proposed between memories that differ in who, when or time zone.

Reproduction (internal check 2026-10-08): "Ilona call Tue 10am Arizona" and
"contractor call Tue 10am Pacific" scored 0.82 and were proposed as a
keep_longer merge, which would have dropped the Ilona memory.
"""

from __future__ import annotations

from mind_mem.granularity_align import find_merge_candidates

ILONA = "Azariel's weekly planning call with Ilona is on Tuesday at 10am Arizona time."
CONTRACTOR = "Azariel's weekly planning call with the engineering contractor is on Tuesday at 10am Pacific time."


def _blk(block_id: str, content: str) -> dict:
    return {"_id": block_id, "content": content, "tags": "schedule, azariel", "Status": "active"}


def _pairs(blocks, threshold=0.5):
    return {(c.to_dict()["id_a"], c.to_dict()["id_b"]) for c in find_merge_candidates(blocks, min_similarity=threshold)}


def test_reported_false_merge_is_not_proposed():
    assert _pairs([_blk("D-1", ILONA), _blk("D-3", CONTRACTOR)]) == set()


def test_different_person_is_not_merged():
    a = _blk("A", "Weekly planning call with Ilona on Tuesday at 10am Arizona time.")
    b = _blk("B", "Weekly planning call with Marcus on Tuesday at 10am Arizona time.")
    assert _pairs([a, b]) == set()


def test_different_time_is_not_merged():
    a = _blk("A", "Weekly planning call with Ilona on Tuesday at 10am Arizona time.")
    b = _blk("B", "Weekly planning call with Ilona on Tuesday at 2pm Arizona time.")
    assert _pairs([a, b]) == set()


def test_different_day_is_not_merged():
    a = _blk("A", "Weekly planning call with Ilona on Tuesday at 10am Arizona time.")
    b = _blk("B", "Weekly planning call with Ilona on Thursday at 10am Arizona time.")
    assert _pairs([a, b]) == set()


def test_different_date_is_not_merged():
    a = _blk("A", "Board review moved to 2026-10-15 per the CFO.")
    b = _blk("B", "Board review moved to 2026-10-22 per the CFO.")
    assert _pairs([a, b]) == set()


def test_different_time_zone_is_not_merged():
    a = _blk("A", "Weekly planning call with Ilona on Tuesday at 10am Arizona time.")
    b = _blk("B", "Weekly planning call with Ilona on Tuesday at 10am Pacific time.")
    assert _pairs([a, b]) == set()


def test_exact_duplicate_is_still_proposed():
    # Positive control: the same statement up to spacing and a trailing full stop -> still a candidate.
    a = _blk("A", "Weekly planning call with Ilona on Tuesday at 10am Arizona time.")
    b = _blk("B", "Weekly planning call with Ilona on Tuesday at  10am Arizona time")
    assert _pairs([a, b]) == {("A", "B")}


def test_reworded_lookalike_is_similar_not_merge():
    from mind_mem.granularity_align import find_similar_pairs

    a = _blk("A", "Weekly planning call with Ilona on Tuesday at 10am Arizona time.")
    b = _blk("B", "Weekly planning call with Ilona, Tuesday 10am Arizona time, recurring.")
    assert _pairs([a, b]) == set()
    assert [(p.id_a, p.id_b) for p in find_similar_pairs([a, b], min_similarity=0.5)] == [("A", "B")]
