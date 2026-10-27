"""A superseded memory never outranks the live memory that replaced it.

Reproduction (internal check 2026-10-08): after a correction moved the Ilona
call from Tuesday to Thursday, the superseded Tuesday block still came back
#1 in recall; only its status field marked it as old.
"""

from __future__ import annotations

from mind_mem._recall_core import _apply_post_filters


def _filters(hits, limit=10):
    return _apply_post_filters(hits, since=None, until=None, lifecycle=None, event_id=None, min_maturity=None, limit=limit)


def _hit(bid, score, status):
    return {"_id": bid, "score": score, "status": status, "excerpt": bid}


def test_superseded_ranks_below_its_replacement():
    hits = [_hit("D-old", 9.0, "superseded"), _hit("D-new", 4.0, "active")]
    assert [h["_id"] for h in _filters(hits)] == ["D-new", "D-old"]


def test_superseded_is_kept_not_dropped():
    out = _filters([_hit("D-old", 9.0, "superseded")])
    assert [h["_id"] for h in out] == ["D-old"]


def test_live_order_is_unchanged():
    hits = [_hit("A", 9.0, "active"), _hit("B", 8.0, "superseded"), _hit("C", 7.0, "active")]
    assert [h["_id"] for h in _filters(hits)] == ["A", "C", "B"]


def test_capitalised_status_field_is_honoured():
    hits = [{"_id": "D-old", "score": 9.0, "Status": "superseded"}, {"_id": "D-new", "score": 1.0, "Status": "active"}]
    assert [h["_id"] for h in _filters(hits)] == ["D-new", "D-old"]


def test_limit_keeps_the_live_replacement():
    hits = [_hit("D-old", 9.0, "superseded"), _hit("D-new", 4.0, "active")]
    assert [h["_id"] for h in _filters(hits, limit=1)] == ["D-new"]
