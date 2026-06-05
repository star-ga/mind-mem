"""recall() applies lifecycle/event_id/min_maturity on EVERY dispatch path.

Regression: the sqlite and vector early-returns applied only the date
filter, silently ignoring lifecycle/event_id/min_maturity (applied only in
the BM25-scan path). All paths now funnel through _apply_post_filters.
"""

from __future__ import annotations

from mind_mem._recall_core import _apply_post_filters


def _h(**kw):
    return {"_id": kw.get("id", "X"), **kw}


def test_event_id_filter_applied():
    hits = [_h(id="A", EventId="evt-1"), _h(id="B", EventId="evt-2")]
    out = _apply_post_filters(hits, since=None, until=None, lifecycle=None, event_id="evt-1", min_maturity=None, limit=10)
    assert [h["_id"] for h in out] == ["A"]


def test_min_maturity_filter_applied():
    hits = [_h(id="A", Maturity=0.9), _h(id="B", Maturity=0.2)]
    out = _apply_post_filters(hits, since=None, until=None, lifecycle=None, event_id=None, min_maturity=0.5, limit=10)
    ids = [h["_id"] for h in out]
    assert "A" in ids and "B" not in ids


def test_no_filters_is_identity():
    hits = [_h(id="A"), _h(id="B")]
    out = _apply_post_filters(hits, since=None, until=None, lifecycle=None, event_id=None, min_maturity=None, limit=10)
    assert out == hits
