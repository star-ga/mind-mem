# Copyright 2026 STARGA, Inc.
"""Both ends of an edge's validity window are read, not just one.

``valid_from`` was validated on write and never read, so an edge dated in
the future was served today and the ``as_of`` replay in
``edge_grounded_answer`` honoured only half its own window. These tests
pin the corrected contract:

* a NULL bound is *unbounded on that side* — the shape of every edge in a
  store written before the column had a reader, and the single most
  likely way a patch release could silently empty a real workspace;
* a future ``valid_from`` withholds the edge **and the edge is still
  there**: every exclusion assertion below is preceded by a direct
  ``SELECT`` proving the row exists, because ``assert x not in results``
  passes trivially against a store where the insert never happened;
* the exclusion is a round trip, not a one-way door — the same row comes
  back once the moment advances past ``valid_from``;
* a malformed bound is not live in either direction, and
  ``include_expired=True`` still returns absolutely everything.

Rows are inserted with raw SQL rather than through ``add_edge`` on
purpose: the fixture must be independent of the read path it is used to
test, and a direct ``SELECT COUNT(*)`` is then an honest positive
control rather than a second use of the same filter.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pytest

from mind_mem import knowledge_graph as kg_mod
from mind_mem.edge_grounded_answer import build_context
from mind_mem.knowledge_graph import KnowledgeGraph, Predicate, _is_live

_T0 = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


def _iso(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def _insert(
    kg: KnowledgeGraph,
    subject: str,
    object_: str,
    source_block_id: str,
    *,
    valid_from: str | None = None,
    valid_until: str | None = None,
    predicate: Predicate = Predicate.DEPENDS_ON,
) -> None:
    """Put a row in ``edges`` directly, bypassing the read path entirely."""
    s_id = kg.entities.resolve(subject)
    o_id = kg.entities.resolve(object_)
    kg._conn.execute(
        "INSERT OR REPLACE INTO edges (subject, predicate, object, source_block_id, "
        "confidence, valid_from, valid_until, metadata) VALUES (?, ?, ?, ?, ?, ?, ?, '{}')",
        (s_id, predicate.value, o_id, source_block_id, 1.0, valid_from, valid_until),
    )
    kg._conn.commit()


def _row_count(kg: KnowledgeGraph) -> int:
    return int(kg._conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0])


class _FrozenClock(datetime):
    """A ``datetime`` whose ``now()`` is whatever the test last set."""

    _frozen: datetime = _T0

    @classmethod
    def now(cls, tz=None):  # type: ignore[override]
        return cls._frozen


@pytest.fixture()
def graph(tmp_path):
    kg = KnowledgeGraph(os.path.join(str(tmp_path), "memory", "knowledge_graph.db"))
    try:
        yield kg
    finally:
        kg.close()


@pytest.fixture()
def frozen(monkeypatch):
    """Pin ``knowledge_graph``'s wall clock so the query path is testable.

    ``_query_edges`` reads ``datetime.now(timezone.utc)``; freezing the
    module's ``datetime`` is what makes "advance the moment past
    ``valid_from``" a deterministic assertion instead of a sleep.
    """

    def _set(moment: datetime) -> None:
        _FrozenClock._frozen = moment

    _set(_T0)
    monkeypatch.setattr(kg_mod, "datetime", _FrozenClock)
    return _set


# ---------------------------------------------------------------------------
# _is_live — the semantics, in one place
# ---------------------------------------------------------------------------


def test_null_interval_is_live_at_every_moment():
    """NULL means unbounded, NOT "not yet valid" — the regression case."""
    for moment in (_T0 - timedelta(days=10_000), _T0, _T0 + timedelta(days=10_000)):
        assert _is_live(None, None, moment) is True


def test_null_lower_bound_is_no_lower_bound():
    assert _is_live(None, _iso(_T0 + timedelta(days=1)), _T0) is True
    assert _is_live(None, _iso(_T0 - timedelta(days=1)), _T0) is False


def test_null_upper_bound_is_no_upper_bound():
    assert _is_live(_iso(_T0 - timedelta(days=1)), None, _T0) is True
    assert _is_live(_iso(_T0 + timedelta(days=1)), None, _T0) is False


def test_both_bounds_are_inclusive():
    stamp = _iso(_T0)
    assert _is_live(stamp, None, _T0) is True
    assert _is_live(None, stamp, _T0) is True
    assert _is_live(stamp, stamp, _T0) is True


def test_malformed_bound_is_not_live_in_either_direction():
    assert _is_live("not-a-timestamp", None, _T0) is False
    assert _is_live(None, "not-a-timestamp", _T0) is False


# ---------------------------------------------------------------------------
# _query_edges — the defect the ruling measured
# ---------------------------------------------------------------------------


def test_future_dated_edge_exists_but_is_not_served(graph, frozen):
    """The 2030 edge: present in the store, absent from the answer."""
    _insert(graph, "alice", "atlas", "B-past", valid_from=_iso(_T0 - timedelta(days=1)))
    _insert(graph, "alice", "orion", "B-future", valid_from=_iso(_T0 + timedelta(days=1)))

    # Positive control: BOTH rows are really there. Without this the
    # exclusion assertion below would pass against an empty table.
    assert _row_count(graph) == 2
    stored = {r[0] for r in graph._conn.execute("SELECT source_block_id FROM edges")}
    assert stored == {"B-past", "B-future"}

    served = {e.source_block_id for e in graph.edges_from("alice")}
    assert "B-past" in served, "an already-open window must still be served"
    assert "B-future" not in served, "an edge whose window has not opened was served"


def test_future_dated_edge_returns_once_the_moment_advances(graph, frozen):
    """Round trip: withheld at T0, served at T0 + 2 days. Same row."""
    _insert(graph, "alice", "orion", "B-future", valid_from=_iso(_T0 + timedelta(days=1)))
    assert _row_count(graph) == 1

    assert [e.source_block_id for e in graph.edges_from("alice")] == []

    frozen(_T0 + timedelta(days=2))
    assert [e.source_block_id for e in graph.edges_from("alice")] == ["B-future"]

    # ...and back again, so the door swings both ways rather than the
    # test having simply observed a store that filled up.
    frozen(_T0)
    assert [e.source_block_id for e in graph.edges_from("alice")] == []


def test_null_interval_edge_is_served_unchanged(graph, frozen):
    """The blast-radius test: an all-NULL store must not lose a single edge.

    Every edge in the live store carries a NULL interval. If reading
    ``valid_from`` withheld any of them, this fix would be a silent
    outage dressed as a patch.
    """
    for i in range(5):
        _insert(graph, "alice", f"node{i}", f"B-null-{i}")
    assert _row_count(graph) == 5
    assert len(graph.edges_from("alice")) == 5

    # Not merely "now": a NULL lower bound must hold at any moment.
    frozen(_T0 - timedelta(days=3650))
    assert len(graph.edges_from("alice")) == 5


def test_expired_upper_bound_still_withheld(graph, frozen):
    """The bound that already worked keeps working."""
    _insert(graph, "alice", "gone", "B-expired", valid_until=_iso(_T0 - timedelta(days=1)))
    _insert(graph, "alice", "here", "B-open", valid_until=_iso(_T0 + timedelta(days=1)))
    assert _row_count(graph) == 2
    assert [e.source_block_id for e in graph.edges_from("alice")] == ["B-open"]


def test_malformed_valid_from_is_withheld(graph, frozen):
    _insert(graph, "alice", "junk", "B-bad", valid_from="whenever")
    assert _row_count(graph) == 1
    assert graph.edges_from("alice") == []


def test_include_expired_bypasses_both_bounds(graph, frozen):
    """No capability lost: the escape hatch still returns everything."""
    _insert(graph, "alice", "orion", "B-future", valid_from=_iso(_T0 + timedelta(days=1)))
    _insert(graph, "alice", "gone", "B-expired", valid_until=_iso(_T0 - timedelta(days=1)))
    _insert(graph, "alice", "junk", "B-bad", valid_from="whenever")
    _insert(graph, "alice", "here", "B-null")
    assert _row_count(graph) == 4

    served = {e.source_block_id for e in graph.edges_from("alice", include_expired=True)}
    assert served == {"B-future", "B-expired", "B-bad", "B-null"}


def test_edges_of_and_neighbors_honour_the_lower_bound(graph, frozen):
    """The other two read doors go through the same filter."""
    _insert(graph, "alice", "orion", "B-future", valid_from=_iso(_T0 + timedelta(days=1)))
    _insert(graph, "alice", "atlas", "B-open")
    assert _row_count(graph) == 2

    assert {e.source_block_id for e in graph.edges_of("alice")} == {"B-open"}
    assert [n["entity"] for n in graph.neighbors("alice")] == [graph.entities.resolve("atlas")]

    frozen(_T0 + timedelta(days=2))
    assert {e.source_block_id for e in graph.edges_of("alice")} == {"B-open", "B-future"}


# ---------------------------------------------------------------------------
# as_of replay — the docstring claim, now true for the valid-time axis
# ---------------------------------------------------------------------------


def test_as_of_replay_honours_the_lower_bound(graph):
    """A point-in-time answer must not contain a claim from its future.

    No clock freezing here: ``build_context`` takes ``as_of`` explicitly,
    which is the whole point of the replay surface.
    """
    _insert(graph, "alice", "orion", "B-future", valid_from=_iso(_T0 + timedelta(days=1)))
    _insert(graph, "alice", "atlas", "B-open", valid_from=_iso(_T0 - timedelta(days=1)))
    assert _row_count(graph) == 2

    before = build_context(graph, "alice", hops=1, as_of=_iso(_T0))
    served_before = {t.source_block_id for t in before.triples}
    assert "B-open" in served_before
    assert "B-future" not in served_before

    after = build_context(graph, "alice", hops=1, as_of=_iso(_T0 + timedelta(days=2)))
    served_after = {t.source_block_id for t in after.triples}
    assert served_after == {"B-open", "B-future"}
