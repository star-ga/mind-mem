"""Minting a new entity records a MERGE CANDIDATE. It never merges.

ROADMAP, two partials whose remaining halves meet here:
  * "Blocking + LLM-arbitration hybrid" — "**Remaining:** the LLM arbitration within a
    block, and wiring blocking into the `capture.py` merge path."
  * "Description-grounded entity resolution" — "**Remaining:** ... and wiring the guard
    into the `capture.py` merge path."

**BOTH POINTERS ARE WRONG, and that is worth recording rather than quietly working
around.** `capture.py` has no merge path — no merge function, no entity resolution, no
`EntityRegistry` reference. The real door is `EntityRegistry.resolve`, which canonicalises
a surface form and CREATES the entity when it is new; that create is the moment two
spellings of one person become two entities, and it is the only place a candidate can be
noticed.

**NOTHING IS MERGED, BY CONSTRUCTION.** `resolve` keeps its exact behaviour: a new surface
still gets its own entity id, every existing caller sees the same return value, and the
candidate is RECORDED for review. That ordering is the whole safety property — an
auto-merge on a blocking hit would fuse "Ada Lovelace" and "Alan Lovelace" on a shared
token, and unmerging is the operation this store cannot offer.

The candidate carries the `merge_guard` verdict, so a reviewer sees WHY a pair was
proposed and whether the guard objected. The guard never authorises a merge either, so
the queue is advisory at both ends.

Flag-gated OFF (`v4.merge_candidates`), probed silently: this writes rows, and a feature
that starts writing to an operator's DB because they upgraded is not additive.
"""

from __future__ import annotations

import sqlite3
import threading

import pytest

from mind_mem.entity_merge_candidates import (
    MERGE_CANDIDATES_FLAG,
    list_candidates,
    note_candidates_for,
)


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    from mind_mem.entity_merge_candidates import SCHEMA

    c.executescript(SCHEMA)
    return c


def test_a_new_name_sharing_a_token_records_a_candidate(conn):
    existing = ["ada lovelace", "charles babbage"]
    noted = note_candidates_for(conn, threading.RLock(), "ada byron lovelace", existing)
    assert noted, "no candidate recorded for a name sharing two tokens"
    rows = list_candidates(conn)
    assert any(r["candidate"] == "ada lovelace" for r in rows), rows


def test_an_unrelated_name_records_NOTHING(conn):
    """POSITIVE CONTROL. A recorder that queued every new name would satisfy the test
    above and hand the reviewer the whole corpus, which is the cost blocking exists to
    avoid."""
    noted = note_candidates_for(conn, threading.RLock(), "grace hopper", ["ada lovelace"])
    assert noted == [], noted
    assert list_candidates(conn) == []


def test_the_candidate_carries_the_GUARD_VERDICT(conn):
    """A reviewer needs to see whether the over-merge guard objected, not just that two
    names share a token. Without the verdict the queue is a list of coincidences."""
    note_candidates_for(conn, threading.RLock(), "ada byron lovelace", ["ada lovelace"])
    row = list_candidates(conn)[0]
    assert row["verdict"], row
    assert row["reason"], row


def test_recording_is_idempotent(conn):
    """One pair is one fact. A second row would double-count it in any review queue, and
    a queue that shows the same pair twice trains a reviewer to skim."""
    lock = threading.RLock()
    note_candidates_for(conn, lock, "ada byron lovelace", ["ada lovelace"])
    note_candidates_for(conn, lock, "ada byron lovelace", ["ada lovelace"])
    assert len(list_candidates(conn)) == 1


def test_a_name_is_never_its_own_candidate(conn):
    note_candidates_for(conn, threading.RLock(), "ada lovelace", ["ada lovelace"])
    assert list_candidates(conn) == []


def test_the_flag_is_declared():
    from mind_mem.v4.feature_flags import ALL_V4_FLAGS

    assert MERGE_CANDIDATES_FLAG in ALL_V4_FLAGS


def test_nothing_in_the_module_can_merge_or_delete():
    """THE SAFETY PROPERTY, asserted over the source rather than promised. Unmerging is
    the operation this store cannot offer, so the recorder must be structurally incapable
    of merging."""
    import inspect

    import mind_mem.entity_merge_candidates as m

    source = inspect.getsource(m)
    for forbidden in ("UPDATE entities", "DELETE FROM entities", "UPDATE aliases",
                      "DELETE FROM aliases"):
        assert forbidden not in source, forbidden


def test_the_forbidden_sql_check_is_not_vacuous():
    """POSITIVE CONTROL for the check above: the module must contain SQL at all, or the
    absence of those statements says nothing."""
    import inspect

    import mind_mem.entity_merge_candidates as m

    assert "INSERT" in inspect.getsource(m)


# ---------------------------------------------------------------------------
# Wiring: EntityRegistry.resolve is the door. A recorder nothing calls is a queue
# that stays empty while the duplicates accumulate.
# ---------------------------------------------------------------------------


def _registry(tmp_path, monkeypatch, *, flag_on: bool):
    import mind_mem.entity_merge_candidates as emc
    from mind_mem.knowledge_graph import KnowledgeGraph

    monkeypatch.setattr(emc, "_flag_on", lambda: flag_on)
    # A KnowledgeGraph takes a DB FILE path, not a directory — it makedirs the parent.
    return KnowledgeGraph(str(tmp_path / "kg.db")).entities


def test_resolve_STILL_returns_a_new_id_for_a_new_surface(tmp_path, monkeypatch):
    """Behaviour is unchanged — this is what makes the change non-breaking. A blocking
    hit must NOT redirect the caller to the existing entity, because that IS the silent
    auto-merge."""
    reg = _registry(tmp_path, monkeypatch, flag_on=True)
    first = reg.resolve("Ada Lovelace")
    second = reg.resolve("Ada Byron Lovelace")
    assert first != second, "resolve silently merged two surfaces into one entity"


def test_resolve_RECORDS_the_candidate_when_the_flag_is_on(tmp_path, monkeypatch):
    reg = _registry(tmp_path, monkeypatch, flag_on=True)
    reg.resolve("Ada Lovelace")
    reg.resolve("Ada Byron Lovelace")
    from mind_mem.entity_merge_candidates import list_candidates

    rows = list_candidates(reg._conn)
    assert rows, "the door recorded no candidate"


def test_resolve_records_NOTHING_when_the_flag_is_off(tmp_path, monkeypatch):
    """Flag-off must be indistinguishable from a build without the feature — including
    on disk. A feature that starts writing rows because someone upgraded is not
    additive."""
    reg = _registry(tmp_path, monkeypatch, flag_on=False)
    reg.resolve("Ada Lovelace")
    reg.resolve("Ada Byron Lovelace")
    tables = {
        r[0]
        for r in reg._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert "entity_merge_candidates" not in tables, tables


def test_a_failing_recorder_never_breaks_resolve(tmp_path, monkeypatch):
    """`resolve` is on the ingestion path. A candidate queue is a convenience; losing an
    entity write because the convenience failed would be a bad trade."""
    import mind_mem.entity_merge_candidates as emc

    monkeypatch.setattr(emc, "_flag_on", lambda: True)

    def _boom(*args, **kwargs):
        raise RuntimeError("queue unavailable")

    monkeypatch.setattr(emc, "note_candidates_for", _boom)
    from mind_mem.knowledge_graph import KnowledgeGraph

    reg = KnowledgeGraph(str(tmp_path / "kg.db")).entities
    assert reg.resolve("Ada Lovelace")  # must not raise
