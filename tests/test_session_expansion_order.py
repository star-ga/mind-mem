"""Chosen sessions come out in session order, not in the corpus's alphabetical key order.

A LoCoMo-style corpus keyed DIA-D1..DIA-D12 iterates D1, D10, D11, D12, D2, ... when the keys
are sorted as strings. The transcript the model reads must still run session 1, 2, ..., 12, and
within a session turn 9 must come before turn 10.
"""

from __future__ import annotations

from mind_mem.session_expansion import expand_sessions


def _corpus() -> dict[str, dict]:
    blocks = {}
    for s in range(1, 13):
        for t in (1, 2, 9, 10):
            bid = f"DIA-D{s}:{t}"
            blocks[bid] = {
                "_id": bid,
                "Statement": f"session {s} turn {t} talks about the garden",
                "Tags": f"session-{s}",
                "DiaID": f"D{s}:{t}",
            }
    return dict(sorted(blocks.items()))  # alphabetical, as a sorted corpus file yields


def test_sessions_and_turns_are_in_order() -> None:
    corpus = _corpus()
    assert list(corpus)[:3] == ["DIA-D10:1", "DIA-D10:10", "DIA-D10:2"]  # positive control: the input really is scrambled
    items = expand_sessions("garden", [], corpus, budget_chars=10**6)
    sessions = []
    for it in items:
        if not sessions or sessions[-1] != it["session"]:
            sessions.append(it["session"])
    assert sessions == [f"T:{s}" for s in range(1, 13)]
    turns_s2 = [it["DiaID"] for it in items if it["session"] == "T:2"]
    assert turns_s2 == ["D2:1", "D2:2", "D2:9", "D2:10"]


def test_dated_sessions_are_chronological() -> None:
    corpus = {
        "a": {"_id": "a", "Statement": "garden one", "Date": "2023-05-08"},
        "b": {"_id": "b", "Statement": "garden two", "Date": "2023-04-30"},
        "c": {"_id": "c", "Statement": "garden three", "Date": "2023-05-10"},
    }
    items = expand_sessions("garden", [], corpus, budget_chars=10**6)
    assert [it["_id"] for it in items] == ["b", "a", "c"]
