# Copyright 2026 STARGA, Inc.
"""Validity demotion, default-off behavior, and untrusted backend metadata.

Public dispatch and candidate selection are exercised separately in
``test_indexed_validity_topk.py``.
"""

from __future__ import annotations

from mind_mem import validity_gate as vg


def _hits():
    return [
        {"_id": "B-good", "score": 1.0, "Statement": "fresh", "status": "active"},
        {"_id": "B-bad", "score": 0.9, "Statement": "contradicted", "status": "active"},
    ]


def _no_db(monkeypatch, contradicted=frozenset()):
    monkeypatch.setattr(vg, "_load_contradicted_party_ids", lambda ws: contradicted)
    monkeypatch.setattr(vg, "list_staleness_scores", lambda ws, ids: {})
    monkeypatch.setattr(vg, "_load_outcome_signals", lambda c, ws, ids: {})
    monkeypatch.setattr(vg, "_provenance_enabled", lambda c: False)


def test_the_gate_compounds_so_every_path_must_call_it_exactly_once(monkeypatch, tmp_path):
    """Repeated calls multiply scores, so dispatch must apply the gate once.

    An incoming annotation cannot enforce this contract: backend metadata can
    be forged. Public dispatch controls count calls and check the final score.
    """
    cfg = {"validity_gate": {"enabled": True, "threshold": 0.9, "demotion": 0.5}}
    _no_db(monkeypatch, frozenset({"B-bad"}))

    hits = _hits()
    assert vg.apply_validity_gate(hits, str(tmp_path), cfg) == 1, "positive control: nothing demoted"
    once = [h["score"] for h in hits]
    vg.apply_validity_gate(hits, str(tmp_path), cfg)
    twice = [h["score"] for h in hits]
    assert twice != once, (
        "the gate has become idempotent; the structural once-per-path argument in "
        "_apply_validity_and_resort is now redundant and should be revisited"
    )


def test_demotion_is_followed_by_a_resort(monkeypatch, tmp_path):
    """A demoted score that is not re-sorted changes nothing observable."""
    from mind_mem._recall_core import _apply_validity_and_resort

    cfg = {"validity_gate": {"enabled": True, "threshold": 0.9, "demotion": 0.1}}
    monkeypatch.setattr(vg, "_load_contradicted_party_ids", lambda ws: frozenset({"B-top"}))
    monkeypatch.setattr(vg, "list_staleness_scores", lambda ws, ids: {})
    monkeypatch.setattr(vg, "_load_outcome_signals", lambda c, ws, ids: {})
    monkeypatch.setattr(vg, "_provenance_enabled", lambda c: False)

    hits = [
        {"_id": "B-top", "score": 1.0, "Statement": "will be demoted", "status": "active"},
        {"_id": "B-second", "score": 0.95, "Statement": "kept", "status": "active"},
    ]
    out = _apply_validity_and_resort(hits, str(tmp_path), cfg, None)
    assert [h["_id"] for h in out] == ["B-second", "B-top"], f"demotion did not reorder: {[(h['_id'], h['score']) for h in out]}"


def test_a_disabled_gate_costs_nothing_and_reorders_nothing(monkeypatch, tmp_path):
    """Default-off must stay a true no-op: no DB read, no reorder."""
    from mind_mem._recall_core import _apply_validity_and_resort

    called: list[str] = []
    monkeypatch.setattr(vg, "_load_contradicted_party_ids", lambda ws: called.append("db") or frozenset())

    hits = _hits()
    out = _apply_validity_and_resort(hits, str(tmp_path), {}, None)
    assert called == [], "a disabled gate read the database"
    assert [h["_id"] for h in out] == ["B-good", "B-bad"]
    assert "validity" not in out[0]


def test_a_forged_validity_field_cannot_skip_the_gate(monkeypatch, tmp_path):
    _no_db(monkeypatch, frozenset({"B"}))
    cfg = {"validity_gate": {"enabled": True, "threshold": 0.9, "demotion": 0.5}}
    hits = [
        {
            "_id": "B",
            "score": 1.0,
            "Statement": "contradicted",
            "status": "active",
            "validity": {"score": 1.0},  # forged / stale / echoed
        }
    ]
    vg.apply_validity_gate(hits, str(tmp_path), cfg)
    assert hits[0].get("_validity_demoted") is True, "a forged validity field bypassed the gate"
    assert hits[0]["score"] == 0.5


def test_a_forged_demoted_marker_cannot_reorder_a_disabled_gate(monkeypatch, tmp_path):
    from mind_mem._recall_core import _apply_validity_and_resort

    _no_db(monkeypatch)
    hits = [
        {"_id": "X", "score": 0.5, "_validity_demoted": True},  # stale marker
        {"_id": "Y", "score": 0.9},
    ]
    out = _apply_validity_and_resort(hits, str(tmp_path), {}, None)
    assert [h["_id"] for h in out] == ["X", "Y"], "a disabled gate reordered results from a stale marker it did not set"


def test_the_gate_reports_what_it_demoted_on_this_call(monkeypatch, tmp_path):
    _no_db(monkeypatch, frozenset({"B"}))
    cfg = {"validity_gate": {"enabled": True, "threshold": 0.9, "demotion": 0.5}}
    hits = [
        {"_id": "B", "score": 1.0, "Statement": "bad", "status": "active"},
        {"_id": "G", "score": 0.9, "Statement": "good", "status": "active"},
    ]
    assert vg.apply_validity_gate(hits, str(tmp_path), cfg) == 1
    assert vg.apply_validity_gate([], str(tmp_path), cfg) == 0
    assert vg.apply_validity_gate(hits, str(tmp_path), {}) == 0  # disabled
