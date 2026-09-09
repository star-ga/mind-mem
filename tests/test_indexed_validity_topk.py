# Copyright 2026 STARGA, Inc.
"""Top-k controls for validity on early indexed recall legs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mind_mem import _recall_core
from mind_mem import validity_gate as vg
from mind_mem.hybrid_recall import RecallResults
from mind_mem.sqlite_index import build_index

TOP_ID = "D-20260909-001"
NEXT_ID = "D-20260909-002"
OTHER_ID = "D-20260909-003"


def _workspace(tmp_path: Path, *, enabled: bool) -> str:
    workspace = tmp_path / ("enabled" if enabled else "disabled")
    decisions = workspace / "decisions"
    decisions.mkdir(parents=True)
    (workspace / "mind-mem.json").write_text(
        json.dumps(
            {
                "recall": {
                    "backend": "sqlite",
                    "expand_query": False,
                    "validity_gate": {"enabled": enabled, "threshold": 0.9, "demotion": 0.1},
                }
            }
        ),
        encoding="utf-8",
    )
    (decisions / "DECISIONS.md").write_text(
        f"[{TOP_ID}]\nType: Decision\nDate: 2026-09-09\nStatus: active\n"
        "Statement: needle needle needle needle needle\n\n"
        f"[{NEXT_ID}]\nType: Decision\nDate: 2026-09-09\nStatus: active\n"
        "Statement: needle\n\n"
        f"[{OTHER_ID}]\nType: Decision\nDate: 2026-09-09\nStatus: active\n"
        "Statement: unrelated control text\n\n",
        encoding="utf-8",
    )
    build_index(str(workspace), incremental=False)
    return str(workspace)


def _gate_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vg, "_load_contradicted_party_ids", lambda workspace: frozenset({TOP_ID}))
    monkeypatch.setattr(vg, "list_staleness_scores", lambda workspace, ids: {})
    monkeypatch.setattr(vg, "_load_outcome_signals", lambda cfg, workspace, ids: {})
    monkeypatch.setattr(vg, "_provenance_enabled", lambda cfg: False)


def test_sqlite_enabled_validity_sees_wide_pool_before_limit(monkeypatch, tmp_path):
    """A demoted indexed top hit must expose the next hit at limit=1."""
    _gate_fixture(monkeypatch)
    _make_sqlite_scores_deterministic(monkeypatch)
    hits = _recall_core.recall(_workspace(tmp_path, enabled=True), "needle", limit=1, rerank=False)
    assert hits, "positive control: SQLite returned no indexed hits"
    assert hits[0]["_id"] == NEXT_ID
    assert hits[0].get("_validity_demoted") is not True


def test_sqlite_disabled_validity_preserves_original_top_k(monkeypatch, tmp_path):
    """Default-off keeps the backend's original limit and order."""
    _gate_fixture(monkeypatch)
    _make_sqlite_scores_deterministic(monkeypatch)
    hits = _recall_core.recall(_workspace(tmp_path, enabled=False), "needle", limit=1, rerank=False)
    assert hits, "positive control: SQLite returned no indexed hits"
    assert hits[0]["_id"] == TOP_ID
    assert "validity" not in hits[0]


def _make_sqlite_scores_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the real SQLite candidate/width selection, normalize tiny fixture scores.

    SQLite's BM25 values are below the product's four-decimal output precision
    for this two-hit fixture, so both would round to zero and demotion could not
    be observed as an ordering change. The wrapper calls the real index query,
    preserving its candidate count and ``return_k`` contract, then gives the
    two real indexed rows distinct deterministic scores for this control.
    """
    from mind_mem import sqlite_index

    real_query_index = sqlite_index.query_index

    def query_with_fixture_scores(*args, **kwargs):
        rows = real_query_index(*args, **kwargs)
        for row in rows:
            row["score"] = 1.0 if row["_id"] == TOP_ID else 0.8 if row["_id"] == NEXT_ID else row["score"]
        return rows

    monkeypatch.setattr(sqlite_index, "query_index", query_with_fixture_scores)


class _LimitedBackend(_recall_core.RecallBackend):
    def __init__(self, *, degraded: bool = False) -> None:
        self.requested_limits: list[int] = []
        self.degraded = degraded

    def search(self, workspace, query, limit=10, active_only=False):
        self.requested_limits.append(limit)
        rows = [
            {"_id": TOP_ID, "score": 1.0, "Statement": "needle", "status": "active"},
            {"_id": NEXT_ID, "score": 0.8, "Statement": "needle", "status": "active"},
        ]
        result = RecallResults(rows[:limit])
        if self.degraded:
            result.degraded = {"leg": "fixture", "reason": "bounded"}
        return result

    def index(self, workspace):
        return None


@pytest.mark.parametrize("recall_config", ["not_a_dict", [], None, 1, False])
def test_malformed_recall_config_preserves_configured_backend(monkeypatch, tmp_path, recall_config):
    """Optional settings cannot turn a working provider into an empty fallback."""
    backend = _LimitedBackend()
    monkeypatch.setattr(_recall_core, "_load_backend", lambda workspace: backend)
    (tmp_path / "mind-mem.json").write_text(json.dumps({"recall": recall_config}), encoding="utf-8")
    hits = _recall_core.recall(str(tmp_path), "needle", limit=1, rerank=False)
    assert backend.requested_limits == [1]
    assert [hit["_id"] for hit in hits] == [TOP_ID]
    assert "validity" not in hits[0]


@pytest.mark.parametrize("enabled, expected_id, expected_limit", [(True, NEXT_ID, 200), (False, TOP_ID, 1)])
def test_recall_backend_validity_respects_enabled_pool_and_limit(monkeypatch, tmp_path, enabled, expected_id, expected_limit):
    """The arbitrary backend must widen only for enabled validity."""
    _gate_fixture(monkeypatch)
    backend = _LimitedBackend()
    monkeypatch.setattr(_recall_core, "_load_backend", lambda workspace: backend)
    workspace = _workspace(tmp_path, enabled=enabled)
    hits = _recall_core.recall(workspace, "needle", limit=1, rerank=False)
    assert backend.requested_limits == [expected_limit]
    assert hits[0]["_id"] == expected_id
    if enabled:
        assert hits[0].get("_validity_demoted") is not True
    else:
        assert "validity" not in hits[0]


def test_forged_validity_and_stale_marker_do_not_control_gate(monkeypatch, tmp_path):
    """Incoming annotations cannot bypass or fabricate the current verdict."""
    _gate_fixture(monkeypatch)
    hits = [
        {
            "_id": TOP_ID,
            "score": 1.0,
            "Statement": "contradicted",
            "status": "active",
            "validity": {"score": 1.0},
            "_validity_demoted": False,
        }
    ]
    cfg = {"validity_gate": {"enabled": True, "threshold": 0.9, "demotion": 0.1}}
    assert vg.apply_validity_gate(hits, str(tmp_path), cfg) == 1
    assert hits[0]["score"] == 0.1
    assert hits[0]["_validity_demoted"] is True


def test_recall_backend_carrier_survives_validity_resort(monkeypatch, tmp_path):
    """Early-leg sorting must retain the backend degradation carrier."""
    _gate_fixture(monkeypatch)
    backend = _LimitedBackend(degraded=True)
    monkeypatch.setattr(_recall_core, "_load_backend", lambda workspace: backend)
    workspace = _workspace(tmp_path, enabled=True)
    hits = _recall_core.recall(workspace, "needle", limit=1, rerank=False)
    assert hits[0]["_id"] == NEXT_ID
    assert getattr(hits, "degraded", None) == {"leg": "fixture", "reason": "bounded"}


@pytest.mark.parametrize("backend_kind", ["sqlite", "configured"])
def test_public_dispatch_applies_validity_once(monkeypatch, tmp_path, backend_kind):
    """Each returned hit receives one demotion, including the early-return legs."""
    _gate_fixture(monkeypatch)
    if backend_kind == "sqlite":
        _make_sqlite_scores_deterministic(monkeypatch)
    else:
        backend = _LimitedBackend()
        monkeypatch.setattr(_recall_core, "_load_backend", lambda workspace: backend)
    real_gate = _recall_core.apply_validity_gate
    calls = []

    def counted_gate(*args, **kwargs):
        calls.append(1)
        return real_gate(*args, **kwargs)

    monkeypatch.setattr(_recall_core, "apply_validity_gate", counted_gate)
    hits = _recall_core.recall(_workspace(tmp_path, enabled=True), "needle", limit=2, rerank=False)
    assert calls == [1]
    assert [hit["_id"] for hit in hits] == [NEXT_ID, TOP_ID]
    assert hits[1]["score"] == 0.1
