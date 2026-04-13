# Copyright 2026 STARGA, Inc.
"""Integration tests for the axis-aware recall orchestrator.

These exercise the fusion/rotation machinery without depending on a real
workspace. ``mind_mem.axis_recall._recall_for_axis`` is monkeypatched so
each axis returns a deterministic result list and we can assert on the
final observation metadata.
"""

from __future__ import annotations

from typing import Any, Mapping

import pytest

from mind_mem import axis_recall as ar
from mind_mem.observation_axis import (
    AxisWeights,
    ObservationAxis,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_result(block_id: str, score: float = 1.0) -> dict[str, Any]:
    return {
        "_id": block_id,
        "file": f"decisions/{block_id}.md",
        "line": 1,
        "score": score,
        "excerpt": f"excerpt for {block_id}",
    }


def _install_stub(monkeypatch: pytest.MonkeyPatch, per_axis: Mapping[ObservationAxis, list[dict]]) -> dict[str, list]:
    """Monkeypatch _recall_for_axis to return canned results per axis.

    Returns a dict with "calls" keyed to the number of times each axis was
    queried so tests can assert the dispatch pattern.
    """
    calls: dict[str, list] = {"invocations": []}

    def _stub(workspace, query, axis, *, limit, active_only, base_recall_kwargs):
        calls["invocations"].append(
            {
                "axis": axis,
                "query": query,
                "limit": limit,
                "active_only": active_only,
                "extra_kwargs": dict(base_recall_kwargs),
            }
        )
        return list(per_axis.get(axis, []))

    monkeypatch.setattr(ar, "_recall_for_axis", _stub)
    return calls


# ---------------------------------------------------------------------------
# Basic axis dispatch
# ---------------------------------------------------------------------------


class TestAxisDispatch:
    def test_default_weights_runs_lexical_and_semantic(self, monkeypatch):
        calls = _install_stub(
            monkeypatch,
            {
                ObservationAxis.LEXICAL: [_fake_result("A")],
                ObservationAxis.SEMANTIC: [_fake_result("A"), _fake_result("B")],
            },
        )
        result = ar.recall_with_axis("/ws", "JWT auth")
        axes_hit = [c["axis"] for c in calls["invocations"]]
        assert ObservationAxis.LEXICAL in axes_hit
        assert ObservationAxis.SEMANTIC in axes_hit
        assert result["diversity"] >= 1

    def test_single_axis_only_runs_that_axis(self, monkeypatch):
        calls = _install_stub(
            monkeypatch,
            {ObservationAxis.TEMPORAL: [_fake_result("X")]},
        )
        result = ar.recall_with_axis(
            "/ws",
            "JWT",
            weights=AxisWeights(lexical=0.0, semantic=0.0, temporal=1.0),
            allow_rotation=False,  # isolate dispatch from rotation
        )
        axes_hit = [c["axis"] for c in calls["invocations"]]
        assert axes_hit == [ObservationAxis.TEMPORAL]
        assert len(result["results"]) == 1

    def test_empty_weights_raises(self, monkeypatch):
        _install_stub(monkeypatch, {})
        with pytest.raises(ValueError, match="at least one active axis"):
            ar.recall_with_axis(
                "/ws",
                "q",
                weights=AxisWeights(lexical=0.0, semantic=0.0),
            )


# ---------------------------------------------------------------------------
# Observation metadata
# ---------------------------------------------------------------------------


class TestObservationMetadata:
    def test_each_result_gets_observation(self, monkeypatch):
        _install_stub(
            monkeypatch,
            {
                ObservationAxis.LEXICAL: [_fake_result("A")],
                ObservationAxis.SEMANTIC: [_fake_result("A")],
            },
        )
        result = ar.recall_with_axis("/ws", "q", allow_rotation=False)
        for res in result["results"]:
            assert "observation" in res
            assert isinstance(res["observation"]["axes"], list)
            assert len(res["observation"]["axes"]) >= 1

    def test_result_tagged_with_multiple_axes_when_both_return_it(self, monkeypatch):
        _install_stub(
            monkeypatch,
            {
                ObservationAxis.LEXICAL: [_fake_result("A")],
                ObservationAxis.SEMANTIC: [_fake_result("A")],
            },
        )
        result = ar.recall_with_axis("/ws", "q", allow_rotation=False)
        top = result["results"][0]
        axes_in_obs = {a["axis"] for a in top["observation"]["axes"]}
        assert "lexical" in axes_in_obs
        assert "semantic" in axes_in_obs

    def test_diversity_counts_contributing_axes(self, monkeypatch):
        _install_stub(
            monkeypatch,
            {
                ObservationAxis.LEXICAL: [_fake_result("A")],
                ObservationAxis.SEMANTIC: [_fake_result("B")],
            },
        )
        result = ar.recall_with_axis("/ws", "q", allow_rotation=False)
        assert result["diversity"] == 2


# ---------------------------------------------------------------------------
# Axis rotation
# ---------------------------------------------------------------------------


class TestAxisRotation:
    def test_rotation_triggers_when_no_results(self, monkeypatch):
        # Both initial axes return nothing — pipeline should rotate.
        per_axis = {
            ObservationAxis.LEXICAL: [],
            ObservationAxis.SEMANTIC: [],
            # Orthogonal rotation candidates both produce a result.
            ObservationAxis.TEMPORAL: [_fake_result("T1")],
            ObservationAxis.ENTITY_GRAPH: [_fake_result("T2")],
        }
        _install_stub(monkeypatch, per_axis)
        result = ar.recall_with_axis("/ws", "q")
        assert result["rotated"] is True
        assert any(r["_id"] in {"T1", "T2"} for r in result["results"])

    def test_rotation_disabled_when_flag_off(self, monkeypatch):
        _install_stub(
            monkeypatch,
            {
                ObservationAxis.LEXICAL: [],
                ObservationAxis.SEMANTIC: [],
            },
        )
        result = ar.recall_with_axis("/ws", "q", allow_rotation=False)
        assert result["rotated"] is False
        assert result["results"] == []

    def test_high_confidence_skips_rotation(self, monkeypatch):
        # Confidence of 1.0 for a rank-1 result should skip rotation.
        _install_stub(
            monkeypatch,
            {
                ObservationAxis.LEXICAL: [_fake_result("A")],
                ObservationAxis.SEMANTIC: [_fake_result("A")],
            },
        )
        result = ar.recall_with_axis("/ws", "q", rotation_threshold=0.5)
        assert result["rotated"] is False

    def test_rotation_mark_sets_observation_rotated_flag(self, monkeypatch):
        per_axis = {
            ObservationAxis.LEXICAL: [],
            ObservationAxis.SEMANTIC: [],
            ObservationAxis.TEMPORAL: [_fake_result("R1")],
            ObservationAxis.ENTITY_GRAPH: [_fake_result("R2")],
        }
        _install_stub(monkeypatch, per_axis)
        result = ar.recall_with_axis("/ws", "q")
        assert result["rotated"] is True
        # R1 and R2 only appeared in the rotation pass, so they MUST be
        # flagged rotated=True.
        for res in result["results"]:
            assert res["observation"]["rotated"] is True

    def test_rotation_only_flags_contributors_not_primary(self, monkeypatch):
        """Primary-only results must keep rotated=False even when rotation fires."""
        per_axis = {
            ObservationAxis.LEXICAL: [_fake_result("P1"), _fake_result("P2")],
            ObservationAxis.SEMANTIC: [],
            # Rotation adds a brand new block R.
            ObservationAxis.TEMPORAL: [_fake_result("R")],
            ObservationAxis.ENTITY_GRAPH: [],
        }
        _install_stub(monkeypatch, per_axis)
        # Force rotation by setting a threshold above any achievable confidence.
        result = ar.recall_with_axis("/ws", "q", limit=10, rotation_threshold=2.0)
        assert result["rotated"] is True
        results_by_id = {r["_id"]: r for r in result["results"]}
        assert "R" in results_by_id
        # Rotation-only result R carries rotated=True
        assert results_by_id["R"]["observation"]["rotated"] is True
        # Primary-only result P1 stayed on the initial axes — the rotated
        # flag must NOT be stamped across it.
        assert "P1" in results_by_id
        assert results_by_id["P1"]["observation"]["rotated"] is False

    def test_rotation_stops_when_all_axes_tried(self, monkeypatch):
        per_axis = {a: [] for a in ObservationAxis}
        _install_stub(monkeypatch, per_axis)
        result = ar.recall_with_axis("/ws", "q")
        # Every axis exhausted → no results but no infinite loop either
        assert result["results"] == []

    def test_attempts_list_records_rotation(self, monkeypatch):
        per_axis = {
            ObservationAxis.LEXICAL: [],
            ObservationAxis.SEMANTIC: [],
            ObservationAxis.TEMPORAL: [_fake_result("T")],
        }
        _install_stub(monkeypatch, per_axis)
        result = ar.recall_with_axis("/ws", "q")
        assert len(result["attempts"]) >= 2  # initial + 1 rotation


# ---------------------------------------------------------------------------
# Adversarial injection
# ---------------------------------------------------------------------------


class TestAdversarialInjection:
    def test_adversarial_flag_adds_contradiction_pass(self, monkeypatch):
        calls = _install_stub(
            monkeypatch,
            {
                ObservationAxis.LEXICAL: [_fake_result("L")],
                ObservationAxis.SEMANTIC: [_fake_result("S")],
                ObservationAxis.CONTRADICTION: [_fake_result("C")],
            },
        )
        result = ar.recall_with_axis("/ws", "q", adversarial=True, allow_rotation=False)
        axes_hit = {c["axis"] for c in calls["invocations"]}
        # LEXICAL + SEMANTIC are the active axes; each pairs with CONTRADICTION.
        assert ObservationAxis.CONTRADICTION in axes_hit
        assert any(r["_id"] == "C" for r in result["results"])

    def test_adversarial_query_prefix_on_adversarial_axis(self, monkeypatch):
        """The adversarial axis should see the query rephrased with NOT "(query)"."""
        calls = _install_stub(
            monkeypatch,
            {
                ObservationAxis.LEXICAL: [_fake_result("L")],
                ObservationAxis.ADVERSARIAL: [_fake_result("A")],
            },
        )
        # Ask for ADVERSARIAL directly — it should receive a NOT "..." wrapped query.
        ar.recall_with_axis(
            "/ws",
            "original query",
            weights=AxisWeights(lexical=1.0, semantic=0.0, adversarial=1.0),
            adversarial=False,
            allow_rotation=False,
        )
        adversarial_calls = [c for c in calls["invocations"] if c["axis"] is ObservationAxis.ADVERSARIAL]
        assert adversarial_calls
        axis_query = adversarial_calls[0]["query"]
        # Full expression wrapped so FTS5 negates the whole phrase, not just the first token.
        assert axis_query == 'NOT "original query"'

    def test_adversarial_query_escapes_double_quotes(self, monkeypatch):
        """Embedded double-quotes must be doubled so they stay inside the phrase."""
        calls = _install_stub(
            monkeypatch,
            {ObservationAxis.ADVERSARIAL: []},
        )
        ar.recall_with_axis(
            "/ws",
            'say "hello" world',
            weights=AxisWeights(lexical=0.0, semantic=0.0, adversarial=1.0),
            adversarial=False,
            allow_rotation=False,
        )
        adversarial_calls = [c for c in calls["invocations"] if c["axis"] is ObservationAxis.ADVERSARIAL]
        assert adversarial_calls
        assert adversarial_calls[0]["query"] == 'NOT "say ""hello"" world"'

    def test_adversarial_empty_query_skipped(self, monkeypatch):
        """Empty query on adversarial axis must not fire a NOT "" that breaks FTS5."""
        calls = _install_stub(
            monkeypatch,
            {ObservationAxis.ADVERSARIAL: [_fake_result("X")]},
        )
        result = ar.recall_with_axis(
            "/ws",
            "   ",
            weights=AxisWeights(lexical=0.0, semantic=0.0, adversarial=1.0),
            adversarial=False,
            allow_rotation=False,
        )
        adversarial_calls = [c for c in calls["invocations"] if c["axis"] is ObservationAxis.ADVERSARIAL]
        # Axis was skipped entirely, so no invocation and no results.
        assert adversarial_calls == []
        assert result["results"] == []


# ---------------------------------------------------------------------------
# Fusion scoring
# ---------------------------------------------------------------------------


class TestFusionScoring:
    def test_higher_ranked_across_axes_wins(self, monkeypatch):
        # A appears first in both axes; B only appears second in one.
        _install_stub(
            monkeypatch,
            {
                ObservationAxis.LEXICAL: [_fake_result("A"), _fake_result("B")],
                ObservationAxis.SEMANTIC: [_fake_result("A")],
            },
        )
        result = ar.recall_with_axis("/ws", "q", allow_rotation=False)
        top = result["results"][0]
        assert top["_id"] == "A"

    def test_axis_weight_influences_ranking(self, monkeypatch):
        # B only comes from SEMANTIC; A only from LEXICAL. Heavy semantic
        # weight should promote B over A.
        _install_stub(
            monkeypatch,
            {
                ObservationAxis.LEXICAL: [_fake_result("A")],
                ObservationAxis.SEMANTIC: [_fake_result("B")],
            },
        )
        result = ar.recall_with_axis(
            "/ws",
            "q",
            weights=AxisWeights(lexical=0.1, semantic=10.0),
            allow_rotation=False,
        )
        assert result["results"][0]["_id"] == "B"

    def test_axis_score_is_deterministic(self, monkeypatch):
        _install_stub(
            monkeypatch,
            {
                ObservationAxis.LEXICAL: [_fake_result("A")],
                ObservationAxis.SEMANTIC: [_fake_result("A")],
            },
        )
        r1 = ar.recall_with_axis("/ws", "q", allow_rotation=False)
        r2 = ar.recall_with_axis("/ws", "q", allow_rotation=False)
        assert r1["results"][0]["_axis_score"] == r2["results"][0]["_axis_score"]
