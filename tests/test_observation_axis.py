# Copyright 2026 STARGA, Inc.
"""Tests for Observer-Dependent Cognition (ODC) primitives.

Covers the value objects (ObservationAxis, AxisWeights, AxisScore,
Observation) and the pure helpers (axis_diversity, rotate_axes,
should_rotate, adversarial_pair). The retrieval-pipeline integration
lives in test_observation_axis_integration.py.
"""

from __future__ import annotations

import math

import pytest

from mind_mem.observation_axis import (
    DEFAULT_ROTATION_THRESHOLD,
    DEFAULT_WEIGHTS,
    AxisScore,
    AxisWeights,
    Observation,
    ObservationAxis,
    adversarial_pair,
    axis_diversity,
    rotate_axes,
    should_rotate,
)


# ---------------------------------------------------------------------------
# ObservationAxis enum
# ---------------------------------------------------------------------------


class TestObservationAxis:
    def test_enum_values_are_lowercase_strings(self) -> None:
        assert ObservationAxis.LEXICAL.value == "lexical"
        assert ObservationAxis.ENTITY_GRAPH.value == "entity_graph"
        assert ObservationAxis.ADVERSARIAL.value == "adversarial"

    def test_inherits_from_str_for_json_serialisation(self) -> None:
        assert isinstance(ObservationAxis.SEMANTIC, str)
        assert ObservationAxis.SEMANTIC == "semantic"

    def test_from_str_exact_match(self) -> None:
        assert ObservationAxis.from_str("lexical") is ObservationAxis.LEXICAL

    def test_from_str_case_insensitive(self) -> None:
        assert ObservationAxis.from_str("LEXICAL") is ObservationAxis.LEXICAL
        assert ObservationAxis.from_str("Semantic") is ObservationAxis.SEMANTIC

    def test_from_str_hyphen_tolerant(self) -> None:
        assert ObservationAxis.from_str("entity-graph") is ObservationAxis.ENTITY_GRAPH
        assert ObservationAxis.from_str("Entity-Graph") is ObservationAxis.ENTITY_GRAPH

    def test_from_str_strips_whitespace(self) -> None:
        assert ObservationAxis.from_str("  temporal  ") is ObservationAxis.TEMPORAL

    def test_from_str_unknown_raises_with_valid_list(self) -> None:
        with pytest.raises(ValueError, match="Unknown observation axis"):
            ObservationAxis.from_str("not-a-real-axis")


# ---------------------------------------------------------------------------
# AxisWeights
# ---------------------------------------------------------------------------


class TestAxisWeights:
    def test_default_lexical_and_semantic_only(self) -> None:
        w = AxisWeights()
        assert w.lexical == 1.0
        assert w.semantic == 1.0
        assert w.temporal == 0.0
        assert w.entity_graph == 0.0
        assert w.contradiction == 0.0
        assert w.adversarial == 0.0

    def test_is_frozen(self) -> None:
        w = AxisWeights()
        with pytest.raises(Exception):
            w.lexical = 0.5  # type: ignore[misc]

    def test_negative_weight_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            AxisWeights(lexical=-0.1)

    def test_nan_weight_rejected(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            AxisWeights(lexical=float("nan"))

    def test_inf_weight_rejected(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            AxisWeights(semantic=float("inf"))

    def test_as_dict_round_trip(self) -> None:
        w = AxisWeights(lexical=0.3, semantic=0.7, temporal=0.5)
        d = w.as_dict()
        assert d["lexical"] == 0.3
        assert d["semantic"] == 0.7
        assert d["temporal"] == 0.5
        assert d["entity_graph"] == 0.0

    def test_active_axes_only_nonzero(self) -> None:
        w = AxisWeights(lexical=1.0, semantic=0.0, temporal=0.5)
        assert w.active_axes() == (ObservationAxis.LEXICAL, ObservationAxis.TEMPORAL)

    def test_active_axes_empty_when_all_zero(self) -> None:
        w = AxisWeights(lexical=0.0, semantic=0.0)
        assert w.active_axes() == ()

    def test_normalised_sums_to_one(self) -> None:
        w = AxisWeights(lexical=2.0, semantic=2.0)
        n = w.normalised()
        assert n.lexical == 0.5
        assert n.semantic == 0.5
        assert math.isclose(sum(n.as_dict().values()), 1.0)

    def test_normalised_all_zero_returns_self(self) -> None:
        w = AxisWeights(lexical=0.0, semantic=0.0)
        assert w.normalised() is w

    def test_from_mapping_hyphen_tolerant(self) -> None:
        w = AxisWeights.from_mapping({"entity-graph": 0.8, "temporal": 0.2})
        assert w.entity_graph == 0.8
        assert w.temporal == 0.2
        assert w.lexical == 0.0

    def test_from_mapping_unknown_key_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown observation axis"):
            AxisWeights.from_mapping({"bogus": 1.0})

    def test_uniform_over_selected_axes(self) -> None:
        w = AxisWeights.uniform([ObservationAxis.CONTRADICTION, ObservationAxis.ADVERSARIAL])
        assert w.contradiction == 1.0
        assert w.adversarial == 1.0
        assert w.lexical == 0.0

    def test_default_weights_module_constant_matches_default_ctor(self) -> None:
        assert DEFAULT_WEIGHTS == AxisWeights()


# ---------------------------------------------------------------------------
# AxisScore
# ---------------------------------------------------------------------------


class TestAxisScore:
    def test_valid_score(self) -> None:
        s = AxisScore(axis=ObservationAxis.LEXICAL, confidence=0.8, rank=3)
        assert s.axis is ObservationAxis.LEXICAL
        assert s.confidence == 0.8
        assert s.rank == 3

    def test_rank_optional(self) -> None:
        s = AxisScore(axis=ObservationAxis.SEMANTIC, confidence=0.5)
        assert s.rank is None

    def test_confidence_out_of_range_rejected(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            AxisScore(axis=ObservationAxis.SEMANTIC, confidence=1.5)
        with pytest.raises(ValueError, match="confidence"):
            AxisScore(axis=ObservationAxis.SEMANTIC, confidence=-0.1)

    def test_nan_confidence_rejected(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            AxisScore(axis=ObservationAxis.SEMANTIC, confidence=float("nan"))

    def test_zero_rank_rejected(self) -> None:
        with pytest.raises(ValueError, match="rank"):
            AxisScore(axis=ObservationAxis.LEXICAL, confidence=0.5, rank=0)

    def test_as_dict_includes_rank_when_present(self) -> None:
        s = AxisScore(axis=ObservationAxis.LEXICAL, confidence=0.25, rank=7)
        d = s.as_dict()
        assert d == {"axis": "lexical", "confidence": 0.25, "rank": 7}

    def test_as_dict_omits_rank_when_absent(self) -> None:
        s = AxisScore(axis=ObservationAxis.SEMANTIC, confidence=0.5)
        assert "rank" not in s.as_dict()


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------


class TestObservation:
    def test_diversity_counts_distinct_axes(self) -> None:
        obs = Observation(
            axes=(
                AxisScore(axis=ObservationAxis.LEXICAL, confidence=0.9),
                AxisScore(axis=ObservationAxis.SEMANTIC, confidence=0.7),
                AxisScore(axis=ObservationAxis.LEXICAL, confidence=0.5),
            )
        )
        # Two distinct axes even though LEXICAL appears twice
        assert obs.diversity() == 2

    def test_diversity_zero_when_empty(self) -> None:
        obs = Observation(axes=())
        assert obs.diversity() == 0

    def test_top_axis_returns_highest_confidence(self) -> None:
        obs = Observation(
            axes=(
                AxisScore(axis=ObservationAxis.LEXICAL, confidence=0.4),
                AxisScore(axis=ObservationAxis.SEMANTIC, confidence=0.8),
                AxisScore(axis=ObservationAxis.TEMPORAL, confidence=0.6),
            )
        )
        top = obs.top_axis()
        assert top is not None
        assert top.axis is ObservationAxis.SEMANTIC

    def test_top_axis_none_when_empty(self) -> None:
        assert Observation(axes=()).top_axis() is None

    def test_as_dict_serialises_all_fields(self) -> None:
        obs = Observation(
            axes=(AxisScore(axis=ObservationAxis.LEXICAL, confidence=0.5),),
            rotated=True,
            notes=("initial axes failed threshold",),
        )
        d = obs.as_dict()
        assert d["rotated"] is True
        assert d["notes"] == ["initial axes failed threshold"]
        assert d["axes"][0]["axis"] == "lexical"


# ---------------------------------------------------------------------------
# axis_diversity helper
# ---------------------------------------------------------------------------


class TestAxisDiversity:
    def test_counts_distinct_axes_across_results(self) -> None:
        results = [
            {"_id": "A", "observation": {"axes": [{"axis": "lexical", "confidence": 0.5}]}},
            {"_id": "B", "observation": {"axes": [{"axis": "semantic", "confidence": 0.7}]}},
            {"_id": "C", "observation": {"axes": [{"axis": "lexical", "confidence": 0.3}]}},
        ]
        assert axis_diversity(results) == 2

    def test_returns_zero_on_empty_input(self) -> None:
        assert axis_diversity([]) == 0

    def test_skips_results_without_observation(self) -> None:
        results = [
            {"_id": "A"},
            {"_id": "B", "observation": {"axes": [{"axis": "temporal", "confidence": 0.5}]}},
        ]
        assert axis_diversity(results) == 1

    def test_skips_malformed_observation(self) -> None:
        results = [
            {"observation": "not a dict"},
            {"observation": {"axes": "not a list"}},
            {"observation": {"axes": [{"axis": "semantic", "confidence": 0.5}]}},
        ]
        assert axis_diversity(results) == 1


# ---------------------------------------------------------------------------
# rotate_axes
# ---------------------------------------------------------------------------


class TestRotateAxes:
    def test_rotate_from_lexical_suggests_orthogonal(self) -> None:
        current = AxisWeights(lexical=1.0, semantic=0.0)
        rotated = rotate_axes(current)
        assert rotated != current
        active = rotated.active_axes()
        assert len(active) >= 1
        assert ObservationAxis.LEXICAL not in active

    def test_rotate_respects_already_tried(self) -> None:
        current = AxisWeights(lexical=1.0, semantic=1.0)
        # If we already tried TEMPORAL too, it should not come back
        rotated = rotate_axes(
            current,
            already_tried=[ObservationAxis.TEMPORAL, ObservationAxis.ENTITY_GRAPH],
        )
        assert ObservationAxis.TEMPORAL not in rotated.active_axes()
        assert ObservationAxis.ENTITY_GRAPH not in rotated.active_axes()

    def test_rotate_returns_current_when_exhausted(self) -> None:
        current = AxisWeights(lexical=1.0, semantic=1.0)
        tried = [
            ObservationAxis.TEMPORAL,
            ObservationAxis.ENTITY_GRAPH,
            ObservationAxis.CONTRADICTION,
            ObservationAxis.ADVERSARIAL,
        ]
        rotated = rotate_axes(current, already_tried=tried)
        assert rotated is current

    def test_rotate_max_new_cap(self) -> None:
        current = AxisWeights(lexical=1.0)
        rotated = rotate_axes(current, max_new=1)
        assert len(rotated.active_axes()) == 1

    def test_rotate_zero_max_new_returns_current(self) -> None:
        current = AxisWeights(lexical=1.0)
        rotated = rotate_axes(current, max_new=0)
        assert rotated is current


# ---------------------------------------------------------------------------
# should_rotate
# ---------------------------------------------------------------------------


class TestShouldRotate:
    def test_below_default_threshold_triggers(self) -> None:
        assert should_rotate(DEFAULT_ROTATION_THRESHOLD - 0.01) is True

    def test_at_threshold_does_not_trigger(self) -> None:
        assert should_rotate(DEFAULT_ROTATION_THRESHOLD) is False

    def test_high_confidence_does_not_trigger(self) -> None:
        assert should_rotate(0.95) is False

    def test_custom_threshold(self) -> None:
        assert should_rotate(0.5, threshold=0.6) is True
        assert should_rotate(0.7, threshold=0.6) is False

    def test_nan_confidence_triggers_rotation(self) -> None:
        assert should_rotate(float("nan")) is True


# ---------------------------------------------------------------------------
# adversarial_pair
# ---------------------------------------------------------------------------


class TestAdversarialPair:
    def test_lexical_pairs_with_contradiction(self) -> None:
        assert adversarial_pair(ObservationAxis.LEXICAL) is ObservationAxis.CONTRADICTION

    def test_semantic_pairs_with_contradiction(self) -> None:
        assert adversarial_pair(ObservationAxis.SEMANTIC) is ObservationAxis.CONTRADICTION

    def test_contradiction_pairs_with_adversarial(self) -> None:
        assert adversarial_pair(ObservationAxis.CONTRADICTION) is ObservationAxis.ADVERSARIAL

    def test_unknown_defaults_to_adversarial(self) -> None:
        assert adversarial_pair(ObservationAxis.ADVERSARIAL) is ObservationAxis.ADVERSARIAL
