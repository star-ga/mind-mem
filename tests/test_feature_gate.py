"""Tests for FeatureGate — the shared config-resolver for retrieval features."""

from __future__ import annotations

import pytest

from mind_mem.feature_gate import (
    FeatureGate,
    FieldSpec,
    has_capitalised_token_detector,
    multi_hop_detector,
    multi_hop_or_temporal_detector,
)


# Build a minimal gate fixture reused across tests.
def _gate() -> FeatureGate:
    return FeatureGate(
        name="test_feature",
        fields={
            "count": FieldSpec(default=3, coerce=int, validate=lambda v: v > 0),
            "ratio": FieldSpec(default=0.5, coerce=float, validate=lambda v: 0 < v <= 1),
            "label": FieldSpec(default="none"),
        },
        auto_detector=lambda q, r: q is not None and "trigger" in q,
    )


class TestFieldSpec:
    def test_default_when_raw_is_none(self) -> None:
        spec = FieldSpec(default=10)
        assert spec.resolve(None) == 10

    def test_coerce_success(self) -> None:
        spec = FieldSpec(default=0, coerce=int)
        assert spec.resolve("42") == 42

    def test_coerce_failure_falls_back(self) -> None:
        spec = FieldSpec(default=7, coerce=int)
        assert spec.resolve("nope") == 7

    def test_validate_success(self) -> None:
        spec = FieldSpec(default=0, coerce=int, validate=lambda v: v > 0)
        assert spec.resolve(5) == 5

    def test_validate_failure_falls_back(self) -> None:
        spec = FieldSpec(default=1, coerce=int, validate=lambda v: v > 0)
        assert spec.resolve(-3) == 1


class TestFeatureGate:
    def test_disabled_when_no_section(self) -> None:
        g = _gate()
        assert g.is_enabled({}) is False
        assert g.is_enabled(None) is False

    def test_enabled_true_wins(self) -> None:
        g = _gate()
        assert g.is_enabled({"retrieval": {"test_feature": {"enabled": True}}}) is True

    def test_auto_enable_fires_on_detector(self) -> None:
        g = _gate()
        cfg = {"retrieval": {"test_feature": {}}}
        assert g.is_enabled(cfg, query="trigger me") is True
        assert g.is_enabled(cfg, query="boring query") is False

    def test_auto_enable_false_overrides(self) -> None:
        g = _gate()
        cfg = {"retrieval": {"test_feature": {"auto_enable": False}}}
        assert g.is_enabled(cfg, query="trigger me") is False

    def test_resolve_with_defaults(self) -> None:
        g = _gate()
        out = g.resolve({})
        assert out == {"count": 3, "ratio": 0.5, "label": "none"}

    def test_resolve_with_valid_overrides(self) -> None:
        g = _gate()
        cfg = {
            "retrieval": {
                "test_feature": {"count": 10, "ratio": 0.8, "label": "custom"},
            }
        }
        assert g.resolve(cfg) == {"count": 10, "ratio": 0.8, "label": "custom"}

    def test_resolve_invalid_falls_back(self) -> None:
        g = _gate()
        cfg = {"retrieval": {"test_feature": {"count": -5, "ratio": 2.0}}}
        out = g.resolve(cfg)
        assert out["count"] == 3
        assert out["ratio"] == pytest.approx(0.5)

    def test_detector_exception_is_safe(self) -> None:
        g = FeatureGate(
            name="t",
            auto_detector=lambda q, r: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        assert g.is_enabled({"retrieval": {"t": {}}}, query="x") is False


class TestPrebakedDetectors:
    def test_multi_hop_detector(self) -> None:
        assert multi_hop_detector("What is the relationship between X and Y?", None) is True
        assert multi_hop_detector("PostgreSQL deployment", None) is False
        assert multi_hop_detector(None, None) is False

    def test_multi_hop_or_temporal_detector(self) -> None:
        assert multi_hop_or_temporal_detector("When did we decide?", None) is True
        assert multi_hop_or_temporal_detector("PostgreSQL", None) is False

    def test_capitalised_token_detector(self) -> None:
        assert has_capitalised_token_detector("What did Alice say?", None) is True
        assert has_capitalised_token_detector("what did she say?", None) is False
        assert has_capitalised_token_detector(None, None) is False
