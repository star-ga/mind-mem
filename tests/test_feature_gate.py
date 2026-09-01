"""Tests for FeatureGate — the shared config-resolver for retrieval features."""

from __future__ import annotations

import json
import logging

import pytest

from mind_mem import entity_prefetch, graph_recall, kg_fusion, session_boost, truth_score
from mind_mem.feature_gate import (
    FeatureGate,
    FieldSpec,
    always_detector,
    has_capitalised_token_detector,
    multi_hop_detector,
    multi_hop_or_temporal_detector,
    strict_int,
    strict_number,
)
from mind_mem.session_boost import _session_of


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


# ===========================================================================
# Migration coverage.
#
# FeatureGate stopped being a library nothing calls: the five hand-rolled
# ``is_*_enabled`` / ``resolve_*_config`` pairs under ``retrieval.*`` now
# resolve through it. Two obligations follow, and each has a class below:
#
#   1. The delegation is REAL — patching FeatureGate has to change what the
#      public functions answer. A test that only imports the module would
#      still pass against the pre-migration tree.
#   2. The delegation is BEHAVIOUR-IDENTICAL — the pre-migration bodies are
#      reproduced verbatim below as an oracle and differenced against the
#      live functions over a config matrix that includes every shape the
#      five copies had drifted on.
# ===========================================================================


class TestStrictCoercers:
    """The shared ``isinstance`` guard the five copies each spelled out."""

    def test_strict_int_passes_ints(self) -> None:
        assert strict_int(5) == 5
        assert strict_int(0) == 0
        assert strict_int(-3) == -3

    def test_strict_int_rejects_numeric_strings(self) -> None:
        """``coerce=int`` would parse "5"; the legacy guard rejected it."""
        with pytest.raises(TypeError):
            strict_int("5")

    def test_strict_int_rejects_floats(self) -> None:
        with pytest.raises(TypeError):
            strict_int(2.5)

    def test_strict_int_accepts_bool_like_isinstance_does(self) -> None:
        """Inherited quirk, kept on purpose: ``isinstance(True, int)`` is True.

        The legacy resolvers all read ``max_hops: true`` as 1. Rejecting
        bools here would be a behaviour change dressed up as a cleanup.
        """
        assert strict_int(True) == 1
        assert strict_int(False) == 0

    def test_strict_number_passes_ints_and_floats(self) -> None:
        assert strict_number(1) == pytest.approx(1.0)
        assert isinstance(strict_number(1), float)
        assert strict_number(0.25) == pytest.approx(0.25)

    def test_strict_number_rejects_strings(self) -> None:
        with pytest.raises(TypeError):
            strict_number("0.5")

    def test_field_spec_falls_back_when_coercer_rejects(self) -> None:
        spec = FieldSpec(default=2, coerce=strict_int, validate=lambda v: v > 0)
        assert spec.resolve("5") == 2
        assert spec.resolve(4) == 4


class TestAlwaysDetector:
    def test_returns_true_for_every_input(self) -> None:
        assert always_detector(None, None) is True
        assert always_detector("", []) is True

    def test_is_not_the_same_as_no_detector(self) -> None:
        """``auto_detector=None`` means auto-enable can never fire."""
        cfg = {"retrieval": {"f": {}}}
        assert FeatureGate(name="f").is_enabled(cfg) is False
        assert FeatureGate(name="f", auto_detector=always_detector).is_enabled(cfg) is True


class TestImplicitSection:
    """An absent section is an empty one — for the gates that shipped that way."""

    def _gates(self) -> tuple[FeatureGate, FeatureGate]:
        strict = FeatureGate(name="f", auto_detector=always_detector)
        implicit = FeatureGate(name="f", auto_detector=always_detector, implicit_section=True)
        return strict, implicit

    def test_absent_section_off_by_default(self) -> None:
        strict, _ = self._gates()
        assert strict.is_enabled({"retrieval": {}}) is False
        assert strict.is_enabled({"other": 1}) is False

    def test_absent_section_auto_enables_when_implicit(self) -> None:
        _, implicit = self._gates()
        assert implicit.is_enabled({"retrieval": {}}) is True
        assert implicit.is_enabled({"other": 1}) is True

    def test_implicit_does_not_rescue_a_falsy_config(self) -> None:
        """``{}`` and ``None`` stay OFF — the legacy guard rejected them first."""
        _, implicit = self._gates()
        assert implicit.is_enabled({}) is False
        assert implicit.is_enabled(None) is False

    def test_implicit_does_not_rescue_a_malformed_section(self) -> None:
        """A section of the wrong type is a hard OFF, never "use defaults"."""
        _, implicit = self._gates()
        assert implicit.is_enabled({"retrieval": {"f": "yes"}}) is False
        assert implicit.is_enabled({"retrieval": {"f": None}}) is False
        assert implicit.is_enabled({"retrieval": "junk"}) is False

    def test_implicit_section_does_not_change_resolve(self) -> None:
        fields = {"n": FieldSpec(default=7, coerce=strict_int, validate=lambda v: v > 0)}
        strict = FeatureGate(name="f", fields=fields)
        implicit = FeatureGate(name="f", fields=fields, implicit_section=True)
        for cfg in ({}, {"retrieval": {}}, {"retrieval": {"f": {"n": 2}}}):
            assert strict.resolve(cfg) == implicit.resolve(cfg)


# --- 1. the delegation is real ---------------------------------------------


class _Marker:
    """Neither True nor False — a bool answer means the gate was bypassed."""


class TestMigratedFeaturesRouteThroughFeatureGate:
    """Fails against the pre-migration tree, and against any re-inlining.

    Every assertion patches :class:`FeatureGate` itself, so it can only pass
    while the public function's answer is the gate's answer. A hand-rolled
    body would return a plain bool / its own dict and miss the marker.
    """

    ENABLE_CALLS = [
        (graph_recall.is_graph_expand_enabled, "multi_hop"),
        (entity_prefetch.is_entity_prefetch_enabled, "entity_prefetch"),
        (session_boost.is_session_boost_enabled, "session_boost"),
        (kg_fusion.is_kg_fusion_enabled, "kg_fusion"),
        (truth_score.is_truth_score_enabled, "truth_score"),
    ]

    RESOLVE_CALLS = [
        (graph_recall.resolve_graph_config, "multi_hop"),
        (entity_prefetch.resolve_entity_prefetch_config, "entity_prefetch"),
        (session_boost.resolve_session_boost_config, "session_boost"),
        (kg_fusion.resolve_kg_fusion_config, "kg_fusion"),
    ]

    @pytest.mark.parametrize("fn,name", ENABLE_CALLS)
    def test_enable_answer_comes_from_the_gate(self, fn, name, monkeypatch) -> None:
        seen: list[str] = []
        marker = _Marker()

        def fake(self, config, *, query=None, results=None):
            seen.append(self.name)
            return marker

        monkeypatch.setattr(FeatureGate, "is_enabled", fake)
        # A config that every legacy body would have answered False on.
        assert fn(None) is marker
        assert seen == [name]

    @pytest.mark.parametrize("fn,name", RESOLVE_CALLS)
    def test_resolved_params_come_from_the_gate(self, fn, name, monkeypatch) -> None:
        def fake(self, config):
            return {"__from_gate__": self.name}

        monkeypatch.setattr(FeatureGate, "resolve", fake)
        assert fn({}) == {"__from_gate__": name}

    def test_each_feature_owns_exactly_one_declared_gate(self) -> None:
        # A list, not a dict: FeatureGate is an unfrozen dataclass and so is
        # unhashable by construction.
        declared = [
            (graph_recall.GRAPH_EXPAND_GATE, "multi_hop", True),
            (entity_prefetch.ENTITY_PREFETCH_GATE, "entity_prefetch", True),
            (session_boost.SESSION_BOOST_GATE, "session_boost", True),
            (kg_fusion.KG_FUSION_GATE, "kg_fusion", False),
            (truth_score.TRUTH_SCORE_GATE, "truth_score", False),
        ]
        for gate, name, implicit in declared:
            assert isinstance(gate, FeatureGate)
            assert gate.name == name
            assert gate.implicit_section is implicit

    def test_query_and_results_reach_the_gate(self, monkeypatch) -> None:
        """The two detector inputs are forwarded, not dropped on the floor."""
        seen: list[tuple] = []

        def fake(self, config, *, query=None, results=None):
            seen.append((self.name, query, results))
            return False

        monkeypatch.setattr(FeatureGate, "is_enabled", fake)
        graph_recall.is_graph_expand_enabled({}, "why did X change?")
        session_boost.is_session_boost_enabled({}, [{"_id": "DIA-D1-3"}])
        assert seen == [
            ("multi_hop", "why did X change?", None),
            ("session_boost", None, [{"_id": "DIA-D1-3"}]),
        ]


# --- 2. the delegation is behaviour-identical -------------------------------
#
# Pre-migration bodies, copied verbatim from the tree before the swap. They
# are the oracle: if a migrated gate and its ancestor ever disagree on any
# config below, the "this is a refactor" claim is false.


def _legacy_is_graph_expand_enabled(config, query=None):
    if not config or not isinstance(config, dict):
        return False
    retrieval = config.get("retrieval", {})
    if not isinstance(retrieval, dict):
        return False
    mh = retrieval.get("multi_hop", {})
    if not isinstance(mh, dict):
        return False
    if mh.get("enabled", False):
        return True
    if not mh.get("auto_enable", True):
        return False
    if not query:
        return False
    try:
        from mind_mem._recall_detection import detect_query_type

        return detect_query_type(query) == "multi-hop"
    except Exception:
        return False


def _legacy_resolve_graph_config(config):
    defaults = {"max_hops": 2, "decay": 0.5, "max_neighbors_per_hop": 5}
    if not config or not isinstance(config, dict):
        return defaults
    retrieval = config.get("retrieval", {})
    if not isinstance(retrieval, dict):
        return defaults
    mh = retrieval.get("multi_hop", {})
    if not isinstance(mh, dict):
        return defaults
    out = dict(defaults)
    if isinstance(mh.get("max_hops"), int) and mh["max_hops"] > 0:
        out["max_hops"] = min(3, int(mh["max_hops"]))
    if isinstance(mh.get("decay"), (int, float)) and 0 < mh["decay"] <= 1:
        out["decay"] = float(mh["decay"])
    if isinstance(mh.get("max_neighbors_per_hop"), int) and mh["max_neighbors_per_hop"] > 0:
        out["max_neighbors_per_hop"] = int(mh["max_neighbors_per_hop"])
    return out


def _legacy_is_entity_prefetch_enabled(config):
    if not config or not isinstance(config, dict):
        return False
    retrieval = config.get("retrieval", {})
    if not isinstance(retrieval, dict):
        return False
    ep = retrieval.get("entity_prefetch", {})
    if not isinstance(ep, dict):
        return False
    if ep.get("enabled", False):
        return True
    return bool(ep.get("auto_enable", True))


def _legacy_resolve_entity_prefetch_config(config):
    defaults = {"max_entities": 3, "max_hops": 1, "entity_score": 5.0}
    if not config or not isinstance(config, dict):
        return defaults
    retrieval = config.get("retrieval", {})
    if not isinstance(retrieval, dict):
        return defaults
    ep = retrieval.get("entity_prefetch", {})
    if not isinstance(ep, dict):
        return defaults
    out = dict(defaults)
    if isinstance(ep.get("max_entities"), int) and ep["max_entities"] > 0:
        out["max_entities"] = int(ep["max_entities"])
    if isinstance(ep.get("max_hops"), int) and ep["max_hops"] >= 0:
        out["max_hops"] = int(ep["max_hops"])
    if isinstance(ep.get("entity_score"), (int, float)) and ep["entity_score"] > 0:
        out["entity_score"] = float(ep["entity_score"])
    return out


def _legacy_is_session_boost_enabled(config, results=None):
    if not config or not isinstance(config, dict):
        return False
    retrieval = config.get("retrieval", {})
    if not isinstance(retrieval, dict):
        return False
    sb = retrieval.get("session_boost", {})
    if not isinstance(sb, dict):
        return False
    if sb.get("enabled", False):
        return True
    if not sb.get("auto_enable", True):
        return False
    if results:
        for b in results[:10]:
            if _session_of(b):
                return True
    return False


def _legacy_resolve_session_boost_config(config):
    defaults = {"top_seed_count": 3, "boost": 0.3}
    if not config or not isinstance(config, dict):
        return defaults
    retrieval = config.get("retrieval", {})
    if not isinstance(retrieval, dict):
        return defaults
    sb = retrieval.get("session_boost", {})
    if not isinstance(sb, dict):
        return defaults
    out = dict(defaults)
    if isinstance(sb.get("top_seed_count"), int) and sb["top_seed_count"] > 0:
        out["top_seed_count"] = int(sb["top_seed_count"])
    if isinstance(sb.get("boost"), (int, float)) and 0 < sb["boost"] <= 5:
        out["boost"] = float(sb["boost"])
    return out


def _legacy_is_kg_fusion_enabled(config):
    if not config or not isinstance(config, dict):
        return False
    retrieval = config.get("retrieval", {})
    if not isinstance(retrieval, dict):
        return False
    kf = retrieval.get("kg_fusion", {})
    if not isinstance(kf, dict):
        return False
    return bool(kf.get("enabled", False))


def _legacy_resolve_kg_fusion_config(config):
    out = {"max_hops": 2, "decay": 0.5, "max_neighbors_per_hop": 5, "max_total_added": 25}
    if not config or not isinstance(config, dict):
        return out
    retrieval = config.get("retrieval", {})
    if not isinstance(retrieval, dict):
        return out
    kf = retrieval.get("kg_fusion", {})
    if not isinstance(kf, dict):
        return out
    if isinstance(kf.get("max_hops"), int) and kf["max_hops"] > 0:
        out["max_hops"] = min(2, int(kf["max_hops"]))
    if isinstance(kf.get("decay"), (int, float)) and 0 < kf["decay"] <= 1:
        out["decay"] = float(kf["decay"])
    if isinstance(kf.get("max_neighbors_per_hop"), int) and kf["max_neighbors_per_hop"] > 0:
        out["max_neighbors_per_hop"] = int(kf["max_neighbors_per_hop"])
    if isinstance(kf.get("max_total_added"), int) and kf["max_total_added"] > 0:
        out["max_total_added"] = int(kf["max_total_added"])
    return out


def _legacy_is_truth_score_enabled(config):
    if not config or not isinstance(config, dict):
        return False
    retrieval = config.get("retrieval", {})
    if not isinstance(retrieval, dict):
        return False
    ts = retrieval.get("truth_score", {})
    if not isinstance(ts, dict):
        return False
    return bool(ts.get("enabled", False))


#: Shapes every gate must agree on, independent of its knobs. The
#: ``{"retrieval": {}}`` / ``{"other": 1}`` rows are the ones the five copies
#: had drifted on — three of them auto-enabled there, two did not.
_SHAPES: list[object] = [
    None,
    {},
    "not-a-dict",
    5,
    [],
    {"other": 1},
    {"retrieval": {}},
    {"retrieval": None},
    {"retrieval": "junk"},
    {"retrieval": []},
    {"retrieval": {"unrelated": {"enabled": True}}},
]

#: Section bodies, spliced under ``retrieval.<name>`` for every feature.
_SECTIONS: list[object] = [
    {},
    None,
    "junk",
    [],
    {"enabled": True},
    {"enabled": False},
    {"enabled": "yes"},
    {"enabled": 0},
    {"auto_enable": False},
    {"auto_enable": True},
    {"auto_enable": 0},
    {"auto_enable": None},
    {"enabled": False, "auto_enable": False},
    {"enabled": True, "auto_enable": False},
]

#: Per-feature knob values, one knob varied at a time.
_KNOBS: dict[str, dict[str, list[object]]] = {
    "multi_hop": {
        "max_hops": [1, 2, 3, 4, 99, 0, -1, True, False, "3", 3.0, None],
        "decay": [0.5, 1, 0, 1.5, -0.1, "0.5", True, None],
        "max_neighbors_per_hop": [1, 5, 10, 0, -1, "x", 2.5, True, None],
    },
    "entity_prefetch": {
        "max_entities": [1, 3, 9, 0, -1, "2", 2.5, True, None],
        "max_hops": [0, 1, 2, -1, "1", 1.5, True, None],
        "entity_score": [1, 5.0, 0, -2, "5", True, None],
    },
    "session_boost": {
        "top_seed_count": [1, 3, 9, 0, -1, "3", 3.5, True, None],
        "boost": [0.3, 1, 5, 5.1, 0, -1, "0.3", True, None],
    },
    "kg_fusion": {
        "max_hops": [1, 2, 3, 0, -1, "2", 2.5, True, None],
        "decay": [0.5, 1, 0, 1.5, "0.5", True, None],
        "max_neighbors_per_hop": [1, 5, 0, -1, "5", True, None],
        "max_total_added": [1, 25, 0, -1, "25", True, None],
    },
    "truth_score": {},
}


def _matrix(name: str) -> list[object]:
    """Every config shape this feature's gate has to answer identically."""
    out = list(_SHAPES)
    out += [{"retrieval": {name: sec}} for sec in _SECTIONS]
    for knob, values in _KNOBS[name].items():
        for value in values:
            out.append({"retrieval": {name: {knob: value}}})
            out.append({"retrieval": {name: {"enabled": True, knob: value}}})
    return out


_QUERIES = [None, "", "PostgreSQL deployment", "What is the relationship between X and Y?", "When did we decide?"]

_RESULT_SETS: list[list[dict] | None] = [
    None,
    [],
    [{"_id": "D-20260420-001", "score": 1.0}],
    [{"_id": "DIA-D1-3", "score": 1.0}],
    [{"dia_id": "D1:3", "score": 1.0}],
    [{"_id": "D-1"}] * 11 + [{"_id": "DIA-D9-1"}],
]


class TestMigratedGatesMatchLegacyExactly:
    """The oracle difference. Any disagreement means the swap changed recall."""

    def test_multi_hop_enable(self) -> None:
        for cfg in _matrix("multi_hop"):
            for query in _QUERIES:
                assert graph_recall.is_graph_expand_enabled(cfg, query) is _legacy_is_graph_expand_enabled(cfg, query), (cfg, query)

    def test_multi_hop_resolve_is_byte_identical(self) -> None:
        for cfg in _matrix("multi_hop"):
            live = graph_recall.resolve_graph_config(cfg)
            # json, not ==: this also pins key ORDER and int-vs-float type.
            assert json.dumps(live) == json.dumps(_legacy_resolve_graph_config(cfg)), cfg

    def test_entity_prefetch_enable(self) -> None:
        for cfg in _matrix("entity_prefetch"):
            assert entity_prefetch.is_entity_prefetch_enabled(cfg) is _legacy_is_entity_prefetch_enabled(cfg), cfg

    def test_entity_prefetch_resolve_is_byte_identical(self) -> None:
        for cfg in _matrix("entity_prefetch"):
            live = entity_prefetch.resolve_entity_prefetch_config(cfg)
            assert json.dumps(live) == json.dumps(_legacy_resolve_entity_prefetch_config(cfg)), cfg

    def test_session_boost_enable(self) -> None:
        for cfg in _matrix("session_boost"):
            for results in _RESULT_SETS:
                live = session_boost.is_session_boost_enabled(cfg, results)
                assert live is _legacy_is_session_boost_enabled(cfg, results), (cfg, results)

    def test_session_boost_resolve_is_byte_identical(self) -> None:
        for cfg in _matrix("session_boost"):
            live = session_boost.resolve_session_boost_config(cfg)
            assert json.dumps(live) == json.dumps(_legacy_resolve_session_boost_config(cfg)), cfg

    def test_kg_fusion_enable(self) -> None:
        for cfg in _matrix("kg_fusion"):
            assert kg_fusion.is_kg_fusion_enabled(cfg) is _legacy_is_kg_fusion_enabled(cfg), cfg

    def test_kg_fusion_resolve_is_byte_identical(self) -> None:
        for cfg in _matrix("kg_fusion"):
            live = kg_fusion.resolve_kg_fusion_config(cfg)
            assert json.dumps(live) == json.dumps(_legacy_resolve_kg_fusion_config(cfg)), cfg

    def test_truth_score_enable(self) -> None:
        for cfg in _matrix("truth_score"):
            assert truth_score.is_truth_score_enabled(cfg) is _legacy_is_truth_score_enabled(cfg), cfg

    def test_the_matrix_actually_covers_both_answers(self) -> None:
        """Guards the oracle itself: an all-False matrix would prove nothing."""
        for name, fn in (
            ("multi_hop", lambda c: graph_recall.is_graph_expand_enabled(c, "relationship between X and Y")),
            ("entity_prefetch", entity_prefetch.is_entity_prefetch_enabled),
            ("session_boost", lambda c: session_boost.is_session_boost_enabled(c, [{"_id": "DIA-D1-3"}])),
            ("kg_fusion", kg_fusion.is_kg_fusion_enabled),
            ("truth_score", truth_score.is_truth_score_enabled),
        ):
            answers = {fn(cfg) for cfg in _matrix(name)}
            assert answers == {True, False}, name


class TestDefaultOffIsUntouched:
    """The zero-config path, stated once per feature rather than inferred."""

    @pytest.mark.parametrize("cfg", [None, {}, {"retrieval": "junk"}, {"retrieval": {"multi_hop": "junk"}}])
    def test_every_gate_is_off(self, cfg: object) -> None:
        assert graph_recall.is_graph_expand_enabled(cfg, "relationship between X and Y") is False
        assert kg_fusion.is_kg_fusion_enabled(cfg) is False
        assert truth_score.is_truth_score_enabled(cfg) is False

    @pytest.mark.parametrize("cfg", [None, {}])
    def test_auto_enabling_gates_need_a_config(self, cfg: object) -> None:
        assert entity_prefetch.is_entity_prefetch_enabled(cfg) is False
        assert session_boost.is_session_boost_enabled(cfg, [{"_id": "DIA-D1-3"}]) is False

    def test_default_params_are_unchanged(self) -> None:
        assert graph_recall.resolve_graph_config(None) == {"max_hops": 2, "decay": 0.5, "max_neighbors_per_hop": 5}
        assert entity_prefetch.resolve_entity_prefetch_config(None) == {"max_entities": 3, "max_hops": 1, "entity_score": 5.0}
        assert session_boost.resolve_session_boost_config(None) == {"top_seed_count": 3, "boost": 0.3}
        assert kg_fusion.resolve_kg_fusion_config(None) == {
            "max_hops": 2,
            "decay": 0.5,
            "max_neighbors_per_hop": 5,
            "max_total_added": 25,
        }


class _Recorder:
    """Stand-in logger that records instead of emitting."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def __getattr__(self, level: str):
        def _record(event: str = "", **kw: object) -> None:
            self.calls.append((level, event, kw))

        return _record


class TestOffProbesAreSilent:
    """A probe that answers "off" must leave no trace that it ran.

    Slice 1's regression: a flag probe called a helper that logged on a
    malformed config, so the flag-OFF build emitted a line the unwired
    build did not. Observability is behaviour.
    """

    MODULES = [graph_recall, entity_prefetch, session_boost, kg_fusion, truth_score]

    def test_the_shared_resolver_has_no_logger_at_all(self) -> None:
        import mind_mem.feature_gate as fg

        assert not hasattr(fg, "_log")
        assert not any(isinstance(v, logging.Logger) for v in vars(fg).values())

    def test_no_feature_logs_while_deciding_it_is_off(self, monkeypatch) -> None:
        recorders = {}
        for mod in self.MODULES:
            rec = _Recorder()
            recorders[mod.__name__] = rec
            monkeypatch.setattr(mod, "_log", rec)

        # Every malformed / empty / explicitly-off shape, through every probe.
        for cfg in _SHAPES + [{"retrieval": {"multi_hop": "junk"}}, {"retrieval": {"entity_prefetch": None}}]:
            for query in _QUERIES:
                graph_recall.is_graph_expand_enabled(cfg, query)
                graph_recall.resolve_graph_config(cfg)
            entity_prefetch.is_entity_prefetch_enabled(cfg)
            entity_prefetch.resolve_entity_prefetch_config(cfg)
            for results in _RESULT_SETS:
                session_boost.is_session_boost_enabled(cfg, results)
            session_boost.resolve_session_boost_config(cfg)
            kg_fusion.is_kg_fusion_enabled(cfg)
            kg_fusion.resolve_kg_fusion_config(cfg)
            truth_score.is_truth_score_enabled(cfg)

        assert {name: rec.calls for name, rec in recorders.items() if rec.calls} == {}


class TestReachableFromTheRecallPipeline:
    """The gate is on the live recall path, not just behind a public helper.

    ``HybridBackend``'s per-feature steps are where recall actually asks
    "is this on?". Patching :class:`FeatureGate` and driving those steps
    shows the question now reaches the shared resolver — the difference
    between a module that is imported and a module that is reached.
    """

    def test_every_pipeline_step_consults_the_gate(self, monkeypatch, tmp_path) -> None:
        from mind_mem.hybrid_recall import HybridBackend

        consulted: list[str] = []

        def fake(self, config, *, query=None, results=None):
            consulted.append(self.name)
            return False

        monkeypatch.setattr(FeatureGate, "is_enabled", fake)

        backend = HybridBackend({})
        hits = [{"_id": "D-20260420-001", "score": 1.0}]
        workspace = str(tmp_path)

        # Each step fails open, so an unreached gate shows up as a missing
        # name rather than as an exception — hence the explicit set compare.
        assert backend._maybe_session_boost(list(hits)) == hits
        assert backend._maybe_truth_score(list(hits)) == hits
        assert backend._maybe_entity_prefetch("Alice outage", workspace, list(hits)) == hits
        assert backend._maybe_graph_expand("Alice outage", workspace, list(hits)) == hits
        assert backend._maybe_kg_expand("Alice outage", workspace, list(hits)) == hits
        assert backend._load_corpus_if_needed("Alice outage", workspace) is None

        assert set(consulted) == {
            "session_boost",
            "truth_score",
            "entity_prefetch",
            "multi_hop",
            "kg_fusion",
        }

    def test_default_config_leaves_every_step_a_no_op(self, tmp_path) -> None:
        """Unpatched, with no config: same list object out, no new keys."""
        from mind_mem.hybrid_recall import HybridBackend

        backend = HybridBackend({})
        hits = [{"_id": "D-20260420-001", "score": 1.0}]
        workspace = str(tmp_path)

        for out in (
            backend._maybe_session_boost(hits),
            backend._maybe_truth_score(hits),
            backend._maybe_entity_prefetch("Alice outage", workspace, hits),
            backend._maybe_graph_expand("Alice outage", workspace, hits),
            backend._maybe_kg_expand("Alice outage", workspace, hits),
        ):
            assert out is hits
        assert hits == [{"_id": "D-20260420-001", "score": 1.0}]
