# Copyright 2026 STARGA, Inc.
"""Tests for OWL-lite ontology + in-process change stream (v2.5.0)."""

from __future__ import annotations

import pytest

from mind_mem.ontology import (
    EntityType,
    Ontology,
    OntologyRegistry,
    software_engineering_ontology,
)
from mind_mem.change_stream import ChangeStream, StreamStats


# ---------------------------------------------------------------------------
# EntityType
# ---------------------------------------------------------------------------


class TestEntityType:
    def test_name_must_be_upper_snake_case(self) -> None:
        EntityType(name="PERSON")
        EntityType(name="ENTITY_TYPE_2")
        with pytest.raises(ValueError, match="UPPER_SNAKE_CASE"):
            EntityType(name="person")
        with pytest.raises(ValueError, match="UPPER_SNAKE_CASE"):
            EntityType(name="Person")
        with pytest.raises(ValueError, match="UPPER_SNAKE_CASE"):
            EntityType(name="")

    def test_required_and_optional_cannot_overlap(self) -> None:
        with pytest.raises(ValueError, match="both required and optional"):
            EntityType(
                name="PERSON",
                required=("name",),
                optional=("name",),
            )


# ---------------------------------------------------------------------------
# Ontology construction + inheritance
# ---------------------------------------------------------------------------


class TestOntology:
    def test_version_required(self) -> None:
        with pytest.raises(ValueError, match="version"):
            Ontology(version="", types={})

    def test_type_name_must_match_key(self) -> None:
        with pytest.raises(ValueError, match="does not match"):
            Ontology(
                version="1",
                types={"FOO": EntityType(name="BAR")},
            )

    def test_unknown_parent_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown parent"):
            Ontology(
                version="1",
                types={"PERSON": EntityType(name="PERSON", parent="ENTITY")},
            )

    def test_inherited_required(self) -> None:
        ont = software_engineering_ontology()
        # PERSON inherits `name` from ENTITY.
        assert "name" in ont.effective_required("PERSON")
        assert "role" in ont.effective_required("PERSON")

    def test_inherited_allowed(self) -> None:
        ont = software_engineering_ontology()
        allowed = ont.effective_allowed("PERSON")
        assert {"name", "description", "tags", "role", "email", "github"} <= allowed

    def test_inherited_property_types(self) -> None:
        ont = software_engineering_ontology()
        pt = ont.effective_property_types("PROJECT")
        assert pt["name"] == "str"   # inherited from ENTITY
        assert pt["status"] == "str"  # declared on PROJECT


# ---------------------------------------------------------------------------
# validate()
# ---------------------------------------------------------------------------


class TestValidate:
    def _ont(self) -> Ontology:
        return software_engineering_ontology()

    def test_missing_required_reports_error(self) -> None:
        errs = self._ont().validate("PERSON", {"role": "engineer"})
        assert any("missing required property: 'name'" in e for e in errs)

    def test_valid_block_empty_errors(self) -> None:
        errs = self._ont().validate(
            "PERSON", {"name": "Alice", "role": "engineer"}
        )
        assert errs == []

    def test_type_mismatch_reported(self) -> None:
        errs = self._ont().validate(
            "PERSON", {"name": "Alice", "role": 42},  # role should be str
        )
        assert any("type mismatch for 'role'" in e for e in errs)

    def test_unexpected_property_strict(self) -> None:
        errs = self._ont().validate(
            "PERSON",
            {"name": "Alice", "role": "engineer", "foo": "bar"},
        )
        assert any("unexpected property: 'foo'" in e for e in errs)

    def test_unexpected_property_non_strict(self) -> None:
        errs = self._ont().validate(
            "PERSON",
            {"name": "Alice", "role": "engineer", "foo": "bar"},
            strict=False,
        )
        assert errs == []

    def test_framework_private_fields_ignored(self) -> None:
        # _id / _score / similar framework keys don't trip strict mode.
        errs = self._ont().validate(
            "PERSON",
            {"name": "Alice", "role": "engineer", "_id": "P-1"},
        )
        assert errs == []

    def test_unknown_type(self) -> None:
        errs = self._ont().validate("NO_SUCH_TYPE", {})
        assert errs == ["unknown type: 'NO_SUCH_TYPE'"]

    def test_float_accepts_int(self) -> None:
        # The "float" label intentionally accepts int (Python coercion).
        errs = self._ont().validate(
            "DECISION",
            {"name": "X", "statement": "hi", "confidence": 1},
        )
        assert errs == []

    def test_none_treated_as_missing(self) -> None:
        errs = self._ont().validate("PERSON", {"name": None, "role": "x"})
        assert any("missing required property: 'name'" in e for e in errs)


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


class TestOntologySerialisation:
    def test_round_trip(self) -> None:
        ont = software_engineering_ontology()
        restored = Ontology.from_dict(ont.to_dict())
        assert restored.version == ont.version
        assert restored.type_names() == ont.type_names()
        assert restored.effective_required("PERSON") == ont.effective_required("PERSON")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestOntologyRegistry:
    def test_load_and_active(self) -> None:
        reg = OntologyRegistry()
        a = Ontology(version="1.0", types={})
        b = Ontology(version="1.1", types={})
        reg.load(a)
        reg.load(b)
        # First load wins as active unless overridden.
        assert reg.active().version == "1.0"
        reg.set_active("1.1")
        assert reg.active().version == "1.1"

    def test_set_active_unknown_raises(self) -> None:
        reg = OntologyRegistry()
        with pytest.raises(KeyError):
            reg.set_active("missing")

    def test_versions_sorted(self) -> None:
        reg = OntologyRegistry()
        reg.load(Ontology(version="2.0", types={}))
        reg.load(Ontology(version="1.5", types={}))
        assert reg.versions() == ["1.5", "2.0"]


# ---------------------------------------------------------------------------
# ChangeStream
# ---------------------------------------------------------------------------


class TestChangeStream:
    def test_constructor_rejects_zero_depth(self) -> None:
        with pytest.raises(ValueError, match="max_queue_depth"):
            ChangeStream(max_queue_depth=0)

    def test_publish_delivers_to_listener(self) -> None:
        stream = ChangeStream()
        received: list = []
        stream.subscribe(lambda ev: received.append(ev))
        stream.publish("block.created", {"_id": "B-1"})
        assert len(received) == 1
        assert received[0].type == "block.created"
        assert received[0].payload == {"_id": "B-1"}

    def test_unsubscribe_stops_delivery(self) -> None:
        stream = ChangeStream()
        received: list = []
        sid = stream.subscribe(lambda ev: received.append(ev))
        stream.publish("block.created", {"_id": "B-1"})
        stream.unsubscribe(sid)
        stream.publish("block.created", {"_id": "B-2"})
        assert [ev.payload["_id"] for ev in received] == ["B-1"]

    def test_unsubscribe_unknown_id_returns_false(self) -> None:
        stream = ChangeStream()
        assert stream.unsubscribe(999) is False

    def test_listener_exception_isolated(self) -> None:
        stream = ChangeStream()

        def bad_listener(ev):
            raise RuntimeError("boom")

        good: list = []
        stream.subscribe(bad_listener)
        stream.subscribe(lambda ev: good.append(ev))
        # Bad listener must not block the good one.
        stream.publish("block.created", {})
        assert len(good) == 1

    def test_overflow_counts_drops(self) -> None:
        stream = ChangeStream(max_queue_depth=2)
        # Listener that never reads its queue — events queue up.
        stream.subscribe(lambda ev: None)
        for _ in range(5):
            stream.publish("block.created", {})
        stats = stream.stats()
        assert stats.published == 5
        assert stats.dropped >= 3  # 5 published, 2 retained = 3 shed

    def test_stats_snapshot_fields(self) -> None:
        stream = ChangeStream()
        stream.subscribe(lambda ev: None)
        stream.publish("block.created", {})
        stats = stream.stats()
        d = stats.as_dict()
        assert set(d.keys()) == {
            "subscribers", "published", "delivered", "dropped", "queue_depth"
        }
