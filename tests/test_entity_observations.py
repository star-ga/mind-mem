# Copyright 2026 STARGA, Inc.
"""First-class entity observations field (roadmap §b).

Covers:
- Idempotent `entities.observations` migration (column-exists guard → re-run
  is a no-op), including on a legacy DB created without the column.
- Feature-flag gating (reuses the v4 flag mechanism): the API refuses to run
  when `entity_observations` is OFF, and the default schema is untouched.
- JSON-list accretion: facts accumulate, dedup, and store in deterministic
  (lex) order — no clock / rand in the write path.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from mind_mem.knowledge_graph import KnowledgeGraph
from mind_mem.v4 import feature_flags
from mind_mem.v4.feature_flags import FeatureDisabledError


@pytest.fixture
def flag_on(tmp_path, monkeypatch):
    """Enable the entity_observations v4 flag via a workspace config."""
    cfg = tmp_path / "mind-mem.json"
    cfg.write_text(json.dumps({"v4": {"entity_observations": {"enabled": True}}}), encoding="utf-8")
    monkeypatch.setenv("MIND_MEM_CONFIG", str(cfg))
    yield


@pytest.fixture
def flag_off(tmp_path, monkeypatch):
    cfg = tmp_path / "mind-mem.json"
    cfg.write_text(json.dumps({"v4": {}}), encoding="utf-8")
    monkeypatch.setenv("MIND_MEM_CONFIG", str(cfg))
    yield


@pytest.fixture
def kg(tmp_path):
    with KnowledgeGraph(str(tmp_path / "kg.db")) as graph:
        yield graph


def _columns(conn: sqlite3.Connection) -> set[str]:
    return {row[1] for row in conn.execute("PRAGMA table_info(entities)").fetchall()}


# ---------------------------------------------------------------------------
# Flag registration
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFlagRegistered:
    def test_flag_in_registry(self) -> None:
        assert "entity_observations" in feature_flags.ALL_V4_FLAGS

    def test_typed_edges_flag_in_registry(self) -> None:
        assert "typed_edges" in feature_flags.ALL_V4_FLAGS


# ---------------------------------------------------------------------------
# Migration idempotency
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMigrationIdempotency:
    def test_default_schema_has_no_observations(self, kg) -> None:
        # The base entities table ships without observations; it is added
        # only by the (flag-gated) migration.
        assert "observations" not in _columns(kg._conn)

    def test_migration_adds_column(self, kg) -> None:
        added = kg.entities.migrate_observations()
        assert added is True
        assert "observations" in _columns(kg._conn)

    def test_migration_rerun_is_noop(self, kg) -> None:
        assert kg.entities.migrate_observations() is True
        # Second call must report no change and not raise.
        assert kg.entities.migrate_observations() is False
        assert kg.entities.migrate_observations() is False

    def test_migration_preserves_existing_rows(self, kg) -> None:
        kg.entities.resolve("STARGA")
        kg.entities.migrate_observations()
        # Existing entity survives the ALTER and gets the default '[]'.
        row = kg._conn.execute("SELECT observations FROM entities WHERE id = ?", ("starga",)).fetchone()
        assert row["observations"] == "[]"

    def test_migration_on_legacy_db(self, tmp_path):
        # Simulate a pre-existing DB whose entities table lacks the column,
        # then open a KnowledgeGraph over it and migrate.
        path = str(tmp_path / "legacy.db")
        raw = sqlite3.connect(path)
        raw.executescript(
            "CREATE TABLE entities (id TEXT PRIMARY KEY, canonical TEXT NOT NULL);INSERT INTO entities VALUES ('acme', 'acme');"
        )
        raw.commit()
        raw.close()
        with KnowledgeGraph(path) as graph:
            assert "observations" not in _columns(graph._conn)
            assert graph.entities.migrate_observations() is True
            assert "observations" in _columns(graph._conn)
            assert graph.entities.migrate_observations() is False


# ---------------------------------------------------------------------------
# Feature-flag gating
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFlagGating:
    def test_add_observation_refused_when_off(self, kg, flag_off) -> None:
        with pytest.raises(FeatureDisabledError):
            kg.entities.add_observation("STARGA", "is a company")

    def test_observations_refused_when_off(self, kg, flag_off) -> None:
        with pytest.raises(FeatureDisabledError):
            kg.entities.observations("STARGA")

    def test_add_observation_allowed_when_on(self, kg, flag_on) -> None:
        result = kg.entities.add_observation("STARGA", "is a company")
        assert result == ["is a company"]

    def test_gating_does_not_migrate_when_off(self, kg, flag_off) -> None:
        # A refused call must not have altered the schema as a side effect.
        with pytest.raises(FeatureDisabledError):
            kg.entities.add_observation("STARGA", "x")
        assert "observations" not in _columns(kg._conn)


# ---------------------------------------------------------------------------
# JSON-list accretion
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAccretion:
    def test_single_fact(self, kg, flag_on) -> None:
        kg.entities.add_observation("STARGA", "builds MIND")
        assert kg.entities.observations("STARGA") == ["builds MIND"]

    def test_facts_accumulate(self, kg, flag_on) -> None:
        kg.entities.add_observation("STARGA", "builds MIND")
        kg.entities.add_observation("STARGA", "founded by Nikolai")
        assert kg.entities.observations("STARGA") == ["builds MIND", "founded by Nikolai"]

    def test_dedup(self, kg, flag_on) -> None:
        kg.entities.add_observation("STARGA", "builds MIND")
        kg.entities.add_observation("STARGA", "builds MIND")
        assert kg.entities.observations("STARGA") == ["builds MIND"]

    def test_deterministic_lex_order(self, kg, flag_on) -> None:
        # Insertion order does not affect stored order (deterministic).
        kg.entities.add_observation("E", "zebra")
        kg.entities.add_observation("E", "apple")
        kg.entities.add_observation("E", "mango")
        assert kg.entities.observations("E") == ["apple", "mango", "zebra"]

    def test_stored_json_is_canonical(self, kg, flag_on) -> None:
        kg.entities.add_observation("E", "b")
        kg.entities.add_observation("E", "a")
        row = kg._conn.execute("SELECT observations FROM entities WHERE id = ?", ("e",)).fetchone()
        # Compact, sorted, valid JSON list.
        assert row["observations"] == '["a","b"]'

    def test_accretion_survives_reopen(self, tmp_path, flag_on) -> None:
        path = str(tmp_path / "kg.db")
        with KnowledgeGraph(path) as g1:
            g1.entities.add_observation("STARGA", "builds MIND")
        with KnowledgeGraph(path) as g2:
            assert g2.entities.observations("STARGA") == ["builds MIND"]

    def test_observations_alias_aware(self, kg, flag_on) -> None:
        # Aliases that canonicalise together share one observation list.
        kg.entities.add_observation("STARGA", "builds MIND")
        assert kg.entities.observations("  starga ") == ["builds MIND"]

    def test_unknown_entity_returns_empty(self, kg, flag_on) -> None:
        assert kg.entities.observations("never seen") == []

    def test_empty_fact_rejected(self, kg, flag_on) -> None:
        with pytest.raises(ValueError):
            kg.entities.add_observation("STARGA", "   ")

    def test_unicode_fact(self, kg, flag_on) -> None:
        kg.entities.add_observation("E", "日本語 fact")
        assert kg.entities.observations("E") == ["日本語 fact"]
