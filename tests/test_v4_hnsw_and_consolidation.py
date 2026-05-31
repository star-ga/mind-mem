"""Tests for HNSW kind-filtered ANN + consolidation worker."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from mind_mem.v4 import FeatureDisabledError
from mind_mem.v4.consolidation_worker import (
    DEFAULT_CONSOLIDATION_CONFIG,
    ConsolidationConfig,
    ConsolidationPlan,
    plan_consolidation,
)
from mind_mem.v4.consolidation_worker import FLAG as CONSOL_FLAG
from mind_mem.v4.hnsw_kind_index import FLAG as HNSW_FLAG
from mind_mem.v4.hnsw_kind_index import (
    backend_status,
    ensure_hnsw_schema,
    knn_by_kind,
    register_block_embedding,
)
from mind_mem.v4.tier_memory import RecallTier

# ---------------------------------------------------------------------------
# HNSW kind index — fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def hnsw_on(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cfg = {"v4": {HNSW_FLAG: {"enabled": True}}}
    (tmp_path / "mind-mem.json").write_text(json.dumps(cfg), encoding="utf-8")
    monkeypatch.setenv("MIND_MEM_CONFIG", str(tmp_path / "mind-mem.json"))
    return tmp_path


@pytest.fixture
def hnsw_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cfg = {"v4": {HNSW_FLAG: {"enabled": False}}}
    (tmp_path / "mind-mem.json").write_text(json.dumps(cfg), encoding="utf-8")
    monkeypatch.setenv("MIND_MEM_CONFIG", str(tmp_path / "mind-mem.json"))
    return tmp_path


# ---------------------------------------------------------------------------
# HNSW — flag + schema
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_hnsw_flag_off_blocks_register(hnsw_off: Path) -> None:
    with pytest.raises(FeatureDisabledError):
        register_block_embedding(hnsw_off, "B-1", "entity", [1.0, 0.0])


@pytest.mark.unit
def test_hnsw_flag_off_blocks_knn(hnsw_off: Path) -> None:
    with pytest.raises(FeatureDisabledError):
        knn_by_kind(hnsw_off, "entity", [1.0, 0.0])


@pytest.mark.unit
def test_hnsw_flag_off_blocks_status(hnsw_off: Path) -> None:
    with pytest.raises(FeatureDisabledError):
        backend_status(hnsw_off)


@pytest.mark.unit
def test_hnsw_schema_idempotent(hnsw_on: Path) -> None:
    ensure_hnsw_schema(hnsw_on)
    ensure_hnsw_schema(hnsw_on)
    ensure_hnsw_schema(hnsw_on)
    with sqlite3.connect(hnsw_on / "index.db") as conn:
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='block_kind_embeddings'").fetchall()
    assert rows == [("block_kind_embeddings",)]


@pytest.mark.unit
def test_hnsw_backend_status_returns_known_value(hnsw_on: Path) -> None:
    """Either sqlite_vec is loadable (production) or brute_force is the fallback."""
    s = backend_status(hnsw_on)
    assert s["backend"] in ("sqlite_vec", "brute_force")


# ---------------------------------------------------------------------------
# HNSW — register + kNN
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_register_then_knn_round_trips(hnsw_on: Path) -> None:
    register_block_embedding(hnsw_on, "B-1", "entity", [1.0, 0.0, 0.0])
    register_block_embedding(hnsw_on, "B-2", "entity", [0.0, 1.0, 0.0])
    register_block_embedding(hnsw_on, "B-3", "entity", [-1.0, 0.0, 0.0])
    out = knn_by_kind(hnsw_on, "entity", [1.0, 0.0, 0.0], k=3)
    # B-1 identical → distance 0; B-3 opposite → distance 2; B-2 orthogonal → 1.
    assert [r[0] for r in out] == ["B-1", "B-2", "B-3"]
    assert out[0][1] == pytest.approx(0.0, abs=1e-6)


@pytest.mark.unit
def test_knn_filters_by_kind(hnsw_on: Path) -> None:
    """Embeddings under one kind don't surface in kNN of another kind."""
    register_block_embedding(hnsw_on, "B-ent", "entity", [1.0, 0.0])
    register_block_embedding(hnsw_on, "B-cod", "code", [1.0, 0.0])
    out = knn_by_kind(hnsw_on, "entity", [1.0, 0.0], k=10)
    assert [r[0] for r in out] == ["B-ent"]


@pytest.mark.unit
def test_knn_respects_k(hnsw_on: Path) -> None:
    for i in range(20):
        # Vectors of varying first-dim magnitude with non-zero second dim
        # so they all have non-zero norm and a defined cosine direction.
        register_block_embedding(hnsw_on, f"B-{i}", "entity", [float(i + 1), 1.0])
    out = knn_by_kind(hnsw_on, "entity", [10.0, 1.0], k=5)
    assert len(out) == 5


@pytest.mark.unit
def test_knn_empty_when_kind_missing(hnsw_on: Path) -> None:
    register_block_embedding(hnsw_on, "B-1", "entity", [1.0, 0.0])
    assert knn_by_kind(hnsw_on, "concept", [1.0, 0.0]) == []


@pytest.mark.unit
def test_knn_empty_when_db_missing(hnsw_on: Path) -> None:
    """Pre-schema knn returns empty list, not crash."""
    assert knn_by_kind(hnsw_on, "entity", [1.0, 0.0]) == []


@pytest.mark.unit
def test_knn_handles_zero_query_norm(hnsw_on: Path) -> None:
    register_block_embedding(hnsw_on, "B-1", "entity", [1.0, 0.0])
    out = knn_by_kind(hnsw_on, "entity", [0.0, 0.0])
    assert out == []  # zero query → no meaningful direction


@pytest.mark.unit
def test_knn_skips_zero_norm_db_vectors(hnsw_on: Path) -> None:
    """A stored zero vector has no direction; skip rather than crash."""
    register_block_embedding(hnsw_on, "B-zero", "entity", [0.0, 0.0])
    register_block_embedding(hnsw_on, "B-real", "entity", [1.0, 0.0])
    out = knn_by_kind(hnsw_on, "entity", [1.0, 0.0])
    assert [r[0] for r in out] == ["B-real"]


@pytest.mark.unit
def test_register_replaces_on_duplicate(hnsw_on: Path) -> None:
    register_block_embedding(hnsw_on, "B-1", "entity", [1.0, 0.0])
    register_block_embedding(hnsw_on, "B-1", "entity", [-1.0, 0.0])
    out = knn_by_kind(hnsw_on, "entity", [1.0, 0.0])
    # New embedding wins; distance 1 - cos(1, -1) = 2.
    assert out[0][1] == pytest.approx(2.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Consolidation worker — fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def consol_on(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cfg = {"v4": {CONSOL_FLAG: {"enabled": True}}}
    (tmp_path / "mind-mem.json").write_text(json.dumps(cfg), encoding="utf-8")
    monkeypatch.setenv("MIND_MEM_CONFIG", str(tmp_path / "mind-mem.json"))
    return tmp_path


@pytest.fixture
def consol_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cfg = {"v4": {CONSOL_FLAG: {"enabled": False}}}
    (tmp_path / "mind-mem.json").write_text(json.dumps(cfg), encoding="utf-8")
    monkeypatch.setenv("MIND_MEM_CONFIG", str(tmp_path / "mind-mem.json"))
    return tmp_path


def _seed_warm(workspace: Path, rows: list[tuple[str, float | None]]) -> None:
    db = workspace / "index.db"
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS block_recall_tier ("
            "block_id TEXT PRIMARY KEY, tier TEXT NOT NULL DEFAULT 'warm', "
            "last_seen_at TEXT NOT NULL, promoted_count INTEGER NOT NULL DEFAULT 0, "
            "last_surprise REAL, block_version INTEGER NOT NULL DEFAULT 0)"
        )
        for bid, surprise in rows:
            conn.execute(
                "INSERT INTO block_recall_tier (block_id, tier, last_seen_at, last_surprise) VALUES (?, 'warm', '2026-05-10T00:00:00Z', ?)",
                (bid, surprise),
            )
        conn.commit()


# ---------------------------------------------------------------------------
# Consolidation — flag + empty paths
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_consol_flag_off_raises(consol_off: Path) -> None:
    with pytest.raises(FeatureDisabledError):
        plan_consolidation(consol_off)


@pytest.mark.unit
def test_consol_empty_when_db_missing(consol_on: Path) -> None:
    p = plan_consolidation(consol_on)
    assert p.total_proposals == 0


@pytest.mark.unit
def test_consol_empty_when_no_warm_blocks(consol_on: Path) -> None:
    db = consol_on / "index.db"
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE block_recall_tier (block_id TEXT PRIMARY KEY, tier TEXT NOT NULL, last_seen_at TEXT NOT NULL, last_surprise REAL)"
        )
        conn.commit()
    p = plan_consolidation(consol_on)
    assert p.total_proposals == 0


# ---------------------------------------------------------------------------
# Consolidation — demotion path (low surprise → COLD)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_low_surprise_demotes_to_cold(consol_on: Path) -> None:
    _seed_warm(
        consol_on,
        [
            ("B-bored", 0.05),  # below default threshold (0.3)
            ("B-active", 0.8),  # above threshold
        ],
    )
    p = plan_consolidation(consol_on)
    demoted_ids = {bid for bid, _t, _r in p.demotions}
    assert "B-bored" in demoted_ids
    assert "B-active" not in demoted_ids
    # Tier target is COLD.
    assert all(t is RecallTier.COLD for _bid, t, _r in p.demotions)


@pytest.mark.unit
def test_demotion_threshold_is_configurable(consol_on: Path) -> None:
    _seed_warm(
        consol_on,
        [("B-1", 0.4), ("B-2", 0.6), ("B-3", 0.2)],
    )
    cfg = ConsolidationConfig(demotion_threshold=0.5)
    p = plan_consolidation(consol_on, cfg=cfg)
    demoted = {bid for bid, _t, _r in p.demotions}
    assert demoted == {"B-1", "B-3"}


@pytest.mark.unit
def test_no_embeddings_means_no_promotions(consol_on: Path) -> None:
    """Without candidate embeddings the planner can demote but not cluster."""
    _seed_warm(
        consol_on,
        [("B-1", 0.05), ("B-2", 0.8), ("B-3", 0.9)],
    )
    p = plan_consolidation(consol_on)
    assert p.promotions == []
    assert any(bid == "B-1" for bid, _t, _r in p.demotions)


# ---------------------------------------------------------------------------
# Consolidation — clustering + promotion path
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_cluster_promotes_centroid_member(consol_on: Path) -> None:
    """3-block cluster with one obvious centroid → that block gets promoted."""
    _seed_warm(
        consol_on,
        [
            ("B-1", 0.7),
            ("B-2", 0.7),
            ("B-3", 0.7),
        ],
    )
    embeddings = {
        "B-1": [1.0, 0.0],
        "B-2": [0.99, 0.01],  # nearly identical to B-1
        "B-3": [0.98, 0.02],  # also nearly identical
    }
    p = plan_consolidation(
        consol_on,
        candidate_embeddings=embeddings,
        cfg=ConsolidationConfig(cluster_count=1, min_cluster_size=3),
    )
    # All three cluster together; representative chosen by lowest surprise
    # to cluster centroid.
    assert len(p.promotions) == 1
    promoted_id = p.promotions[0][0]
    assert promoted_id in {"B-1", "B-2", "B-3"}
    # cluster_summaries reports member list.
    assert len(p.cluster_summaries) == 1
    summary = p.cluster_summaries[0]
    assert summary["size"] == 3
    assert sorted(summary["members"]) == ["B-1", "B-2", "B-3"]


@pytest.mark.unit
def test_cluster_below_min_size_does_not_promote(consol_on: Path) -> None:
    _seed_warm(
        consol_on,
        [("B-1", 0.7), ("B-2", 0.7)],
    )
    embeddings = {"B-1": [1.0, 0.0], "B-2": [-1.0, 0.0]}
    p = plan_consolidation(
        consol_on,
        candidate_embeddings=embeddings,
        cfg=ConsolidationConfig(cluster_count=2, min_cluster_size=3),
    )
    assert p.promotions == []


@pytest.mark.unit
def test_promotion_target_is_cold(consol_on: Path) -> None:
    _seed_warm(
        consol_on,
        [("B-1", 0.7), ("B-2", 0.7), ("B-3", 0.7)],
    )
    embeddings = {"B-1": [1.0, 0.0], "B-2": [1.0, 0.0], "B-3": [1.0, 0.0]}
    p = plan_consolidation(
        consol_on,
        candidate_embeddings=embeddings,
        cfg=ConsolidationConfig(cluster_count=1, min_cluster_size=3),
    )
    assert p.promotions[0][1] is RecallTier.COLD


# ---------------------------------------------------------------------------
# Plan immutability + total
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_consolidation_plan_is_frozen() -> None:
    p = ConsolidationPlan()
    with pytest.raises((AttributeError, Exception)):
        p.promotions = []  # type: ignore[misc]


@pytest.mark.unit
def test_consolidation_plan_total_proposals_counts_both() -> None:
    p = ConsolidationPlan(
        promotions=[("B-1", RecallTier.COLD, "x")],
        demotions=[("B-2", RecallTier.COLD, "y"), ("B-3", RecallTier.COLD, "z")],
    )
    assert p.total_proposals == 3


@pytest.mark.unit
def test_default_config_round_trips() -> None:
    assert DEFAULT_CONSOLIDATION_CONFIG.cluster_count == 5
    assert DEFAULT_CONSOLIDATION_CONFIG.min_cluster_size == 3
    assert DEFAULT_CONSOLIDATION_CONFIG.demotion_threshold == 0.3
