"""Tests for the HNSW kind-filtered ANN index."""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

import pytest
from _platform_compat import is_root

from mind_mem.v4 import FeatureDisabledError
from mind_mem.v4.hnsw_kind_index import FLAG as HNSW_FLAG
from mind_mem.v4.hnsw_kind_index import (
    backend_status,
    ensure_hnsw_schema,
    knn_by_kind,
    register_block_embedding,
)

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
# HNSW — backend honesty + read-path purity (regression)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_backend_status_reports_the_backend_that_actually_serves(hnsw_on: Path) -> None:
    """backend_status used to answer 'sqlite_vec' whenever the extension
    loaded, while every query was served by the brute-force scan — a health
    surface reporting an ANN index that ran no queries."""
    register_block_embedding(hnsw_on, "B-1", "entity", [1.0, 0.0])
    s = backend_status(hnsw_on)
    assert s["backend"] == "brute_force"
    # Extension availability is still reported, just as its own signal.
    assert isinstance(s["sqlite_vec_available"], bool)


@pytest.mark.unit
def test_knn_does_not_write_to_the_index(hnsw_on: Path) -> None:
    """The read path used to CREATE VIRTUAL TABLE bke_vec on first touch —
    a table it never queried, and a write that fails outright on a
    read-only index."""
    register_block_embedding(hnsw_on, "B-1", "entity", [1.0, 0.0])
    assert knn_by_kind(hnsw_on, "entity", [1.0, 0.0]) != []

    with sqlite3.connect(hnsw_on / "index.db") as conn:
        names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master").fetchall()}
    assert not any(n.startswith("bke_vec") for n in names), f"read path wrote tables: {sorted(names)}"


@pytest.mark.unit
def test_knn_works_against_a_readonly_index(hnsw_on: Path) -> None:
    """First query against a read-only index.db raised
    OperationalError('attempt to write a readonly database')."""

    if is_root():  # pragma: no cover - root ignores the mode bits
        pytest.skip("running as root; file mode does not deny writes")
    register_block_embedding(hnsw_on, "B-1", "entity", [1.0, 0.0])
    db = hnsw_on / "index.db"
    db.chmod(0o444)
    try:
        out = knn_by_kind(hnsw_on, "entity", [1.0, 0.0])
    finally:
        db.chmod(0o644)
    assert [r[0] for r in out] == ["B-1"]


@pytest.mark.unit
def test_knn_warns_when_every_row_is_skipped_on_dimension(hnsw_on: Path, caplog: pytest.LogCaptureFixture) -> None:
    """A dimension mismatch across the whole partition returns [] — the same
    answer as an empty partition. Say which one it is, or an embedder swap
    looks like an empty index."""
    register_block_embedding(hnsw_on, "B-1", "entity", [1.0, 0.0, 0.0])
    with caplog.at_level(logging.WARNING, logger="mind_mem.v4.hnsw_kind_index"):
        out = knn_by_kind(hnsw_on, "entity", [1.0] * 128)
    assert out == []
    assert "skipped_dim_mismatch=1" in caplog.text
    assert "query_dim=128" in caplog.text


@pytest.mark.unit
def test_knn_silent_when_partition_is_genuinely_empty(hnsw_on: Path, caplog: pytest.LogCaptureFixture) -> None:
    """The warning must distinguish the two cases, not fire on both."""
    register_block_embedding(hnsw_on, "B-1", "entity", [1.0, 0.0])
    with caplog.at_level(logging.WARNING, logger="mind_mem.v4.hnsw_kind_index"):
        assert knn_by_kind(hnsw_on, "concept", [1.0, 0.0]) == []
    assert "skipped_dim_mismatch" not in caplog.text
