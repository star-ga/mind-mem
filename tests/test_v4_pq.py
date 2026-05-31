"""Tests for v4 product-quantization (PQ) encoding."""

from __future__ import annotations

import json
import math
import random
import sqlite3
from pathlib import Path

import pytest
from mind_mem.v4 import FeatureDisabledError
from mind_mem.v4.pq import (
    DEFAULT_PQ_CONFIG,
    FLAG,
    Codebook,
    PQConfig,
    asymmetric_distance,
    decode,
    encode,
    ensure_pq_schema,
    load_code,
    load_codebook,
    set_codebook_trainer,
    store_code,
    store_codebook,
    train_codebook,
)


@pytest.fixture
def cfg_on(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cfg = {"v4": {FLAG: {"enabled": True, "subvectors": 4, "centroids": 4, "kmeans_iters": 25, "kmeans_seed": 7}}}
    (tmp_path / "mind-mem.json").write_text(json.dumps(cfg), encoding="utf-8")
    monkeypatch.setenv("MIND_MEM_CONFIG", str(tmp_path / "mind-mem.json"))
    return tmp_path


@pytest.fixture
def cfg_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cfg = {"v4": {FLAG: {"enabled": False}}}
    (tmp_path / "mind-mem.json").write_text(json.dumps(cfg), encoding="utf-8")
    monkeypatch.setenv("MIND_MEM_CONFIG", str(tmp_path / "mind-mem.json"))
    return tmp_path


def _gauss_vector(rng: random.Random, dim: int) -> list[float]:
    return [rng.gauss(0.0, 1.0) for _ in range(dim)]


# ---------------------------------------------------------------------------
# Flag gating
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_flag_off_blocks_train(cfg_off: Path) -> None:
    with pytest.raises(FeatureDisabledError):
        train_codebook([[1.0]], DEFAULT_PQ_CONFIG)


@pytest.mark.unit
def test_flag_off_blocks_encode(cfg_off: Path) -> None:
    cb = Codebook(cfg=DEFAULT_PQ_CONFIG, centroids=tuple())
    with pytest.raises(FeatureDisabledError):
        encode([1.0], cb)


@pytest.mark.unit
def test_flag_off_blocks_decode(cfg_off: Path) -> None:
    cb = Codebook(cfg=DEFAULT_PQ_CONFIG, centroids=tuple())
    with pytest.raises(FeatureDisabledError):
        decode(b"\x00\x00", cb)


@pytest.mark.unit
def test_flag_off_blocks_distance(cfg_off: Path) -> None:
    cb = Codebook(cfg=DEFAULT_PQ_CONFIG, centroids=tuple())
    with pytest.raises(FeatureDisabledError):
        asymmetric_distance([1.0], b"\x00", cb)


@pytest.mark.unit
def test_flag_off_blocks_persistence(cfg_off: Path) -> None:
    with pytest.raises(FeatureDisabledError):
        ensure_pq_schema(cfg_off)
    with pytest.raises(FeatureDisabledError):
        store_codebook(cfg_off, "x", Codebook(cfg=DEFAULT_PQ_CONFIG, centroids=tuple()))
    with pytest.raises(FeatureDisabledError):
        load_codebook(cfg_off, "x")


# ---------------------------------------------------------------------------
# Train / encode / decode round-trip
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_empty_training_yields_empty_codebook(cfg_on: Path) -> None:
    cb = train_codebook([], DEFAULT_PQ_CONFIG)
    assert cb.centroids == ()


@pytest.mark.unit
def test_train_rejects_non_divisible_dim(cfg_on: Path) -> None:
    cfg = PQConfig(subvectors=3, centroids=2)
    with pytest.raises(ValueError):
        train_codebook([[1.0, 2.0, 3.0, 4.0]], cfg)  # 4 not divisible by 3


@pytest.mark.unit
def test_train_with_unique_points_yields_perfect_reconstruction(cfg_on: Path) -> None:
    """K equal to N training points → each point is its own centroid →
    encode/decode round-trips exactly."""
    cfg = PQConfig(subvectors=2, centroids=4)  # K = number of points
    rng = random.Random(0)
    points = [_gauss_vector(rng, 4) for _ in range(4)]
    cb = train_codebook(points, cfg)
    for p in points:
        recon = decode(encode(p, cb), cb)
        assert len(recon) == 4
        for a, b in zip(p, recon):
            assert a == pytest.approx(b, abs=1e-6)


@pytest.mark.unit
def test_train_quantizes_clusters(cfg_on: Path) -> None:
    """Two well-separated clusters should round-trip to their centroids,
    not their exact original positions."""
    cfg = PQConfig(subvectors=2, centroids=2)
    points = [
        [10.0, 10.0, 10.0, 10.0],
        [10.1, 10.0, 10.0, 10.1],
        [-10.0, -10.0, -10.0, -10.0],
        [-10.1, -10.0, -10.0, -10.1],
    ]
    cb = train_codebook(points, cfg)
    encoded = encode([10.05, 10.0, 10.0, 10.05], cb)
    decoded = decode(encoded, cb)
    # Should reconstruct to ~+10 cluster centroid, not the -10 cluster.
    assert all(v > 0 for v in decoded)


@pytest.mark.unit
def test_encode_returns_m_bytes(cfg_on: Path) -> None:
    cfg = PQConfig(subvectors=4, centroids=4)
    points = [[float(i + j) for j in range(8)] for i in range(8)]
    cb = train_codebook(points, cfg)
    code = encode(points[0], cb)
    assert len(code) == cfg.subvectors


@pytest.mark.unit
def test_encode_mismatched_dim_returns_empty_bytes(cfg_on: Path) -> None:
    cfg = PQConfig(subvectors=2, centroids=2)
    points = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
    cb = train_codebook(points, cfg)
    assert encode([1.0, 2.0], cb) == b""  # wrong dim


@pytest.mark.unit
def test_decode_handles_corrupt_code(cfg_on: Path) -> None:
    cfg = PQConfig(subvectors=2, centroids=2)
    points = [[1.0, 2.0, 3.0, 4.0]]
    cb = train_codebook(points, cfg)
    # Code byte points to a centroid index out of range.
    bad = bytes([99, 0])
    out = decode(bad, cb)
    # First subvector becomes zeros (corrupt-fallback); second uses
    # centroid 0.
    assert len(out) == 4


# ---------------------------------------------------------------------------
# Asymmetric distance
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_asymmetric_distance_to_self_is_low(cfg_on: Path) -> None:
    cfg = PQConfig(subvectors=2, centroids=4)
    rng = random.Random(0)
    points = [_gauss_vector(rng, 4) for _ in range(4)]
    cb = train_codebook(points, cfg)
    code = encode(points[0], cb)
    self_d = asymmetric_distance(points[0], code, cb)
    other_d = asymmetric_distance(points[3], code, cb)
    assert self_d < other_d


@pytest.mark.unit
def test_asymmetric_distance_handles_bad_inputs(cfg_on: Path) -> None:
    cfg = PQConfig(subvectors=2, centroids=2)
    cb = train_codebook([[1.0, 2.0, 3.0, 4.0]], cfg)
    # Mismatched query dim.
    assert asymmetric_distance([1.0], encode([1.0, 2.0, 3.0, 4.0], cb), cb) == math.inf
    # Mismatched code length.
    assert asymmetric_distance([1.0, 2.0, 3.0, 4.0], b"x", cb) == math.inf
    # Empty code.
    assert asymmetric_distance([1.0, 2.0, 3.0, 4.0], b"", cb) == math.inf


# ---------------------------------------------------------------------------
# Compression ratio
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_compression_ratio_at_default_config() -> None:
    """For 768-dim float32 input, default M=32 K=256 yields 96× compression."""
    raw = 768 * 4  # float32
    pq = DEFAULT_PQ_CONFIG.subvectors  # 1 byte per subvector at K=256
    assert raw == 3072
    assert pq == 32
    assert raw // pq == 96


# ---------------------------------------------------------------------------
# Persistence round-trip
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_codebook_round_trips_through_sqlite(cfg_on: Path) -> None:
    cfg = PQConfig(subvectors=2, centroids=4)
    rng = random.Random(7)
    points = [_gauss_vector(rng, 4) for _ in range(8)]
    cb = train_codebook(points, cfg)
    store_codebook(cfg_on, "default", cb)
    loaded = load_codebook(cfg_on, "default")
    assert loaded is not None
    assert loaded.cfg == cb.cfg
    assert loaded.subvector_dim == cb.subvector_dim
    # Centroids round-trip exactly (float32 precision, but small values).
    for m in range(cfg.subvectors):
        for k in range(cfg.centroids):
            for a, b in zip(loaded.centroids[m][k], cb.centroids[m][k]):
                assert a == pytest.approx(b, abs=1e-5)


@pytest.mark.unit
def test_load_codebook_returns_none_when_absent(cfg_on: Path) -> None:
    ensure_pq_schema(cfg_on)
    assert load_codebook(cfg_on, "no-such-name") is None


@pytest.mark.unit
def test_load_codebook_returns_none_when_db_missing(cfg_on: Path) -> None:
    assert load_codebook(cfg_on, "default") is None


@pytest.mark.unit
def test_code_round_trips_through_sqlite(cfg_on: Path) -> None:
    cfg = PQConfig(subvectors=2, centroids=4)
    points = [[1.0, 2.0, 3.0, 4.0]]
    cb = train_codebook(points, cfg)
    code = encode(points[0], cb)
    store_code(cfg_on, "B-1", "default", code)
    loaded = load_code(cfg_on, "B-1", "default")
    assert loaded == code


@pytest.mark.unit
def test_load_code_returns_none_when_absent(cfg_on: Path) -> None:
    ensure_pq_schema(cfg_on)
    assert load_code(cfg_on, "B-never", "default") is None


@pytest.mark.unit
def test_store_code_replaces_on_duplicate(cfg_on: Path) -> None:
    cfg = PQConfig(subvectors=2, centroids=2)
    cb = train_codebook([[1.0, 2.0, 3.0, 4.0]], cfg)
    code1 = encode([1.0, 2.0, 3.0, 4.0], cb)
    code2 = bytes([1, 0])  # arbitrary
    store_code(cfg_on, "B-1", "default", code1)
    store_code(cfg_on, "B-1", "default", code2)
    assert load_code(cfg_on, "B-1", "default") == code2


# ---------------------------------------------------------------------------
# Schema bootstrap
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_ensure_schema_idempotent(cfg_on: Path) -> None:
    ensure_pq_schema(cfg_on)
    ensure_pq_schema(cfg_on)
    ensure_pq_schema(cfg_on)
    db = cfg_on / "index.db"
    with sqlite3.connect(db) as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'pq_%'")}
    assert tables == {"pq_codebook", "pq_codes"}


# ---------------------------------------------------------------------------
# Pluggable trainer
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_set_codebook_trainer_swaps_implementation(cfg_on: Path) -> None:
    """Production deployments install a faster trainer; the encode/decode
    surface uses whatever codebook the trainer produced."""
    captured: list[Sequence[Sequence[float]]] = []

    def fake_trainer(training, cfg):
        captured.append(training)
        # centroids[m][k] = D/M-dim centroid k for subvector position m.
        # For M=1, K=1, dim=2: centroids = ((c0,)) where c0 = (0.0, 0.0).
        return Codebook(cfg=cfg, centroids=(((0.0, 0.0),),))

    set_codebook_trainer(fake_trainer)
    try:
        cb = train_codebook([[1.0, 2.0]], PQConfig(subvectors=1, centroids=1))
        assert cb.centroids == (((0.0, 0.0),),)
        assert captured == [[[1.0, 2.0]]]
    finally:
        # Restore default trainer to avoid bleeding into other tests.
        from mind_mem.v4.pq import _stdlib_train_codebook

        set_codebook_trainer(_stdlib_train_codebook)


# Sequence type-import for the fake_trainer signature above.
from collections.abc import Sequence  # noqa: E402  (used in test fixture)
