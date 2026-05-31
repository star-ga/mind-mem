"""Tests for the v4 surprise-weighted retrieval scoring module."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from mind_mem.v4 import FeatureDisabledError
from mind_mem.v4.surprise_retrieval import (
    DEFAULT_PROMOTE_THRESHOLD,
    FLAG,
    centroid,
    compute_surprise,
    should_promote_on_surprise,
    surprise_threshold,
)


@pytest.fixture
def cfg_on(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cfg = {"v4": {FLAG: {"enabled": True}}}
    (tmp_path / "mind-mem.json").write_text(json.dumps(cfg), encoding="utf-8")
    monkeypatch.setenv("MIND_MEM_CONFIG", str(tmp_path / "mind-mem.json"))
    return tmp_path


@pytest.fixture
def cfg_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cfg = {"v4": {FLAG: {"enabled": False}}}
    (tmp_path / "mind-mem.json").write_text(json.dumps(cfg), encoding="utf-8")
    monkeypatch.setenv("MIND_MEM_CONFIG", str(tmp_path / "mind-mem.json"))
    return tmp_path


# ---------------------------------------------------------------------------
# compute_surprise — math
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_identical_vectors_zero_surprise() -> None:
    v = [1.0, 2.0, 3.0]
    assert compute_surprise(v, v) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.unit
def test_orthogonal_vectors_half_surprise() -> None:
    """cosine_sim = 0 ⇒ surprise = 0.5."""
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert compute_surprise(a, b) == pytest.approx(0.5, abs=1e-9)


@pytest.mark.unit
def test_opposite_vectors_max_surprise() -> None:
    """cosine_sim = -1 ⇒ surprise = 1.0."""
    a = [1.0, 0.0]
    b = [-1.0, 0.0]
    assert compute_surprise(a, b) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.unit
def test_partial_alignment_yields_partial_surprise() -> None:
    """45° apart ⇒ cos_sim = √2/2 ⇒ surprise = (1 - √2/2)/2 ≈ 0.146."""
    a = [1.0, 0.0]
    b = [1.0, 1.0]
    expected = (1.0 - (math.sqrt(2.0) / 2.0)) / 2.0
    assert compute_surprise(a, b) == pytest.approx(expected, abs=1e-9)


@pytest.mark.unit
def test_scale_invariance() -> None:
    """Cosine distance ignores magnitude — only direction matters."""
    a = [1.0, 0.0, 0.0]
    b1 = [1.0, 1.0, 0.0]
    b2 = [100.0, 100.0, 0.0]
    assert compute_surprise(a, b1) == pytest.approx(compute_surprise(a, b2), abs=1e-9)


@pytest.mark.unit
def test_surprise_is_symmetric() -> None:
    a = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0]
    b = [2.0, 7.0, 1.0, 8.0, 2.0, 8.0]
    assert compute_surprise(a, b) == pytest.approx(compute_surprise(b, a), abs=1e-9)


@pytest.mark.unit
def test_surprise_always_in_unit_interval() -> None:
    cases = [
        ([1.0, 0.0], [1.0, 0.0]),
        ([1.0, 0.0], [-1.0, 0.0]),
        ([0.5, 0.7, -0.3], [-0.4, 0.9, 0.1]),
    ]
    for a, b in cases:
        s = compute_surprise(a, b)
        assert 0.0 <= s <= 1.0


# ---------------------------------------------------------------------------
# compute_surprise — degenerate inputs
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_none_context_returns_mild_surprise() -> None:
    assert compute_surprise([1.0, 0.0], None) == 0.5


@pytest.mark.unit
def test_empty_candidate_returns_mild_surprise() -> None:
    assert compute_surprise([], [1.0, 0.0]) == 0.5


@pytest.mark.unit
def test_empty_context_returns_mild_surprise() -> None:
    assert compute_surprise([1.0, 0.0], []) == 0.5


@pytest.mark.unit
def test_mismatched_dimensions_return_mild_surprise() -> None:
    assert compute_surprise([1.0, 0.0], [1.0, 0.0, 0.0]) == 0.5


@pytest.mark.unit
def test_zero_norm_returns_mild_surprise() -> None:
    """A zero vector has no direction, so cosine distance is undefined."""
    assert compute_surprise([0.0, 0.0], [1.0, 1.0]) == 0.5
    assert compute_surprise([1.0, 1.0], [0.0, 0.0]) == 0.5


# ---------------------------------------------------------------------------
# centroid
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_centroid_single_vector_round_trips() -> None:
    v = [1.0, 2.0, 3.0]
    assert centroid([v]) == [1.0, 2.0, 3.0]


@pytest.mark.unit
def test_centroid_two_vectors_is_componentwise_mean() -> None:
    a = [1.0, 0.0, -1.0]
    b = [3.0, 4.0, 1.0]
    assert centroid([a, b]) == [2.0, 2.0, 0.0]


@pytest.mark.unit
def test_centroid_skips_empty_vectors() -> None:
    """An empty vector in the iterable is ignored, not an error."""
    assert centroid([[], [1.0, 2.0], [3.0, 4.0]]) == [2.0, 3.0]


@pytest.mark.unit
def test_centroid_returns_none_on_dimension_mismatch() -> None:
    assert centroid([[1.0, 0.0], [1.0, 0.0, 0.0]]) is None


@pytest.mark.unit
def test_centroid_empty_iterable_returns_none() -> None:
    assert centroid([]) is None


@pytest.mark.unit
def test_centroid_all_empty_returns_none() -> None:
    assert centroid([[], [], []]) is None


@pytest.mark.unit
def test_centroid_then_surprise_round_trip() -> None:
    """A candidate identical to the centroid produces zero surprise."""
    v1 = [1.0, 0.0, 0.0]
    v2 = [0.0, 1.0, 0.0]
    c = centroid([v1, v2])
    assert c is not None
    # The centroid is [0.5, 0.5, 0.0] — same direction as itself.
    assert compute_surprise(c, c) == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Promotion gate — flag gating
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_promote_raises_when_flag_off(cfg_off: Path) -> None:
    with pytest.raises(FeatureDisabledError):
        should_promote_on_surprise(0.9)


@pytest.mark.unit
def test_compute_surprise_works_with_flag_off(cfg_off: Path) -> None:
    """Pure math is not gated — only the *act* of promoting is."""
    assert compute_surprise([1.0, 0.0], [1.0, 0.0]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Promotion gate — threshold semantics
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_promote_above_threshold_is_true(cfg_on: Path) -> None:
    assert should_promote_on_surprise(0.9) is True


@pytest.mark.unit
def test_promote_at_threshold_is_true(cfg_on: Path) -> None:
    """Boundary is inclusive: surprise == threshold ⇒ promote."""
    t = surprise_threshold()
    assert should_promote_on_surprise(t) is True


@pytest.mark.unit
def test_promote_below_threshold_is_false(cfg_on: Path) -> None:
    t = surprise_threshold()
    assert should_promote_on_surprise(max(0.0, t - 0.1)) is False


@pytest.mark.unit
def test_promote_with_explicit_threshold(cfg_on: Path) -> None:
    """Per-call override beats the configured value."""
    assert should_promote_on_surprise(0.5, threshold=0.4) is True
    assert should_promote_on_surprise(0.5, threshold=0.9) is False


# ---------------------------------------------------------------------------
# Threshold loader
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_default_threshold_value() -> None:
    """Documented default to keep tests + tier_memory in sync."""
    assert DEFAULT_PROMOTE_THRESHOLD == 0.65


@pytest.mark.unit
def test_threshold_loads_from_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = {"v4": {FLAG: {"enabled": True, "promote_threshold": 0.42}}}
    (tmp_path / "mind-mem.json").write_text(json.dumps(cfg), encoding="utf-8")
    monkeypatch.setenv("MIND_MEM_CONFIG", str(tmp_path / "mind-mem.json"))
    assert surprise_threshold() == pytest.approx(0.42)


@pytest.mark.unit
def test_threshold_falls_back_on_bad_value(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = {"v4": {FLAG: {"enabled": True, "promote_threshold": "very surprising"}}}
    (tmp_path / "mind-mem.json").write_text(json.dumps(cfg), encoding="utf-8")
    monkeypatch.setenv("MIND_MEM_CONFIG", str(tmp_path / "mind-mem.json"))
    assert surprise_threshold() == DEFAULT_PROMOTE_THRESHOLD


@pytest.mark.unit
def test_threshold_clamps_to_unit_interval(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A wild value can't accidentally promote everything (>1) or nothing (<0)."""
    for bad, expected in [(2.5, 1.0), (-3.0, 0.0)]:
        cfg = {"v4": {FLAG: {"enabled": True, "promote_threshold": bad}}}
        (tmp_path / "mind-mem.json").write_text(json.dumps(cfg), encoding="utf-8")
        monkeypatch.setenv("MIND_MEM_CONFIG", str(tmp_path / "mind-mem.json"))
        assert surprise_threshold() == expected


@pytest.mark.unit
def test_threshold_returns_default_when_block_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No config block at all ⇒ default."""
    cfg = {"v4": {FLAG: {"enabled": True}}}
    (tmp_path / "mind-mem.json").write_text(json.dumps(cfg), encoding="utf-8")
    monkeypatch.setenv("MIND_MEM_CONFIG", str(tmp_path / "mind-mem.json"))
    assert surprise_threshold() == DEFAULT_PROMOTE_THRESHOLD
