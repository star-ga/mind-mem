# Copyright 2026 STARGA, Inc.
"""Tests for LLM noise profiler — per-provider, per-domain reliability tracking.

Red-Green-Refactor: these tests were written before the implementation.
"""

from __future__ import annotations

import json
import os

import pytest

from mind_mem.llm_noise_profile import LLMNoiseProfiler, NoiseProfile

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def profiler():
    return LLMNoiseProfiler()


@pytest.fixture
def profiler_with_providers():
    p = LLMNoiseProfiler()
    p.register_provider("gpt-4", initial_reliability=0.9)
    p.register_provider("mistral", initial_reliability=0.7)
    p.register_provider("deepseek", initial_reliability=0.6)
    return p


# ---------------------------------------------------------------------------
# NoiseProfile dataclass
# ---------------------------------------------------------------------------


def test_noise_profile_default_fields():
    np = NoiseProfile(provider_id="test-llm")
    assert np.provider_id == "test-llm"
    assert 0.0 <= np.global_reliability <= 1.0
    assert isinstance(np.domain_reliability, dict)
    assert np.total_observations == 0
    assert np.error_count == 0
    assert np.last_calibrated is not None


def test_noise_profile_custom_reliability():
    np = NoiseProfile(provider_id="x", global_reliability=0.85)
    assert np.global_reliability == 0.85


def test_noise_profile_domain_reliability_dict():
    np = NoiseProfile(provider_id="x", domain_reliability={"code": 0.9, "math": 0.6})
    assert np.domain_reliability["code"] == 0.9
    assert np.domain_reliability["math"] == 0.6


# ---------------------------------------------------------------------------
# register_provider
# ---------------------------------------------------------------------------


def test_register_new_provider(profiler):
    profiler.register_provider("llama3", initial_reliability=0.75)
    assert profiler.get_reliability("llama3") == pytest.approx(0.75)


def test_register_provider_default_reliability(profiler):
    profiler.register_provider("qwen")
    assert profiler.get_reliability("qwen") == pytest.approx(0.7)


def test_register_same_provider_twice_is_idempotent(profiler):
    profiler.register_provider("gpt", initial_reliability=0.8)
    profiler.register_provider("gpt", initial_reliability=0.5)  # second call ignored
    assert profiler.get_reliability("gpt") == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# record_outcome — EMA update
# ---------------------------------------------------------------------------


def test_record_correct_outcome_raises_global_reliability(profiler):
    profiler.register_provider("p1", initial_reliability=0.5)
    profiler.record_outcome("p1", "code", was_correct=True)
    # EMA: 0.5 * 0.95 + 1.0 * 0.05 = 0.525
    assert profiler.get_reliability("p1") == pytest.approx(0.525)


def test_record_incorrect_outcome_lowers_global_reliability(profiler):
    profiler.register_provider("p1", initial_reliability=0.8)
    profiler.record_outcome("p1", "code", was_correct=False)
    # EMA: 0.8 * 0.95 + 0.0 * 0.05 = 0.76
    assert profiler.get_reliability("p1") == pytest.approx(0.76)


def test_record_outcome_updates_domain_reliability(profiler):
    profiler.register_provider("p1", initial_reliability=0.7)
    profiler.record_outcome("p1", "math", was_correct=True)
    domain_rel = profiler.get_reliability("p1", domain="math")
    # First domain observation: global_rel * 0.95 + 1.0 * 0.05
    assert domain_rel > 0.7


def test_record_outcome_increments_total_observations(profiler):
    profiler.register_provider("p1")
    profiler.record_outcome("p1", "code", was_correct=True)
    profiler.record_outcome("p1", "code", was_correct=False)
    profile = profiler._profiles["p1"]
    assert profile.total_observations == 2


def test_record_outcome_increments_error_count_on_failure(profiler):
    profiler.register_provider("p1")
    profiler.record_outcome("p1", "code", was_correct=False)
    profile = profiler._profiles["p1"]
    assert profile.error_count == 1


def test_record_outcome_unknown_provider_raises(profiler):
    with pytest.raises(KeyError):
        profiler.record_outcome("nonexistent", "code", was_correct=True)


# ---------------------------------------------------------------------------
# get_reliability
# ---------------------------------------------------------------------------


def test_get_reliability_no_domain_returns_global(profiler_with_providers):
    r = profiler_with_providers.get_reliability("gpt-4")
    assert r == pytest.approx(0.9)


def test_get_reliability_with_domain_falls_back_to_global_when_no_domain_data(
    profiler_with_providers,
):
    r = profiler_with_providers.get_reliability("gpt-4", domain="unseen-domain")
    assert r == pytest.approx(0.9)


def test_get_reliability_unknown_provider_raises(profiler):
    with pytest.raises(KeyError):
        profiler.get_reliability("nobody")


# ---------------------------------------------------------------------------
# get_observation_noise
# ---------------------------------------------------------------------------


def test_noise_is_inverse_of_reliability(profiler):
    profiler.register_provider("p1", initial_reliability=0.8)
    noise = profiler.get_observation_noise("p1")
    assert noise == pytest.approx(1.0 - 0.8)


def test_noise_with_domain(profiler):
    profiler.register_provider("p1", initial_reliability=0.6)
    profiler.record_outcome("p1", "math", was_correct=True)
    noise = profiler.get_observation_noise("p1", domain="math")
    rel = profiler.get_reliability("p1", domain="math")
    assert noise == pytest.approx(1.0 - rel)


def test_noise_clamps_to_zero_for_perfect_reliability(profiler):
    profiler.register_provider("p1", initial_reliability=1.0)
    # Feed many correct answers to push reliability toward 1.0
    for _ in range(50):
        profiler.record_outcome("p1", "x", was_correct=True)
    noise = profiler.get_observation_noise("p1")
    assert noise >= 0.0


# ---------------------------------------------------------------------------
# get_best_provider
# ---------------------------------------------------------------------------


def test_get_best_provider_global(profiler_with_providers):
    best = profiler_with_providers.get_best_provider()
    assert best == "gpt-4"  # highest at 0.9


def test_get_best_provider_by_domain(profiler):
    profiler.register_provider("a", initial_reliability=0.7)
    profiler.register_provider("b", initial_reliability=0.8)
    # Give "a" many correct domain hits to raise its domain reliability above "b"
    for _ in range(40):
        profiler.record_outcome("a", "poetry", was_correct=True)
    best = profiler.get_best_provider(domain="poetry")
    assert best == "a"


def test_get_best_provider_no_providers_raises(profiler):
    with pytest.raises(ValueError, match="no providers"):
        profiler.get_best_provider()


# ---------------------------------------------------------------------------
# ranking
# ---------------------------------------------------------------------------


def test_ranking_sorted_descending(profiler_with_providers):
    ranks = profiler_with_providers.ranking()
    scores = [s for _, s in ranks]
    assert scores == sorted(scores, reverse=True)


def test_ranking_contains_all_providers(profiler_with_providers):
    ranks = profiler_with_providers.ranking()
    ids = {pid for pid, _ in ranks}
    assert ids == {"gpt-4", "mistral", "deepseek"}


def test_ranking_with_domain(profiler_with_providers):
    ranks = profiler_with_providers.ranking(domain="code")
    assert len(ranks) == 3


# ---------------------------------------------------------------------------
# JSON persistence — save / load
# ---------------------------------------------------------------------------


def test_save_and_load_roundtrip(profiler, tmp_path):
    path = str(tmp_path / "noise_profiles.json")
    profiler.register_provider("m1", initial_reliability=0.88)
    profiler.record_outcome("m1", "code", was_correct=True)
    profiler.save(path)

    profiler2 = LLMNoiseProfiler()
    profiler2.load(path)
    assert profiler2.get_reliability("m1") == pytest.approx(profiler.get_reliability("m1"), rel=1e-6)


def test_save_creates_parent_dirs(profiler, tmp_path):
    path = str(tmp_path / "nested" / "dir" / "profiles.json")
    profiler.register_provider("m1")
    profiler.save(path)
    assert os.path.isfile(path)


def test_load_nonexistent_file_is_noop(profiler, tmp_path):
    path = str(tmp_path / "does_not_exist.json")
    profiler.load(path)  # must not raise
    assert len(profiler._profiles) == 0


def test_load_corrupted_json_is_noop(profiler, tmp_path):
    path = str(tmp_path / "bad.json")
    with open(path, "w") as f:
        f.write("{not valid json")
    profiler.load(path)  # must not raise
    assert len(profiler._profiles) == 0


def test_saved_json_has_expected_structure(profiler, tmp_path):
    path = str(tmp_path / "p.json")
    profiler.register_provider("a", initial_reliability=0.65)
    profiler.save(path)
    with open(path) as f:
        data = json.load(f)
    assert "version" in data
    assert "profiles" in data
    assert "a" in data["profiles"]
