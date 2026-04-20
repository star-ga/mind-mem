"""Tests for v3.2.0 distributed recall cache (LRU + Redis)."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from mind_mem.recall_cache import (
    LRUCache,
    RecallCache,
    cached_recall,
    invalidate,
    make_cache_key,
    reset_singleton,
)


@pytest.fixture(autouse=True)
def reset_cache_singleton() -> None:
    reset_singleton()
    yield
    reset_singleton()


class TestCacheKey:
    def test_stable_for_identical_inputs(self) -> None:
        a = make_cache_key("hello", namespace="ns", limit=10, backend="bm25")
        b = make_cache_key("hello", namespace="ns", limit=10, backend="bm25")
        assert a == b

    def test_different_query_different_key(self) -> None:
        a = make_cache_key("hello")
        b = make_cache_key("world")
        assert a != b

    def test_different_namespace_different_key(self) -> None:
        a = make_cache_key("hello", namespace="tenant-a")
        b = make_cache_key("hello", namespace="tenant-b")
        assert a != b

    def test_different_backend_different_key(self) -> None:
        a = make_cache_key("q", backend="bm25")
        b = make_cache_key("q", backend="hybrid")
        assert a != b

    def test_key_contains_namespace_prefix(self) -> None:
        key = make_cache_key("q", namespace="mytenant")
        assert "mindmem:recall:mytenant:" in key


class TestLRUCache:
    def test_set_and_get(self) -> None:
        cache = LRUCache()
        cache.set("k", "v")
        assert cache.get("k") == "v"

    def test_missing_returns_none(self) -> None:
        assert LRUCache().get("missing") is None

    def test_lru_eviction_when_full(self) -> None:
        cache = LRUCache(max_entries=2)
        cache.set("a", "1")
        cache.set("b", "2")
        cache.set("c", "3")  # evicts "a" (LRU)
        assert cache.get("a") is None
        assert cache.get("b") == "2"
        assert cache.get("c") == "3"

    def test_get_updates_recency(self) -> None:
        cache = LRUCache(max_entries=2)
        cache.set("a", "1")
        cache.set("b", "2")
        cache.get("a")  # touch a
        cache.set("c", "3")  # evicts "b" now
        assert cache.get("a") == "1"
        assert cache.get("b") is None
        assert cache.get("c") == "3"

    def test_ttl_expiry(self) -> None:
        cache = LRUCache()
        cache.set("k", "v", ttl_seconds=1)
        assert cache.get("k") == "v"
        # Fake the clock.
        with patch("mind_mem.recall_cache.time.time", return_value=time.time() + 10):
            assert cache.get("k") is None

    def test_ttl_zero_means_no_expiry(self) -> None:
        cache = LRUCache()
        cache.set("k", "v", ttl_seconds=0)
        with patch("mind_mem.recall_cache.time.time", return_value=time.time() + 1e9):
            assert cache.get("k") == "v"

    def test_delete(self) -> None:
        cache = LRUCache()
        cache.set("k", "v")
        assert cache.delete("k") is True
        assert cache.delete("k") is False

    def test_invalidate_namespace_removes_matching(self) -> None:
        cache = LRUCache()
        cache.set("mindmem:recall:ns1:abc", "v1")
        cache.set("mindmem:recall:ns1:def", "v2")
        cache.set("mindmem:recall:ns2:xyz", "v3")
        dropped = cache.invalidate_namespace("ns1")
        assert dropped == 2
        assert cache.get("mindmem:recall:ns1:abc") is None
        assert cache.get("mindmem:recall:ns2:xyz") == "v3"

    def test_clear(self) -> None:
        cache = LRUCache()
        cache.set("a", "1")
        cache.set("b", "2")
        cache.clear()
        assert len(cache) == 0


class TestRecallCache:
    def test_lru_only_when_no_redis_url(self) -> None:
        cache = RecallCache()
        assert cache.backend_label == "lru"

    def test_gets_from_lru_first(self) -> None:
        cache = RecallCache()
        cache.set("k", "v")
        assert cache.get("k") == "v"

    def test_miss_returns_none(self) -> None:
        assert RecallCache().get("missing") is None

    def test_redis_init_failure_falls_back_to_lru(self) -> None:
        with patch("importlib.util.find_spec", return_value=object()):
            # Redis module pretends to exist.
            with patch("mind_mem.recall_cache._RedisCache", side_effect=RuntimeError("boom")):
                cache = RecallCache(redis_url="redis://fake:6379")
                # Still usable — LRU stays up.
                assert cache.backend_label == "lru"
                cache.set("k", "v")
                assert cache.get("k") == "v"

    def test_redis_gets_populated_on_miss_when_l2_hit(self) -> None:
        """L2 hit repopulates L1."""
        fake_redis = MagicMock()
        fake_redis.get.return_value = "from-redis"
        with patch("importlib.util.find_spec", return_value=object()):
            with patch("mind_mem.recall_cache._RedisCache", return_value=fake_redis):
                cache = RecallCache(redis_url="redis://fake:6379")
                assert cache.get("missing-in-lru") == "from-redis"
                # Second get hits LRU — no additional Redis call.
                fake_redis.get.reset_mock()
                assert cache.get("missing-in-lru") == "from-redis"
                fake_redis.get.assert_not_called()

    def test_invalidate_namespace_hits_both_tiers(self) -> None:
        fake_redis = MagicMock()
        fake_redis.get.return_value = None
        fake_redis.invalidate_namespace.return_value = 5
        with patch("importlib.util.find_spec", return_value=object()):
            with patch("mind_mem.recall_cache._RedisCache", return_value=fake_redis):
                cache = RecallCache(redis_url="redis://fake:6379")
                # Seed LRU
                cache.set("mindmem:recall:default:foo", "bar")
                dropped = cache.invalidate_namespace("default")
                # 1 from LRU + 5 from Redis
                assert dropped == 6


class TestCachedRecall:
    def test_wraps_inner_and_caches(self) -> None:
        call_count = {"n": 0}

        def inner(query: str, *, limit: int, active_only: bool, backend: str) -> str:
            call_count["n"] += 1
            return f'{{"query": "{query}", "n": {call_count["n"]}}}'

        first = cached_recall(inner, "hello", limit=5)
        second = cached_recall(inner, "hello", limit=5)
        assert first == second
        assert call_count["n"] == 1  # only one inner call — second came from cache

    def test_different_limits_bypass_cache(self) -> None:
        call_count = {"n": 0}

        def inner(query: str, *, limit: int, active_only: bool, backend: str) -> str:
            call_count["n"] += 1
            return f"limit={limit}"

        cached_recall(inner, "q", limit=5)
        cached_recall(inner, "q", limit=10)
        assert call_count["n"] == 2

    def test_ttl_honored(self) -> None:
        call_count = {"n": 0}

        def inner(query: str, *, limit: int, active_only: bool, backend: str) -> str:
            call_count["n"] += 1
            return "val"

        cached_recall(inner, "q", ttl_seconds=1)
        with patch("mind_mem.recall_cache.time.time", return_value=time.time() + 10):
            cached_recall(inner, "q", ttl_seconds=1)
        assert call_count["n"] == 2

    def test_invalidate_forces_refresh(self) -> None:
        call_count = {"n": 0}

        def inner(query: str, *, limit: int, active_only: bool, backend: str) -> str:
            call_count["n"] += 1
            return "val"

        cached_recall(inner, "q", namespace="my-ns")
        invalidate("my-ns")
        cached_recall(inner, "q", namespace="my-ns")
        assert call_count["n"] == 2
