# Copyright 2026 STARGA, Inc.
"""Tests for the LLM prefix cache (v2.0.0b1 inference acceleration)."""

from __future__ import annotations

import threading
import time

import pytest

from mind_mem import prefix_cache as pc
from mind_mem.prefix_cache import (
    CacheStats,
    PrefixCache,
    all_stats,
    get_cache,
    reset_all,
)


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_rejects_empty_namespace(self) -> None:
        with pytest.raises(ValueError, match="namespace"):
            PrefixCache("")

    def test_rejects_zero_max_size(self) -> None:
        with pytest.raises(ValueError, match="max_size"):
            PrefixCache("ns", max_size=0)

    def test_rejects_negative_max_size(self) -> None:
        with pytest.raises(ValueError, match="max_size"):
            PrefixCache("ns", max_size=-1)

    def test_accepts_none_ttl(self) -> None:
        cache = PrefixCache("ns", ttl_seconds=None)
        assert cache.stats().ttl_seconds is None

    def test_zero_ttl_treated_as_disabled(self) -> None:
        cache = PrefixCache("ns", ttl_seconds=0)
        assert cache.stats().ttl_seconds is None

    def test_negative_ttl_treated_as_disabled(self) -> None:
        cache = PrefixCache("ns", ttl_seconds=-5)
        assert cache.stats().ttl_seconds is None


# ---------------------------------------------------------------------------
# Hit / miss
# ---------------------------------------------------------------------------


class TestHitMiss:
    def test_get_miss_returns_false_none(self) -> None:
        cache = PrefixCache("ns")
        hit, value = cache.get("prefix", {"q": "a"})
        assert hit is False
        assert value is None

    def test_put_then_get_hit(self) -> None:
        cache = PrefixCache("ns")
        cache.put("prefix", {"q": "a"}, "response")
        hit, value = cache.get("prefix", {"q": "a"})
        assert hit is True
        assert value == "response"

    def test_different_prefix_separate_entry(self) -> None:
        cache = PrefixCache("ns")
        cache.put("prefix-a", {"q": "x"}, "A")
        cache.put("prefix-b", {"q": "x"}, "B")
        assert cache.get("prefix-a", {"q": "x"}) == (True, "A")
        assert cache.get("prefix-b", {"q": "x"}) == (True, "B")

    def test_different_payload_separate_entry(self) -> None:
        cache = PrefixCache("ns")
        cache.put("p", {"q": "a"}, "A")
        cache.put("p", {"q": "b"}, "B")
        assert cache.get("p", {"q": "a"}) == (True, "A")
        assert cache.get("p", {"q": "b"}) == (True, "B")

    def test_dict_order_does_not_affect_key(self) -> None:
        cache = PrefixCache("ns")
        cache.put("p", {"a": 1, "b": 2}, "value")
        # Same payload, different insertion order in the dict literal
        hit, value = cache.get("p", {"b": 2, "a": 1})
        assert hit is True
        assert value == "value"

    def test_bytes_payload_supported(self) -> None:
        cache = PrefixCache("ns")
        cache.put("p", b"raw bytes", "v")
        assert cache.get("p", b"raw bytes") == (True, "v")

    def test_none_payload_supported(self) -> None:
        cache = PrefixCache("ns")
        cache.put("p", None, "v")
        assert cache.get("p", None) == (True, "v")

    def test_put_overwrites_existing(self) -> None:
        cache = PrefixCache("ns")
        cache.put("p", "x", "first")
        cache.put("p", "x", "second")
        hit, value = cache.get("p", "x")
        assert hit is True
        assert value == "second"


# ---------------------------------------------------------------------------
# LRU eviction
# ---------------------------------------------------------------------------


class TestLRUEviction:
    def test_eviction_when_full(self) -> None:
        cache = PrefixCache("ns", max_size=2)
        cache.put("p", "a", 1)
        cache.put("p", "b", 2)
        cache.put("p", "c", 3)  # Evicts "a" (LRU)
        assert cache.get("p", "a") == (False, None)
        assert cache.get("p", "b") == (True, 2)
        assert cache.get("p", "c") == (True, 3)

    def test_get_refreshes_lru_order(self) -> None:
        cache = PrefixCache("ns", max_size=2)
        cache.put("p", "a", 1)
        cache.put("p", "b", 2)
        # Touch "a" so "b" becomes LRU
        cache.get("p", "a")
        cache.put("p", "c", 3)  # Evicts "b"
        assert cache.get("p", "a") == (True, 1)
        assert cache.get("p", "b") == (False, None)
        assert cache.get("p", "c") == (True, 3)

    def test_eviction_counter_increments(self) -> None:
        cache = PrefixCache("ns", max_size=1)
        cache.put("p", "a", 1)
        cache.put("p", "b", 2)
        assert cache.stats().evictions == 1


# ---------------------------------------------------------------------------
# TTL expiration
# ---------------------------------------------------------------------------


class TestTTL:
    def test_entry_expires_after_ttl(self, monkeypatch) -> None:
        fake_now = [0.0]

        def fake_monotonic() -> float:
            return fake_now[0]

        monkeypatch.setattr(time, "monotonic", fake_monotonic)
        cache = PrefixCache("ns", ttl_seconds=10.0)
        cache.put("p", "x", "v")

        fake_now[0] = 5.0
        assert cache.get("p", "x") == (True, "v")

        fake_now[0] = 11.0
        assert cache.get("p", "x") == (False, None)

    def test_expiration_counter_increments(self, monkeypatch) -> None:
        fake_now = [0.0]
        monkeypatch.setattr(time, "monotonic", lambda: fake_now[0])
        cache = PrefixCache("ns", ttl_seconds=1.0)
        cache.put("p", "x", "v")
        fake_now[0] = 2.0
        cache.get("p", "x")
        assert cache.stats().expirations == 1

    def test_put_after_expire_recreates_entry(self, monkeypatch) -> None:
        fake_now = [0.0]
        monkeypatch.setattr(time, "monotonic", lambda: fake_now[0])
        cache = PrefixCache("ns", ttl_seconds=1.0)
        cache.put("p", "x", "v1")
        fake_now[0] = 2.0
        cache.get("p", "x")  # expires
        cache.put("p", "x", "v2")
        fake_now[0] = 2.5
        assert cache.get("p", "x") == (True, "v2")


# ---------------------------------------------------------------------------
# Invalidate / clear
# ---------------------------------------------------------------------------


class TestInvalidate:
    def test_invalidate_existing_returns_true(self) -> None:
        cache = PrefixCache("ns")
        cache.put("p", "x", "v")
        assert cache.invalidate("p", "x") is True
        assert cache.get("p", "x") == (False, None)

    def test_invalidate_missing_returns_false(self) -> None:
        cache = PrefixCache("ns")
        assert cache.invalidate("p", "x") is False

    def test_clear_removes_everything_and_resets_counters(self) -> None:
        cache = PrefixCache("ns")
        cache.put("p", "a", 1)
        cache.put("p", "b", 2)
        cache.get("p", "a")
        cache.clear()
        stats = cache.stats()
        assert stats.size == 0
        assert stats.hits == 0
        assert stats.misses == 0


# ---------------------------------------------------------------------------
# CacheStats
# ---------------------------------------------------------------------------


class TestCacheStats:
    def test_hit_rate_zero_when_empty(self) -> None:
        cache = PrefixCache("ns")
        assert cache.stats().hit_rate == 0.0

    def test_hit_rate_computed(self) -> None:
        cache = PrefixCache("ns")
        cache.put("p", "x", "v")
        cache.get("p", "x")  # hit
        cache.get("p", "x")  # hit
        cache.get("p", "y")  # miss
        stats = cache.stats()
        assert stats.hits == 2
        assert stats.misses == 1
        assert abs(stats.hit_rate - 2 / 3) < 1e-9

    def test_as_dict_round_trip(self) -> None:
        cache = PrefixCache("ns", max_size=10, ttl_seconds=30.0)
        d = cache.stats().as_dict()
        assert d["namespace"] == "ns"
        assert d["max_size"] == 10
        assert d["ttl_seconds"] == 30.0
        assert set(d.keys()) >= {
            "namespace",
            "size",
            "max_size",
            "hits",
            "misses",
            "evictions",
            "expirations",
            "ttl_seconds",
            "hit_rate",
        }


# ---------------------------------------------------------------------------
# Namespace isolation in keys
# ---------------------------------------------------------------------------


class TestNamespaceIsolation:
    def test_same_prefix_different_namespace_different_keys(self) -> None:
        a = PrefixCache("A")
        b = PrefixCache("B")
        ka = a.key("p", "x")
        kb = b.key("p", "x")
        assert ka != kb


# ---------------------------------------------------------------------------
# Registry + all_stats
# ---------------------------------------------------------------------------


class TestRegistry:
    def setup_method(self) -> None:
        reset_all()
        pc._caches.clear()

    def test_get_cache_returns_same_instance(self) -> None:
        c1 = get_cache("my_ns")
        c2 = get_cache("my_ns")
        assert c1 is c2

    def test_get_cache_first_caller_wins_config(self) -> None:
        c1 = get_cache("my_ns", max_size=5)
        c2 = get_cache("my_ns", max_size=500)  # ignored — cache already exists
        assert c1 is c2
        assert c2.max_size == 5

    def test_all_stats_ordered_by_namespace(self) -> None:
        get_cache("zebra")
        get_cache("alpha")
        get_cache("mike")
        ns = [s.namespace for s in all_stats()]
        assert ns == ["alpha", "mike", "zebra"]

    def test_reset_all_clears_every_cache(self) -> None:
        c = get_cache("my_ns")
        c.put("p", "x", "v")
        assert c.stats().size == 1
        reset_all()
        assert c.stats().size == 0

    def test_registry_namespace_cap(self) -> None:
        """Dynamic namespaces must not grow the registry without bound."""
        from mind_mem.prefix_cache import set_max_namespaces

        pc._caches.clear()
        set_max_namespaces(3)
        get_cache("tenant-1")
        get_cache("tenant-2")
        get_cache("tenant-3")
        get_cache("tenant-4")  # evicts tenant-1 (LRU)
        assert "tenant-1" not in pc._caches
        assert "tenant-4" in pc._caches
        # Restore default so other tests don't see a tight cap.
        set_max_namespaces(64)

    def test_get_cache_refreshes_lru_on_repeat_access(self) -> None:
        from mind_mem.prefix_cache import set_max_namespaces

        pc._caches.clear()
        set_max_namespaces(3)
        get_cache("a")
        get_cache("b")
        get_cache("c")
        # Touch "a" so "b" becomes LRU.
        get_cache("a")
        get_cache("d")  # evicts "b"
        assert "a" in pc._caches
        assert "b" not in pc._caches
        assert "c" in pc._caches
        assert "d" in pc._caches
        set_max_namespaces(64)

    def test_set_max_namespaces_rejects_zero(self) -> None:
        from mind_mem.prefix_cache import set_max_namespaces

        with pytest.raises(ValueError):
            set_max_namespaces(0)


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


class TestConcurrency:
    def test_concurrent_put_get_no_corruption(self) -> None:
        # max_size sized above the total working set so the test measures
        # data-race safety, not LRU churn. The LRU eviction path is
        # covered separately in TestLRUEviction.
        threads_n = 16
        per_thread = 200
        cache = PrefixCache("ns", max_size=threads_n * per_thread + 16)
        errors: list[BaseException] = []
        barrier = threading.Barrier(threads_n)

        def worker(tid: int) -> None:
            try:
                barrier.wait()
                for i in range(per_thread):
                    cache.put("p", (tid, i), f"v{tid}-{i}")
                    hit, value = cache.get("p", (tid, i))
                    assert hit is True
                    assert value == f"v{tid}-{i}"
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(threads_n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors, f"Worker errors: {errors[:3]}"
