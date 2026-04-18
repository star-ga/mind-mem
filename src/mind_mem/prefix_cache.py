# Copyright 2026 STARGA, Inc.
"""Prefix cache for LLM-backed mind-mem hot paths.

Cross-encoder reranking, intent routing, and LLM query expansion all call
an LLM with prompts that share a large fixed prefix (system instructions,
governance context, candidate boilerplate). Under load this prefix is
tokenised and shipped over the wire repeatedly even though the model
would produce an identical response for the same input.

:class:`PrefixCache` is an in-memory LRU cache keyed on
``(namespace, prefix_hash, payload_hash)``. It records hit / miss
counters so :func:`mind_mem.mcp_server.index_stats` can surface cache
efficiency, and it supports an optional TTL so a misbehaving model
response does not live forever.

Pure-Python, stdlib only. No network, no disk — this is a process-local
cache. Persisting across processes is out of scope for v2.0.0b1.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Optional

# Upper bound on how many PrefixCache instances the module-level registry
# will keep alive. Every well-known subsystem (cross-encoder, intent
# router, query expansion) gets exactly one cache, so a few dozen is
# ample. A cap avoids a subtle DoS where a caller that passes dynamic
# namespace strings (e.g. ``get_cache(f"tenant-{id}")``) grows the
# registry without bound.
_DEFAULT_MAX_NAMESPACES: int = 64


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hash_text(text: str) -> str:
    """SHA-256 hex digest of UTF-8 text. Stable, collision-resistant."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _hash_obj(value: Any) -> str:
    """Deterministic SHA-256 of an arbitrary JSON-compatible value.

    ``str`` / ``bytes`` are hashed directly so they do not pay the JSON
    round-trip cost. Everything else goes through a canonical JSON
    encoding so dict key order does not perturb the key.
    """
    import json

    if value is None:
        return _hash_text("")
    if isinstance(value, str):
        return _hash_text(value)
    if isinstance(value, bytes):
        return hashlib.sha256(value).hexdigest()
    return _hash_text(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str))


# ---------------------------------------------------------------------------
# Cache entry
# ---------------------------------------------------------------------------


@dataclass
class _Entry:
    """Single cache entry. Mutable on access to update ``last_hit_at``."""

    value: Any
    stored_at: float
    last_hit_at: float
    hit_count: int = 0

    def is_expired(self, now: float, ttl: Optional[float]) -> bool:
        if ttl is None or ttl <= 0:
            return False
        return (now - self.stored_at) >= ttl


# ---------------------------------------------------------------------------
# Cache statistics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CacheStats:
    """Immutable snapshot of cache statistics for MCP consumption."""

    namespace: str
    size: int
    max_size: int
    hits: int
    misses: int
    evictions: int
    expirations: int
    ttl_seconds: Optional[float]

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return self.hits / total

    def as_dict(self) -> dict[str, Any]:
        return {
            "namespace": self.namespace,
            "size": self.size,
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "expirations": self.expirations,
            "ttl_seconds": self.ttl_seconds,
            "hit_rate": round(self.hit_rate, 4),
        }


# ---------------------------------------------------------------------------
# PrefixCache
# ---------------------------------------------------------------------------


class PrefixCache:
    """LRU cache for LLM prefix responses with optional TTL.

    Args:
        namespace: Label included in the key and in :class:`CacheStats`.
            Each subsystem (cross-encoder, intent router, LLM expansion)
            uses its own namespace so their hit / miss counters stay
            separate for observability.
        max_size: Maximum number of live entries. Once exceeded the
            least-recently-used entry is evicted. Must be >= 1.
        ttl_seconds: Optional time-to-live. ``None`` disables expiration.
            A value of 0 or less is treated the same as ``None``.
    """

    def __init__(
        self,
        namespace: str,
        *,
        max_size: int = 256,
        ttl_seconds: Optional[float] = None,
    ) -> None:
        if not namespace:
            raise ValueError("PrefixCache namespace must be a non-empty string")
        if max_size < 1:
            raise ValueError("PrefixCache.max_size must be >= 1")

        self._namespace = namespace
        self._max_size = int(max_size)
        self._ttl = ttl_seconds if (ttl_seconds is not None and ttl_seconds > 0) else None

        # OrderedDict acts as the LRU data structure. move_to_end on hit
        # keeps the most-recently-used entry at the right.
        self._entries: "OrderedDict[str, _Entry]" = OrderedDict()
        # Separate lock so concurrent get/put do not corrupt the LRU order.
        self._lock = threading.RLock()

        # Counters — read by get_stats(), written under the lock.
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._expirations = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def namespace(self) -> str:
        return self._namespace

    @property
    def max_size(self) -> int:
        return self._max_size

    def key(self, prefix: str, payload: Any) -> str:
        """Build a cache key from a text prefix plus a payload.

        The returned key includes the namespace so two caches with the
        same prefix but different namespaces never collide, even when a
        caller accidentally shares the underlying store.
        """
        return f"{self._namespace}:{_hash_text(prefix)}:{_hash_obj(payload)}"

    def get(self, prefix: str, payload: Any) -> tuple[bool, Any]:
        """Return ``(hit, value)`` for the given prefix + payload.

        On a miss returns ``(False, None)``. On an expired entry the
        entry is evicted and the call counts as a miss so the caller
        refills it on the same hot path without seeing stale data.
        """
        k = self.key(prefix, payload)
        now = time.monotonic()
        with self._lock:
            entry = self._entries.get(k)
            if entry is None:
                self._misses += 1
                return False, None
            if entry.is_expired(now, self._ttl):
                del self._entries[k]
                self._expirations += 1
                self._misses += 1
                return False, None
            entry.hit_count += 1
            entry.last_hit_at = now
            self._entries.move_to_end(k, last=True)
            self._hits += 1
            return True, entry.value

    def put(self, prefix: str, payload: Any, value: Any) -> None:
        """Insert (or refresh) an entry. Evicts the LRU when full."""
        k = self.key(prefix, payload)
        now = time.monotonic()
        with self._lock:
            if k in self._entries:
                existing = self._entries[k]
                existing.value = value
                existing.stored_at = now
                existing.last_hit_at = now
                self._entries.move_to_end(k, last=True)
                return
            self._entries[k] = _Entry(
                value=value,
                stored_at=now,
                last_hit_at=now,
                hit_count=0,
            )
            if len(self._entries) > self._max_size:
                self._entries.popitem(last=False)
                self._evictions += 1

    def invalidate(self, prefix: str, payload: Any) -> bool:
        """Remove a single entry. Returns True when something was removed."""
        k = self.key(prefix, payload)
        with self._lock:
            entry = self._entries.pop(k, None)
            return entry is not None

    def clear(self) -> None:
        """Remove every entry and reset counters.

        Counters reset too so ``hit_rate`` after clear cannot be
        dominated by stale history from the previous corpus / config.
        """
        with self._lock:
            self._entries.clear()
            self._hits = 0
            self._misses = 0
            self._evictions = 0
            self._expirations = 0

    def stats(self) -> CacheStats:
        """Snapshot current statistics. Cheap; safe to call often."""
        with self._lock:
            return CacheStats(
                namespace=self._namespace,
                size=len(self._entries),
                max_size=self._max_size,
                hits=self._hits,
                misses=self._misses,
                evictions=self._evictions,
                expirations=self._expirations,
                ttl_seconds=self._ttl,
            )


# ---------------------------------------------------------------------------
# Module-level registry
# ---------------------------------------------------------------------------


_registry_lock = threading.RLock()
_caches: "OrderedDict[str, PrefixCache]" = OrderedDict()
_max_namespaces: int = _DEFAULT_MAX_NAMESPACES


def set_max_namespaces(limit: int) -> None:
    """Override the registry cap. Intended for operators and tests."""
    global _max_namespaces
    if limit < 1:
        raise ValueError("_max_namespaces must be >= 1")
    with _registry_lock:
        _max_namespaces = int(limit)
        while len(_caches) > _max_namespaces:
            _caches.popitem(last=False)


def get_cache(
    namespace: str,
    *,
    max_size: int = 256,
    ttl_seconds: Optional[float] = None,
) -> PrefixCache:
    """Return the shared :class:`PrefixCache` for *namespace*.

    First caller wins on configuration — subsequent callers receive the
    already-initialised cache regardless of the ``max_size`` / ``ttl``
    they pass. This matches the usage pattern where the MCP server, the
    cross-encoder, and the intent router each reach for the same cache
    without coordinating.

    Registry size is bounded (see :data:`_DEFAULT_MAX_NAMESPACES`) so
    callers that build per-tenant namespaces cannot grow the registry
    without bound. When the cap is exceeded the least-recently-accessed
    namespace is evicted.
    """
    with _registry_lock:
        cache = _caches.get(namespace)
        if cache is None:
            cache = PrefixCache(
                namespace,
                max_size=max_size,
                ttl_seconds=ttl_seconds,
            )
            _caches[namespace] = cache
            while len(_caches) > _max_namespaces:
                _caches.popitem(last=False)
        else:
            _caches.move_to_end(namespace, last=True)
        return cache


def all_stats() -> list[CacheStats]:
    """Return a snapshot of every registered cache's statistics.

    Ordered by namespace so MCP clients can diff snapshots reliably.
    """
    with _registry_lock:
        namespaces = sorted(_caches.keys())
        return [_caches[ns].stats() for ns in namespaces]


def reset_all() -> None:
    """Clear every registered cache and its counters.

    Intended for tests and for operators responding to a corpus refresh
    where cached LLM responses reference stale block ids or content.
    """
    with _registry_lock:
        for cache in _caches.values():
            cache.clear()


__all__ = [
    "PrefixCache",
    "CacheStats",
    "get_cache",
    "all_stats",
    "reset_all",
    "set_max_namespaces",
]
