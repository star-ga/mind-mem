"""v3.2.0 — distributed recall result cache (Redis + in-process LRU fallback).

Caches the JSON envelope returned by ``recall(query, …)`` keyed on
``(query_hash, namespace, limit, backend, active_only)``. The cache
is **transparent** — callers opt in by wiring through
:func:`cached_recall` / :func:`invalidate` rather than the raw
recall functions.

Two backends:

* **Redis** — when ``cache.redis_url`` is set in ``mind-mem.json``.
  Used for multi-process / multi-host deployments so all mind-mem
  processes pointing at the same cache see each other's writes.
  Fails open (treats a Redis outage as a cache miss) so recall
  stays available even when Redis is unavailable.
* **In-process LRU** — default fallback. Pure-Python, stdlib
  ``collections.OrderedDict`` with LRU eviction. Used when Redis
  isn't configured or the ``redis`` package isn't installed.

Invalidation — **by construction, through the key**. The recall path folds
the governed-ledger head (``index_anchor``) into :func:`make_cache_key`, so a
governed write moves the head, the head moves the key, and every entry
computed against the previous corpus state becomes unresolvable. That holds
for a write through *any* door — the governance MCP tools, the CLI, the HTTP
transport, the apply engine, federation replication — because it does not
depend on that door remembering to call anything.

The explicit namespace-wide :func:`invalidate` remains, and the governance
MCP tools still call it. It is now a *promptness* measure (it frees the
memory immediately rather than at the next eviction), not the correctness
mechanism it used to be. It was never sufficient as one: it is called from
three MCP tools, and the other write doors never called it at all.

Targeted per-query invalidation is still not offered — it would require
tracking which queries touched which blocks, and the anchor makes it
unnecessary.

Metrics exported on the Prometheus exporter (when installed):

* ``recall_cache_hits_total{backend}``
* ``recall_cache_misses_total{backend}``
* ``recall_cache_evictions_total{backend}``
* ``recall_cache_size`` (in-process backend only)
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import threading
import time
from collections import OrderedDict
from typing import Any

from .observability import get_logger, metrics

_log = get_logger("recall_cache")

# Default TTL: one hour. Callers can override per query. This is a *memory*
# bound, not a freshness mechanism — freshness comes from ``index_anchor`` in
# the key (see ``make_cache_key``). An entry whose corpus state has moved is
# already unresolvable long before its TTL expires; the TTL only decides how
# long an entry nobody will ask for again keeps occupying a slot.
_DEFAULT_TTL_SECONDS = 3600

# In-process LRU cap. Each entry averages ~8 KiB (envelope JSON) so
# 1024 entries ≈ 8 MiB of process memory, which is negligible.
_LRU_MAX_ENTRIES = 1024


# ---------------------------------------------------------------------------
# Cache-key derivation
# ---------------------------------------------------------------------------


def make_cache_key(
    query: str,
    *,
    namespace: str = "default",
    limit: int = 10,
    backend: str = "auto",
    active_only: bool = False,
    scoring_instant: str = "",
    index_anchor: str = "",
) -> str:
    """Derive a stable cache key for a recall invocation.

    SHA-256 of the JSON-serialized parameter tuple. JSON rather than
    repr() so the key is portable across Python versions and doesn't
    leak object-id randomness. Namespace-prefixed so multi-tenant
    deployments don't collide.

    ``scoring_instant`` is part of the key because it is part of the answer:
    recall is deterministic given (corpus, config, scoring_instant), so two
    requests differing only in the instant are two different queries. Omitting
    it would serve one instant's ranking under another instant's attestation.

    ``index_anchor`` — the head of the governed hash chain — is in the key for
    the same reason and a sharper one. Recall is deterministic given *(corpus, config,
    scoring_instant)*, and the anchor is how a run names the corpus it read;
    it is the third coordinate, and the TTL below is not a substitute for it.
    Without the anchor in the key, a governed write that lands through any door
    other than the governance MCP tools — the CLI, the HTTP transport, the
    apply engine, federation replication — leaves this cache serving the
    pre-write answer while :mod:`mind_mem.recall_attestation` stamps the
    post-write anchor onto it. The run then attests a corpus state it did not
    read, and replaying at that anchor does not reproduce the answer. With the
    anchor in the key a superseded entry is simply unresolvable: the write
    moved the head, the head is in the key, the key no longer matches. No
    invalidation call to remember, so no door that can forget to call it.

    The default of ``""`` keeps the signature backward-compatible for callers
    outside the recall path; the recall path always supplies it.
    """
    payload = {
        "query": query,
        "namespace": namespace,
        "limit": int(limit),
        "backend": backend,
        "active_only": bool(active_only),
        "scoring_instant": str(scoring_instant),
        "index_anchor": str(index_anchor),
    }
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    return f"mindmem:recall:{namespace}:{digest[:24]}"


# ---------------------------------------------------------------------------
# In-process LRU
# ---------------------------------------------------------------------------


class LRUCache:
    """Thread-safe LRU with per-entry expiry.

    Used as the in-process fallback + as the L1 in front of Redis.
    """

    def __init__(self, max_entries: int = _LRU_MAX_ENTRIES) -> None:
        self._max = max(1, max_entries)
        self._data: "OrderedDict[str, tuple[float, str]]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> str | None:
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            expires_at, value = entry
            if expires_at > 0 and time.time() > expires_at:
                self._data.pop(key, None)
                return None
            self._data.move_to_end(key)
            return value

    def set(self, key: str, value: str, ttl_seconds: int = _DEFAULT_TTL_SECONDS) -> None:
        expires_at = time.time() + ttl_seconds if ttl_seconds > 0 else 0.0
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
            self._data[key] = (expires_at, value)
            while len(self._data) > self._max:
                self._data.popitem(last=False)
                metrics.inc("recall_cache_evictions_total")

    def delete(self, key: str) -> bool:
        with self._lock:
            return self._data.pop(key, None) is not None

    def invalidate_namespace(self, namespace: str) -> int:
        """Remove every entry whose key contains the namespace."""
        prefix = f"mindmem:recall:{namespace}:"
        with self._lock:
            to_delete = [k for k in self._data if k.startswith(prefix)]
            for k in to_delete:
                self._data.pop(k, None)
            return len(to_delete)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)


# ---------------------------------------------------------------------------
# Redis adapter
# ---------------------------------------------------------------------------


class _RedisCache:
    """Thin wrapper around ``redis.Redis`` with fail-open semantics."""

    def __init__(self, url: str):
        import redis  # type: ignore

        self._client = redis.from_url(url, decode_responses=True, socket_timeout=1.0)

    def get(self, key: str) -> str | None:
        try:
            val = self._client.get(key)
            return val if isinstance(val, str) else None
        except Exception as exc:
            _log.debug("redis_get_failed", error=str(exc))
            return None

    def set(self, key: str, value: str, ttl_seconds: int = _DEFAULT_TTL_SECONDS) -> None:
        try:
            if ttl_seconds > 0:
                self._client.setex(key, ttl_seconds, value)
            else:
                self._client.set(key, value)
        except Exception as exc:
            _log.debug("redis_set_failed", error=str(exc))

    def invalidate_namespace(self, namespace: str) -> int:
        """SCAN + DEL every key under the namespace prefix."""
        try:
            prefix = f"mindmem:recall:{namespace}:*"
            count = 0
            for batch in _scan_iter_chunks(self._client, prefix, 500):
                if batch:
                    count += self._client.delete(*batch)
            return count
        except Exception as exc:
            _log.debug("redis_invalidate_failed", error=str(exc))
            return 0


def _scan_iter_chunks(client, match_pattern: str, chunk_size: int):
    """Yield lists of keys in chunks so DEL calls stay bounded."""
    cursor = 0
    buf: list[str] = []
    while True:
        cursor, keys = client.scan(cursor=cursor, match=match_pattern, count=chunk_size)
        buf.extend(keys)
        if len(buf) >= chunk_size or cursor == 0:
            yield buf
            buf = []
        if cursor == 0:
            break


# ---------------------------------------------------------------------------
# Cache façade
# ---------------------------------------------------------------------------


class RecallCache:
    """Two-tier cache: in-process LRU L1 + optional Redis L2.

    Writes go to both tiers; reads prefer L1, fall back to L2, and
    repopulate L1 on L2 hit.
    """

    def __init__(self, redis_url: str | None = None, lru_max: int = _LRU_MAX_ENTRIES) -> None:
        self._lru = LRUCache(max_entries=lru_max)
        self._redis: _RedisCache | None = None
        self._backend_label = "lru"
        if redis_url and importlib.util.find_spec("redis") is not None:
            try:
                self._redis = _RedisCache(redis_url)
                self._backend_label = "redis"
            except Exception as exc:
                _log.warning("redis_init_failed", url=redis_url, error=str(exc))
                self._redis = None

    @property
    def backend_label(self) -> str:
        return self._backend_label

    def get(self, key: str) -> str | None:
        hit = self._lru.get(key)
        if hit is not None:
            metrics.inc("recall_cache_hits_total")
            return hit
        if self._redis is not None:
            hit = self._redis.get(key)
            if hit is not None:
                self._lru.set(key, hit)
                metrics.inc("recall_cache_hits_total")
                return hit
        metrics.inc("recall_cache_misses_total")
        return None

    def set(self, key: str, value: str, ttl_seconds: int = _DEFAULT_TTL_SECONDS) -> None:
        self._lru.set(key, value, ttl_seconds=ttl_seconds)
        if self._redis is not None:
            self._redis.set(key, value, ttl_seconds=ttl_seconds)

    def invalidate_namespace(self, namespace: str = "default") -> int:
        """Drop every cached entry for *namespace* across both tiers."""
        dropped = self._lru.invalidate_namespace(namespace)
        if self._redis is not None:
            dropped += self._redis.invalidate_namespace(namespace)
        _log.info("recall_cache_invalidated", namespace=namespace, dropped=dropped)
        return dropped

    def clear(self) -> None:
        self._lru.clear()


# ---------------------------------------------------------------------------
# Module-level singleton helpers
# ---------------------------------------------------------------------------

_singleton_lock = threading.Lock()
_singleton: RecallCache | None = None


def get_cache(config: dict[str, Any] | None = None) -> RecallCache:
    """Return the process-wide RecallCache singleton."""
    global _singleton
    with _singleton_lock:
        if _singleton is None:
            redis_url: str | None = None
            if config is not None:
                cache_cfg = config.get("cache", {}) if isinstance(config, dict) else {}
                if isinstance(cache_cfg, dict):
                    redis_url = cache_cfg.get("redis_url")
            _singleton = RecallCache(redis_url=redis_url)
        return _singleton


def cached_recall(
    inner: Any,
    query: str,
    *,
    namespace: str = "default",
    limit: int = 10,
    backend: str = "auto",
    active_only: bool = False,
    ttl_seconds: int = _DEFAULT_TTL_SECONDS,
    config: dict[str, Any] | None = None,
    scoring_instant: str = "",
    index_anchor: str = "",
) -> str:
    """Cache-wrapped call to a recall function.

    *inner* is any callable with the signature
    ``(query, limit=..., active_only=..., backend=...) -> str`` — the
    same shape ``_recall_impl`` and its consumers use. Returns the
    same string envelope.

    ``scoring_instant`` (an ISO-8601 UTC date, or ``""`` for callers that do not
    use the seam) and ``index_anchor`` (the governed-ledger head, or ``""``)
    participate in the key only — *inner* is called with its original
    signature, so a caller that closes over either keeps working unchanged.
    See :func:`make_cache_key` for why the anchor, not the TTL, is what makes a
    superseded entry unservable.
    """
    key = make_cache_key(
        query,
        namespace=namespace,
        limit=limit,
        backend=backend,
        active_only=active_only,
        scoring_instant=scoring_instant,
        index_anchor=index_anchor,
    )
    cache = get_cache(config)
    hit = cache.get(key)
    if hit is not None:
        return hit
    envelope: str = str(inner(query, limit=limit, active_only=active_only, backend=backend))
    cache.set(key, envelope, ttl_seconds=ttl_seconds)
    return envelope


def invalidate(namespace: str = "default", config: dict[str, Any] | None = None) -> int:
    """Drop cached recall results for *namespace*. Returns entries dropped."""
    return get_cache(config).invalidate_namespace(namespace)


def reset_singleton() -> None:
    """Clear the module-level singleton. Test-only hook."""
    global _singleton
    with _singleton_lock:
        _singleton = None
