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
from .recall_attestation import INDEX_ANCHOR_UNRESOLVED

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


#: Cache-key contract version. Entries produced before workspace, retrieval
#: projection, schema, and filter coordinates were all bound must never be
#: adopted into the stronger contract.
_KEY_FORMAT = "v3"


def canonical_workspace_id(workspace: str | None) -> str:
    """Return the local filesystem identity used to isolate cache entries.

    ``realpath`` makes aliases of one local root share an entry. This is a
    local path identity, not a portable authority or authenticated principal.
    An absent workspace gets a reserved sentinel instead of colliding with the
    empty governed anchor.
    """
    import os

    if not workspace or not str(workspace).strip():
        return "\x00unscoped"
    try:
        return os.path.realpath(str(workspace))
    except (OSError, ValueError):
        return str(workspace)


#: A file whose mtime is this close to "now" cannot vouch for its own content:
#: filesystem timestamps tick coarsely (a jiffy on ext4, a second or more on
#: some network filesystems), so a same-size in-place rewrite inside one tick
#: leaves ``(size, mtime)`` unchanged. Inside the window the stamp refuses to
#: name the state and the request bypasses the cache — the racy-timestamp rule
#: a content-addressed index uses for exactly the same reason.
_RACY_WINDOW_NS = 1_000_000_000

#: Index-directory files that are written by reads or are not retrieval input.
#: ``-shm`` is SQLite's shared-memory index (touched by readers); ``usage.json``
#: is the token-metering ledger.
_STAMP_EXCLUDED_INDEX_SUFFIXES = ("-shm",)
_STAMP_EXCLUDED_INDEX_NAMES = frozenset({"usage.json"})

#: Retrieval inputs outside ``.mind-mem-index`` and the corpus directories.
_STAMP_EXTRA_FILES = (
    "memory/knowledge_graph.db",
    "memory/knowledge_graph.db-wal",
    # v4 PQ side store (``mind_mem.v4.pq``), written by a vector rebuild.
    "index.db",
    "index.db-wal",
)

#: Default local vector-index directory (``recall.index_path``). Hybrid and
#: vector recall read ``<dir>/index.json`` on every call, and a vector-only
#: rebuild (``recall_vector.rebuild_index``) writes nothing else, so the whole
#: directory listing is part of the stamp.
_DEFAULT_VECTOR_INDEX_DIR = ".mind-mem-vectors"


def _vector_stamp_inputs(workspace: str, config: dict | None) -> tuple[list[str], list[str]] | None:
    """Absolute directories and files the vector leg reads, per the ``recall`` config.

    Every path is resolved by the READER's own resolver, never re-joined here:
    ``recall_vector.local_index_file`` (what ``VectorBackend._get_index_path``
    and ``rebuild_index`` use) for the index directory, and
    ``VectorBackend._sqlite_vec_db_path`` for the sqlite-vec database. A
    relative, absolute, ``~``-prefixed or symlinked ``recall.index_path`` /
    ``recall.sqlite_vec_db`` therefore names the file recall actually opens.
    ``index_path`` is coerced with ``str()`` exactly as ``VectorBackend``
    coerces it. The default directory is always included.

    Pure: no I/O, no logging (``VectorBackend`` is never constructed — its
    ``__init__`` logs). Returns ``None`` (the caller bypasses the cache) when a
    configured value is one the reader's resolver itself cannot resolve.
    """
    import os
    from types import SimpleNamespace

    from .recall_vector import VectorBackend, local_index_file

    recall_cfg = config.get("recall") if isinstance(config, dict) else None
    if not isinstance(recall_cfg, dict):
        recall_cfg = {}
    try:
        index_paths = [_DEFAULT_VECTOR_INDEX_DIR]
        if "index_path" in recall_cfg:
            index_paths.append(str(recall_cfg["index_path"]))
        dirs = [os.path.dirname(local_index_file(workspace, ip)) for ip in index_paths]
        vec_db = VectorBackend._sqlite_vec_db_path(SimpleNamespace(config=recall_cfg), workspace)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return list(dict.fromkeys(dirs)), [vec_db, f"{vec_db}-wal"]


def corpus_stamp(workspace: str, config: dict | None = None) -> str | None:
    """Stat-only identity of everything recall reads, or ``None`` if unnameable.

    The governed-ledger head in the key moves only on governed writes. Several
    writes that change what recall may serve never touch that ledger: an admin
    ``reindex`` after a status change on disk, ``delete_memory_item``,
    ``compact``, ``lint_autofix``, lineage / edge writes (``recall.db``), KG
    writes, and any out-of-band edit. This stamp covers them by construction —
    every file in ``.mind-mem-index`` (bar the exclusions above), the
    knowledge-graph database and every corpus file
    (:func:`~mind_mem.corpus_registry.discover_corpus_files`), each as
    ``(name, size, mtime_ns, ctime_ns, inode)`` or ``absent``. The vector leg's
    inputs are covered too: every file in the local vector-index directory
    (``.mind-mem-vectors`` and, when *config* names one, ``recall.index_path``),
    a configured ``recall.sqlite_vec_db`` and the v4 PQ side store — a
    vector-only rebuild or an out-of-band vector-index edit writes nothing
    else. No file is read: the cost is one ``listdir`` per directory plus one
    ``stat`` per file. *config* is the already-loaded ``mind-mem.json`` dict;
    the stamp never opens it.

    Returns ``None`` — the caller must bypass the cache — when a stat fails for
    a reason other than absence, or when any file was modified inside
    :data:`_RACY_WINDOW_NS`.
    """
    import os

    from .corpus_registry import discover_corpus_files

    root = str(workspace)
    now_ns = time.time_ns()
    parts: list[str] = []

    def _add(label: str, path: str) -> bool:
        try:
            st = os.stat(path)
        except FileNotFoundError:
            parts.append(f"{label}\x00absent")
            return True
        except OSError:
            return False
        if st.st_mtime_ns > now_ns - _RACY_WINDOW_NS:
            return False
        parts.append(f"{label}\x00{st.st_size}\x00{st.st_mtime_ns}\x00{st.st_ctime_ns}\x00{st.st_ino}")
        return True

    def _listing(label_dir: str, path_dir: str) -> list[tuple[str, str]] | None:
        try:
            names = sorted(os.listdir(path_dir))
        except (FileNotFoundError, NotADirectoryError):
            parts.append(f"{label_dir}/\x00absent")
            return []
        except OSError:
            return None
        return [
            (f"{label_dir}/{name}", os.path.join(path_dir, name))
            for name in names
            if name not in _STAMP_EXCLUDED_INDEX_NAMES and not name.endswith(_STAMP_EXCLUDED_INDEX_SUFFIXES)
        ]

    def _ws(rel: str) -> tuple[str, str]:
        return rel, os.path.join(root, *rel.split("/"))

    vector_inputs = _vector_stamp_inputs(root, config)
    if vector_inputs is None:
        return None
    vector_dirs, vector_files = vector_inputs
    # Workspace-relative inputs are labelled by their relative name; vector
    # inputs by the absolute path their reader resolved (it may lie outside
    # the workspace, e.g. an absolute ``recall.index_path``).
    dir_inputs = [_ws(".mind-mem-index"), *((d, d) for d in vector_dirs)]
    inputs: list[tuple[str, str]] = []
    for label_dir, path_dir in dir_inputs:
        listed = _listing(label_dir, path_dir)
        if listed is None:
            return None
        inputs.extend(listed)
    inputs.extend(_ws(rel) for rel in _STAMP_EXTRA_FILES)
    inputs.extend((f, f) for f in vector_files)
    try:
        inputs.extend(_ws(rel) for _label, rel in discover_corpus_files(root))
    except OSError:
        return None
    for label, path in inputs:
        if not _add(label, path):
            return None
    return hashlib.sha256("\x01".join(parts).encode("utf-8")).hexdigest()[:32]


#: Version of the retrieval-config projection itself. A projection change must
#: change the digest even for the overwhelmingly common empty/default config.
PROJECTION_VERSION = "p2"

#: Returned only for programmatic, non-JSON configurations. The public config
#: loader parses JSON, so ordinary files cannot produce this value; callers
#: that construct an invalid mapping must bypass the cache rather than share a
#: sentinel key with another invalid mapping.
UNCACHEABLE_CONFIG_FINGERPRINT = "\x00unserialisable"


#: Storage-only settings do not change an answer. Every other top-level field
#: is retained conservatively: a newly introduced provider or policy section
#: must cause an extra miss until its semantics are understood, never reuse an
#: answer produced under another configuration.
_CACHE_ONLY_SECTIONS = frozenset({"cache"})


def effective_retrieval_projection(config: dict | None) -> dict:
    """Project every raw non-cache config field into cache identity.

    This intentionally does not resolve defaults. Explicit and implicit forms
    may therefore miss each other's cache entries, which is safe; treating two
    differently resolved configs as identical could serve the wrong answer.
    Environment and model-artifact state remain outside this raw-file contract.
    """
    if not isinstance(config, dict):
        return {}
    return {key: value for key, value in config.items() if key not in _CACHE_ONLY_SECTIONS}


def retrieval_config_fingerprint(config: dict | None) -> str:
    """Digest the versioned JSON retrieval projection, including its empty state."""
    relevant = effective_retrieval_projection(config)
    try:
        body = json.dumps(
            {"v": PROJECTION_VERSION, "sections": relevant},
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError):
        return UNCACHEABLE_CONFIG_FINGERPRINT
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def make_cache_key(
    query: str,
    *,
    namespace: str = "default",
    limit: int = 10,
    backend: str = "auto",
    active_only: bool = False,
    scoring_instant: str = "",
    index_anchor: str = "",
    workspace: str = "",
    config_fingerprint: str = "",
    schema_version: str = "",
    filters: dict | None = None,
    agent_id: str | None = None,
    corpus_stamp: str = "",
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

    ``corpus_stamp`` (see :func:`corpus_stamp`) is the second corpus
    coordinate. The anchor only moves on governed writes; reindex, delete,
    compact, lint autofix, lineage / edge writes and out-of-band edits change
    what recall serves without moving it. The stamp is folded in only when
    non-empty, so callers that do not supply it keep their existing keys.
    """
    payload = {
        "workspace": canonical_workspace_id(workspace),
        "config": config_fingerprint,
        "schema": schema_version,
        "query": query,
        "namespace": namespace,
        "limit": int(limit),
        "backend": backend,
        "active_only": bool(active_only),
        "scoring_instant": str(scoring_instant),
        "index_anchor": str(index_anchor),
    }
    # Post-retrieval filters (since / until / lifecycle / event_id /
    # min_maturity) change the ANSWER, so they belong in the key. Omitting them
    # would let a filtered query be served from the UNFILTERED entry: the filter
    # would pass a unit test and silently do nothing in a warm process.
    #
    # Only non-empty values are folded in and the sub-key is omitted entirely
    # when nothing is set, so a caller that passes no filter gets a
    # byte-identical key to before and no warm cache is invalidated by this
    # field merely existing.
    active_filters = {k: v for k, v in (filters or {}).items() if v is not None and v != ""}
    if active_filters:
        payload["filters"] = {k: str(v) for k, v in sorted(active_filters.items())}
    if agent_id:
        payload["agent_id"] = str(agent_id)
    if corpus_stamp:
        payload["corpus_stamp"] = str(corpus_stamp)
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    return f"mindmem:recall:{_KEY_FORMAT}:{namespace}:{digest[:24]}"


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
        prefix = f"mindmem:recall:{_KEY_FORMAT}:{namespace}:"
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
            prefix = f"mindmem:recall:{_KEY_FORMAT}:{namespace}:*"
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
    workspace: str = "",
    config_fingerprint: str = "",
    schema_version: str = "",
    filters: dict | None = None,
    agent_id: str | None = None,
    corpus_stamp: str = "",
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
    if config_fingerprint == UNCACHEABLE_CONFIG_FINGERPRINT:
        # A shared sentinel in the key would make every distinct malformed
        # config collide. The file-backed public config is JSON and never takes
        # this branch; it protects programmatic callers without guessing an
        # identity for data the canonical serializer refused.
        return str(inner(query, limit=limit, active_only=active_only, backend=backend, **(filters or {})))

    if index_anchor == INDEX_ANCHOR_UNRESOLVED:
        # A failed governed-head read is not a corpus generation. Never let a
        # caller accidentally turn its explicit failure state into a reusable
        # cache entry by invoking this primitive directly.
        return str(inner(query, limit=limit, active_only=active_only, backend=backend, **(filters or {})))

    key = make_cache_key(
        query,
        namespace=namespace,
        limit=limit,
        backend=backend,
        active_only=active_only,
        scoring_instant=scoring_instant,
        index_anchor=index_anchor,
        workspace=workspace,
        config_fingerprint=config_fingerprint,
        schema_version=schema_version,
        filters=filters,
        agent_id=agent_id,
        corpus_stamp=corpus_stamp,
    )
    cache = get_cache(config)
    hit = cache.get(key)
    if hit is not None:
        return hit
    # Filters are forwarded to `inner` as well as keyed: keying alone would give
    # every filtered query its own entry and then fill it with the UNFILTERED
    # answer.
    envelope: str = str(inner(query, limit=limit, active_only=active_only, backend=backend, **(filters or {})))
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
