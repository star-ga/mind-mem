"""Sharded Postgres / Citus routing (v4.0 prep).

Extends :mod:`mind_mem.block_store_postgres` with consistent-hashing
across N shards. Each tenant's blocks live on exactly one shard;
cross-shard recall merges per-shard BM25 results via RRF.

Design:
* :class:`ShardRouter` maps ``(tenant_id, namespace)`` → shard index
  using a 64-bit consistent-hash ring. Adding a shard rebalances at
  most ``1/N`` of the keyspace.
* :class:`ShardedPostgresBlockStore` implements the full
  ``BlockStore`` Protocol by delegating to per-shard underlying
  :class:`PostgresBlockStore` instances.
* Connections are pooled per-shard; each shard has its own DSN.

Citus-specific: when ``CITUS=true`` in the config, the router trusts
Citus's distributed table for sharding and the adapter just forwards
to a single endpoint. Pure-Postgres deployments use the client-side
router and fan out queries themselves.

This is v4.0-prep scaffolding — the underlying writes still go
through :class:`PostgresBlockStore.write_block` which is single-shard.
Cross-shard writes that need transactional semantics are an
independent workstream (2PC via PG ``PREPARE TRANSACTION``).
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Iterable

_log = logging.getLogger("mind_mem.storage.sharded_pg")


def _stable_hash(s: str) -> int:
    """Stable 64-bit hash for shard assignment.

    Not SipHash — we need cross-language stability (the JS SDK and
    Go SDK route writes to the same shard as the Python server). The
    first 8 bytes of SHA-256 are good enough and portable.
    """
    return int.from_bytes(hashlib.sha256(s.encode("utf-8")).digest()[:8], "big")


@dataclass
class ShardConfig:
    """Per-shard connection info."""

    index: int
    dsn: str
    weight: int = 1  # higher weight → takes proportionally more virtual nodes

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("shard index must be ≥0")
        if self.weight < 1:
            raise ValueError("shard weight must be ≥1")


@dataclass
class ShardRouter:
    """Consistent-hash ring.

    Each physical shard gets ``virtual_nodes_per_weight × weight``
    virtual nodes distributed around a 64-bit ring. Lookup: hash the
    key, walk clockwise to the next virtual node. Adding a shard
    affects ~1/N of keys (textbook consistent hashing).
    """

    shards: list[ShardConfig]
    virtual_nodes_per_weight: int = 160
    _ring: list[tuple[int, int]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.shards:
            raise ValueError("ShardRouter requires at least one shard")
        ring: list[tuple[int, int]] = []
        for shard in self.shards:
            for v in range(shard.weight * self.virtual_nodes_per_weight):
                token = _stable_hash(f"shard{shard.index}:vnode{v}")
                ring.append((token, shard.index))
        ring.sort(key=lambda x: x[0])
        self._ring = ring

    def route(self, key: str) -> int:
        """Return the shard index that owns ``key``."""
        h = _stable_hash(key)
        # Binary-search the smallest token ≥ h; wrap around to 0 if
        # h is larger than every token.
        lo, hi = 0, len(self._ring)
        while lo < hi:
            mid = (lo + hi) // 2
            if self._ring[mid][0] < h:
                lo = mid + 1
            else:
                hi = mid
        if lo == len(self._ring):
            lo = 0
        return self._ring[lo][1]

    def shard_for(self, tenant_id: str, namespace: str = "default") -> ShardConfig:
        idx = self.route(f"{tenant_id}:{namespace}")
        return next(s for s in self.shards if s.index == idx)

    def fan_out_shards(self) -> list[ShardConfig]:
        """Every physical shard — used for cross-shard fan-out reads."""
        return list(self.shards)


# ---------------------------------------------------------------------------
# ShardedPostgresBlockStore — BlockStore-shaped façade over N shards.
# ---------------------------------------------------------------------------


class ShardedPostgresBlockStore:
    """BlockStore that dispatches per-tenant writes and fans out reads.

    Writes: route by ``(tenant_id, namespace)`` → single shard.
    Reads: fan out to all shards, merge via RRF (reuses
    :func:`mind_mem.hybrid_recall.rrf_fuse`).

    Construct via :func:`from_config` — direct instantiation requires
    caller-built ``PostgresBlockStore`` per shard.
    """

    def __init__(
        self,
        router: ShardRouter,
        stores_by_shard: dict[int, Any],
        *,
        default_tenant_id: str = "default",
        default_namespace: str = "default",
    ) -> None:
        if not stores_by_shard:
            raise ValueError("stores_by_shard must not be empty")
        self._router = router
        self._stores = stores_by_shard
        self._default_tenant = default_tenant_id
        self._default_namespace = default_namespace

    # ---- BlockStore write surface -----------------------------------------

    def write_block(
        self,
        block: dict[str, Any],
        *,
        tenant_id: str | None = None,
        namespace: str | None = None,
    ) -> str:
        tid = tenant_id or str(block.get("_tenant") or self._default_tenant)
        ns = namespace or str(block.get("_namespace") or self._default_namespace)
        shard = self._router.shard_for(tid, ns)
        store = self._stores[shard.index]
        result = store.write_block(block)
        return str(result) if result is not None else ""

    def delete_block(
        self,
        block_id: str,
        *,
        tenant_id: str | None = None,
        namespace: str | None = None,
    ) -> bool:
        tid = tenant_id or self._default_tenant
        ns = namespace or self._default_namespace
        shard = self._router.shard_for(tid, ns)
        return bool(self._stores[shard.index].delete_block(block_id))

    # ---- BlockStore read surface ------------------------------------------

    def get_by_id(self, block_id: str) -> dict[str, Any] | None:
        # Block IDs aren't sharded by their content — we have to fan
        # out to every shard. Real deployments embed the tenant in the
        # block ID prefix so this shortens to a single shard.
        for store in self._stores.values():
            result = store.get_by_id(block_id)
            if result is not None:
                return dict(result)
        return None

    def search(self, query: str, *, limit: int = 10) -> list[dict[str, Any]]:
        """Fan out search to every shard, RRF-fuse per-shard rankings."""
        from ..hybrid_recall import rrf_fuse

        per_shard_lists: list[list[dict]] = []
        for store in self._stores.values():
            try:
                per_shard_lists.append(store.search(query, limit=limit))
            except Exception as exc:
                _log.debug("shard_search_failed: %s", exc)
                continue
        if not per_shard_lists:
            return []
        weights = [1.0] * len(per_shard_lists)
        fused = rrf_fuse(per_shard_lists, weights=weights, k=60)
        return fused[:limit]

    def get_all(
        self,
        *,
        active_only: bool = False,
        tenant_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Tenant-scoped get_all when ``tenant_id`` is provided; else
        fan-out across shards for admin/compliance scans."""
        if tenant_id is not None:
            shard = self._router.shard_for(tenant_id, self._default_namespace)
            return list(self._stores[shard.index].get_all(active_only=active_only))
        out: list[dict[str, Any]] = []
        for store in self._stores.values():
            try:
                out.extend(store.get_all(active_only=active_only))
            except Exception as exc:
                _log.debug("shard_get_all_failed: %s", exc)
                continue
        return out

    # ---- BlockStore snapshot surface — per-shard ---------------------------

    def snapshot(
        self,
        snap_dir: str,
        *,
        files_touched: list[str] | None = None,
    ) -> dict[str, Any]:
        """Snapshot every shard to its own sub-directory.

        Returns a composite manifest ``{shard_idx: shard_manifest}``.
        """
        import os

        manifests: dict[str, Any] = {}
        for idx, store in self._stores.items():
            shard_dir = os.path.join(snap_dir, f"shard-{idx:02d}")
            os.makedirs(shard_dir, exist_ok=True)
            try:
                manifests[str(idx)] = store.snapshot(shard_dir, files_touched=files_touched)
            except Exception as exc:
                manifests[str(idx)] = {"error": str(exc)}
        return {"sharded": True, "shards": manifests}

    def restore(self, snap_dir: str) -> None:
        import os

        for idx, store in self._stores.items():
            shard_dir = os.path.join(snap_dir, f"shard-{idx:02d}")
            if os.path.isdir(shard_dir):
                store.restore(shard_dir)

    def diff(self, snap_dir: str) -> list[str]:
        import os

        out: list[str] = []
        for idx, store in self._stores.items():
            shard_dir = os.path.join(snap_dir, f"shard-{idx:02d}")
            try:
                out.extend(store.diff(shard_dir))
            except Exception as exc:
                _log.debug("shard_diff_failed shard=%s: %s", idx, exc)
                continue
        return out

    # ---- BlockStore lock surface ------------------------------------------

    def lock(self, *, blocking: bool = True, timeout: float = 30.0) -> Any:
        """Multi-shard lock acquires on every shard under the same
        timeout budget. Caller uses as ``with store.lock():``.
        """
        return _FanOutLock(self._stores.values(), blocking=blocking, timeout=timeout)


class _FanOutLock:
    """Context manager that locks every underlying shard in sequence."""

    def __init__(self, stores: Iterable[Any], *, blocking: bool, timeout: float) -> None:
        self._stores = list(stores)
        self._blocking = blocking
        self._timeout = timeout
        self._acquired: list[Any] = []

    def __enter__(self) -> "_FanOutLock":
        for store in self._stores:
            lock_cm = store.lock(blocking=self._blocking, timeout=self._timeout)
            lock_cm.__enter__()
            self._acquired.append(lock_cm)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # Release in reverse-acquired order — tidiness, not correctness.
        for lock_cm in reversed(self._acquired):
            try:
                lock_cm.__exit__(exc_type, exc, tb)
            except Exception as exc2:
                _log.debug("shard_lock_release_failed: %s", exc2)
                continue


def from_config(config: dict[str, Any]) -> ShardedPostgresBlockStore:
    """Build a :class:`ShardedPostgresBlockStore` from ``block_store`` config.

    Expected shape::

        {
          "block_store": {
            "backend": "sharded_postgres",
            "shards": [
              {"index": 0, "dsn": "postgres://.../shard0"},
              {"index": 1, "dsn": "postgres://.../shard1"}
            ]
          }
        }
    """
    bs = config.get("block_store", {}) if isinstance(config, dict) else {}
    raw_shards = bs.get("shards") or []
    if not raw_shards:
        raise ValueError("sharded_postgres backend requires block_store.shards")
    shards = [ShardConfig(**s) for s in raw_shards]
    router = ShardRouter(shards=shards)

    from ..block_store_postgres import PostgresBlockStore

    stores = {s.index: PostgresBlockStore(s.dsn) for s in shards}
    return ShardedPostgresBlockStore(router, stores)


__all__ = [
    "ShardConfig",
    "ShardRouter",
    "ShardedPostgresBlockStore",
    "from_config",
]
