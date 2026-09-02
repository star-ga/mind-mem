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

Routing-key discipline: the ring is keyed on ``(tenant_id, namespace)``,
so one tenant that wrote under several namespaces is spread over
several shards. Every tenant-scoped operation therefore either names
the namespace (single-shard route, identical to the key the write
used) or fans out over every shard. It never guesses ``default`` for a
namespace the caller did not give — that guess is what makes a block
written under ``_namespace="prod"`` invisible to the matching read and
undeletable by the matching erasure request.

Failure discipline: a shard that raises is never folded into an empty
or partial answer. :meth:`~ShardedPostgresBlockStore.search` raises
when *every* shard failed (reporting a total outage as "no results"
fabricates an answer); :meth:`~ShardedPostgresBlockStore.get_all`,
:meth:`~ShardedPostgresBlockStore.list_blocks` and
:meth:`~ShardedPostgresBlockStore.diff` raise when *any* shard failed,
because their contract is completeness and an unmarked subset lets a
compliance scan certify a corpus it never read.

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
import os
from dataclasses import dataclass, field
from typing import Any, Iterable

from ..admission import require_admission, require_delete_admission
from ..block_store import BlockStoreError

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

    Shard indices must be distinct: the index is both the ring's
    ownership label and the key into the store map, so a repeated
    index would mint a second set of virtual nodes for a shard whose
    connection has already been overwritten — the ring would send a
    share of the keyspace to a database that does not hold it.
    """

    shards: list[ShardConfig]
    virtual_nodes_per_weight: int = 160
    _ring: list[tuple[int, int]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.shards:
            raise ValueError("ShardRouter requires at least one shard")
        seen: set[int] = set()
        for shard in self.shards:
            if shard.index in seen:
                raise ValueError(f"duplicate shard index {shard.index} — every shard needs a distinct index")
            seen.add(shard.index)
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
        routed = {s.index for s in router.shards}
        held = set(stores_by_shard)
        if routed != held:
            # Fail here rather than with a bare KeyError on the first
            # routed write, or a shard that quietly never sees traffic.
            raise ValueError(
                "router shards and stores_by_shard disagree: "
                f"no store for shard(s) {sorted(routed - held)}, "
                f"store for unrouted shard(s) {sorted(held - routed)}"
            )
        self._router = router
        self._stores = stores_by_shard
        self._default_tenant = default_tenant_id
        self._default_namespace = default_namespace

    # ---- routing ----------------------------------------------------------

    def _shard_indices(self, tenant_id: str, namespace: str | None) -> list[int]:
        """Shard indices that may hold ``tenant_id``'s blocks.

        With a namespace the routing key is complete and exactly one
        shard owns it. Without one the tenant may have written under
        any namespace (``write_block`` honours the block's own
        ``_namespace``), so every shard is a candidate — assuming the
        default namespace here would silently miss those blocks.
        """
        if namespace is None:
            return list(self._stores)
        return [self._router.shard_for(tenant_id, namespace).index]

    # ---- BlockStore write surface -----------------------------------------

    def write_block(
        self,
        block: dict[str, Any],
        *,
        tenant_id: str | None = None,
        namespace: str | None = None,
    ) -> str:
        require_admission(str(block.get("_id") or ""), status=block.get("Status"))
        tid = tenant_id or str(block.get("_tenant") or self._default_tenant)
        # The payload's own ``_namespace`` is part of the routing key,
        # so a read that does not name the same namespace has to fan
        # out to find this block again (see :meth:`_shard_indices`).
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
        """Delete ``block_id`` from the shard that owns it.

        Without a ``namespace`` the owning shard is not determined by
        the tenant alone, so every shard is tried until one reports a
        deletion. Returning ``False`` while the block survives on
        another shard would report a completed erasure that did not
        happen.

        Admission is required here, before the fan-out, rather than left
        to whichever shard happens to own the block: with no shard
        configured for the tenant the loop body never runs, so a router
        that resolves to nothing would otherwise turn an ungated delete
        into a quiet ``False`` instead of a refusal. This wrapper opens
        no scope and records no removal — the owning shard does both.
        """
        require_delete_admission(str(block_id))
        tid = tenant_id or self._default_tenant
        for idx in self._shard_indices(tid, namespace):
            if bool(self._stores[idx].delete_block(block_id)):
                return True
        return False

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
        """Fan out search to every shard, RRF-fuse per-shard rankings.

        A shard that raises is logged and skipped — ranked retrieval
        degrades to the shards that answered. If *every* shard fails
        the call raises: a backend outage presented as "no results" is
        a fabricated answer, the one thing a memory product must never
        return.
        """
        from ..hybrid_recall import rrf_fuse

        per_shard_lists: list[list[dict]] = []
        failures: list[str] = []
        for idx, store in self._stores.items():
            try:
                per_shard_lists.append(store.search(query, limit=limit))
            except Exception as exc:
                failures.append(f"shard {idx}: {exc}")
                _log.warning("shard_search_failed shard=%s: %s", idx, exc)
                continue
        if failures and not per_shard_lists:
            raise BlockStoreError("search failed on every shard: " + "; ".join(failures))
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
        namespace: str | None = None,
    ) -> list[dict[str, Any]]:
        """Blocks for one tenant, or every block for admin/compliance scans.

        ``tenant_id`` + ``namespace`` names the single shard the
        matching write used. ``tenant_id`` alone fans out — the tenant
        may have written under any namespace — and the merged result is
        filtered back down to that tenant. No ``tenant_id`` is the
        unscoped scan.

        Raises:
            BlockStoreError: if any consulted shard failed. This method's
                contract is completeness; a silently short result would
                let a compliance export certify blocks it never read.
        """
        indices = list(self._stores) if tenant_id is None else self._shard_indices(tenant_id, namespace)
        out: list[dict[str, Any]] = []
        failures: list[str] = []
        for idx in indices:
            try:
                out.extend(self._stores[idx].get_all(active_only=active_only))
            except Exception as exc:
                failures.append(f"shard {idx}: {exc}")
                _log.warning("shard_get_all_failed shard=%s: %s", idx, exc)
                continue
        if failures:
            raise BlockStoreError("get_all is incomplete: " + "; ".join(failures))
        if tenant_id is None:
            return out
        return [b for b in out if str(b.get("_tenant") or self._default_tenant) == tenant_id]

    def list_blocks(self) -> list[str]:
        """Union of every shard's artifact list.

        Raises:
            BlockStoreError: if any shard failed — same completeness
                contract as :meth:`get_all`.
        """
        names: set[str] = set()
        failures: list[str] = []
        for idx, store in self._stores.items():
            try:
                names.update(str(name) for name in store.list_blocks())
            except Exception as exc:
                failures.append(f"shard {idx}: {exc}")
                _log.warning("shard_list_blocks_failed shard=%s: %s", idx, exc)
                continue
        if failures:
            raise BlockStoreError("list_blocks is incomplete: " + "; ".join(failures))
        return sorted(names)

    # ---- BlockStore snapshot surface — per-shard ---------------------------

    def snapshot(
        self,
        snap_dir: str,
        *,
        files_touched: list[str] | None = None,
    ) -> dict[str, Any]:
        """Snapshot every shard to its own sub-directory.

        Returns a composite manifest. ``ok`` is False and
        ``failed_shards`` lists the shards whose snapshot raised, so a
        partial backup cannot be recorded as a backup taken — the
        per-shard ``error`` key alone is too easy to miss.
        """
        manifests: dict[str, Any] = {}
        failed: list[str] = []
        for idx, store in self._stores.items():
            shard_dir = os.path.join(snap_dir, f"shard-{idx:02d}")
            os.makedirs(shard_dir, exist_ok=True)
            try:
                manifests[str(idx)] = store.snapshot(shard_dir, files_touched=files_touched)
            except Exception as exc:
                _log.warning("shard_snapshot_failed shard=%s: %s", idx, exc)
                manifests[str(idx)] = {"error": str(exc)}
                failed.append(str(idx))
        return {
            "sharded": True,
            "ok": not failed,
            "failed_shards": failed,
            "shards": manifests,
        }

    def restore(self, snap_dir: str) -> None:
        """Restore every shard from ``snap_dir/shard-NN``.

        Raises:
            BlockStoreError: if any shard's directory is absent, before
                anything is restored. :meth:`snapshot` creates one
                directory per shard, so a missing one means this is not
                this cluster's snapshot (wrong path, partial copy, or a
                shard added since). Restoring only the shards that
                happen to be present and returning ``None`` is
                indistinguishable from a full restore.
        """
        missing = [f"shard-{idx:02d}" for idx in self._stores if not os.path.isdir(os.path.join(snap_dir, f"shard-{idx:02d}"))]
        if missing:
            raise BlockStoreError(f"snapshot {snap_dir} has no directory for {', '.join(missing)} — refusing a partial restore")
        for idx, store in self._stores.items():
            store.restore(os.path.join(snap_dir, f"shard-{idx:02d}"))

    def diff(self, snap_dir: str) -> list[str]:
        """Per-shard diff against a snapshot.

        Raises:
            BlockStoreError: if any shard failed — a swallowed shard
                error reads as "this shard matches the snapshot".
        """
        out: list[str] = []
        failures: list[str] = []
        for idx, store in self._stores.items():
            shard_dir = os.path.join(snap_dir, f"shard-{idx:02d}")
            try:
                out.extend(store.diff(shard_dir))
            except Exception as exc:
                failures.append(f"shard {idx}: {exc}")
                _log.warning("shard_diff_failed shard=%s: %s", idx, exc)
                continue
        if failures:
            raise BlockStoreError("diff is incomplete: " + "; ".join(failures))
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
        try:
            for store in self._stores:
                lock_cm = store.lock(blocking=self._blocking, timeout=self._timeout)
                lock_cm.__enter__()
                self._acquired.append(lock_cm)
        except BaseException:
            # Python never calls ``__exit__`` for a context manager
            # whose ``__enter__`` raised, and this object is dropped
            # with the exception — so nothing else would ever release
            # the shards already locked. The Postgres lock is a row
            # with no finalizer: an orphan outlives the process and
            # wedges every later acquirer until it times out.
            self._release_all(None, None, None)
            raise
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._release_all(exc_type, exc, tb)

    def _release_all(self, exc_type, exc, tb) -> None:
        # Release in reverse-acquired order — tidiness, not correctness.
        acquired, self._acquired = self._acquired, []
        for lock_cm in reversed(acquired):
            try:
                lock_cm.__exit__(exc_type, exc, tb)
            except Exception as exc2:
                _log.warning("shard_lock_release_failed: %s", exc2)
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

    A repeated ``index`` is rejected by :class:`ShardRouter` before any
    connection is built: the store map is keyed by index, so the later
    DSN would silently replace the earlier one while the ring still
    handed that index a double share of the keyspace.
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
