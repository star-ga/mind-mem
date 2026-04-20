"""v3.2.0 — read-replica routing for PostgresBlockStore.

Wraps a primary ``PostgresBlockStore`` with a pool of read replicas
so read-heavy MCP tools (``recall``, ``find_similar``,
``hybrid_search``, ``prefetch``) spread load across replicas while
writes always hit the primary.

Usage — ``mind-mem.json``::

    {
        "block_store": {
            "backend": "postgres",
            "dsn": "postgresql://mindmem@primary:5432/mindmem",
            "replicas": [
                "postgresql://mindmem@replica-1:5432/mindmem",
                "postgresql://mindmem@replica-2:5432/mindmem"
            ]
        }
    }

The replicated store is a transparent :class:`BlockStore` wrapper:

* ``get_all`` / ``get_by_id`` / ``search`` / ``list_blocks`` / ``diff``
  → route to a replica via round-robin.
* ``write_block`` / ``delete_block`` / ``snapshot`` / ``restore`` /
  ``lock`` → always go to the primary.

When a replica raises on a read, the wrapper falls back to the
primary (fail-open). A repeated-failure circuit breaker marks a
replica unhealthy after 3 consecutive failures so subsequent reads
skip it for a 30-second cool-down.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from .block_store import BlockStore, BlockStoreError
from .block_store_postgres import PostgresBlockStore
from .observability import get_logger, metrics

_log = get_logger("postgres_replica")

_CIRCUIT_BREAKER_FAILURES = 3
_CIRCUIT_BREAKER_COOLDOWN_SECONDS = 30.0


@dataclass
class _ReplicaState:
    """Per-replica health state used by the circuit breaker."""

    store: PostgresBlockStore
    failure_count: int = 0
    cooling_until: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def healthy(self) -> bool:
        return time.time() >= self.cooling_until

    def record_failure(self) -> None:
        with self.lock:
            self.failure_count += 1
            if self.failure_count >= _CIRCUIT_BREAKER_FAILURES:
                self.cooling_until = time.time() + _CIRCUIT_BREAKER_COOLDOWN_SECONDS
                _log.warning(
                    "replica_unhealthy",
                    failure_count=self.failure_count,
                    cooldown_until=self.cooling_until,
                )

    def record_success(self) -> None:
        with self.lock:
            self.failure_count = 0
            self.cooling_until = 0.0


class ReplicatedPostgresBlockStore:
    """Read-replica-aware BlockStore wrapper.

    Constructed via :class:`mind_mem.storage.get_block_store` when the
    config declares ``block_store.replicas``. Reads route through the
    replicas; writes always hit the primary.
    """

    def __init__(
        self,
        primary_dsn: str,
        replica_dsns: list[str],
        *,
        schema: str = "mind_mem",
        workspace: str | None = None,
    ) -> None:
        if not replica_dsns:
            raise ValueError("ReplicatedPostgresBlockStore requires ≥1 replica DSN")
        self._primary = PostgresBlockStore(primary_dsn, schema=schema, workspace=workspace)
        self._replicas: list[_ReplicaState] = [
            _ReplicaState(PostgresBlockStore(dsn, schema=schema, workspace=workspace))
            for dsn in replica_dsns
        ]
        self._rr_counter = 0
        self._rr_lock = threading.Lock()

    # ─── replica selection ───────────────────────────────────────────

    def _pick_replica(self) -> _ReplicaState | None:
        """Round-robin over healthy replicas. None when all are cooling."""
        n = len(self._replicas)
        with self._rr_lock:
            start = self._rr_counter
            self._rr_counter = (self._rr_counter + 1) % n
        for offset in range(n):
            idx = (start + offset) % n
            rep = self._replicas[idx]
            if rep.healthy:
                return rep
        return None

    def _run_on_replica(self, method_name: str, *args, **kwargs) -> Any:
        """Route a read through a replica; fall back to primary on failure."""
        rep = self._pick_replica()
        if rep is None:
            metrics.inc("replica_all_cooling_fallback_primary")
            return getattr(self._primary, method_name)(*args, **kwargs)
        try:
            result = getattr(rep.store, method_name)(*args, **kwargs)
            rep.record_success()
            metrics.inc("replica_read_success")
            return result
        except (BlockStoreError, Exception) as exc:
            rep.record_failure()
            metrics.inc("replica_read_failure")
            _log.warning("replica_read_failed", method=method_name, error=str(exc))
            return getattr(self._primary, method_name)(*args, **kwargs)

    # ─── Read surface → replica ──────────────────────────────────────

    def get_all(self, *, active_only: bool = False) -> list[dict[str, Any]]:
        return self._run_on_replica("get_all", active_only=active_only)

    def get_by_id(self, block_id: str) -> dict[str, Any] | None:
        return self._run_on_replica("get_by_id", block_id)

    def search(self, query: str, *, limit: int = 10) -> list[dict[str, Any]]:
        return self._run_on_replica("search", query, limit=limit)

    def list_blocks(self) -> list[str]:
        return self._run_on_replica("list_blocks")

    def diff(self, snap_dir: str) -> list[str]:
        return self._run_on_replica("diff", snap_dir)

    # ─── Write surface → primary ─────────────────────────────────────

    def write_block(self, block: dict[str, Any]) -> str:
        metrics.inc("replica_write_primary")
        return self._primary.write_block(block)

    def delete_block(self, block_id: str) -> bool:
        metrics.inc("replica_write_primary")
        return self._primary.delete_block(block_id)

    def snapshot(
        self,
        snap_dir: str,
        *,
        files_touched: list[str] | None = None,
    ) -> dict[str, Any]:
        return self._primary.snapshot(snap_dir, files_touched=files_touched)

    def restore(self, snap_dir: str) -> None:
        self._primary.restore(snap_dir)

    def lock(self, *, blocking: bool = True, timeout: float = 30.0) -> Any:
        return self._primary.lock(blocking=blocking, timeout=timeout)

    # ─── Protocol deprecation shim ──────────────────────────────────

    def list_files(self) -> list[str]:
        """Deprecated alias for :meth:`list_blocks` — removed in v4.0."""
        import warnings

        warnings.warn(
            "BlockStore.list_files() is deprecated; use list_blocks() instead. "
            "The alias will be removed in v4.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.list_blocks()


def build_from_config(config: dict, workspace: str | None = None) -> BlockStore:
    """Construct the right store shape based on ``block_store.replicas``.

    Returns ``PostgresBlockStore`` when no replicas are configured, or
    ``ReplicatedPostgresBlockStore`` when ``block_store.replicas`` is
    a non-empty list. Caller is the storage factory
    (:mod:`mind_mem.storage`).
    """
    bs_cfg = config.get("block_store", {}) if isinstance(config, dict) else {}
    if not isinstance(bs_cfg, dict):
        bs_cfg = {}
    primary_dsn = bs_cfg.get("dsn")
    if not primary_dsn:
        raise ValueError("block_store.dsn is required for the postgres backend")
    schema = bs_cfg.get("schema", "mind_mem")
    replicas = bs_cfg.get("replicas") or []
    if not isinstance(replicas, list):
        raise ValueError("block_store.replicas must be a list of DSN strings")
    replicas = [r for r in replicas if isinstance(r, str) and r.strip()]

    if replicas:
        return ReplicatedPostgresBlockStore(
            primary_dsn=primary_dsn,
            replica_dsns=replicas,
            schema=schema,
            workspace=workspace,
        )
    return PostgresBlockStore(primary_dsn, schema=schema, workspace=workspace)
