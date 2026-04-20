"""v3.2.0 — tests for read-replica routing in ReplicatedPostgresBlockStore."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def rep_store() -> MagicMock:
    """Build a ReplicatedPostgresBlockStore with fully-mocked Postgres backends."""
    # Patch ``PostgresBlockStore`` so construction doesn't touch a real DB.
    with patch("mind_mem.block_store_postgres_replica.PostgresBlockStore") as mock_cls:

        def _make_instance(dsn: str, **_: object) -> MagicMock:
            inst = MagicMock()
            inst._dsn = dsn
            # Default behaviours — override per test as needed.
            inst.get_all.return_value = [{"_id": "x", "dsn": dsn}]
            inst.get_by_id.return_value = {"_id": "x", "dsn": dsn}
            inst.search.return_value = [{"_id": "x", "dsn": dsn}]
            inst.list_blocks.return_value = [f"{dsn}/file.md"]
            inst.diff.return_value = []
            inst.write_block.return_value = "x"
            inst.delete_block.return_value = True
            inst.snapshot.return_value = {"files": []}
            inst.restore.return_value = None
            return inst

        mock_cls.side_effect = _make_instance

        from mind_mem.block_store_postgres_replica import ReplicatedPostgresBlockStore

        store = ReplicatedPostgresBlockStore(
            primary_dsn="postgresql://primary/db",
            replica_dsns=[
                "postgresql://replica-a/db",
                "postgresql://replica-b/db",
            ],
        )
        yield store


class TestReadRouting:
    def test_get_all_uses_replica(self, rep_store) -> None:
        result = rep_store.get_all()
        assert isinstance(result, list)
        # Either replica serves the request; primary is not touched.
        rep_store._primary.get_all.assert_not_called()

    def test_get_by_id_uses_replica(self, rep_store) -> None:
        rep_store.get_by_id("X")
        rep_store._primary.get_by_id.assert_not_called()

    def test_search_uses_replica(self, rep_store) -> None:
        rep_store.search("foo", limit=5)
        rep_store._primary.search.assert_not_called()

    def test_round_robin_between_replicas(self, rep_store) -> None:
        """Successive reads alternate between the two replicas."""
        dsns_used: list[str] = []
        # Seed both replicas with returning-their-dsn fakes.
        for rep in rep_store._replicas:

            def _probe(*args, _dsn=rep.store._dsn, **kwargs):
                dsns_used.append(_dsn)
                return []

            rep.store.list_blocks.side_effect = _probe

        for _ in range(4):
            rep_store.list_blocks()

        # 4 calls → 2 each across the 2 replicas.
        counts = {d: dsns_used.count(d) for d in set(dsns_used)}
        assert len(counts) == 2
        for dsn, n in counts.items():
            assert n == 2, f"{dsn} got {n} calls; expected 2"


class TestWriteRouting:
    def test_write_block_always_primary(self, rep_store) -> None:
        rep_store.write_block({"_id": "D-1"})
        rep_store._primary.write_block.assert_called_once_with({"_id": "D-1"})
        for rep in rep_store._replicas:
            rep.store.write_block.assert_not_called()

    def test_delete_block_always_primary(self, rep_store) -> None:
        rep_store.delete_block("D-1")
        rep_store._primary.delete_block.assert_called_once_with("D-1")

    def test_snapshot_always_primary(self, rep_store) -> None:
        rep_store.snapshot("/tmp/snap")
        rep_store._primary.snapshot.assert_called_once()

    def test_restore_always_primary(self, rep_store) -> None:
        rep_store.restore("/tmp/snap")
        rep_store._primary.restore.assert_called_once_with("/tmp/snap")


class TestCircuitBreaker:
    def test_replica_failure_falls_back_to_primary(self, rep_store) -> None:
        rep_store._replicas[0].store.get_by_id.side_effect = RuntimeError("replica dead")
        rep_store._replicas[1].store.get_by_id.return_value = {"_id": "from-rep-b"}

        # First call hits replica-a (fails, falls back to primary — returns
        # whatever the primary mock returns).
        rep_store.get_by_id("X")
        rep_store._primary.get_by_id.assert_called_once()

    def test_repeated_failures_trip_breaker(self, rep_store) -> None:
        """After 3 failures on replica-a the breaker opens."""
        rep_store._replicas[0].store.get_by_id.side_effect = RuntimeError("replica dead")
        # We'll hit replica-a repeatedly; round-robin still alternates,
        # so we need 3 direct hits. Use the state API directly to
        # drive failures without depending on rr ordering.
        for _ in range(3):
            rep_store._replicas[0].record_failure()
        assert not rep_store._replicas[0].healthy

    def test_healthy_replica_recovers(self, rep_store) -> None:
        rep = rep_store._replicas[0]
        rep.record_failure()
        rep.record_failure()
        rep.record_success()
        assert rep.failure_count == 0
        assert rep.healthy

    def test_all_replicas_cooling_falls_back_to_primary(self, rep_store) -> None:
        for rep in rep_store._replicas:
            rep.cooling_until = time.time() + 60
        result = rep_store.get_all()
        # Either the primary got called, or we got an iterable back.
        rep_store._primary.get_all.assert_called_once()
        assert isinstance(result, list)


class TestBuildFromConfig:
    def test_no_replicas_returns_plain_postgres_store(self) -> None:
        with patch("mind_mem.block_store_postgres_replica.PostgresBlockStore") as mock_cls:
            from mind_mem.block_store_postgres_replica import build_from_config

            build_from_config({"block_store": {"dsn": "postgresql://x/y"}})
            # One construction — the primary. No ReplicatedPostgresBlockStore.
            assert mock_cls.call_count == 1

    def test_replicas_returns_replicated_store(self) -> None:
        with patch("mind_mem.block_store_postgres_replica.PostgresBlockStore"):
            from mind_mem.block_store_postgres_replica import (
                ReplicatedPostgresBlockStore,
                build_from_config,
            )

            store = build_from_config(
                {
                    "block_store": {
                        "dsn": "postgresql://primary/db",
                        "replicas": ["postgresql://r1/db", "postgresql://r2/db"],
                    }
                }
            )
            assert isinstance(store, ReplicatedPostgresBlockStore)

    def test_missing_dsn_raises(self) -> None:
        from mind_mem.block_store_postgres_replica import build_from_config

        with pytest.raises(ValueError, match="block_store.dsn"):
            build_from_config({"block_store": {}})

    def test_malformed_replicas_list_raises(self) -> None:
        from mind_mem.block_store_postgres_replica import build_from_config

        with pytest.raises(ValueError, match="replicas"):
            build_from_config(
                {"block_store": {"dsn": "x", "replicas": "not-a-list"}}
            )
