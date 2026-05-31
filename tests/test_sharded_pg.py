"""v4.0 prep — sharded Postgres routing tests (mock underlying stores)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from mind_mem.storage.sharded_pg import (
    ShardConfig,
    ShardedPostgresBlockStore,
    ShardRouter,
)


class TestShardConfig:
    def test_rejects_negative_index(self) -> None:
        with pytest.raises(ValueError):
            ShardConfig(index=-1, dsn="x")

    def test_rejects_zero_weight(self) -> None:
        with pytest.raises(ValueError):
            ShardConfig(index=0, dsn="x", weight=0)


class TestShardRouter:
    def test_rejects_empty_shards(self) -> None:
        with pytest.raises(ValueError):
            ShardRouter(shards=[])

    def test_single_shard_routes_every_key(self) -> None:
        r = ShardRouter(shards=[ShardConfig(index=0, dsn="x")])
        for key in ("alpha", "beta", "gamma", ""):
            assert r.route(key) == 0

    def test_deterministic_routing(self) -> None:
        r = ShardRouter(shards=[ShardConfig(index=0, dsn="a"), ShardConfig(index=1, dsn="b")])
        first = r.route("tenant:alice")
        second = r.route("tenant:alice")
        assert first == second
        assert first in (0, 1)

    def test_different_tenants_can_hit_different_shards(self) -> None:
        r = ShardRouter(shards=[ShardConfig(index=0, dsn="a"), ShardConfig(index=1, dsn="b")])
        placements = {r.shard_for(f"tenant-{i}").index for i in range(50)}
        # With two shards + 50 tenants, both should see traffic.
        assert placements == {0, 1}

    def test_fan_out_shards_returns_all(self) -> None:
        shards = [ShardConfig(index=0, dsn="a"), ShardConfig(index=1, dsn="b")]
        r = ShardRouter(shards=shards)
        assert {s.index for s in r.fan_out_shards()} == {0, 1}

    def test_weight_affects_load(self) -> None:
        shards = [
            ShardConfig(index=0, dsn="a", weight=1),
            ShardConfig(index=1, dsn="b", weight=4),
        ]
        r = ShardRouter(shards=shards)
        counts = {0: 0, 1: 0}
        for i in range(2000):
            counts[r.shard_for(f"tenant-{i}").index] += 1
        # Shard 1 should get roughly 4x the traffic of shard 0.
        assert counts[1] > counts[0] * 2  # generous bound for variance


class TestShardedPostgresBlockStore:
    def _fake_store(self, name: str) -> MagicMock:
        store = MagicMock()
        store.name = name
        store.write_block.return_value = f"block-id-from-{name}"
        store.get_by_id.return_value = None
        store.search.return_value = [{"_id": f"D-{name}", "score": 1.0}]
        store.get_all.return_value = [{"_id": f"D-{name}-all"}]
        return store

    def _build(self) -> tuple[ShardedPostgresBlockStore, MagicMock, MagicMock]:
        a = self._fake_store("A")
        b = self._fake_store("B")
        shards = [ShardConfig(index=0, dsn="a"), ShardConfig(index=1, dsn="b")]
        router = ShardRouter(shards=shards)
        store = ShardedPostgresBlockStore(router, {0: a, 1: b})
        return store, a, b

    def test_write_routes_to_single_shard(self) -> None:
        store, a, b = self._build()
        store.write_block({"_id": "D-1"}, tenant_id="acme", namespace="default")
        # Exactly one underlying shard got the write.
        assert (a.write_block.call_count + b.write_block.call_count) == 1

    def test_delete_routes_to_single_shard(self) -> None:
        store, a, b = self._build()
        a.delete_block.return_value = True
        b.delete_block.return_value = True
        store.delete_block("D-1", tenant_id="acme")
        assert (a.delete_block.call_count + b.delete_block.call_count) == 1

    def test_search_fans_out_and_fuses(self) -> None:
        store, a, b = self._build()
        out = store.search("hello", limit=5)
        a.search.assert_called_once()
        b.search.assert_called_once()
        assert len(out) == 2
        # Both shards' results made it into the fused list.
        ids = {r["_id"] for r in out}
        assert "D-A" in ids and "D-B" in ids

    def test_get_by_id_stops_on_first_hit(self) -> None:
        store, a, b = self._build()
        a.get_by_id.return_value = {"_id": "D-1", "from": "A"}
        result = store.get_by_id("D-1")
        assert result is not None
        assert result["from"] == "A"
        # b may or may not be consulted depending on dict iteration
        # order; no contract there, so don't assert.

    def test_get_all_all_tenants_fans_out(self) -> None:
        store, a, b = self._build()
        out = store.get_all()
        assert len(out) == 2  # one from each shard

    def test_get_all_with_tenant_routes_to_one_shard(self) -> None:
        store, a, b = self._build()
        store.get_all(tenant_id="acme")
        # Only one of the underlying stores gets the call.
        assert (a.get_all.call_count + b.get_all.call_count) == 1
