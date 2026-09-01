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

    def test_write_routes_to_single_shard(self, admitted) -> None:
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

    def test_get_all_with_tenant_and_namespace_routes_to_one_shard(self) -> None:
        store, a, b = self._build()
        # (tenant, namespace) is the whole routing key, so naming both
        # identifies a single owning shard. Without the namespace the
        # tenant has no single owner and the call must fan out (see
        # TestNamespaceRoutingSymmetry).
        store.get_all(tenant_id="acme", namespace="default")
        # Only one of the underlying stores gets the call.
        assert (a.get_all.call_count + b.get_all.call_count) == 1


# ---------------------------------------------------------------------------
# Regression tests for the 2026-08-29 audit findings.
# ---------------------------------------------------------------------------


class _FakeShard:
    """In-memory stand-in for one ``PostgresBlockStore``."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.blocks: dict[str, dict] = {}
        self.restored_from: list[str] = []

    def write_block(self, block: dict) -> str:
        self.blocks[str(block["_id"])] = dict(block)
        return str(block["_id"])

    def delete_block(self, block_id: str) -> bool:
        return self.blocks.pop(block_id, None) is not None

    def get_all(self, *, active_only: bool = False) -> list[dict]:
        return [dict(b) for b in self.blocks.values()]

    def get_by_id(self, block_id: str) -> dict | None:
        found = self.blocks.get(block_id)
        return dict(found) if found else None

    def search(self, query: str, *, limit: int = 10) -> list[dict]:
        return [dict(b) for b in self.blocks.values()][:limit]

    def list_blocks(self) -> list[str]:
        return [f"{self.name}.md"]

    def snapshot(self, snap_dir: str, *, files_touched: list[str] | None = None) -> dict:
        return {"shard": self.name, "files": len(self.blocks)}

    def restore(self, snap_dir: str) -> None:
        self.restored_from.append(snap_dir)

    def diff(self, snap_dir: str) -> list[str]:
        return []


class _ExplodingShard(_FakeShard):
    """A shard whose every operation raises — an unreachable database."""

    def _boom(self, *args, **kwargs):
        raise RuntimeError(f"{self.name} unreachable")

    write_block = _boom
    delete_block = _boom
    get_all = _boom
    search = _boom
    list_blocks = _boom
    snapshot = _boom
    restore = _boom
    diff = _boom


def _cluster(n: int = 4, failing: set[int] | None = None):
    """Build an ``n``-shard store; indices in ``failing`` raise on use."""
    failing = failing or set()
    shards = [ShardConfig(index=i, dsn=f"pg://shard{i}") for i in range(n)]
    router = ShardRouter(shards=shards)
    stores = {i: (_ExplodingShard(f"s{i}") if i in failing else _FakeShard(f"s{i}")) for i in range(n)}
    return ShardedPostgresBlockStore(router, stores), stores, router


def _namespace_off_the_default_shard(router: ShardRouter, tenant: str) -> str:
    """A namespace whose (tenant, namespace) key lands on another shard."""
    home = router.shard_for(tenant, "default").index
    for candidate in ("prod", "staging", "eu", "ns-1", "ns-2", "ns-3", "ns-4"):
        if router.shard_for(tenant, candidate).index != home:
            return candidate
    raise AssertionError("no namespace routes off the default shard — fixture assumption broken")


class TestNamespaceRoutingSymmetry:
    """A block written under its own ``_namespace`` must stay findable.

    The write side routes on the payload's ``_namespace``; a read or an
    erasure that has no namespace to give must therefore not assume
    ``default`` — that assumption returns ``[]`` for a block that exists
    and reports ``False`` for a deletion that never happened.
    """

    def test_get_all_finds_a_block_written_under_another_namespace(self, admitted) -> None:
        store, stores, router = _cluster()
        ns = _namespace_off_the_default_shard(router, "acme")
        store.write_block({"_id": "D-1", "_tenant": "acme", "_namespace": ns})

        found = store.get_all(tenant_id="acme")

        assert [b["_id"] for b in found] == ["D-1"]

    def test_delete_reaches_a_block_written_under_another_namespace(self, admitted) -> None:
        store, stores, router = _cluster()
        ns = _namespace_off_the_default_shard(router, "acme")
        store.write_block({"_id": "D-1", "_tenant": "acme", "_namespace": ns})

        assert store.delete_block("D-1", tenant_id="acme") is True
        assert store.get_by_id("D-1") is None

    def test_tenant_scoped_get_all_excludes_other_tenants(self, admitted) -> None:
        store, stores, router = _cluster()
        store.write_block({"_id": "D-1", "_tenant": "acme"})
        store.write_block({"_id": "D-2", "_tenant": "globex"})

        assert [b["_id"] for b in store.get_all(tenant_id="acme")] == ["D-1"]


class TestShardFailuresAreNotSwallowed:
    def test_search_raises_when_every_shard_fails(self) -> None:
        from mind_mem.block_store import BlockStoreError

        store, _stores, _router = _cluster(failing={0, 1, 2, 3})
        with pytest.raises(BlockStoreError):
            store.search("pqc signing decision")

    def test_search_degrades_when_one_shard_fails(self, admitted) -> None:
        store, stores, _router = _cluster(failing={2})
        stores[0].blocks["D-1"] = {"_id": "D-1"}
        assert [h["_id"] for h in store.search("anything")] == ["D-1"]

    def test_get_all_raises_when_one_shard_fails(self) -> None:
        from mind_mem.block_store import BlockStoreError

        store, _stores, _router = _cluster(failing={2})
        with pytest.raises(BlockStoreError):
            store.get_all()

    def test_list_blocks_raises_when_one_shard_fails(self) -> None:
        from mind_mem.block_store import BlockStoreError

        store, _stores, _router = _cluster(failing={2})
        with pytest.raises(BlockStoreError):
            store.list_blocks()

    def test_diff_raises_when_one_shard_fails(self) -> None:
        from mind_mem.block_store import BlockStoreError

        store, _stores, _router = _cluster(failing={2})
        with pytest.raises(BlockStoreError):
            store.diff("/tmp/does-not-matter")


class TestSnapshotAndRestoreHonesty:
    def test_snapshot_manifest_flags_a_failed_shard(self, tmp_path) -> None:
        store, _stores, _router = _cluster(failing={2})
        manifest = store.snapshot(str(tmp_path))
        assert manifest["ok"] is False
        assert manifest["failed_shards"] == ["2"]

    def test_snapshot_manifest_is_ok_when_every_shard_succeeded(self, tmp_path) -> None:
        store, _stores, _router = _cluster()
        manifest = store.snapshot(str(tmp_path))
        assert manifest["ok"] is True
        assert manifest["failed_shards"] == []

    def test_restore_refuses_a_directory_with_no_shard_dirs(self, tmp_path) -> None:
        from mind_mem.block_store import BlockStoreError

        store, stores, _router = _cluster()
        with pytest.raises(BlockStoreError):
            store.restore(str(tmp_path))
        assert all(not s.restored_from for s in stores.values())

    def test_restore_refuses_a_partial_snapshot_without_touching_a_shard(self, tmp_path) -> None:
        from mind_mem.block_store import BlockStoreError

        store, stores, _router = _cluster()
        for idx in (0, 1, 2):  # shard-03 missing
            (tmp_path / f"shard-{idx:02d}").mkdir()
        with pytest.raises(BlockStoreError):
            store.restore(str(tmp_path))
        assert all(not s.restored_from for s in stores.values())

    def test_restore_runs_when_every_shard_dir_is_present(self, tmp_path) -> None:
        store, stores, _router = _cluster()
        for idx in range(4):
            (tmp_path / f"shard-{idx:02d}").mkdir()
        store.restore(str(tmp_path))
        assert all(len(s.restored_from) == 1 for s in stores.values())


class _RecordingLock:
    """Minimal ``PostgresBlockStore.lock()`` stand-in that records state."""

    def __init__(self, holder: "_LockingShard") -> None:
        self._holder = holder

    def __enter__(self) -> "_RecordingLock":
        if self._holder.explode:
            raise TimeoutError(f"{self._holder.name} lock timed out")
        self._holder.held = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._holder.held = False
        self._holder.released = True


class _LockingShard:
    def __init__(self, name: str, *, explode: bool = False) -> None:
        self.name = name
        self.explode = explode
        self.held = False
        self.released = False

    def lock(self, *, blocking: bool = True, timeout: float = 30.0) -> _RecordingLock:
        return _RecordingLock(self)


class TestFanOutLock:
    def test_a_failing_shard_releases_the_locks_already_taken(self) -> None:
        """No orphaned shard locks when one shard's lock raises.

        The Postgres lock is a row with no finalizer, so a lock left
        held by a dropped context manager outlives the process and
        wedges every later acquirer on the healthy shards.
        """
        shards = [
            _LockingShard("s0"),
            _LockingShard("s1"),
            _LockingShard("s2", explode=True),
            _LockingShard("s3"),
        ]
        router = ShardRouter(shards=[ShardConfig(index=i, dsn=f"pg://{i}") for i in range(4)])
        store = ShardedPostgresBlockStore(router, dict(enumerate(shards)))

        with pytest.raises(TimeoutError):
            with store.lock():
                pass  # pragma: no cover - never reached

        assert [s.held for s in shards] == [False, False, False, False]
        assert [s.released for s in shards[:2]] == [True, True]

    def test_normal_exit_still_releases_every_shard(self) -> None:
        shards = [_LockingShard(f"s{i}") for i in range(3)]
        router = ShardRouter(shards=[ShardConfig(index=i, dsn=f"pg://{i}") for i in range(3)])
        store = ShardedPostgresBlockStore(router, dict(enumerate(shards)))
        with store.lock():
            assert all(s.held for s in shards)
        assert all(s.released and not s.held for s in shards)


class TestConfigMistakesFailLoudly:
    def test_router_rejects_a_duplicate_shard_index(self) -> None:
        with pytest.raises(ValueError, match="duplicate shard index"):
            ShardRouter(
                shards=[
                    ShardConfig(index=0, dsn="pg://a"),
                    ShardConfig(index=1, dsn="pg://b"),
                    ShardConfig(index=1, dsn="pg://c"),
                ]
            )

    def test_from_config_rejects_a_duplicate_shard_index(self) -> None:
        from mind_mem.storage.sharded_pg import from_config

        config = {
            "block_store": {
                "backend": "sharded_postgres",
                "shards": [
                    {"index": 0, "dsn": "pg://a"},
                    {"index": 1, "dsn": "pg://b"},
                    {"index": 1, "dsn": "pg://c"},
                ],
            }
        }
        with pytest.raises(ValueError, match="duplicate shard index"):
            from_config(config)

    def test_store_map_must_cover_exactly_the_routed_shards(self) -> None:
        router = ShardRouter(shards=[ShardConfig(index=0, dsn="a"), ShardConfig(index=1, dsn="b")])
        with pytest.raises(ValueError, match="disagree"):
            ShardedPostgresBlockStore(router, {0: _FakeShard("s0")})


def test_implements_the_block_store_protocol() -> None:
    """The module docstring claims full Protocol coverage — prove it."""
    from mind_mem.block_store import BlockStore

    store, _stores, _router = _cluster(n=2)
    assert isinstance(store, BlockStore)
