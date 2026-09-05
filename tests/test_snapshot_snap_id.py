# Copyright 2026 STARGA, Inc.
"""A snapshot has an identity; only some backends need it to have a location.

Roadmap item ``PostgresBlockStore.snapshot(snap_id=…)``. Until 5.0.2 every
backend read the snapshot's identity off the basename of a directory it was
handed, which meant the Postgres backend — whose blocks of record are rows,
not files — could not snapshot at all on a deployment where the API host
shares no filesystem with the operator. ``snap_id`` is now a first-class
parameter on ``snapshot`` / ``restore`` / ``diff`` across every backend, and
the on-disk ``MANIFEST.json`` is an *export* rather than the record.

Two halves are proved separately because they fail separately:

* the identity rules (:mod:`mind_mem.block_store` helpers) — pure logic, run
  everywhere, including the traversal refusals the manifest-containment
  hardening in ``apply_engine._block_ids_in_snapshot`` paid for; and
* the backends — Markdown always, Postgres only against a live database.

Every Postgres test here calls ``pytest.skip`` with a named reason when
``MIND_MEM_TEST_PG_DSN`` is unset, so a run with no database reports skips
and never a pass. Run them with::

    MIND_MEM_TEST_PG_DSN="postgresql://…" pytest tests/test_snapshot_snap_id.py -rs
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Iterator

import pytest
from _restore_scope import restoring

from mind_mem.admission import UngatedRestoreError
from mind_mem.block_store import (
    MarkdownBlockStore,
    reject_unsafe_snap_id,
    resolve_snapshot_target,
    validate_snap_id,
)

_DSN_ENV = "MIND_MEM_TEST_PG_DSN"


# ─── The identity rules ───────────────────────────────────────────────────────


class TestSnapshotIdentityRules:
    """``resolve_snapshot_target`` is the single definition of the pair."""

    def test_directory_only_keeps_the_basename_rule(self) -> None:
        assert resolve_snapshot_target("/srv/snaps/20260904-120000", None) == (
            "/srv/snaps/20260904-120000",
            "20260904-120000",
        )

    def test_trailing_separator_does_not_swallow_the_identity(self) -> None:
        assert resolve_snapshot_target("/srv/snaps/snap-1/", None)[1] == "snap-1"

    def test_id_only_has_no_location(self) -> None:
        assert resolve_snapshot_target(None, "snap-1") == (None, "snap-1")

    def test_both_are_allowed_when_they_agree(self) -> None:
        assert resolve_snapshot_target("/srv/snaps/snap-1", "snap-1") == ("/srv/snaps/snap-1", "snap-1")

    def test_both_are_refused_when_they_disagree(self) -> None:
        """A row filed under one id and a MANIFEST.json in a directory named
        another is the exact trap the cross-backend export exists to avoid:
        the export is there so a Markdown store can restore the same
        snapshot, and it can only find it by name."""
        with pytest.raises(ValueError, match="disagrees"):
            resolve_snapshot_target("/srv/snaps/snap-1", "snap-2")

    def test_neither_is_refused(self) -> None:
        with pytest.raises(ValueError, match="neither was given"):
            resolve_snapshot_target(None, None)

    @pytest.mark.parametrize("bad", ["../escape", "a/b", "a\\b", "..", ".", "", "nul\x00id", "bell\x07id"])
    def test_a_traversing_id_is_refused(self, bad: str) -> None:
        """The floor, applied to an id however it was obtained.

        ``_block_ids_in_snapshot`` was reachable by a crafted manifest that
        named a file outside the snapshot; the fix routes manifest entries
        through ``_safe_child_path``. An id is the other half of the same
        surface — on a filesystem-backed store it becomes a directory name —
        so it never gets to carry a separator either.
        """
        with pytest.raises(ValueError):
            reject_unsafe_snap_id(bad)

    @pytest.mark.parametrize("bad", ["2026-09-04T12:00:00", "sn*ap", 'q"id', "a|b", "x" * 201])
    def test_an_explicit_id_must_also_be_a_legal_path_component(self, bad: str) -> None:
        """Explicit ids get the portability rules as well: an id minted on a
        database-only host becomes a directory name the moment the same
        snapshot is exported or taken through a filesystem-backed store."""
        with pytest.raises(ValueError):
            validate_snap_id(bad)

    def test_a_derived_id_is_not_held_to_the_portability_rules(self) -> None:
        """Deliberate asymmetry, and the reason it is safe.

        A directory named ``2026-09-04T12:00:00`` is a snapshot that already
        exists on someone's disk. The portability rules constrain a NEW id;
        they must not start refusing a caller that worked, so the derived
        path gets the security floor only.
        """
        assert resolve_snapshot_target("/srv/snaps/2026-09-04T12:00:00", None)[1] == "2026-09-04T12:00:00"


# ─── Markdown backend: an id resolves to a location ───────────────────────────


def _md_block(bid: str, statement: str) -> dict:
    return {"_id": bid, "_source_file": "decisions/DECISIONS.md", "Statement": statement, "Status": "active"}


class TestMarkdownSnapId:
    def test_id_addressed_snapshot_round_trips(self, tmp_path: Path, admitted: None) -> None:
        """Seed known ids, snapshot BY ID, mutate, restore BY ID, prove the
        exact ids and content came back — not merely that the calls returned.
        """
        ws = tmp_path
        (ws / "decisions").mkdir()
        store = MarkdownBlockStore(str(ws))
        store.write_block(_md_block("D-1", "before"))
        store.write_block(_md_block("D-2", "also before"))

        manifest = store.snapshot(snap_id="snap-alpha")

        # The snapshot CONTAINS the blocks: its own copy of the corpus file
        # is on disk under the resolved id, and it holds both seeded ids.
        snap_root = ws / "intelligence" / "applied" / "snap-alpha"
        assert (snap_root / "MANIFEST.json").is_file()
        assert "decisions/DECISIONS.md" in manifest["files"]
        captured = (snap_root / "decisions" / "DECISIONS.md").read_text(encoding="utf-8")
        assert "D-1" in captured and "D-2" in captured, "the snapshot does not contain the seeded ids"

        store.write_block(_md_block("D-1", "after"))
        store.write_block(_md_block("D-3", "added after the snapshot"))
        assert store.diff(snap_id="snap-alpha") != [], "positive control: the mutation must be visible as a diff"

        with restoring(str(ws), batch_id="restore:snap-alpha", block_ids=("D-1", "D-3")):
            store.restore(snap_id="snap-alpha")

        got = store.get_by_id("D-1")
        assert got is not None and got["Statement"] == "before"
        assert store.get_by_id("D-2") is not None
        assert store.get_by_id("D-3") is None
        assert store.diff(snap_id="snap-alpha") == []

    def test_path_addressed_snapshot_is_unchanged(self, tmp_path: Path, admitted: None) -> None:
        """Regression control for the additive claim: the old surface still
        writes where it always did and still round-trips."""
        ws = tmp_path / "ws"
        (ws / "decisions").mkdir(parents=True)
        store = MarkdownBlockStore(str(ws))
        store.write_block(_md_block("D-9", "original"))
        snap_dir = tmp_path / "snap-legacy"
        store.snapshot(str(snap_dir))
        assert (snap_dir / "MANIFEST.json").is_file()

        store.write_block(_md_block("D-9", "mutated"))
        with restoring(str(ws), batch_id="restore:legacy", block_ids=("D-9",)):
            store.restore(str(snap_dir))
        got = store.get_by_id("D-9")
        assert got is not None and got["Statement"] == "original"

    def test_disagreeing_pair_is_refused_before_anything_is_written(self, tmp_path: Path, admitted: None) -> None:
        ws = tmp_path / "ws"
        (ws / "decisions").mkdir(parents=True)
        store = MarkdownBlockStore(str(ws))
        with pytest.raises(ValueError, match="disagrees"):
            store.snapshot(str(tmp_path / "snap-x"), snap_id="snap-y")
        assert not (tmp_path / "snap-x").exists(), "a refused snapshot created its directory anyway"

    def test_an_ungated_id_addressed_restore_is_refused(self, tmp_path: Path, admitted: None) -> None:
        """Positive control for the new addressing mode: it did not open a
        second, unadmitted door into ``restore``. Refused under the very
        receipt the rest of this class runs under — the proposal-scoped one,
        which is ambient authority over every id."""
        ws = tmp_path
        (ws / "decisions").mkdir()
        store = MarkdownBlockStore(str(ws))
        store.write_block(_md_block("D-1", "before"))
        store.snapshot(snap_id="snap-gated")
        store.write_block(_md_block("D-1", "after"))

        with pytest.raises(UngatedRestoreError):
            store.restore(snap_id="snap-gated")

        got = store.get_by_id("D-1")
        assert got is not None, "positive control failed — the block is not there to be reverted"
        assert got["Statement"] == "after", "the refused restore reverted the corpus anyway"


# ─── Postgres backend: an id needs no location at all ─────────────────────────

psycopg = pytest.importorskip("psycopg", reason="psycopg not installed; skipping Postgres tests")

from mind_mem.block_store_postgres import PostgresBlockStore  # noqa: E402


def _pg_block(bid: str, statement: str) -> dict:
    return {
        "_id": bid,
        "_source_file": "decisions/DECISIONS.md",
        "Statement": statement,
        "Status": "active",
        "Date": "2026-09-04",
    }


@pytest.fixture
def pg_store(tmp_path: Path) -> Iterator[PostgresBlockStore]:
    dsn = os.environ.get(_DSN_ENV)
    if not dsn:
        pytest.skip(f"{_DSN_ENV} not set — no live Postgres available")
    schema = f"mm_snapid_{uuid.uuid4().hex[:10]}"
    store = PostgresBlockStore(dsn=dsn, schema=schema, workspace=str(tmp_path))
    store._ensure_schema()
    try:
        yield store
    finally:
        from psycopg import sql

        with psycopg.connect(dsn, autocommit=True) as c:
            c.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))
        store.close()


def _snapshot_row_ids(store: PostgresBlockStore, snap_id: str) -> set[str]:
    """The ids the snapshot actually holds, read straight out of the rows.

    Not from the returned manifest and not from the live table: the claim
    is that the snapshot CONTAINS the blocks, and a manifest is a
    description of a snapshot, not the snapshot.
    """
    pool = store._get_pool()
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(f'SELECT block_id FROM "{store._schema}".snapshot_blocks WHERE snap_id = %s', (snap_id,))  # noqa: S608 — schema validated by _validate_schema_name
        return {str(r[0]) for r in cur.fetchall()}


def _stored_manifest(store: PostgresBlockStore, snap_id: str) -> dict:
    pool = store._get_pool()
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(f'SELECT manifest FROM "{store._schema}".snapshots WHERE snap_id = %s', (snap_id,))  # noqa: S608 — schema validated by _validate_schema_name
        row = cur.fetchone()
    assert row is not None, f"no manifest row for snap_id={snap_id!r}"
    return json.loads(row[0]) if isinstance(row[0], str) else dict(row[0])


class TestPostgresSnapId:
    def test_snapshot_by_id_touches_no_filesystem_and_still_holds_the_blocks(
        self, pg_store: PostgresBlockStore, tmp_path: Path, admitted: None
    ) -> None:
        """The defect, closed: a snapshot with no directory anywhere.

        The trap this avoids is a test that asserts "it returned a manifest"
        — true of a snapshot that captured nothing. Both halves are asserted:
        the workspace stays empty AND the exact seeded ids are readable back
        out of ``snapshot_blocks``.
        """
        pg_store.write_block(_pg_block("D-100", "alpha"))
        pg_store.write_block(_pg_block("D-101", "beta"))
        before = sorted(p.name for p in tmp_path.iterdir())

        manifest = pg_store.snapshot(snap_id="snap-nofs")

        assert sorted(p.name for p in tmp_path.iterdir()) == before, "an id-addressed snapshot wrote to the filesystem"
        assert manifest["snap_id"] == "snap-nofs"
        assert manifest["files"] == ["decisions/DECISIONS.md"]
        assert _snapshot_row_ids(pg_store, "snap-nofs") == {"D-100", "D-101"}, "the snapshot does not contain the seeded ids"
        # The manifest is the record, and it lives beside the rows.
        assert _stored_manifest(pg_store, "snap-nofs")["files"] == ["decisions/DECISIONS.md"]

    def test_id_addressed_snapshot_restore_round_trip(self, pg_store: PostgresBlockStore, tmp_path: Path, admitted: None) -> None:
        """The whole round trip with no path in sight: seed, snapshot by id,
        mutate in three ways, restore by id, prove the exact pre-snapshot
        state is back."""
        pg_store.write_block(_pg_block("D-200", "kept"))
        pg_store.write_block(_pg_block("D-201", "will be edited"))
        pg_store.write_block(_pg_block("D-202", "will be deleted"))

        pg_store.snapshot(snap_id="snap-round")
        assert _snapshot_row_ids(pg_store, "snap-round") == {"D-200", "D-201", "D-202"}

        pg_store.write_block(_pg_block("D-201", "edited"))
        # A delete needs its own scope: the ``admitted`` fixture's receipt
        # authorises writes, and the seam refuses to transfer it.
        from mind_mem.governance_gate import get_gate

        gate = get_gate(str(tmp_path))
        assert gate is not None
        with gate.admit_delete("D-202", rationale="snapshot round-trip test", actor="pytest"):
            assert pg_store.delete_block("D-202") is True
        pg_store.write_block(_pg_block("D-203", "born after the snapshot"))
        assert pg_store.diff(snap_id="snap-round") != [], "positive control: the mutations must be visible as a diff"

        with restoring(str(tmp_path), batch_id="restore:snap-round", block_ids=("D-201", "D-202", "D-203")):
            pg_store.restore(snap_id="snap-round")

        edited = pg_store.get_by_id("D-201")
        assert edited is not None and edited["Statement"] == "will be edited"
        deleted = pg_store.get_by_id("D-202")
        assert deleted is not None and deleted["Statement"] == "will be deleted", "restore did not reinstate the deleted block"
        assert pg_store.get_by_id("D-203") is None, "restore did not withdraw the post-snapshot block"
        assert pg_store.get_by_id("D-200") is not None
        # ``_source_file`` survives an id-addressed round trip too.
        assert edited["_source_file"] == "decisions/DECISIONS.md"
        assert pg_store.diff(snap_id="snap-round") == []

    def test_path_addressed_snapshot_still_exports_its_manifest(self, pg_store: PostgresBlockStore, tmp_path: Path, admitted: None) -> None:
        """Regression control: the existing caller is untouched — same
        identity from the basename, same ``MANIFEST.json`` on disk, and the
        rows are filed under that same id."""
        pg_store.write_block(_pg_block("D-300", "legacy"))
        snap_dir = tmp_path / "snap-legacy"
        manifest = pg_store.snapshot(str(snap_dir))

        assert manifest["version"] == 2
        exported = json.loads((snap_dir / "MANIFEST.json").read_text(encoding="utf-8"))
        assert exported["files"] == manifest["files"]
        assert _snapshot_row_ids(pg_store, "snap-legacy") == {"D-300"}

        with restoring(str(tmp_path), batch_id="restore:snap-legacy", block_ids=("D-300",)):
            pg_store.restore(str(snap_dir))
        assert pg_store.get_by_id("D-300") is not None

    def test_a_snapshot_taken_by_path_is_restorable_by_id(self, pg_store: PostgresBlockStore, tmp_path: Path, admitted: None) -> None:
        """The two addressing modes name the same snapshot — which is what
        makes the migration to id-addressing safe on an existing corpus."""
        pg_store.write_block(_pg_block("D-400", "original"))
        pg_store.snapshot(str(tmp_path / "snap-bridge"))
        pg_store.write_block(_pg_block("D-400", "mutated"))

        with restoring(str(tmp_path), batch_id="restore:snap-bridge", block_ids=("D-400",)):
            pg_store.restore(snap_id="snap-bridge")
        got = pg_store.get_by_id("D-400")
        assert got is not None and got["Statement"] == "original"

    def test_export_directory_and_identity_must_agree(self, pg_store: PostgresBlockStore, tmp_path: Path, admitted: None) -> None:
        with pytest.raises(ValueError, match="disagrees"):
            pg_store.snapshot(str(tmp_path / "dir-name"), snap_id="row-name")
        assert not (tmp_path / "dir-name").exists()

    @pytest.mark.parametrize("bad", ["../../etc", "a/b", ".."])
    def test_a_traversing_id_never_reaches_the_database(self, pg_store: PostgresBlockStore, bad: str, admitted: None) -> None:
        with pytest.raises(ValueError):
            pg_store.snapshot(snap_id=bad)

    def test_an_ungated_id_addressed_restore_is_refused(self, pg_store: PostgresBlockStore, tmp_path: Path, admitted: None) -> None:
        """Positive control: ``snap_id`` did not open a second, unadmitted
        door into the most destructive operation the product has. Refused
        under the proposal-scoped receipt, and the corpus is re-read
        afterwards — a check that ran after the transaction would raise and
        still have reverted everything."""
        pg_store.write_block(_pg_block("D-500", "before"))
        pg_store.snapshot(snap_id="snap-gated")
        pg_store.write_block(_pg_block("D-500", "after"))

        with pytest.raises(UngatedRestoreError, match="proposal-scoped"):
            pg_store.restore(snap_id="snap-gated")

        got = pg_store.get_by_id("D-500")
        assert got is not None, "positive control failed — the block is not there to be reverted"
        assert got["Statement"] == "after", "the refused restore reverted the corpus anyway"

    def test_restoring_an_unknown_id_changes_nothing(self, pg_store: PostgresBlockStore, tmp_path: Path, admitted: None) -> None:
        """A missing snapshot must fail before the live table is cleared."""
        pg_store.write_block(_pg_block("D-600", "still here"))
        from mind_mem.block_store_postgres import BlockStoreError

        with restoring(str(tmp_path), batch_id="restore:absent", block_ids=("D-600",)):
            with pytest.raises(BlockStoreError, match="not found"):
                pg_store.restore(snap_id="snap-never-taken")
        assert pg_store.get_by_id("D-600") is not None, "a failed restore emptied the live table"


# ─── Sharded cluster: the id has to carry the shard index ─────────────────────


class _RecordingShard:
    """The minimum of a ``BlockStore`` the sharded snapshot surface calls."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.snapshotted: list[tuple[str | None, str | None]] = []
        self.restored: list[tuple[str | None, str | None]] = []

    def snapshot(
        self,
        snap_dir: str | None = None,
        *,
        snap_id: str | None = None,
        files_touched: list[str] | None = None,
    ) -> dict:
        self.snapshotted.append((snap_dir, snap_id))
        return {"shard": self.name, "snap_id": snap_id}

    def restore(self, snap_dir: str | None = None, *, snap_id: str | None = None) -> None:
        self.restored.append((snap_dir, snap_id))

    def diff(self, snap_dir: str | None = None, *, snap_id: str | None = None) -> list[str]:
        return []


def _sharded_cluster(n: int = 3):
    from mind_mem.storage.sharded_pg import ShardConfig, ShardedPostgresBlockStore, ShardRouter

    router = ShardRouter(shards=[ShardConfig(index=i, dsn=f"pg://shard{i}") for i in range(n)])
    stores = {i: _RecordingShard(f"s{i}") for i in range(n)}
    return ShardedPostgresBlockStore(router, stores), stores


class TestShardedSnapId:
    def test_each_shard_gets_its_own_id_and_no_directory(self, tmp_path: Path) -> None:
        """The shards do not share a ``snapshots`` table, so a bare
        ``snap_id`` would name a *different* set of rows in each of them —
        the index has to be part of the id, exactly as it is part of the
        directory name on the path-addressed side."""
        store, stores = _sharded_cluster()
        manifest = store.snapshot(snap_id="cluster-1")

        assert manifest["ok"] is True
        assert manifest["snap_id"] == "cluster-1"
        assert [s.snapshotted for s in stores.values()] == [
            [(None, "cluster-1-shard-00")],
            [(None, "cluster-1-shard-01")],
            [(None, "cluster-1-shard-02")],
        ]
        assert list(tmp_path.iterdir()) == [], "an id-addressed cluster snapshot wrote to the filesystem"

    def test_id_addressed_restore_reaches_every_shard(self, tmp_path: Path, admitted: None) -> None:
        store, stores = _sharded_cluster()
        with restoring(str(tmp_path), batch_id="restore:cluster-1", block_ids=("D-1",)):
            store.restore(snap_id="cluster-1")
        assert [s.restored for s in stores.values()] == [
            [(None, "cluster-1-shard-00")],
            [(None, "cluster-1-shard-01")],
            [(None, "cluster-1-shard-02")],
        ]

    def test_an_ungated_id_addressed_cluster_restore_is_refused(self, tmp_path: Path, admitted: None) -> None:
        """Positive control: the fan-out is still behind the RESTORE scope,
        and refused *before* any shard is reached."""
        store, stores = _sharded_cluster()
        with pytest.raises(UngatedRestoreError):
            store.restore(snap_id="cluster-1")
        assert all(s.restored == [] for s in stores.values()), "a refused restore still reached a shard"

    def test_path_addressed_cluster_snapshot_keeps_its_layout(self, tmp_path: Path) -> None:
        """Regression control: ``restore`` locates shards by the
        ``<snap_dir>/shard-NN`` layout, so the directory-addressed path must
        keep producing exactly that."""
        store, stores = _sharded_cluster()
        store.snapshot(str(tmp_path / "snap-cluster"))
        assert [s.snapshotted for s in stores.values()] == [
            [(str(tmp_path / "snap-cluster" / "shard-00"), None)],
            [(str(tmp_path / "snap-cluster" / "shard-01"), None)],
            [(str(tmp_path / "snap-cluster" / "shard-02"), None)],
        ]
        for idx in range(3):
            assert (tmp_path / "snap-cluster" / f"shard-{idx:02d}").is_dir()
