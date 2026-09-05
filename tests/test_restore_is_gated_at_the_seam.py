# Copyright 2026 STARGA, Inc.
"""Every ``BlockStore`` shape refuses an ungated ``restore`` at its own seam.

``restore`` is the third way content leaves a workspace, beside ``write_block``
and ``delete_block``. Measured on ``5.0.1`` (HEAD ``e16ef2c``), with the write
and delete gates already closed at that same seam::

    --- DEFECT: store.restore() with NO scope open ---
    restore returned normally
    block D-20260902-002 readable before/after: True False
    (evidence, hash_chain) before/after: (2, 2) (2, 2)
    --- POSITIVE CONTROL: the SAME seam, delete and write, no scope ---
    ungated delete_block -> raised UngatedDeleteError
    ungated write_block  -> raised UngatedWriteError

A governed block died and neither ledger moved.

``tests/test_governed_restore_seam.py`` covers the Markdown seam in depth and
pins ``apply_engine.restore_snapshot`` as the only opener. **This file covers
the other axis: the store-shape matrix.** A ``BlockStore`` is whatever
``storage.get_block_store`` returns — Markdown, encrypted, Postgres, the
replica adapter, the shard fan-out — and the invariant is worth nothing if it
holds for one of them.

The two adapters used to *inherit* their refusal by handing the call to a store
that enforces. Measured, that is not the same property. Over an inner store
that does not check::

    ShardedPostgresBlockStore.restore   : returned normally  <-- UNGATED RESTORE SUCCEEDED
    ReplicatedPostgresBlockStore.restore: returned normally  <-- UNGATED RESTORE SUCCEEDED
    EncryptedBlockStore.restore         : raised UngatedRestoreError   (inner saw 0 calls)

and the shard fan-out additionally answered *"is this my snapshot?"* — a
``BlockStoreError`` naming the missing shard directory — to a caller it had not
authorised. Both now call ``require_restore_admission`` first and still delegate
the work: authorisation at the door, one scope, one chain record.
"""

from __future__ import annotations

import importlib
import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Iterator

import pytest
from _restore_scope import restoring
from _write_path_scan import RESTORE_ENFORCEMENT_EXEMPT, iter_source_files, scan_restore_defs

from mind_mem.admission import UngatedRestoreError
from mind_mem.apply_engine import RESTORE_VERB, create_snapshot, restore_snapshot
from mind_mem.block_store import BlockStoreError, MarkdownBlockStore
from mind_mem.block_store_encrypted import EncryptedBlockStore
from mind_mem.block_store_postgres import PostgresBlockStore
from mind_mem.block_store_postgres_replica import ReplicatedPostgresBlockStore
from mind_mem.enums import IngestTier
from mind_mem.governance_gate import evict_gate, get_gate
from mind_mem.storage import get_block_store

#: ``sharded_pg`` is deliberately NOT imported by name here — it is resolved
#: per call by :func:`sharded_module`, for the reason that function documents.

CORPUS = ("decisions", "tasks", "entities", "intelligence", "memory", "summaries")

#: A DSN that resolves to a closed port. Every Postgres construction below is
#: lazy — no connection is opened until a query runs — so reaching the network
#: at all is itself a measurable failure of "the check came first".
DEAD_DSN = "postgresql://nobody@127.0.0.1:1/none"


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def workspace(tmp_path: Path) -> Iterator[str]:
    ws = tmp_path / "ws"
    for sub in CORPUS:
        (ws / sub).mkdir(parents=True, exist_ok=True)
    (ws / "mind-mem.json").write_text(json.dumps({"block_store": {"backend": "markdown"}}), encoding="utf-8")
    (ws / "decisions" / "DECISIONS.md").write_text("# Decisions\n\n", encoding="utf-8")
    (ws / "memory" / "intel-state.json").write_text("{}\n", encoding="utf-8")
    try:
        yield str(ws)
    finally:
        evict_gate(str(ws))


def write_governed_block(ws: str, block_id: str) -> None:
    gate = get_gate(ws)
    store = get_block_store(ws)
    with gate.admit_block("WRITE", block_id, "body", tier=IngestTier.EXTERNAL_INGEST):
        store.write_block({"_id": block_id, "Statement": "body", "Status": "quarantined", "Date": "2026-09-02"})


def ledger_counts(ws: str) -> tuple[int, int]:
    """``(evidence rows, hash-chain rows)`` — the pair the defect left unmoved."""
    ev = os.path.join(ws, "memory", "evidence_chain.jsonl")
    n_ev = 0
    if os.path.isfile(ev):
        with open(ev, encoding="utf-8") as fh:
            n_ev = sum(1 for line in fh if line.strip())
    db = os.path.join(ws, "memory", "hash_chain_v2.db")
    n_hc = 0
    if os.path.isfile(db):
        con = sqlite3.connect(db)
        try:
            n_hc = int(con.execute("SELECT COUNT(*) FROM hash_chain").fetchone()[0])
        finally:
            con.close()
    return n_ev, n_hc


def restore_rows(ws: str) -> list[dict[str, Any]]:
    path = os.path.join(ws, "memory", "evidence_chain.jsonl")
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    return [r for r in rows if (r.get("metadata") or {}).get("action_verb") == RESTORE_VERB]


@pytest.fixture
def live_snapshot(workspace: str) -> tuple[str, str]:
    """``(snap_dir, ws)`` where a restore WOULD withdraw ``D-20260902-002``.

    Without the second block, "the block is still readable" measures a restore
    that had nothing to do rather than a refusal.
    """
    write_governed_block(workspace, "D-20260902-001")
    snap_dir = create_snapshot(workspace, "20260902-190000", files_touched=None)
    write_governed_block(workspace, "D-20260902-002")
    assert get_block_store(workspace).get_by_id("D-20260902-002") is not None, "fixture did not land the withdrawable block"
    return snap_dir, workspace


class PermissiveInnerStore:
    """A ``BlockStore``-shaped inner store that enforces nothing.

    The instrument for the adapter half of this file. An adapter that owns its
    seam refuses before reaching this object; an adapter that only inherits one
    lets the call through — and ``restored`` counts which happened, so the
    difference is measured rather than argued.
    """

    def __init__(self) -> None:
        self.restored: list[str] = []

    # Addressed by directory, by ``snap_id``, or by both, like every real
    # ``BlockStore.restore``: an instrument that accepts only the directory
    # would raise a ``TypeError`` on the id-addressed path and read as a
    # refusal the adapter never made.
    def restore(self, snap_dir: str | None = None, *, snap_id: str | None = None) -> None:
        self.restored.append(snap_dir if snap_dir is not None else f"snap_id:{snap_id}")


def sharded_module() -> Any:
    """The ``mind_mem.storage.sharded_pg`` the import system resolves *now*.

    Not the one this file bound at collection time, because those are not
    always the same object. ``tests/test_mm_doctor_postgres_hint.py`` evicts
    every ``mind_mem.storage*`` key from ``sys.modules`` with a bare
    ``sys.modules.pop`` — not ``monkeypatch.delitem`` — so nothing puts them
    back. A later ``from mind_mem.storage import sharded_pg`` re-executes the
    module and returns a **second** module object; measured::

        module re-executed, B is A: False
        class the test file holds reads globals of A: True
        ...and NOT of B: False

    A twin that patches B while constructing A's class then neutralises
    nothing: the store reads A's globals, the real check runs, and a store
    that *is* correctly gated is reported as an ungated one. Resolving the
    store and the namespace the twin patches from this one call keeps them
    the same object however the rest of the suite has treated
    ``sys.modules`` — which is what "patch the module the code under test
    actually reads" means when the module can be re-imported underneath it.
    """
    return importlib.import_module("mind_mem.storage.sharded_pg")


def sharded_over(inner: Any) -> Any:
    """A one-shard fan-out over ``inner``, built from :func:`sharded_module`."""
    mod = sharded_module()
    return mod.ShardedPostgresBlockStore(mod.ShardRouter([mod.ShardConfig(index=0, dsn=DEAD_DSN)]), {0: inner})


def replica_over(inner: Any) -> ReplicatedPostgresBlockStore:
    store = ReplicatedPostgresBlockStore(DEAD_DSN, [DEAD_DSN])
    store._primary = inner  # the adapter's own seam is what is under test
    return store


def snap_dir_with_shard_layout(tmp_path: Path) -> str:
    """A directory the shard fan-out accepts, so its refusal is not about layout."""
    snap = tmp_path / "shard-snapshot"
    (snap / "shard-00").mkdir(parents=True, exist_ok=True)
    return str(snap)


# ---------------------------------------------------------------------------
# The matrix is closed by construction, not by this file remembering
# ---------------------------------------------------------------------------


class TestEveryRestoreImplementationOwnsItsSeam:
    def test_the_only_unenforced_restore_is_the_protocol_stub(self) -> None:
        """A new backend cannot ship an ungated restore without failing here."""
        defs = scan_restore_defs(iter_source_files())
        # Positive control: an empty scan satisfies any assertion about it.
        assert defs, "the scanner found no `def restore` in src/ — everything below is vacuous"
        assert any(enforces for *_rest, enforces in defs), "the scanner sees enforcement nowhere; its matcher is broken"

        unenforced = sorted((rel, qualname) for rel, qualname, _line, enforces in defs if not enforces)
        assert unenforced == [("src/mind_mem/block_store.py", "BlockStore.restore")], (
            f"a `def restore` in src/ does not call require_restore_admission: {unenforced}"
        )
        assert set(unenforced) == set(RESTORE_ENFORCEMENT_EXEMPT), "the scanner's exemption set and this file disagree"

    def test_all_five_store_shapes_enforce_by_name(self) -> None:
        """Named, not counted — the set-difference above passes on an exemption."""
        enforcing = {(rel, qualname) for rel, qualname, _line, enforces in scan_restore_defs(iter_source_files()) if enforces}
        for shape in (
            ("src/mind_mem/block_store.py", "MarkdownBlockStore.restore"),
            ("src/mind_mem/block_store_postgres.py", "PostgresBlockStore.restore"),
            ("src/mind_mem/block_store_encrypted.py", "EncryptedBlockStore.restore"),
            ("src/mind_mem/block_store_postgres_replica.py", "ReplicatedPostgresBlockStore.restore"),
            ("src/mind_mem/storage/sharded_pg.py", "ShardedPostgresBlockStore.restore"),
        ):
            assert shape in enforcing, f"{shape} does not call require_restore_admission"


# ---------------------------------------------------------------------------
# Behaviour: the block existed, the ungated restore raised, the block survived
# ---------------------------------------------------------------------------


class TestTheStoresOfRecordRefuseAnUngatedRestore:
    def test_markdown_refuses_and_the_block_and_the_ledgers_survive(self, live_snapshot: tuple[str, str]) -> None:
        """The reproduced defect, head-on, on the zero-config default."""
        snap_dir, ws = live_snapshot
        store = get_block_store(ws)
        # Positive control: the block this restore would withdraw is really there.
        assert store.get_by_id("D-20260902-002") is not None
        before = ledger_counts(ws)

        with pytest.raises(UngatedRestoreError, match="no governance admission is open"):
            store.restore(snap_dir)

        assert store.get_by_id("D-20260902-002") is not None, "the refused restore withdrew the block anyway"
        assert ledger_counts(ws) == before, "the refused restore moved a ledger"

    def test_encrypted_refuses_at_its_own_seam(self, live_snapshot: tuple[str, str], tmp_path: Path) -> None:
        snap_dir, ws = live_snapshot
        inner = PermissiveInnerStore()
        store = EncryptedBlockStore(ws, passphrase="not-a-real-secret", inner=inner)

        with pytest.raises(UngatedRestoreError):
            store.restore(snap_dir)

        assert inner.restored == [], "the wrapper forwarded an ungated restore to the inner store"

    def test_postgres_refuses_before_it_opens_a_connection(self, workspace: str) -> None:
        """Authorisation precedes I/O, so an ungated caller cannot even connect.

        The positive control is the second half: under a correct RESTORE scope
        the same call gets *past* the seam and fails on the dead DSN instead.
        Without it, "raises UngatedRestoreError" would also be satisfied by a
        store that refuses everything.
        """
        store = PostgresBlockStore(DEAD_DSN, workspace=workspace)

        with pytest.raises(UngatedRestoreError):
            store.restore("/snapshots/20260902-190000")

        with restoring(workspace, batch_id="restore:20260902-190000", block_ids=["D-20260902-001"]):
            with pytest.raises(Exception) as excinfo:  # noqa: PT011 — the DB error is the point
                store.restore("/snapshots/20260902-190000")
        assert not isinstance(excinfo.value, UngatedRestoreError), (
            "an admitted restore was still refused by the seam — the check is not what stopped the ungated one"
        )


class TestTheAdaptersRefuseAtTheirOwnSeam:
    """Inherited enforcement is a property of the inner store, not of the adapter."""

    def test_replica_refuses_over_a_non_enforcing_primary(self, tmp_path: Path) -> None:
        inner = PermissiveInnerStore()
        store = replica_over(inner)

        with pytest.raises(UngatedRestoreError):
            store.restore(str(tmp_path / "any-snapshot"))

        assert inner.restored == [], "the adapter forwarded an ungated restore to the primary"

    def test_sharded_refuses_over_non_enforcing_shards(self, tmp_path: Path) -> None:
        inner = PermissiveInnerStore()
        store = sharded_over(inner)

        with pytest.raises(UngatedRestoreError):
            store.restore(snap_dir_with_shard_layout(tmp_path))

        assert inner.restored == [], "the fan-out forwarded an ungated restore to a shard"

    def test_an_admitted_restore_still_reaches_the_inner_store(self, workspace: str, tmp_path: Path) -> None:
        """Positive control for both refusals above.

        A door that refuses everything is not a door. Under the scope the
        sanctioned opener mints, both adapters delegate exactly as before.
        """
        snap = snap_dir_with_shard_layout(tmp_path)
        replica_inner, sharded_inner = PermissiveInnerStore(), PermissiveInnerStore()

        with restoring(workspace, batch_id="restore:20260902-191000", block_ids=["D-20260902-001"]):
            replica_over(replica_inner).restore(snap)
            sharded_over(sharded_inner).restore(snap)

        assert replica_inner.restored == [snap], "the admitted restore never reached the primary"
        assert sharded_inner.restored == [os.path.join(snap, "shard-00")], "the admitted restore never reached the shard"

    def test_the_fan_out_no_longer_answers_existence_before_authorisation(self, workspace: str, tmp_path: Path) -> None:
        """An ungated caller must not learn whether a path is this cluster's snapshot.

        Before the seam check, the shard-directory pre-check ran first and
        raised ``BlockStoreError`` naming the missing shard. Both halves are
        asserted: the ungated caller gets the authorisation refusal, and the
        *admitted* caller still gets the existence refusal — so the pre-check
        was reordered, not removed.
        """
        no_shards = str(tmp_path / "not-a-cluster-snapshot")
        store = sharded_over(PermissiveInnerStore())

        with pytest.raises(UngatedRestoreError):
            store.restore(no_shards)

        with restoring(workspace, batch_id="restore:20260902-192000", block_ids=["D-20260902-001"]):
            with pytest.raises(BlockStoreError, match="refusing a partial restore"):
                store.restore(no_shards)


# ---------------------------------------------------------------------------
# The sanctioned door still works end to end
# ---------------------------------------------------------------------------


class TestAnAdmittedRestoreStillWorks:
    def test_the_sanctioned_restore_lands_and_records_one_row(self, live_snapshot: tuple[str, str]) -> None:
        snap_dir, ws = live_snapshot
        assert restore_rows(ws) == []

        restore_snapshot(ws, snap_dir)

        store = get_block_store(ws)
        assert store.get_by_id("D-20260902-002") is None, "the sanctioned restore did not withdraw the block"
        assert store.get_by_id("D-20260902-001") is not None, "the sanctioned restore did not reinstate the snapshot"
        rows = restore_rows(ws)
        assert len(rows) == 1, f"expected exactly one RESTORE row, got {len(rows)}"
        assert "D-20260902-002" in rows[0]["metadata"]["withdrawn_block_ids"]


# ---------------------------------------------------------------------------
# Mutation twins — a check never observed failing is not a check
# ---------------------------------------------------------------------------


class TestMutationTwin:
    """Break exactly the call this file asserts, and watch the defect return.

    One twin per module the lane touched. Each neutralises
    ``require_restore_admission`` *in that module's namespace only*, so a twin
    that goes green because some other module's check caught the call is not
    possible.
    """

    @staticmethod
    def _neutralise(monkeypatch: pytest.MonkeyPatch, module: Any) -> None:
        class _NoReceipt:
            entry_id = "mutation-twin"

        monkeypatch.setattr(module, "require_restore_admission", lambda snap: _NoReceipt())

    def test_markdown_twin(self, live_snapshot: tuple[str, str], monkeypatch: pytest.MonkeyPatch) -> None:
        from mind_mem import block_store as module

        snap_dir, ws = live_snapshot
        before = ledger_counts(ws)
        self._neutralise(monkeypatch, module)

        MarkdownBlockStore(ws).restore(snap_dir)

        assert get_block_store(ws).get_by_id("D-20260902-002") is None, (
            "with the check neutralised the ungated restore did NOT run — this file is not measuring that check"
        )
        assert ledger_counts(ws) == before, "an ungated restore moved a ledger"
        assert restore_rows(ws) == []

    def test_replica_twin(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from mind_mem import block_store_postgres_replica as module

        inner = PermissiveInnerStore()
        self._neutralise(monkeypatch, module)

        replica_over(inner).restore(str(tmp_path / "any-snapshot"))

        assert inner.restored, "the adapter's refusal does not come from the call this file asserts"

    def test_sharded_twin(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        module = sharded_module()

        inner = PermissiveInnerStore()
        snap = snap_dir_with_shard_layout(tmp_path)
        self._neutralise(monkeypatch, module)
        store = sharded_over(inner)
        # The twin's own precondition: the namespace just neutralised is the
        # one `store.restore` reads. Without it, a re-imported `sharded_pg`
        # (see `sharded_module`) turns "the check is load-bearing" into "some
        # other copy of the check ran", and the twin reads as a defect in a
        # store that is gated correctly.
        assert type(store).restore.__globals__ is module.__dict__, (
            "the twin patched a different module object than the store under test reads"
        )

        store.restore(snap)

        assert inner.restored == [os.path.join(snap, "shard-00")], "the fan-out's refusal does not come from the call this file asserts"
