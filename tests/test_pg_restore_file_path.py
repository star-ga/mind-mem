"""restore() must preserve each block's file_path (routing metadata).

Regression: snapshot stored block metadata with _-prefixed keys stripped,
so _source_file was lost; snapshot_blocks had no file_path column; and
restore() read COALESCE(metadata->>'_source_file','') => '' for every row.
A single snapshot/restore cycle therefore wiped file_path on every block.

From 5.0.2 ``restore`` is admitted at the store seam like ``write_block`` and
``delete_block``, so the restore below runs inside the RESTORE batch scope
``apply_engine.restore_snapshot`` opens (:func:`tests._restore_scope.restoring`).
The ``admitted`` fixture this test used to lean on is not enough and must not
be made enough: its receipt is proposal-scoped, which is ambient authority
over every id, and ``require_restore_admission`` refuses exactly that. The
routing-metadata claim is unchanged -- it is proved through the supported
door now -- and :func:`test_an_ungated_restore_is_still_refused` is the
positive control that the door is still shut.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Iterator

import pytest
from _restore_scope import restoring

psycopg = pytest.importorskip("psycopg")

from mind_mem.admission import UngatedRestoreError  # noqa: E402
from mind_mem.block_store_postgres import PostgresBlockStore  # noqa: E402

_DSN_ENV = "MIND_MEM_TEST_PG_DSN"


@pytest.fixture
def live_store(tmp_path: Path) -> Iterator[tuple[PostgresBlockStore, str]]:
    """A store in a throwaway schema, plus the snapshot dir to use with it.

    The store's workspace is ``tmp_path`` -- the same workspace the
    ``admitted`` fixture and :func:`restoring` key the gate on, so the
    admission and the store agree on one workspace the way
    ``apply_engine.restore_snapshot`` does. It also stops the test writing
    into a fixed ``/tmp`` path shared by every concurrent run.
    """
    dsn = os.environ.get(_DSN_ENV)
    if not dsn:
        pytest.skip(f"{_DSN_ENV} not set -- no live Postgres available")
    schema = f"mm_rfp_{uuid.uuid4().hex[:10]}"
    store = PostgresBlockStore(dsn=dsn, schema=schema, workspace=str(tmp_path))
    store._ensure_schema()
    try:
        yield store, str(tmp_path / f"{schema}-snap")
    finally:
        from psycopg import sql

        with psycopg.connect(dsn, autocommit=True) as c:
            c.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))
        store.close()


def _seed(store: PostgresBlockStore, statement: str) -> None:
    store.write_block(
        {
            "_id": "D-1",
            "_source_file": "decisions/DECISIONS.md",
            "Statement": statement,
            "Status": "active",
        }
    )


def test_restore_preserves_file_path(live_store: tuple[PostgresBlockStore, str], admitted: None, tmp_path: Path) -> None:
    store, snap = live_store
    _seed(store, "keep this path")
    store.snapshot(snap)
    # Mutate then restore.
    _seed(store, "changed")
    with restoring(str(tmp_path), batch_id=f"restore:{os.path.basename(snap)}", block_ids=("D-1",)):
        store.restore(snap)

    block = store.get_by_id("D-1")
    assert block is not None
    assert block["Statement"] == "keep this path"
    # The bug: file_path came back as '' after restore.
    assert block["_source_file"] == "decisions/DECISIONS.md"


def test_an_ungated_restore_is_still_refused(live_store: tuple[PostgresBlockStore, str], admitted: None) -> None:
    """Positive control for the scope above: the seam still refuses.

    Without it, :func:`restoring` could quietly become a no-op, or the
    ``require_restore_admission`` call could be deleted from the store, and
    the test above would go on passing while ``store.restore()`` was once
    again callable from anywhere. The refusal is asserted on the same store
    and the same snapshot, under the very receipt this file used to rely on
    -- the proposal-scoped one the ``admitted`` fixture opens.

    The corpus is then re-read: an admission check that ran *after* the
    transaction would raise and still have reverted the workspace, which is
    the failure mode the ordering in ``PostgresBlockStore.restore`` exists
    to prevent, so "it raised" is not on its own evidence that nothing moved.
    """
    store, snap = live_store
    _seed(store, "keep this path")
    store.snapshot(snap)
    _seed(store, "changed")

    with pytest.raises(UngatedRestoreError, match="proposal-scoped"):
        store.restore(snap)

    block = store.get_by_id("D-1")
    assert block is not None, "positive control failed -- the seed block is not there to be reverted"
    assert block["Statement"] == "changed", "the refused restore reverted the corpus anyway"
