"""The block-metadata store must exist, and one importance scale must reach the planner.

Three defects met here, all of the same shape — a value or a file that no
code path could ever produce, sitting behind a check that therefore always
answered the same way:

1. ``.mind-mem/`` was in no ``DIRS`` list and nothing created it, so
   :class:`~mind_mem.block_metadata.BlockMetadataManager` was never built on
   any workspace and ``record_access`` never ran once.
2. Three modules named three different files as "the block_meta store", so
   no reader ever saw a writer's row.
3. The planner compared the writer's ``[0.8, 1.5]`` importance against a
   ``[0, 1]`` threshold of 0.25 — unmeetable by construction, which reads as
   "nothing needs forgetting".

Every assertion below is paired with the control that makes it able to fail.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

from mind_mem.block_metadata import (
    IMPORTANCE_CEILING,
    IMPORTANCE_FLOOR,
    BlockMetadataManager,
    block_meta_db_path,
    compute_importance,
    keep_value,
)
from mind_mem.cognitive_forget import ConsolidationConfig

NOW = datetime(2026, 9, 3, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# 1. The store directory exists
# ---------------------------------------------------------------------------


def test_init_workspace_scaffolds_the_store_directory(tmp_path) -> None:
    from mind_mem.init_workspace import DIRS, init

    assert ".mind-mem" in DIRS, "the store directory must be scaffolded by init"

    ws = tmp_path / "ws"
    init(str(ws))
    assert (ws / ".mind-mem").is_dir()


def test_manager_creates_its_own_directory_on_first_use(tmp_path) -> None:
    """The installed base is never re-inited, so the owner must create it."""
    ws = tmp_path / "existing-workspace"
    ws.mkdir()
    assert not (ws / ".mind-mem").exists(), "positive control: the directory really is absent"

    mgr = BlockMetadataManager(block_meta_db_path(str(ws)))
    try:
        mgr.record_access(["D-20200101-001"])
    finally:
        mgr.close()

    assert (ws / ".mind-mem" / "block_meta.db").is_file()


def test_record_access_never_invents_an_instant(tmp_path) -> None:
    """The recall path is clock-free; a telemetry write must not break that.

    ``tests/_recall_clock_sentinel.py`` fails a recall that reads any date
    clock inside ``mind_mem``. ``record_access`` runs on that path, so the
    stamp is a parameter: omitted, the access is counted and
    ``last_accessed`` is left alone; supplied, it is written verbatim.
    """
    import sqlite3

    path = block_meta_db_path(str(tmp_path))
    mgr = BlockMetadataManager(path)
    try:
        mgr.record_access(["D-20200101-001"])
        mgr.record_access(["D-20200101-002"], now=NOW)
    finally:
        mgr.close()

    conn = sqlite3.connect(path)
    try:
        rows = dict(conn.execute("SELECT id, last_accessed FROM block_meta").fetchall())
        counts = dict(conn.execute("SELECT id, access_count FROM block_meta").fetchall())
    finally:
        conn.close()

    # Positive control: both rows exist and both accesses were counted, so
    # the NULL below is a deliberate absence and not a failed write.
    assert counts == {"D-20200101-001": 1, "D-20200101-002": 1}
    assert rows["D-20200101-001"] is None
    assert rows["D-20200101-002"] == NOW.isoformat()


def test_recorded_access_without_an_instant_still_moves_importance(tmp_path) -> None:
    """The count half of the telemetry works with no clock at all."""
    mgr = BlockMetadataManager(block_meta_db_path(str(tmp_path)))
    try:
        for _ in range(8):
            mgr.record_access(["D-20200101-001"])
        assert mgr.update_importance("D-20200101-001") > IMPORTANCE_FLOOR
    finally:
        mgr.close()


def test_unwritable_store_path_degrades_instead_of_raising(tmp_path) -> None:
    """Creating the directory must not turn a bad path into a new crash."""
    mgr = BlockMetadataManager("/nonexistent/path/block_meta.db")
    mgr.record_access(["D-20200101-001"])  # must not raise
    assert mgr.get_importance_boost("D-20200101-001") == 1.0


# ---------------------------------------------------------------------------
# 2. One store path, resolved from one place
# ---------------------------------------------------------------------------


def test_every_block_meta_consumer_resolves_the_same_file(tmp_path) -> None:
    """The MCP readers and the recall writer must name one file.

    ``_recall_core`` spells the path as a literal; this pins the resolver to
    that same literal so the two cannot drift apart again.
    """
    ws = str(tmp_path / "ws")
    assert block_meta_db_path(ws) == os.path.join(ws, ".mind-mem", "block_meta.db")


# ---------------------------------------------------------------------------
# 3. One importance scale, reachable at both ends
# ---------------------------------------------------------------------------


def test_the_writer_can_only_produce_values_on_the_declared_scale() -> None:
    """Positive control for the scale claim itself."""
    extremes = [
        compute_importance(access_count=0, last_accessed=None, connection_count=0, now=NOW),
        compute_importance(access_count=10**6, last_accessed=NOW.isoformat(), connection_count=50, now=NOW),
    ]
    for value in extremes:
        assert IMPORTANCE_FLOOR <= value <= IMPORTANCE_CEILING


def test_keep_value_spans_the_whole_unit_interval() -> None:
    """The conversion must be able to answer 0 and 1, not a narrow band."""
    assert keep_value(IMPORTANCE_FLOOR) == 0.0
    assert keep_value(IMPORTANCE_CEILING) == 1.0
    # Out-of-band inputs clamp rather than escaping BlockCognition's contract.
    assert keep_value(0.0) == 0.0
    assert keep_value(99.0) == 1.0


def test_default_mark_threshold_is_meetable_and_refusable() -> None:
    """The threshold must separate two blocks the writer can actually produce.

    This is the assertion the old code could not have passed: with
    importance read on the ``[0.8, 1.5]`` scale against a ``[0, 1]``
    threshold, *no* input is below 0.25.
    """
    threshold = ConsolidationConfig().importance_threshold

    never_read = keep_value(compute_importance(access_count=0, last_accessed=None, connection_count=0, now=NOW))
    just_read = keep_value(compute_importance(access_count=1, last_accessed=NOW.isoformat(), connection_count=0, now=NOW))
    long_ago = keep_value(
        compute_importance(
            access_count=1,
            last_accessed=(NOW - timedelta(days=90)).isoformat(),
            connection_count=0,
            now=NOW,
        )
    )

    assert never_read < threshold, never_read
    assert long_ago < threshold, long_ago
    assert just_read > threshold, just_read


def test_stored_importance_stays_the_rerank_multiplier(tmp_path) -> None:
    """Recall's boost must remain neutral for a block it merely recorded.

    ``record_access`` writes no importance, so the column default applies and
    the reranker multiplies by exactly 1.0 — the ordering an existing
    workspace already has. A change here re-ranks every user's recall.
    """
    mgr = BlockMetadataManager(block_meta_db_path(str(tmp_path)))
    try:
        mgr.record_access(["D-20200101-001"])
        assert mgr.get_importance_boost("D-20200101-001") == 1.0
        # Positive control: the row really exists, so this is not a
        # "no row, default 1.0" answer.
        assert mgr.get_co_occurring_blocks("D-20200101-001") == []
        updated = mgr.update_importance("D-20200101-001")
        assert updated != 1.0, "positive control: an explicit update does move the value"
        assert IMPORTANCE_FLOOR <= updated <= IMPORTANCE_CEILING
    finally:
        mgr.close()
