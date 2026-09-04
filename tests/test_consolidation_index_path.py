"""Regression: consolidation tools must read the index the product writes.

``plan_consolidation``, ``propagate_staleness`` and ``project_profile`` each
built their own literal index path (a ``.sqlite_index/index.db`` directory
that no writer in the tree ever creates) instead of deriving it from
:mod:`mind_mem.sqlite_index`, the module that owns ``DB_REL_PATH`` and is the
only writer of that database. ``os.path.isfile`` was therefore always False on
a real workspace: all three tools reported an empty corpus and still answered
``success``, so the archive/forget planner was blind to the whole corpus and
staleness never left its seed set.

These tests deliberately never spell the index location themselves. The
fixture writes through ``sqlite_index._connect`` / ``_init_schema`` — the
writer's own path resolution and the writer's own schema — so any reader that
resolves a different directory, or reads columns the real schema does not
have, fails here.

The same discipline now covers the *telemetry* the planner scores on. The
fixture used to hand-write ``block_meta`` rows with ``importance = 0.1`` into
the recall index — a file the telemetry writer does not write, holding a
value its ``[0.8, 1.5]`` clamp cannot produce. Both halves were impossible,
so the test passed while the shipped planner returned an empty plan for every
corpus. Telemetry here goes in through ``BlockMetadataManager.record_access``,
the only writer of it in the tree.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from mind_mem.block_metadata import BlockMetadataManager, block_meta_db_path
from mind_mem.mcp.infra.workspace import use_workspace
from mind_mem.sqlite_index import _connect, _db_path, _init_schema

#: Never accessed, created long ago — the block that must be marked.
BLOCK_A = "DEC-20200101-001"
#: Recorded by the real writer just now — must survive on importance alone.
BLOCK_B = "DEC-20200101-002"
#: Never accessed but created today — must survive on age alone.
BLOCK_C = "DEC-20200101-003"

OLD_DATE = "2020-01-01"


@pytest.fixture()
def indexed_workspace(tmp_path):
    """A workspace whose index AND telemetry were written by their own writers."""
    ws = tmp_path / "ws"
    (ws / "decisions").mkdir(parents=True)

    today = datetime.now(timezone.utc).date().isoformat()
    conn = _connect(str(ws))  # the writer decides where the DB lives
    try:
        _init_schema(conn)
        for bid, statement, date in (
            (BLOCK_A, "First decision", OLD_DATE),
            (BLOCK_B, "Second decision", OLD_DATE),
            (BLOCK_C, "Third decision", today),
        ):
            conn.execute(
                "INSERT INTO blocks (id, type, file, line, status, date, json_blob) "
                "VALUES (?, 'decision', 'decisions/DECISIONS.md', 1, 'active', ?, ?)",
                (bid, date, json.dumps({"Statement": statement, "_id": bid})),
            )
        conn.execute("INSERT INTO xref_edges (src, dst) VALUES (?, ?)", (BLOCK_A, BLOCK_B))
        conn.commit()
    finally:
        conn.close()

    # Telemetry through the real writer, at the path the real writer picks.
    # The instant is passed rather than read: ``record_access`` never invents
    # one, because the recall path that calls it is contractually clock-free.
    mgr = BlockMetadataManager(block_meta_db_path(str(ws)))
    try:
        mgr.record_access([BLOCK_B], query="second", now=datetime.now(timezone.utc))
    finally:
        mgr.close()
    return ws


def test_index_path_comes_from_the_writer(indexed_workspace) -> None:
    """The tools' path helper resolves to the writer's own location."""
    from mind_mem.mcp.tools.consolidation import _index_db_path

    resolved = _index_db_path(str(indexed_workspace))
    assert resolved == _db_path(str(indexed_workspace))
    assert (indexed_workspace / ".mind-mem-index" / "recall.db").is_file()


def test_plan_consolidation_emits_the_block_that_should_be_marked(indexed_workspace) -> None:
    """POSITIVE CONTROL: a block the real writer's scale calls low-value is emitted.

    ``BLOCK_A`` was never accessed and was created in 2020. Under the shipped
    thresholds it must appear in ``mark``. Before the fix this list was empty
    for every corpus of every size, and an empty plan reads as "nothing needs
    forgetting" — the false green this test exists to make impossible.
    """
    from mind_mem.mcp.tools.consolidation import plan_consolidation

    with use_workspace(str(indexed_workspace)):
        payload = json.loads(plan_consolidation())

    plan = payload["plan"]
    assert BLOCK_A in plan["mark"], plan
    # BLOCK_C is the same telemetry (none at all) with a recent creation date.
    # If ``created_at`` were dropped again, staleness would fall through to
    # True and this block would be marked too.
    assert BLOCK_C not in plan["mark"], plan


def test_recorded_access_alone_keeps_a_block_out_of_the_plan(indexed_workspace) -> None:
    """The scale, isolated: ``stale_days=0`` makes every block stale.

    With staleness satisfied for everything, the only thing separating
    ``BLOCK_A`` from ``BLOCK_B`` is the importance the real writer's telemetry
    produces. If the planner reads a store the writer does not write — the
    defect this file is named for — ``BLOCK_B`` looks untouched and is marked.
    """
    from mind_mem.mcp.tools.consolidation import plan_consolidation

    with use_workspace(str(indexed_workspace)):
        payload = json.loads(plan_consolidation(stale_days=0))

    plan = payload["plan"]
    assert BLOCK_A in plan["mark"], plan
    assert BLOCK_C in plan["mark"], plan  # positive control: age is what spared it before
    assert BLOCK_B not in plan["mark"], plan


def test_plan_is_a_dry_run_and_writes_nothing(indexed_workspace) -> None:
    """A preview must not create the telemetry store it reads."""
    from mind_mem.mcp.tools.consolidation import plan_consolidation

    ws = indexed_workspace.parent / "unwritten"
    (ws / "decisions").mkdir(parents=True)
    conn = _connect(str(ws))
    try:
        _init_schema(conn)
        conn.execute(
            "INSERT INTO blocks (id, type, file, line, status, date, json_blob) "
            "VALUES (?, 'decision', 'decisions/DECISIONS.md', 1, 'active', ?, '{}')",
            (BLOCK_A, OLD_DATE),
        )
        conn.commit()
    finally:
        conn.close()

    with use_workspace(str(ws)):
        plan = json.loads(plan_consolidation())["plan"]

    assert BLOCK_A in plan["mark"], "positive control: the planner really ran over this corpus"
    assert not (ws / ".mind-mem").exists()


def test_maturity_gate_reads_the_indexed_frontmatter(indexed_workspace) -> None:
    from mind_mem.mcp.tools.consolidation import plan_consolidation

    with use_workspace(str(indexed_workspace)):
        payload = json.loads(plan_consolidation(maturity_gate=True, min_maturity=0.5))

    gate = payload["maturity_gate"]
    # Every indexed block reaches the gate; an unread index leaves both lists empty.
    assert sorted([*gate["admitted"], *gate["held"]]) == [BLOCK_A, BLOCK_B, BLOCK_C]


def test_propagate_staleness_walks_the_indexed_xrefs(indexed_workspace) -> None:
    from mind_mem.mcp.tools.consolidation import propagate_staleness

    with use_workspace(str(indexed_workspace)):
        payload = json.loads(propagate_staleness(BLOCK_A, max_hops=2))

    # Without the xref edge the seed is the only scored block.
    assert BLOCK_B in payload["scores"], payload["scores"]
    assert payload["scores"][BLOCK_B] > 0.0


def test_project_profile_counts_the_indexed_blocks(indexed_workspace) -> None:
    from mind_mem.mcp.tools.consolidation import project_profile

    with use_workspace(str(indexed_workspace)):
        payload = json.loads(project_profile())

    assert payload["total_blocks"] == 3, payload
    assert payload["block_types"] == {"decision": 3}


def test_missing_index_still_degrades_to_an_empty_plan(tmp_path) -> None:
    """No index at all is a clean empty answer, not a crash."""
    from mind_mem.mcp.tools.consolidation import plan_consolidation, project_profile

    ws = tmp_path / "empty"
    (ws / "decisions").mkdir(parents=True)

    with use_workspace(str(ws)):
        plan = json.loads(plan_consolidation())["plan"]
        profile = json.loads(project_profile())

    assert plan["total"] == 0
    assert profile["total_blocks"] == 0
