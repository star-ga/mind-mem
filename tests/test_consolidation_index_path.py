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
"""

from __future__ import annotations

import json

import pytest

from mind_mem.mcp.infra.workspace import use_workspace
from mind_mem.sqlite_index import _connect, _db_path, _init_schema

BLOCK_A = "DEC-20200101-001"
BLOCK_B = "DEC-20200101-002"


@pytest.fixture()
def indexed_workspace(tmp_path):
    """A workspace whose index was written by ``sqlite_index`` itself."""
    ws = tmp_path / "ws"
    (ws / "decisions").mkdir(parents=True)

    conn = _connect(str(ws))  # the writer decides where the DB lives
    try:
        _init_schema(conn)
        for bid, statement in ((BLOCK_A, "First decision"), (BLOCK_B, "Second decision")):
            conn.execute(
                "INSERT INTO blocks (id, type, file, line, status, date, json_blob) "
                "VALUES (?, 'decision', 'decisions/DECISIONS.md', 1, 'active', '2020-01-01', ?)",
                (bid, json.dumps({"Statement": statement, "_id": bid})),
            )
            conn.execute(
                "INSERT INTO block_meta (id, importance, access_count, last_accessed) VALUES (?, 0.1, 0, '2020-01-02T00:00:00Z')",
                (bid,),
            )
        conn.execute("INSERT INTO xref_edges (src, dst) VALUES (?, ?)", (BLOCK_A, BLOCK_B))
        conn.commit()
    finally:
        conn.close()
    return ws


def test_index_path_comes_from_the_writer(indexed_workspace) -> None:
    """The tools' path helper resolves to the writer's own location."""
    from mind_mem.mcp.tools.consolidation import _index_db_path

    resolved = _index_db_path(str(indexed_workspace))
    assert resolved == _db_path(str(indexed_workspace))
    assert (indexed_workspace / ".mind-mem-index" / "recall.db").is_file()


def test_plan_consolidation_sees_the_indexed_corpus(indexed_workspace) -> None:
    from mind_mem.mcp.tools.consolidation import plan_consolidation

    with use_workspace(str(indexed_workspace)):
        payload = json.loads(plan_consolidation())

    plan = payload["plan"]
    assert plan["total"] == 2, plan
    assert sorted(plan["mark"]) == [BLOCK_A, BLOCK_B]


def test_maturity_gate_reads_the_indexed_frontmatter(indexed_workspace) -> None:
    from mind_mem.mcp.tools.consolidation import plan_consolidation

    with use_workspace(str(indexed_workspace)):
        payload = json.loads(plan_consolidation(maturity_gate=True, min_maturity=0.5))

    gate = payload["maturity_gate"]
    # Every indexed block reaches the gate; an unread index leaves both lists empty.
    assert sorted([*gate["admitted"], *gate["held"]]) == [BLOCK_A, BLOCK_B]


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

    assert payload["total_blocks"] == 2, payload
    assert payload["block_types"] == {"decision": 2}


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
