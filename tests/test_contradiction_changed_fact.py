"""A changed fact (Tuesday -> Thursday) is flagged as a contradiction.

Reproduction (internal check 2026-10-08): a correction moving the Ilona call
from Tuesday to Thursday was not flagged against the committed Tuesday block.
Only the Ilona block may be named: the investor call and the contractor call
share the day and the time but are different meetings.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from mind_mem.contradiction_detector import _classify_conflict, check_proposal_contradictions

SEED = """# DECISIONS

[D-20261001-001]
Date: 2026-10-01
Status: active
Scope: project:azariel
Statement: Azariel's weekly planning call with Ilona is on Tuesday at 10am Arizona time.
Rationale: Curated from history, session 12.
Supersedes: none
Tags: schedule, azariel
Sources: export session 12

[D-20261001-002]
Date: 2026-10-01
Status: active
Scope: project:azariel
Statement: Azariel's monthly investor review call is on Tuesday at 2pm Arizona time.
Rationale: Curated from history, session 19.
Supersedes: none
Tags: schedule, azariel
Sources: export session 19

[D-20261001-003]
Date: 2026-10-01
Status: active
Scope: project:azariel
Statement: Azariel's weekly planning call with the engineering contractor is on Tuesday at 10am Pacific time.
Rationale: Curated from history, session 23.
Supersedes: none
Tags: schedule, azariel
Sources: export session 23
"""

CORRECTION = "Azariel's weekly planning call with Ilona is on Thursday at 10am Arizona time."


@pytest.fixture()
def ws(tmp_path: Path) -> str:
    os.makedirs(tmp_path / "decisions")
    (tmp_path / "decisions" / "DECISIONS.md").write_text(SEED, encoding="utf-8")
    return str(tmp_path)


def test_changed_day_is_a_contradiction():
    old = "Azariel's weekly planning call with Ilona is on Tuesday at 10am Arizona time."
    assert _classify_conflict(CORRECTION, old, 0.95) == "contradiction"


def test_changed_time_is_a_contradiction():
    old = "Weekly planning call with Ilona on Tuesday at 10am Arizona time."
    new = "Weekly planning call with Ilona on Tuesday at 11am Arizona time."
    assert _classify_conflict(new, old, 0.9) == "contradiction"


def test_different_person_is_not_a_contradiction():
    other = "Azariel's weekly planning call with the engineering contractor is on Tuesday at 10am Pacific time."
    assert _classify_conflict(CORRECTION, other, 0.8) != "contradiction"


def test_same_fact_reworded_is_not_a_contradiction():
    same = "Weekly planning call with Ilona: Thursday, 10am Arizona time."
    assert _classify_conflict(CORRECTION, same, 0.8) != "contradiction"


def test_correction_flags_only_the_ilona_block(ws):
    report = check_proposal_contradictions(ws, {"Statement": CORRECTION})
    flagged = sorted(c["block_id"] for c in report["conflicts"] if c["conflict_type"] == "contradiction")
    assert flagged == ["D-20261001-001"]
    assert report["has_contradictions"] is True


@pytest.fixture()
def mcp_ws(tmp_path: Path, monkeypatch) -> str:
    from mind_mem.init_workspace import init

    ws = str(tmp_path / "mcpws")
    os.makedirs(ws, exist_ok=True)
    init(ws)
    with open(os.path.join(ws, "decisions", "DECISIONS.md"), "a", encoding="utf-8") as handle:
        handle.write("\n" + SEED.split("\n", 2)[2])
    monkeypatch.setenv("MIND_MEM_SCOPE", "admin")
    return ws


def _propose(ws: str, statement: str) -> dict:
    import json

    import mind_mem.mcp.tools.governance as gov
    from mind_mem.mcp.infra.workspace import use_workspace

    with use_workspace(ws):
        return json.loads(gov.propose_update(block_type="decision", statement=statement, rationale="Correction from Ilona on 2026-10-07."))


def test_propose_update_reports_the_contradicted_block(mcp_ws):
    envelope = _propose(mcp_ws, CORRECTION)
    assert envelope["status"] == "proposed"
    assert [c["block_id"] for c in envelope["contradicts"]] == ["D-20261001-001"]


def test_propose_update_without_conflict_keeps_its_shape(mcp_ws):
    envelope = _propose(mcp_ws, "Azariel prefers written agendas sent a day before every call.")
    assert envelope["status"] == "proposed"
    assert "contradicts" not in envelope
