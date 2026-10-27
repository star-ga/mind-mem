"""A correction proposed through MCP never lands as a second active truth.

Reproduction (architecture review F2/F17): ``propose_update`` reported the
Tuesday block as contradicted, but an apply of the same correction staged as
an ``append_block`` proposal saw "No conflicts" -- the proposal carries its
Statement inside the op patch, so the detector compared field labels instead
of the claim -- and appended a second active truth. No MCP path could express
the supersession either. These tests pin the loop closed end to end:

* ``propose_update(..., supersedes=<id>)`` stages a ``supersede_decision``;
  approving it retires the old block (``SupersededBy`` set) and recall returns
  the correction first.
* Propose time and apply time read the same Statement and agree.
* A superseded block surfaced by the BM25 leg is never a live contradiction.
* ``contradicts`` names the id to pass back as ``supersedes``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from mind_mem.apply_engine import compute_fingerprint, find_proposal
from mind_mem.block_parser import parse_file
from mind_mem.block_store import _render_block
from mind_mem.contradiction_detector import check_proposal_contradictions
from mind_mem.mm_cli import config_set

TUESDAY_ID = "D-20261001-001"
TUESDAY = "Azariel's weekly planning call with Ilona is on Tuesday at 10am Arizona time."
THURSDAY = "Azariel's weekly planning call with Ilona is on Thursday at 10am Arizona time."
RATIONALE = "Correction from Ilona on 2026-10-07."

SEED = f"""
[{TUESDAY_ID}]
Date: 2026-10-01
Status: active
Scope: project:azariel
Statement: {TUESDAY}
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
"""


@pytest.fixture()
def ws(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    from mind_mem.init_workspace import init

    root = str(tmp_path / "ws")
    os.makedirs(root, exist_ok=True)
    init(root)
    with open(os.path.join(root, "decisions", "DECISIONS.md"), "a", encoding="utf-8") as handle:
        handle.write(SEED)
    # Applies are off in a fresh workspace (detect_only); enable them through
    # the bound config the way an operator does.
    config_set(os.path.join(root, "mind-mem.json"), "governance_mode", "propose_apply")
    monkeypatch.setenv("MIND_MEM_SCOPE", "admin")
    return root


def _gov():
    import mind_mem.mcp.tools.governance as gov

    return gov


def _call(ws: str, fn, **kwargs: Any) -> dict:
    from mind_mem.mcp.infra.workspace import use_workspace

    with use_workspace(ws):
        return json.loads(fn(**kwargs))


def _propose(ws: str, statement: str, **extra: Any) -> dict:
    return _call(ws, _gov().propose_update, block_type="decision", statement=statement, rationale=RATIONALE, **extra)


def _decision(ws: str, block_id: str) -> dict:
    for block in parse_file(os.path.join(ws, "decisions", "DECISIONS.md")):
        if block.get("_id") == block_id:
            return block
    raise AssertionError(f"{block_id} not in DECISIONS.md")


def _stage_append(ws: str, statement: str) -> str:
    """Stage the correction the way a plain append proposal carries it."""
    block = {
        "_id": "D-20261007-001",
        "Statement": statement,
        "Date": "2026-10-07",
        "Status": "active",
        "Type": "decision",
        "Scope": "project:azariel",
        "Rationale": RATIONALE,
        "Supersedes": "none",
        "Tags": "schedule, azariel",
        "Sources": ["mcp:propose_update"],
    }
    patch = _render_block(block).rstrip("\n")
    proposal = {
        "ProposalId": "P-20261007-001",
        "Type": "decision",
        "TargetBlock": "D-20261007-001",
        "Risk": "medium",
        "Evidence": ["Correction relayed by Ilona"],
        "Rollback": "restore_snapshot",
        "Ops": [{"op": "append_block", "file": "decisions/DECISIONS.md", "patch": patch}],
    }
    fingerprint = compute_fingerprint(proposal)
    ops = "\n".join(f"    {line}" for line in patch.splitlines())
    text = (
        "\n[P-20261007-001]\nProposalId: P-20261007-001\nType: decision\nTargetBlock: D-20261007-001\n"
        "Risk: medium\nEvidence:\n- Correction relayed by Ilona\nRollback: restore_snapshot\nOps:\n"
        f"- op: append_block\n  file: decisions/DECISIONS.md\n  patch: |\n{ops}\n"
        f"Fingerprint: {fingerprint}\nStatus: staged\nFilesTouched:\n- decisions/DECISIONS.md\n"
    )
    path = os.path.join(ws, "intelligence", "proposed", "DECISIONS_PROPOSED.md")
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(text)
    staged, _ = find_proposal(ws, "P-20261007-001")
    assert staged is not None and staged.get("Fingerprint") == fingerprint
    return "P-20261007-001"


def _contradiction_ids(report: dict) -> list[str]:
    return sorted(c["block_id"] for c in report["conflicts"] if c["conflict_type"] == "contradiction")


# --- apply time sees what propose time saw ---------------------------------


def test_apply_time_reads_the_statement_inside_the_patch(ws):
    pid = _stage_append(ws, THURSDAY)
    proposal, _ = find_proposal(ws, pid)
    apply_report = check_proposal_contradictions(ws, proposal)
    propose_report = check_proposal_contradictions(ws, {"Statement": THURSDAY})
    assert _contradiction_ids(propose_report) == [TUESDAY_ID]
    assert _contradiction_ids(apply_report) == _contradiction_ids(propose_report)


def test_approve_apply_of_a_correction_without_supersedes_is_blocked(ws):
    pid = _stage_append(ws, THURSDAY)
    result = _call(ws, _gov().approve_apply, proposal_id=pid, dry_run=False)
    assert result["status"] == "blocked_contradictions"
    assert [c["block_id"] for c in result["contradictions"]["conflicts"] if c["conflict_type"] == "contradiction"] == [TUESDAY_ID]
    # Nothing landed: Tuesday is still the one active truth.
    assert _decision(ws, TUESDAY_ID)["Status"] == "active"
    assert all(b.get("Statement") != THURSDAY for b in parse_file(os.path.join(ws, "decisions", "DECISIONS.md")))


# --- BM25 leg honours committed statuses -----------------------------------


def test_bm25_candidate_that_is_superseded_is_not_a_contradiction(ws, monkeypatch):
    import mind_mem.contradiction_detector as cd

    superseded = {"_id": "D-20260901-009", "Status": "superseded", "Statement": TUESDAY, "excerpt": TUESDAY}
    live = {"_id": TUESDAY_ID, "Status": "active", "Statement": TUESDAY, "excerpt": TUESDAY}
    monkeypatch.setattr(cd, "_bm25_recall", lambda *_a, **_k: [superseded, live])
    report = check_proposal_contradictions(ws, {"Statement": THURSDAY})
    ids = _contradiction_ids(report)
    # Positive control: the same text on an active block IS flagged.
    assert TUESDAY_ID in ids
    assert "D-20260901-009" not in ids


def test_bm25_candidate_without_status_uses_the_corpus_status(ws, monkeypatch):
    import mind_mem.contradiction_detector as cd

    path = os.path.join(ws, "decisions", "DECISIONS.md")
    text = Path(path).read_text(encoding="utf-8")
    Path(path).write_text(
        text.replace(
            "Status: active\nScope: project:azariel\nStatement: " + TUESDAY,
            "Status: superseded\nScope: project:azariel\nStatement: " + TUESDAY,
        ),
        encoding="utf-8",
    )
    assert _decision(ws, TUESDAY_ID)["Status"] == "superseded"
    monkeypatch.setattr(cd, "_bm25_recall", lambda *_a, **_k: [{"_id": TUESDAY_ID, "excerpt": TUESDAY}])
    assert _contradiction_ids(check_proposal_contradictions(ws, {"Statement": THURSDAY})) == []


# --- contradicts carries the id to pass back -------------------------------


def test_contradicts_names_the_supersedes_argument(ws):
    envelope = _propose(ws, THURSDAY)
    assert envelope["status"] == "proposed"
    assert [c["block_id"] for c in envelope["contradicts"]] == [TUESDAY_ID]
    assert envelope["contradicts"][0]["supersedes"] == TUESDAY_ID
    assert "supersedes" in envelope["next_step"]


# --- propose_update(supersedes=...) end to end ------------------------------


def test_supersedes_round_trip_retires_the_old_truth(ws):
    staged = _propose(ws, THURSDAY, supersedes=TUESDAY_ID)
    assert staged["status"] == "staged", staged
    assert staged["target_block"] == TUESDAY_ID
    assert "contradicts" not in staged  # the contradiction is the one being resolved
    pid = staged["proposal_id"]
    new_id = staged["new_block"]

    proposal, _ = find_proposal(ws, pid)
    assert proposal is not None
    assert [op["op"] for op in proposal["Ops"]] == ["supersede_decision"]
    # Same verdict at apply time: the retired target is shown, not counted.
    report = check_proposal_contradictions(ws, proposal)
    assert _contradiction_ids(report) == []
    assert report["has_contradictions"] is False
    assert [c["block_id"] for c in report["conflicts"] if c.get("resolved_by_supersede")] == [TUESDAY_ID]

    applied = _call(ws, _gov().approve_apply, proposal_id=pid, dry_run=False)
    assert applied["status"] == "applied", applied

    old = _decision(ws, TUESDAY_ID)
    assert old["Status"] == "superseded"
    assert old["SupersededBy"] == new_id
    new = _decision(ws, new_id)
    assert new["Status"] == "active"
    assert new["Statement"] == THURSDAY
    assert new["Supersedes"] == TUESDAY_ID

    from mind_mem.mcp.tools.recall import recall

    hits = _call(ws, recall, query="Azariel weekly planning call with Ilona", limit=1)
    results = hits["results"] if isinstance(hits, dict) else hits
    assert results[0]["_id"] == new_id


def test_supersedes_still_reports_other_contradictions(ws):
    # The investor call is a different fact; superseding it with the Ilona
    # correction must not hide the Tuesday contradiction.
    staged = _propose(ws, THURSDAY, supersedes="D-20261001-002")
    assert staged["status"] == "staged"
    assert [c["block_id"] for c in staged["contradicts"]] == [TUESDAY_ID]


@pytest.mark.parametrize(
    ("target", "reason"),
    [
        ("D-20991231-999", "not found"),
        ("not an id", "invalid"),
        ("D-20261001-001\nStatus: x", "invalid"),
    ],
)
def test_supersedes_refuses_bad_targets_and_writes_nothing(ws, target, reason):
    before = Path(ws, "intelligence", "proposed", "EDITS_PROPOSED.md").read_text(encoding="utf-8")
    signals_before = Path(ws, "intelligence", "SIGNALS.md").read_text(encoding="utf-8")
    envelope = _propose(ws, THURSDAY, supersedes=target)
    assert "error" in envelope and reason in envelope["error"]
    assert Path(ws, "intelligence", "proposed", "EDITS_PROPOSED.md").read_text(encoding="utf-8") == before
    assert Path(ws, "intelligence", "SIGNALS.md").read_text(encoding="utf-8") == signals_before


def test_supersedes_refuses_an_already_superseded_target(ws):
    first = _propose(ws, THURSDAY, supersedes=TUESDAY_ID)
    assert _call(ws, _gov().approve_apply, proposal_id=first["proposal_id"], dry_run=False)["status"] == "applied"
    again = _propose(ws, "Azariel's weekly planning call with Ilona is on Friday at 10am Arizona time.", supersedes=TUESDAY_ID)
    assert "error" in again and "not active" in again["error"]


def test_supersedes_is_decision_only(ws):
    envelope = _call(
        ws,
        _gov().propose_update,
        block_type="task",
        statement="Move the Ilona planning call to Thursday.",
        rationale=RATIONALE,
        supersedes=TUESDAY_ID,
    )
    assert "error" in envelope and "decision" in envelope["error"]


def test_contradicted_then_resubmitted_as_supersede_passes_a_strict_gate(ws):
    # The documented loop: propose, read `contradicts`, re-propose with the id.
    # Under a strict quality gate the caller's own identical signal must not
    # read as a near-duplicate of the supersession it is converting into.
    config_set(os.path.join(ws, "mind-mem.json"), "quality_gate.mode", "strict")
    first = _propose(ws, THURSDAY)
    assert first["status"] == "proposed"
    staged = _propose(ws, THURSDAY, supersedes=first["contradicts"][0]["supersedes"])
    assert staged["status"] == "staged", staged
    # Positive control: a plain re-proposal of the same text IS a near-duplicate.
    again = _propose(ws, THURSDAY)
    assert again.get("error") == "quality_gate_rejection"


def test_supersede_staging_is_deduplicated(ws):
    first = _propose(ws, THURSDAY, supersedes=TUESDAY_ID)
    assert first["status"] == "staged"
    second = _propose(ws, THURSDAY, supersedes=TUESDAY_ID)
    assert "error" in second and "already staged" in second["error"]
