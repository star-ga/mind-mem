"""M4 closed-set slots: declaration, staging, and governed supersession."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from mind_mem.closed_slots import (
    PendingSlotProposalError,
    SlotConfigError,
    UnknownSlotError,
    find_active_slot_occupant,
    load_slot_declarations,
    stage_slot_update,
)


def _workspace(tmp_path: Path, *, declarations: object) -> Path:
    for rel in ("decisions", "tasks", "entities", "intelligence/proposed", "memory"):
        (tmp_path / rel).mkdir(parents=True)
    (tmp_path / "decisions/DECISIONS.md").write_text("", encoding="utf-8")
    (tmp_path / "intelligence/proposed/EDITS_PROPOSED.md").write_text("", encoding="utf-8")
    (tmp_path / "mind-mem.json").write_text(json.dumps({"closed_slots": declarations}), encoding="utf-8")
    return tmp_path


def _decl() -> dict[str, object]:
    return {"version": 1, "namespaces": {"profile": {"version": 1, "slots": ["status", "tier"]}}}


def test_unknown_namespace_and_member_are_rejected(tmp_path: Path) -> None:
    ws = _workspace(tmp_path, declarations=_decl())
    with pytest.raises(UnknownSlotError, match="not a declared"):
        stage_slot_update(str(ws), "settings", "status", "on", rationale="valid authored reason")
    with pytest.raises(UnknownSlotError, match="allowed"):
        stage_slot_update(str(ws), "profile", "colour", "blue", rationale="valid authored reason")
    assert (ws / "decisions/DECISIONS.md").read_text(encoding="utf-8") == ""


def test_malformed_declaration_fails_closed(tmp_path: Path) -> None:
    ws = _workspace(tmp_path, declarations={"version": 1, "namespaces": {"profile": {"version": 1, "slots": ["status", "status"]}}})
    with pytest.raises(SlotConfigError, match="duplicate"):
        load_slot_declarations(str(ws))


def test_stage_is_reviewable_and_same_value_is_reassertion(tmp_path: Path) -> None:
    ws = _workspace(tmp_path, declarations=_decl())
    first = stage_slot_update(
        str(ws), "profile", "status", "active", rationale="the profile is active", now=datetime(2026, 9, 14, tzinfo=timezone.utc)
    )
    assert first["status"] == "staged"
    proposal = (ws / "intelligence/proposed/EDITS_PROPOSED.md").read_text(encoding="utf-8")
    assert first["proposal_id"] in proposal
    assert "approve_apply" not in proposal  # proposal text remains data, not an approval call

    # A repeated request is idempotent while the first proposal awaits review.
    repeated = stage_slot_update(str(ws), "profile", "status", "active", rationale="the profile is active")
    assert repeated["status"] == "already_staged"
    assert repeated["proposal_id"] == first["proposal_id"]


def test_same_active_value_is_reassertion_without_lineage_churn(tmp_path: Path) -> None:
    ws = _workspace(tmp_path, declarations=_decl())
    (ws / "decisions/DECISIONS.md").write_text(
        "[D-20260914-001]\nStatement: active\nStatus: active\nType: decision\n"
        "Tags: closed-slot, profile, status\nSlotNamespace: profile\nSlotName: status\n"
        "SlotSetVersion: 1\n\n---\n",
        encoding="utf-8",
    )
    result = stage_slot_update(str(ws), "profile", "status", "active", rationale="the profile is active")
    assert result["status"] == "reasserted"
    assert result["occupant_id"] == "D-20260914-001"
    assert not (ws / "intelligence/proposed/EDITS_PROPOSED.md").read_text(encoding="utf-8").strip()


def test_different_pending_value_is_refused_without_overwrite(tmp_path: Path) -> None:
    ws = _workspace(tmp_path, declarations=_decl())
    stage_slot_update(str(ws), "profile", "status", "active", rationale="the profile is active")
    with pytest.raises(PendingSlotProposalError, match="already has staged"):
        stage_slot_update(str(ws), "profile", "status", "inactive", rationale="the profile changed")
    assert (ws / "decisions/DECISIONS.md").read_text(encoding="utf-8") == ""


def test_active_slot_lookup_is_exact_and_free_form_is_ignored(tmp_path: Path) -> None:
    ws = _workspace(tmp_path, declarations=_decl())
    (ws / "decisions/DECISIONS.md").write_text(
        "[D-20260914-001]\nStatement: active\nStatus: active\nType: decision\n"
        "SlotNamespace: profile\nSlotName: status\n\n---\n"
        "[D-20260914-002]\nStatement: unrelated\nStatus: active\nType: decision\n\n---\n",
        encoding="utf-8",
    )
    occupant = find_active_slot_occupant(str(ws), "profile", "status")
    assert occupant is not None and occupant["_id"] == "D-20260914-001"
    assert find_active_slot_occupant(str(ws), "profile", "tier") is None


def test_approval_gate_records_supersession_lineage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The public governance entrypoint applies, and only applies, the upsert."""
    from mind_mem.init_workspace import init
    from mind_mem.mcp.tools import governance
    from mind_mem.spec_binding import SpecBindingManager

    init(str(tmp_path))
    config_path = tmp_path / "mind-mem.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.update({"governance_mode": "propose", "closed_slots": _decl()})
    config_path.write_text(json.dumps(config), encoding="utf-8")
    SpecBindingManager(str(config_path)).rebind(str(config_path))
    monkeypatch.setenv("MIND_MEM_WORKSPACE", str(tmp_path))

    first = json.loads(governance.propose_slot_update.__wrapped__("profile", "status", "active", "the profile is active"))
    applied = json.loads(governance.approve_apply.__wrapped__(first["proposal_id"], dry_run=False))
    assert applied["status"] == "applied"

    # The ordinary no-touch policy is not the slot contract; age the test
    # workspace's local apply marker so the second explicit approval can run.
    state_path = tmp_path / "memory/intel-state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["last_apply_ts"] = "2020-01-01T00:00:00Z"
    state_path.write_text(json.dumps(state), encoding="utf-8")

    second = json.loads(governance.propose_slot_update.__wrapped__("profile", "status", "inactive", "the profile changed"))
    applied2 = json.loads(governance.approve_apply.__wrapped__(second["proposal_id"], dry_run=False))
    assert applied2["status"] == "applied"
    decisions = (tmp_path / "decisions/DECISIONS.md").read_text(encoding="utf-8")
    assert "Status: superseded" in decisions
    assert "SupersededBy: D-" in decisions
    assert "Supersedes: D-" in decisions
    assert decisions.count("SlotName: status") == 2


def test_decorated_slot_stage_runs_provenance_gate_and_accepts_attribution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The public admin tool cannot bypass the ordinary pre-write door."""
    from mind_mem.init_workspace import init
    from mind_mem.mcp.tools import governance
    from mind_mem.spec_binding import SpecBindingManager

    init(str(tmp_path))
    config_path = tmp_path / "mind-mem.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.update(
        {
            "governance_mode": "propose",
            "closed_slots": _decl(),
            "v4": {"provenance": {"enabled": True, "policy": "required", "fields": ["ActorId"]}},
        }
    )
    config_path.write_text(json.dumps(config), encoding="utf-8")
    SpecBindingManager(str(config_path)).rebind(str(config_path))
    monkeypatch.setenv("MIND_MEM_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("MIND_MEM_SCOPE", "admin")

    refused = json.loads(governance.propose_slot_update("profile", "status", "active", "the profile is active"))
    assert refused["error"] == "provenance_required"
    assert "ProposalId: P-" not in (tmp_path / "intelligence/proposed/EDITS_PROPOSED.md").read_text(encoding="utf-8")

    admitted = json.loads(
        governance.propose_slot_update(
            "profile",
            "status",
            "active",
            "the profile is active",
            actor_id="agent-7",
        )
    )
    assert admitted["status"] == "staged"
    staged = (tmp_path / "intelligence/proposed/EDITS_PROPOSED.md").read_text(encoding="utf-8")
    assert "ActorId: agent-7" in staged


def test_apply_rejects_slot_proposal_after_declaration_changes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Approval revalidates the authored declaration under the apply lock."""
    from mind_mem.init_workspace import init
    from mind_mem.mcp.tools import governance
    from mind_mem.spec_binding import SpecBindingManager

    init(str(tmp_path))
    config_path = tmp_path / "mind-mem.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.update({"governance_mode": "propose", "closed_slots": _decl()})
    config_path.write_text(json.dumps(config), encoding="utf-8")
    SpecBindingManager(str(config_path)).rebind(str(config_path))
    monkeypatch.setenv("MIND_MEM_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("MIND_MEM_SCOPE", "admin")

    staged = json.loads(governance.propose_slot_update.__wrapped__("profile", "status", "active", "the profile is active"))
    changed = json.loads(config_path.read_text(encoding="utf-8"))
    changed["closed_slots"]["namespaces"]["profile"]["version"] = 2
    changed["closed_slots"]["namespaces"]["profile"]["slots"] = ["tier"]
    config_path.write_text(json.dumps(changed), encoding="utf-8")
    SpecBindingManager(str(config_path)).rebind(str(config_path))

    result = json.loads(governance.approve_apply.__wrapped__(staged["proposal_id"], dry_run=False))
    assert result["status"] == "failed"
    assert "SlotName: status" not in (tmp_path / "decisions/DECISIONS.md").read_text(encoding="utf-8")


def test_apply_rejects_multi_block_payload_containing_a_slot(tmp_path: Path) -> None:
    """A slot operation cannot smuggle a second parsed block past the guard."""
    from mind_mem.closed_slots import SlotInvariantError, _validate_slot_payload

    ws = _workspace(tmp_path, declarations=_decl())
    slot = {
        "_id": "D-20260914-001",
        "Statement": "active",
        "Status": "active",
        "SlotNamespace": "profile",
        "SlotName": "status",
        "SlotSetVersion": "1",
        "SlotValueDigest": hashlib.sha256(b"active").hexdigest(),
    }
    ordinary = {"_id": "D-20260914-002", "Statement": "free form", "Status": "active"}
    with pytest.raises(SlotInvariantError, match="exactly one block"):
        _validate_slot_payload(str(ws), [slot, ordinary], active_blocks=[])
