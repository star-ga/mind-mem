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


def test_decorated_slot_approval_rechecks_current_provenance_policy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An admin approval cannot apply a proposal after its policy changes."""
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

    positive = json.loads(governance.propose_slot_update("profile", "tier", "gold", "the profile tier is gold", actor_id="agent-7"))
    assert positive["status"] == "staged"
    applied = json.loads(governance.approve_apply(positive["proposal_id"], dry_run=False))
    assert applied["status"] == "applied"

    state_path = tmp_path / "memory/intel-state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["last_apply_ts"] = "2020-01-01T00:00:00Z"
    state_path.write_text(json.dumps(state), encoding="utf-8")

    staged = json.loads(governance.propose_slot_update("profile", "status", "active", "the profile is active", actor_id="agent-7"))
    assert staged["status"] == "staged"
    changed = json.loads(config_path.read_text(encoding="utf-8"))
    changed["v4"]["provenance"]["fields"] = ["ActorRole"]
    config_path.write_text(json.dumps(changed), encoding="utf-8")
    SpecBindingManager(str(config_path)).rebind(str(config_path))
    from mind_mem.governance_gate import evict_gate

    evict_gate(str(tmp_path))

    refused = json.loads(governance.approve_apply(staged["proposal_id"], dry_run=False))
    assert refused["status"] == "failed"
    assert "current policy" in refused["message"]
    assert "SlotName: status" not in (tmp_path / "decisions/DECISIONS.md").read_text(encoding="utf-8")


def test_slot_stage_reuses_provenance_redaction_and_codepoint_gates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Slot metadata follows shared length, vocabulary, redaction, and Unicode controls."""
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

    overlong = json.loads(governance.propose_slot_update("profile", "status", "active", "the profile is active", actor_id="x" * 257))
    assert "exceeds" in overlong["error"]

    admitted = json.loads(governance.propose_slot_update("profile", "status", "active", "the profile is active", actor_id="agent\u200b-7"))
    assert admitted["status"] == "staged"
    staged = (tmp_path / "intelligence/proposed/EDITS_PROPOSED.md").read_text(encoding="utf-8")
    assert "ActorId: agent-7" in staged
    assert "\u200b" not in staged

    # A reject-mode detector must produce the same stable refusal envelope as
    # ordinary propose_update, before a proposal is appended.
    reject_config = json.loads(config_path.read_text(encoding="utf-8"))
    reject_config["v4"]["redaction"] = {"enabled": True, "mode": "reject", "detectors": ["email"]}
    config_path.write_text(json.dumps(reject_config), encoding="utf-8")
    SpecBindingManager(str(config_path)).rebind(str(config_path))
    refused = json.loads(
        governance.propose_slot_update("profile", "tier", "mail ops@example.com", "the tier contains an address", actor_id="agent-7")
    )
    assert refused["error"] == "redaction_refused"
    assert "SlotName: tier" not in (tmp_path / "intelligence/proposed/EDITS_PROPOSED.md").read_text(encoding="utf-8")


def test_slot_approval_rechecks_changed_redaction_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A staged clear-text slot must be restaged when current redaction would reject it."""
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

    staged = json.loads(governance.propose_slot_update.__wrapped__("profile", "status", "mail ops@example.com", "the profile has mail"))
    assert staged["status"] == "staged"
    changed = json.loads(config_path.read_text(encoding="utf-8"))
    changed["v4"] = {"redaction": {"enabled": True, "mode": "reject", "detectors": ["email"]}}
    config_path.write_text(json.dumps(changed), encoding="utf-8")
    SpecBindingManager(str(config_path)).rebind(str(config_path))
    from mind_mem.governance_gate import evict_gate

    evict_gate(str(tmp_path))

    refused = json.loads(governance.approve_apply.__wrapped__(staged["proposal_id"], dry_run=False))
    assert refused["status"] == "failed"
    assert "restaging" in refused["message"]
    assert "SlotName: status" not in (tmp_path / "decisions/DECISIONS.md").read_text(encoding="utf-8")


def test_slot_stage_uses_enabled_v4_field_vocabulary_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An enabled workspace vocabulary also governs closed-slot metadata."""
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
            "v4": {"block_metadata": {"enabled": True}, "vocabulary": {"enabled": True}},
            "vocabularies": {"actor_role": {"values": ["admin"], "mode": "reject"}},
        }
    )
    config_path.write_text(json.dumps(config), encoding="utf-8")
    SpecBindingManager(str(config_path)).rebind(str(config_path))
    monkeypatch.setenv("MIND_MEM_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("MIND_MEM_SCOPE", "admin")

    refused = json.loads(governance.propose_slot_update("profile", "status", "active", "the profile is active", actor_role="planner"))
    assert refused["error"].startswith("schema_validation_rejection")
    assert "SlotName: status" not in (tmp_path / "intelligence/proposed/EDITS_PROPOSED.md").read_text(encoding="utf-8")


def test_explicit_workspace_binds_v4_flags_during_staging(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A direct slot call cannot borrow v4 policy from the ambient workspace."""
    from mind_mem.closed_slots import ClosedSlotError

    explicit = _workspace(tmp_path / "explicit", declarations=_decl())
    ambient = _workspace(tmp_path / "ambient", declarations=_decl())
    (explicit / "mind-mem.json").write_text(
        json.dumps(
            {
                "governance_mode": "propose",
                "closed_slots": _decl(),
                "v4": {"block_metadata": {"enabled": True}, "vocabulary": {"enabled": True}},
                "vocabularies": {"actor_role": {"values": ["admin"], "mode": "reject"}},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("MIND_MEM_WORKSPACE", str(ambient))

    with pytest.raises(ClosedSlotError, match="schema_validation_rejection"):
        stage_slot_update(str(explicit), "profile", "status", "active", rationale="the profile is active", actor_role="planner")
    assert (explicit / "intelligence/proposed/EDITS_PROPOSED.md").read_text(encoding="utf-8") == ""


def test_explicit_workspace_binds_v4_flags_during_approval(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Approval rechecks the explicit workspace even when ambient config differs."""
    from mind_mem.apply_engine import apply_proposal
    from mind_mem.closed_slots import stage_slot_update
    from mind_mem.init_workspace import init
    from mind_mem.spec_binding import SpecBindingManager
    from mind_mem.v4 import block_metadata

    explicit = tmp_path / "explicit"
    ambient = tmp_path / "ambient"
    init(str(explicit))
    init(str(ambient))
    (explicit / "mind-mem.json").write_text(
        json.dumps({"governance_mode": "propose", "closed_slots": _decl(), "v4": {"block_metadata": {"enabled": True}}}),
        encoding="utf-8",
    )
    (ambient / "mind-mem.json").write_text(json.dumps({"governance_mode": "propose", "closed_slots": _decl()}), encoding="utf-8")
    SpecBindingManager(str(explicit / "mind-mem.json")).rebind(str(explicit / "mind-mem.json"))

    original = dict(block_metadata._validators)
    try:
        monkeypatch.setenv("MIND_MEM_WORKSPACE", str(explicit))
        block_metadata.register_schema_validator("decision", lambda payload: block_metadata.SchemaValidationResult(ok=True))
        monkeypatch.setenv("MIND_MEM_WORKSPACE", str(ambient))
        staged = stage_slot_update(str(explicit), "profile", "status", "active", rationale="the profile is active")
        assert staged["status"] == "staged"

        monkeypatch.setenv("MIND_MEM_WORKSPACE", str(explicit))
        block_metadata.register_schema_validator(
            "decision", lambda payload: block_metadata.SchemaValidationResult(ok=False, reason="approval policy changed")
        )
        monkeypatch.setenv("MIND_MEM_WORKSPACE", str(ambient))
        success, message = apply_proposal(str(explicit), staged["proposal_id"], dry_run=False)
        assert not success
        assert "schema_validation_rejection" in message or "current policy" in message
        assert "SlotName: status" not in (explicit / "decisions/DECISIONS.md").read_text(encoding="utf-8")
    finally:
        block_metadata._validators.clear()
        block_metadata._validators.update(original)


def test_slot_approval_rechecks_new_strict_quality_policy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A proposal staged under advisory quality cannot bypass a later strict gate."""
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

    staged = json.loads(governance.propose_slot_update.__wrapped__("profile", "status", "x", "the profile is x"))
    assert staged["status"] == "staged"
    changed = json.loads(config_path.read_text(encoding="utf-8"))
    changed["quality_gate"] = {"mode": "strict"}
    config_path.write_text(json.dumps(changed), encoding="utf-8")
    SpecBindingManager(str(config_path)).rebind(str(config_path))
    from mind_mem.governance_gate import evict_gate

    evict_gate(str(tmp_path))

    refused = json.loads(governance.approve_apply.__wrapped__(staged["proposal_id"], dry_run=False))
    assert refused["status"] == "failed"
    assert "quality_gate_rejection" in refused["message"]
    assert "SlotName: status" not in (tmp_path / "decisions/DECISIONS.md").read_text(encoding="utf-8")


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
