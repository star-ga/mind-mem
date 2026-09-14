"""Public apply keeps provenance refusals structured and transactional."""

from __future__ import annotations

import json
from pathlib import Path

from mind_mem import apply_engine
from mind_mem.apply_engine import compute_fingerprint
from mind_mem.block_parser import parse_file
from mind_mem.enums import IngestTier
from mind_mem.governance_gate import evict_gate, get_gate
from mind_mem.init_workspace import init
from mind_mem.mcp.infra.workspace import use_workspace
from mind_mem.mcp.tools import governance
from mind_mem.spec_binding import SpecBindingManager
from mind_mem.storage import get_block_store

PROVENANCE = {
    "actor_id": "e3-apply-actor",
    "actor_role": "operator",
    "session_id": "e3-apply-session",
    "tool_id": "e3-apply-test",
    "purpose": "prove transactional provenance refusal",
}
PROVENANCE_FIELDS = {
    "actor_id": "ActorId",
    "actor_role": "ActorRole",
    "session_id": "SessionId",
    "tool_id": "ToolId",
    "purpose": "Purpose",
}


def _block(block_id: str, statement: str) -> dict[str, str]:
    return {
        "_id": block_id,
        "Status": "quarantined",
        "Statement": statement,
        **{field: PROVENANCE[param] for param, field in PROVENANCE_FIELDS.items()},
    }


def _workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    init(str(workspace))
    config_path = workspace / "mind-mem.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.update(
        {
            "governance_mode": "propose",
            "v4": {"provenance": {"enabled": True, "policy": "required"}},
        }
    )
    config_path.write_text(json.dumps(config), encoding="utf-8")
    SpecBindingManager(str(config_path)).rebind(str(config_path))
    (workspace / "intelligence/proposed").mkdir(parents=True, exist_ok=True)
    (workspace / "intelligence/proposed/DECISIONS_PROPOSED.md").write_text("", encoding="utf-8")
    (workspace / "decisions/DECISIONS.md").write_text("", encoding="utf-8")
    evict_gate(str(workspace))
    return workspace


def _land_initial(workspace: Path, block: dict[str, str]) -> None:
    store = get_block_store(str(workspace))
    with get_gate(str(workspace)).admit_block(
        "WRITE",
        block["_id"],
        block["Statement"],
        tier=IngestTier.EXTERNAL_INGEST,
        provenance=PROVENANCE,
    ):
        store.write_block(block)


def _write_proposal(workspace: Path, proposal_id: str, ops: list[dict[str, object]], target: str) -> None:
    fingerprint = compute_fingerprint({"Type": "edit", "TargetBlock": target, "Ops": ops})
    lines = [
        f"[{proposal_id}]",
        f"ProposalId: {proposal_id}",
        "Type: edit",
        f"TargetBlock: {target}",
        "Risk: low",
        "Status: staged",
        "Evidence:",
        "- E3 apply provenance transaction control",
        "Rollback: restore the snapshot",
        f"Fingerprint: {fingerprint}",
        "Ops:",
    ]
    for op in ops:
        lines.append(f"- op: {op['op']}")
        lines.append(f"  file: {op['file']}")
        if op.get("target"):
            lines.append(f"  target: {op['target']}")
        for key in ("field", "value", "status"):
            if key in op:
                lines.append(f"  {key}: {op[key]}")
        if "patch" in op:
            lines.append("  patch: |")
            lines.extend(f"    {line}" for line in str(op["patch"]).splitlines())
    lines.extend(
        [
            "FilesTouched:",
            "- decisions/DECISIONS.md",
            "Sources:",
            "- E3 test fixture",
            "ActorId: e3-apply-actor",
            "ActorRole: operator",
            "SessionId: e3-apply-session",
            "ToolId: e3-apply-test",
            "Purpose: prove transactional provenance refusal",
            "",
            "---",
            "",
        ]
    )
    (workspace / "intelligence/proposed/DECISIONS_PROPOSED.md").write_text("\n".join(lines), encoding="utf-8")


def _apply(workspace: Path, proposal_id: str, monkeypatch) -> dict[str, object]:
    monkeypatch.setattr(apply_engine, "check_preconditions", lambda _workspace: (True, ["fixture preconditions: PASS"]))
    monkeypatch.setenv("MIND_MEM_WORKSPACE", str(workspace))
    monkeypatch.setenv("MIND_MEM_SCOPE", "admin")
    with use_workspace(str(workspace)):
        return json.loads(governance.approve_apply.__wrapped__(proposal_id, dry_run=False))


def test_full_required_provenance_apply_persists_all_fields(tmp_path: Path, monkeypatch) -> None:
    workspace = _workspace(tmp_path)
    block = _block("D-20260914-981", "full provenance apply positive")
    patch = f"[{block['_id']}]\n" + "\n".join(f"{key}: {value}" for key, value in block.items() if key != "_id")
    ops = [{"op": "append_block", "file": "decisions/DECISIONS.md", "target": block["_id"], "patch": patch}]
    _write_proposal(workspace, "P-20260914-981", ops, block["_id"])
    result = _apply(workspace, "P-20260914-981", monkeypatch)
    assert result["status"] == "applied"
    landed = parse_file(str(workspace / "decisions/DECISIONS.md"))
    stored = next(item for item in landed if item.get("_id") == block["_id"])
    for param, field in PROVENANCE_FIELDS.items():
        assert stored[field] == PROVENANCE[param]


def test_forged_nested_provenance_returns_failure_and_rolls_back_prior_op(tmp_path: Path, monkeypatch) -> None:
    workspace = _workspace(tmp_path)
    original = _block("D-20260914-982", "original transaction value")
    _land_initial(workspace, original)
    forged = _block("D-20260914-983", "forged nested payload")
    forged["ActorId"] = "forged-by-fixture"
    forged_patch = f"[{forged['_id']}]\n" + "\n".join(f"{key}: {value}" for key, value in forged.items() if key != "_id")
    ops = [
        {
            "op": "update_field",
            "file": "decisions/DECISIONS.md",
            "target": original["_id"],
            "field": "Statement",
            "value": "transient first operation",
        },
        {"op": "append_block", "file": "decisions/DECISIONS.md", "target": forged["_id"], "patch": forged_patch},
    ]
    _write_proposal(workspace, "P-20260914-982", ops, original["_id"])
    before = (workspace / "decisions/DECISIONS.md").read_bytes()
    result = _apply(workspace, "P-20260914-982", monkeypatch)
    after = (workspace / "decisions/DECISIONS.md").read_bytes()

    assert result["status"] == "failed"
    assert result["success"] is False
    assert "provenance does not match" in str(result["message"])
    assert after == before
    assert parse_file(str(workspace / "decisions/DECISIONS.md"))[0]["Statement"] == original["Statement"]
    assert not any(item.get("_id") == forged["_id"] for item in parse_file(str(workspace / "decisions/DECISIONS.md")))
