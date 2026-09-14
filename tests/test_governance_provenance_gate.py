"""Governance receipt scopes enforce the workspace provenance policy."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mind_mem.compliance.provenance_policy import ProvenanceRequired
from mind_mem.enums import IngestTier
from mind_mem.governance_gate import evict_gate, get_gate
from mind_mem.init_workspace import init
from mind_mem.storage import get_block_store

PROVENANCE = {
    "actor_id": "agent-1",
    "actor_role": "operator",
    "session_id": "session-1",
    "tool_id": "mm",
    "purpose": "governed test write",
}


def _set_policy(workspace: Path, policy: str) -> None:
    config_path = workspace / "mind-mem.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.setdefault("v4", {})["provenance"] = {"enabled": True, "policy": policy}
    config_path.write_text(json.dumps(config), encoding="utf-8")


def _block() -> dict[str, object]:
    return {
        "_id": "D-20260914-001",
        "Status": "quarantined",
        "Statement": "a provenance gate control",
        "ActorId": PROVENANCE["actor_id"],
        "ActorRole": PROVENANCE["actor_role"],
        "SessionId": PROVENANCE["session_id"],
        "ToolId": PROVENANCE["tool_id"],
        "Purpose": PROVENANCE["purpose"],
    }


def test_required_policy_refuses_direct_gate_scope_before_evidence_or_store(tmp_path: Path) -> None:
    init(str(tmp_path))
    _set_policy(tmp_path, "required")
    block = _block()
    store = get_block_store(str(tmp_path))

    with pytest.raises(ProvenanceRequired, match="provenance policy 'required'"):
        with get_gate(str(tmp_path)).admit_block(
            "INGEST",
            str(block["_id"]),
            str(block["Statement"]),
            tier=IngestTier.EXTERNAL_INGEST,
        ):
            store.write_block(block)

    assert store.get_by_id(str(block["_id"])) is None
    evidence = tmp_path / "memory" / "evidence_chain.jsonl"
    assert not evidence.exists() or evidence.read_text(encoding="utf-8").strip() == ""


def test_required_policy_accepts_explicit_attribution_and_lands_block(tmp_path: Path) -> None:
    init(str(tmp_path))
    _set_policy(tmp_path, "required")
    block = _block()
    store = get_block_store(str(tmp_path))

    with get_gate(str(tmp_path)).admit_block(
        "INGEST",
        str(block["_id"]),
        str(block["Statement"]),
        tier=IngestTier.EXTERNAL_INGEST,
        provenance=PROVENANCE,
    ):
        store.write_block(block)

    assert store.get_by_id(str(block["_id"])) is not None


def test_off_policy_preserves_legacy_direct_gate_write(tmp_path: Path) -> None:
    init(str(tmp_path))
    block = _block()
    store = get_block_store(str(tmp_path))

    with get_gate(str(tmp_path)).admit_block(
        "INGEST",
        str(block["_id"]),
        str(block["Statement"]),
        tier=IngestTier.EXTERNAL_INGEST,
    ):
        store.write_block(block)

    assert store.get_by_id(str(block["_id"])) is not None


@pytest.mark.parametrize("scope", ("batch", "proposal", "edge", "artifact"))
def test_required_policy_refuses_every_write_scope(scope: str, tmp_path: Path) -> None:
    init(str(tmp_path))
    _set_policy(tmp_path, "required")
    gate = get_gate(str(tmp_path))
    try:
        with pytest.raises(ProvenanceRequired):
            if scope == "batch":
                with gate.admit_batch("WRITE", "batch-1", ("D-20260914-002",), "body", tier=IngestTier.EXTERNAL_INGEST):
                    pass
            elif scope == "proposal":
                with gate.admit_proposal("P-20260914-002", "[]"):
                    pass
            elif scope == "edge":
                with gate.admit_edge("E-20260914-002", "edge"):
                    pass
            else:
                with gate.admit_artifact("CT-20260914-002", "artifact"):
                    pass
    finally:
        evict_gate(str(tmp_path))


def test_staged_proposal_scope_reads_current_required_policy(tmp_path: Path) -> None:
    init(str(tmp_path))
    gate = get_gate(str(tmp_path))
    try:
        _set_policy(tmp_path, "required")
        with pytest.raises(ProvenanceRequired, match="provenance policy 'required'"):
            with gate.admit_proposal("P-20260914-001", "[]", actor="apply_engine"):
                pytest.fail("required policy must be checked before apply writes")
    finally:
        evict_gate(str(tmp_path))
