# Copyright 2026 STARGA, Inc.
"""Public proposals preserve reasons and screen persisted fields before writing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mind_mem.block_parser import parse_file
from mind_mem.init_workspace import init
from mind_mem.mcp.tools.governance import propose_update

# AWS's documentation fixture, not a credential.
CANARY = "AKIAIOSFODNN7EXAMPLE"
REASON = "Independent review requires a reproducible result."


def _workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str | None) -> Path:
    ws = tmp_path / "ws"
    init(str(ws))
    cfg = ws / "mind-mem.json"
    data = json.loads(cfg.read_text())
    data.setdefault("v4", {})["redaction"] = (
        {"enabled": True, "mode": mode, "detectors": ["aws_access_key_id"]} if mode else {"enabled": False}
    )
    cfg.write_text(json.dumps(data))
    monkeypatch.setenv("MIND_MEM_WORKSPACE", str(ws))
    monkeypatch.setenv("MIND_MEM_CONFIG", str(cfg))
    monkeypatch.setenv("MIND_MEM_SCOPE", "admin")
    return ws


def _propose(**changes: str) -> dict:
    args = {
        "block_type": "decision",
        "statement": "Publish the measured result with its source receipt.",
        "rationale": REASON,
        "actor_id": "audit-operator",
    }
    args.update(changes)
    return json.loads(propose_update(**args))


def _records(ws: Path) -> list[dict]:
    return parse_file(str(ws / "intelligence" / "SIGNALS.md"))


@pytest.mark.parametrize("block_type", ["decision", "task"])
def test_supplied_reason_survives_public_proposal(tmp_path, monkeypatch, block_type):
    ws = _workspace(tmp_path, monkeypatch, None)
    assert _propose(block_type=block_type)["written"] == 1
    records = _records(ws)
    assert len(records) == 1
    assert records[0]["Rationale"] == REASON
    assert records[0]["Status"] in {"pending-review", "quarantined", "pending"}


@pytest.mark.parametrize("field", ["statement", "rationale", "tags", "purpose", "actor_id", "confidence"])
def test_reject_covers_every_persisted_input_without_leaving_secret_on_disk(tmp_path, monkeypatch, field):
    ws = _workspace(tmp_path, monkeypatch, "reject")
    signals = ws / "intelligence" / "SIGNALS.md"
    before = signals.read_bytes()
    result = _propose(**{field: CANARY})
    assert result["error"] == "redaction_refused"
    assert signals.read_bytes() == before
    # Includes the redaction/audit stores, not only SIGNALS.md. In particular
    # an actor value must not leak into audit metadata before being screened.
    assert all(CANARY.encode() not in p.read_bytes() for p in ws.rglob("*") if p.is_file())
    assert _propose()["written"] == 1  # The same policy admits a clean write.


@pytest.mark.parametrize("field, stored", [("statement", "Excerpt"), ("rationale", "Rationale"), ("tags", "Tags"), ("purpose", "Purpose")])
def test_redact_preserves_proposal_with_redacted_content(tmp_path, monkeypatch, field, stored):
    ws = _workspace(tmp_path, monkeypatch, "redact")
    assert _propose(**{field: CANARY})["written"] == 1
    record = _records(ws)[0]
    assert "REDACTED" in str(record[stored])
    assert record["ActorId"] == "audit-operator"
    assert all(CANARY.encode() not in p.read_bytes() for p in ws.rglob("*") if p.is_file())


@pytest.mark.parametrize("field", ["actor_id", "confidence"])
def test_redact_refuses_to_invent_an_identity_or_class(tmp_path, monkeypatch, field):
    ws = _workspace(tmp_path, monkeypatch, "redact")
    before = (ws / "intelligence" / "SIGNALS.md").read_bytes()
    assert _propose(**{field: CANARY}) == {"error": "redaction_identity_refused", "field": field}
    assert (ws / "intelligence" / "SIGNALS.md").read_bytes() == before
    assert all(CANARY.encode() not in p.read_bytes() for p in ws.rglob("*") if p.is_file())


@pytest.mark.parametrize("mode", [None, "flag"])
def test_non_rewriting_modes_preserve_metadata_and_reason(tmp_path, monkeypatch, mode):
    ws = _workspace(tmp_path, monkeypatch, mode)
    assert _propose(tags=CANARY, purpose=CANARY, rationale=CANARY)["written"] == 1
    record = _records(ws)[0]
    assert CANARY in str(record["Tags"])
    assert record["Purpose"] == CANARY
    assert record["Rationale"] == CANARY


@pytest.mark.parametrize("bad_config", [{"mode": "unknown"}, {"detectors": ["unregistered-detector"]}])
def test_bad_redaction_configuration_returns_a_structured_refusal(tmp_path, monkeypatch, bad_config):
    ws = _workspace(tmp_path, monkeypatch, "reject")
    cfg = ws / "mind-mem.json"
    data = json.loads(cfg.read_text())
    data["v4"]["redaction"].update(bad_config)
    cfg.write_text(json.dumps(data))
    before = (ws / "intelligence" / "SIGNALS.md").read_bytes()
    assert _propose()["error"] == "compliance_config_invalid"
    assert (ws / "intelligence" / "SIGNALS.md").read_bytes() == before
