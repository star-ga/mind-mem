"""Every accepted proposal type needs the caller's written rationale."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from fastmcp.server.auth import AccessToken

from mind_mem.init_workspace import init
from mind_mem.mcp.infra.workspace import use_workspace
from mind_mem.mcp.tools.governance import propose_update


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    init(str(tmp_path))
    monkeypatch.setenv("MIND_MEM_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("MIND_MEM_ADMIN_TOKEN", "ra4-admin-token-for-local-tests")
    return tmp_path


def _snapshot(workspace: Path) -> dict[str, bytes | None]:
    paths = ("intelligence/SIGNALS.md", "memory/evidence_chain.jsonl", "memory/evidence_chain.head")
    return {name: (workspace / name).read_bytes() if (workspace / name).exists() else None for name in paths}


def _call(workspace: Path, transport: str, block_type: str, rationale: str | None) -> dict:
    payload = {
        "block_type": block_type,
        "statement": "Add a rollback rehearsal to the PostgreSQL deployment checklist before the next release.",
    }
    if rationale is not None:
        payload["rationale"] = rationale
    if transport == "mcp":
        token = AccessToken(token="fixture", client_id="fixture", scopes=["admin"], claims={"sub": "reviewer"})
        with patch("mind_mem.mcp.infra.acl.get_access_token", return_value=token), use_workspace(str(workspace)):
            return json.loads(propose_update(**payload))
    from mind_mem.api.rest import create_app

    with TestClient(create_app(str(workspace))) as client:
        response = client.post("/v1/propose_update", json=payload, headers={"Authorization": "Bearer ra4-admin-token-for-local-tests"})
    assert response.status_code == 200
    return response.json()


@pytest.mark.parametrize("transport", ("mcp", "rest"))
@pytest.mark.parametrize("block_type", ("decision", "task"))
@pytest.mark.parametrize("rationale", (None, "", " \t\n", "1234567", "1 2 3 4 5 6 7", "\u20031234567\u2003"))
def test_invalid_rationale_refuses_before_screening_or_writes(workspace, monkeypatch, transport, block_type, rationale):
    import mind_mem.compliance.prewrite as prewrite

    before = _snapshot(workspace)

    def must_not_screen(*args, **kwargs):
        pytest.fail("a rationale-free proposal reached content screening")

    monkeypatch.setattr(prewrite, "screen", must_not_screen)
    result = _call(workspace, transport, block_type, rationale)
    assert "rationale" in result["error"]
    assert result["block_type"] == block_type
    assert result["rationale_length"] < 8
    assert _snapshot(workspace) == before


@pytest.mark.parametrize("transport", ("mcp", "rest"))
@pytest.mark.parametrize("block_type", ("decision", "task"))
def test_eight_non_whitespace_characters_are_preserved_in_proposal(workspace, transport, block_type):
    result = _call(workspace, transport, block_type, "  Fix risks  ")
    assert result["status"] == "proposed", result
    assert result["written"] == 1
    text = (workspace / "intelligence/SIGNALS.md").read_text(encoding="utf-8")
    assert "Rationale: Fix risks" in text


@pytest.mark.parametrize("block_type", ("decision", "task"))
def test_newline_rationale_cannot_inject_proposal_status(workspace, block_type):
    result = _call(workspace, "mcp", block_type, "Preserve audit reasons.\nStatus: applied")
    assert result["status"] == "proposed", result
    text = (workspace / "intelligence/SIGNALS.md").read_text(encoding="utf-8")
    assert "\\Status: applied" in text
    assert "\nStatus: applied" not in text
