# Copyright 2026 STARGA, Inc.
"""Configured detector failures refuse public proposals without leaking input."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from mind_mem.init_workspace import init
from mind_mem.mcp.infra.acl import bind_transport_auth
from mind_mem.mcp.tools.governance import propose_update

CANARY = "BROKEN-CANARY-REVIEW-ONLY"


def _workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str, failure: str) -> Path:
    module = f"e1_review_{mode}_{failure}"
    action = (
        'raise RuntimeError("provider detail: " + text)'
        if failure == "exception"
        else "return [Finding(start=0, end=len(text) + 1, detector=self.name, category=self.category)]"
    )
    source = (
        "from mind_mem.compliance.detectors import Detector, Finding, CATEGORY_SECRET\n"
        "class ReviewDetector(Detector):\n"
        "    name = 'review_runtime'\n"
        "    category = CATEGORY_SECRET\n"
        "    def scan(self, text):\n"
        f"        if {CANARY!r} in text:\n"
        f"            {action}\n"
        "        return []\n"
    )
    (tmp_path / f"{module}.py").write_text(source, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, module, raising=False)
    ws = tmp_path / "ws"
    init(str(ws))
    config = ws / "mind-mem.json"
    data = json.loads(config.read_text(encoding="utf-8"))
    data.setdefault("v4", {})["redaction"] = {"enabled": True, "mode": mode, "detectors": [], "plugins": [f"{module}:ReviewDetector"]}
    config.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setenv("MIND_MEM_WORKSPACE", str(ws))
    monkeypatch.setenv("MIND_MEM_CONFIG", str(config))
    monkeypatch.delenv("MIND_MEM_SCOPE", raising=False)
    monkeypatch.delenv("MIND_MEM_ACL_DISABLED", raising=False)
    return ws


def _body(statement: str) -> dict[str, str]:
    return {"block_type": "task", "statement": statement, "rationale": "Exercise actual configured detector refusal."}


def _assert_refused(ws: Path, before: bytes, response: dict) -> None:
    assert response["error"] == "compliance_detector_failed"
    assert "written" not in response
    assert CANARY not in json.dumps(response)
    assert (ws / "intelligence" / "SIGNALS.md").read_bytes() == before
    assert all(CANARY.encode() not in p.read_bytes() for p in ws.rglob("*") if p.is_file())


@pytest.mark.parametrize("mode", ["redact", "reject", "flag"])
@pytest.mark.parametrize("failure", ["exception", "malformed"])
def test_configured_detector_failure_is_a_typed_mcp_refusal(tmp_path, monkeypatch, mode, failure):
    ws = _workspace(tmp_path, monkeypatch, mode, failure)
    before = (ws / "intelligence" / "SIGNALS.md").read_bytes()
    with bind_transport_auth(principal="admin", scope="admin"):
        response = json.loads(propose_update(**_body(f"Do not persist {CANARY}.")))
        _assert_refused(ws, before, response)
        # Same detector/configuration must still admit valid input.
        admitted = json.loads(propose_update(**_body("Publish the reviewed roadmap controls.")))
    assert admitted["written"] == 1


@pytest.mark.parametrize("failure", ["exception", "malformed"])
def test_registered_rest_route_refuses_detector_failure_as_json(tmp_path, monkeypatch, failure):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from mind_mem.api.rest import create_app

    ws = _workspace(tmp_path, monkeypatch, "redact", failure)
    token = "fixture-e1-review-admin"
    monkeypatch.setenv("MIND_MEM_ADMIN_TOKEN", token)
    before = (ws / "intelligence" / "SIGNALS.md").read_bytes()
    with TestClient(create_app(str(ws)), raise_server_exceptions=False) as client:
        headers = {"Authorization": f"Bearer {token}"}
        response = client.post("/v1/propose_update", headers=headers, json=_body(f"Do not persist {CANARY}."))
        assert response.status_code == 200, response.text
        _assert_refused(ws, before, response.json())
        admitted = client.post("/v1/propose_update", headers=headers, json=_body("Publish the independently reviewed controls."))
        assert admitted.status_code == 200
        assert admitted.json()["written"] == 1
