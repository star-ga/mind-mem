# Copyright 2026 STARGA, Inc.
"""SDK fixture shapes checked against the real authenticated REST handlers.

Client tests consume these envelopes; this gate prevents both clients and
their mock server from agreeing on a response the product never serves.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from mind_mem.api.rest import create_app  # noqa: E402
from mind_mem.init_workspace import init  # noqa: E402
from mind_mem.mcp.infra.workspace import use_workspace  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _shape_contains(actual, fixture, path="response") -> None:
    assert type(actual) is type(fixture), f"{path}: {type(actual).__name__} != {type(fixture).__name__}"
    if isinstance(fixture, dict):
        for key, value in fixture.items():
            assert key in actual, f"{path}.{key} disappeared"
            _shape_contains(actual[key], value, path + "." + key)
    elif isinstance(fixture, list):
        assert len(actual) == len(fixture), f"{path}: fixture case cardinality changed"
        for index, (item, expected) in enumerate(zip(actual, fixture)):
            _shape_contains(item, expected, f"{path}[{index}]")


def test_sdk_envelopes_are_produced_by_current_rest(tmp_path: Path, monkeypatch) -> None:
    init(str(tmp_path))
    (tmp_path / "decisions/DECISIONS.md").write_text(
        "[D-20260914-001]\nDate: 2026-09-14\nStatus: active\nStatement: Orchid is the SDK contract sentinel.\n\n"
    )
    token = "fixture-sdk-contract-token"
    monkeypatch.setenv("MIND_MEM_TOKEN", token)
    monkeypatch.setenv("MIND_MEM_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("MIND_MEM_CONFIG", str(tmp_path / "mind-mem.json"))
    monkeypatch.setenv("MIND_MEM_SCOPE", "user")
    requests = (
        ("recall", "POST", "/v1/recall", {"query": "orchid", "backend": "bm25", "scoring_instant": "2026-09-14"}),
        ("block", "GET", "/v1/block/D-20260914-001", None),
        ("health", "GET", "/v1/health", None),
        ("contradictions", "GET", "/v1/contradictions", None),
        ("scan", "GET", "/v1/scan", None),
    )
    with use_workspace(str(tmp_path)), TestClient(create_app(str(tmp_path))) as client:
        for name, method, path, body in requests:
            response = client.request(method, path, headers={"Authorization": "Bearer " + token}, json=body)
            assert response.status_code == 200, response.text
            actual = response.json()
            fixture = json.loads((ROOT / "sdk/spec/fixtures" / f"{name}.json").read_text())
            _shape_contains(actual, fixture, name)
            if name == "recall":
                assert actual["count"] == 1
                assert actual["results"][0]["_id"] == fixture["results"][0]["_id"]
                assert actual["results"][0]["excerpt"] == fixture["results"][0]["excerpt"]
            elif name == "block":
                assert actual["found"] and actual["block"]["Statement"] == fixture["block"]["Statement"]


def test_go_module_carries_the_same_contract_fixtures() -> None:
    fixtures = list((ROOT / "sdk/spec/fixtures").glob("*.json"))
    assert {path.stem for path in fixtures} == {"recall", "block", "health", "contradictions", "scan"}
    for path in fixtures:
        assert path.read_bytes() == (ROOT / "sdk/go/testdata/contract" / path.name).read_bytes()


def test_contract_gate_rejects_the_old_nested_recall_shape() -> None:
    fixture = json.loads((ROOT / "sdk/spec/fixtures/recall.json").read_text())
    old = dict(fixture)
    old["results"] = [{"block": {"id": "D-20260914-001", "content": "Orchid"}, "score": 1.0, "rank": 1}]
    with pytest.raises(AssertionError, match="_id disappeared"):
        _shape_contains(old, fixture)
