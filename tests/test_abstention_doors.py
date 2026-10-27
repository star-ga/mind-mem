# Copyright 2026 STARGA, Inc.
"""Abstention on the doors beyond the MCP handler: CLI, stdlib HTTP, REST.

``mm recall`` and the stdlib ``POST /query`` route reach the engine through the
serving entry (``mind_mem.recall.recall``), not through ``_recall_impl``; the
FastAPI ``POST /v1/recall`` route delegates to ``_recall_impl``. A verdict that
only one of those doors could produce would make "the response tells you
whether any of it is a clear match" a property of a handler instead of the
product -- the exact trap the attestation work already fell into once.

The shape contract on the two non-envelope doors is the strict part. The CLI
prints a bare JSON LIST and the stdlib route a fixed payload; neither may change
unless the caller asked, so a script that parses them today keeps working.
"""

from __future__ import annotations

import json
import os
import sqlite3
from typing import Any

import pytest

from tests._abstention_fixture import make_workspace, preload_serving_stack, queries

preload_serving_stack()  # before any capsys test can bind a logger to a per-test stream


@pytest.fixture()
def ws(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> str:
    path = make_workspace(str(tmp_path))
    monkeypatch.setenv("MIND_MEM_WORKSPACE", path)
    monkeypatch.setenv("MIND_MEM_CONFIG", os.path.join(path, "mind-mem.json"))
    return path


def _present_query() -> str:
    return queries()[0][0]


def _rows(ws: str) -> list[tuple[Any, ...]]:
    conn = sqlite3.connect(os.path.join(ws, ".mind-mem-index", "recall.db"))
    try:
        try:
            return conn.execute("SELECT run_id, intended_use, band FROM recall_abstention").fetchall()
        except sqlite3.OperationalError:
            return []
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli(capsys: pytest.CaptureFixture[str], *argv: str) -> Any:
    from mind_mem import mm_cli

    code = mm_cli.main(["recall", *argv])
    out = capsys.readouterr().out
    assert code == 0, out
    return json.loads(out)


def test_cli_without_the_flag_prints_the_same_bare_list(ws: str, capsys: pytest.CaptureFixture[str]) -> None:
    printed = _cli(capsys, _present_query(), "--limit", "5")
    assert isinstance(printed, list) and printed, "pre-feature shape is a bare non-empty list"
    assert {"_id", "score"} <= set(printed[0])
    assert _rows(ws) == []  # nothing was recorded either


def test_cli_with_intended_use_prints_results_and_the_verdict(ws: str, capsys: pytest.CaptureFixture[str]) -> None:
    payload = _cli(capsys, _present_query(), "--limit", "5", "--intended-use", "governed_write")
    assert isinstance(payload, dict)
    assert payload["count"] == len(payload["results"]) > 0
    verdict = payload["abstention"]
    assert verdict["intended_use"] == "governed_write"
    assert verdict["band"] in ("act", "proceed_with_caution", "do_not_act")
    assert verdict["candidates"] == len(payload["results"])
    assert verdict["thresholds"]["act"] == 0.70
    # The verdict describes the SAME list the flag-less call prints.
    plain = _cli(capsys, _present_query(), "--limit", "5")
    assert [h["_id"] for h in payload["results"]] == [h["_id"] for h in plain]


def test_cli_verdict_is_recorded_against_the_published_run_identity(ws: str, capsys: pytest.CaptureFixture[str]) -> None:
    payload = _cli(capsys, _present_query(), "--limit", "5", "--intended-use", "answer")
    run_id = payload["attestation"]["query_id"]
    assert len(run_id) == 64
    assert _rows(ws) == [(run_id, "answer", payload["abstention"]["band"])]


def test_cli_rejects_an_unknown_use_at_parse_time(ws: str, capsys: pytest.CaptureFixture[str]) -> None:
    from mind_mem import mm_cli

    with pytest.raises(SystemExit) as exit_info:
        mm_cli.main(["recall", "anything", "--intended-use", "yolo"])
    assert exit_info.value.code == 2
    assert "summary" in capsys.readouterr().err  # the valid choices are named


def test_cli_use_flag_is_exclusive_with_the_other_output_modes(ws: str) -> None:
    from mind_mem import mm_cli

    for other in (["--receipt-envelope"], ["--kernel", "default"]):
        with pytest.raises(SystemExit) as exit_info:
            mm_cli.main(["recall", "anything", "--intended-use", "answer", *other])
        assert exit_info.value.code == 2


def test_python_callers_can_judge_the_library_result_directly(ws: str) -> None:
    """The library door returns a list of hit dicts; the pure core takes it as is."""
    from mind_mem.recall import recall
    from mind_mem.recall_abstention import assess

    hits = recall(ws, _present_query(), limit=5)
    verdict = assess(hits, intended_use="summary")
    assert verdict["candidates"] == len(hits) > 0
    assert verdict["band"] in ("act", "proceed_with_caution", "do_not_act")
    assert _rows(ws) == []  # a pure judgement records nothing


# ---------------------------------------------------------------------------
# stdlib HTTP transport
# ---------------------------------------------------------------------------


def test_stdlib_query_route_is_unchanged_without_a_request(ws: str) -> None:
    from mind_mem import http_transport

    status, payload = http_transport._handle_query(ws, {"query": _present_query(), "limit": 5})
    assert status == 200
    assert set(payload) == {"query", "results", "count", "attestation"}
    assert _rows(ws) == []


def test_stdlib_query_route_returns_the_verdict_when_asked(ws: str) -> None:
    from mind_mem import http_transport

    status, payload = http_transport._handle_query(ws, {"query": _present_query(), "limit": 5, "intended_use": "summary"})
    assert status == 200
    assert payload["abstention"]["intended_use"] == "summary"
    assert payload["abstention"]["candidates"] == payload["count"]
    run_id = payload["attestation"]["query_id"]
    assert _rows(ws) == [(run_id, "summary", payload["abstention"]["band"])]


def test_stdlib_query_route_refuses_a_bad_use(ws: str) -> None:
    from mind_mem import http_transport

    status, payload = http_transport._handle_query(ws, {"query": _present_query(), "intended_use": "yolo"})
    assert status == 400
    assert "intended_use" in payload["error"] and "governed_write" in payload["error"]
    status, payload = http_transport._handle_query(ws, {"query": _present_query(), "intended_use": 7})
    assert status == 400


def test_stdlib_query_route_ignores_the_workspace_flag_without_a_request(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """The route's payload is a fixed shape clients already parse, and honouring
    the workspace-wide flag would cost a config stat on every request that never
    asked. The flag governs the envelope doors; this route is per-request."""
    from mind_mem import http_transport

    path = make_workspace(str(tmp_path), config={"abstention": {"enabled": True, "bands": {"summary": {"act": 0.3, "caution": 0.05}}}})
    monkeypatch.setenv("MIND_MEM_WORKSPACE", path)
    status, payload = http_transport._handle_query(path, {"query": _present_query(), "limit": 5})
    assert status == 200
    assert "abstention" not in payload
    # Positive control: the same workspace DOES return a verdict when asked, and
    # it applies the workspace's configured thresholds.
    status, asked = http_transport._handle_query(path, {"query": _present_query(), "limit": 5, "intended_use": "summary"})
    assert status == 200
    assert asked["abstention"]["thresholds"] == {"act": 0.3, "caution": 0.05}


# ---------------------------------------------------------------------------
# gRPC (grpcio-free handler)
# ---------------------------------------------------------------------------


def test_grpc_recall_honours_only_the_workspace_switch(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """gRPC has no ``intended_use`` field in its message schema, so the workspace
    switch is the only way a verdict reaches it. Off, the payload is the one it
    always was; on, it carries the configured default use's verdict."""
    from mind_mem.api.grpc_server import RecallRequest, handle_recall

    off = make_workspace(str(tmp_path / "off"))
    on = make_workspace(str(tmp_path / "on"), config={"abstention": {"enabled": True, "default_use": "governed_write"}})
    request = RecallRequest(query=_present_query(), limit=5)

    monkeypatch.setenv("MIND_MEM_WORKSPACE", off)
    monkeypatch.setenv("MIND_MEM_CONFIG", os.path.join(off, "mind-mem.json"))
    plain = json.loads(handle_recall(request).payload)
    assert plain["results"], "positive control: the door served hits"
    assert "abstention" not in plain

    monkeypatch.setenv("MIND_MEM_WORKSPACE", on)
    monkeypatch.setenv("MIND_MEM_CONFIG", os.path.join(on, "mind-mem.json"))
    asked = json.loads(handle_recall(request).payload)
    assert asked["abstention"]["intended_use"] == "governed_write"
    assert [h["_id"] for h in asked["results"]] == [h["_id"] for h in plain["results"]]
    assert _rows(on), "the verdict is recorded so report_outcome can score it"
    assert not _rows(off)


# ---------------------------------------------------------------------------
# FastAPI REST
# ---------------------------------------------------------------------------

fastapi = pytest.importorskip("fastapi", reason="fastapi not installed")
from fastapi.testclient import TestClient  # noqa: E402

from mind_mem.api import rest  # noqa: E402


@pytest.fixture()
def client(ws: str, monkeypatch: pytest.MonkeyPatch) -> Any:
    for name in ("MIND_MEM_API_KEY_DB", "MIND_MEM_TOKEN", "MIND_MEM_ADMIN_TOKEN", "OIDC_ISSUER", "OIDC_AUDIENCE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MIND_MEM_ALLOW_UNAUTHENTICATED_LOCALHOST", "1")
    capability = rest._enforce_fail_closed("127.0.0.1", True)
    app = rest.create_app(ws, _local_anonymous_capability=capability)
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def test_rest_recall_accepts_and_returns_the_verdict(client: Any) -> None:
    plain = client.post("/v1/recall", json={"query": _present_query(), "limit": 5})
    assert plain.status_code == 200, plain.text
    assert "abstention" not in plain.json()  # positive control: the route works and is silent by default
    asked = client.post("/v1/recall", json={"query": _present_query(), "limit": 5, "intended_use": "governed_write"})
    assert asked.status_code == 200, asked.text
    assert asked.json()["abstention"]["intended_use"] == "governed_write"
    assert [h["_id"] for h in asked.json()["results"]] == [h["_id"] for h in plain.json()["results"]]


def test_rest_recall_rejects_an_unknown_use(client: Any) -> None:
    resp = client.post("/v1/recall", json={"query": _present_query(), "intended_use": "yolo"})
    assert resp.status_code in (400, 422)
    assert "intended_use" in resp.text
