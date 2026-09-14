"""Real REST -> observed-MCP admin scope handoff controls (RA.4).

The endpoint is exercised through FastAPI's registered route.  Fixture token
values are deliberately local and are never included in assertions or output.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from mind_mem.api.rest import create_app  # noqa: E402
from mind_mem.init_workspace import init  # noqa: E402
from mind_mem.mcp.infra.acl import bind_transport_auth, enforce_capability_acl  # noqa: E402

_ADMIN = "fixture-ra4-admin"
_USER = "fixture-ra4-user"


def _proposal(label: str) -> dict[str, Any]:
    return {
        "block_type": "decision",
        "statement": f"REST scope handoff fixture proposal {label}.",
        "rationale": "A sufficiently long rationale for the REST scope fixture.",
        "confidence": "high",
    }


@pytest.fixture
def configured_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    init(str(tmp_path))
    monkeypatch.setenv("MIND_MEM_ADMIN_TOKEN", _ADMIN)
    monkeypatch.setenv("MIND_MEM_TOKEN", _USER)
    monkeypatch.delenv("MIND_MEM_SCOPE", raising=False)
    monkeypatch.delenv("MIND_MEM_ACL_DISABLED", raising=False)
    return str(tmp_path)


def _post(app: Any, body: dict[str, Any], token: str | None, **headers: str) -> Any:
    request_headers = {"Authorization": f"Bearer {token}"} if token else {}
    request_headers.update(headers)
    with TestClient(app, raise_server_exceptions=False) as client:
        return client.post("/v1/propose_update", json=body, headers=request_headers)


def test_registered_admin_route_uses_verified_bearer_without_ambient_scope(
    configured_workspace: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An ambient user setting must not deny a separately authenticated admin
    # request; the trusted request snapshot owns the decision.
    monkeypatch.setenv("MIND_MEM_SCOPE", "user")
    app = create_app(configured_workspace)
    response = _post(app, _proposal("no-ambient"), _ADMIN)
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["written"] == 1, response.text
    signals = Path(configured_workspace, "intelligence", "SIGNALS.md").read_text(encoding="utf-8")
    assert "ActorId: admin" in signals


def test_user_and_invalid_authorities_are_refused_before_write(configured_workspace: str) -> None:
    app = create_app(configured_workspace)
    before = Path(configured_workspace, "intelligence", "SIGNALS.md").read_bytes()
    user = _post(app, _proposal("user"), _USER)
    invalid = _post(app, _proposal("invalid"), "fixture-ra4-invalid")
    assert user.status_code == 403
    assert invalid.status_code == 401
    assert Path(configured_workspace, "intelligence", "SIGNALS.md").read_bytes() == before


def test_claimed_actor_header_cannot_change_authenticated_identity_or_scope(configured_workspace: str) -> None:
    app = create_app(configured_workspace)
    response = _post(app, _proposal("spoof"), _ADMIN, **{"X-MindMem-Actor": "fixture-eve", "X-MindMem-Purpose": "fixture-purpose"})
    assert response.status_code == 200, response.text
    assert response.json()["written"] == 1
    signals = Path(configured_workspace, "intelligence", "SIGNALS.md").read_text(encoding="utf-8")
    assert "ActorId: admin" in signals
    assert "fixture-eve" not in signals
    assert "Purpose: fixture-purpose" in signals


def test_admin_user_admin_sequence_does_not_leak_scope(configured_workspace: str) -> None:
    app = create_app(configured_workspace)
    first = _post(app, _proposal("first-admin"), _ADMIN)
    denied = _post(app, _proposal("middle-user"), _USER)
    second = _post(app, _proposal("second-admin"), _ADMIN)
    assert first.status_code == second.status_code == 200
    assert first.json()["written"] == second.json()["written"] == 1
    assert denied.status_code == 403


def test_concurrent_registered_admin_requests_keep_their_verified_scope(configured_workspace: str) -> None:
    app = create_app(configured_workspace)

    def invoke(index: int) -> tuple[int, int]:
        response = _post(app, _proposal(f"parallel-{index}"), _ADMIN)
        return response.status_code, int(response.json().get("written", 0))

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(invoke, range(4)))
    assert results == [(200, 1)] * 4


def test_conflicting_nested_transport_decision_fails_closed() -> None:
    with bind_transport_auth(principal="fixture-admin", scope="admin"):
        with bind_transport_auth(principal="fixture-other", scope="user") as conflict:
            assert conflict.status == "denied"
            denied = enforce_capability_acl("propose_update")
            assert denied is not None
            assert "authentication context unavailable" in denied
