# Copyright 2026 STARGA, Inc.
"""REST anonymous-local mode is an app-scoped result of bind validation."""

from __future__ import annotations

from typing import Any

import pytest

fastapi = pytest.importorskip("fastapi", reason="fastapi not installed")

from fastapi.testclient import TestClient  # noqa: E402

from mind_mem.api import rest  # noqa: E402

_AUTH_ENV = (
    "MIND_MEM_TOKEN",
    "MIND_MEM_ADMIN_TOKEN",
    "MIND_MEM_API_KEY_DB",
    "OIDC_ISSUER",
    "OIDC_AUDIENCE",
    "MIND_MEM_ALLOW_UNAUTHENTICATED_LOCALHOST",
    "MIND_MEM_BIND_HOST",
)


@pytest.fixture(autouse=True)
def clean_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _AUTH_ENV:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def workspace(tmp_path: Any) -> str:
    for name in ("decisions", "tasks", "entities", "intelligence", "memory"):
        (tmp_path / name).mkdir()
    return str(tmp_path)


def _metrics_status(application: Any) -> int:
    with TestClient(application, raise_server_exceptions=False) as client:
        return client.get("/v1/metrics").status_code


def test_direct_asgi_does_not_trust_environment_optin(
    workspace: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MIND_MEM_ALLOW_UNAUTHENTICATED_LOCALHOST", "1")
    monkeypatch.setenv("MIND_MEM_BIND_HOST", "0.0.0.0")
    assert _metrics_status(rest.create_app(workspace)) == 401


def test_arbitrary_truthy_factory_argument_is_not_a_capability(workspace: str) -> None:
    app = rest.create_app(workspace, _local_anonymous_capability=True)
    assert _metrics_status(app) == 401


def test_checked_loopback_capability_allows_local_app(workspace: str) -> None:
    capability = rest._enforce_fail_closed("127.0.0.1", True)
    app = rest.create_app(workspace, _local_anonymous_capability=capability)
    assert _metrics_status(app) != 401


def test_public_launcher_supplies_checked_capability(
    workspace: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[Any] = []
    monkeypatch.setattr("uvicorn.run", lambda application, **_kwargs: captured.append(application))
    rest.run(
        host="127.0.0.1",
        workspace=workspace,
        allow_unauthenticated_localhost=True,
    )
    assert len(captured) == 1
    assert _metrics_status(captured[0]) != 401


def test_public_launcher_refuses_routable_anonymous_bind(workspace: str) -> None:
    with pytest.raises(SystemExit):
        rest.run(
            host="0.0.0.0",  # noqa: S104 - the refusal is the assertion
            workspace=workspace,
            allow_unauthenticated_localhost=True,
        )


def test_empty_user_token_is_absent_and_direct_app_stays_closed(
    workspace: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MIND_MEM_TOKEN", "")
    monkeypatch.setenv("MIND_MEM_ALLOW_UNAUTHENTICATED_LOCALHOST", "1")
    assert rest._auth_is_configured() is False
    assert _metrics_status(rest.create_app(workspace)) == 401


def test_empty_user_token_allows_checked_loopback_mode(
    workspace: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MIND_MEM_TOKEN", "")
    capability = rest._enforce_fail_closed("127.0.0.1", True)
    app = rest.create_app(workspace, _local_anonymous_capability=capability)
    assert _metrics_status(app) != 401
