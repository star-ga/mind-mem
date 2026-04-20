"""Tests for OIDC callback + admin API key endpoints (v3.2.0)."""

from __future__ import annotations

from typing import Any, Generator
from unittest.mock import patch

import pytest

fastapi = pytest.importorskip("fastapi", reason="fastapi not installed")

from fastapi.testclient import TestClient  # noqa: E402

from mind_mem.api.rest import create_app  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures (mirror test_rest_api.py style)
# ---------------------------------------------------------------------------

ADMIN_TOKEN = "test-admin-token-for-mind-mem-rest-api-oidc"
USER_TOKEN = "test-user-token-for-mind-mem-rest-api-oidc"
_ADMIN_HEADER = {"Authorization": f"Bearer {ADMIN_TOKEN}"}
_USER_HEADER = {"Authorization": f"Bearer {USER_TOKEN}"}


@pytest.fixture()
def workspace(tmp_path: Any) -> str:
    for subdir in ("decisions", "tasks", "entities", "intelligence", "memory"):
        (tmp_path / subdir).mkdir()
    return str(tmp_path)


@pytest.fixture()
def admin_client(workspace: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> Generator[TestClient, None, None]:
    """TestClient with admin token + API key store configured."""
    db_path = str(tmp_path / "keys.db")
    monkeypatch.setenv("MIND_MEM_TOKEN", USER_TOKEN)
    monkeypatch.setenv("MIND_MEM_ADMIN_TOKEN", ADMIN_TOKEN)
    monkeypatch.setenv("MIND_MEM_WORKSPACE", workspace)
    monkeypatch.setenv("MIND_MEM_API_KEY_DB", db_path)
    monkeypatch.setenv("MIND_MEM_ENV", "production")
    app = create_app(workspace)
    with TestClient(app, raise_server_exceptions=False) as tc:
        yield tc


@pytest.fixture()
def user_client(workspace: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> Generator[TestClient, None, None]:
    """TestClient with user token only (no admin), API key store configured."""
    db_path = str(tmp_path / "keys_user.db")
    monkeypatch.setenv("MIND_MEM_TOKEN", USER_TOKEN)
    monkeypatch.setenv("MIND_MEM_ADMIN_TOKEN", ADMIN_TOKEN)
    monkeypatch.setenv("MIND_MEM_WORKSPACE", workspace)
    monkeypatch.setenv("MIND_MEM_API_KEY_DB", db_path)
    app = create_app(workspace)
    with TestClient(app, raise_server_exceptions=False) as tc:
        yield tc


@pytest.fixture()
def no_store_client(workspace: str, monkeypatch: pytest.MonkeyPatch) -> Generator[TestClient, None, None]:
    """TestClient with admin token but NO API key store configured."""
    monkeypatch.setenv("MIND_MEM_TOKEN", USER_TOKEN)
    monkeypatch.setenv("MIND_MEM_ADMIN_TOKEN", ADMIN_TOKEN)
    monkeypatch.setenv("MIND_MEM_WORKSPACE", workspace)
    monkeypatch.delenv("MIND_MEM_API_KEY_DB", raising=False)
    app = create_app(workspace)
    with TestClient(app, raise_server_exceptions=False) as tc:
        yield tc


# ---------------------------------------------------------------------------
# OIDC callback
# ---------------------------------------------------------------------------


class TestOIDCCallback:
    def test_oidc_not_configured_returns_501(self, admin_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OIDC_ISSUER", raising=False)
        resp = admin_client.post(
            "/v1/auth/oidc/callback",
            headers={"Authorization": "Bearer sometoken"},
        )
        assert resp.status_code == 501

    def test_oidc_missing_token_returns_401(self, admin_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OIDC_ISSUER", "https://example.com")
        monkeypatch.setenv("OIDC_CLIENT_ID", "cid")
        monkeypatch.setenv("OIDC_AUDIENCE", "aud")
        resp = admin_client.post("/v1/auth/oidc/callback")
        assert resp.status_code == 401

    def test_oidc_invalid_token_returns_401(self, admin_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OIDC_ISSUER", "https://example.com")
        monkeypatch.setenv("OIDC_CLIENT_ID", "cid")
        monkeypatch.setenv("OIDC_AUDIENCE", "aud")
        from mind_mem.api.auth import AuthError

        with patch("mind_mem.api.auth.OIDCProvider.verify", side_effect=AuthError("bad token")):
            resp = admin_client.post(
                "/v1/auth/oidc/callback",
                headers={"Authorization": "Bearer invalid.token.here"},
            )
        assert resp.status_code == 401

    def test_oidc_valid_token_returns_authenticated(self, admin_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OIDC_ISSUER", "https://example.com")
        monkeypatch.setenv("OIDC_CLIENT_ID", "cid")
        monkeypatch.setenv("OIDC_AUDIENCE", "aud")
        fake_claims = {"sub": "user-abc", "scope": "openid email"}

        with patch("mind_mem.api.auth.OIDCProvider._get_jwks", return_value={"keys": []}):
            with patch("mind_mem.api.auth.jwt.decode", return_value=fake_claims):
                resp = admin_client.post(
                    "/v1/auth/oidc/callback",
                    headers={"Authorization": "Bearer valid.jwt.token"},
                )
        assert resp.status_code == 200
        data = resp.json()
        assert data["authenticated"] is True
        assert data["agent_id"] == "user-abc"
        assert "openid" in data["scopes"]


# ---------------------------------------------------------------------------
# Admin API key management — end-to-end
# ---------------------------------------------------------------------------


class TestAdminAPIKeysEndToEnd:
    def test_create_key_returns_201_with_raw_key(self, admin_client: TestClient) -> None:
        resp = admin_client.post(
            "/v1/admin/api_keys",
            json={"agent_id": "bot-1", "scopes": ["user"], "expires_in_days": 30},
            headers=_ADMIN_HEADER,
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["key"].startswith("mmk_live_")
        assert data["agent_id"] == "bot-1"

    def test_list_keys_after_create(self, admin_client: TestClient) -> None:
        admin_client.post(
            "/v1/admin/api_keys",
            json={"agent_id": "bot-list", "scopes": ["user"]},
            headers=_ADMIN_HEADER,
        )
        resp = admin_client.get("/v1/admin/api_keys", headers=_ADMIN_HEADER)
        assert resp.status_code == 200
        keys = resp.json()["keys"]
        assert any(k["agent_id"] == "bot-list" for k in keys)

    def test_list_keys_filtered_by_agent(self, admin_client: TestClient) -> None:
        admin_client.post(
            "/v1/admin/api_keys",
            json={"agent_id": "filter-bot", "scopes": ["user"]},
            headers=_ADMIN_HEADER,
        )
        resp = admin_client.get(
            "/v1/admin/api_keys",
            params={"agent_id": "filter-bot"},
            headers=_ADMIN_HEADER,
        )
        assert resp.status_code == 200
        keys = resp.json()["keys"]
        assert all(k["agent_id"] == "filter-bot" for k in keys)

    def test_revoke_key_returns_200(self, admin_client: TestClient) -> None:
        admin_client.post(
            "/v1/admin/api_keys",
            json={"agent_id": "revoke-bot", "scopes": ["user"]},
            headers=_ADMIN_HEADER,
        )
        list_resp = admin_client.get("/v1/admin/api_keys", headers=_ADMIN_HEADER)
        key_id = next(
            k["key_id"]
            for k in list_resp.json()["keys"]
            if k["agent_id"] == "revoke-bot"
        )
        resp = admin_client.delete(f"/v1/admin/api_keys/{key_id}", headers=_ADMIN_HEADER)
        assert resp.status_code == 200
        assert resp.json()["revoked"] is True

    def test_revoke_nonexistent_key_returns_404(self, admin_client: TestClient) -> None:
        resp = admin_client.delete("/v1/admin/api_keys/no-such-key", headers=_ADMIN_HEADER)
        assert resp.status_code == 404

    def test_rotate_key_returns_new_key(self, admin_client: TestClient) -> None:
        create_resp = admin_client.post(
            "/v1/admin/api_keys",
            json={"agent_id": "rotate-bot", "scopes": ["user"]},
            headers=_ADMIN_HEADER,
        )
        old_key = create_resp.json()["key"]
        list_resp = admin_client.get("/v1/admin/api_keys", headers=_ADMIN_HEADER)
        key_id = next(
            k["key_id"]
            for k in list_resp.json()["keys"]
            if k["agent_id"] == "rotate-bot"
        )
        rotate_resp = admin_client.post(
            f"/v1/admin/api_keys/{key_id}/rotate",
            headers=_ADMIN_HEADER,
        )
        assert rotate_resp.status_code == 200
        new_key = rotate_resp.json()["key"]
        assert new_key != old_key
        assert new_key.startswith("mmk_live_")

    def test_rotate_nonexistent_returns_404(self, admin_client: TestClient) -> None:
        resp = admin_client.post(
            "/v1/admin/api_keys/ghost-id/rotate",
            headers=_ADMIN_HEADER,
        )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Admin scope enforcement
# ---------------------------------------------------------------------------


class TestAdminScopeEnforcement:
    def test_create_key_with_user_token_returns_403(self, user_client: TestClient) -> None:
        resp = user_client.post(
            "/v1/admin/api_keys",
            json={"agent_id": "bad-actor", "scopes": ["user"]},
            headers=_USER_HEADER,
        )
        assert resp.status_code == 403

    def test_list_keys_with_user_token_returns_403(self, user_client: TestClient) -> None:
        resp = user_client.get("/v1/admin/api_keys", headers=_USER_HEADER)
        assert resp.status_code == 403

    def test_revoke_key_with_user_token_returns_403(self, user_client: TestClient) -> None:
        resp = user_client.delete("/v1/admin/api_keys/any-id", headers=_USER_HEADER)
        assert resp.status_code == 403

    def test_rotate_key_with_user_token_returns_403(self, user_client: TestClient) -> None:
        resp = user_client.post("/v1/admin/api_keys/any-id/rotate", headers=_USER_HEADER)
        assert resp.status_code == 403


# ---------------------------------------------------------------------------
# Store not configured
# ---------------------------------------------------------------------------


class TestStoreNotConfigured:
    def test_create_returns_501_without_store(self, no_store_client: TestClient) -> None:
        resp = no_store_client.post(
            "/v1/admin/api_keys",
            json={"agent_id": "bot", "scopes": ["user"]},
            headers=_ADMIN_HEADER,
        )
        assert resp.status_code == 501

    def test_list_returns_501_without_store(self, no_store_client: TestClient) -> None:
        resp = no_store_client.get("/v1/admin/api_keys", headers=_ADMIN_HEADER)
        assert resp.status_code == 501
