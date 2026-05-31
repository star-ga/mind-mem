"""v3.2.1 — OIDC JWTs must pass through ``_require_admin`` checks.

Pre-v3.2.1, ``_require_admin`` only recognised ``MIND_MEM_ADMIN_TOKEN``
matches and mmk_* keys with an explicit admin scope. An OIDC JWT —
even one carrying ``admin`` or ``mind-mem.admin`` scope — was silently
downgraded to user-tier because the admin path never consulted the
validated JWT claims.

These tests pin the fix: an OIDC JWT with an admin scope authenticates
as admin on admin-gated endpoints; one without is rejected with 403
when an admin gate is configured.
"""

from __future__ import annotations

from typing import Any, Generator
from unittest.mock import patch

import pytest

fastapi = pytest.importorskip("fastapi", reason="fastapi not installed")

from fastapi.testclient import TestClient  # noqa: E402

from mind_mem.api.rest import create_app  # noqa: E402


@pytest.fixture()
def workspace(tmp_path: Any) -> str:
    for subdir in ("decisions", "tasks", "entities", "intelligence", "memory"):
        (tmp_path / subdir).mkdir()
    return str(tmp_path)


@pytest.fixture()
def oidc_client(workspace: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> Generator[TestClient, None, None]:
    """TestClient with OIDC configured and no static admin token.

    The whole point of OIDC-only mode is: the admin gate must still
    work, driven purely by JWT scopes.
    """
    monkeypatch.setenv("MIND_MEM_WORKSPACE", workspace)
    monkeypatch.delenv("MIND_MEM_TOKEN", raising=False)
    monkeypatch.delenv("MIND_MEM_ADMIN_TOKEN", raising=False)
    monkeypatch.setenv("OIDC_ISSUER", "https://idp.example.com")
    monkeypatch.setenv("OIDC_CLIENT_ID", "mm-client")
    monkeypatch.setenv("OIDC_AUDIENCE", "mind-mem-api")
    app = create_app(workspace)
    with TestClient(app, raise_server_exceptions=False) as tc:
        yield tc


def _mock_jwt(claims: dict) -> Any:
    """Context manager that makes OIDCProvider.verify return *claims*."""
    return patch("mind_mem.api.auth.jwt.decode", return_value=claims)


def _mock_jwks() -> Any:
    return patch("mind_mem.api.auth.OIDCProvider._get_jwks", return_value={"keys": []})


# Valid JWT format (three dot-separated parts) — the actual decoding is mocked.
_JWT = "header.payload.signature"


class TestOIDCAdminEnforcement:
    def test_oidc_admin_scope_passes_admin_gate(self, oidc_client: TestClient) -> None:
        """OIDC JWT with ``admin`` scope authenticates on admin endpoints."""
        claims = {"sub": "user-admin", "scope": "openid mind-mem.admin"}
        with _mock_jwks(), _mock_jwt(claims):
            resp = oidc_client.get(
                "/v1/admin/api_keys",
                headers={"Authorization": f"Bearer {_JWT}"},
            )
        # 501 is acceptable (no key store configured), 200 is also fine.
        # What matters is *not* 401/403 — admin gate passed.
        assert resp.status_code in {200, 501}

    def test_oidc_user_scope_rejected_from_admin_gate(self, oidc_client: TestClient) -> None:
        """OIDC JWT without admin scope gets 403 on admin endpoints."""
        claims = {"sub": "user-regular", "scope": "openid email"}
        with _mock_jwks(), _mock_jwt(claims):
            resp = oidc_client.get(
                "/v1/admin/api_keys",
                headers={"Authorization": f"Bearer {_JWT}"},
            )
        assert resp.status_code == 403

    def test_oidc_token_passes_user_gate(self, oidc_client: TestClient) -> None:
        """OIDC JWT (any valid) authenticates on user-tier endpoints."""
        claims = {"sub": "user-regular", "scope": "openid email"}
        with _mock_jwks(), _mock_jwt(claims):
            resp = oidc_client.post(
                "/v1/recall",
                headers={"Authorization": f"Bearer {_JWT}"},
                json={"query": "test", "limit": 5},
            )
        # User-tier is open to any validated JWT.
        assert resp.status_code == 200

    def test_invalid_oidc_token_rejected_as_401(self, oidc_client: TestClient) -> None:
        """A JWT that fails signature/audience validation → 401."""
        from mind_mem.api.auth import AuthError

        with (
            _mock_jwks(),
            patch(
                "mind_mem.api.auth.OIDCProvider.verify",
                side_effect=AuthError("token expired"),
            ),
        ):
            resp = oidc_client.post(
                "/v1/recall",
                headers={"Authorization": f"Bearer {_JWT}"},
                json={"query": "test"},
            )
        assert resp.status_code == 401

    def test_admin_scope_names_env_override(self, oidc_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        """``MIND_MEM_OIDC_ADMIN_SCOPES`` env var changes which scopes grant admin."""
        monkeypatch.setenv("MIND_MEM_OIDC_ADMIN_SCOPES", "my-custom-admin")
        # JWT with just the new custom scope should pass admin gate.
        claims = {"sub": "user-admin", "scope": "openid my-custom-admin"}
        with _mock_jwks(), _mock_jwt(claims):
            resp = oidc_client.get(
                "/v1/admin/api_keys",
                headers={"Authorization": f"Bearer {_JWT}"},
            )
        assert resp.status_code in {200, 501}

        # JWT with the *default* admin scope should now be rejected (since
        # we overrode the scope list).
        claims = {"sub": "user-not-admin", "scope": "openid admin mind-mem.admin"}
        with _mock_jwks(), _mock_jwt(claims):
            resp = oidc_client.get(
                "/v1/admin/api_keys",
                headers={"Authorization": f"Bearer {_JWT}"},
            )
        assert resp.status_code == 403

    def test_no_oidc_config_skips_oidc_path(self, workspace: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """When OIDC env is unset, bearer tokens are not probed as JWTs.

        Important: the OIDC verify path is expensive (JWKS fetch).
        When OIDC isn't configured we must skip it for every request.
        """
        monkeypatch.setenv("MIND_MEM_WORKSPACE", workspace)
        monkeypatch.setenv("MIND_MEM_TOKEN", "plain-user-token-not-a-jwt")
        monkeypatch.delenv("OIDC_ISSUER", raising=False)
        monkeypatch.delenv("OIDC_AUDIENCE", raising=False)

        app = create_app(workspace)
        with TestClient(app, raise_server_exceptions=False) as client:
            # OIDCProvider.verify should NEVER be called — if it were,
            # it would fail (no env set) but we assert it's not invoked.
            with patch("mind_mem.api.auth.OIDCProvider.verify") as mock_verify:
                resp = client.post(
                    "/v1/recall",
                    headers={"Authorization": "Bearer plain-user-token-not-a-jwt"},
                    json={"query": "test"},
                )
            assert resp.status_code == 200
            mock_verify.assert_not_called()
