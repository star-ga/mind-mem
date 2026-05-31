"""Tests for OIDCProvider / OIDCConfig in src/mind_mem/api/auth.py."""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("jose", reason="python-jose not installed; skipping OIDC tests")
pytest.importorskip("httpx", reason="httpx not installed; skipping OIDC tests")

from jose import jwt  # noqa: E402

from mind_mem.api.auth import AuthError, OIDCConfig, OIDCProvider  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ISSUER = "https://test.example.com"
_AUDIENCE = "mind-mem-api"
_CLIENT_ID = "test-client"

# Minimal RSA-like test key generated with python-jose for tests only.
# We use HS256 (symmetric) so tests stay dependency-free and fast;
# the OIDCProvider algorithm list is patched to allow HS256 in tests.
_SECRET = "test-secret-not-used-in-production"


def _make_token(
    issuer: str = _ISSUER,
    audience: str = _AUDIENCE,
    sub: str = "user-123",
    exp_offset: int = 3600,
    extra: dict | None = None,
) -> str:
    """Build a HS256 JWT for test use."""
    payload: dict[str, Any] = {
        "iss": issuer,
        "aud": audience,
        "sub": sub,
        "iat": int(time.time()),
        "exp": int(time.time()) + exp_offset,
    }
    if extra:
        payload.update(extra)
    return jwt.encode(payload, _SECRET, algorithm="HS256")


def _provider(issuer: str = _ISSUER, audience: str = _AUDIENCE) -> OIDCProvider:
    """Return an OIDCProvider that never hits the network (JWKS mocked)."""
    config = OIDCConfig(
        issuer=issuer,
        client_id=_CLIENT_ID,
        client_secret="secret",
        audience=audience,
    )
    provider = OIDCProvider(config)
    # Inject a fake JWKS dict; the actual key material is irrelevant because
    # we patch jwt.decode in tests that need successful verification.
    provider._jwks = {"keys": []}
    return provider


# ---------------------------------------------------------------------------
# OIDCConfig
# ---------------------------------------------------------------------------


class TestOIDCConfig:
    def test_jwks_uri_appends_well_known(self) -> None:
        config = OIDCConfig(
            issuer="https://login.example.com",
            client_id="c",
            client_secret="s",
            audience="a",
        )
        assert config.jwks_uri == "https://login.example.com/.well-known/jwks.json"

    def test_jwks_uri_strips_trailing_slash(self) -> None:
        config = OIDCConfig(
            issuer="https://login.example.com/",
            client_id="c",
            client_secret="s",
            audience="a",
        )
        assert config.jwks_uri == "https://login.example.com/.well-known/jwks.json"

    def test_default_scopes(self) -> None:
        config = OIDCConfig(issuer="https://x", client_id="c", client_secret="s", audience="a")
        assert "openid" in config.scopes


# ---------------------------------------------------------------------------
# OIDCProvider — JWKS fetch
# ---------------------------------------------------------------------------


class TestOIDCProviderInit:
    def test_jwks_fetched_lazily(self) -> None:
        """JWKS should only be fetched when verify() is first called."""
        config = OIDCConfig(issuer=_ISSUER, client_id=_CLIENT_ID, client_secret="s", audience=_AUDIENCE)
        provider = OIDCProvider(config)
        assert provider._jwks is None

    def test_jwks_fetch_succeeds(self) -> None:
        fake_jwks = {"keys": [{"kty": "RSA", "kid": "1", "n": "abc", "e": "AQAB"}]}
        mock_response = MagicMock()
        mock_response.json.return_value = fake_jwks
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.get", return_value=mock_response):
            provider = _provider()
            provider._jwks = None  # force refetch
            provider._fetch_jwks()
            assert provider._fetch_jwks() == fake_jwks

    def test_jwks_fetch_failure_raises_auth_error(self) -> None:
        import httpx

        with patch("httpx.get", side_effect=httpx.ConnectError("refused")):
            provider = _provider()
            provider._jwks = None
            with pytest.raises(AuthError, match="Failed to fetch JWKS"):
                provider._fetch_jwks()

    def test_jwks_cached_after_first_call(self) -> None:
        fake_jwks = {"keys": []}
        mock_response = MagicMock()
        mock_response.json.return_value = fake_jwks
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.get", return_value=mock_response) as mock_get:
            provider = _provider()
            provider._jwks = None
            provider._get_jwks()
            provider._get_jwks()
            # Second call must NOT hit the network again
            mock_get.assert_called_once()


# ---------------------------------------------------------------------------
# OIDCProvider — verify()
# ---------------------------------------------------------------------------


class TestOIDCProviderVerify:
    def test_valid_token_returns_claims(self) -> None:
        token = _make_token()
        provider = _provider()
        with patch.object(
            OIDCProvider,
            "_get_jwks",
            return_value={"keys": []},
        ):
            with patch(
                "mind_mem.api.auth.jwt.decode",
                return_value={"sub": "user-123", "iss": _ISSUER, "aud": _AUDIENCE},
            ):
                claims = provider.verify(token)
        assert claims["sub"] == "user-123"

    def test_expired_token_raises_auth_error(self) -> None:
        from jose import ExpiredSignatureError

        provider = _provider()
        with patch.object(provider, "_get_jwks", return_value={"keys": []}):
            with patch(
                "mind_mem.api.auth.jwt.decode",
                side_effect=ExpiredSignatureError("expired"),
            ):
                with pytest.raises(AuthError) as exc_info:
                    provider.verify("any.token.here")
        assert exc_info.value.code == "token_expired"

    def test_wrong_audience_raises_auth_error(self) -> None:
        from jose import JWTError

        provider = _provider()
        with patch.object(provider, "_get_jwks", return_value={"keys": []}):
            with patch(
                "mind_mem.api.auth.jwt.decode",
                side_effect=JWTError("audience"),
            ):
                with pytest.raises(AuthError) as exc_info:
                    provider.verify("any.token.here")
        assert exc_info.value.code in ("wrong_audience", "invalid_token")

    def test_wrong_issuer_raises_auth_error(self) -> None:
        from jose import JWTError

        provider = _provider()
        with patch.object(provider, "_get_jwks", return_value={"keys": []}):
            with patch(
                "mind_mem.api.auth.jwt.decode",
                side_effect=JWTError("issuer"),
            ):
                with pytest.raises(AuthError) as exc_info:
                    provider.verify("bad.issuer.token")
        assert exc_info.value.code in ("wrong_issuer", "invalid_token")

    def test_malformed_token_raises_auth_error(self) -> None:
        from jose import JWTError

        provider = _provider()
        with patch.object(provider, "_get_jwks", return_value={"keys": []}):
            with patch("mind_mem.api.auth.jwt.decode", side_effect=JWTError("bad")):
                with pytest.raises(AuthError):
                    provider.verify("notajwt")


# ---------------------------------------------------------------------------
# OIDCProvider — extract_scopes()
# ---------------------------------------------------------------------------


class TestExtractScopes:
    def test_scope_string(self) -> None:
        provider = _provider()
        scopes = provider.extract_scopes({"scope": "openid email read:data"})
        assert scopes == ["openid", "email", "read:data"]

    def test_scopes_list(self) -> None:
        provider = _provider()
        scopes = provider.extract_scopes({"scopes": ["admin", "user"]})
        assert "admin" in scopes

    def test_roles_list(self) -> None:
        provider = _provider()
        scopes = provider.extract_scopes({"roles": ["org:admin"]})
        assert "org:admin" in scopes

    def test_combined_claims_deduplicated(self) -> None:
        provider = _provider()
        scopes = provider.extract_scopes({"scope": "openid", "scopes": ["openid", "user"], "roles": ["user"]})
        assert scopes.count("openid") == 1
        assert scopes.count("user") == 1

    def test_empty_claims_returns_empty(self) -> None:
        provider = _provider()
        assert provider.extract_scopes({}) == []


# ---------------------------------------------------------------------------
# Preset factories
# ---------------------------------------------------------------------------


class TestPresetFactories:
    def test_okta_preset_issuer(self) -> None:
        p = OIDCProvider.for_okta(
            domain="dev-123.okta.com",
            client_id="cid",
            client_secret="sec",
            audience="api://default",
        )
        # Validate the issuer hostname by parsing the URL and checking the
        # *full* hostname component.  A plain `in` or `endswith` on the raw
        # issuer string would be vulnerable to partial-match bypasses
        # (e.g. "https://evil.notokta.com/…").  Parsing with urlparse and
        # comparing the hostname field ensures we match on a label boundary.
        # The hostname can only be "okta.com" exactly, or a label-prefixed
        # subdomain ending at ".okta.com" (e.g. "dev-123.okta.com").
        from urllib.parse import urlparse as _urlparse

        _parsed = _urlparse(p._config.issuer)
        _host = _parsed.hostname or ""
        # Split on "." and verify the last two labels are "okta" and "com".
        _labels = _host.split(".")
        assert len(_labels) >= 2 and _labels[-2] == "okta" and _labels[-1] == "com", f"Issuer hostname {_host!r} is not an okta.com domain"
        assert p._config.issuer.endswith("/default")

    def test_okta_custom_auth_server(self) -> None:
        p = OIDCProvider.for_okta(
            domain="corp.okta.com",
            client_id="c",
            client_secret="s",
            audience="a",
            authorization_server="myserver",
        )
        assert p._config.issuer.endswith("/myserver")

    def test_auth0_preset_issuer(self) -> None:
        p = OIDCProvider.for_auth0(
            domain="myapp.us.auth0.com",
            client_id="c",
            client_secret="s",
            audience="a",
        )
        assert p._config.issuer == "https://myapp.us.auth0.com/"

    def test_google_workspace_preset_issuer(self) -> None:
        p = OIDCProvider.for_google_workspace(client_id="c", client_secret="s", audience="a")
        assert p._config.issuer == "https://accounts.google.com"

    def test_azure_ad_preset_issuer(self) -> None:
        p = OIDCProvider.for_azure_ad(
            tenant_id="tenant-guid",
            client_id="c",
            client_secret="s",
            audience="a",
        )
        assert "tenant-guid" in p._config.issuer
        assert p._config.issuer.endswith("/v2.0")

    def test_all_presets_return_oidc_provider(self) -> None:
        providers = [
            OIDCProvider.for_okta("d.okta.com", "c", "s", "a"),
            OIDCProvider.for_auth0("d.auth0.com", "c", "s", "a"),
            OIDCProvider.for_google_workspace("c", "s", "a"),
            OIDCProvider.for_azure_ad("tid", "c", "s", "a"),
        ]
        for p in providers:
            assert isinstance(p, OIDCProvider)
            assert isinstance(p._config, OIDCConfig)
