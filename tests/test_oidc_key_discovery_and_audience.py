"""Regression tests: where the signing keys come from, and who a token is for.

Two defects in :mod:`mind_mem.api.auth`:

* The JWKS endpoint was *derived* as ``{issuer}/.well-known/jwks.json``.
  OIDC defines no such derivation — ``jwks_uri`` is a field inside the
  discovery document at ``{issuer}/.well-known/openid-configuration``, and
  real providers serve their keys from a different host entirely. The
  derived URL 404s, so ``_fetch_jwks`` raised before a token was ever
  decoded and every JWT was rejected.
* ``aud`` is an OPTIONAL JWT claim, and the JWT library validates it only
  when the token carries one. A token that simply omits ``aud`` therefore
  sailed past ``verify()`` — a guard that cannot fail on the input an
  attacker chooses. A token this issuer minted for another relying party
  must not authenticate here.
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("jose", reason="python-jose not installed; skipping OIDC tests")
pytest.importorskip("httpx", reason="httpx not installed; skipping OIDC tests")

from jose import jwt  # noqa: E402

from mind_mem.api.auth import AuthError, OIDCConfig, OIDCProvider  # noqa: E402

_ISSUER = "https://idp.example.com"
_AUDIENCE = "https://mind-mem.example/api"
_JWKS_URI = "https://keys.example.net/oauth2/v3/certs"
_DISCOVERY_URI = f"{_ISSUER}/.well-known/openid-configuration"
_SECRET = "test-secret-not-used-in-production"


def _provider(jwks_uri: str = "") -> OIDCProvider:
    return OIDCProvider(
        OIDCConfig(
            issuer=_ISSUER,
            client_id="mm-client",
            client_secret="secret",
            audience=_AUDIENCE,
            jwks_uri=jwks_uri,
        )
    )


def _json_response(payload: Any) -> MagicMock:
    response = MagicMock()
    response.json.return_value = payload
    response.raise_for_status = MagicMock()
    return response


class _Fetcher:
    """Records every URL fetched and answers from a URL→payload table."""

    def __init__(self, table: dict[str, Any]) -> None:
        self.table = table
        self.urls: list[str] = []

    def __call__(self, url: str, **_kwargs: Any) -> MagicMock:
        self.urls.append(url)
        if url not in self.table:
            import httpx

            raise httpx.HTTPStatusError("404", request=MagicMock(), response=MagicMock())
        return _json_response(self.table[url])


# ---------------------------------------------------------------------------
# Key location
# ---------------------------------------------------------------------------


class TestKeysComeFromDiscovery:
    def test_jwks_uri_is_read_from_the_discovery_document(self) -> None:
        keys = {"keys": [{"kty": "RSA", "kid": "1", "n": "abc", "e": "AQAB"}]}
        fetcher = _Fetcher({_DISCOVERY_URI: {"jwks_uri": _JWKS_URI}, _JWKS_URI: keys})

        with patch("httpx.get", side_effect=fetcher):
            assert _provider()._fetch_jwks() == keys

        assert fetcher.urls == [_DISCOVERY_URI, _JWKS_URI]

    def test_derived_well_known_jwks_path_is_never_requested(self) -> None:
        """The old guess. An IdP that only serves the real endpoints must work."""
        fetcher = _Fetcher({_DISCOVERY_URI: {"jwks_uri": _JWKS_URI}, _JWKS_URI: {"keys": []}})

        with patch("httpx.get", side_effect=fetcher):
            _provider()._fetch_jwks()

        assert f"{_ISSUER}/.well-known/jwks.json" not in fetcher.urls

    def test_explicit_jwks_uri_skips_discovery(self) -> None:
        fetcher = _Fetcher({_JWKS_URI: {"keys": []}})

        with patch("httpx.get", side_effect=fetcher):
            assert _provider(jwks_uri=_JWKS_URI)._fetch_jwks() == {"keys": []}

        assert fetcher.urls == [_JWKS_URI]

    def test_discovery_without_jwks_uri_fails_closed(self) -> None:
        fetcher = _Fetcher({_DISCOVERY_URI: {"issuer": _ISSUER}})

        with patch("httpx.get", side_effect=fetcher):
            with pytest.raises(AuthError) as exc_info:
                _provider()._fetch_jwks()

        assert exc_info.value.code == "jwks_fetch_failed"

    def test_plaintext_jwks_uri_is_refused(self) -> None:
        """Signing keys fetched over http are keys an on-path attacker picks."""
        fetcher = _Fetcher({_DISCOVERY_URI: {"jwks_uri": "http://keys.example.net/certs"}})

        with patch("httpx.get", side_effect=fetcher):
            with pytest.raises(AuthError, match="https jwks_uri"):
                _provider()._fetch_jwks()

    def test_google_shaped_issuer_resolves_offhost_keys(self) -> None:
        """The concrete case the derivation broke: keys on another host."""
        provider = OIDCProvider.for_google_workspace(client_id="c", client_secret="s", audience="c")
        discovery = "https://accounts.google.com/.well-known/openid-configuration"
        offhost = "https://www.googleapis.com/oauth2/v3/certs"
        fetcher = _Fetcher({discovery: {"jwks_uri": offhost}, offhost: {"keys": []}})

        with patch("httpx.get", side_effect=fetcher):
            assert provider._fetch_jwks() == {"keys": []}

        assert fetcher.urls == [discovery, offhost]


# ---------------------------------------------------------------------------
# Audience
# ---------------------------------------------------------------------------


class TestAudienceIsRequired:
    def test_library_alone_accepts_a_token_with_no_aud(self) -> None:
        """Pin the upstream behaviour this guard exists for.

        The JWT library skips audience validation when the claim is absent,
        so ``verify_aud=True`` constrains only tokens that opted in.
        """
        token = jwt.encode(
            {"iss": _ISSUER, "sub": "user-1", "exp": int(time.time()) + 3600},
            _SECRET,
            algorithm="HS256",
        )
        claims = jwt.decode(
            token,
            _SECRET,
            algorithms=["HS256"],
            audience=_AUDIENCE,
            issuer=_ISSUER,
            options={"verify_exp": True, "verify_iss": True, "verify_aud": True},
        )
        assert "aud" not in claims

    def test_verify_rejects_a_token_with_no_aud(self) -> None:
        provider = _provider(jwks_uri=_JWKS_URI)
        decoded = {"iss": _ISSUER, "sub": "user-1", "exp": int(time.time()) + 3600}

        with patch.object(OIDCProvider, "_get_jwks", return_value={"keys": []}):
            with patch("mind_mem.api.auth.jwt.decode", return_value=decoded):
                with pytest.raises(AuthError) as exc_info:
                    provider.verify("any.token.here")

        assert exc_info.value.code == "wrong_audience"

    def test_verify_accepts_the_configured_audience(self) -> None:
        provider = _provider(jwks_uri=_JWKS_URI)
        decoded = {"iss": _ISSUER, "sub": "user-1", "aud": _AUDIENCE}

        with patch.object(OIDCProvider, "_get_jwks", return_value={"keys": []}):
            with patch("mind_mem.api.auth.jwt.decode", return_value=decoded):
                assert provider.verify("any.token.here") == decoded

    def test_verify_accepts_an_aud_array_containing_us(self) -> None:
        provider = _provider(jwks_uri=_JWKS_URI)
        decoded = {"iss": _ISSUER, "sub": "user-1", "aud": ["other-app", _AUDIENCE]}

        with patch.object(OIDCProvider, "_get_jwks", return_value={"keys": []}):
            with patch("mind_mem.api.auth.jwt.decode", return_value=decoded):
                assert provider.verify("any.token.here") == decoded

    @pytest.mark.parametrize(
        "aud",
        [
            "other-app",
            ["other-app"],
            [],
            [_AUDIENCE.encode()],
            {"aud": _AUDIENCE},
            123,
        ],
    )
    def test_verify_rejects_a_foreign_or_malformed_aud(self, aud: object) -> None:
        provider = _provider(jwks_uri=_JWKS_URI)
        decoded = {"iss": _ISSUER, "sub": "user-1", "aud": aud}

        with patch.object(OIDCProvider, "_get_jwks", return_value={"keys": []}):
            with patch("mind_mem.api.auth.jwt.decode", return_value=decoded):
                with pytest.raises(AuthError) as exc_info:
                    provider.verify("any.token.here")

        assert exc_info.value.code == "wrong_audience"
