"""OIDC/SSO authentication for the mind-mem REST API.

Validates JWTs issued by Okta, Auth0, Google Workspace, or Azure AD.
Fetches JWKS on first use and caches the result in-process.

Dependencies (api extra): python-jose[cryptography], httpx
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

try:
    import httpx
    from jose import ExpiredSignatureError, JWTError, jwt
    from jose.backends.rsa_backend import RSAKey  # noqa: F401 — presence check
except ImportError as _err:  # pragma: no cover
    raise ImportError(
        "OIDC auth requires the 'api' extra: pip install 'mind-mem[api]'"
    ) from _err


# ---------------------------------------------------------------------------
# AuthError
# ---------------------------------------------------------------------------


class AuthError(Exception):
    """Raised when JWT validation fails for any reason."""

    def __init__(self, message: str, code: str = "auth_error") -> None:
        super().__init__(message)
        self.code = code


# ---------------------------------------------------------------------------
# OIDCConfig
# ---------------------------------------------------------------------------


@dataclass
class OIDCConfig:
    """Configuration for a single OIDC issuer.

    Args:
        issuer:        Issuer URL (must match the ``iss`` claim exactly).
        client_id:     Registered client / application ID.
        client_secret: Client secret (kept server-side only).
        audience:      Expected ``aud`` claim value.
        scopes:        Requested scopes (informational; not validated per-call).
    """

    issuer: str
    client_id: str
    client_secret: str
    audience: str
    scopes: list[str] = field(default_factory=lambda: ["openid", "profile", "email"])

    @property
    def jwks_uri(self) -> str:
        """Standard OIDC discovery JWKS URI derived from the issuer URL."""
        base = self.issuer.rstrip("/")
        return f"{base}/.well-known/jwks.json"


# ---------------------------------------------------------------------------
# OIDCProvider
# ---------------------------------------------------------------------------


class OIDCProvider:
    """JWT validator backed by the issuer's JWKS endpoint.

    JWKS are fetched lazily on first call to :meth:`verify` and cached
    for the lifetime of the process. Call :meth:`refresh_jwks` to force
    a reload (e.g. on a 401 from a downstream resource).

    Args:
        config: An :class:`OIDCConfig` instance for the desired issuer.
    """

    def __init__(self, config: OIDCConfig) -> None:
        self._config = config
        self._jwks: dict[str, Any] | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify(self, token: str) -> dict:
        """Validate *token* and return its claims.

        Checks:
        - Signature (using the issuer's JWKS)
        - ``iss`` matches :attr:`OIDCConfig.issuer`
        - ``aud`` matches :attr:`OIDCConfig.audience`
        - ``exp`` has not passed

        Returns:
            Decoded claims dict.

        Raises:
            AuthError: On any validation failure.
        """
        jwks = self._get_jwks()
        try:
            claims: dict = jwt.decode(
                token,
                jwks,
                algorithms=["RS256", "RS384", "RS512", "ES256", "ES384", "ES512"],
                audience=self._config.audience,
                issuer=self._config.issuer,
                options={"verify_exp": True, "verify_iss": True, "verify_aud": True},
            )
        except ExpiredSignatureError as exc:
            raise AuthError("Token has expired", code="token_expired") from exc
        except JWTError as exc:
            msg = str(exc).lower()
            if "issuer" in msg:
                raise AuthError(f"Token issuer mismatch: {exc}", code="wrong_issuer") from exc
            if "audience" in msg:
                raise AuthError(f"Token audience mismatch: {exc}", code="wrong_audience") from exc
            raise AuthError(f"Token validation failed: {exc}", code="invalid_token") from exc
        return claims

    def extract_scopes(self, claims: dict) -> list[str]:
        """Pull scopes from standard JWT claim fields.

        Checks ``scope`` (space-separated string), ``scopes`` (list),
        and ``roles`` (list) — covering Okta, Auth0, Azure AD, and
        Google Workspace conventions.
        """
        result: list[str] = []
        scope_str = claims.get("scope", "")
        if isinstance(scope_str, str) and scope_str:
            result.extend(scope_str.split())
        scopes_list = claims.get("scopes", [])
        if isinstance(scopes_list, list):
            result.extend(str(s) for s in scopes_list)
        roles = claims.get("roles", [])
        if isinstance(roles, list):
            result.extend(str(r) for r in roles)
        return list(dict.fromkeys(result))  # deduplicate, preserve order

    def refresh_jwks(self) -> None:
        """Force a JWKS reload from the remote endpoint."""
        with self._lock:
            self._jwks = None
            self._jwks = self._fetch_jwks()

    # ------------------------------------------------------------------
    # Preset factories
    # ------------------------------------------------------------------

    @classmethod
    def for_okta(
        cls,
        domain: str,
        client_id: str,
        client_secret: str,
        audience: str,
        authorization_server: str = "default",
    ) -> "OIDCProvider":
        """Return an OIDCProvider pre-configured for an Okta tenant.

        Args:
            domain:               Your Okta domain, e.g. ``dev-12345.okta.com``.
            client_id:            Application Client ID from Okta console.
            client_secret:        Application Client Secret.
            audience:             API audience identifier.
            authorization_server: Okta auth server (default ``"default"``).
        """
        issuer = f"https://{domain}/oauth2/{authorization_server}"
        config = OIDCConfig(
            issuer=issuer,
            client_id=client_id,
            client_secret=client_secret,
            audience=audience,
            scopes=["openid", "profile", "email"],
        )
        return cls(config)

    @classmethod
    def for_auth0(
        cls,
        domain: str,
        client_id: str,
        client_secret: str,
        audience: str,
    ) -> "OIDCProvider":
        """Return an OIDCProvider pre-configured for an Auth0 tenant.

        Args:
            domain:        Your Auth0 domain, e.g. ``your-app.us.auth0.com``.
            client_id:     Application Client ID.
            client_secret: Application Client Secret.
            audience:      API identifier from Auth0 console.
        """
        issuer = f"https://{domain}/"
        config = OIDCConfig(
            issuer=issuer,
            client_id=client_id,
            client_secret=client_secret,
            audience=audience,
            scopes=["openid", "profile", "email"],
        )
        return cls(config)

    @classmethod
    def for_google_workspace(
        cls,
        client_id: str,
        client_secret: str,
        audience: str,
    ) -> "OIDCProvider":
        """Return an OIDCProvider pre-configured for Google Workspace / GCP.

        Args:
            client_id:     OAuth 2.0 Client ID from Google Cloud Console.
            client_secret: Client Secret.
            audience:      Audience — typically the client_id itself for ID tokens,
                           or the resource server URL for access tokens.
        """
        config = OIDCConfig(
            issuer="https://accounts.google.com",
            client_id=client_id,
            client_secret=client_secret,
            audience=audience,
            scopes=["openid", "profile", "email"],
        )
        return cls(config)

    @classmethod
    def for_azure_ad(
        cls,
        tenant_id: str,
        client_id: str,
        client_secret: str,
        audience: str,
    ) -> "OIDCProvider":
        """Return an OIDCProvider pre-configured for Microsoft Azure AD / Entra ID.

        Args:
            tenant_id:     Azure AD tenant GUID or ``common`` / ``organizations``.
            client_id:     Application (client) ID from Azure portal.
            client_secret: Client secret value.
            audience:      Application ID URI or ``api://<client_id>``.
        """
        issuer = f"https://login.microsoftonline.com/{tenant_id}/v2.0"
        config = OIDCConfig(
            issuer=issuer,
            client_id=client_id,
            client_secret=client_secret,
            audience=audience,
            scopes=["openid", "profile", "email"],
        )
        return cls(config)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_jwks(self) -> dict[str, Any]:
        """Return cached JWKS, fetching on first call."""
        with self._lock:
            if self._jwks is None:
                self._jwks = self._fetch_jwks()
            return self._jwks

    def _fetch_jwks(self) -> dict[str, Any]:
        """HTTP GET the JWKS URI and return the parsed JSON."""
        uri = self._config.jwks_uri
        try:
            response = httpx.get(uri, timeout=10.0)
            response.raise_for_status()
            result: dict[str, Any] = response.json()
            return result
        except httpx.HTTPError as exc:
            raise AuthError(
                f"Failed to fetch JWKS from {uri}: {exc}",
                code="jwks_fetch_failed",
            ) from exc
