"""OIDC/SSO authentication for the mind-mem REST API.

Validates JWTs issued by Okta, Auth0, Google Workspace, or Azure AD.
The signing keys are located the way OIDC specifies — read ``jwks_uri``
out of the issuer's discovery document — then fetched on first use and
cached in-process. Set :attr:`OIDCConfig.jwks_uri` to skip discovery.

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
    raise ImportError("OIDC auth requires the 'api' extra: pip install 'mind-mem[api]'") from _err


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
        jwks_uri:      Explicit JWKS endpoint. Leave empty (the default) to
                       read it out of the issuer's discovery document, which
                       is the only place OIDC defines it. Set it for an
                       issuer that publishes no discovery document, or to
                       pin the endpoint in an air-gapped deployment.
    """

    issuer: str
    client_id: str
    client_secret: str
    audience: str
    scopes: list[str] = field(default_factory=lambda: ["openid", "profile", "email"])
    jwks_uri: str = ""

    @property
    def discovery_uri(self) -> str:
        """The OIDC Discovery document URL for this issuer.

        Defined by OpenID Connect Discovery 1.0 §4: the provider
        configuration lives at ``{issuer}/.well-known/openid-configuration``
        and *names* its ``jwks_uri`` inside that document. There is no
        specified way to derive the JWKS URL from the issuer directly —
        real providers put it on a different host or path entirely — so
        deriving one is guesswork that fails closed on every token.
        """
        base = self.issuer.rstrip("/")
        return f"{base}/.well-known/openid-configuration"


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
        - ``aud`` is **present** and contains :attr:`OIDCConfig.audience`
        - ``exp`` has not passed

        The audience check is re-asserted here rather than left to the JWT
        library: ``aud`` is an OPTIONAL claim (RFC 7519 §4.1.3), so a
        library that validates it only "when present" accepts any token
        that simply omits it — a guard that cannot fail on the one input
        an attacker chooses. A token minted by this issuer for a different
        relying party must not authenticate here.

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
        self._require_audience(claims)
        return claims

    def _require_audience(self, claims: dict) -> None:
        """Reject a token whose ``aud`` is missing, malformed, or foreign.

        ``aud`` is a single string or an array of strings; either way the
        configured audience must appear in it.
        """
        expected = self._config.audience
        raw = claims.get("aud")
        if raw is None:
            raise AuthError(
                f"Token has no 'aud' claim; expected {expected!r}",
                code="wrong_audience",
            )
        values = [raw] if isinstance(raw, str) else raw
        if not isinstance(values, (list, tuple)) or not all(isinstance(v, str) for v in values):
            raise AuthError("Token 'aud' claim is malformed", code="wrong_audience")
        if expected not in values:
            raise AuthError(
                f"Token audience mismatch: expected {expected!r}",
                code="wrong_audience",
            )

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

    def _resolve_jwks_uri(self) -> str:
        """Return the issuer's JWKS endpoint.

        An explicitly configured :attr:`OIDCConfig.jwks_uri` wins. Otherwise
        the endpoint is read from the issuer's discovery document, which is
        where OIDC Discovery 1.0 puts it — providers really do serve their
        keys from another host, so there is nothing to derive it from.
        """
        configured = self._config.jwks_uri.strip()
        if configured:
            return configured

        uri = self._config.discovery_uri
        try:
            response = httpx.get(uri, timeout=10.0)
            response.raise_for_status()
            document: Any = response.json()
        except httpx.HTTPError as exc:
            raise AuthError(
                f"Failed to fetch OIDC discovery document from {uri}: {exc}",
                code="jwks_fetch_failed",
            ) from exc
        except ValueError as exc:
            raise AuthError(
                f"OIDC discovery document at {uri} is not JSON: {exc}",
                code="jwks_fetch_failed",
            ) from exc

        jwks_uri = document.get("jwks_uri") if isinstance(document, dict) else None
        if not isinstance(jwks_uri, str) or not jwks_uri.startswith("https://"):
            # No usable key endpoint: fail closed rather than guessing one.
            # A non-HTTPS jwks_uri is refused outright — signing keys fetched
            # over a downgradeable channel are keys an on-path attacker picks.
            raise AuthError(
                f"OIDC discovery document at {uri} declares no https jwks_uri",
                code="jwks_fetch_failed",
            )
        return jwks_uri

    def _fetch_jwks(self) -> dict[str, Any]:
        """HTTP GET the JWKS URI and return the parsed JSON."""
        uri = self._resolve_jwks_uri()
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
