"""REST-layer defects found inside files that also carried a HIGH finding.

Each test names the mechanism it pins, not just the symptom.
"""

from __future__ import annotations

import os
import stat
from typing import Any, Generator

import pytest
from _platform_compat import chmod_denies_write, is_root

fastapi = pytest.importorskip("fastapi", reason="fastapi not installed; skipping REST API tests")

from fastapi.testclient import TestClient  # noqa: E402

from mind_mem.api import rest  # noqa: E402
from mind_mem.api.rest import create_app  # noqa: E402


@pytest.fixture()
def workspace(tmp_path: Any) -> str:
    for subdir in ("decisions", "tasks", "entities", "intelligence", "memory"):
        (tmp_path / subdir).mkdir()
    return str(tmp_path)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    for var in (
        "MIND_MEM_TOKEN",
        "MIND_MEM_ADMIN_TOKEN",
        "MIND_MEM_API_KEY_DB",
        "OIDC_ISSUER",
        "OIDC_AUDIENCE",
        "OIDC_CLIENT_ID",
        "OIDC_CLIENT_SECRET",
        "MIND_MEM_ALLOW_UNAUTHENTICATED_LOCALHOST",
    ):
        monkeypatch.delenv(var, raising=False)
    rest._OIDC_PROVIDER_CACHE.clear()
    rest._API_KEY_STORE_CACHE.clear()
    yield
    rest._OIDC_PROVIDER_CACHE.clear()
    rest._API_KEY_STORE_CACHE.clear()


# ---------------------------------------------------------------------------
# OIDC provider is cached, so its in-process JWKS cache is real
# ---------------------------------------------------------------------------


def test_oidc_provider_is_reused_across_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """A new provider per call means a new JWKS fetch per call.

    OIDCProvider caches JWKS on the INSTANCE (self._jwks) and documents
    that cache as living "for the lifetime of the process", but both
    call sites constructed a fresh provider, so each authenticated
    request made its own blocking HTTPS fetch to the IdP.
    """
    pytest.importorskip("mind_mem.api.auth")
    monkeypatch.setenv("OIDC_ISSUER", "https://idp.example")
    monkeypatch.setenv("OIDC_AUDIENCE", "mind-mem")

    first = rest._oidc_provider("https://idp.example", "cid", "secret", "mind-mem")
    second = rest._oidc_provider("https://idp.example", "cid", "secret", "mind-mem")
    assert first is second, "provider rebuilt — the JWKS cache is per-call, not per-process"

    # A configuration change must NOT hand back the stale provider.
    rotated = rest._oidc_provider("https://idp.example", "cid", "rotated-secret", "mind-mem")
    assert rotated is not first

    # And the cache stays bounded rather than growing per distinct config.
    for i in range(rest._OIDC_PROVIDER_CACHE_MAX * 3):
        rest._oidc_provider(f"https://idp{i}.example", "cid", "s", "aud")
    assert len(rest._OIDC_PROVIDER_CACHE) <= rest._OIDC_PROVIDER_CACHE_MAX


def test_verify_oidc_token_reuses_the_cached_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """The request path itself goes through the cache."""
    pytest.importorskip("mind_mem.api.auth")
    monkeypatch.setenv("OIDC_ISSUER", "https://idp.example")
    monkeypatch.setenv("OIDC_AUDIENCE", "mind-mem")

    built: list[object] = []
    real = rest._oidc_provider

    def _counting(*args: str) -> object:
        provider = real(*args)
        built.append(provider)
        return provider

    monkeypatch.setattr(rest, "_oidc_provider", _counting)
    for _ in range(3):
        rest._verify_oidc_token("aaa.bbb.ccc")
    assert len(built) == 3
    assert len({id(p) for p in built}) == 1, "each request built its own provider"


# ---------------------------------------------------------------------------
# A broken API-key store is not "no API-key store"
# ---------------------------------------------------------------------------


@pytest.mark.skipif(is_root(), reason="root ignores directory permissions")
def test_unopenable_api_key_store_is_not_reported_as_unconfigured(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """A configured-but-unopenable store must not masquerade as unset.

    APIKeyStore.__init__ does real I/O (makedirs + connect + CREATE
    TABLE). A bare `except Exception: return None` turned an unwritable
    parent directory into "not configured", so key holders got 401 and
    the admin endpoint answered 501 "set MIND_MEM_API_KEY_DB" with the
    variable set to exactly that path.
    """
    locked = tmp_path / "locked"
    locked.mkdir()
    db_path = locked / "keys.db"
    if not chmod_denies_write(tmp_path):
        pytest.skip("this filesystem does not enforce the write bit (e.g. Windows)")
    os.chmod(locked, stat.S_IRUSR | stat.S_IXUSR)  # r-x------ : cannot create files
    monkeypatch.setenv("MIND_MEM_API_KEY_DB", str(db_path))
    try:
        with pytest.raises(rest.APIKeyStoreUnavailable) as excinfo:
            rest._get_api_key_store()
        assert "keys.db" in str(excinfo.value)
        # And the admin helper reports 503 (broken), never 501 (absent).
        with pytest.raises(rest.HTTPException) as http_exc:
            rest._require_api_key_store()
        assert http_exc.value.status_code == 503
    finally:
        os.chmod(locked, stat.S_IRWXU)


def test_unconfigured_api_key_store_still_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """The other half of the split: unset really is None, not an error."""
    monkeypatch.delenv("MIND_MEM_API_KEY_DB", raising=False)
    assert rest._get_api_key_store() is None
    with pytest.raises(rest.HTTPException) as http_exc:
        rest._require_api_key_store()
    assert http_exc.value.status_code == 501


def test_api_key_store_is_cached(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """The store was rebuilt (and a sqlite file opened) per request."""
    monkeypatch.setenv("MIND_MEM_API_KEY_DB", str(tmp_path / "keys.db"))
    assert rest._get_api_key_store() is rest._get_api_key_store()


# ---------------------------------------------------------------------------
# Unauthenticated exposure of /v1/health and /v1/metrics
# ---------------------------------------------------------------------------


def test_health_hides_workspace_from_unauthenticated_callers(workspace: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Liveness stays open; the host description does not.

    With MIND_MEM_TOKEN configured, GET /v1/health used to answer 200
    with the absolute workspace path and whether it exists, to a caller
    holding no credential at all.
    """
    monkeypatch.setenv("MIND_MEM_TOKEN", "s3cret-token-value")
    monkeypatch.setenv("MIND_MEM_WORKSPACE", workspace)
    app = create_app(workspace)
    with TestClient(app, raise_server_exceptions=False) as tc:
        anon = tc.get("/v1/health")
        assert anon.status_code == 200, "liveness must stay reachable"
        body = anon.json()
        assert body["status"] == "ok"
        assert "workspace" not in body
        assert "workspace_exists" not in body

        auth = tc.get("/v1/health", headers={"Authorization": "Bearer s3cret-token-value"})
        assert auth.status_code == 200
        assert auth.json()["workspace"] == workspace
        assert auth.json()["workspace_exists"] is True


def test_metrics_requires_auth(workspace: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """The Prometheus registry names mcp_http_auth_failures — i.e. whether
    credential guessing is underway — and was served to anyone."""
    monkeypatch.setenv("MIND_MEM_TOKEN", "s3cret-token-value")
    monkeypatch.setenv("MIND_MEM_WORKSPACE", workspace)
    app = create_app(workspace)
    with TestClient(app, raise_server_exceptions=False) as tc:
        assert tc.get("/v1/metrics").status_code == 401
        authed = tc.get("/v1/metrics", headers={"Authorization": "Bearer s3cret-token-value"})
        assert authed.status_code in (200, 404)  # 404 only when prometheus_client is absent


# ---------------------------------------------------------------------------
# The bind gate must test the same predicate the credential code tests
# ---------------------------------------------------------------------------


def test_empty_admin_token_is_not_configured_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    """An exported-but-empty token is not a credential.

    _auth_is_configured tested presence (`is not None`) while
    http_auth._build_http_auth_tokens registers a token only
    `if admin_token:`. With MIND_MEM_ADMIN_TOKEN='' the gate certified a
    routable bind as authenticated while no credential existed.
    """
    monkeypatch.setenv("MIND_MEM_ADMIN_TOKEN", "")
    assert rest._auth_is_configured() is False
    with pytest.raises(SystemExit):
        rest._enforce_fail_closed("0.0.0.0", False)  # noqa: S104 - the point of the test


def test_empty_user_token_is_not_configured_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MIND_MEM_TOKEN", "")
    assert rest._auth_is_configured() is False
    with pytest.raises(SystemExit):
        rest._enforce_fail_closed("0.0.0.0", False)  # noqa: S104 - the point of the test


def test_real_admin_token_still_configures_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    """The fix must not make a genuine token look unconfigured."""
    monkeypatch.setenv("MIND_MEM_ADMIN_TOKEN", "a-real-token")
    assert rest._auth_is_configured() is True
    rest._enforce_fail_closed("0.0.0.0", False)  # noqa: S104 - must NOT raise


def test_empty_admin_token_still_closes_the_admin_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """_admin_gate_is_configured keeps its fail-closed `is not None` superset."""
    monkeypatch.setenv("MIND_MEM_ADMIN_TOKEN", "")
    assert rest._admin_gate_is_configured() is True
