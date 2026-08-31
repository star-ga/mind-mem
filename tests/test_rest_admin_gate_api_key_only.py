"""The REST admin gate must fire in an API-key-only deployment.

``_require_admin`` decided whether to enforce the admin-scope check with
its own hand-written predicate listing ``MIND_MEM_TOKEN`` /
``MIND_MEM_ADMIN_TOKEN`` / OIDC. The startup bind gate
(``_auth_is_configured``) listed those *plus* ``MIND_MEM_API_KEY_DB``.

The two disagreed for exactly one deployment shape: per-agent ``mmk_*``
keys and nothing else. There the server starts happily ("authentication
configured, bind allowed") while ``_require_admin``'s whole body is
skipped — ``_has_admin_scope`` and ``_api_key_has_admin_scope`` are
never called. Any valid key, including one minted with ``scopes=[]``,
could then call ``POST /v1/admin/api_keys`` and mint itself a key with
``scopes=["admin"]``. Privilege escalation from any low-privilege key.

These tests pin three things:

* a ``scopes=[]`` key is refused (403) by every admin endpoint,
* that refusal comes from the *scope* check and not from the gate
  denying everything — the same key still works on user-tier ``/v1/recall``
  and an ``admin``-scoped key still gets through,
* the drift itself cannot come back: whatever ``_auth_is_configured``
  counts, the admin gate counts too.
"""

from __future__ import annotations

from typing import Any, Generator

import pytest

fastapi = pytest.importorskip("fastapi", reason="fastapi not installed")

from fastapi.testclient import TestClient  # noqa: E402

from mind_mem.api.rest import (  # noqa: E402
    _admin_gate_is_configured,
    _auth_is_configured,
    create_app,
)

_AUTH_ENV = (
    "MIND_MEM_TOKEN",
    "MIND_MEM_ADMIN_TOKEN",
    "MIND_MEM_API_KEY_DB",
    "MIND_MEM_ALLOW_UNAUTHENTICATED_LOCALHOST",
    "OIDC_ISSUER",
    "OIDC_AUDIENCE",
)


@pytest.fixture()
def workspace(tmp_path: Any) -> str:
    for subdir in ("decisions", "tasks", "entities", "intelligence", "memory"):
        (tmp_path / subdir).mkdir()
    return str(tmp_path)


@pytest.fixture()
def clean_auth_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip every auth mechanism so each test states its own."""
    for name in _AUTH_ENV:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture()
def api_key_only(
    workspace: str,
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    clean_auth_env: None,
) -> Generator[tuple[TestClient, str, str], None, None]:
    """API-key-only deployment: a low-privilege key and an admin key.

    No ``MIND_MEM_TOKEN``, no ``MIND_MEM_ADMIN_TOKEN``, no OIDC — the
    exact shape whose admin gate used to be skipped.
    """
    from mind_mem.api.api_keys import APIKeyStore

    db_path = str(tmp_path / "keys.db")
    store = APIKeyStore(db_path, production=True)
    low_key = store.create(agent_id="bot-lowpriv", scopes=[], expires_in_days=30)
    admin_key = store.create(agent_id="ops", scopes=["admin"], expires_in_days=30)

    monkeypatch.setenv("MIND_MEM_WORKSPACE", workspace)
    monkeypatch.setenv("MIND_MEM_API_KEY_DB", db_path)
    monkeypatch.setenv("MIND_MEM_ENV", "production")

    with TestClient(create_app(workspace), raise_server_exceptions=False) as client:
        yield client, low_key, admin_key


def _bearer(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


class TestAPIKeyOnlyAdminGate:
    def test_low_privilege_key_cannot_create_admin_key(
        self,
        api_key_only: tuple[TestClient, str, str],
    ) -> None:
        """The escalation itself: scopes=[] must not mint scopes=["admin"]."""
        client, low_key, _ = api_key_only
        resp = client.post(
            "/v1/admin/api_keys",
            headers=_bearer(low_key),
            json={"agent_id": "attacker", "scopes": ["admin"], "expires_in_days": 30},
        )
        assert resp.status_code == 403
        assert "key" not in resp.json()

    @pytest.mark.parametrize(
        ("method", "path"),
        [
            ("get", "/v1/admin/api_keys"),
            ("delete", "/v1/admin/api_keys/some-key-id"),
            ("post", "/v1/admin/api_keys/some-key-id/rotate"),
        ],
    )
    def test_low_privilege_key_refused_by_every_admin_endpoint(
        self,
        api_key_only: tuple[TestClient, str, str],
        method: str,
        path: str,
    ) -> None:
        client, low_key, _ = api_key_only
        resp = getattr(client, method)(path, headers=_bearer(low_key))
        assert resp.status_code == 403

    def test_low_privilege_key_cannot_apply_governance_proposals(
        self,
        api_key_only: tuple[TestClient, str, str],
    ) -> None:
        """The skipped gate covered the governance routes too, not just key admin.

        ``/v1/approve_apply`` and ``/v1/rollback_proposal`` are declared
        admin-only and depend on ``_require_admin``; with the gate
        skipped they were reachable by any valid key.
        """
        client, low_key, _ = api_key_only
        applied = client.post(
            "/v1/approve_apply",
            headers=_bearer(low_key),
            json={"proposal_id": "P-20260830-001", "dry_run": True},
        )
        assert applied.status_code == 403

        rolled_back = client.post(
            "/v1/rollback_proposal",
            headers=_bearer(low_key),
            json={"receipt_ts": "20260830-000000", "reason": "regression-test"},
        )
        assert rolled_back.status_code == 403

    def test_low_privilege_key_still_works_on_user_tier(
        self,
        api_key_only: tuple[TestClient, str, str],
    ) -> None:
        """Guards against 'fixing' this by denying the key everything."""
        client, low_key, _ = api_key_only
        resp = client.post("/v1/recall", headers=_bearer(low_key), json={"query": "hello", "limit": 3})
        assert resp.status_code == 200

    def test_admin_scoped_key_passes_the_gate(
        self,
        api_key_only: tuple[TestClient, str, str],
    ) -> None:
        """The 403 above must come from the scope check, not a blanket denial."""
        client, _, admin_key = api_key_only
        created = client.post(
            "/v1/admin/api_keys",
            headers=_bearer(admin_key),
            json={"agent_id": "bot-2", "scopes": ["user"], "expires_in_days": 30},
        )
        assert created.status_code == 201
        assert created.json()["key"].startswith("mmk_live_")

        listed = client.get("/v1/admin/api_keys", headers=_bearer(admin_key))
        assert listed.status_code == 200

    def test_unknown_key_is_unauthenticated_not_forbidden(
        self,
        api_key_only: tuple[TestClient, str, str],
    ) -> None:
        client, _, _ = api_key_only
        resp = client.get("/v1/admin/api_keys", headers=_bearer("mmk_live_" + "0" * 64))
        assert resp.status_code == 401


class TestAdminGateTracksAuthPredicate:
    """The admin gate must never be narrower than the startup bind gate."""

    @pytest.mark.parametrize(
        "env",
        [
            {"MIND_MEM_TOKEN": "u" * 32},
            {"MIND_MEM_ADMIN_TOKEN": "a" * 32},
            {"MIND_MEM_API_KEY_DB": "/tmp/mind-mem-keys.db"},
            {"OIDC_ISSUER": "https://idp.example.com", "OIDC_AUDIENCE": "mind-mem"},
        ],
    )
    def test_configured_auth_always_arms_the_admin_gate(
        self,
        monkeypatch: pytest.MonkeyPatch,
        clean_auth_env: None,
        env: dict[str, str],
    ) -> None:
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        assert _auth_is_configured() is True
        assert _admin_gate_is_configured() is True

    def test_no_auth_configured_leaves_the_gate_open(
        self,
        monkeypatch: pytest.MonkeyPatch,
        clean_auth_env: None,
    ) -> None:
        """Unchanged behaviour: with nothing configured there is no admin tier.

        ``_require_auth`` already fails closed here unless the operator
        set the loopback opt-in, so the gate has nothing to enforce.
        """
        assert _auth_is_configured() is False
        assert _admin_gate_is_configured() is False

    def test_present_but_empty_oidc_arms_the_gate(
        self,
        monkeypatch: pytest.MonkeyPatch,
        clean_auth_env: None,
    ) -> None:
        """Misconfiguration resolves closed, not open."""
        monkeypatch.setenv("OIDC_ISSUER", "")
        monkeypatch.setenv("OIDC_AUDIENCE", "")
        assert _auth_is_configured() is False
        assert _admin_gate_is_configured() is True
