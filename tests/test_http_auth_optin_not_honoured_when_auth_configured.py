# Copyright 2026 STARGA, Inc.
"""The unauthenticated opt-in must not grant access when auth IS configured.

MEASURED on 28b9d7f1 before this fix:

    MIND_MEM_ADMIN_TOKEN=the-real-admin-token   (MIND_MEM_TOKEN unset)
    MIND_MEM_ALLOW_UNAUTHENTICATED_LOCALHOST=1
    verify_token({})                               -> True
    verify_token({"Authorization": "Bearer WRONG"}) -> True

Two functions disagreed about what "configured" means:

* ``rest._auth_is_configured`` counts MIND_MEM_TOKEN **or**
  MIND_MEM_ADMIN_TOKEN **or** MIND_MEM_API_KEY_DB **or** OIDC_ISSUER+AUDIENCE;
* ``http_auth._check_token`` reads **only** MIND_MEM_TOKEN.

So an operator who configured admin auth got, at startup,
``_enforce_fail_closed`` returning early on ``_auth_is_configured()`` -- never
reaching its loopback check, so ``--host 0.0.0.0`` was permitted -- and at
request time ``verify_token`` seeing ``expected is None`` and taking the
"no auth configured, operator opted into anonymous localhost" branch.

The opt-in means "I accept NO auth because I am on loopback". If any auth
mechanism IS configured that statement is false, so the opt-in must not be
honoured. ``_auth_is_configured``'s own docstring already states the principle
this violated: "A gate must not certify a property it did not check, whichever
direction the mismatch happens to fail in today."

This grants the admin token no new reach: it does not become a user credential.
It only stops anonymous access being handed out on a server whose operator
configured authentication.
"""

from __future__ import annotations

import importlib

import pytest

OPT_IN = "MIND_MEM_ALLOW_UNAUTHENTICATED_LOCALHOST"
OTHERS = ("MIND_MEM_ADMIN_TOKEN", "MIND_MEM_API_KEY_DB", "OIDC_ISSUER", "OIDC_AUDIENCE")


@pytest.fixture
def auth(monkeypatch):
    for var in ("MIND_MEM_TOKEN", OPT_IN, *OTHERS):
        monkeypatch.delenv(var, raising=False)
    from mind_mem.mcp.infra import http_auth

    return importlib.reload(http_auth)


def test_positive_control_optin_still_works_with_no_auth_at_all(auth, monkeypatch):
    """The opt-in must KEEP working in the case it exists for: genuinely no auth.

    Without this control the assertions below would pass just as happily if the
    opt-in had been broken outright.
    """
    monkeypatch.setenv(OPT_IN, "1")
    assert auth.verify_token({}) is True


def test_admin_token_configured_revokes_the_anonymous_optin(auth, monkeypatch):
    monkeypatch.setenv("MIND_MEM_ADMIN_TOKEN", "the-real-admin-token")
    monkeypatch.setenv(OPT_IN, "1")
    assert auth.verify_token({}) is False
    assert auth.verify_token({"Authorization": "Bearer totally-wrong"}) is False


@pytest.mark.parametrize("var", ["MIND_MEM_API_KEY_DB", "OIDC_ISSUER"])
def test_every_other_configured_mechanism_revokes_it_too(auth, monkeypatch, var):
    monkeypatch.setenv(var, "configured")
    if var == "OIDC_ISSUER":
        monkeypatch.setenv("OIDC_AUDIENCE", "aud")
    monkeypatch.setenv(OPT_IN, "1")
    assert auth.verify_token({}) is False


def test_empty_admin_token_does_not_revoke_it(auth, monkeypatch):
    """Present-but-empty is NOT a usable credential, matching _auth_is_configured's
    truthiness test. An exported-but-empty var must not silently 401 a loopback dev
    server that has no auth at all."""
    monkeypatch.setenv("MIND_MEM_ADMIN_TOKEN", "")
    monkeypatch.setenv(OPT_IN, "1")
    assert auth.verify_token({}) is True


def test_empty_user_token_is_normalized_to_unconfigured(auth, monkeypatch):
    """An empty user token follows the same truthiness rule as startup."""
    monkeypatch.setenv("MIND_MEM_TOKEN", "")
    monkeypatch.setenv(OPT_IN, "1")
    assert auth._check_token() is None
    assert auth.auth_is_configured() is False
    assert auth.verify_token({}) is True


def test_real_user_token_still_authenticates_normally(auth, monkeypatch):
    monkeypatch.setenv("MIND_MEM_TOKEN", "user-token-value")
    monkeypatch.setenv("MIND_MEM_ADMIN_TOKEN", "admin-token-value")
    assert auth.verify_token({"Authorization": "Bearer user-token-value"}) is True
    assert auth.verify_token({"Authorization": "Bearer admin-token-value"}) is False
    assert auth.verify_token({}) is False
