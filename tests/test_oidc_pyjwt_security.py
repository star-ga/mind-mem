# Copyright 2026 STARGA, Inc.
"""Security controls for the PyJWT-backed OIDC verification boundary."""

from __future__ import annotations

import json
import time
from typing import Any

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ec, rsa
from jwt.algorithms import ECAlgorithm, RSAAlgorithm

from mind_mem.api.auth import _ALLOWED_JWT_KEYS, AuthError, OIDCConfig, OIDCProvider

_ISSUER = "https://issuer.example"
_AUDIENCE = "mind-mem-api"
_KID = "current-signing-key"


def _provider(jwk: dict) -> OIDCProvider:
    provider = OIDCProvider(
        OIDCConfig(
            issuer=_ISSUER,
            client_id="client",
            client_secret="secret",
            audience=_AUDIENCE,
            jwks_uri="https://issuer.example/jwks",
        )
    )
    provider._jwks = {"keys": [jwk]}
    return provider


def _claims(**changes: object) -> dict[str, object]:
    now = int(time.time())
    claims: dict[str, object] = {
        "iss": _ISSUER,
        "aud": _AUDIENCE,
        "sub": "subject",
        "iat": now,
        "exp": now + 300,
    }
    claims.update(changes)
    return claims


@pytest.fixture(scope="module")
def rsa_material() -> tuple[Any, dict[str, Any]]:
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_jwk = json.loads(RSAAlgorithm.to_jwk(private.public_key()))
    public_jwk.update({"kid": _KID, "use": "sig", "alg": "RS256", "key_ops": ["verify"]})
    return private, public_jwk


@pytest.fixture(scope="module")
def ec_material() -> tuple[Any, dict[str, Any]]:
    private = ec.generate_private_key(ec.SECP256R1())
    public_jwk = json.loads(ECAlgorithm.to_jwk(private.public_key()))
    public_jwk.update({"kid": _KID, "use": "sig", "alg": "ES256", "key_ops": ["verify"]})
    return private, public_jwk


def _encode(claims: dict[str, object], key: Any, algorithm: str, kid: str = _KID) -> str:
    return jwt.encode(claims, key, algorithm=algorithm, headers={"kid": kid})


def test_algorithm_allowlist_is_unchanged() -> None:
    assert set(_ALLOWED_JWT_KEYS) == {"RS256", "RS384", "RS512", "ES256", "ES384", "ES512"}


@pytest.mark.parametrize("algorithm", sorted(_ALLOWED_JWT_KEYS))
def test_each_allowlisted_algorithm_verifies_a_real_asymmetric_token(
    algorithm: str,
    rsa_material: tuple[Any, dict[str, Any]],
) -> None:
    private, rsa_jwk = rsa_material
    if algorithm.startswith("RS"):
        jwk = dict(rsa_jwk, alg=algorithm)
    else:
        curves = {
            "ES256": ec.SECP256R1,
            "ES384": ec.SECP384R1,
            "ES512": ec.SECP521R1,
        }
        private = ec.generate_private_key(curves[algorithm]())
        jwk = json.loads(ECAlgorithm.to_jwk(private.public_key()))
        jwk.update({"kid": _KID, "use": "sig", "alg": algorithm, "key_ops": ["verify"]})
    claims = _provider(jwk).verify(_encode(_claims(), private, algorithm))
    assert claims["sub"] == "subject"


def test_hs_algorithm_cannot_confuse_an_es_public_key(
    ec_material: tuple[Any, dict[str, Any]],
) -> None:
    _, ec_jwk = ec_material
    token = jwt.encode(
        _claims(),
        "attacker-secret-that-is-at-least-32-bytes",
        algorithm="HS256",
        headers={"kid": _KID},
    )
    with pytest.raises(AuthError) as exc_info:
        _provider(ec_jwk).verify(token)
    assert exc_info.value.code == "invalid_token"


def test_unsigned_none_algorithm_is_rejected(rsa_material: tuple[object, dict]) -> None:
    _, rsa_jwk = rsa_material
    token = jwt.encode(_claims(), "", algorithm="none", headers={"kid": _KID})
    with pytest.raises(AuthError) as exc_info:
        _provider(rsa_jwk).verify(token)
    assert exc_info.value.code == "invalid_token"


def test_header_algorithm_cannot_select_the_wrong_key_family(
    rsa_material: tuple[Any, dict[str, Any]],
    ec_material: tuple[Any, dict[str, Any]],
) -> None:
    private, _ = rsa_material
    _, ec_jwk = ec_material
    token = _encode(_claims(), private, "RS256")
    with pytest.raises(AuthError, match="0 compatible keys"):
        _provider(ec_jwk).verify(token)


def test_duplicate_kid_is_ambiguous_and_rejected(rsa_material: tuple[Any, dict[str, Any]]) -> None:
    private, jwk = rsa_material
    provider = _provider(jwk)
    provider._jwks = {"keys": [jwk, dict(jwk)]}
    with pytest.raises(AuthError, match="2 compatible keys"):
        provider.verify(_encode(_claims(), private, "RS256"))


def test_jwk_algorithm_must_match_the_token_header(rsa_material: tuple[Any, dict[str, Any]]) -> None:
    private, jwk = rsa_material
    with pytest.raises(AuthError, match="0 compatible keys"):
        _provider(dict(jwk, alg="RS512")).verify(_encode(_claims(), private, "RS256"))


def test_token_without_kid_uses_one_unambiguous_compatible_key(
    rsa_material: tuple[Any, dict[str, Any]],
) -> None:
    private, jwk = rsa_material
    token = jwt.encode(_claims(), private, algorithm="RS256")
    assert _provider(jwk).verify(token)["sub"] == "subject"


@pytest.mark.parametrize(
    ("claim", "code"),
    [
        ("exp", "token_expired"),
        ("nbf", "invalid_token"),
        ("iss", "wrong_issuer"),
        ("aud", "wrong_audience"),
    ],
)
def test_registered_claim_failures_remain_fail_closed(
    claim: str,
    code: str,
    rsa_material: tuple[Any, dict[str, Any]],
) -> None:
    private, jwk = rsa_material
    # Build temporal values when the case executes, rather than at collection.
    # The full matrix can spend longer than the five-minute nbf horizon between
    # collection and this test, turning the intended future token into a valid one.
    now = int(time.time())
    change: dict[str, object] = {
        "exp": now - 10,
        "nbf": now + 300,
        "iss": "https://attacker.example",
        "aud": "other-service",
    }
    with pytest.raises(AuthError) as exc_info:
        _provider(jwk).verify(_encode(_claims(**{claim: change[claim]}), private, "RS256"))
    assert exc_info.value.code == code


def test_malformed_token_is_wrapped_as_auth_error(rsa_material: tuple[Any, dict[str, Any]]) -> None:
    _, jwk = rsa_material
    with pytest.raises(AuthError) as exc_info:
        _provider(jwk).verify("not-a-jwt")
    assert exc_info.value.code == "invalid_token"


def test_missing_exp_preserves_the_existing_optional_exp_contract(rsa_material: tuple[Any, dict[str, Any]]) -> None:
    private, jwk = rsa_material
    claims = _claims()
    claims.pop("exp")
    assert _provider(jwk).verify(_encode(claims, private, "RS256"))["sub"] == "subject"


def test_missing_iat_preserves_the_existing_optional_iat_contract(
    rsa_material: tuple[Any, dict[str, Any]],
) -> None:
    private, jwk = rsa_material
    claims = _claims()
    claims.pop("iat")
    assert _provider(jwk).verify(_encode(claims, private, "RS256"))["sub"] == "subject"


def test_future_iat_is_rejected_with_typed_auth_error(rsa_material: tuple[Any, dict[str, Any]]) -> None:
    private, jwk = rsa_material
    with pytest.raises(AuthError) as exc_info:
        _provider(jwk).verify(_encode(_claims(iat=int(time.time()) + 3600), private, "RS256"))
    assert exc_info.value.code == "invalid_token"


@pytest.mark.parametrize("iat", ["not-an-integer", None], ids=["string", "null"])
def test_malformed_iat_is_rejected_with_typed_auth_error(
    iat: object,
    rsa_material: tuple[Any, dict[str, Any]],
) -> None:
    private, jwk = rsa_material
    with pytest.raises(AuthError) as exc_info:
        _provider(jwk).verify(_encode(_claims(iat=iat), private, "RS256"))
    assert exc_info.value.code == "invalid_token"


@pytest.mark.parametrize(("claim", "code"), [("aud", "wrong_audience"), ("iss", "wrong_issuer")])
def test_missing_required_identity_claim_is_mapped(
    claim: str,
    code: str,
    rsa_material: tuple[Any, dict[str, Any]],
) -> None:
    private, jwk = rsa_material
    claims = _claims()
    claims.pop(claim)
    with pytest.raises(AuthError) as exc_info:
        _provider(jwk).verify(_encode(claims, private, "RS256"))
    assert exc_info.value.code == code


def test_invalid_signature_is_wrapped_as_auth_error(rsa_material: tuple[Any, dict[str, Any]]) -> None:
    _, jwk = rsa_material
    attacker_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    with pytest.raises(AuthError) as exc_info:
        _provider(jwk).verify(_encode(_claims(), attacker_key, "RS256"))
    assert exc_info.value.code == "invalid_token"


def test_unknown_kid_is_rejected(rsa_material: tuple[Any, dict[str, Any]]) -> None:
    private, jwk = rsa_material
    token = _encode(_claims(), private, "RS256", kid="unknown")
    with pytest.raises(AuthError, match="0 compatible keys"):
        _provider(jwk).verify(token)
