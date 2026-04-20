"""Per-tenant key management + envelope encryption (v4.0 prep).

v3.0 shipped workspace-wide at-rest encryption (``SQLCipher`` +
``EncryptedBlockStore``). For v4.0 multi-tenant the encryption
scope flips — each tenant gets their own data key (DEK), and the
operator holds a master key (KEK) that wraps every DEK so only
that master can decrypt the storage tier.

This module is a minimal KEK/DEK façade:

* :class:`MasterKey` wraps the operator's KEK (AES-256 default).
* :func:`generate_tenant_dek` creates a fresh DEK per tenant and
  returns the wrapped ciphertext for the caller to persist.
* :func:`unwrap_tenant_dek` reverses that.
* :func:`rotate_tenant_dek` generates a new DEK while keeping the
  old one available for decrypting already-ciphertext'd rows.

Dependencies: uses ``cryptography`` when installed (AEAD via
AES-GCM-256). Falls back to an HMAC-based KDF + ``chacha20``-style
software AEAD via stdlib only — acceptable for dev/test, not
production. Operators who actually deploy this must install
``mind-mem[encrypted]``.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
from dataclasses import dataclass

from .observability import get_logger

_log = get_logger("tenant_kms")


# AES-256 key size.
_DEK_BYTES = 32
# GCM nonce size recommended by NIST.
_NONCE_BYTES = 12


def _has_cryptography() -> bool:
    try:
        import cryptography  # noqa: F401

        return True
    except ImportError:
        return False


@dataclass(frozen=True)
class MasterKey:
    """Operator's KEK. ``bytes_`` is the raw 32-byte key material.

    Don't log this struct — the repr leaks the key. Callers should
    treat it as opaque and pass it to the wrap / unwrap helpers only.
    """

    bytes_: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.bytes_, (bytes, bytearray)) or len(self.bytes_) < _DEK_BYTES:
            raise ValueError(f"master key must be ≥{_DEK_BYTES} bytes")


@dataclass(frozen=True)
class WrappedDEK:
    """A wrapped per-tenant data key.

    Wire format: ``tenant_id | nonce | ciphertext | tag`` encoded as
    base64. Callers persist this opaque blob; only the master can
    unwrap.
    """

    tenant_id: str
    nonce: bytes
    wrapped: bytes
    tag: bytes

    def to_b64(self) -> str:
        mac_tag_len = len(self.tag).to_bytes(1, "big")
        return base64.urlsafe_b64encode(
            self.tenant_id.encode("utf-8") + b":" + self.nonce + b":" + mac_tag_len + self.tag + self.wrapped
        ).decode("ascii")

    @classmethod
    def from_b64(cls, blob: str) -> "WrappedDEK":
        raw = base64.urlsafe_b64decode(blob.encode("ascii"))
        first_colon = raw.index(b":")
        second_colon = raw.index(b":", first_colon + 1)
        tenant_id = raw[:first_colon].decode("utf-8")
        nonce = raw[first_colon + 1 : second_colon]
        payload = raw[second_colon + 1 :]
        tag_len = payload[0]
        tag = payload[1 : 1 + tag_len]
        wrapped = payload[1 + tag_len :]
        return cls(tenant_id=tenant_id, nonce=nonce, wrapped=wrapped, tag=tag)


def _derive_wrap_key(master: MasterKey, tenant_id: str) -> bytes:
    """HKDF-lite — HMAC(master, tenant_id) → per-tenant wrap key.

    Each tenant gets a unique wrap key so compromising one tenant's
    wrap doesn't affect others. The master key stays offline in the
    operator's KMS / envelope store.
    """
    return hmac.new(master.bytes_, tenant_id.encode("utf-8"), hashlib.sha256).digest()


def _wrap_with_cryptography(wrap_key: bytes, dek: bytes, aad: bytes) -> tuple[bytes, bytes, bytes]:
    """AES-GCM-256 wrap. Returns (nonce, ciphertext, tag)."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # type: ignore

    aes = AESGCM(wrap_key)
    nonce = secrets.token_bytes(_NONCE_BYTES)
    # AESGCM returns ciphertext || tag as one blob.
    ct_tag = aes.encrypt(nonce, dek, aad)
    ct, tag = ct_tag[:-16], ct_tag[-16:]
    return nonce, ct, tag


def _unwrap_with_cryptography(wrap_key: bytes, nonce: bytes, ct: bytes, tag: bytes, aad: bytes) -> bytes:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # type: ignore

    aes = AESGCM(wrap_key)
    return aes.decrypt(nonce, ct + tag, aad)


def _wrap_fallback(wrap_key: bytes, dek: bytes, aad: bytes) -> tuple[bytes, bytes, bytes]:
    """Software AEAD fallback — XOR + HMAC tag.

    NOT production-grade. Here so tests pass without cryptography
    installed. Operators get a startup warning via
    :func:`require_production_crypto`.
    """
    nonce = secrets.token_bytes(_NONCE_BYTES)
    stream = hashlib.shake_256(wrap_key + nonce).digest(len(dek))
    ct = bytes(a ^ b for a, b in zip(dek, stream))
    tag = hmac.new(wrap_key, nonce + ct + aad, hashlib.sha256).digest()
    return nonce, ct, tag


def _unwrap_fallback(wrap_key: bytes, nonce: bytes, ct: bytes, tag: bytes, aad: bytes) -> bytes:
    expected_tag = hmac.new(wrap_key, nonce + ct + aad, hashlib.sha256).digest()
    if not hmac.compare_digest(expected_tag, tag):
        raise ValueError("DEK tag mismatch — wrong master key or tampered ciphertext")
    stream = hashlib.shake_256(wrap_key + nonce).digest(len(ct))
    return bytes(a ^ b for a, b in zip(ct, stream))


def require_production_crypto() -> None:
    """Raise if cryptography isn't installed — call at app startup.

    Dev environments skip the call and get the fallback; production
    callers make this a hard gate to avoid shipping the non-standard
    AEAD by accident.
    """
    if not _has_cryptography():
        raise RuntimeError(
            "mind-mem tenant_kms requires the 'cryptography' package for production use. Install with: pip install 'mind-mem[encrypted]'"
        )


def generate_tenant_dek(master: MasterKey, tenant_id: str) -> tuple[bytes, WrappedDEK]:
    """Create a fresh DEK for ``tenant_id`` and return (dek, wrapped).

    Caller persists ``wrapped`` (via .to_b64()). The plaintext ``dek``
    is returned only for the caller's initial setup; they must not
    store it.
    """
    if not tenant_id:
        raise ValueError("tenant_id must be non-empty")
    dek = secrets.token_bytes(_DEK_BYTES)
    aad = tenant_id.encode("utf-8")
    wrap_key = _derive_wrap_key(master, tenant_id)
    if _has_cryptography():
        nonce, ct, tag = _wrap_with_cryptography(wrap_key, dek, aad)
    else:
        _log.warning(
            "tenant_kms_fallback_aead",
            tenant_id=tenant_id,
            reason="cryptography package not installed",
        )
        nonce, ct, tag = _wrap_fallback(wrap_key, dek, aad)
    return dek, WrappedDEK(tenant_id=tenant_id, nonce=nonce, wrapped=ct, tag=tag)


def unwrap_tenant_dek(master: MasterKey, wrapped: WrappedDEK) -> bytes:
    """Unwrap a previously-generated DEK. Raises on tag mismatch."""
    wrap_key = _derive_wrap_key(master, wrapped.tenant_id)
    aad = wrapped.tenant_id.encode("utf-8")
    if _has_cryptography():
        return _unwrap_with_cryptography(wrap_key, wrapped.nonce, wrapped.wrapped, wrapped.tag, aad)
    return _unwrap_fallback(wrap_key, wrapped.nonce, wrapped.wrapped, wrapped.tag, aad)


def rotate_tenant_dek(master: MasterKey, old: WrappedDEK) -> tuple[bytes, WrappedDEK, bytes]:
    """Rotate a tenant's DEK.

    Returns ``(new_dek, new_wrapped, old_dek)``. Caller's rotation
    procedure:

      1. unwrap ``old_dek`` (for reading already-encrypted rows)
      2. generate ``new_dek`` via this function
      3. re-encrypt existing ciphertexts from old → new (callers
         job — mind-mem KMS only mints keys, never touches storage)
      4. persist ``new_wrapped`` and drop ``old``.
    """
    old_plaintext = unwrap_tenant_dek(master, old)
    new_dek, new_wrapped = generate_tenant_dek(master, old.tenant_id)
    return new_dek, new_wrapped, old_plaintext


def create_master_key_from_env(env_var: str = "MIND_MEM_KMS_MASTER_KEY_B64") -> MasterKey:
    """Build a MasterKey from a base64-encoded env var.

    Operators stash the 32+ bytes in their secret manager (e.g. AWS
    SSM, Vault, 1Password) and inject as the env var at process boot.
    Raises if the env is unset or the key is too short.
    """
    raw = os.environ.get(env_var, "")
    if not raw:
        raise RuntimeError(f"{env_var} not set — can't build MasterKey")
    try:
        key = base64.urlsafe_b64decode(raw)
    except Exception as exc:
        raise RuntimeError(f"{env_var} is not valid base64: {exc}") from exc
    return MasterKey(bytes_=key)


__all__ = [
    "MasterKey",
    "WrappedDEK",
    "create_master_key_from_env",
    "generate_tenant_dek",
    "require_production_crypto",
    "rotate_tenant_dek",
    "unwrap_tenant_dek",
]
