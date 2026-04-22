"""v4.0 prep — per-tenant KMS envelope encryption."""

from __future__ import annotations

import base64
import secrets

import pytest

from mind_mem.tenant_kms import (
    MasterKey,
    WrappedDEK,
    create_master_key_from_env,
    generate_tenant_dek,
    rotate_tenant_dek,
    unwrap_tenant_dek,
)


@pytest.fixture
def master() -> MasterKey:
    return MasterKey(bytes_=secrets.token_bytes(32))


class TestMasterKey:
    def test_rejects_short_key(self) -> None:
        with pytest.raises(ValueError, match="≥32 bytes"):
            MasterKey(bytes_=b"too short")

    def test_rejects_non_bytes(self) -> None:
        with pytest.raises(ValueError):
            MasterKey(bytes_="not bytes")  # type: ignore[arg-type]


class TestGenerateAndUnwrap:
    def test_roundtrip(self, master: MasterKey) -> None:
        dek, wrapped = generate_tenant_dek(master, "acme")
        unwrapped = unwrap_tenant_dek(master, wrapped)
        assert unwrapped == dek
        assert len(dek) == 32

    def test_each_tenant_gets_unique_dek(self, master: MasterKey) -> None:
        dek_a, _ = generate_tenant_dek(master, "acme")
        dek_b, _ = generate_tenant_dek(master, "globex")
        assert dek_a != dek_b

    def test_same_tenant_twice_different_deks(self, master: MasterKey) -> None:
        """Deks are random; not derived from tenant_id alone."""
        dek1, _ = generate_tenant_dek(master, "acme")
        dek2, _ = generate_tenant_dek(master, "acme")
        assert dek1 != dek2

    def test_empty_tenant_id_rejected(self, master: MasterKey) -> None:
        with pytest.raises(ValueError):
            generate_tenant_dek(master, "")

    def test_unwrap_with_wrong_master_fails(self, master: MasterKey) -> None:
        wrong = MasterKey(bytes_=secrets.token_bytes(32))
        _, wrapped = generate_tenant_dek(master, "acme")
        with pytest.raises(Exception):
            unwrap_tenant_dek(wrong, wrapped)


class TestWrappedDEKSerialisation:
    def test_to_from_b64_roundtrip(self, master: MasterKey) -> None:
        _, wrapped = generate_tenant_dek(master, "acme")
        blob = wrapped.to_b64()
        restored = WrappedDEK.from_b64(blob)
        assert restored.tenant_id == wrapped.tenant_id
        assert restored.nonce == wrapped.nonce
        assert restored.wrapped == wrapped.wrapped
        assert restored.tag == wrapped.tag


class TestRotateTenantDEK:
    def test_rotation_produces_new_key_and_preserves_old(self, master: MasterKey) -> None:
        old_dek, old_wrapped = generate_tenant_dek(master, "acme")
        new_dek, new_wrapped, returned_old = rotate_tenant_dek(master, old_wrapped)
        assert new_dek != old_dek
        assert returned_old == old_dek
        # New wrapped decodes back to new_dek.
        assert unwrap_tenant_dek(master, new_wrapped) == new_dek
        # Old wrapped still decodes to old_dek (caller needs both
        # during the re-encrypt window).
        assert unwrap_tenant_dek(master, old_wrapped) == old_dek


class TestCreateMasterKeyFromEnv:
    def test_reads_b64_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        raw = secrets.token_bytes(32)
        monkeypatch.setenv("TEST_MASTER", base64.urlsafe_b64encode(raw).decode())
        mk = create_master_key_from_env("TEST_MASTER")
        assert mk.bytes_ == raw

    def test_missing_env_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TEST_MASTER", raising=False)
        with pytest.raises(RuntimeError, match="not set"):
            create_master_key_from_env("TEST_MASTER")

    def test_non_base64_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_MASTER", "!!!not base64!!!")
        with pytest.raises((RuntimeError, ValueError)):
            create_master_key_from_env("TEST_MASTER")


# v3.6.2: legacy-format compatibility
class TestLegacyWireFormatCompat:
    """WrappedDEK.from_b64 must accept both the pre-v3.5.0-rc colon
    format AND the current length-prefixed format so an operator who
    persisted blobs under the v4.0-prep preview still decrypts them
    after upgrade.
    """

    def _manual_legacy_blob(self, tenant_id: str, nonce: bytes, tag: bytes, wrapped: bytes) -> str:
        import base64

        payload = tenant_id.encode("utf-8") + b":" + nonce + b":" + len(tag).to_bytes(1, "big") + tag + wrapped
        return base64.urlsafe_b64encode(payload).decode("ascii")

    def test_legacy_colon_blob_parses(self):
        from mind_mem.tenant_kms import WrappedDEK

        blob = self._manual_legacy_blob(
            tenant_id="acme",
            nonce=b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c",
            tag=b"\xde\xad\xbe\xef" * 4,
            wrapped=b"wrapped-ciphertext-bytes" * 2,
        )
        d = WrappedDEK.from_b64(blob)
        assert d.tenant_id == "acme"
        assert d.nonce == b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c"
        assert len(d.tag) == 16
        assert d.wrapped.startswith(b"wrapped-ciphertext-bytes")

    def test_round_trip_uses_length_prefixed(self):
        from mind_mem.tenant_kms import WrappedDEK

        src = WrappedDEK(
            tenant_id="tenant-x",
            nonce=b"\x3a" * 12,  # 0x3a == ':' — would collide in legacy format
            wrapped=b"w" * 32,
            tag=b"t" * 16,
        )
        d = WrappedDEK.from_b64(src.to_b64())
        assert d.tenant_id == "tenant-x"
        assert d.nonce == b"\x3a" * 12
        assert d.wrapped == b"w" * 32
        assert d.tag == b"t" * 16

    def test_garbage_raises_clear_error(self):
        import pytest

        from mind_mem.tenant_kms import WrappedDEK

        with pytest.raises(ValueError, match="did not match"):
            WrappedDEK.from_b64("aaaaaaaa")  # valid b64, invalid both formats
