# Copyright 2026 STARGA, Inc.
"""Acceptance gate for the ``tenant_kms`` wiring (restoration slice).

``tenant_kms`` shipped a complete KEK/DEK envelope façade — AES-256-GCM when
``cryptography`` is installed, a documented non-AEAD fallback when it is not —
kept its own unit tests, and was called by **nothing**. Meanwhile the shipped
at-rest path (:mod:`mind_mem.encryption`) hand-rolls an HMAC-SHA256 keystream
under encrypt-then-MAC and says so in its own docstring. The product owned a
real AEAD and did not use it.

This file pins the substitution: with the operator opting in, the at-rest path
seals new records with AES-256-GCM under a data key that ``tenant_kms`` minted
and wrapped. Wired SINGLE-TENANT (``tenant_id="default"``) — the multi-tenant
surface therefore arrives already exercised.

Six contracts, one class each:

1. **the default is untouched** — with the env var unset the manager builds no
   envelope cipher, writes ``MMENC1`` records, and the probe leaves no trace.
   The comparison's teeth are checked by a positive control that makes the
   very same assertions FAIL once the opt-in is supplied;
2. **the opt-in actually calls tenant_kms** — the key the file cipher uses is
   provably the one behind the wrapped-DEK blob on disk, recovered through
   ``tenant_kms.unwrap_tenant_dek``. This is the assertion that fails if the
   ``_kms_envelope_cipher(...)`` call is deleted from ``EncryptionManager``;
3. **both formats keep opening** — a workspace that flips the mode on mid-life
   still reads every legacy record it already wrote;
4. **it is an AEAD, not decoration** — a flipped byte anywhere in the record
   fails authentication, and a wrong master key cannot open it;
5. **absence degrades, it never crashes or corrupts** — no ``cryptography``
   means the legacy path, with a warning; and a wrapped-DEK blob that does not
   unwrap is never silently replaced;
6. **no silent downgrade** — passphrase rotation refuses rather than rewriting
   AES-GCM records with the keystream construction.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import pytest

from mind_mem import tenant_kms
from mind_mem.encryption import _KMS_DEK_FILE, _MAGIC, _MAGIC_KMS, EncryptionManager

_PASSPHRASE = "test-passphrase-not-a-secret"
_PLAINTEXT = b"# Decisions\n\nD-20260901-001 the corpus bytes that must stay confidential.\n"

pytest.importorskip("cryptography", reason="envelope mode requires the optional 'cryptography' package")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _master_b64() -> str:
    """A deterministic 32-byte KEK. Test material only — never a real key."""
    return base64.urlsafe_b64encode(bytes(range(32))).decode("ascii")


def _workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, flag: bool) -> str:
    """Workspace whose ``mind-mem.json`` carries the v4 flag, config resolved."""
    cfg = {"v4": {"tenant_kms": {"enabled": flag}}}
    (tmp_path / "mind-mem.json").write_text(json.dumps(cfg), encoding="utf-8")
    monkeypatch.setenv("MIND_MEM_CONFIG", str(tmp_path / "mind-mem.json"))
    return str(tmp_path)


def _opt_in(monkeypatch: pytest.MonkeyPatch, master_b64: str | None = None) -> None:
    monkeypatch.setenv("MIND_MEM_KMS_MASTER_KEY_B64", master_b64 or _master_b64())


@pytest.fixture(autouse=True)
def _no_ambient_master(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never inherit an operator's real master key into a test."""
    monkeypatch.delenv("MIND_MEM_KMS_MASTER_KEY_B64", raising=False)


# ===========================================================================
# 1. The default path is untouched
# ===========================================================================


class TestDefaultUnchanged:
    def test_no_env_var_means_no_envelope(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        ws = _workspace(tmp_path, monkeypatch, flag=True)  # flag ON, env absent
        mgr = EncryptionManager(ws, _PASSPHRASE)
        assert mgr._kms is None
        record = mgr.encrypt(_PLAINTEXT)
        assert record[:6] == _MAGIC
        assert mgr.decrypt(record) == _PLAINTEXT

    def test_env_var_without_the_flag_means_no_envelope(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        ws = _workspace(tmp_path, monkeypatch, flag=False)
        _opt_in(monkeypatch)
        mgr = EncryptionManager(ws, _PASSPHRASE)
        assert mgr._kms is None
        assert mgr.encrypt(_PLAINTEXT)[:6] == _MAGIC

    def test_off_writes_no_key_material(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A DEK blob appearing in a default install would itself be a change."""
        ws = _workspace(tmp_path, monkeypatch, flag=False)
        EncryptionManager(ws, _PASSPHRASE).encrypt(_PLAINTEXT)
        assert not os.path.exists(os.path.join(ws, ".mind-mem-keys", _KMS_DEK_FILE))

    def test_the_off_comparison_has_teeth(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Positive control.

        Every assertion above is of the form "nothing happened", and a test
        that cannot distinguish is not a test. Supply the opt-in and each of
        them must now be FALSE — that is what proves they were measuring the
        wiring rather than measuring nothing.
        """
        ws = _workspace(tmp_path, monkeypatch, flag=True)
        _opt_in(monkeypatch)
        mgr = EncryptionManager(ws, _PASSPHRASE)
        assert mgr._kms is not None
        assert mgr.encrypt(_PLAINTEXT)[:6] != _MAGIC
        assert os.path.exists(os.path.join(ws, ".mind-mem-keys", _KMS_DEK_FILE))


# ===========================================================================
# 2. The opt-in genuinely calls tenant_kms
# ===========================================================================


class TestEnvelopeIsWired:
    def test_records_are_aes_gcm_under_the_kms_minted_dek(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The load-bearing assertion of this file.

        It does not merely check that a different magic appears — it recovers
        the data key from the on-disk wrapped blob **through tenant_kms** and
        decrypts the record with it directly. That can only pass if the file
        cipher's key is the key ``generate_tenant_dek`` minted, so deleting the
        ``_kms_envelope_cipher(...)`` call from ``EncryptionManager.__init__``
        (or swapping in any other key source) fails it.
        """
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        ws = _workspace(tmp_path, monkeypatch, flag=True)
        _opt_in(monkeypatch)

        record = EncryptionManager(ws, _PASSPHRASE).encrypt(_PLAINTEXT)
        assert record[:6] == _MAGIC_KMS

        blob = Path(ws, ".mind-mem-keys", _KMS_DEK_FILE).read_text(encoding="utf-8").strip()
        wrapped = tenant_kms.WrappedDEK.from_b64(blob)
        assert wrapped.tenant_id == "default"

        master = tenant_kms.MasterKey(bytes_=base64.urlsafe_b64decode(_master_b64()))
        dek = tenant_kms.unwrap_tenant_dek(master, wrapped)
        assert len(dek) == 32

        nonce, ct = record[6:18], record[18:]
        assert AESGCM(dek).decrypt(nonce, ct, _MAGIC_KMS) == _PLAINTEXT

    def test_round_trip_through_the_manager(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        ws = _workspace(tmp_path, monkeypatch, flag=True)
        _opt_in(monkeypatch)
        mgr = EncryptionManager(ws, _PASSPHRASE)
        assert mgr.decrypt(mgr.encrypt(_PLAINTEXT)) == _PLAINTEXT
        assert mgr.decrypt(mgr.encrypt(b"")) == b""

    def test_the_dek_is_stable_across_managers(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A second manager must reuse the wrapped DEK, not mint a new one."""
        ws = _workspace(tmp_path, monkeypatch, flag=True)
        _opt_in(monkeypatch)
        blob_path = Path(ws, ".mind-mem-keys", _KMS_DEK_FILE)
        record = EncryptionManager(ws, _PASSPHRASE).encrypt(_PLAINTEXT)
        first = blob_path.read_text(encoding="utf-8")
        assert EncryptionManager(ws, _PASSPHRASE).decrypt(record) == _PLAINTEXT
        assert blob_path.read_text(encoding="utf-8") == first

    def test_file_surface_routes_on_the_new_magic(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """``is_encrypted`` / ``encrypt_file`` must recognise envelope records.

        A bare ``== _MAGIC`` header compare reads one as plaintext, which makes
        ``encrypt_file`` double-encrypt and ``decrypt_file`` hand ciphertext
        back to the caller.
        """
        ws = _workspace(tmp_path, monkeypatch, flag=True)
        _opt_in(monkeypatch)
        target = Path(ws, "corpus.md")
        target.write_bytes(_PLAINTEXT)

        mgr = EncryptionManager(ws, _PASSPHRASE)
        mgr.encrypt_file(str(target))
        sealed = target.read_bytes()
        assert sealed[:6] == _MAGIC_KMS
        assert mgr.is_encrypted(str(target))

        mgr.encrypt_file(str(target))  # idempotent — must NOT double-encrypt
        assert target.read_bytes() == sealed
        assert mgr.decrypt_file(str(target)) == _PLAINTEXT


# ===========================================================================
# 3. Mixed-format workspaces
# ===========================================================================


class TestBothFormatsOpen:
    def test_legacy_records_survive_the_switch(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        ws = _workspace(tmp_path, monkeypatch, flag=True)
        legacy = EncryptionManager(ws, _PASSPHRASE).encrypt(_PLAINTEXT)
        assert legacy[:6] == _MAGIC

        _opt_in(monkeypatch)
        mgr = EncryptionManager(ws, _PASSPHRASE)
        assert mgr._kms is not None
        assert mgr.decrypt(legacy) == _PLAINTEXT  # old record, new mode
        assert mgr.encrypt(_PLAINTEXT)[:6] == _MAGIC_KMS  # new record, new format

    def test_envelope_record_refuses_to_open_without_the_mode(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Fail closed, and say which knob is missing — never guess a key."""
        ws = _workspace(tmp_path, monkeypatch, flag=True)
        _opt_in(monkeypatch)
        record = EncryptionManager(ws, _PASSPHRASE).encrypt(_PLAINTEXT)

        monkeypatch.delenv("MIND_MEM_KMS_MASTER_KEY_B64")
        plain_mgr = EncryptionManager(ws, _PASSPHRASE)
        with pytest.raises(ValueError, match="MMKMS1"):
            plain_mgr.decrypt(record)


# ===========================================================================
# 4. It is a real AEAD
# ===========================================================================


class TestAuthentication:
    @pytest.mark.parametrize("offset", [0, 3, 7, 20, -1])
    def test_any_flipped_byte_fails_authentication(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, offset: int) -> None:
        ws = _workspace(tmp_path, monkeypatch, flag=True)
        _opt_in(monkeypatch)
        mgr = EncryptionManager(ws, _PASSPHRASE)
        record = bytearray(mgr.encrypt(_PLAINTEXT))
        record[offset] ^= 0x01
        with pytest.raises(ValueError):
            mgr.decrypt(bytes(record))

    def test_a_different_master_key_cannot_open_the_workspace(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        ws = _workspace(tmp_path, monkeypatch, flag=True)
        _opt_in(monkeypatch)
        EncryptionManager(ws, _PASSPHRASE).encrypt(_PLAINTEXT)

        _opt_in(monkeypatch, base64.urlsafe_b64encode(bytes(32)).decode("ascii"))
        with pytest.raises(ValueError, match="could not be unwrapped"):
            EncryptionManager(ws, _PASSPHRASE)


# ===========================================================================
# 5. Absence degrades; damage is never overwritten
# ===========================================================================


class TestDegradesCleanly:
    def test_missing_cryptography_falls_back_to_the_legacy_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """No AEAD available must mean the old behaviour, not a crash.

        ``cryptography`` is an optional dependency and the default install is
        zero-dep, so this is the state most deployments that set the env var
        by mistake would be in.
        """
        ws = _workspace(tmp_path, monkeypatch, flag=True)
        _opt_in(monkeypatch)
        monkeypatch.setattr(tenant_kms, "_has_cryptography", lambda: False)

        mgr = EncryptionManager(ws, _PASSPHRASE)
        assert mgr._kms is None
        assert mgr.encrypt(_PLAINTEXT)[:6] == _MAGIC
        assert not os.path.exists(os.path.join(ws, ".mind-mem-keys", _KMS_DEK_FILE))

    def test_a_damaged_dek_blob_is_never_replaced(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Minting over an unopenable DEK destroys the corpus and hides it."""
        ws = _workspace(tmp_path, monkeypatch, flag=True)
        _opt_in(monkeypatch)
        EncryptionManager(ws, _PASSPHRASE)  # mints the blob

        blob_path = Path(ws, ".mind-mem-keys", _KMS_DEK_FILE)
        damaged = "!!!not-a-wrapped-dek!!!"
        blob_path.write_text(damaged, encoding="utf-8")

        with pytest.raises(ValueError, match="Refusing to mint a replacement"):
            EncryptionManager(ws, _PASSPHRASE)
        assert blob_path.read_text(encoding="utf-8") == damaged


# ===========================================================================
# 6. No silent downgrade
# ===========================================================================


class TestNoSilentDowngrade:
    def test_passphrase_rotation_refuses_in_envelope_mode(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        ws = _workspace(tmp_path, monkeypatch, flag=True)
        _opt_in(monkeypatch)
        target = Path(ws, "corpus.md")
        target.write_bytes(_PLAINTEXT)
        mgr = EncryptionManager(ws, _PASSPHRASE)
        mgr.encrypt_file(str(target))

        with pytest.raises(RuntimeError, match="rotate_tenant_dek"):
            mgr.rotate_key("another-passphrase", [str(target)])

        # And the file is exactly as it was — a refused rotation touches nothing.
        assert target.read_bytes()[:6] == _MAGIC_KMS
        assert mgr.decrypt_file(str(target)) == _PLAINTEXT
