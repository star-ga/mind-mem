"""Tests for mind-mem encryption at rest."""

import os
import stat

import pytest

from mind_mem.encryption import _MAGIC, _SALT_SIZE, EncryptionManager, _pbkdf2


@pytest.fixture
def workspace(tmp_path):
    return str(tmp_path)


@pytest.fixture
def mgr(workspace):
    return EncryptionManager(workspace, "test-passphrase-12345")


class TestPBKDF2:
    def test_deterministic(self):
        salt = b"fixed-salt-for-test-1234567890ab"
        k1 = _pbkdf2("password", salt, iterations=1000)
        k2 = _pbkdf2("password", salt, iterations=1000)
        assert k1 == k2

    def test_different_passwords(self):
        salt = b"fixed-salt-for-test-1234567890ab"
        k1 = _pbkdf2("password1", salt, iterations=1000)
        k2 = _pbkdf2("password2", salt, iterations=1000)
        assert k1 != k2

    def test_key_length(self):
        salt = b"fixed-salt-for-test-1234567890ab"
        k = _pbkdf2("password", salt, iterations=1000)
        assert len(k) == 32  # AES-256


class TestEncryptDecrypt:
    def test_round_trip(self, mgr):
        plaintext = b"Hello, this is sensitive data!"
        ciphertext = mgr.encrypt(plaintext)
        decrypted = mgr.decrypt(ciphertext)
        assert decrypted == plaintext

    def test_ciphertext_different_from_plaintext(self, mgr):
        plaintext = b"Secret message that should be hidden"
        ciphertext = mgr.encrypt(plaintext)
        # Ciphertext should not contain plaintext
        assert plaintext not in ciphertext

    def test_magic_header(self, mgr):
        ciphertext = mgr.encrypt(b"test")
        assert ciphertext[:6] == _MAGIC

    def test_different_nonces(self, mgr):
        plaintext = b"same plaintext"
        c1 = mgr.encrypt(plaintext)
        c2 = mgr.encrypt(plaintext)
        # Different nonces → different ciphertexts
        assert c1 != c2

    def test_tamper_detection(self, mgr):
        ciphertext = mgr.encrypt(b"test data")
        # Tamper with ciphertext
        tampered = bytearray(ciphertext)
        tampered[20] ^= 0xFF  # Flip a byte in the ciphertext area
        with pytest.raises(ValueError, match="MAC verification failed"):
            mgr.decrypt(bytes(tampered))

    def test_invalid_header(self, mgr):
        # Data long enough to pass length check but with wrong header
        fake = b"BADHED" + os.urandom(60)
        with pytest.raises(ValueError, match="Invalid encryption header"):
            mgr.decrypt(fake)

    def test_too_short(self, mgr):
        with pytest.raises(ValueError, match="too short"):
            mgr.decrypt(b"short")

    def test_empty_plaintext(self, mgr):
        ciphertext = mgr.encrypt(b"")
        decrypted = mgr.decrypt(ciphertext)
        assert decrypted == b""

    def test_large_data(self, mgr):
        plaintext = os.urandom(1024 * 100)  # 100KB
        ciphertext = mgr.encrypt(plaintext)
        decrypted = mgr.decrypt(ciphertext)
        assert decrypted == plaintext


class TestFileEncryption:
    def test_encrypt_decrypt_file(self, workspace, mgr):
        path = os.path.join(workspace, "test.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write("# Secret Decision\nUse AES-256 for encryption.")

        mgr.encrypt_file(path)
        assert mgr.is_encrypted(path)

        # Read encrypted — should not contain plaintext
        with open(path, "rb") as f:
            data = f.read()
        assert b"Secret Decision" not in data

        # Decrypt
        plaintext = mgr.decrypt_file(path)
        assert b"Secret Decision" in plaintext

    def test_decrypt_in_place(self, workspace, mgr):
        path = os.path.join(workspace, "test.md")
        original = "# Original content"
        with open(path, "w", encoding="utf-8") as f:
            f.write(original)

        mgr.encrypt_file(path)
        mgr.decrypt_file_in_place(path)

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        assert content == original

    def test_double_encrypt_noop(self, workspace, mgr):
        path = os.path.join(workspace, "test.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write("content")

        mgr.encrypt_file(path)
        size1 = os.path.getsize(path)
        mgr.encrypt_file(path)  # Should be noop
        size2 = os.path.getsize(path)
        assert size1 == size2

    def test_is_encrypted(self, workspace, mgr):
        path = os.path.join(workspace, "test.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write("plain text")

        assert not mgr.is_encrypted(path)
        mgr.encrypt_file(path)
        assert mgr.is_encrypted(path)

    def test_decrypt_unencrypted_returns_asis(self, workspace, mgr):
        path = os.path.join(workspace, "plain.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write("not encrypted")

        content = mgr.decrypt_file(path)
        assert content == b"not encrypted"


class TestKeyManagement:
    def test_short_passphrase_rejected(self, workspace):
        with pytest.raises(ValueError, match="at least 8"):
            EncryptionManager(workspace, "short")

    def test_salt_persisted(self, workspace):
        mgr1 = EncryptionManager(workspace, "test-passphrase-12345")
        salt_path = os.path.join(workspace, ".mind-mem-keys", "salt")
        assert os.path.isfile(salt_path)

        # Same salt on second init
        mgr2 = EncryptionManager(workspace, "test-passphrase-12345")
        assert mgr1._salt == mgr2._salt

    def test_same_passphrase_same_key(self, workspace):
        mgr1 = EncryptionManager(workspace, "test-passphrase-12345")
        mgr2 = EncryptionManager(workspace, "test-passphrase-12345")

        plaintext = b"test data"
        ciphertext = mgr1.encrypt(plaintext)
        decrypted = mgr2.decrypt(ciphertext)
        assert decrypted == plaintext

    def test_different_passphrase_fails(self, workspace):
        mgr1 = EncryptionManager(workspace, "passphrase-one-12345")
        ciphertext = mgr1.encrypt(b"secret")

        mgr2 = EncryptionManager(workspace, "passphrase-two-12345")
        with pytest.raises(ValueError, match="MAC verification failed"):
            mgr2.decrypt(ciphertext)

    def test_key_rotation(self, workspace):
        mgr = EncryptionManager(workspace, "old-passphrase-12345")

        # Encrypt a file
        path = os.path.join(workspace, "data.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write("sensitive content")
        mgr.encrypt_file(path)

        # Rotate key
        count = mgr.rotate_key("new-passphrase-12345", [path])
        assert count == 1

        # Decrypt with new key (mgr is updated in-place)
        plaintext = mgr.decrypt_file(path)
        assert plaintext == b"sensitive content"

    def test_key_rotation_short_passphrase(self, workspace, mgr):
        with pytest.raises(ValueError, match="at least 8"):
            mgr.rotate_key("short", [])


class TestSaltIntegrity:
    """A damaged salt must never be silently replaced.

    The salt is half the key material and cannot be recomputed. Minting a
    fresh one over a truncated file destroys every ciphertext in the
    workspace and then misreports the loss as tampering.
    """

    @staticmethod
    def _salt_path(workspace):
        return os.path.join(workspace, ".mind-mem-keys", "salt")

    def test_truncated_salt_is_not_overwritten(self, workspace):
        mgr1 = EncryptionManager(workspace, "test-passphrase-12345")
        ciphertext = mgr1.encrypt(b"irreplaceable corpus content")

        salt_path = self._salt_path(workspace)
        with open(salt_path, "rb") as f:
            original = f.read()
        assert len(original) == _SALT_SIZE

        # Simulate a truncated / partially written salt file.
        with open(salt_path, "r+b") as f:
            f.truncate(16)

        with pytest.raises(ValueError, match="refusing to overwrite key material"):
            EncryptionManager(workspace, "test-passphrase-12345")

        # The surviving bytes are still on disk — nothing was regenerated.
        with open(salt_path, "rb") as f:
            assert f.read() == original[:16]

        # And once the operator restores the salt, the old ciphertext is
        # readable again: the failure was recoverable, not terminal.
        with open(salt_path, "wb") as f:
            f.write(original)
        mgr2 = EncryptionManager(workspace, "test-passphrase-12345")
        assert mgr2.decrypt(ciphertext) == b"irreplaceable corpus content"

    def test_empty_salt_file_is_reported_not_replaced(self, workspace):
        EncryptionManager(workspace, "test-passphrase-12345")
        salt_path = self._salt_path(workspace)
        with open(salt_path, "wb"):
            pass

        with pytest.raises(ValueError, match="0 bytes, expected 32"):
            EncryptionManager(workspace, "test-passphrase-12345")
        assert os.path.getsize(salt_path) == 0

    def test_oversized_salt_file_is_reported_not_replaced(self, workspace):
        EncryptionManager(workspace, "test-passphrase-12345")
        salt_path = self._salt_path(workspace)
        junk = b"\xab" * (_SALT_SIZE + 8)
        with open(salt_path, "wb") as f:
            f.write(junk)

        with pytest.raises(ValueError, match="refusing to overwrite key material"):
            EncryptionManager(workspace, "test-passphrase-12345")
        with open(salt_path, "rb") as f:
            assert f.read() == junk

    def test_salt_creation_is_atomic_no_partial_file(self, workspace, monkeypatch):
        """An interrupted salt write leaves no salt at all, never a short one."""
        keys_dir = os.path.join(workspace, ".mind-mem-keys")

        def boom(src, dst):
            raise OSError("simulated disk failure during salt commit")

        monkeypatch.setattr("mind_mem.encryption.os.replace", boom)
        with pytest.raises(OSError, match="simulated disk failure"):
            EncryptionManager(workspace, "test-passphrase-12345")

        assert not os.path.exists(os.path.join(keys_dir, "salt"))
        # No staged temp file left behind to be mistaken for key material.
        assert os.listdir(keys_dir) == []

    def test_created_salt_is_owner_only_with_no_leftovers(self, workspace):
        EncryptionManager(workspace, "test-passphrase-12345")
        keys_dir = os.path.join(workspace, ".mind-mem-keys")
        salt_path = self._salt_path(workspace)

        assert os.path.getsize(salt_path) == _SALT_SIZE
        assert os.listdir(keys_dir) == ["salt"]
        if os.name == "posix":
            assert stat.S_IMODE(os.stat(salt_path).st_mode) == 0o600

    def test_rotated_salt_is_owner_only(self, workspace):
        mgr = EncryptionManager(workspace, "old-passphrase-12345")
        path = os.path.join(workspace, "data.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write("sensitive content")
        mgr.encrypt_file(path)

        mgr.rotate_key("new-passphrase-12345", [path])

        salt_path = self._salt_path(workspace)
        assert os.path.getsize(salt_path) == _SALT_SIZE
        if os.name == "posix":
            assert stat.S_IMODE(os.stat(salt_path).st_mode) == 0o600
        # The rotated salt is the one a fresh manager loads.
        mgr2 = EncryptionManager(workspace, "new-passphrase-12345")
        assert mgr2.decrypt_file(path) == b"sensitive content"
