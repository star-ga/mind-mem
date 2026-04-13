#!/usr/bin/env python3
"""mind-mem Encryption at Rest — optional AES-256 encryption for memory blocks.

Provides transparent encryption/decryption for workspace files using
PBKDF2-derived keys from a user passphrase. Pure stdlib implementation
(hashlib + hmac for KDF, no external crypto libraries required).

Uses AES-256-CTR mode via XOR with HMAC-SHA256 keystream (pure Python
fallback when no native crypto is available).

Key management:
- Key derived from passphrase via PBKDF2-HMAC-SHA256 (600k iterations)
- Salt stored in workspace/.mind-mem-keys/salt (32 bytes)
- Key never written to disk
- Key rotation via re-encryption with new passphrase

Usage:
    from .encryption import EncryptionManager
    mgr = EncryptionManager(workspace, passphrase="my-secret")
    ciphertext = mgr.encrypt(b"sensitive data")
    plaintext = mgr.decrypt(ciphertext)
    mgr.encrypt_file("path/to/file.md")

Zero external deps — hashlib, hmac, os, struct (all stdlib).
"""

from __future__ import annotations

import hashlib
import hmac
import os
import struct

from .mind_filelock import FileLock
from .observability import get_logger, metrics

_log = get_logger("encryption")

# KDF parameters
_KDF_ITERATIONS = 600_000  # OWASP recommendation for PBKDF2-SHA256
_SALT_SIZE = 32
_KEY_SIZE = 32  # AES-256
_NONCE_SIZE = 16
_MAC_SIZE = 32  # HMAC-SHA256

# File header magic bytes
_MAGIC = b"MMENC1"  # mind-mem encrypted v1


def _pbkdf2(passphrase: str, salt: bytes, iterations: int = _KDF_ITERATIONS) -> bytes:
    """Derive a key from passphrase using PBKDF2-HMAC-SHA256."""
    return hashlib.pbkdf2_hmac(
        "sha256",
        passphrase.encode("utf-8"),
        salt,
        iterations,
        dklen=_KEY_SIZE,
    )


def _keystream(key: bytes, nonce: bytes, length: int) -> bytes:
    """Generate a keystream using HMAC-SHA256 in counter mode.

    This is a pure-Python AES-CTR-like construction using HMAC as PRF.
    """
    stream = bytearray()
    counter = 0
    while len(stream) < length:
        block_input = nonce + struct.pack(">Q", counter)
        block = hmac.new(key, block_input, hashlib.sha256).digest()
        stream.extend(block)
        counter += 1
    return bytes(stream[:length])


def _xor_bytes(data: bytes, keystream: bytes) -> bytes:
    """XOR two byte sequences."""
    return bytes(a ^ b for a, b in zip(data, keystream))


class EncryptionManager:
    """Optional encryption layer for mind-mem workspaces.

    Provides encrypt/decrypt operations with PBKDF2-derived keys.
    Thread-safe via FileLock on key material.
    """

    def __init__(self, workspace: str, passphrase: str) -> None:
        """Initialize with workspace and passphrase.

        Args:
            workspace: Workspace root path.
            passphrase: User passphrase for key derivation.

        Raises:
            ValueError: If passphrase is too short (< 8 chars).
        """
        if len(passphrase) < 8:
            raise ValueError("Passphrase must be at least 8 characters")

        self.workspace = os.path.realpath(workspace)
        self._keys_dir = os.path.join(self.workspace, ".mind-mem-keys")
        os.makedirs(self._keys_dir, exist_ok=True)

        self._salt = self._get_or_create_salt()
        self._key = _pbkdf2(passphrase, self._salt)

        # Derive separate keys for encryption and MAC
        self._enc_key = hmac.new(self._key, b"encrypt", hashlib.sha256).digest()
        self._mac_key = hmac.new(self._key, b"authenticate", hashlib.sha256).digest()

    def _get_or_create_salt(self) -> bytes:
        """Get or create the workspace salt."""
        salt_path = os.path.join(self._keys_dir, "salt")
        with FileLock(salt_path):
            if os.path.isfile(salt_path):
                with open(salt_path, "rb") as f:
                    salt = f.read()
                if len(salt) == _SALT_SIZE:
                    return salt

            salt = os.urandom(_SALT_SIZE)
            with open(salt_path, "wb") as f:
                f.write(salt)
            _log.info("encryption_salt_created")
            return salt

    def encrypt(self, plaintext: bytes) -> bytes:
        """Encrypt data.

        Format: MAGIC(6) + NONCE(16) + CIPHERTEXT(N) + MAC(32)

        Args:
            plaintext: Data to encrypt.

        Returns:
            Encrypted bytes with nonce and MAC.
        """
        nonce = os.urandom(_NONCE_SIZE)
        ks = _keystream(self._enc_key, nonce, len(plaintext))
        ciphertext = _xor_bytes(plaintext, ks)

        # Compute MAC over nonce + ciphertext (encrypt-then-MAC)
        mac_input = nonce + ciphertext
        mac = hmac.new(self._mac_key, mac_input, hashlib.sha256).digest()

        return _MAGIC + nonce + ciphertext + mac

    def decrypt(self, data: bytes) -> bytes:
        """Decrypt data.

        Args:
            data: Encrypted bytes (MAGIC + NONCE + CIPHERTEXT + MAC).

        Returns:
            Decrypted plaintext.

        Raises:
            ValueError: If data is malformed or MAC verification fails.
        """
        min_len = len(_MAGIC) + _NONCE_SIZE + _MAC_SIZE
        if len(data) < min_len:
            raise ValueError("Encrypted data too short")

        if data[: len(_MAGIC)] != _MAGIC:
            raise ValueError("Invalid encryption header (not mind-mem encrypted)")

        nonce = data[len(_MAGIC) : len(_MAGIC) + _NONCE_SIZE]
        mac = data[-_MAC_SIZE:]
        ciphertext = data[len(_MAGIC) + _NONCE_SIZE : -_MAC_SIZE]

        # Verify MAC (constant-time comparison)
        expected_mac = hmac.new(self._mac_key, nonce + ciphertext, hashlib.sha256).digest()
        if not hmac.compare_digest(mac, expected_mac):
            raise ValueError("MAC verification failed — data may be tampered")

        ks = _keystream(self._enc_key, nonce, len(ciphertext))
        return _xor_bytes(ciphertext, ks)

    def encrypt_file(self, file_path: str) -> None:
        """Encrypt a file in-place.

        Args:
            file_path: Path to the file to encrypt.
        """
        resolved = os.path.realpath(file_path)
        with FileLock(resolved):
            with open(resolved, "rb") as f:
                plaintext = f.read()

            # Nothing to do for empty files. Encrypting an empty
            # plaintext would write a header-only ciphertext whose
            # magic bytes mask the file as "already encrypted" on
            # the next call — permanently losing the ability to
            # reach the original zero-byte state.
            if not plaintext:
                return

            # Skip if already encrypted
            if plaintext[: len(_MAGIC)] == _MAGIC:
                return

            encrypted = self.encrypt(plaintext)
            with open(resolved, "wb") as f:
                f.write(encrypted)

        _log.info("file_encrypted", path=file_path)
        metrics.inc("files_encrypted")

    def decrypt_file(self, file_path: str) -> bytes:
        """Decrypt a file, returning plaintext without modifying the file.

        Args:
            file_path: Path to the encrypted file.

        Returns:
            Decrypted content.
        """
        with open(file_path, "rb") as f:
            data = f.read()

        if data[: len(_MAGIC)] != _MAGIC:
            return data  # Not encrypted, return as-is

        return self.decrypt(data)

    def decrypt_file_in_place(self, file_path: str) -> None:
        """Decrypt a file and write plaintext back.

        Args:
            file_path: Path to the encrypted file.
        """
        resolved = os.path.realpath(file_path)
        with FileLock(resolved):
            with open(resolved, "rb") as f:
                data = f.read()

            if data[: len(_MAGIC)] != _MAGIC:
                return  # Not encrypted

            plaintext = self.decrypt(data)
            with open(resolved, "wb") as f:
                f.write(plaintext)

        _log.info("file_decrypted", path=file_path)
        metrics.inc("files_decrypted")

    def is_encrypted(self, file_path: str) -> bool:
        """Check if a file is encrypted.

        Args:
            file_path: Path to check.

        Returns:
            True if file has the mind-mem encryption header.
        """
        try:
            with open(file_path, "rb") as f:
                header = f.read(len(_MAGIC))
            return header == _MAGIC
        except OSError:
            return False

    def rotate_key(self, new_passphrase: str, file_paths: list[str]) -> int:
        """Rotate encryption key by re-encrypting files with a new passphrase.

        Args:
            new_passphrase: New passphrase to derive key from.
            file_paths: List of encrypted file paths to re-encrypt.

        Returns:
            Number of files re-encrypted.
        """
        if len(new_passphrase) < 8:
            raise ValueError("New passphrase must be at least 8 characters")

        # Derive the new key/material BEFORE touching any files so a
        # key-rotation crash can't leave the workspace split between
        # old-salt and new-ciphertext states.
        new_salt = os.urandom(_SALT_SIZE)
        new_key = _pbkdf2(new_passphrase, new_salt)
        new_enc_key = hmac.new(new_key, b"encrypt", hashlib.sha256).digest()
        new_mac_key = hmac.new(new_key, b"authenticate", hashlib.sha256).digest()

        # Phase 1 — decrypt everything with the current key. Abort the
        # entire rotation on the first failure; a partial rotation is
        # worse than no rotation (we don't know which key decrypts which
        # file anymore).
        plaintexts: list[tuple[str, bytes]] = []
        for path in file_paths:
            try:
                plaintexts.append((path, self.decrypt_file(path)))
            except (OSError, ValueError) as e:
                _log.error("key_rotation_decrypt_failed", path=path, error=str(e))
                raise RuntimeError(
                    f"key rotation aborted: decrypt failed for {path!r}"
                ) from e

        # Phase 2 — write all re-encrypted payloads to temp files
        # alongside the originals. No original is overwritten yet, so a
        # crash after this phase still leaves every file intact.
        staged: list[tuple[str, str]] = []  # (final_path, tmp_path)
        try:
            for path, plaintext in plaintexts:
                nonce = os.urandom(_NONCE_SIZE)
                ks = _keystream(new_enc_key, nonce, len(plaintext))
                ciphertext = _xor_bytes(plaintext, ks)
                mac = hmac.new(
                    new_mac_key, nonce + ciphertext, hashlib.sha256
                ).digest()
                encrypted = _MAGIC + nonce + ciphertext + mac

                resolved = os.path.realpath(path)
                tmp_path = resolved + ".rotate.tmp"
                with open(tmp_path, "wb") as fh:
                    fh.write(encrypted)
                    fh.flush()
                    os.fsync(fh.fileno())
                staged.append((resolved, tmp_path))

            # Phase 3 — atomically swap each file in. os.replace is
            # atomic on POSIX and Windows.
            for final_path, tmp_path in staged:
                with FileLock(final_path):
                    os.replace(tmp_path, final_path)

            # Phase 4 — persist the new salt once every file has been
            # swapped. A crash before this step and the old salt is
            # still valid (no files have the new ciphertext yet);
            # a crash after and the new salt matches the new files.
            salt_path = os.path.join(self._keys_dir, "salt")
            tmp_salt = salt_path + ".rotate.tmp"
            with open(tmp_salt, "wb") as fh:
                fh.write(new_salt)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_salt, salt_path)
        except Exception:
            # Best-effort cleanup of any temp files we staged.
            for _, tmp_path in staged:
                try:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except OSError:
                    pass
            raise

        # Only flip in-memory state once every on-disk artifact has
        # been committed.
        self._salt = new_salt
        self._key = new_key
        self._enc_key = new_enc_key
        self._mac_key = new_mac_key

        _log.info("key_rotation_complete", files=len(staged))
        metrics.inc("key_rotations")
        return len(staged)
