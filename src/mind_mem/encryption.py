#!/usr/bin/env python3
"""mind-mem Encryption at Rest — optional authenticated encryption for blocks.

Provides transparent encryption/decryption for workspace files using
PBKDF2-derived keys from a user passphrase. Pure stdlib implementation
(hashlib + hmac, no external crypto libraries required).

Construction (NOT AES / NOT SQLCipher — do not represent it as such):
a 256-bit-keyed **HMAC-SHA256 counter-mode keystream cipher** XORed with the
plaintext, under an **encrypt-then-MAC** scheme with separate encryption and
authentication keys. The construction is sound (random per-message nonce,
constant-time MAC check), but it is a hand-rolled stream cipher, not a
NIST/AEAD primitive. Operators with FIPS/AES compliance requirements should
not rely on this module as "AES-256". NOTE: the FTS5/sqlite-vec recall index
is NOT encrypted — only the on-disk block files are; an attacker with
filesystem read access can recover indexed content from recall.db.

OPT-IN KMS ENVELOPE MODE (``v4.tenant_kms``, default OFF): when the operator
sets ``MIND_MEM_KMS_MASTER_KEY_B64``, enables the flag, and has the
``cryptography`` package installed, new ciphertexts are real AES-256-GCM
records under a per-tenant data key wrapped by that master key
(:mod:`mind_mem.tenant_kms`). That path IS an AEAD and may be described as
AES-256-GCM. The default path above is not, and must not be. Reads route on
the record header, so a workspace holding both formats opens both. There is
no automatic migration: existing files stay in the format they were written
in until they are re-encrypted.

Key management:
- Key derived from passphrase via PBKDF2-HMAC-SHA256 (600k iterations)
- Salt stored in workspace/.mind-mem-keys/salt (32 bytes), written atomically
- A salt file of the wrong length is reported, never silently regenerated —
  regenerating it would make every existing ciphertext unreadable forever
- Key never written to disk
- Key rotation via re-encryption with new passphrase

Usage:
    from .encryption import EncryptionManager
    mgr = EncryptionManager(workspace, passphrase="my-secret")
    ciphertext = mgr.encrypt(b"sensitive data")
    plaintext = mgr.decrypt(ciphertext)
    mgr.encrypt_file("path/to/file.md")

Zero external deps — hashlib, hmac, os, struct, tempfile (all stdlib).
"""

from __future__ import annotations

import hashlib
import hmac
import os
import struct
import tempfile
from typing import TYPE_CHECKING

from .mind_filelock import FileLock
from .observability import get_logger, metrics

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .tenant_kms import MasterKey

_log = get_logger("encryption")

# KDF parameters
_KDF_ITERATIONS = 600_000  # OWASP recommendation for PBKDF2-SHA256
_SALT_SIZE = 32
_KEY_SIZE = 32  # 256-bit key
_NONCE_SIZE = 16
_MAC_SIZE = 32  # HMAC-SHA256

# Hard cap on a single encrypt/decrypt payload. The keystream XOR is a
# pure-Python per-byte loop, so an unbounded input blocks the event loop /
# exhausts memory (DoS). 256 MiB is far above any real memory block while
# still bounding worst-case work. Override only with eyes open.
_MAX_PAYLOAD_BYTES = 256 * 1024 * 1024

# Envelope overhead on a ciphertext (MAGIC + NONCE + MAC), used to bound the
# on-disk ciphertext size against the plaintext cap.
_ENVELOPE_SIZE = 6 + _NONCE_SIZE + _MAC_SIZE  # _MAGIC is 6 bytes


def _reject_oversized_file(path: str, max_bytes: int) -> None:
    """Reject an oversized file BEFORE it is read into memory.

    The encrypt/decrypt caps protect the pure-Python XOR loop, but the file
    entry points ``read()`` the whole file first — a multi-GB file would
    buffer fully into RAM before the in-memory cap fires. A stat-size
    pre-check closes that gap. Missing/unstattable files fall through so the
    subsequent ``open()`` surfaces the real error.
    """
    try:
        size = os.path.getsize(path)
    except OSError:
        return
    if size > max_bytes:
        raise ValueError(f"file {path!r} ({size} bytes) exceeds {max_bytes} byte cap")


def _fsync_dir(directory: str) -> None:
    """Best-effort fsync of a directory so a rename inside it is durable.

    Not supported everywhere (Windows refuses to open a directory), hence
    best-effort: durability is an improvement on the rename, never a
    precondition for it.
    """
    try:
        dir_fd = os.open(directory, os.O_RDONLY)
    except OSError:  # nosec B110 — directories are not openable on all platforms
        return
    try:
        os.fsync(dir_fd)
    except OSError:  # nosec B110 — best-effort durability
        pass
    finally:
        os.close(dir_fd)


def _write_key_material(path: str, data: bytes) -> None:
    """Write key material to ``path`` atomically and owner-only.

    The salt is the one piece of key material that cannot be recomputed
    from anything else, so it must never be observable in a half-written
    state: a torn write leaves a wrong-length salt that a later run cannot
    distinguish from deliberate corruption. Stage into a temp file in the
    same directory (``mkstemp`` creates it 0600), fsync it, then
    ``os.replace`` — atomic on POSIX and Windows — and fsync the directory
    so the rename itself survives a crash. Any failure removes the temp
    file and leaves the existing salt, if any, untouched.
    """
    directory = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(prefix=".salt-", dir=directory)
    try:
        try:
            written = os.write(fd, data)
            # A short write is the exact failure this helper exists to
            # prevent — surface it instead of committing a truncated file.
            if written != len(data):
                raise OSError(f"short write to {tmp_path!r}: {written}/{len(data)} bytes")
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:  # nosec B110 — temp file already gone
            pass
        raise
    _fsync_dir(directory)


# File header magic bytes
_MAGIC = b"MMENC1"  # mind-mem encrypted v1

# ---------------------------------------------------------------------------
# Opt-in KMS envelope mode (v4.tenant_kms)
# ---------------------------------------------------------------------------
#
# The default construction above is a keystream cipher, and the docstring says
# so. :mod:`mind_mem.tenant_kms` holds the product's only real AEAD (AES-256-GCM
# via ``cryptography``), and it was minting keys nobody used. This section is
# the substitution: with the flag on and a master key supplied, the at-rest
# path stops hand-rolling a cipher and uses AES-GCM under a KMS-wrapped data
# key instead. The KEK/DEK split is exactly the shape tenant_kms already
# implements — it is wired SINGLE-TENANT here (``tenant_id="default"``), so the
# multi-tenant surface arrives already exercised rather than untested.
#
# OFF BY DEFAULT, and off means off: :func:`_kms_envelope_cipher` returns
# ``None`` after a single ``os.environ`` lookup, so an install that never sets
# the env var performs no config read, no import of ``cryptography``, and no
# log line. Nothing about the default ciphertext changes.
_MAGIC_KMS = b"MMKMS1"  # mind-mem KMS envelope record v1

#: Every magic a mind-mem ciphertext can open with. Both are 6 bytes, so the
#: legacy offset arithmetic keyed on ``len(_MAGIC)`` is unaffected; the values
#: differ so a reader can ROUTE on the header. That matters because a workspace
#: which flips envelope mode on mid-life holds both kinds of record at once and
#: every one of them must keep opening.
_MAGICS: tuple[bytes, ...] = (_MAGIC, _MAGIC_KMS)

#: AES-GCM parameters for the envelope payload.
_KMS_NONCE_SIZE = 12  # NIST-recommended GCM nonce
_KMS_TAG_SIZE = 16
_KMS_RECORD_OVERHEAD = len(_MAGIC_KMS) + _KMS_NONCE_SIZE + _KMS_TAG_SIZE

#: Env var holding the operator's base64 KEK. Absent means envelope mode is
#: off, which is the default and the only state a stock install is ever in.
_KMS_MASTER_ENV = "MIND_MEM_KMS_MASTER_KEY_B64"

#: Single-tenant id for the workspace-scoped wiring. v4.0 multi-tenant flips
#: the encryption scope to one DEK per tenant; until a tenant identity exists
#: on the server surface there is exactly one, and naming it here keeps the
#: wire format and the on-disk blob identical to what multi-tenant will read.
_KMS_TENANT_ID = "default"

#: Wrapped-DEK blob, inside the existing 0700 ``.mind-mem-keys`` directory.
_KMS_DEK_FILE = "tenant-default.dek"


def has_magic(data: bytes) -> bool:
    """True when *data* opens with any mind-mem ciphertext magic.

    Every "is this file encrypted?" test must go through here rather than
    comparing against :data:`_MAGIC` alone: a bare ``_MAGIC`` comparison reads
    a KMS envelope record as PLAINTEXT, which would make ``encrypt_file``
    double-encrypt it and ``decrypt_file`` hand ciphertext back to the caller.
    """
    return any(data[: len(m)] == m for m in _MAGICS)


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

    Pure-Python counter-mode keystream with HMAC-SHA256 as the PRF (NOT
    AES). Each block is HMAC(key, nonce || counter); blocks are concatenated
    and truncated to ``length``.
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


def _discard_previous_salt(keys_dir: str) -> None:
    """Remove the retained rotation salt. Safe when it is absent."""
    try:
        os.remove(os.path.join(keys_dir, "salt.previous"))
    except OSError:
        pass


class _KmsEnvelopeCipher:
    """AES-256-GCM payload cipher under a KMS-wrapped per-tenant data key.

    Record layout: ``_MAGIC_KMS(6) | nonce(12) | ciphertext || tag``.

    The magic is passed as the AEAD's associated data, so a record whose
    header was rewritten to look like a legacy ``MMENC1`` file (or vice
    versa) fails authentication instead of decrypting to something.

    This is a real AEAD — one primitive doing confidentiality and integrity,
    not a keystream plus a bolted-on MAC. That distinction is the entire
    reason for the wiring, so it is stated here and not overstated anywhere
    else: the LEGACY path in this module remains an HMAC-SHA256 keystream
    under encrypt-then-MAC, and it is what a workspace gets unless an
    operator opts in.
    """

    __slots__ = ("_aead",)

    def __init__(self, dek: bytes) -> None:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # noqa: PLC0415

        self._aead = AESGCM(dek)

    def seal(self, plaintext: bytes) -> bytes:
        nonce = os.urandom(_KMS_NONCE_SIZE)
        return _MAGIC_KMS + nonce + self._aead.encrypt(nonce, plaintext, _MAGIC_KMS)

    def open(self, data: bytes) -> bytes:
        body = data[len(_MAGIC_KMS) :]
        if len(body) < _KMS_NONCE_SIZE + _KMS_TAG_SIZE:
            raise ValueError("KMS envelope record too short")
        nonce, ct = body[:_KMS_NONCE_SIZE], body[_KMS_NONCE_SIZE:]
        try:
            return bytes(self._aead.decrypt(nonce, ct, _MAGIC_KMS))
        except Exception as exc:
            # Surface as ValueError like every other failure in this module,
            # so callers keep one exception contract. InvalidTag carries no
            # message worth forwarding.
            raise ValueError("AES-GCM authentication failed — wrong data key or tampered ciphertext") from exc


def _load_or_mint_tenant_dek(master: "MasterKey", keys_dir: str) -> bytes:
    """Return the workspace's plaintext DEK, minting one on first use.

    A wrapped-DEK file that exists but does not unwrap is NEVER replaced, for
    the same reason ``_get_or_create_salt`` refuses to replace a damaged salt:
    minting a fresh DEK over it destroys every envelope ciphertext in the
    workspace *and* disguises the loss, since each later decrypt then reports
    authentication failure and points the operator at an attacker instead of
    at their own file. Fail closed and name the path so it can be restored.
    """
    from .tenant_kms import WrappedDEK, generate_tenant_dek, unwrap_tenant_dek  # noqa: PLC0415

    path = os.path.join(keys_dir, _KMS_DEK_FILE)
    with FileLock(path):
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as fh:
                blob = fh.read().strip()
            try:
                return unwrap_tenant_dek(master, WrappedDEK.from_b64(blob))
            except Exception as exc:
                _log.error("kms_tenant_dek_unwrap_failed", path=path)
                raise ValueError(
                    f"wrapped data key {path!r} could not be unwrapped with the master key in "
                    f"{_KMS_MASTER_ENV} ({exc}). Refusing to mint a replacement, which would make "
                    "every already-encrypted file permanently unreadable. Restore the blob from "
                    "backup, or supply the master key it was wrapped under."
                ) from exc

        dek, wrapped = generate_tenant_dek(master, _KMS_TENANT_ID)
        _write_key_material(path, wrapped.to_b64().encode("ascii"))
        _log.info("kms_tenant_dek_created", tenant_id=_KMS_TENANT_ID)
        return dek


def _kms_envelope_cipher(keys_dir: str) -> "_KmsEnvelopeCipher | None":
    """Build the opt-in KMS envelope cipher, or ``None`` for the legacy path.

    Three conditions, checked cheapest-first so a default install pays exactly
    one ``os.environ`` lookup and nothing else — no config stat, no JSON parse,
    no import, no log line:

    1. :data:`_KMS_MASTER_ENV` is set — the operator supplies the KEK;
    2. the ``v4.tenant_kms`` flag is on (quiet probe: an OFF answer must leave
       no trace);
    3. ``cryptography`` is importable.

    Condition 3 failing is a clean degrade, not a crash: the workspace keeps
    using the legacy construction and gets a warning saying why. It is safe to
    degrade on the WRITE side precisely because reads route on the record
    magic — an envelope record written earlier still refuses to open under a
    build that lost the dependency, rather than being mis-read.
    """
    if not os.environ.get(_KMS_MASTER_ENV, "").strip():
        return None

    from .v4.feature_flags import is_enabled_quiet  # noqa: PLC0415

    if not is_enabled_quiet("tenant_kms"):
        return None

    from .tenant_kms import create_master_key_from_env, require_production_crypto  # noqa: PLC0415

    try:
        require_production_crypto()
    except RuntimeError as exc:
        _log.warning("kms_envelope_unavailable", reason=str(exc))
        return None
    try:
        master = create_master_key_from_env(_KMS_MASTER_ENV)
    except (RuntimeError, ValueError) as exc:
        _log.warning("kms_envelope_master_key_rejected", reason=str(exc))
        return None

    return _KmsEnvelopeCipher(_load_or_mint_tenant_dek(master, keys_dir))


class EncryptionManager:
    """Optional encryption layer for mind-mem workspaces.

    Provides encrypt/decrypt operations with PBKDF2-derived keys.
    Thread-safe via FileLock on key material.

    When KMS envelope mode is opted into (see :func:`_kms_envelope_cipher`)
    new ciphertexts are AES-256-GCM records under a ``tenant_kms``-wrapped
    data key instead. Reads route on the record magic, so both formats keep
    opening from the same manager.
    """

    def __init__(self, workspace: str, passphrase: str) -> None:
        """Initialize with workspace and passphrase.

        Args:
            workspace: Workspace root path.
            passphrase: User passphrase for key derivation.

        Raises:
            ValueError: If passphrase is too short (< 8 chars), or if the
                workspace salt file exists but has the wrong length.
        """
        if len(passphrase) < 8:
            raise ValueError("Passphrase must be at least 8 characters")

        self.workspace = os.path.realpath(workspace)
        self._keys_dir = os.path.join(self.workspace, ".mind-mem-keys")
        # Owner-only (0700): the key-material directory must not be world- or
        # group-readable on shared hosts. exist_ok tolerates a pre-existing
        # dir; tighten its mode below in case it was created world-readable.
        os.makedirs(self._keys_dir, mode=0o700, exist_ok=True)
        try:
            os.chmod(self._keys_dir, 0o700)
        except OSError:  # nosec B110 — best-effort on filesystems w/o chmod
            pass

        self._salt = self._get_or_create_salt()
        self._passphrase = passphrase
        self._key = _pbkdf2(passphrase, self._salt)

        # Derive separate keys for encryption and MAC
        self._enc_key = hmac.new(self._key, b"encrypt", hashlib.sha256).digest()
        self._mac_key = hmac.new(self._key, b"authenticate", hashlib.sha256).digest()

        # Opt-in KMS envelope provider. ``None`` — the default, and the only
        # value a stock install ever sees — leaves every byte of the path
        # below untouched.
        self._kms: _KmsEnvelopeCipher | None = _kms_envelope_cipher(self._keys_dir)

    def _get_or_create_salt(self) -> bytes:
        """Load the workspace salt, minting one only when none exists.

        A salt file that exists but is not exactly ``_SALT_SIZE`` bytes is
        NEVER overwritten. The salt is half the key material and cannot be
        recovered from anything else, so replacing a damaged one with fresh
        entropy destroys every ciphertext in the workspace *and* disguises
        the loss: each later ``decrypt`` fails MAC verification and reports
        tampering, pointing the operator at an attacker instead of at their
        own truncated file. Fail closed and name the file, so the salt that
        is still on disk can be restored from backup.

        Raises:
            ValueError: If the salt file exists with the wrong length.
        """
        salt_path = os.path.join(self._keys_dir, "salt")
        with FileLock(salt_path):
            if os.path.isfile(salt_path):
                with open(salt_path, "rb") as f:
                    salt = f.read()
                if len(salt) == _SALT_SIZE:
                    return salt
                _log.error(
                    "encryption_salt_corrupt",
                    path=salt_path,
                    expected_bytes=_SALT_SIZE,
                    actual_bytes=len(salt),
                )
                raise ValueError(
                    f"salt file {salt_path!r} is {len(salt)} bytes, expected {_SALT_SIZE}: "
                    "refusing to overwrite key material, which would make every "
                    "already-encrypted file permanently unreadable. Restore the file "
                    "from backup, or delete it if nothing was encrypted under it."
                )

            salt = os.urandom(_SALT_SIZE)
            _write_key_material(salt_path, salt)
            _log.info("encryption_salt_created")
            return salt

    def encrypt(self, plaintext: bytes) -> bytes:
        """Encrypt data.

        Format: MAGIC(6) + NONCE(16) + CIPHERTEXT(N) + MAC(32) — an
        HMAC-SHA256 keystream under encrypt-then-MAC, not an AEAD.

        In KMS envelope mode (opt-in) the record is instead
        ``_MAGIC_KMS(6) + NONCE(12) + AES-256-GCM(ciphertext||tag)``.

        Args:
            plaintext: Data to encrypt.

        Returns:
            Encrypted bytes with nonce and MAC.
        """
        if len(plaintext) > _MAX_PAYLOAD_BYTES:
            raise ValueError(f"plaintext exceeds {_MAX_PAYLOAD_BYTES} byte encrypt cap")
        if self._kms is not None:
            return self._kms.seal(plaintext)
        nonce = os.urandom(_NONCE_SIZE)
        ks = _keystream(self._enc_key, nonce, len(plaintext))
        ciphertext = _xor_bytes(plaintext, ks)

        # Compute MAC over nonce + ciphertext (encrypt-then-MAC)
        mac_input = nonce + ciphertext
        mac = hmac.new(self._mac_key, mac_input, hashlib.sha256).digest()

        return _MAGIC + nonce + ciphertext + mac

    def _previous_keys(self) -> "tuple[bytes, bytes] | None":
        """(mac_key, enc_key) from the retained rotation salt, or None.

        Present only while a rotation is in flight or crashed partway. Read
        fresh each time rather than cached: a rotation completing in another
        process removes the file, and a stale cache would keep offering a key
        that should no longer decrypt anything.
        """
        path = os.path.join(self._keys_dir, "salt.previous")
        try:
            with open(path, "rb") as fh:
                salt = fh.read()
        except OSError:
            return None
        if len(salt) != _SALT_SIZE:
            return None
        key = _pbkdf2(self._passphrase, salt)
        return (
            hmac.new(key, b"authenticate", hashlib.sha256).digest(),
            hmac.new(key, b"encrypt", hashlib.sha256).digest(),
        )

    def decrypt(self, data: bytes) -> bytes:
        """Decrypt data.

        Args:
            data: Encrypted bytes (MAGIC + NONCE + CIPHERTEXT + MAC).

        Returns:
            Decrypted plaintext.

        Raises:
            ValueError: If data is malformed or MAC verification fails.
        """
        # Route on the header FIRST. A KMS envelope record is shorter than the
        # legacy minimum (34 bytes vs 54 for an empty payload), so the legacy
        # length guard below would reject a perfectly valid one as "too short".
        if data[: len(_MAGIC_KMS)] == _MAGIC_KMS:
            if len(data) > _MAX_PAYLOAD_BYTES + _KMS_RECORD_OVERHEAD:
                raise ValueError(f"ciphertext exceeds {_MAX_PAYLOAD_BYTES} byte decrypt cap")
            if self._kms is None:
                raise ValueError(
                    "this file is a KMS envelope record (MMKMS1) but envelope mode is not "
                    f"available here: set {_KMS_MASTER_ENV}, enable v4.tenant_kms in "
                    "mind-mem.json, and install 'cryptography'. Refusing to guess a key."
                )
            return self._kms.open(data)

        min_len = len(_MAGIC) + _NONCE_SIZE + _MAC_SIZE
        if len(data) < min_len:
            raise ValueError("Encrypted data too short")
        if len(data) > _MAX_PAYLOAD_BYTES + min_len:
            raise ValueError(f"ciphertext exceeds {_MAX_PAYLOAD_BYTES} byte decrypt cap")

        if data[: len(_MAGIC)] != _MAGIC:
            raise ValueError("Invalid encryption header (not mind-mem encrypted)")

        nonce = data[len(_MAGIC) : len(_MAGIC) + _NONCE_SIZE]
        mac = data[-_MAC_SIZE:]
        ciphertext = data[len(_MAGIC) + _NONCE_SIZE : -_MAC_SIZE]

        # Verify MAC (constant-time comparison)
        expected_mac = hmac.new(self._mac_key, nonce + ciphertext, hashlib.sha256).digest()
        if not hmac.compare_digest(mac, expected_mac):
            # Before declaring tampering, try the retained previous key. It
            # exists only while a rotation is in flight (or crashed partway),
            # and it is exactly the case where a file legitimately carries the
            # OLD key while `salt` already names the new one. Reporting
            # "data may be tampered" for a half-finished rotation sends the
            # operator hunting an attacker instead of re-running the rotation.
            previous = self._previous_keys()
            if previous is not None:
                prev_mac_key, prev_enc_key = previous
                prev_expected = hmac.new(prev_mac_key, nonce + ciphertext, hashlib.sha256).digest()
                if hmac.compare_digest(mac, prev_expected):
                    return _xor_bytes(ciphertext, _keystream(prev_enc_key, nonce, len(ciphertext)))
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
            _reject_oversized_file(resolved, _MAX_PAYLOAD_BYTES)
            with open(resolved, "rb") as f:
                plaintext = f.read()

            # Nothing to do for empty files. Encrypting an empty
            # plaintext would write a header-only ciphertext whose
            # magic bytes mask the file as "already encrypted" on
            # the next call — permanently losing the ability to
            # reach the original zero-byte state.
            if not plaintext:
                return

            # Skip if already encrypted, in EITHER record format.
            if has_magic(plaintext):
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
        _reject_oversized_file(file_path, _MAX_PAYLOAD_BYTES + _ENVELOPE_SIZE)
        with open(file_path, "rb") as f:
            data = f.read()

        if not has_magic(data):
            return data  # Not encrypted, return as-is

        return self.decrypt(data)

    def decrypt_file_in_place(self, file_path: str) -> None:
        """Decrypt a file and write plaintext back.

        Args:
            file_path: Path to the encrypted file.
        """
        resolved = os.path.realpath(file_path)
        with FileLock(resolved):
            _reject_oversized_file(resolved, _MAX_PAYLOAD_BYTES + _ENVELOPE_SIZE)
            with open(resolved, "rb") as f:
                data = f.read()

            if not has_magic(data):
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
            return has_magic(header)
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

        # In envelope mode the passphrase is not what protects the payload —
        # the KMS-wrapped data key is. Phase 2 below re-encrypts with the
        # LEGACY keystream construction, so running this here would silently
        # downgrade every AES-GCM record to the hand-rolled cipher while
        # reporting a successful "key rotation". Refuse and name the real
        # rotation path instead of quietly weakening the workspace.
        if self._kms is not None:
            raise RuntimeError(
                "passphrase rotation is not the rotation for KMS envelope mode: it would "
                "rewrite AES-GCM records with the legacy keystream construction. Rotate the "
                "data key with mind_mem.tenant_kms.rotate_tenant_dek and re-encrypt under the "
                "new DEK, or turn v4.tenant_kms off before rotating the passphrase."
            )

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
                raise RuntimeError(f"key rotation aborted: decrypt failed for {path!r}") from e

        # Phase 2 — write all re-encrypted payloads to temp files
        # alongside the originals. No original is overwritten yet, so a
        # crash after this phase still leaves every file intact.
        staged: list[tuple[str, str]] = []  # (final_path, tmp_path)
        try:
            for path, plaintext in plaintexts:
                if len(plaintext) > _MAX_PAYLOAD_BYTES:
                    raise ValueError(f"key rotation payload for {path!r} exceeds {_MAX_PAYLOAD_BYTES} byte cap")
                nonce = os.urandom(_NONCE_SIZE)
                ks = _keystream(new_enc_key, nonce, len(plaintext))
                ciphertext = _xor_bytes(plaintext, ks)
                mac = hmac.new(new_mac_key, nonce + ciphertext, hashlib.sha256).digest()
                encrypted = _MAGIC + nonce + ciphertext + mac

                resolved = os.path.realpath(path)
                tmp_path = resolved + ".rotate.tmp"
                with open(tmp_path, "wb") as fh:
                    fh.write(encrypted)
                    fh.flush()
                    os.fsync(fh.fileno())
                staged.append((resolved, tmp_path))

            # Phase 2.5 — retain the OUTGOING salt before the first swap.
            #
            # Phase 3 replaces files one at a time. The moment the first
            # os.replace lands, the corpus holds a MIX: some files under the
            # new key, the rest under the old, while ``salt`` still names the
            # old one. A crash there used to leave data that NEITHER key could
            # read in full — irreversible loss, not a stalled operation. The
            # comment below Phase 4 claimed "no files have the new ciphertext
            # yet", which stops being true after a single swap.
            #
            # Keeping the outgoing salt makes every intermediate state
            # recoverable: decrypt falls back to it, so a half-rotated corpus
            # reads correctly under either key and the rotation can simply be
            # re-run.
            _write_key_material(os.path.join(self._keys_dir, "salt.previous"), self._salt)

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
            _write_key_material(salt_path, new_salt)
            # The rotation is complete, so the previous salt is no longer
            # needed to read anything. Dropping it here — and ONLY here —
            # keeps the recovery window exactly as long as the risk.
            _discard_previous_salt(self._keys_dir)
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
