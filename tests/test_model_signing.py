"""Tests for ``mind_mem.model_signing`` — Ed25519 manifest signing."""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path

import pytest

from mind_mem.model_signing import (
    ED25519_PRIVATE_KEY_BYTES,
    ED25519_PUBLIC_KEY_BYTES,
    ED25519_SIGNATURE_BYTES,
    MANIFEST_FILENAME,
    PUBKEY_FILENAME,
    SIGNATURE_FILENAME,
    compute_manifest_text,
    generate_keypair,
    public_key_from_private,
    sign_manifest,
    sign_model,
    verify_manifest,
    verify_model,
)


@pytest.fixture
def tiny_ckpt(tmp_path: Path) -> Path:
    """Two-file synthetic checkpoint, deterministic content."""
    root = tmp_path / "ckpt"
    root.mkdir()
    (root / "config.json").write_text('{"model_type":"qwen3"}')
    body = b'{"weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}'
    (root / "model.safetensors").write_bytes(struct.pack("<Q", len(body)) + body + b"\x00" * 8)
    return root


# ---------------------------------------------------------------------------
# Manifest computation
# ---------------------------------------------------------------------------


class TestComputeManifestText:
    def test_format_is_sha256sum_compatible(self, tiny_ckpt: Path) -> None:
        text, by_path = compute_manifest_text(tiny_ckpt)
        for line in text.splitlines():
            digest, _, path = line.partition("  ")
            assert len(digest) == 64
            assert all(c in "0123456789abcdef" for c in digest)
            assert by_path[path] == digest

    def test_paths_sorted_for_determinism(self, tiny_ckpt: Path) -> None:
        # Re-run twice; identical input → identical bytes, regardless of
        # underlying readdir ordering.
        a, _ = compute_manifest_text(tiny_ckpt)
        b, _ = compute_manifest_text(tiny_ckpt)
        assert a == b

    def test_skips_sidecar_files(self, tiny_ckpt: Path) -> None:
        # Pre-seed the directory with sidecars; they MUST NOT enter the
        # manifest (otherwise signing becomes a fixed-point problem).
        (tiny_ckpt / MANIFEST_FILENAME).write_text("placeholder\n")
        (tiny_ckpt / SIGNATURE_FILENAME).write_bytes(b"\x00" * 64)
        (tiny_ckpt / PUBKEY_FILENAME).write_bytes(b"\x00" * 32)
        _, by_path = compute_manifest_text(tiny_ckpt)
        assert MANIFEST_FILENAME not in by_path
        assert SIGNATURE_FILENAME not in by_path
        assert PUBKEY_FILENAME not in by_path

    def test_missing_root_raises(self, tmp_path: Path) -> None:
        with pytest.raises(NotADirectoryError):
            compute_manifest_text(tmp_path / "nope")


# ---------------------------------------------------------------------------
# Key management
# ---------------------------------------------------------------------------


class TestKeypairs:
    def test_generate_keypair_sizes(self) -> None:
        sk, pk = generate_keypair()
        assert len(sk) == ED25519_PRIVATE_KEY_BYTES
        assert len(pk) == ED25519_PUBLIC_KEY_BYTES

    def test_public_key_from_private_round_trip(self) -> None:
        sk, pk = generate_keypair()
        derived = public_key_from_private(sk)
        assert derived == pk

    def test_public_key_from_private_rejects_bad_length(self) -> None:
        with pytest.raises(ValueError, match="32 bytes"):
            public_key_from_private(b"\x00" * 31)


# ---------------------------------------------------------------------------
# Sign / verify a manifest in isolation
# ---------------------------------------------------------------------------


class TestSignVerifyManifest:
    def test_round_trip_passes(self) -> None:
        sk, pk = generate_keypair()
        text = "abc123  config.json\n"
        sig = sign_manifest(text, sk)
        assert len(sig) == ED25519_SIGNATURE_BYTES
        assert verify_manifest(text, sig, pk)

    def test_tampered_text_fails(self) -> None:
        sk, pk = generate_keypair()
        text = "abc123  config.json\n"
        sig = sign_manifest(text, sk)
        assert not verify_manifest("abc123  TAMPERED.json\n", sig, pk)

    def test_wrong_pubkey_fails(self) -> None:
        sk, _ = generate_keypair()
        _, other_pk = generate_keypair()
        text = "abc123  x\n"
        sig = sign_manifest(text, sk)
        assert not verify_manifest(text, sig, other_pk)

    def test_signature_bad_length_fails_gracefully(self) -> None:
        _, pk = generate_keypair()
        # 32 bytes, not 64 → returns False, doesn't raise
        assert not verify_manifest("x", b"\x00" * 32, pk)

    def test_sign_rejects_bad_private_length(self) -> None:
        with pytest.raises(ValueError, match="32 bytes"):
            sign_manifest("x", b"\x00" * 16)

    def test_verify_rejects_bad_public_length(self) -> None:
        with pytest.raises(ValueError, match="32 bytes"):
            verify_manifest("x", b"\x00" * 64, b"\x00" * 16)


# ---------------------------------------------------------------------------
# Sign / verify a checkpoint end-to-end
# ---------------------------------------------------------------------------


class TestSignModel:
    def test_writes_three_sidecars(self, tiny_ckpt: Path) -> None:
        sk, _ = generate_keypair()
        result = sign_model(tiny_ckpt, sk)
        assert (tiny_ckpt / MANIFEST_FILENAME).is_file()
        assert (tiny_ckpt / SIGNATURE_FILENAME).is_file()
        assert (tiny_ckpt / PUBKEY_FILENAME).is_file()
        assert result.manifest_path == tiny_ckpt / MANIFEST_FILENAME
        assert len(result.signature) == ED25519_SIGNATURE_BYTES
        assert len(result.public_key) == ED25519_PUBLIC_KEY_BYTES

    def test_no_sidecars_when_write_disabled(self, tiny_ckpt: Path) -> None:
        sk, _ = generate_keypair()
        result = sign_model(tiny_ckpt, sk, write_sidecars=False)
        assert not (tiny_ckpt / MANIFEST_FILENAME).exists()
        assert not (tiny_ckpt / SIGNATURE_FILENAME).exists()
        assert not (tiny_ckpt / PUBKEY_FILENAME).exists()
        # In-memory result still populated
        assert result.manifest_text
        assert len(result.signature) == ED25519_SIGNATURE_BYTES

    def test_manifest_sha256_matches(self, tiny_ckpt: Path) -> None:
        sk, _ = generate_keypair()
        result = sign_model(tiny_ckpt, sk, write_sidecars=False)
        assert result.manifest_sha256 == hashlib.sha256(result.manifest_text.encode()).hexdigest()


class TestVerifyModel:
    def test_clean_signed_checkpoint_verifies(self, tiny_ckpt: Path) -> None:
        sk, _ = generate_keypair()
        sign_model(tiny_ckpt, sk)
        result = verify_model(tiny_ckpt)
        assert result.passed
        assert result.error_kind is None

    def test_explicit_pubkey_overrides_sidecar(self, tiny_ckpt: Path) -> None:
        sk, pk = generate_keypair()
        sign_model(tiny_ckpt, sk)
        # Replace the .pub sidecar with garbage but pass the real key
        # explicitly — verification should still succeed because the
        # caller's pinned key is the source of truth.
        (tiny_ckpt / PUBKEY_FILENAME).write_bytes(b"\xff" * 32)
        result = verify_model(tiny_ckpt, public_key=pk)
        assert result.passed

    def test_modified_file_triggers_manifest_mismatch(self, tiny_ckpt: Path) -> None:
        sk, _ = generate_keypair()
        sign_model(tiny_ckpt, sk)
        # Tamper with one of the signed files
        (tiny_ckpt / "config.json").write_text('{"model_type":"BACKDOORED"}')
        result = verify_model(tiny_ckpt)
        assert not result.passed
        assert result.error_kind == "manifest_mismatch"

    def test_corrupted_signature_triggers_bad_signature(self, tiny_ckpt: Path) -> None:
        sk, _ = generate_keypair()
        sign_model(tiny_ckpt, sk)
        # Flip a byte in the signature — same manifest, broken sig
        sig_path = tiny_ckpt / SIGNATURE_FILENAME
        bytes_ = bytearray(sig_path.read_bytes())
        bytes_[0] ^= 0xFF
        sig_path.write_bytes(bytes(bytes_))
        result = verify_model(tiny_ckpt)
        assert not result.passed
        assert result.error_kind == "bad_signature"

    def test_missing_manifest_reports_cleanly(self, tiny_ckpt: Path) -> None:
        result = verify_model(tiny_ckpt)
        assert not result.passed
        assert result.error_kind == "missing_file"
        assert MANIFEST_FILENAME in (result.error_detail or "")

    def test_missing_signature_reports_cleanly(self, tiny_ckpt: Path) -> None:
        sk, _ = generate_keypair()
        sign_model(tiny_ckpt, sk)
        (tiny_ckpt / SIGNATURE_FILENAME).unlink()
        result = verify_model(tiny_ckpt)
        assert not result.passed
        assert result.error_kind == "missing_file"
        assert SIGNATURE_FILENAME in (result.error_detail or "")

    def test_missing_pubkey_and_no_explicit_key_reports_cleanly(self, tiny_ckpt: Path) -> None:
        sk, _ = generate_keypair()
        sign_model(tiny_ckpt, sk)
        (tiny_ckpt / PUBKEY_FILENAME).unlink()
        result = verify_model(tiny_ckpt)
        assert not result.passed
        assert result.error_kind == "missing_file"
        assert PUBKEY_FILENAME in (result.error_detail or "")
