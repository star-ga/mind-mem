"""Ed25519 manifest signing for ``mm audit-model`` checkpoints.

Companion to :mod:`mind_mem.model_audit`. ``audit_model`` produces a
SHA-256 manifest of every file in a checkpoint; this module signs that
manifest with an Ed25519 keypair so a third party can verify the
checkpoint hasn't been tampered with since the audit.

Wire format (text, ``MODEL_MANIFEST.txt``, ``sha256sum -c`` compatible)::

    <sha256-hex>  <relative-path>
    ...

Sidecar files written next to the manifest::

    MODEL_MANIFEST.txt        — the manifest itself
    MODEL_MANIFEST.txt.sig    — 64-byte Ed25519 signature over the
                                manifest (raw bytes, not base64)
    MODEL_PUBKEY.pub          — 32-byte Ed25519 public key (raw bytes,
                                not PEM) — present alongside .sig so
                                downstream consumers can verify without
                                key distribution overhead

Operators that need a centrally-managed publisher key can omit the
``.pub`` sidecar and compare the signature against a key fetched from
their internal key registry; the contract here is "everything needed
to verify is reproducible from the checkpoint directory".

Zero-coupling to ``model_audit``: this module computes its own
manifest from the checkpoint root so it can be used standalone or
chained after ``audit_model`` (in which case the ``audit_report``
parameter short-circuits the re-hash).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
)

MANIFEST_FILENAME = "MODEL_MANIFEST.txt"
SIGNATURE_FILENAME = "MODEL_MANIFEST.txt.sig"
PUBKEY_FILENAME = "MODEL_PUBKEY.pub"

# Ed25519 sizes — reference: RFC 8032 §5.1.5 / §5.1.6
ED25519_PRIVATE_KEY_BYTES = 32
ED25519_PUBLIC_KEY_BYTES = 32
ED25519_SIGNATURE_BYTES = 64


@dataclass
class SignResult:
    """Outcome of a ``sign_model`` call.

    ``manifest_path`` / ``signature_path`` / ``pubkey_path`` may all be
    ``None`` if the caller asked the writer to stay in-memory (used by
    the verify path that just wants the recomputed digest).
    """

    manifest_text: str
    manifest_sha256: str
    signature: bytes
    public_key: bytes
    manifest_path: Path | None = None
    signature_path: Path | None = None
    pubkey_path: Path | None = None


@dataclass
class VerifyResult:
    """Outcome of a ``verify_model`` call.

    A clean verify sets ``passed=True`` and ``error_kind=None``. Any
    integrity failure sets ``passed=False`` and a fine-grained
    ``error_kind`` so callers (CLI / MCP tool) can render an actionable
    message instead of "verification failed".
    """

    passed: bool
    manifest_sha256: str
    error_kind: str | None = None  # "manifest_mismatch" | "bad_signature" | "missing_file"
    error_detail: str | None = None


# --- Manifest computation ----------------------------------------------------


def compute_manifest_text(root: Path) -> tuple[str, dict[str, str]]:
    """Walk ``root`` and produce a deterministic ``sha256sum -c``-compatible
    manifest. Sort by relative path so the byte stream is stable across
    filesystems with different ``readdir`` ordering.
    """
    if not root.is_dir():
        raise NotADirectoryError(f"checkpoint root not a directory: {root}")
    by_path: dict[str, str] = {}
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.name not in _SKIP_NAMES)
    for p in files:
        h = hashlib.sha256()
        with p.open("rb") as f:
            while chunk := f.read(1 << 20):
                h.update(chunk)
        by_path[p.relative_to(root).as_posix()] = h.hexdigest()
    lines = [f"{digest}  {path}\n" for path, digest in sorted(by_path.items())]
    return "".join(lines), by_path


# Files that ``sign_model`` must NOT include in the manifest itself —
# otherwise signing becomes a fixed-point problem (the manifest hashes
# its own outputs). These are the sidecar files this module writes.
_SKIP_NAMES = frozenset({MANIFEST_FILENAME, SIGNATURE_FILENAME, PUBKEY_FILENAME})


# --- Key management ----------------------------------------------------------


def generate_keypair() -> tuple[bytes, bytes]:
    """Generate a fresh Ed25519 keypair.

    Returns ``(private_key_raw_32_bytes, public_key_raw_32_bytes)``.
    Raw byte form (not PEM / DER) keeps the wire format STARGA-native
    and avoids ASN.1 parsing on the verify side.
    """
    sk = Ed25519PrivateKey.generate()
    sk_raw = sk.private_bytes(encoding=Encoding.Raw, format=PrivateFormat.Raw, encryption_algorithm=NoEncryption())
    pk_raw = sk.public_key().public_bytes(encoding=Encoding.Raw, format=PublicFormat.Raw)
    return sk_raw, pk_raw


def public_key_from_private(private_key: bytes) -> bytes:
    """Derive the 32-byte public key from a 32-byte private key."""
    if len(private_key) != ED25519_PRIVATE_KEY_BYTES:
        raise ValueError(f"Ed25519 private key must be {ED25519_PRIVATE_KEY_BYTES} bytes, got {len(private_key)}")
    sk = Ed25519PrivateKey.from_private_bytes(private_key)
    return sk.public_key().public_bytes(encoding=Encoding.Raw, format=PublicFormat.Raw)


# --- Sign / verify -----------------------------------------------------------


def sign_manifest(manifest_text: str, private_key: bytes) -> bytes:
    """Sign the manifest text with an Ed25519 private key.

    Returns the 64-byte signature. Manifest is signed as UTF-8 bytes —
    operators must keep the manifest in canonical form (LF-only line
    endings, sorted by path) so a re-hash under a different OS still
    verifies.
    """
    if len(private_key) != ED25519_PRIVATE_KEY_BYTES:
        raise ValueError(f"Ed25519 private key must be {ED25519_PRIVATE_KEY_BYTES} bytes, got {len(private_key)}")
    sk = Ed25519PrivateKey.from_private_bytes(private_key)
    return sk.sign(manifest_text.encode("utf-8"))


def verify_manifest(manifest_text: str, signature: bytes, public_key: bytes) -> bool:
    """Verify an Ed25519 signature over the manifest text."""
    if len(public_key) != ED25519_PUBLIC_KEY_BYTES:
        raise ValueError(f"Ed25519 public key must be {ED25519_PUBLIC_KEY_BYTES} bytes, got {len(public_key)}")
    if len(signature) != ED25519_SIGNATURE_BYTES:
        return False
    pk = Ed25519PublicKey.from_public_bytes(public_key)
    try:
        pk.verify(signature, manifest_text.encode("utf-8"))
        return True
    except InvalidSignature:
        return False


# --- High-level: sign / verify a whole checkpoint ----------------------------


def sign_model(
    checkpoint_path: Path | str,
    private_key: bytes,
    *,
    write_sidecars: bool = True,
) -> SignResult:
    """Sign every file under ``checkpoint_path``.

    Writes ``MODEL_MANIFEST.txt`` + ``MODEL_MANIFEST.txt.sig`` +
    ``MODEL_PUBKEY.pub`` next to the checkpoint root unless
    ``write_sidecars=False`` (in which case only the in-memory
    :class:`SignResult` is returned).
    """
    root = Path(checkpoint_path).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"checkpoint root not a directory: {root}")

    manifest_text, _ = compute_manifest_text(root)
    manifest_sha = hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()
    signature = sign_manifest(manifest_text, private_key)
    public_key = public_key_from_private(private_key)

    manifest_path: Path | None = None
    signature_path: Path | None = None
    pubkey_path: Path | None = None

    if write_sidecars:
        manifest_path = root / MANIFEST_FILENAME
        signature_path = root / SIGNATURE_FILENAME
        pubkey_path = root / PUBKEY_FILENAME
        manifest_path.write_text(manifest_text)
        signature_path.write_bytes(signature)
        pubkey_path.write_bytes(public_key)

    return SignResult(
        manifest_text=manifest_text,
        manifest_sha256=manifest_sha,
        signature=signature,
        public_key=public_key,
        manifest_path=manifest_path,
        signature_path=signature_path,
        pubkey_path=pubkey_path,
    )


def verify_model(
    checkpoint_path: Path | str,
    *,
    public_key: bytes | None = None,
) -> VerifyResult:
    """Verify a previously-signed checkpoint.

    If ``public_key`` is omitted the verifier reads
    ``MODEL_PUBKEY.pub`` from the checkpoint directory. Operators
    pinning to a centrally-managed key should pass it explicitly so a
    swapped sidecar can't impersonate.
    """
    root = Path(checkpoint_path).expanduser().resolve()
    manifest_path = root / MANIFEST_FILENAME
    signature_path = root / SIGNATURE_FILENAME

    if not manifest_path.is_file():
        return VerifyResult(
            passed=False,
            manifest_sha256="",
            error_kind="missing_file",
            error_detail=f"{MANIFEST_FILENAME} not found in {root}",
        )
    if not signature_path.is_file():
        return VerifyResult(
            passed=False,
            manifest_sha256="",
            error_kind="missing_file",
            error_detail=f"{SIGNATURE_FILENAME} not found in {root}",
        )

    if public_key is None:
        pubkey_path = root / PUBKEY_FILENAME
        if not pubkey_path.is_file():
            return VerifyResult(
                passed=False,
                manifest_sha256="",
                error_kind="missing_file",
                error_detail=(f"{PUBKEY_FILENAME} not found and no public_key passed; either ship the .pub sidecar or pass --pubkey"),
            )
        public_key = pubkey_path.read_bytes()

    stored_manifest = manifest_path.read_text()
    stored_signature = signature_path.read_bytes()

    # Recompute the manifest from disk and compare to the stored one.
    # Any drift (added file, removed file, mutated bytes) shows up here.
    recomputed_manifest, _ = compute_manifest_text(root)
    if recomputed_manifest != stored_manifest:
        return VerifyResult(
            passed=False,
            manifest_sha256=hashlib.sha256(recomputed_manifest.encode("utf-8")).hexdigest(),
            error_kind="manifest_mismatch",
            error_detail="recomputed manifest differs from stored MODEL_MANIFEST.txt",
        )

    if not verify_manifest(stored_manifest, stored_signature, public_key):
        return VerifyResult(
            passed=False,
            manifest_sha256=hashlib.sha256(stored_manifest.encode("utf-8")).hexdigest(),
            error_kind="bad_signature",
            error_detail="Ed25519 signature verification failed against MODEL_PUBKEY.pub",
        )

    return VerifyResult(
        passed=True,
        manifest_sha256=hashlib.sha256(stored_manifest.encode("utf-8")).hexdigest(),
    )
