# Copyright 2026 STARGA, Inc.
"""EncryptedBlockStore — transparent at-rest encryption for Markdown
corpora (v3.0.0 — GH #504).

Wraps any :class:`BlockStore` with an EncryptionManager so every read
from disk goes through decrypt_file and every write-back (via
parse_file-compatible consumers) reads ciphertext transparently.

Activation is opt-in via the ``MIND_MEM_ENCRYPTION_PASSPHRASE`` env
var — when unset the factory returns the plain BlockStore so no new
user breaks.

**Scope notes:**

- This ships the Markdown-file wrapper, *not* SQLCipher for the FTS5
  / sqlite-vec indices. A fully encrypted index requires the
  ``pysqlcipher3`` optional dep; left for a later patch so this
  change can land without new deps. Operators who want the
  index-level protection should keep the workspace on an encrypted
  filesystem (LUKS, FileVault, BitLocker).

- Encryption is a no-op for files that don't start with
  :data:`encryption._MAGIC` — this keeps existing unencrypted
  workspaces readable during rollout.

- ``encrypt_workspace(workspace)`` is provided as a one-shot
  migration helper that walks every Markdown file under the corpus
  directories and encrypts each in-place.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from .block_parser import get_active, get_by_id, parse_file
from .block_store import BlockStore, MarkdownBlockStore
from .corpus_registry import CORPUS_DIRS
from .observability import get_logger

_log = get_logger("block_store_encrypted")


def _passphrase() -> str | None:
    raw = os.environ.get("MIND_MEM_ENCRYPTION_PASSPHRASE", "").strip()
    return raw or None


# ---------------------------------------------------------------------------
# EncryptedBlockStore — transparent-decrypt read path
# ---------------------------------------------------------------------------


class EncryptedBlockStore:
    """BlockStore wrapper that transparently decrypts files at read.

    Exposes the same public surface as :class:`MarkdownBlockStore`.
    Every file path returned by the inner store is intercepted: if
    the file's leading bytes match the encryption magic header, the
    ciphertext is decrypted into a temp plaintext that the parser
    consumes, and the temp is deleted immediately after parse.

    Plain (unencrypted) files pass through unchanged — the wrapper is
    safe to deploy against a partially-migrated workspace.
    """

    def __init__(self, workspace: str, *, passphrase: str, inner: BlockStore | None = None) -> None:
        if not passphrase:
            raise ValueError("EncryptedBlockStore requires a non-empty passphrase")
        self._workspace = workspace
        self._passphrase = passphrase
        self._inner = inner or MarkdownBlockStore(workspace)

        # Lazy-import EncryptionManager so unit-test workloads that
        # never decrypt anything skip the hashlib startup cost.
        from .encryption import EncryptionManager

        self._em = EncryptionManager(workspace, passphrase)

    # ------------------------------------------------------------------
    # BlockStore protocol
    # ------------------------------------------------------------------

    def get_all(self, *, active_only: bool = False) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []
        for fpath in self.list_files():
            parsed = self._parse_maybe_encrypted(fpath)
            if active_only:
                parsed = get_active(parsed)
            blocks.extend(parsed)
        return blocks

    def get_by_id(self, block_id: str) -> Optional[dict[str, Any]]:
        for fpath in self.list_files():
            parsed = self._parse_maybe_encrypted(fpath)
            result = get_by_id(parsed, block_id)
            if result:
                return result
        return None

    def search(self, query: str, *, limit: int = 10) -> list[dict[str, Any]]:
        q = query.lower()
        matches: list[dict[str, Any]] = []
        for block in self.get_all():
            text = " ".join(str(v) for v in block.values()).lower()
            if q in text:
                matches.append(block)
                if len(matches) >= limit:
                    break
        return matches

    def list_files(self) -> list[str]:
        return self._inner.list_files()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _parse_maybe_encrypted(self, fpath: str) -> list[dict[str, Any]]:
        """Parse *fpath*, transparently decrypting on the fly.

        Uses a tempfile only when we detect ciphertext so the happy
        path (plaintext file) matches the inner store's performance.
        """
        from .encryption import _MAGIC

        try:
            with open(fpath, "rb") as fh:
                head = fh.read(len(_MAGIC))
        except OSError:
            return []
        if head != _MAGIC:
            return parse_file(fpath)

        try:
            plaintext = self._em.decrypt_file(fpath)
        except Exception as exc:  # pragma: no cover — best-effort
            _log.warning("encrypted_read_failed", path=fpath, error=str(exc))
            return []

        # Write to a tempfile so parse_file can read it; fall back to a
        # string-based parse when block_parser exposes it. The temp is
        # unlinked immediately after parse.
        import tempfile

        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".md", prefix=".mm-decrypt-")
        try:
            with os.fdopen(tmp_fd, "wb") as fh:
                fh.write(plaintext)
            return parse_file(tmp_path)
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Factory + migration
# ---------------------------------------------------------------------------


def get_block_store(workspace: str) -> BlockStore:
    """Return an encrypted or plain BlockStore depending on env config.

    When ``MIND_MEM_ENCRYPTION_PASSPHRASE`` is set in the environment
    a transparent EncryptedBlockStore is returned; otherwise a plain
    MarkdownBlockStore. New code should route through this factory
    instead of instantiating MarkdownBlockStore directly.
    """
    passphrase = _passphrase()
    if passphrase:
        try:
            return EncryptedBlockStore(workspace, passphrase=passphrase)
        except Exception as exc:
            _log.warning(
                "encrypted_block_store_init_failed",
                error=str(exc),
                fallback="MarkdownBlockStore",
            )
    return MarkdownBlockStore(workspace)


def encrypt_workspace(workspace: str) -> dict[str, int]:
    """One-shot migration — encrypt every Markdown file in the corpus.

    Idempotent: files already prefixed with the magic header are
    skipped. Empty files are skipped (see EncryptionManager for the
    rationale). Returns a summary dict for the caller to print.

    Requires ``MIND_MEM_ENCRYPTION_PASSPHRASE`` in the environment
    (raises :class:`RuntimeError` otherwise).
    """
    passphrase = _passphrase()
    if not passphrase:
        raise RuntimeError("encrypt_workspace requires MIND_MEM_ENCRYPTION_PASSPHRASE")

    from .encryption import _MAGIC, EncryptionManager

    em = EncryptionManager(workspace, passphrase)
    encrypted = skipped = failed = 0
    for d in CORPUS_DIRS:
        dir_path = os.path.join(workspace, d)
        if not os.path.isdir(dir_path):
            continue
        for root, _dirs, files in os.walk(dir_path):
            for fname in files:
                if not fname.endswith(".md"):
                    continue
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, "rb") as fh:
                        head = fh.read(len(_MAGIC))
                    if head == _MAGIC:
                        skipped += 1
                        continue
                    em.encrypt_file(fpath)
                    encrypted += 1
                except Exception as exc:  # pragma: no cover
                    _log.warning("encrypt_workspace_failed", path=fpath, error=str(exc))
                    failed += 1
    return {"encrypted": encrypted, "skipped": skipped, "failed": failed}


__all__ = [
    "EncryptedBlockStore",
    "get_block_store",
    "encrypt_workspace",
]
