# Copyright 2026 STARGA, Inc.
"""EncryptedBlockStore — transparent at-rest encryption for Markdown
corpora (v3.0.0 — GH #504).

Wraps any :class:`BlockStore` with an EncryptionManager: reads decrypt
a ciphertext file on the way through the parser, and writes unseal the
target file for the inner store's read-modify-write and re-seal it
afterwards, so a governed apply against an encrypted workspace behaves
exactly as it does against a plain one.

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

- Encryption is a no-op for files that don't start with a mind-mem
  ciphertext magic (:func:`encryption.has_magic`, which covers both the
  legacy ``MMENC1`` record and the opt-in ``MMKMS1`` KMS envelope
  record) — this keeps existing unencrypted
  workspaces readable during rollout, and a write to a file the
  operator has left in plaintext leaves it in plaintext. A corpus
  file that does not exist yet follows its neighbours — created
  sealed in a migrated workspace, plain in one that holds no
  ciphertext yet (see :meth:`EncryptedBlockStore._decrypted_target`).

- ``encrypt_workspace(workspace)`` is provided as a one-shot
  migration helper that walks every Markdown file under the corpus
  directories and encrypts each in-place.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator, Optional, cast

from .admission import require_admission, require_delete_admission
from .block_parser import get_active, get_by_id, parse_file
from .block_store import BlockStore, BlockStoreError, MarkdownBlockStore, _resolve_block_file
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

    Implements the whole :class:`~mind_mem.block_store.BlockStore`
    surface — read, write, snapshot and lock — by delegating to an inner
    store with the ciphertext opened around the calls that need
    plaintext. The write and snapshot halves are not optional
    decoration: ``get_block_store`` hands this object to the apply
    engine, which takes a ``snapshot()`` before applying a proposal and
    calls ``write_block()`` inside it, so a wrapper that implements only
    the read half raises ``AttributeError`` mid-transaction while the
    factory's ``cast(BlockStore, ...)`` asserts it cannot.

    **Read.** Every file path returned by the inner store is
    intercepted: if the file's leading bytes match the encryption magic
    header, the ciphertext is decrypted into a temp plaintext that the
    parser consumes, and the temp is deleted immediately after parse.
    Plain (unencrypted) files pass through unchanged — the wrapper is
    safe to deploy against a partially-migrated workspace. A file that
    carries the header and does not decrypt raises
    :class:`~mind_mem.block_store.BlockStoreError`; it is never reported
    as a file that happens to hold no blocks.

    **Write.** ``write_block`` / ``delete_block`` need the inner store's
    read-modify-write to see Markdown, so the target file is unsealed
    for the call and re-sealed in a ``finally`` — see
    :meth:`_decrypted_target` for which files end up encrypted and for
    the two properties that dance does not have.

    **Snapshot / lock.** ``snapshot`` / ``restore`` / ``diff`` / ``lock``
    are byte- or path-level: they never parse block text, so they
    delegate untouched and a snapshot of an encrypted corpus stays
    encrypted.
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
        for fpath in self.list_blocks():
            parsed = self._parse_maybe_encrypted(fpath)
            if active_only:
                parsed = get_active(parsed)
            blocks.extend(parsed)
        return blocks

    def get_by_id(self, block_id: str) -> Optional[dict[str, Any]]:
        for fpath in self.list_blocks():
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

    def list_blocks(self) -> list[str]:
        """v3.2.0 §1.4: forwards to the underlying store's list_blocks."""
        return self._inner.list_blocks()

    def list_files(self) -> list[str]:
        """Deprecated alias for :meth:`list_blocks` — removed in v4.0."""
        import warnings

        warnings.warn(
            "BlockStore.list_files() is deprecated; use list_blocks() instead. The alias will be removed in v4.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.list_blocks()

    # ------------------------------------------------------------------
    # Write surface — unseal, delegate, re-seal
    # ------------------------------------------------------------------

    def write_block(self, block: dict[str, Any]) -> str:
        """Persist or replace a block, leaving its file as it found it.

        The inner store rewrites the target as UTF-8 text, which is
        meaningless against ciphertext: reading it raises
        ``UnicodeDecodeError`` at the first non-UTF-8 envelope byte, and
        a ciphertext that did happen to decode would be overwritten with
        plaintext, destroying every block already in the file. So the
        target is unsealed for the duration of the delegated call — see
        :meth:`_decrypted_target`.

        Raises:
            UngatedWriteError: no governance admission is open for this
                block (:mod:`mind_mem.admission`).
            ValueError: block is missing ``_id``, or the inner store
                rejects it (unknown prefix, malformed id).
        """
        block_id = block.get("_id")
        if not block_id:
            raise ValueError("block is missing '_id'; cannot write")
        # Enforced here as well as on the inner store: this wrapper is a
        # BlockStore in its own right, so a caller holding only the
        # encrypted store must not get a laxer write surface than one
        # holding the plain store. It runs BEFORE the unseal so an
        # ungated caller cannot make this wrapper drop plaintext on disk.
        require_admission(str(block_id), status=block.get("Status"))
        with self._decrypted_target(str(block_id)):
            return self._inner.write_block(block)

    def delete_block(self, block_id: str) -> bool:
        """Remove a block, leaving its file as it found it.

        Same unseal/re-seal as :meth:`write_block`. The removed content
        is journalled by the inner store to ``memory/deleted_blocks.jsonl``,
        which is **not** encrypted — deleting a block from an encrypted
        corpus therefore leaves its plaintext in the recovery journal.
        That is the inner store's recovery contract, unchanged here and
        stated so an operator can decide about the journal rather than
        discover it.

        The admission check runs **before** the unseal for the same
        reason :meth:`write_block` checks before it: ``_decrypted_target``
        writes the corpus file back out in plaintext for the duration of
        the operation, so relying on the inner store to refuse would let
        an ungated caller put plaintext on disk first and be told "no"
        second. This wrapper opens no scope of its own and records no
        removal — the inner store does both, so a delete leaves exactly
        one chain record however it was reached.
        """
        require_delete_admission(str(block_id))
        with self._decrypted_target(str(block_id)):
            return self._inner.delete_block(block_id)

    # ------------------------------------------------------------------
    # Snapshot surface — byte-level, delegated untouched
    # ------------------------------------------------------------------

    def snapshot(
        self,
        snap_dir: str,
        *,
        files_touched: list[str] | None = None,
    ) -> dict[str, Any]:
        """Create a point-in-time snapshot. Returns the manifest dict.

        Delegated verbatim: the inner store copies file bytes and never
        parses them, so an encrypted corpus is snapshotted as ciphertext
        and restored byte-identically. The snapshot needs no separate
        protection because it never holds plaintext the corpus did not.
        """
        return self._inner.snapshot(snap_dir, files_touched=files_touched)

    def restore(self, snap_dir: str) -> None:
        """Restore the workspace from a snapshot directory (see :meth:`snapshot`)."""
        self._inner.restore(snap_dir)

    def diff(self, snap_dir: str) -> list[str]:
        """Per-file diff vs. a snapshot — a SHA-256 compare of raw bytes.

        Ciphertext against ciphertext, which is sound but conservative:
        the envelope carries a fresh random nonce per encryption, so a
        file rewritten with identical plaintext reads as changed. The
        error is one-sided — a real change is never reported as
        unchanged — which is the direction a rollback needs.
        """
        return self._inner.diff(snap_dir)

    # ------------------------------------------------------------------
    # Lock / cache surface
    # ------------------------------------------------------------------

    def lock(self, *, blocking: bool = True, timeout: float = 30.0) -> Any:
        """Acquire the inner store's exclusive workspace-wide lock.

        Path-level, so nothing here needs decrypting. Note this is the
        lock a caller must take to serialize a whole read-modify-write
        against another process — :meth:`write_block` cannot take it
        internally without deadlocking a caller that already holds it.
        """
        return self._inner.lock(blocking=blocking, timeout=timeout)

    def invalidate_cache(self) -> None:
        """Clear the inner store's file-discovery cache, when it has one.

        Not part of the :class:`~mind_mem.block_store.BlockStore`
        Protocol — only the filesystem-backed stores cache a file list —
        so it is forwarded when present rather than assumed.
        """
        invalidate = getattr(self._inner, "invalidate_cache", None)
        if callable(invalidate):
            invalidate()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @contextmanager
    def _decrypted_target(self, block_id: str) -> Iterator[None]:
        """Hold the canonical file for *block_id* in plaintext for one write.

        Which files end up encrypted afterwards:

        * already ciphertext → unsealed for the write, re-sealed after;
        * present with plaintext content → left plaintext. A workspace
          mid-rollout keeps whatever the operator has not migrated;
          :func:`encrypt_workspace` is the one-shot that brings it over.
        * absent, or present but zero-length → this store is creating the
          file's content, so it follows the corpus around it: born sealed
          in a migrated workspace, born plain in one that has no
          ciphertext yet (:meth:`_corpus_is_sealed`). Writing a new block
          must not be the thing that seals the first file in a workspace
          — a plaintext-only reader would then go quiet on a file it read
          a moment ago. Zero-length counts as absent on purpose:
          ``delete_block`` can empty a file, and reading the empty result
          as "plaintext the operator chose" would drop that file out of
          the sealed set on its next write.

        Two properties this does not have, stated rather than papered
        over:

        * **A plaintext window exists on disk** between the unseal and
          the re-seal. The ``finally`` bounds it — a write that raises
          still re-seals — but a crash inside the window leaves the file
          plaintext. That is recoverable rather than lossy: the read
          path handles plain files, and ``encrypt_workspace`` re-seals
          them. The alternative, reproducing the inner store's
          read-modify-write here so the file is never written in the
          clear, buys a smaller window with a second copy of the write
          semantics to keep in sync.
        * **It is not atomic against a concurrent writer.** The inner
          store's per-file :class:`~mind_mem.mind_filelock.FileLock`
          covers its own read-modify-write, not this unseal/re-seal
          around it, and cannot: that lock is not reentrant, so taking
          it here would deadlock against the inner call. Callers needing
          the whole sequence serialized take :meth:`lock` first.
        """
        target = _resolve_block_file(self._workspace, block_id)
        if target is None:
            # Unmapped prefix: the inner store refuses the write (or
            # reports "not found" for a delete). No file to unseal.
            yield
            return

        had_content = os.path.isfile(target) and os.path.getsize(target) > 0
        was_sealed = had_content and self._em.is_encrypted(target)
        # Short-circuits: the corpus probe only runs on the create path.
        seal_after = was_sealed or (not had_content and self._corpus_is_sealed())
        if was_sealed:
            self._em.decrypt_file_in_place(target)
        try:
            yield
        finally:
            if seal_after and os.path.isfile(target):
                # encrypt_file is a no-op on an already-sealed or
                # zero-length file, so this is safe to call blind.
                self._em.encrypt_file(target)

    def _corpus_is_sealed(self) -> bool:
        """True when any block file the inner store manages is ciphertext.

        The question a create-path write has to answer: is this workspace
        migrated? A file that cannot be opened does not get a vote — it
        cannot tell us it is sealed, and the read path is where being
        unable to read it becomes an error rather than a guess.
        """
        from .encryption import _MAGIC, has_magic

        for fpath in self.list_blocks():
            try:
                with open(fpath, "rb") as fh:
                    if has_magic(fh.read(len(_MAGIC))):
                        return True
            except OSError:
                continue
        return False

    def _parse_maybe_encrypted(self, fpath: str) -> list[dict[str, Any]]:
        """Parse *fpath*, transparently decrypting on the fly.

        Uses a tempfile only when we detect ciphertext so the happy
        path (plaintext file) matches the inner store's performance.
        """
        from .encryption import _MAGIC, has_magic

        try:
            with open(fpath, "rb") as fh:
                head = fh.read(len(_MAGIC))
        except FileNotFoundError:
            # Raced with a delete between list_blocks and here — the file
            # genuinely holds no blocks. Any other OSError (permissions,
            # I/O) propagates: a file we cannot read is not a file with
            # nothing in it.
            return []
        if not has_magic(head):
            return parse_file(fpath)

        try:
            plaintext = self._em.decrypt_file(fpath)
        except (OSError, ValueError) as exc:
            # Never report an unreadable file as an empty one. A wrong or
            # rotated passphrase fails the MAC check here; swallowing it
            # made the whole corpus look empty, which reads to every
            # consumer (recall, scan, governance, export) as "there is
            # nothing here" instead of "you cannot read this" — and a
            # governance pass over a corpus it believes is empty is far
            # worse than one that refuses to run.
            #
            # BlockStoreError deliberately does not subclass ValueError:
            # the corpus walk skips a single unreadable file on
            # OSError/ValueError, and a passphrase failure is not one bad
            # file, it is the whole workspace.
            _log.error("encrypted_read_failed", path=fpath, error=str(exc))
            raise BlockStoreError(
                f"cannot decrypt {os.path.basename(fpath)!r}: {exc}. The workspace passphrase "
                "(MIND_MEM_ENCRYPTION_PASSPHRASE) may be wrong or rotated, or the file may be corrupt."
            ) from exc

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
            # EncryptedBlockStore duck-types the BlockStore Protocol —
            # mypy can't infer structural conformance across the
            # dependent-type wrapper, hence the cast. Fixed in v3.2.1
            # per docs/review-architecture-v3.2.0.md §5.
            return cast(BlockStore, EncryptedBlockStore(workspace, passphrase=passphrase))
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

    from .encryption import _MAGIC, EncryptionManager, has_magic

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
                    # Either record format counts as already-encrypted; a bare
                    # ``_MAGIC`` compare would re-encrypt KMS envelope records.
                    if has_magic(head):
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
