# Copyright 2026 STARGA, Inc.
"""Redaction orchestration — destroy the content, keep the proof.

:mod:`mind_mem.tombstone` owns the ledger record; this module owns the
*order of operations* that makes a redaction safe, and the derived-copy
sweep that makes it real.

Order (deliberate, and the reason this lives in one place):

1. Resolve the block's preserved Merkle leaf **before** anything is
   destroyed — ``index_meta`` is authoritative because that is the exact
   hash the live tree holds.
2. Admit ``REDACT`` through :class:`~mind_mem.governance_gate.GovernanceGate`
   (evidence chain + SHA3-512 hash chain) and append a ``delete_block``
   entry to the SHA-256 :class:`~mind_mem.audit_chain.AuditChain`. Both
   store digests only — the content itself is never persisted.
3. Write the tombstone, binding those receipts and back-linking to the
   previous tombstone.
4. **Then** destroy the corpus text (caller-supplied callable).
5. Sweep the derived copies: FTS rows, cached embeddings, block metadata
   and any pre-existing plaintext recovery receipt.

Steps 1–3 happen before 4 on purpose. If a chain write fails the content
is still there and the caller can retry; the reverse ordering could
destroy content and leave no proof it ever existed, which is the one
failure this feature exists to prevent. Between 3 and 4 a block is
already unreachable through recall (the ledger filter short-circuits it),
so a crash in that window fails safe.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
from datetime import datetime, timezone
from typing import Callable, Final

from .observability import get_logger, metrics
from .tombstone import Tombstone, TombstoneError, append_tombstone, get_tombstone

_log = get_logger("tombstone_redact")

#: Derived-copy tables swept on redaction. Absent tables are skipped —
#: a workspace may never have built vectors or the FTS schema.
_PURGE_TABLES: Final[tuple[tuple[str, str], ...]] = (
    ("blocks", "id"),
    ("blocks_fts", "block_id"),
    ("index_meta", "block_id"),
    ("block_meta", "id"),
    ("block_vectors", "id"),
    ("embedding_cache", "block_id"),
    ("vec_blocks", "block_id"),
)

_RECOVERY_LOG_REL: Final[str] = os.path.join("memory", "deleted_blocks.jsonl")


# ---------------------------------------------------------------------------
# Leaf resolution
# ---------------------------------------------------------------------------


def resolve_leaf_hash(
    workspace: str,
    block_id: str,
    *,
    raw_text: str,
    parsed_block: dict | None = None,
) -> tuple[str, str]:
    """Return ``(leaf_hash, leaf_source)`` for the block about to be redacted.

    Preference order matches :data:`mind_mem.tombstone.LEAF_SOURCES`:

    * ``index_meta`` — the hash the live Merkle tree actually holds.
    * ``parsed_block`` — recomputed with the indexer's own function, so
      it equals what the tree *would* hold once built.
    * ``raw_text`` — last resort when the block cannot be parsed; the
      leaf is then a digest of the source text and is flagged as such
      so an auditor knows it will not reproduce a historical proof.
    """
    from .sqlite_index import _compute_block_hash, _db_path

    db_path = _db_path(workspace)
    if os.path.isfile(db_path):
        conn = None
        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            row = conn.execute(
                "SELECT content_hash FROM index_meta WHERE block_id = ? ORDER BY file_path LIMIT 1",
                (block_id,),
            ).fetchone()
            if row and row[0]:
                return str(row[0]), "index_meta"
        except sqlite3.Error:
            _log.debug("tombstone_leaf_index_lookup_failed", block_id=block_id)
        finally:
            if conn is not None:
                conn.close()

    if parsed_block is not None:
        return _compute_block_hash(parsed_block), "parsed_block"

    return hashlib.sha256(raw_text.encode("utf-8")).hexdigest(), "raw_text"


# ---------------------------------------------------------------------------
# Derived-copy sweep
# ---------------------------------------------------------------------------


def purge_index(workspace: str, block_id: str) -> dict[str, int]:
    """Delete every index-side copy of *block_id* (and its fact children).

    Returns ``{table: rows_deleted}``. Missing tables and a missing
    database are not errors — they mean there is nothing to destroy.
    """
    from .sqlite_index import _db_path

    deleted: dict[str, int] = {}
    db_path = _db_path(workspace)
    if not os.path.isfile(db_path):
        return deleted

    conn = None
    try:
        conn = sqlite3.connect(db_path, timeout=30.0)
        conn.execute("PRAGMA busy_timeout=30000")
        # Erasure, not unlinking: without secure_delete SQLite leaves the
        # deleted cell bytes readable in the freed page, so `strings` on
        # recall.db would still yield the "destroyed" block.
        conn.execute("PRAGMA secure_delete=ON")
        ids = [block_id]
        try:
            ids.extend(str(r[0]) for r in conn.execute("SELECT id FROM blocks WHERE parent_id = ?", (block_id,)).fetchall())
        except sqlite3.Error:
            pass  # `blocks` absent — FTS schema never built for this workspace.

        placeholders = ",".join("?" for _ in ids)
        for table, column in _PURGE_TABLES:
            try:
                cur = conn.execute(
                    f"DELETE FROM {table} WHERE {column} IN ({placeholders})",  # nosec B608 — table/column from the fixed _PURGE_TABLES tuple; ids bound as params
                    ids,
                )
                if cur.rowcount > 0:
                    deleted[table] = cur.rowcount
            except sqlite3.Error:
                continue  # table not present in this workspace
        try:
            cur = conn.execute(
                f"DELETE FROM xref_edges WHERE src IN ({placeholders}) OR dst IN ({placeholders})",  # nosec B608 — placeholders is `? * N`, ids bound as params
                ids + ids,
            )
            if cur.rowcount > 0:
                deleted["xref_edges"] = cur.rowcount
        except sqlite3.Error:
            pass
        conn.commit()
        _rebuild_fts(conn, block_id)
        _scrub_free_pages(conn, block_id)
    except sqlite3.Error as exc:
        _log.warning("tombstone_purge_index_failed", block_id=block_id, error=str(exc))
    finally:
        if conn is not None:
            conn.close()
    return deleted


def _rebuild_fts(conn: sqlite3.Connection, block_id: str) -> None:
    """Rebuild the FTS5 index so the deleted document's terms are gone.

    ``DELETE FROM blocks_fts`` is a *logical* delete: FTS5 records a
    delete marker and leaves the original terms sitting in the segment
    b-tree until a merge. A redacted block's distinctive words are
    therefore still readable in ``blocks_fts_data`` after an ordinary
    delete — proven by the erasure test, which greps the raw database
    file. ``rebuild`` regenerates the whole index from the content
    table, which no longer holds the row.
    """
    try:
        conn.execute("INSERT INTO blocks_fts(blocks_fts) VALUES('rebuild')")
        conn.commit()
    except sqlite3.Error as exc:
        _log.warning("tombstone_fts_rebuild_failed", block_id=block_id, error=str(exc))


def _scrub_free_pages(conn: sqlite3.Connection, block_id: str) -> None:
    """Remove the copies SQLite keeps outside the live rows.

    Two of them, both of which hand back "deleted" text verbatim:

    * the **write-ahead log** still holds the pre-delete page images
      until a checkpoint — ``TRUNCATE`` folds them in and zeroes the
      ``-wal`` file;
    * **free pages** inside the main database can retain content written
      before ``secure_delete`` was switched on — ``VACUUM`` rebuilds the
      file without them.

    Best-effort: a concurrent reader can hold the lock VACUUM needs. The
    block is already unreachable at that point, so a failure is a loud
    warning rather than a rollback — but it is a warning an operator
    must act on before claiming erasure, hence the explicit log event.
    """
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    except sqlite3.Error as exc:
        _log.warning("tombstone_wal_checkpoint_failed", block_id=block_id, error=str(exc))
    try:
        conn.isolation_level = None  # VACUUM cannot run inside a transaction
        conn.execute("VACUUM")
    except sqlite3.Error as exc:
        _log.warning("tombstone_vacuum_failed", block_id=block_id, error=str(exc))
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    except sqlite3.Error as exc:
        _log.debug("tombstone_post_vacuum_checkpoint_failed", block_id=block_id, error=str(exc))


def scrub_recovery_log(workspace: str, block_id: str, record_hash: str) -> int:
    """Replace plaintext receipts for *block_id* in ``deleted_blocks.jsonl``.

    A block deleted before tombstones were enabled (or deleted, recreated
    and redacted) leaves a full plaintext copy in the recovery journal.
    Right-to-forget means that copy goes too. The receipt is not dropped
    — it is rewritten without ``content`` and pointed at the tombstone,
    so the journal still shows a deletion happened.

    Returns the number of receipts scrubbed.
    """
    from .mind_filelock import FileLock

    path = os.path.join(os.path.abspath(workspace), _RECOVERY_LOG_REL)
    if not os.path.isfile(path):
        return 0

    scrubbed = 0
    with FileLock(path + ".lock"):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                lines = fh.readlines()
        except OSError as exc:
            _log.warning("tombstone_scrub_read_failed", block_id=block_id, error=str(exc))
            return 0

        rewritten: list[str] = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                entry = json.loads(stripped)
            except ValueError:
                rewritten.append(stripped + "\n")
                continue
            if entry.get("block_id") != block_id or "content" not in entry:
                rewritten.append(stripped + "\n")
                continue
            scrubbed += 1
            rewritten.append(
                json.dumps(
                    {
                        "block_id": block_id,
                        "deleted_at": entry.get("deleted_at", ""),
                        "content_redacted": True,
                        "tombstone_record_hash": record_hash,
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )

        if scrubbed:
            try:
                _atomic_rewrite(path, "".join(rewritten))
            except OSError as exc:
                # Expected when the operator has applied the OS-level
                # append-only attribute to the recovery journal (see
                # docs/append-only-audit-logs.md): the rewrite the scrub
                # needs is exactly what `chattr +a` forbids. The block is
                # already redacted everywhere else, so this is a loud
                # warning naming the file an operator must clear by hand,
                # never a failed redaction.
                _log.warning("tombstone_scrub_write_failed", block_id=block_id, path=path, error=str(exc))
                return 0
    return scrubbed


def _atomic_rewrite(path: str, text: str) -> None:
    """Replace *path* with *text* via a same-directory temp file + rename."""
    directory = os.path.dirname(path)
    fd, tmp_path = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------


def redact_block(
    workspace: str,
    block_id: str,
    *,
    content: str,
    source_file: str,
    actor: str,
    reason: str,
    parsed_block: dict | None = None,
    destroy: Callable[[], None] | None = None,
) -> Tombstone:
    """Redact a live block: chain the event, then destroy the content.

    Args:
        workspace: Workspace root.
        block_id: Block being redacted (must currently exist).
        content: The block's raw text — hashed for the chains, never stored.
        source_file: Workspace-relative path the block lived in.
        actor: Who requested the deletion. Recorded in every chain.
        reason: Why. Recorded in every chain; must not be empty.
        parsed_block: Parsed block dict, when available, for leaf recompute.
        destroy: Injected callable that removes the content from the corpus.
            Called only after the tombstone is durable. Not called at all
            when ``None`` (caller destroys the content itself afterwards).

    Returns:
        The written :class:`~mind_mem.tombstone.Tombstone`.

    Raises:
        TombstoneError: reason/actor missing, or the ledger refuses the record.
        Exception: propagated from the governance gate (spec-hash drift)
            or the audit chain — content is untouched in that case.
    """
    if not reason or not reason.strip():
        raise TombstoneError("redaction requires an explicit reason — it is chained as the justification for destroying content")
    if not actor or not actor.strip():
        raise TombstoneError("redaction requires an actor")

    leaf_hash, leaf_source = resolve_leaf_hash(workspace, block_id, raw_text=content, parsed_block=parsed_block)
    content_bytes = len(content.encode("utf-8"))
    content_sha3_512 = hashlib.sha3_512(content.encode("utf-8")).hexdigest()
    redacted_at = datetime.now(timezone.utc).isoformat()

    # Step 2a — governance gate: evidence chain + SHA3-512 hash chain.
    from .governance_gate import get_gate

    receipt = get_gate(workspace).admit_receipt(
        "REDACT",
        block_id,
        content,
        actor=actor,
        target_file=source_file,
        metadata={
            "reason": reason,
            "tombstone": "v1",
            "leaf_hash": leaf_hash,
            "leaf_source": leaf_source,
            "content_bytes": content_bytes,
        },
    )

    # Step 2b — SHA-256 mutation ledger (carries agent + reason as fields).
    from .audit_chain import AuditChain

    audit_entry = AuditChain(workspace).append(
        "delete_block",
        source_file,
        agent=actor,
        reason=reason,
        payload={
            "block_id": block_id,
            "leaf_hash": leaf_hash,
            "leaf_source": leaf_source,
            "content_sha3_512": content_sha3_512,
            "content_bytes": content_bytes,
            "tombstone": "v1",
        },
    )

    # Step 3 — the tombstone itself, binding both receipts.
    record = append_tombstone(
        workspace,
        block_id=block_id,
        redacted_at=redacted_at,
        actor=actor,
        reason=reason,
        leaf_hash=leaf_hash,
        leaf_source=leaf_source,
        content_sha3_512=content_sha3_512,
        content_bytes=content_bytes,
        source_file=source_file,
        evidence_id=receipt.evidence_id,
        evidence_hash=receipt.evidence_hash,
        chain_entry_id=receipt.chain_entry_id,
        chain_entry_hash=receipt.chain_entry_hash,
        audit_seq=audit_entry.seq,
        audit_entry_hash=audit_entry.entry_hash,
    )

    # Step 4 — destroy the corpus content.
    if destroy is not None:
        destroy()

    # Step 5 — sweep derived copies. Best-effort by design: the block is
    # already unreachable via the ledger filter, so a sweep failure is a
    # loud warning, not a reason to leave a half-redacted block behind.
    try:
        purged = purge_index(workspace, block_id)
        scrubbed = scrub_recovery_log(workspace, block_id, record.record_hash)
    except OSError as exc:
        # The tombstone is durable and the corpus text is gone; a sweep
        # failure must not turn a completed redaction into an exception
        # the caller reads as "nothing happened".
        _log.warning("tombstone_sweep_failed", block_id=block_id, error=str(exc))
        purged, scrubbed = {}, 0

    _log.info(
        "tombstone_redacted",
        block_id=block_id,
        actor=actor,
        purged=sum(purged.values()),
        scrubbed=scrubbed,
    )
    metrics.inc("tombstone_redactions")
    return record


def already_tombstoned(workspace: str, block_id: str) -> Tombstone | None:
    """Return the tombstone for *block_id* when one exists (idempotency check)."""
    return get_tombstone(workspace, block_id)


__all__ = [
    "already_tombstoned",
    "purge_index",
    "redact_block",
    "resolve_leaf_hash",
    "scrub_recovery_log",
]

# deferred: erasure is guaranteed at the store's logical surfaces (corpus
# file, index rows, WAL, free pages, recovery journal), NOT against raw-disk
# forensics — the corpus rewrite is a temp-file rename, so the old extent may
# survive on the block device — upgrade path: document the filesystem-level
# requirement (encrypted-at-rest volume or fs-level secure erase) in the
# compliance guide, and offer an opt-in overwrite-in-place corpus writer.
# deferred: non-Markdown block stores (Postgres) keep the block of record in
# the store, so their delete path must purge the row there too — upgrade path:
# add a store-level `redact(block_id)` to the BlockStore protocol and call it
# from redact_block's step 4 destroy callable.
# deferred: external vector backends (chroma / faiss directories under
# .mind-mem-index/) are not swept — upgrade path: extend purge_index with a
# backend-dispatch that calls each configured vector backend's delete-by-id.
