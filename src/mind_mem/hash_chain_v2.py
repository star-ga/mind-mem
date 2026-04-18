# Copyright 2026 STARGA, Inc.
"""HashChainV2 — SHA3-512 per-block hash chain with SQLite persistence.

Upgrade from audit_chain (SHA256) to SHA3-512 with per-block sub-chains.
Each entry commits to the full global chain via previous_hash, and the
block_id grouping allows efficient per-block history queries.

Schema (SQLite):
    hash_chain(
        rowid        INTEGER PRIMARY KEY,
        entry_id     TEXT    NOT NULL UNIQUE,
        timestamp    TEXT    NOT NULL,
        block_id     TEXT    NOT NULL,
        action       TEXT    NOT NULL,
        content_hash TEXT    NOT NULL,  -- SHA3-512(content)
        previous_hash TEXT   NOT NULL,  -- SHA3-512 of prior entry, or GENESIS
        entry_hash   TEXT    NOT NULL   -- SHA3-512(canonical fields)
    )

Usage:
    from mind_mem.hash_chain_v2 import HashChainV2

    chain = HashChainV2("/path/to/chain.db")
    entry = chain.append("block-42", "create", "initial content")
    valid, broken_at = chain.verify_chain()

Migration from v1:
    from mind_mem.hash_chain_v2 import convert_from_v1
    new_chain = convert_from_v1("/path/to/v1_chain.jsonl", "/path/to/new.db")
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from .preimage import preimage

# Genesis sentinel — 128 zeros (SHA3-512 produces 128 hex chars)
GENESIS_HASH: str = "0" * 128

_SCHEMA = """
CREATE TABLE IF NOT EXISTS hash_chain (
    rowid        INTEGER PRIMARY KEY AUTOINCREMENT,
    entry_id     TEXT    NOT NULL UNIQUE,
    timestamp    TEXT    NOT NULL,
    block_id     TEXT    NOT NULL,
    action       TEXT    NOT NULL,
    content_hash TEXT    NOT NULL,
    previous_hash TEXT   NOT NULL,
    entry_hash   TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_block_id ON hash_chain (block_id);
"""


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HashEntry:
    """Immutable record of a single hash-chain entry."""

    entry_id: str
    timestamp: str
    block_id: str
    action: str
    content_hash: str  # SHA3-512(content)
    previous_hash: str  # global chain linkage
    entry_hash: str  # SHA3-512(canonical representation)


class MigrationError(Exception):
    """Raised when a v1 → v2 migration cannot be completed."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sha3(data: str) -> str:
    return hashlib.sha3_512(data.encode("utf-8")).hexdigest()


def _compute_content_hash(content: str) -> str:
    return _sha3(content)


def _compute_entry_hash_v1(
    entry_id: str,
    timestamp: str,
    block_id: str,
    action: str,
    content_hash: str,
    previous_hash: str,
) -> str:
    """Legacy v1 entry-hash scheme.

    Fields joined by ``|`` — kept for backward verification of chains
    written before v2.10.0. New entries hash via :func:`_compute_entry_hash_v3`.
    """
    canonical = f"{entry_id}|{timestamp}|{block_id}|{action}|{content_hash}|{previous_hash}"
    return _sha3(canonical)


def _compute_entry_hash_v3(
    entry_id: str,
    timestamp: str,
    block_id: str,
    action: str,
    content_hash: str,
    previous_hash: str,
) -> str:
    """v3 entry-hash scheme (v2.10.0+): TAG_v1 NUL-separated preimage.

    Uses :mod:`preimage` so field values containing ``|`` (or any other
    ambiguous boundary character) can't craft collisions. Still SHA3-512.
    """
    pre = preimage(
        "CHAIN_v1",
        entry_id,
        timestamp,
        block_id,
        action,
        content_hash,
        previous_hash,
    )
    return hashlib.sha3_512(pre).hexdigest()


# Public alias — new entries hash via v3. Verification code must try v3
# first then fall back to v1 so pre-v2.10.0 chains continue to verify.
_compute_entry_hash = _compute_entry_hash_v3


def _row_to_entry(row: sqlite3.Row) -> HashEntry:
    return HashEntry(
        entry_id=row["entry_id"],
        timestamp=row["timestamp"],
        block_id=row["block_id"],
        action=row["action"],
        content_hash=row["content_hash"],
        previous_hash=row["previous_hash"],
        entry_hash=row["entry_hash"],
    )


# ---------------------------------------------------------------------------
# HashChainV2
# ---------------------------------------------------------------------------


class HashChainV2:
    """SHA3-512 append-only hash chain with SQLite backend.

    Thread-safety: each public method opens and closes its own connection.
    SQLite WAL mode is enabled for concurrent read performance.
    """

    def __init__(self, db_path: str, *, readonly: bool = False) -> None:
        self._db_path = os.path.realpath(db_path)
        # Serialize appends across threads: SQLite writer serialization alone
        # is not enough when the same process holds multiple connections and
        # each reads-then-writes the chain head (TOCTOU on previous_hash).
        self._lock = threading.RLock()
        self._readonly = bool(readonly)
        if not self._readonly:
            os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
            self._init_db()

    @classmethod
    def open_readonly(cls, db_path: str) -> "HashChainV2":
        """Open an existing chain database without writing to it.

        The standalone verifier (:mod:`mind_mem.verify_cli`) uses this so
        auditing a workspace never mutates the ledger — not even via the
        otherwise-idempotent ``CREATE TABLE IF NOT EXISTS`` schema
        touch. Append / import paths raise on a read-only instance.
        """
        return cls(db_path, readonly=True)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        # isolation_level="DEFERRED" keeps explicit transaction control
        # (autocommit via None silently defeats BEGIN EXCLUSIVE / BEGIN IMMEDIATE).
        # timeout=30s avoids immediate OperationalError on transient locks.
        if self._readonly:
            # URI form with mode=ro opens the DB read-only without
            # creating it and without acquiring a write lock. We do
            # NOT use immutable=1 because that flag tells SQLite to
            # skip the -wal file, which would hide recent committed
            # writes from the verifier on WAL-mode databases.
            uri = f"file:{self._db_path}?mode=ro"
            conn = sqlite3.connect(uri, uri=True, timeout=30.0)
        else:
            conn = sqlite3.connect(self._db_path, timeout=30.0, isolation_level="DEFERRED")
        conn.row_factory = sqlite3.Row
        if not self._readonly:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def length(self) -> int:
        """Total number of entries in the chain."""
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM hash_chain").fetchone()
            return int(row[0])

    def append(
        self,
        block_id: str,
        action: str,
        content: str,
        *,
        timestamp: Optional[str] = None,
    ) -> HashEntry:
        """Append a new entry to the global chain.

        Args:
            block_id: Logical identifier for the block being mutated.
            action:   Verb describing the mutation (create, update, delete, …).
            content:  Raw content whose SHA3-512 digest is stored.
            timestamp: Optional ISO8601 timestamp override (private use for
                migration). Public callers should let the default (now) apply.

        Returns:
            The newly created, immutable HashEntry.
        """
        if self._readonly:
            raise PermissionError("HashChainV2 opened read-only; append() is not permitted")
        entry_id = str(uuid.uuid4())
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat()
        content_hash = _compute_content_hash(content)

        # Serialize reads-then-writes across threads sharing this instance.
        # SQLite BEGIN IMMEDIATE alone serializes writers at the DB level, but
        # Python-level lock avoids raising OperationalError on concurrent
        # intra-process appends that would otherwise collide.
        with self._lock:
            with self._connect() as conn:
                conn.execute("BEGIN IMMEDIATE")
                last_row = conn.execute("SELECT entry_hash FROM hash_chain ORDER BY rowid DESC LIMIT 1").fetchone()
                previous_hash = last_row["entry_hash"] if last_row else GENESIS_HASH

                entry_hash = _compute_entry_hash(entry_id, timestamp, block_id, action, content_hash, previous_hash)

                conn.execute(
                    """
                    INSERT INTO hash_chain
                        (entry_id, timestamp, block_id, action, content_hash, previous_hash, entry_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (entry_id, timestamp, block_id, action, content_hash, previous_hash, entry_hash),
                )
                conn.commit()

        return HashEntry(
            entry_id=entry_id,
            timestamp=timestamp,
            block_id=block_id,
            action=action,
            content_hash=content_hash,
            previous_hash=previous_hash,
            entry_hash=entry_hash,
        )

    def verify_entry(self, entry: HashEntry) -> bool:
        """Verify a single entry's internal consistency.

        Tries the v3 scheme (v2.10.0+) first and falls back to the v1
        legacy scheme so pre-v2.10.0 chains still verify. Does NOT
        check linkage to adjacent entries.
        """
        args = (
            entry.entry_id,
            entry.timestamp,
            entry.block_id,
            entry.action,
            entry.content_hash,
            entry.previous_hash,
        )
        return entry.entry_hash == _compute_entry_hash_v3(*args) or entry.entry_hash == _compute_entry_hash_v1(*args)

    def verify_chain(self) -> tuple[bool, int]:
        """Verify the full global chain integrity.

        Walks every entry in insertion order, checking:
        - entry_hash matches recomputed value
        - previous_hash links to the prior entry's entry_hash
        - once a v3 entry is seen, NO downgrade to v1 is tolerated
          (downgrade attack mitigation — without this rule an attacker
          could forge a v1-hashed entry after a v3 entry since the v1
          scheme is separator-injection-vulnerable)

        Returns:
            (valid: bool, first_broken_index: int)
            first_broken_index is -1 when the chain is valid, or the
            0-based index of the first broken entry.
        """
        # Stream rows with fetchmany so a million-entry ledger doesn't
        # materialise the whole table in-process when an MCP caller
        # triggers verification.
        prev_hash = GENESIS_HASH
        idx = -1
        seen_v3 = False
        with self._connect() as conn:
            cur = conn.execute("SELECT * FROM hash_chain ORDER BY rowid ASC")
            while True:
                batch = cur.fetchmany(1024)
                if not batch:
                    break
                for row in batch:
                    idx += 1
                    entry = _row_to_entry(row)

                    if entry.previous_hash != prev_hash:
                        return False, idx

                    args = (
                        entry.entry_id,
                        entry.timestamp,
                        entry.block_id,
                        entry.action,
                        entry.content_hash,
                        entry.previous_hash,
                    )
                    v3_ok = entry.entry_hash == _compute_entry_hash_v3(*args)
                    if v3_ok:
                        seen_v3 = True
                    elif seen_v3:
                        # Downgrade blocked: chain already produced a v3
                        # entry, this one is not v3 → reject without
                        # consulting the legacy v1 scheme.
                        return False, idx
                    elif entry.entry_hash != _compute_entry_hash_v1(*args):
                        return False, idx

                    prev_hash = entry.entry_hash

        return True, -1

    def get_block_chain(self, block_id: str) -> list[HashEntry]:
        """Return all entries for a specific block in insertion order.

        Args:
            block_id: The block identifier to filter by.

        Returns:
            List of HashEntry objects (may be empty).
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM hash_chain WHERE block_id = ? ORDER BY rowid ASC",
                (block_id,),
            ).fetchall()
        return [_row_to_entry(r) for r in rows]

    def get_latest(self, n: int = 10) -> list[HashEntry]:
        """Return the n most recent entries (chronological order).

        Args:
            n: Maximum number of entries to return (default 10).

        Returns:
            List of HashEntry objects, oldest first within the window.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM hash_chain ORDER BY rowid DESC LIMIT ?",
                (n,),
            ).fetchall()
        # Reverse so returned order is oldest-first (insertion order)
        return [_row_to_entry(r) for r in reversed(rows)]

    def export_jsonl(self, output_path: str) -> int:
        """Export the full chain to a JSONL file.

        Each line is a JSON object with all HashEntry fields.

        Args:
            output_path: Destination file path.

        Returns:
            Number of entries exported.
        """
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM hash_chain ORDER BY rowid ASC").fetchall()

        with open(output_path, "w", encoding="utf-8") as fh:
            for row in rows:
                entry = _row_to_entry(row)
                fh.write(
                    json.dumps(
                        {
                            "entry_id": entry.entry_id,
                            "timestamp": entry.timestamp,
                            "block_id": entry.block_id,
                            "action": entry.action,
                            "content_hash": entry.content_hash,
                            "previous_hash": entry.previous_hash,
                            "entry_hash": entry.entry_hash,
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )

        return len(rows)

    def import_jsonl(self, input_path: str) -> int:
        """Import a JSONL file produced by export_jsonl.

        Validates each entry's internal consistency before writing.
        Raises ValueError if any entry fails verification.

        Args:
            input_path: Path to the JSONL file.

        Returns:
            Number of entries imported.

        Raises:
            ValueError: If any entry is tampered/invalid/corrupt.
            FileNotFoundError: If input_path does not exist.
        """
        if self._readonly:
            raise PermissionError("HashChainV2 opened read-only; import_jsonl is not permitted")
        entries = _load_jsonl_entries(input_path)

        # Lock for the entire head-read + validate + insert sequence so a
        # concurrent append() cannot land an entry between our head snapshot
        # and our bulk insert (which would produce entries whose
        # previous_hash links to a stale head — verify_chain would then flag
        # the entire import as tampered).
        with self._lock:
            with self._connect() as conn:
                conn.execute("BEGIN IMMEDIATE")
                head_row = conn.execute("SELECT entry_hash FROM hash_chain ORDER BY rowid DESC LIMIT 1").fetchone()
                prev_hash = head_row["entry_hash"] if head_row else GENESIS_HASH

                for idx, entry in enumerate(entries):
                    if not self.verify_entry(entry):
                        conn.rollback()
                        raise ValueError(f"Entry at line {idx + 1} (id={entry.entry_id}) is tampered or corrupt")
                    if entry.previous_hash != prev_hash:
                        conn.rollback()
                        raise ValueError(
                            f"Entry at line {idx + 1} (id={entry.entry_id}) breaks chain linkage: "
                            f"expected previous_hash={prev_hash[:16]}… "
                            f"got {entry.previous_hash[:16]}…"
                        )
                    prev_hash = entry.entry_hash

                # Transaction already open from the earlier BEGIN IMMEDIATE
                # above; just commit the batch we validated.
                conn.executemany(
                    """
                    INSERT INTO hash_chain
                        (entry_id, timestamp, block_id, action, content_hash, previous_hash, entry_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            e.entry_id,
                            e.timestamp,
                            e.block_id,
                            e.action,
                            e.content_hash,
                            e.previous_hash,
                            e.entry_hash,
                        )
                        for e in entries
                    ],
                )
                conn.commit()

        return len(entries)


# ---------------------------------------------------------------------------
# Internal load helper
# ---------------------------------------------------------------------------


def _load_jsonl_entries(path: str) -> list[HashEntry]:
    """Parse a JSONL file into a list of HashEntry objects."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Chain file not found: {path}")

    entries: list[HashEntry] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line_num, line in enumerate(fh, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_num}: {exc}") from exc
            try:
                entries.append(
                    HashEntry(
                        entry_id=obj["entry_id"],
                        timestamp=obj["timestamp"],
                        block_id=obj["block_id"],
                        action=obj["action"],
                        content_hash=obj["content_hash"],
                        previous_hash=obj["previous_hash"],
                        entry_hash=obj["entry_hash"],
                    )
                )
            except KeyError as exc:
                raise ValueError(f"Missing field {exc} at line {line_num}") from exc

    return entries


# ---------------------------------------------------------------------------
# Migration helper
# ---------------------------------------------------------------------------


def convert_from_v1(old_chain_path: str, new_db_path: str) -> HashChainV2:
    """Migrate a v1 SHA256 audit chain to a v2 SHA3-512 hash chain.

    Reads each entry from the v1 JSONL file and re-inserts it into a
    fresh HashChainV2 database. The v1 payload_hash (SHA256) is stored
    verbatim as the content for the new entry so the SHA3-512
    content_hash commits to the original digest rather than raw content.

    Args:
        old_chain_path: Path to v1 JSONL chain file.
        new_db_path:    Destination SQLite database path.

    Returns:
        The newly created HashChainV2 instance.

    Raises:
        FileNotFoundError: If old_chain_path does not exist.
        MigrationError: If the v1 file is malformed.
    """
    if not os.path.isfile(old_chain_path):
        raise FileNotFoundError(f"v1 chain not found: {old_chain_path}")

    v1_entries: list[dict] = []
    with open(old_chain_path, "r", encoding="utf-8") as fh:
        for line_num, line in enumerate(fh, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                v1_entries.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise MigrationError(f"Malformed JSON at line {line_num}: {exc}") from exc

    new_chain = HashChainV2(new_db_path)

    for v1 in v1_entries:
        try:
            block_id = v1.get("target", "unknown")
            action = v1.get("operation", "unknown")
            # Use the v1 payload_hash as content; preserves the original digest
            content = v1.get("payload_hash", "")
            # Preserve the v1 timestamp so temporal queries still work on
            # migrated chains. Fall back to now() when absent.
            original_ts = v1.get("timestamp")
        except (KeyError, TypeError) as exc:
            raise MigrationError(f"Cannot read v1 entry: {exc}") from exc

        new_chain.append(block_id, action, content, timestamp=original_ts)

    return new_chain
