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
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterator, Optional

from .preimage import preimage

# Genesis sentinel — 128 zeros (SHA3-512 produces 128 hex chars)
GENESIS_HASH: str = "0" * 128

#: Extension of the sidecar sealing the chain head, beside the database.
#:
#: THE DEFECT IT CLOSES, measured on a fresh workspace. Two governed
#: writes (four chain rows, four evidence rows), then
#: ``DELETE FROM hash_chain WHERE rowid IN (… ORDER BY rowid DESC LIMIT
#: 2)``::
#:
#:     AFTER   chain=2 evidence=4
#:     [ok] hash_chain: 2 entries verified
#:     verify ok: True  exit: 0
#:
#: and after ``DELETE FROM hash_chain`` outright, ``[ok] hash_chain: 0
#: entries verified``. :meth:`HashChainV2.verify_chain` walks links
#: between the rows that are *present*, so removing a contiguous tail
#: leaves a chain that is internally perfect and shorter than it was.
#: The last row has no successor to bind it and the database has nothing
#: outside itself to be compared against, which is why truncation was
#: undetectable rather than merely unreported.
#:
#: :mod:`~mind_mem.served_ledger` already had this seal
#: (``.mind-mem-ledger/served.head``) and the reasoning is copied
#: wholesale: the sidecar is written after the row, read back
#: unconditionally, and an ABSENT seal is a distinct fact from a blank
#: one — collapsing them is what makes deleting the seal the way to
#: unseal the tail.
HEAD_SUFFIX: str = ".head"


def head_path(db_path: str) -> str:
    """Path of the head sidecar for the chain at *db_path*.

    Derived from the one path :class:`HashChainV2` is given rather than
    from a workspace, so a chain opened anywhere seals beside itself.
    For the workspace ledger this is ``memory/hash_chain_v2.head``, which
    is declared in :data:`~mind_mem.corpus_registry.LEDGER_FILES` — a
    seal a snapshot could capture and a restore could put back is a seal
    that rewinds with the thing it is sealing.
    """
    return os.path.splitext(db_path)[0] + HEAD_SUFFIX


def write_head(db_path: str, head: str) -> None:
    """Replace the head sidecar in one step — temp file, then ``os.replace``.

    ``open(path, "w")`` truncates first and writes second, so between
    those two the seal on disk is zero-length. A blank seal is not a
    neutral value — :func:`read_head` reports it as an OVERWRITTEN seal,
    which would convict an untouched chain — so the rename is atomic and
    no reader ever observes a partial one. The temp name carries the pid
    so a file left by a crashed writer is never the one being renamed.

    Durability is NOT claimed: neither the SQLite commit nor this replace
    is fsync-paired with the other, so a power loss can leave the seal one
    row behind the chain. :func:`verify_head` admits exactly that lag and
    nothing else — see its docstring for why no removal can forge it.
    """
    final = head_path(db_path)
    tmp = f"{final}.{os.getpid()}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as handle:
            handle.write(head + "\n")
        os.replace(tmp, final)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:  # pragma: no cover - the replace already consumed it
                pass


def _seal_tail(conn: sqlite3.Connection, db_path: str) -> None:
    """Seal the chain's CURRENT tail, under the write lock that orders commits.

    Called after the insert has committed, never before: a seal written
    ahead of its row is the exact state :func:`verify_head` convicts, so
    doing it first would manufacture a tamper report out of every failed
    commit.

    The second ``BEGIN IMMEDIATE`` is the whole point, and it is about
    PROCESSES rather than threads. :class:`HashChainV2` serialises its own
    threads with an ``RLock`` and leaves cross-process ordering to SQLite's
    write lock — correct for the rows, and the sidecar is not a row. Two
    processes appending to one workspace (an MCP server and a ``mm``
    command, the ordinary pairing that made
    :func:`mind_mem.served_ledger.append_served_run` take a cross-process
    lock) can therefore commit in one order and write their seals in the
    other, leaving the sidecar naming an entry two or more rows behind the
    tail — which :func:`verify_head` reports as tampering. Re-taking the
    write lock puts every seal write into the same total order as every
    commit, so the last seal written is always the last row committed.

    Read the tail back rather than sealing the caller's own ``entry_hash``:
    under that lock the tail may legitimately be another process's newer
    row, and sealing our own would be writing a seal that is already stale.

    Measured cost of the extra lock round trip, three runs on a loaded
    box, each comparing the two shapes inside ONE process so the load is
    common to both: +0.010, +0.014 and +0.027 ms per append over sealing
    straight after the commit — against ``HashChainV2.append``'s own
    ~0.85 ms, roughly 1-3%. (A two-process A/B of the whole ``append``
    swung from +98% to -11% on the same box and is reported here as
    unusable rather than as a number.) ``BEGIN IMMEDIATE`` + ``commit``
    was also measured against ``BEGIN IMMEDIATE`` + ``rollback``; the
    rollback is not cheaper, so the commit stays.

    The window this closes did not reproduce in 7 200 appends across 6
    concurrent processes, so it is bought on structure rather than on an
    observed failure. What it prevents is a sticky FALSE RED on a tamper
    verifier — a seal left naming an older entry reads exactly like a
    truncation and clears only on the next admission.
    """
    conn.execute("BEGIN IMMEDIATE")
    row = conn.execute("SELECT entry_hash FROM hash_chain ORDER BY rowid DESC LIMIT 1").fetchone()
    if row is not None:
        write_head(db_path, row["entry_hash"])
    conn.commit()


def read_head(db_path: str) -> Optional[str]:
    """The sealed head, or ``None`` when the sidecar is **absent**.

    Absent and empty are different facts and stay distinguishable.
    Collapsing both to ``""`` is what lets a deleted sidecar read as "no
    seal to check against", and a truthiness test on the result then
    skips the comparison entirely — removing the seal would remove the
    check with it.
    """
    try:
        with open(head_path(db_path), encoding="utf-8") as handle:
            return handle.read().strip()
    except OSError:
        return None


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


#: v4 flag gating the sequence-aware import check. See
#: :meth:`HashChainV2.import_jsonl`.
_SEQUENCE_VERIFY_FLAG = "mind_kernels"


def _sequence_verify_enabled() -> bool:
    """Read ``v4.mind_kernels``, fail-closed and QUIET.

    Uses ``is_enabled_quiet``, never ``is_enabled``: the latter warns
    ``v4_config_unreadable`` on a malformed config, and a probe that decides
    whether a feature is on must not itself be observable when the answer is
    no. With the flag OFF this call emits nothing and ``import_jsonl``
    behaves exactly as it did before the kernel was wired.
    """
    try:
        from .v4.feature_flags import is_enabled_quiet

        return is_enabled_quiet(_SEQUENCE_VERIFY_FLAG)
    except Exception:
        return False


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

    @contextmanager
    def _session(self) -> Iterator[sqlite3.Connection]:
        """Open a connection, commit-or-rollback, then CLOSE it.

        Every query below goes through here rather than through
        ``with self._connect() as conn``, because a bare sqlite3
        connection used as a context manager commits (or, on an
        exception, rolls back) and then **leaves the handle open** —
        its ``__exit__`` documents exactly that and nothing more.

        Refcounting does not clean up after it. A ``sqlite3.Connection``
        holds its prepared-statement cache, and that cache holds the
        connection back, so every connection this class opens sits in a
        reference cycle and is reclaimed only when the cyclic collector
        happens to run. Until then the process keeps an open descriptor
        on the database *and* on its ``-wal`` / ``-shm`` sidecars.

        That is a leak on every platform and a correctness bug on
        Windows, where an open handle makes ``os.unlink`` fail: a
        directory holding a chain — a review sandbox, a test workspace —
        could not be deleted. Closing here is also what lets SQLite
        checkpoint and remove the sidecars, so the directory empties.

        Transaction semantics are unchanged: the inner ``with conn``
        still commits on success and rolls back on an exception, and it
        does so *before* the close. ``close()`` on its own never
        commits, so the ordering cannot turn a rollback into a commit.
        """
        conn = self._connect()
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._session() as conn:
            conn.executescript(_SCHEMA)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def length(self) -> int:
        """Total number of entries in the chain."""
        with self._session() as conn:
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
            with self._session() as conn:
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
                _seal_tail(conn, self._db_path)

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
        with self._session() as conn:
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

    def get_by_entry_id(self, entry_id: str) -> Optional[HashEntry]:
        """Return the entry with *entry_id*, or ``None`` if absent.

        ``entry_id`` is UNIQUE in the schema, so this resolves at most one
        row. Used by :meth:`GovernanceGate.admit` to read an entry back out
        of the durable chain before it hands the caller a receipt — an
        admission whose chain entry does not resolve must not authorise a
        write.
        """
        with self._session() as conn:
            row = conn.execute("SELECT * FROM hash_chain WHERE entry_id = ?", (entry_id,)).fetchone()
        return _row_to_entry(row) if row is not None else None

    def get_block_chain(self, block_id: str) -> list[HashEntry]:
        """Return all entries for a specific block in insertion order.

        Args:
            block_id: The block identifier to filter by.

        Returns:
            List of HashEntry objects (may be empty).
        """
        with self._session() as conn:
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
        with self._session() as conn:
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
        with self._session() as conn:
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

        Downgrade monotonicity (flag ``v4.mind_kernels``, default OFF)
        -------------------------------------------------------------
        The per-entry gate below is :meth:`verify_entry`, which accepts the
        v3 scheme **or** the legacy v1 one — it has no memory of what the
        segment has already proven, so it cannot see a v3 entry followed by
        a v1 entry as the downgrade it is. :meth:`verify_chain` *can*, and
        rejects exactly that. The two doors therefore disagreed: a crafted
        export could pass import and then make the whole ledger read as
        broken from the forged entry onward.

        With the flag ON the incoming segment is first checked as a
        SEQUENCE by ``mind_kernels.sha3_512_chain_verify`` — the same
        monotonicity rule ``verify_chain`` applies — anchored to the current
        ledger head. The per-entry loop is left exactly as it was, so every
        input that was rejected before is still rejected with the same
        message and line number; the flag only ever refuses MORE.

        deferred: with the flag OFF the disagreement above is still live.
        It is not fixed unconditionally only because 5.0.1's restoration
        lands every wiring default-OFF and byte-identical. Upgrade path:
        default ``v4.mind_kernels`` to ON, then delete the flag and make
        the sequence check unconditional.

        Args:
            input_path: Path to the JSONL file.

        Returns:
            Number of entries imported.

        Raises:
            ValueError: If any entry is tampered/invalid/corrupt, or (flag
                ON) if the segment downgrades from the v3 entry-hash scheme
                back to v1.
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
            with self._session() as conn:
                conn.execute("BEGIN IMMEDIATE")
                head_row = conn.execute("SELECT entry_hash FROM hash_chain ORDER BY rowid DESC LIMIT 1").fetchone()
                prev_hash = head_row["entry_hash"] if head_row else GENESIS_HASH

                if entries and _sequence_verify_enabled():
                    from .mind_kernels import sha3_512_chain_verify

                    if not sha3_512_chain_verify(
                        [
                            {
                                "entry_id": e.entry_id,
                                "timestamp": e.timestamp,
                                "block_id": e.block_id,
                                "action": e.action,
                                "content_hash": e.content_hash,
                                "previous_hash": e.previous_hash,
                                "entry_hash": e.entry_hash,
                            }
                            for e in entries
                        ],
                        previous_hash=prev_hash,
                    ):
                        conn.rollback()
                        raise ValueError(
                            "Imported segment fails sequence verification: an entry-hash scheme downgrade "
                            "(v3 -> legacy v1) or a broken link. Per-entry verification cannot see this; "
                            "verify_chain() would reject the ledger after the import."
                        )

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
                # The head moved, so the seal moves with it. Without this an
                # ordinary import left the sidecar naming the pre-import tail
                # and `verify_head` would convict a chain nobody touched — a
                # seal that only some writers maintain is a seal that reports
                # legitimate writes as tampering.
                if entries:
                    _seal_tail(conn, self._db_path)

        return len(entries)


# ---------------------------------------------------------------------------
# The head seal
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HeadVerdict:
    """Outcome of :func:`verify_head`.

    ``ok`` means *no disagreement was found*, which is not the same as
    *the tail was checked*: on a chain with no sidecar there is nothing to
    disagree with, and ``ok`` is ``True`` with :attr:`sealed` ``False``.
    A caller that reads ``ok`` alone will report an unsealed chain as a
    verified one — the exact shape "0 entries verified, GREEN" had. Read
    both.
    """

    ok: bool
    #: A sidecar was on disk. ``False`` means the comparison did not run.
    sealed: bool
    length: int
    #: The chain's actual tail ``entry_hash``, or :data:`GENESIS_HASH`.
    head: str
    #: What the sidecar said, or ``None`` when it is absent.
    recorded: Optional[str]
    reason: str


def verify_head(db_path: str) -> HeadVerdict:
    """Compare the chain's tail against its head sidecar. Read-only.

    The comparison is **unconditional**, and both halves of that matter,
    for the same reasons :func:`mind_mem.served_ledger.verify_served_chain`
    gives:

    * it runs when the chain holds NO rows. A recorded head with nothing
      left to name means every row was removed — the one deletion the
      link walk cannot see, because an emptied table has no broken link.
    * it runs when the sidecar is ABSENT with rows present. That state is
      reported (``sealed=False``), never silently passed: skipping the
      check because the seal is gone makes deleting the seal the way to
      unseal the tail.

    THE ONE ALLOWED DISAGREEMENT. :meth:`HashChainV2.append` commits the
    row and *then* replaces the seal, so a process killed between the two
    leaves a chain one row ahead of its sidecar. Refusing that would make
    an ordinary ``SIGKILL`` a permanent one-way door. It is admitted only
    when the tail row NAMES the sealed value as its ``previous_hash``, and
    that state cannot be manufactured by a removal: taking the tail away
    leaves the seal naming a row that is *gone* (neither branch matches),
    and taking any earlier row away breaks the links
    :meth:`HashChainV2.verify_chain` still convicts.

    WHAT THIS DOES NOT CLOSE, named rather than implied: a chain written
    before 5.0.2 has no sidecar, so its tail is unsealed until the next
    admission seals it. Removing the database AND the sidecar together
    leaves a directory indistinguishable from one that never ran — the
    residual :func:`verify_served_chain` names for its own ledger, and for
    the same reason: both records live inside the workspace.
    """
    if not os.path.isfile(db_path):
        orphan = read_head(db_path)
        if orphan is None:
            return HeadVerdict(ok=True, sealed=False, length=0, head=GENESIS_HASH, recorded=None, reason="")
        # A seal that outlived its ledger. Fatal in both modes, and NOT the
        # same finding as ``check_hash_chain``'s "no ledger present": an
        # absent database is a workspace that may never have written, while
        # an absent database beside a surviving seal is positive evidence
        # that one did and the ledger was removed. Treating this as merely
        # unsealed would make "delete the database" quieter than "empty the
        # table", which is the wrong way round.
        return HeadVerdict(
            ok=False,
            sealed=True,
            length=0,
            head=GENESIS_HASH,
            recorded=orphan,
            reason=(
                f"the ledger {os.path.basename(db_path)} is gone but "
                f"{os.path.basename(head_path(db_path))} still seals {(orphan or 'a blank value')[:16]}…: "
                "the chain was removed"
            ),
        )
    try:
        chain = HashChainV2.open_readonly(db_path)
        length = chain.length
        latest = chain.get_latest(n=1) if length else []
    except (sqlite3.DatabaseError, OSError) as exc:
        return HeadVerdict(
            ok=False,
            sealed=False,
            length=0,
            head=GENESIS_HASH,
            recorded=read_head(db_path),
            reason=f"cannot read ledger: {exc}",
        )

    recorded = read_head(db_path)
    head = latest[-1].entry_hash if latest else GENESIS_HASH

    if not latest:
        if recorded is None:
            return HeadVerdict(ok=True, sealed=False, length=0, head=head, recorded=None, reason="")
        return HeadVerdict(
            ok=False,
            sealed=True,
            length=0,
            head=head,
            recorded=recorded,
            reason=(
                f"the chain holds no entry but {os.path.basename(head_path(db_path))} still seals "
                f"{(recorded or 'a blank value')[:16]}…: the rows were removed"
            ),
        )
    if recorded is None:
        return HeadVerdict(
            ok=True,
            sealed=False,
            length=length,
            head=head,
            recorded=None,
            reason=f"{length} entries, tail unsealed — no {os.path.basename(head_path(db_path))} on disk",
        )
    if recorded == head:
        return HeadVerdict(ok=True, sealed=True, length=length, head=head, recorded=recorded, reason="")
    if recorded and recorded == latest[-1].previous_hash:
        # The crash window: the seal lags the tail by exactly one row, and
        # that row names the seal as its predecessor. See the docstring for
        # why no removal reaches this state.
        return HeadVerdict(
            ok=True,
            sealed=True,
            length=length,
            head=head,
            recorded=recorded,
            reason="seal lags the tail by one row (append committed, seal not yet replaced)",
        )
    return HeadVerdict(
        ok=False,
        sealed=True,
        length=length,
        head=head,
        recorded=recorded,
        reason=(
            f"the chain ends at {head[:16]}… but {os.path.basename(head_path(db_path))} seals "
            f"{(recorded or 'a blank value')[:16]}…, which is neither that entry nor its "
            "predecessor: entries were removed or edited"
        ),
    )


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


def _validated_timestamp(value: object, idx: int) -> "str | None":
    """Return a usable timestamp, or raise before anything is written.

    ``None`` is legitimate (the writer falls back to now()); anything that is
    not a string is not, and must be refused during validation rather than at
    append time — otherwise it raises partway through the write loop, after
    earlier entries have already committed.
    """
    if value is None or isinstance(value, str):
        return value
    raise MigrationError(
        f"v1 entry {idx} has a non-string timestamp ({type(value).__name__}); refusing to migrate rather than write a partial chain"
    )


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
        MigrationError: If the v1 file is malformed — a line that is not
            a JSON object, or an entry carrying no ``payload_hash``. The
            whole file is validated before the destination database is
            opened, so a rejected migration leaves no half-written chain
            behind.
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
                parsed = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise MigrationError(f"Malformed JSON at line {line_num}: {exc}") from exc
            # A v1 line that parses but is not an object (a bare list,
            # string or number) used to reach ``v1.get(...)`` and escape as
            # a raw AttributeError, contradicting the documented
            # MigrationError contract.
            if not isinstance(parsed, dict):
                raise MigrationError(f"Malformed v1 entry at line {line_num}: expected a JSON object, got {type(parsed).__name__}")
            v1_entries.append(parsed)

    # Validate every entry BEFORE opening the destination. The v1
    # payload_hash is the payload commitment the migrated chain inherits;
    # an entry without one used to migrate to ``content=""``, and the
    # resulting ledger then verified as fully intact — a chain that
    # certifies itself while having dropped what it commits to. Refuse the
    # migration instead, and refuse it before any of it is written.
    prepared: list[tuple[str, str, str, Optional[str]]] = []
    for idx, v1 in enumerate(v1_entries, 1):
        content = v1.get("payload_hash")
        if not isinstance(content, str) or not content:
            raise MigrationError(f"v1 entry {idx} has no usable payload_hash; migrating it would drop the payload commitment it carries")
        prepared.append(
            (
                str(v1.get("target", "unknown")),
                str(v1.get("operation", "unknown")),
                content,
                # Preserve the v1 timestamp so temporal queries still work
                # on migrated chains. Fall back to now() when absent.
                # Type-checked HERE, in the validate-everything-first phase,
                # not discovered at append time: a non-string timestamp reaching
                # the write loop raises after earlier entries have committed.
                _validated_timestamp(v1.get("timestamp"), idx),
            )
        )

    # Build into a staging database and swap it in only once EVERY entry has
    # landed. Appending straight to new_db_path commits entry by entry, so a
    # failure on entry N leaves 1..N-1 behind: a destination chain that verifies
    # internally while being silently truncated, which is worse than no chain.
    # os.replace is atomic on POSIX and Windows.
    staging = f"{new_db_path}.migrating"

    def _clear_staging() -> None:
        for leftover in (staging, f"{staging}-wal", f"{staging}-shm"):
            try:
                os.remove(leftover)
            except OSError:
                pass

    _clear_staging()
    try:
        staged = HashChainV2(staging)
        for block_id, action, content, original_ts in prepared:
            staged.append(block_id, action, content, timestamp=original_ts)
        os.replace(staging, new_db_path)
    except BaseException:
        _clear_staging()
        raise

    return HashChainV2(new_db_path)
