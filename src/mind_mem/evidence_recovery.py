# Copyright 2026 STARGA, Inc.
"""Seal a forked evidence chain and start a new one that anchors it.

:class:`~mind_mem.evidence_objects.EvidenceChain` refuses to append to a
store whose history did not load intact, and the refusal is right: the
in-memory chain is empty after a failed load, so an append would take
``_GENESIS_HASH`` as its ``previous_hash`` and root a second chain behind
the untrusted tail. What the refusal did not come with was a way back, so
a workspace whose ledger forked once could take no governed write ever
again.

This module is that way back, and it is deliberately not a repair.
``EvidenceChain._freeze_and_raise`` sets the constraint the whole design
obeys — *"repairing the history by rewriting hashes is never this code's
decision"* — and its message names the remedy: archive the stored chain
and re-anchor as a deliberate operation. So:

1. **Survey.** Read the damaged file end to end (the loader stops at the
   first break; a census must not), verify every record's self-hash, and
   classify every linkage break.
2. **Archive.** Copy the file byte-for-byte to a timestamped sibling,
   prove the copy is faithful by digest, and drop the write bits. Not one
   byte of the damaged history is edited, dropped or reordered.
3. **Anchor.** Mint one record whose ``previous_hash`` is
   ``_GENESIS_HASH``, whose ``payload_hash`` **is** the archive's sha256,
   and whose metadata carries the record count, the head hash and the
   break census. Write it as the whole new store.

The break therefore never disappears. The archived file still fails to
verify — that is the point of keeping it — and the digest that pins it is
covered by the anchor's own evidence hash.

**What that does and does not buy.** An edit to the archive is detected by
:func:`verify_archives`, which re-hashes the file the anchor names and
compares both the digest and the byte size; ``mm chain verify-archive`` is
that check as a gate. Say tamper-EVIDENT against a partial edit, never
tamper-proof: this chain is unkeyed SHA-256 with no signature and no external
witness, so an actor who rewrites the whole store — anchor included — still
produces something that verifies clean. Recovery lowers the cost of that
attack, because forging one anchor now stands in for forging N records. The
defence against it is a signature or an off-box witness, and neither exists
here yet.

**Explicit only.** Nothing here is reachable from a failed load, from
``create()``, or from any read path. A ledger that heals itself when
somebody opens it is not tamper-evident, so recovery is an operator
action: :func:`recover_chain` refuses without ``confirm=True``, and
``mm chain recover`` refuses without ``--confirm``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import stat
from datetime import datetime, timezone
from typing import Mapping

from .evidence_objects import (
    _GENESIS_HASH,
    EVIDENCE_SCHEMA_VERSION,
    EvidenceAction,
    EvidenceChain,
    EvidenceObject,
)
from .mind_filelock import FileLock
from .observability import get_logger, metrics

_log = get_logger("evidence_recovery")

# ---------------------------------------------------------------------------
# Break taxonomy
# ---------------------------------------------------------------------------

#: A record claims the genesis hash as its parent while records precede it.
#: The signature of a writer that loaded nothing and started over — the
#: 3.8.x append path, which took ``_GENESIS_HASH`` whenever its in-memory
#: chain was empty and had no refusal to stop it.
BREAK_GENESIS_RESTART = "genesis_restart"

#: A record links to a hash that really is in the file, but earlier than
#: the record before it. Two writers each believed they owned the tail, so
#: the file holds a branch: two records naming one parent.
BREAK_FORK_FROM_STALE_HEAD = "fork_from_stale_head"

#: A record links to a hash that appears nowhere at or before it. Its
#: parent was never in this file — a foreign record, or one whose parent
#: was removed.
BREAK_UNKNOWN_PARENT = "unknown_parent"

#: The first record does not start at the genesis hash.
BREAK_NON_GENESIS_ROOT = "non_genesis_root"

#: A record's own ``evidence_hash`` does not match its fields under either
#: hashing scheme. Tampering, not forking.
BREAK_SELF_HASH_MISMATCH = "self_hash_mismatch"

#: A line that is not a readable evidence record at all.
BREAK_UNREADABLE_RECORD = "unreadable_record"

#: A record verifying only under the legacy v1 scheme after a v3 record has
#: been seen — the downgrade ``EvidenceChain.verify_chain`` rejects.
BREAK_SCHEME_DOWNGRADE = "scheme_downgrade"

#: Every kind, in report order. A census always carries all of them, so a
#: reader never has to distinguish "zero" from "this build had no name for
#: it".
BREAK_KINDS: tuple[str, ...] = (
    BREAK_GENESIS_RESTART,
    BREAK_FORK_FROM_STALE_HEAD,
    BREAK_UNKNOWN_PARENT,
    BREAK_NON_GENESIS_ROOT,
    BREAK_SELF_HASH_MISMATCH,
    BREAK_UNREADABLE_RECORD,
    BREAK_SCHEME_DOWNGRADE,
)

# ---------------------------------------------------------------------------
# Anchor vocabulary
# ---------------------------------------------------------------------------

#: The verb this operation records, carried in ``metadata`` rather than as
#: an :class:`~mind_mem.evidence_objects.EvidenceAction` member.
#:
#: Inventing a member would break forward compatibility exactly the way
#: ``lifecycle_evidence`` documents: ``EvidenceObject.from_dict`` in a
#: release older than this one performs a strict ``EvidenceAction(value)``
#: lookup, so a new verb makes that reader fail to load — and a chain
#: nobody older can read is a poor answer to a chain nobody can append to.
#: ``VERIFY`` is the honest existing word: the anchor attests to a digest.
RECOVERY_VERB = "REANCHOR"

#: ``metadata`` key carrying :data:`RECOVERY_VERB`.
RECOVERY_VERB_KEY = "recovery_verb"

#: Hashed anchor fields binding the retained companion hash-chain prefix.
COMPANION_PRESENT_KEY = "companion_hash_chain_present"
COMPANION_ENTRIES_KEY = "companion_hash_chain_entries"
COMPANION_TAIL_KEY = "companion_hash_chain_tail"

#: The :class:`~mind_mem.evidence_objects.EvidenceAction` the anchor is
#: written under. See :data:`RECOVERY_VERB`.
RECOVERY_ACTION = EvidenceAction.VERIFY

#: Infix separating the store's name from the archive's timestamp. Chosen
#: so the archive matches ``corpus_registry.LEDGER_PATTERNS``'
#: ``memory/evidence_chain.jsonl.*`` — an archived ledger is still a ledger
#: of record, and a snapshot must refuse to carry it off.
ARCHIVE_INFIX = ".damaged-"

#: Seconds the recovery waits for the store's cross-process append lock.
#: Matches the append path's budget: recovery is rarer than an append but
#: no more entitled to skip the queue.
_RECOVERY_LOCK_TIMEOUT_SECONDS = 30.0

#: Bytes moved per read/write step when copying the damaged file.
_COPY_CHUNK_BYTES = 1 << 20


class ChainRecoveryRefused(ValueError):
    """The recovery declined to act, and nothing on disk was changed.

    Raised for every refusal: a missing confirmation, an absent or empty
    store, a chain that verifies clean, an archive path already taken, and
    a store that moved under an unlocked writer mid-operation. All of them
    mean the same thing to a caller — no archive was kept, no store was
    replaced — which is why they share a type rather than making a caller
    enumerate outcomes to learn that nothing happened.
    """


# ---------------------------------------------------------------------------
# Survey
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ChainBreak:
    """One place the stored chain stops being a chain."""

    line: int
    """1-based line number in the store file."""

    kind: str
    """One of :data:`BREAK_KINDS`."""

    expected_previous: str
    """The hash this record's ``previous_hash`` had to be."""

    found_previous: str
    """The hash it actually carried."""

    evidence_id: str
    """The offending record's id, or ``""`` when it could not be read."""

    def to_dict(self) -> dict:
        """Serialise to a JSON-compatible dict."""
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class DamageSurvey:
    """What a full read of a stored chain found — a report, not a repair.

    Deliberately not built on ``EvidenceChain.verify_chain``: that reads
    the *loaded* entries, and a forked store loads zero of them. The
    survey reads the file.
    """

    path: str
    """The file surveyed."""

    lines: int
    """Non-blank lines in the file."""

    records: int
    """Lines that parsed as evidence records."""

    breaks: tuple[ChainBreak, ...]
    """Every break found, in line order."""

    census: Mapping[str, int]
    """Break count per kind, carrying every key in :data:`BREAK_KINDS`."""

    head_hash: str | None
    """``evidence_hash`` of the last readable record, or None if there is none."""

    byte_size: int
    """Size of the file when it was surveyed."""

    sha256: str
    """Digest of the file's bytes when it was surveyed."""

    @property
    def is_damaged(self) -> bool:
        """True when the stored chain does not verify as one unbroken chain."""
        return bool(self.breaks)

    @property
    def first_break_line(self) -> int | None:
        """Line of the earliest break, or None when there is none."""
        return self.breaks[0].line if self.breaks else None

    @property
    def last_break_line(self) -> int | None:
        """Line of the latest break, or None when there is none."""
        return self.breaks[-1].line if self.breaks else None

    def to_dict(self) -> dict:
        """Serialise to a JSON-compatible dict, breaks included."""
        return {
            "path": self.path,
            "lines": self.lines,
            "records": self.records,
            "byte_size": self.byte_size,
            "sha256": self.sha256,
            "head_hash": self.head_hash,
            "is_damaged": self.is_damaged,
            "break_count": len(self.breaks),
            "census": dict(self.census),
            "first_break_line": self.first_break_line,
            "last_break_line": self.last_break_line,
            "breaks": [b.to_dict() for b in self.breaks],
        }


def _sha256_and_size(path: str) -> tuple[str, int]:
    """Digest and byte length of *path*, read in chunks."""
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_COPY_CHUNK_BYTES), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _classify(found_previous: str, seen_hashes: set[str], line: int) -> str:
    """Name the break a record with ``previous_hash`` *found_previous* makes.

    *seen_hashes* holds every ``evidence_hash`` read strictly before this
    line, so "points backwards into this file" and "points nowhere in this
    file" are distinguishable — the difference between a writer that raced
    another writer and a record that never belonged here.
    """
    if line == 1:
        return BREAK_NON_GENESIS_ROOT
    if found_previous == _GENESIS_HASH:
        return BREAK_GENESIS_RESTART
    if found_previous in seen_hashes:
        return BREAK_FORK_FROM_STALE_HEAD
    return BREAK_UNKNOWN_PARENT


def survey_chain_file(path: str) -> DamageSurvey:
    """Read *path* end to end and report every break, without changing it.

    Purely diagnostic: opens the file read-only, writes nothing, creates
    nothing, and takes no lock — so a verifier, a CLI report and a
    recovery can all call it. :func:`recover_chain` re-runs it under the
    append lock before acting, because an unlocked survey can only ever
    describe the file as it was at the moment it was read.

    Unlike :meth:`~mind_mem.evidence_objects.EvidenceChain._load_from_file`
    this does **not** stop at the first bad record. Stopping is right for a
    loader — a verified prefix must never pass for the whole history — and
    wrong for a census, which exists to say how much damage there is.

    Args:
        path: JSONL evidence store to survey.

    Returns:
        A :class:`DamageSurvey`. An absent file surveys as zero lines,
        zero records and no breaks; callers that need the distinction
        check ``os.path.isfile`` themselves.

    Raises:
        OSError: If the file exists but cannot be read.
    """
    if not os.path.isfile(path):
        return DamageSurvey(
            path=path,
            lines=0,
            records=0,
            breaks=(),
            census={kind: 0 for kind in BREAK_KINDS},
            head_hash=None,
            byte_size=0,
            sha256=hashlib.sha256(b"").hexdigest(),
        )

    sha256, byte_size = _sha256_and_size(path)
    verifier = EvidenceChain()
    line_cap = EvidenceChain._MAX_LOAD_LINE_BYTES

    breaks: list[ChainBreak] = []
    census = {kind: 0 for kind in BREAK_KINDS}
    seen_hashes: set[str] = set()
    lines = 0
    records = 0
    head_hash: str | None = None
    seen_v3 = False
    # The hash the next record must carry. "" is the unmatchable sentinel a
    # record leaves behind when it could not be read: whatever follows it
    # cannot link to a hash nobody knows, and saying so is more honest than
    # silently re-basing on the last good record.
    expected_previous = _GENESIS_HASH

    def _record_break(line_no: int, kind: str, expected: str, found: str, evidence_id: str) -> None:
        breaks.append(
            ChainBreak(
                line=line_no,
                kind=kind,
                expected_previous=expected,
                found_previous=found,
                evidence_id=evidence_id,
            )
        )
        census[kind] += 1

    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line_no, raw in enumerate(handle, 1):
            stripped = raw.strip()
            if not stripped:
                continue
            lines += 1
            if len(raw.encode("utf-8")) > line_cap:
                _record_break(line_no, BREAK_UNREADABLE_RECORD, expected_previous, "", "")
                expected_previous = ""
                continue
            try:
                ev = EvidenceObject.from_dict(json.loads(stripped))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                _record_break(line_no, BREAK_UNREADABLE_RECORD, expected_previous, "", "")
                expected_previous = ""
                continue

            records += 1
            scheme = verifier._verify_scheme(ev)
            if scheme is None:
                _record_break(line_no, BREAK_SELF_HASH_MISMATCH, expected_previous, ev.previous_hash, ev.evidence_id)
            elif scheme == "v3":
                seen_v3 = True
            elif scheme == "v1" and seen_v3:
                _record_break(line_no, BREAK_SCHEME_DOWNGRADE, expected_previous, ev.previous_hash, ev.evidence_id)

            if ev.previous_hash != expected_previous:
                _record_break(
                    line_no,
                    _classify(ev.previous_hash, seen_hashes, line_no),
                    expected_previous,
                    ev.previous_hash,
                    ev.evidence_id,
                )

            seen_hashes.add(ev.evidence_hash)
            expected_previous = ev.evidence_hash
            head_hash = ev.evidence_hash

    return DamageSurvey(
        path=path,
        lines=lines,
        records=records,
        breaks=tuple(breaks),
        census=census,
        head_hash=head_hash,
        byte_size=byte_size,
        sha256=sha256,
    )


# ---------------------------------------------------------------------------
# Recovery
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class RecoveryResult:
    """What one completed re-anchor did."""

    store_path: str
    """The store that now holds the new segment."""

    archive_path: str
    """Where the damaged history was archived, unmodified."""

    archive_sha256: str
    """Digest of the archived bytes, as read back off disk after the copy."""

    archive_bytes: int
    """Size of the archive."""

    archived_records: int
    """Records the archive holds."""

    breaks: int
    """Breaks the archive holds."""

    census: Mapping[str, int]
    """Break count per kind in the archive."""

    anchor: EvidenceObject
    """The new segment's first and, at this instant, only record."""

    def to_dict(self) -> dict:
        """Serialise to a JSON-compatible dict."""
        return {
            "store_path": self.store_path,
            "archive_path": self.archive_path,
            "archive_sha256": self.archive_sha256,
            "archive_bytes": self.archive_bytes,
            "archived_records": self.archived_records,
            "breaks": self.breaks,
            "census": dict(self.census),
            "anchor": self.anchor.to_dict(),
        }


def _archive_stamp() -> str:
    """UTC stamp for an archive filename, second resolution.

    Its own function so a test can freeze it and prove that a second
    recovery landing on a taken name is refused rather than allowed to
    overwrite archived history.
    """
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def archive_path_for(store_path: str, stamp: str | None = None) -> str:
    """Where a recovery of *store_path* would archive the damaged file."""
    return f"{store_path}{ARCHIVE_INFIX}{stamp or _archive_stamp()}"


def _copy_file(source: str, destination: str) -> None:
    """Copy *source* to *destination* byte-for-byte and fsync the result.

    ``open(..., "xb")`` rather than ``"wb"``: the caller has already
    established the destination is free, and this closes the window
    between that check and the write. Losing the race is an error, never
    an overwrite of an archive.
    """
    with open(source, "rb") as src, open(destination, "xb") as dst:
        for chunk in iter(lambda: src.read(_COPY_CHUNK_BYTES), b""):
            dst.write(chunk)
        dst.flush()
        os.fsync(dst.fileno())


def _fsync_directory(path: str) -> None:
    """Flush *path*'s directory entry, so a rename survives a crash.

    Best effort: Windows cannot open a directory for the purpose, and a
    filesystem is free to refuse. The archive and the store are both
    fsync'd as files first, so the worst a failure here costs is the
    ordering guarantee between them.
    """
    directory = os.path.dirname(os.path.abspath(path))
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:  # pragma: no cover - platform dependent
        return
    try:
        os.fsync(fd)
    except OSError:  # pragma: no cover - platform dependent
        pass
    finally:
        os.close(fd)


def _companion_baseline(store_path: str) -> tuple[bool, int, str]:
    """Read the companion hash-chain count and tail without creating it."""
    from .hash_chain_v2 import GENESIS_HASH, HashChainV2

    db_path = os.path.join(os.path.dirname(os.path.abspath(store_path)), "hash_chain_v2.db")
    if not os.path.isfile(db_path):
        return False, 0, GENESIS_HASH
    try:
        chain = HashChainV2.open_readonly(db_path)
        ok, broken_at = chain.verify_chain()
        if not ok:
            raise ChainRecoveryRefused(f"refusing to bind companion hash chain {db_path!r}: its own chain breaks at entry {broken_at}")
        entries = chain.length
        latest = chain.get_latest(1) if entries else []
    except ChainRecoveryRefused:
        raise
    except Exception as exc:
        raise ChainRecoveryRefused(f"refusing to bind unreadable companion hash chain {db_path!r}: {exc}") from exc
    return True, entries, latest[-1].entry_hash if latest else GENESIS_HASH


def _anchor_metadata(
    survey: DamageSurvey,
    archive_path: str,
    archive_sha256: str,
    reason: str,
    companion: tuple[bool, int, str],
) -> dict:
    """The claim the anchor makes about the history it seals.

    Every value here is covered by the record's ``evidence_hash`` — the v3
    preimage hashes ``metadata`` as a sorted-key JSON slot — so the census
    and the digest are as tamper-evident as the record itself.
    """
    return {
        "evidence_schema": EVIDENCE_SCHEMA_VERSION,
        RECOVERY_VERB_KEY: RECOVERY_VERB,
        "continues_predecessor_chain": False,
        "predecessor_trust_restored": False,
        COMPANION_PRESENT_KEY: companion[0],
        COMPANION_ENTRIES_KEY: companion[1],
        COMPANION_TAIL_KEY: companion[2],
        "archived_chain": os.path.basename(archive_path),
        "archived_sha256": archive_sha256,
        "archived_bytes": survey.byte_size,
        "archived_records": survey.records,
        "archived_lines": survey.lines,
        "archived_head_hash": survey.head_hash or "",
        "archived_break_count": len(survey.breaks),
        "archived_break_census": {kind: count for kind, count in survey.census.items() if count},
        "archived_first_break_line": survey.first_break_line,
        "archived_last_break_line": survey.last_break_line,
        "reason": reason,
    }


def _remove_uncommitted_archive(path: str, expected_sha256: str, expected_bytes: int) -> None:
    """Remove the exact partial archive made by this recovery attempt.

    Successful archives stay read-only.  An aborted attempt owns its partial
    archive, but Windows will not unlink that file until its read-only bit is
    cleared.  Re-hash before changing the mode and again before unlinking so a
    path replaced or modified during recovery is preserved rather than treated
    as this operation's disposable output.
    """

    def inspect() -> tuple[os.stat_result, str, int]:
        info = os.stat(path, follow_symlinks=False)
        if not stat.S_ISREG(info.st_mode) or os.path.islink(path):
            raise OSError(f"refusing to remove changed recovery archive {path!r}: path is not a regular file")
        digest, byte_size = _sha256_and_size(path)
        return info, digest, byte_size

    if ARCHIVE_INFIX not in os.path.basename(path):
        raise OSError(f"refusing to remove non-archive path {path!r}")

    before, digest, byte_size = inspect()
    if digest != expected_sha256 or byte_size != expected_bytes:
        raise OSError(
            f"refusing to remove changed recovery archive {path!r}: expected "
            f"sha256 {expected_sha256} and {expected_bytes} bytes, found {digest} and {byte_size} bytes"
        )

    original_mode = stat.S_IMODE(before.st_mode)
    made_writable = not original_mode & stat.S_IWRITE
    try:
        if made_writable:
            os.chmod(path, original_mode | stat.S_IWRITE)

        after, digest, byte_size = inspect()
        if (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino):
            raise OSError(f"refusing to remove changed recovery archive {path!r}: path identity changed")
        if digest != expected_sha256 or byte_size != expected_bytes:
            raise OSError(
                f"refusing to remove changed recovery archive {path!r}: expected "
                f"sha256 {expected_sha256} and {expected_bytes} bytes, found {digest} and {byte_size} bytes"
            )
        os.unlink(path)
    except Exception:
        if made_writable:
            try:
                current = os.stat(path, follow_symlinks=False)
                if stat.S_ISREG(current.st_mode) and (current.st_dev, current.st_ino) == (before.st_dev, before.st_ino):
                    os.chmod(path, original_mode)
            except OSError as restore_exc:
                raise OSError(f"could not restore read-only mode on retained recovery archive {path!r}") from restore_exc
        raise


def recover_chain(
    store_path: str,
    *,
    actor: str,
    reason: str = "",
    confirm: bool = False,
    expected_sha256: str | None = None,
) -> RecoveryResult:
    """Seal a damaged evidence store and re-anchor it into a new segment.

    Archives *store_path* byte-for-byte to a timestamped sibling, then
    replaces it with a single record — the anchor — that names the archive
    and its sha256 and links from ``_GENESIS_HASH``. Nothing in the
    archived history is edited, dropped or reordered, and the archive
    still fails to verify afterwards. That is the design: the break is
    made permanent and citable rather than made to go away.

    The whole operation runs under the store's cross-process append lock
    and re-surveys the file inside it, so a writer that appended between a
    caller's report and this call cannot have its record silently archived
    on the strength of a stale census. The store's digest is re-checked
    after the copy and again immediately before the replace.

    **Residual race, stated rather than papered over.** The release that
    causes this damage, 3.8.3, takes no lock, so on a workspace where one
    is still installed the two digest checks narrow the window but cannot
    close it: a record appended in the instant between the final digest and
    the rename is destroyed with no trace. Stop every writer on the
    workspace before recovering.

    Args:
        store_path: The damaged JSONL evidence store.
        actor: Who is performing the recovery; recorded in the anchor.
        reason: Free text recorded in the anchor's metadata.
        confirm: Must be ``True``. Without it nothing is read, copied or
            written — recovery is an operator decision, so the default
            has to be refusal rather than a convenience.
        expected_sha256: Optional reviewed digest of the damaged store.
            It is compared with the in-lock survey before any archive or
            pending segment is created.

    Returns:
        A :class:`RecoveryResult` describing the archive and the anchor.

    Raises:
        ChainRecoveryRefused: If *confirm* is not ``True``, if the store is
            absent or holds no record, if the stored chain verifies clean,
            if the archive path is already taken, or if the store changed
            mid-operation. Nothing on disk has changed in any of those
            cases — the last one unwinds its own partial archive.
        LockTimeout: If the store's append lock could not be taken.
        OSError: If the archive could not be written or verified.
    """
    if not confirm:
        raise ChainRecoveryRefused(
            f"refusing to re-anchor {store_path!r}: recovery seals a governance history and "
            "starts a new segment, so it is never implicit — pass confirm=True (mm chain "
            "recover --confirm) once the damage census has been read"
        )

    with FileLock(store_path, timeout=_RECOVERY_LOCK_TIMEOUT_SECONDS):
        if not os.path.isfile(store_path):
            raise ChainRecoveryRefused(f"refusing to re-anchor {store_path!r}: there is no stored chain at that path")

        survey = survey_chain_file(store_path)
        if expected_sha256 is not None:
            expected = expected_sha256.strip().lower()
            if len(expected) != 64 or any(ch not in "0123456789abcdef" for ch in expected):
                raise ChainRecoveryRefused("expected_sha256 must be exactly 64 hexadecimal characters")
            if expected != survey.sha256:
                raise ChainRecoveryRefused(
                    f"refusing to re-anchor {store_path!r}: reviewed sha256 pin {expected} "
                    f"does not match the in-lock survey {survey.sha256}; nothing was written"
                )
        if survey.records == 0:
            raise ChainRecoveryRefused(f"refusing to re-anchor {store_path!r}: the store holds no evidence record to archive")
        if not survey.is_damaged:
            raise ChainRecoveryRefused(
                f"refusing to re-anchor {store_path!r}: the stored chain verifies clean over "
                f"{survey.records} record(s). Recovery seals a broken history; it is not a "
                "tidy-up, and running it on an intact chain would retire a verifiable ledger "
                "for nothing"
            )

        companion = _companion_baseline(store_path)

        pending = f"{store_path}.reanchor-pending"
        if os.path.lexists(pending):
            raise ChainRecoveryRefused(
                f"refusing to re-anchor {store_path!r}: pending path {pending!r} already exists; "
                "its contents belong to a previous operation and were not changed"
            )

        archive_path = archive_path_for(store_path)
        if os.path.exists(archive_path):
            raise ChainRecoveryRefused(
                f"refusing to re-anchor {store_path!r}: {archive_path!r} already exists and "
                "would have to be overwritten — that file is archived governance history"
            )

        _copy_file(store_path, archive_path)

        # Prove the copy is faithful, and that the source did not move under
        # us while it was being read. Both digests are taken off disk after
        # the copy, so neither is an assumption about what the copy did.
        archive_sha256, archive_bytes = _sha256_and_size(archive_path)
        source_sha256, _ = _sha256_and_size(store_path)
        if archive_sha256 != survey.sha256 or source_sha256 != survey.sha256:
            _remove_uncommitted_archive(archive_path, archive_sha256, archive_bytes)
            raise OSError(
                f"archive of {store_path!r} is not byte-identical to the surveyed file "
                f"(surveyed {survey.sha256}, archive {archive_sha256}, source now "
                f"{source_sha256}); the archive has been removed and the store is untouched"
            )

        _fsync_directory(archive_path)
        # An archive is not a place anything appends to again. Dropping the
        # write bits will not stop a determined root, and is not meant to:
        # it stops the ordinary accident of a tool opening the nearest
        # matching filename for append.
        try:
            os.chmod(archive_path, 0o444)
        except OSError:  # pragma: no cover - platform dependent
            _log.info("evidence_archive_chmod_skipped", path=archive_path)

        anchor = EvidenceChain()._forge(
            previous_hash=_GENESIS_HASH,
            action=RECOVERY_ACTION,
            actor=actor,
            target_block_id="",
            target_file=os.path.basename(archive_path),
            # The payload being attested to IS the archived history, so its
            # digest is the payload hash — not a value carried alongside one.
            payload_hash=archive_sha256,
            metadata=_anchor_metadata(survey, archive_path, archive_sha256, reason, companion),
            confidence=1.0,
        )

        # Write the new segment beside the store and rename it into place, so
        # the store is never observed half-replaced: a reader sees either the
        # damaged history or the anchored new one.
        line = json.dumps(anchor.to_dict(), separators=(",", ":")) + "\n"
        try:
            with open(pending, "x", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
        except FileExistsError as exc:
            # A path can appear after the precheck. Never follow a symlink or
            # truncate that winner; remove only the archive this operation made.
            _remove_uncommitted_archive(archive_path, archive_sha256, archive_bytes)
            raise ChainRecoveryRefused(
                f"refusing to re-anchor {store_path!r}: pending path {pending!r} appeared during recovery; "
                "the store and competing pending path were not changed"
            ) from exc

        # Last look before the store is replaced. The append lock serialises
        # every writer in this release, but the release that caused this
        # damage — 3.8.3 — has no lock at all, so on a workspace where one is
        # still installed the store can move while a locked process works.
        # A record that landed after the archive was taken is not in the
        # archive, and `os.replace` would destroy it with no trace. Checking
        # here narrows the window to the instant between this digest and the
        # rename; it cannot close it, which is why the operator is told to
        # stop the writers first.
        final_sha256, _ = _sha256_and_size(store_path)
        try:
            final_companion = _companion_baseline(store_path)
        except ChainRecoveryRefused:
            os.unlink(pending)
            _remove_uncommitted_archive(archive_path, archive_sha256, archive_bytes)
            raise
        if final_sha256 != survey.sha256 or final_companion != companion:
            # Unwind completely rather than leave a half-done recovery: the
            # archive is now a *prefix* of the store, not the history, and a
            # file named for archived history that is missing a record is
            # worse than no file. Both are this operation's own, created
            # moments ago, so removing them restores the disk exactly.
            os.unlink(pending)
            _remove_uncommitted_archive(archive_path, archive_sha256, archive_bytes)
            raise ChainRecoveryRefused(
                f"refusing to re-anchor {store_path!r}: the store changed while the recovery "
                f"ran (evidence sha256 {survey.sha256} -> {final_sha256}; companion "
                f"{companion!r} -> {final_companion!r}) — another writer appended, "
                "and replacing the store now would destroy a record the archive does not hold. "
                "Nothing was kept: the store is untouched and the partial archive was removed. "
                "Stop every writer on this workspace and re-run"
            )

        os.replace(pending, store_path)
        _fsync_directory(store_path)

    _log.warning(
        "evidence_chain_reanchored",
        store=store_path,
        archive=archive_path,
        archived_records=survey.records,
        breaks=len(survey.breaks),
        actor=actor,
    )
    metrics.inc("evidence_chain_reanchored")

    return RecoveryResult(
        store_path=store_path,
        archive_path=archive_path,
        archive_sha256=archive_sha256,
        archive_bytes=archive_bytes,
        archived_records=survey.records,
        breaks=len(survey.breaks),
        census={kind: count for kind, count in survey.census.items() if count},
        anchor=anchor,
    )


# ---------------------------------------------------------------------------
# Reading the anchor's claim BACK
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ArchiveCheck:
    """What became of one archive an anchor record attests.

    ``status`` is the load-bearing field: OK / MISSING / MISMATCH / UNREADABLE.
    """

    anchor_id: str
    archive_name: str
    archive_path: str
    status: str
    expected_sha256: str
    actual_sha256: str
    expected_bytes: int
    actual_bytes: int
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status == ARCHIVE_OK


#: The archive is present and hashes to exactly what its anchor recorded.
ARCHIVE_OK = "OK"
#: The anchor names an archive that is not on disk beside the store.
ARCHIVE_MISSING = "MISSING"
#: The archive is present and does NOT hash to what its anchor recorded.
ARCHIVE_MISMATCH = "MISMATCH"
#: The archive is present and could not be read at all.
ARCHIVE_UNREADABLE = "UNREADABLE"


def verify_archives(store_path: str) -> tuple[ArchiveCheck, ...]:
    """Re-hash every archive the store's anchors attest, and report each.

    Recording a digest that nothing ever reads back is not tamper-evidence, it
    is a note. Before this existed, deleting an archive left ``verify_chain``
    returning ``(True, [])`` and a survey reporting "intact" — the anchor named
    a file that was gone and nothing noticed. This is the leg that closes that:
    it resolves each ``archived_chain`` beside the store, re-hashes the bytes,
    and compares against the ``archived_sha256`` and ``archived_bytes`` the
    anchor committed to.

    Both halves are checked because they fail differently: a size change is a
    truncation or an append, a digest change with the same size is an edit in
    place.

    Returns one :class:`ArchiveCheck` per anchor, in file order. An empty tuple
    means the store has no anchors — which is NOT a pass, and callers must not
    render it as one.
    """
    checks: list[ArchiveCheck] = []
    store = os.path.abspath(store_path)
    directory = os.path.dirname(store)
    try:
        with open(store, "r", encoding="utf-8") as handle:
            lines = handle.readlines()
    except FileNotFoundError:
        return ()

    for raw in lines:
        stripped = raw.strip()
        if not stripped:
            continue
        try:
            record = json.loads(stripped)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(record, Mapping):
            continue
        metadata = record.get("metadata")
        if not isinstance(metadata, dict):
            continue
        if metadata.get(RECOVERY_VERB_KEY) != RECOVERY_VERB:
            continue

        name = str(metadata.get("archived_chain") or "")
        expected_sha = str(metadata.get("archived_sha256") or "")
        raw_expected_bytes = metadata.get("archived_bytes")
        expected_bytes = raw_expected_bytes if isinstance(raw_expected_bytes, int) and not isinstance(raw_expected_bytes, bool) else -1
        anchor_id = str(record.get("evidence_id") or "")
        # Resolved beside the store by BASENAME only: the anchor records a
        # basename, and honouring a path from the record would let a rewritten
        # record redirect the check at a file of its choosing.
        path = os.path.join(directory, os.path.basename(name)) if name else ""

        valid_sha = len(expected_sha) == 64 and all(ch in "0123456789abcdef" for ch in expected_sha)
        if not valid_sha or expected_bytes < 0:
            checks.append(
                ArchiveCheck(
                    anchor_id=anchor_id,
                    archive_name=name,
                    archive_path=path,
                    status=ARCHIVE_MISMATCH,
                    expected_sha256=expected_sha,
                    actual_sha256="",
                    expected_bytes=expected_bytes,
                    actual_bytes=0,
                    detail="the recovery anchor carries an invalid archive digest or byte count",
                )
            )
            continue

        if not name or not os.path.isfile(path):
            checks.append(
                ArchiveCheck(
                    anchor_id=anchor_id,
                    archive_name=name,
                    archive_path=path,
                    status=ARCHIVE_MISSING,
                    expected_sha256=expected_sha,
                    actual_sha256="",
                    expected_bytes=expected_bytes,
                    actual_bytes=0,
                    detail="the anchor names an archive that is not beside the store",
                )
            )
            continue

        try:
            digest = hashlib.sha256()
            size = 0
            with open(path, "rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
                    size += len(chunk)
            actual_sha = digest.hexdigest()
        except OSError as exc:
            checks.append(
                ArchiveCheck(
                    anchor_id=anchor_id,
                    archive_name=name,
                    archive_path=path,
                    status=ARCHIVE_UNREADABLE,
                    expected_sha256=expected_sha,
                    actual_sha256="",
                    expected_bytes=expected_bytes,
                    actual_bytes=0,
                    detail=str(exc),
                )
            )
            continue

        if actual_sha == expected_sha and size == expected_bytes:
            status, detail = ARCHIVE_OK, ""
        elif size != expected_bytes:
            status = ARCHIVE_MISMATCH
            detail = f"size changed: {expected_bytes} -> {size} bytes"
        else:
            status = ARCHIVE_MISMATCH
            detail = "same size, different bytes: edited in place"

        checks.append(
            ArchiveCheck(
                anchor_id=anchor_id,
                archive_name=name,
                archive_path=path,
                status=status,
                expected_sha256=expected_sha,
                actual_sha256=actual_sha,
                expected_bytes=expected_bytes,
                actual_bytes=size,
                detail=detail,
            )
        )
    return tuple(checks)
