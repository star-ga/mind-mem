#!/usr/bin/env python3
"""mind-mem Backup & Restore CLI. Zero external deps.

Provides:
- Full workspace backup (tar.gz or JSON export)
- Selective restore with conflict detection
- Git-friendly export (structured JSON, one block per line)
- WAL (write-ahead log) for crash-safe writes

Usage:
    python3 -m mind_mem.backup_restore backup workspace/ --output backup.tar.gz
    python3 -m mind_mem.backup_restore export workspace/ --output export.jsonl
    python3 -m mind_mem.backup_restore restore workspace/ --input backup.tar.gz
    python3 -m mind_mem.backup_restore wal-replay workspace/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
from datetime import datetime

from .block_parser import parse_blocks, parse_file
from .corpus_registry import BACKUP_DIRS, is_ledger_path, iter_ledger_paths
from .enums import IngestTier
from .observability import get_logger, metrics

_log = get_logger("backup_restore")

# BACKUP_DIRS imported from corpus_registry

BACKUP_FILES = ["mind-mem.json", "mind-mem-acl.json"]

#: Where the append-only ledgers ride in the archive.
#:
#: Not their live path, and that is the whole point. ``BACKUP_DIRS``
#: includes ``memory/``, so before 5.0.2 an archive held
#: ``memory/hash_chain_v2.db`` under its real name and ``restore --force``
#: wrote it straight back over the live chain: measured evidence −1,
#: hash_chain −1, and a block written after the backup gone with no record
#: it had existed.
#:
#: Excluding the ledgers outright would have closed the rewind by throwing
#: away a real capability — a backup is how an audit chain survives a lost
#: machine. Relocating closes it instead: the live path is *absent from the
#: archive*, so no restore can write it however hard it tries, while the
#: bytes are still there for an operator to take out by hand::
#:
#:     tar -xzf backup.tar.gz ledger-archive/
#:
#: :func:`restore_workspace` never extracts this prefix; it names it in the
#: result so the operator knows the chain is in the archive and where.
LEDGER_ARCHIVE_PREFIX = "ledger-archive"

#: Verb the recorded restore writes. Classified by
#: ``governance_gate._ACTION_MAP`` onto the EXISTING ``ROLLBACK`` evidence
#: action, so no enum member is added and a 5.0.1 reader parses it.
RESTORE_VERB = "RESTORE"


def _exclude_ledgers(tarinfo: tarfile.TarInfo) -> tarfile.TarInfo | None:
    """``tar.add`` filter: drop any member whose name is a ledger path."""
    return None if is_ledger_path(tarinfo.name) else tarinfo


# ---------------------------------------------------------------------------
# WAL (Write-Ahead Log)
# ---------------------------------------------------------------------------


class WAL:
    """Write-ahead log for crash-safe Markdown mutations.

    Before modifying any Markdown file, write the intention to the WAL.
    On next startup, replay any incomplete WAL entries.

    WAL location: workspace/.mind-mem-wal/
    """

    def __init__(self, workspace: str) -> None:
        self.workspace = os.path.realpath(workspace)
        self.wal_dir = os.path.join(self.workspace, ".mind-mem-wal")
        os.makedirs(self.wal_dir, exist_ok=True)

    _counter = 0  # Monotonic counter to avoid timestamp collisions on Windows

    def begin(self, operation: str, target_path: str, content: str) -> str:
        """Write a WAL entry before performing the operation.

        Returns the WAL entry ID (used to commit/rollback).
        """
        resolved = os.path.realpath(target_path)
        ws_real = os.path.realpath(self.workspace)
        if not resolved.startswith(ws_real + os.sep) and resolved != ws_real:
            raise ValueError(f"WAL.begin: target_path escapes workspace: {target_path}")

        ts = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        WAL._counter += 1
        entry_id = f"wal-{ts}-{WAL._counter}"
        entry_path = os.path.join(self.wal_dir, f"{entry_id}.json")

        entry = {
            "id": entry_id,
            "operation": operation,
            "target": os.path.relpath(target_path, self.workspace),
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "pid": os.getpid(),
            "status": "pending",
        }

        # Save backup of target file if it exists
        if os.path.isfile(target_path):
            backup_path = os.path.join(self.wal_dir, f"{entry_id}.backup")
            shutil.copy2(target_path, backup_path)
            entry["backup"] = f"{entry_id}.backup"

        with open(entry_path, "w", encoding="utf-8") as f:
            json.dump(entry, f, indent=2)

        return entry_id

    def commit(self, entry_id: str) -> None:
        """Mark a WAL entry as committed (operation completed successfully)."""
        entry_path = os.path.join(self.wal_dir, f"{entry_id}.json")
        if os.path.isfile(entry_path):
            with open(entry_path, "r", encoding="utf-8") as f:
                entry = json.load(f)
            entry["status"] = "committed"
            with open(entry_path, "w", encoding="utf-8") as f:
                json.dump(entry, f, indent=2)
            # Clean up backup file
            backup = entry.get("backup")
            if backup:
                backup_path = os.path.join(self.wal_dir, backup)
                if os.path.isfile(backup_path):
                    os.unlink(backup_path)
            # Remove committed WAL entry
            os.unlink(entry_path)

    def rollback(self, entry_id: str) -> bool:
        """Rollback a pending WAL entry, restoring the backup."""
        entry_path = os.path.join(self.wal_dir, f"{entry_id}.json")
        if not os.path.isfile(entry_path):
            return False

        with open(entry_path, "r", encoding="utf-8") as f:
            entry = json.load(f)

        target = os.path.realpath(os.path.join(self.workspace, entry["target"]))
        ws_real = os.path.realpath(self.workspace)
        if not target.startswith(ws_real + os.sep):
            _log.error("wal_rollback_blocked", entry_id=entry_id, reason="target escapes workspace")
            os.unlink(entry_path)
            return False
        backup = entry.get("backup")

        if backup:
            backup_path = os.path.join(self.wal_dir, backup)
            if os.path.isfile(backup_path):
                shutil.copy2(backup_path, target)
                os.unlink(backup_path)
        elif os.path.isfile(target):
            # No backup means the file was new — remove it
            os.unlink(target)

        os.unlink(entry_path)
        _log.info("wal_rollback", entry_id=entry_id, target=entry["target"])
        return True

    def replay(self) -> int:
        """Replay any pending WAL entries (crash recovery).

        Pending entries indicate a crash during write. We rollback them
        to restore the pre-write state.

        Returns the number of entries that were **actually** rolled back.
        A ``rollback`` that restored nothing (missing entry file, target
        escaping the workspace) is not counted and is logged as a
        failure, so a caller cannot report "workspace restored" over a
        half-written file that was never touched.
        """
        replayed = 0
        failed = 0
        if not os.path.isdir(self.wal_dir):
            return 0

        for fname in sorted(os.listdir(self.wal_dir)):
            if not fname.endswith(".json"):
                continue
            entry_path = os.path.join(self.wal_dir, fname)
            try:
                with open(entry_path, "r", encoding="utf-8") as f:
                    entry = json.load(f)
            except (json.JSONDecodeError, OSError) as exc:
                # A WAL record truncated by the very crash this log exists
                # to recover from must not vanish silently — recovery for
                # that entry is impossible, so say so.
                failed += 1
                _log.error("wal_entry_unreadable", entry=fname, error=str(exc))
                continue

            if entry.get("status") == "pending":
                entry_id = entry.get("id", fname[:-5])
                if self.rollback(entry_id):
                    replayed += 1
                    _log.info("wal_replay_rollback", entry_id=entry_id)
                else:
                    failed += 1
                    _log.error("wal_replay_failed", entry_id=entry_id, entry=fname)

        if replayed or failed:
            _log.info("wal_replay_complete", entries=replayed, failed=failed)
        return replayed

    def pending_count(self) -> int:
        """Count pending WAL entries (readable ones only).

        Entries too damaged to parse are reported by
        :meth:`unreadable_count` instead — they are not "clean", they are
        unrecoverable, and lumping them in here would let a caller treat a
        corrupt log as an empty one.
        """
        return self._scan()[0]

    def unreadable_count(self) -> int:
        """Count WAL entries that cannot be parsed (truncated / corrupt)."""
        return self._scan()[1]

    def _scan(self) -> tuple[int, int]:
        """Return ``(pending, unreadable)`` counts for the WAL directory."""
        if not os.path.isdir(self.wal_dir):
            return 0, 0
        pending = 0
        unreadable = 0
        for fname in os.listdir(self.wal_dir):
            if not fname.endswith(".json"):
                continue
            try:
                with open(os.path.join(self.wal_dir, fname), "r", encoding="utf-8") as f:
                    entry = json.load(f)
            except (json.JSONDecodeError, OSError, UnicodeDecodeError):
                unreadable += 1
                continue
            if isinstance(entry, dict) and entry.get("status") == "pending":
                pending += 1
        return pending, unreadable


# ---------------------------------------------------------------------------
# Backup
# ---------------------------------------------------------------------------


def backup_workspace(workspace: str, output: str, *, allow_empty: bool = False) -> str:
    """Create a tar.gz backup of the workspace.

    Args:
        workspace: Path to workspace
        output: Output file path (.tar.gz)
        allow_empty: Permit an archive that captured nothing. Off by
            default: a workspace holding none of ``BACKUP_DIRS`` /
            ``BACKUP_FILES`` is almost always a mistyped path, and an
            empty archive is only discovered at restore time.

    Returns:
        Path to created backup file

    Raises:
        FileNotFoundError: *workspace* is not a directory.
        ValueError: nothing matched and ``allow_empty`` is False. The
            partial archive is removed so it cannot be mistaken for a
            real backup.
    """
    ws = os.path.abspath(workspace)
    output = os.path.abspath(output)

    if not os.path.isdir(ws):
        raise FileNotFoundError(f"workspace is not a directory: {ws}")

    archived: list[str] = []
    #: Ledgers ride under :data:`LEDGER_ARCHIVE_PREFIX`, and deliberately do
    #: NOT count towards ``archived``: a directory holding nothing but a
    #: hash chain is still a workspace this backup captured no corpus from,
    #: and letting a ledger satisfy the non-empty guard would turn the
    #: mistyped-path check into a no-op.
    ledgers = iter_ledger_paths(ws)
    with tarfile.open(output, "w:gz") as tar:
        for d in BACKUP_DIRS:
            path = os.path.join(ws, d)
            if os.path.isdir(path):
                tar.add(path, arcname=d, filter=_exclude_ledgers)
                archived.append(d)

        for f in BACKUP_FILES:
            path = os.path.join(ws, f)
            if os.path.isfile(path):
                tar.add(path, arcname=f)
                archived.append(f)

        for rel in ledgers:
            tar.add(os.path.join(ws, rel), arcname=f"{LEDGER_ARCHIVE_PREFIX}/{rel}")

    if not archived and not allow_empty:
        _log.error("backup_empty", workspace=ws, output=output)
        try:
            os.unlink(output)
        except OSError:  # nosec B110 — best effort; the raise below is the signal
            pass
        expected = ", ".join(list(BACKUP_DIRS) + BACKUP_FILES)
        raise ValueError(f"backup captured nothing from {ws} — no corpus directory or config file found (expected one of: {expected})")

    size = os.path.getsize(output)
    _log.info(
        "backup_created",
        output=output,
        size_bytes=size,
        members=len(archived),
        ledgers_relocated=len(ledgers),
        ledger_prefix=LEDGER_ARCHIVE_PREFIX,
    )
    metrics.inc("backups_created")
    return output


def export_jsonl(workspace: str, output: str) -> int:
    """Export all blocks as JSONL (one JSON object per line). Git-friendly.

    Args:
        workspace: Path to workspace
        output: Output file path (.jsonl)

    Returns:
        Number of blocks exported
    """
    ws = os.path.abspath(workspace)
    count = 0

    corpus_files = {
        "decisions": "decisions/DECISIONS.md",
        "tasks": "tasks/TASKS.md",
        "projects": "entities/projects.md",
        "people": "entities/people.md",
        "tools": "entities/tools.md",
        "incidents": "entities/incidents.md",
        "contradictions": "intelligence/CONTRADICTIONS.md",
        "drift": "intelligence/DRIFT.md",
        "signals": "intelligence/SIGNALS.md",
    }

    with open(output, "w", encoding="utf-8") as out:
        for label, rel_path in corpus_files.items():
            path = os.path.join(ws, rel_path)
            if not os.path.isfile(path):
                continue
            try:
                blocks = parse_file(path)
            except (OSError, UnicodeDecodeError, ValueError):
                continue
            for block in blocks:
                block["_source"] = label
                block["_file"] = rel_path
                out.write(json.dumps(block, default=str) + "\n")
                count += 1

    _log.info("export_complete", blocks=count, output=output)
    metrics.inc("blocks_exported", count)
    return count


def _is_safe_tar_member(member: tarfile.TarInfo, ws: str) -> bool:
    """Validate a tar member is safe to extract into workspace.

    Rejects: absolute paths, traversal via .., symlinks, hardlinks,
    device files, and any path that resolves outside the workspace.
    """
    # Reject absolute paths
    if os.path.isabs(member.name) or member.name.startswith("/"):
        return False
    # Reject .. components
    if ".." in member.name.split(os.sep) or ".." in member.name.split("/"):
        return False
    # Reject symlinks and hardlinks (can point outside workspace)
    if member.issym() or member.islnk():
        return False
    # Reject device files
    if member.isdev():
        return False
    # Final check: resolved path must be inside workspace
    dest = os.path.realpath(os.path.join(ws, member.name))
    ws_real = os.path.realpath(ws)
    try:
        if os.path.commonpath([ws_real, dest]) != ws_real:
            return False
    except ValueError:
        # Different drives on Windows
        return False
    return True


#: Members this restore parses to name the blocks it is putting back.
#: Bounded so a crafted archive cannot make the pre-scan read an arbitrary
#: amount into memory before a single byte has been written.
_MAX_PRESCAN_MEMBER_BYTES = 8 * 1024 * 1024


def _block_ids_in_archive(tar: tarfile.TarFile, ws: str) -> list[str]:
    """Block ids the archive will reinstate, read from the archive itself.

    Read *before* extraction, because the scope has to be open before the
    workspace changes and afterwards the archive's version and the live
    version are the same thing.
    """
    ids: list[str] = []
    seen: set[str] = set()
    for member in tar.getmembers():
        if not member.isfile() or not member.name.endswith(".md"):
            continue
        if member.name.startswith(LEDGER_ARCHIVE_PREFIX + "/") or is_ledger_path(member.name):
            continue
        if not _is_safe_tar_member(member, ws) or member.size > _MAX_PRESCAN_MEMBER_BYTES:
            continue
        handle = tar.extractfile(member)
        if handle is None:
            continue
        with handle:
            try:
                text = handle.read().decode("utf-8")
            except (UnicodeDecodeError, OSError):
                continue
        try:
            blocks = parse_blocks(text)
        except (ValueError, TypeError):
            continue
        for block in blocks:
            bid = str(block.get("_id", "") or "")
            if bid and bid not in seen:
                seen.add(bid)
                ids.append(bid)
    return ids


#: See ``apply_engine._MAX_IDS_IN_RECORD`` — same reasoning, same cap.
_MAX_IDS_IN_RECORD = 500


def _live_block_ids(ws: str) -> list[str]:
    """Block ids in the workspace a restore is about to write over."""
    ids: list[str] = []
    seen: set[str] = set()
    for d in BACKUP_DIRS:
        root_dir = os.path.join(ws, d)
        if not os.path.isdir(root_dir):
            continue
        for dirpath, _dirnames, filenames in os.walk(root_dir):
            for name in sorted(filenames):
                if not name.endswith(".md"):
                    continue
                try:
                    blocks = parse_file(os.path.join(dirpath, name))
                except (OSError, UnicodeDecodeError, ValueError):
                    continue
                for block in blocks:
                    bid = str(block.get("_id", "") or "")
                    if bid and bid not in seen:
                        seen.add(bid)
                        ids.append(bid)
    return ids


def _id_digest(ids: list[str]) -> str:
    """SHA-256 over the canonical sorted id list."""
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def restore_workspace(workspace: str, backup_path: str, force: bool = False) -> dict:
    """Restore a workspace from a tar.gz backup, inside a recorded scope.

    Two things this does that the pre-5.0.2 version did not.

    **It cannot rewind the audit chain.** A member whose destination is a
    ledger of record is refused, whatever the archive says. Archives written
    from 5.0.2 on hold no such member — the ledgers ride under
    :data:`LEDGER_ARCHIVE_PREFIX` — but archives already on disk do, and
    they are exactly the ones an operator reaches for in an emergency.
    Refusing on the reader side is what makes the older archives safe too.

    **It records itself.** The restore runs inside one
    ``admit_batch(action="RESTORE")`` scope naming the archive digest and
    the block ids it reinstates, so a whole-corpus overwrite is an event in
    the chain rather than an invisible one. The gate refusing means the
    restore does not run: an unrecordable restore is refused, not performed
    quietly.

    Args:
        workspace: Target workspace path
        backup_path: Path to .tar.gz backup
        force: Overwrite existing files without prompting

    Returns:
        Summary dict. ``refused_ledgers`` counts members refused because
        their destination was a ledger; ``ledger_archive`` counts the
        relocated copies left in the archive for the operator.

    Raises:
        GovernanceBypassError: the gate refused; nothing was extracted.
    """
    from .governance_gate import get_gate

    ws = os.path.abspath(workspace)
    backup_path = os.path.abspath(backup_path)
    restored = 0
    skipped = 0
    blocked = 0
    refused_ledgers = 0
    ledger_archive = 0
    conflicts: list[str] = []

    with open(backup_path, "rb") as fh:
        archive_digest = hashlib.sha256(fh.read()).hexdigest()

    with tarfile.open(backup_path, "r:gz") as prescan:
        reinstated = _block_ids_in_archive(prescan, ws)

    # A ``force`` restore overwrites the live corpus with the archive's
    # version, so anything live and absent from the archive dies here. Only
    # with ``force``: without it every conflicting member is skipped, so
    # nothing is withdrawn and claiming otherwise would be a false record.
    withdrawn = sorted(set(_live_block_ids(ws)) - set(reinstated)) if force else []
    block_ids = sorted(set(reinstated) | set(withdrawn))

    record = {
        "archive": os.path.basename(backup_path),
        "archive_sha256": archive_digest,
        "reinstated_block_ids": sorted(reinstated),
        "withdrawn_block_ids": withdrawn,
    }

    with (
        tarfile.open(backup_path, "r:gz") as tar,
        get_gate(ws).admit_batch(
            action=RESTORE_VERB,
            batch_id=f"restore:{os.path.basename(backup_path)}",
            block_ids=block_ids,
            content=json.dumps(record, sort_keys=True, default=str),
            tier=IngestTier.RESTAMP,
            actor="backup_restore",
            target_file=os.path.basename(backup_path),
            metadata={
                "door": "backup_restore.restore_workspace",
                "archive": os.path.basename(backup_path),
                "archive_sha256": archive_digest,
                "reinstated_count": len(reinstated),
                "reinstated_digest": _id_digest(reinstated),
                "withdrawn_count": len(withdrawn),
                "withdrawn_digest": _id_digest(withdrawn),
                "withdrawn_block_ids": withdrawn[:_MAX_IDS_IN_RECORD],
                "withdrawn_truncated": len(withdrawn) > _MAX_IDS_IN_RECORD,
                "force": bool(force),
            },
        ) as receipt,
    ):
        for member in tar.getmembers():
            # The ledgers ride under a prefix no restore extracts: they are
            # in the archive for the operator, not for this loop.
            if member.name == LEDGER_ARCHIVE_PREFIX or member.name.startswith(LEDGER_ARCHIVE_PREFIX + "/"):
                ledger_archive += 1
                continue
            # A pre-5.0.2 archive names the ledgers under their live path.
            # Writing one back IS the rewind this function exists to stop.
            if is_ledger_path(member.name):
                _log.warning("restore_ledger_refused", member=member.name, reason="a restore may not overwrite a ledger of record")
                refused_ledgers += 1
                continue
            # Security: validate every member before extraction
            if not _is_safe_tar_member(member, ws):
                _log.warning("tar_member_blocked", member=member.name, reason="path traversal or unsafe member type")
                metrics.inc("restore_workspace_blocked_members")
                blocked += 1
                continue

            member_path = os.path.join(ws, member.name)
            if os.path.exists(member_path) and not force:
                conflicts.append(member.name)
                skipped += 1
            else:
                # Extract by streaming content to a file we open ourselves,
                # rather than using tar.extract which follows symlinks.
                if member.isfile():
                    dest = os.path.join(ws, member.name)
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    src = tar.extractfile(member)
                    if src is not None:
                        with src, open(dest, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                elif member.isdir():
                    os.makedirs(os.path.join(ws, member.name), exist_ok=True)
                else:
                    # Skip anything else (fifos, etc.)
                    continue
                restored += 1

        admission = receipt.entry_id

    result: dict = {
        "restored": restored,
        "skipped": skipped,
        "blocked": blocked,
        "conflicts": conflicts,
        "refused_ledgers": refused_ledgers,
        "ledger_archive": ledger_archive,
        "admission": admission,
    }
    _log.info(
        "restore_complete",
        restored=restored,
        skipped=skipped,
        blocked=blocked,
        conflicts=len(conflicts),
        refused_ledgers=refused_ledgers,
        ledger_archive=ledger_archive,
        admission=admission,
    )
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="mind-mem Backup & Restore")
    sub = parser.add_subparsers(dest="command")

    # Backup
    bp = sub.add_parser("backup", help="Create workspace backup")
    bp.add_argument("workspace", help="Workspace path")
    bp.add_argument("--output", "-o", help="Output file (default: mind-mem-backup-<date>.tar.gz)")
    bp.add_argument(
        "--allow-empty",
        action="store_true",
        help="Write the archive even when the workspace holds nothing to back up (default: fail)",
    )

    # Export
    ep = sub.add_parser("export", help="Export blocks as JSONL")
    ep.add_argument("workspace", help="Workspace path")
    ep.add_argument("--output", "-o", help="Output file (default: mind-mem-export-<date>.jsonl)")

    # Restore
    rp = sub.add_parser("restore", help="Restore from backup")
    rp.add_argument("workspace", help="Target workspace path")
    rp.add_argument("--input", "-i", required=True, help="Backup file path")
    rp.add_argument("--force", action="store_true", help="Overwrite existing files")

    # WAL replay
    wp = sub.add_parser("wal-replay", help="Replay pending WAL entries (crash recovery)")
    wp.add_argument("workspace", help="Workspace path")

    args = parser.parse_args()

    if args.command == "backup":
        ws = os.path.abspath(args.workspace)
        output = args.output or f"mind-mem-backup-{datetime.now().strftime('%Y%m%d-%H%M%S')}.tar.gz"
        try:
            path = backup_workspace(ws, output, allow_empty=args.allow_empty)
        except (FileNotFoundError, ValueError) as exc:
            print(f"Backup failed: {exc}")
            raise SystemExit(1) from exc
        with tarfile.open(path, "r:gz") as tar:
            members = len(tar.getnames())
        print(f"Backup created: {path} ({os.path.getsize(path)} bytes, {members} entr{'y' if members == 1 else 'ies'})")

    elif args.command == "export":
        ws = os.path.abspath(args.workspace)
        output = args.output or f"mind-mem-export-{datetime.now().strftime('%Y%m%d-%H%M%S')}.jsonl"
        count = export_jsonl(ws, output)
        print(f"Exported {count} blocks → {output}")

    elif args.command == "restore":
        ws = os.path.abspath(args.workspace)
        result = restore_workspace(ws, args.input, args.force)
        print(f"Restored: {result['restored']} file(s)")
        if result["skipped"]:
            print(f"Skipped: {result['skipped']} (existing files)")
        if result["conflicts"]:
            print("Conflicts (use --force to overwrite):")
            for c in result["conflicts"][:10]:
                print(f"  {c}")

    elif args.command == "wal-replay":
        ws = os.path.abspath(args.workspace)
        wal = WAL(ws)
        pending, unreadable = wal._scan()
        if pending == 0 and unreadable == 0:
            print("No pending WAL entries. Workspace is clean.")
        else:
            replayed = wal.replay()
            if replayed == pending and unreadable == 0:
                print(f"Replayed {replayed} pending WAL entry(ies). Workspace restored to consistent state.")
            else:
                print(f"Replayed {replayed} of {pending} pending WAL entry(ies).")
                if unreadable:
                    print(f"WARNING: {unreadable} WAL entry(ies) are unreadable (truncated or corrupt) and could not be recovered.")
                if replayed < pending:
                    print(f"WARNING: {pending - replayed} pending entry(ies) could not be rolled back.")
                print(f"Workspace may be inconsistent — inspect {wal.wal_dir} before continuing.")
                raise SystemExit(1)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
