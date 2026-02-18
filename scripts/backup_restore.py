#!/usr/bin/env python3
"""mind-mem Backup & Restore CLI. Zero external deps.

Provides:
- Full workspace backup (tar.gz or JSON export)
- Selective restore with conflict detection
- Git-friendly export (structured JSON, one block per line)
- WAL (write-ahead log) for crash-safe writes

Usage:
    python3 scripts/backup_restore.py backup workspace/ --output backup.tar.gz
    python3 scripts/backup_restore.py export workspace/ --output export.jsonl
    python3 scripts/backup_restore.py restore workspace/ --input backup.tar.gz
    python3 scripts/backup_restore.py wal-replay workspace/
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tarfile
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from block_parser import parse_file
from observability import get_logger, metrics

_log = get_logger("backup_restore")

# Directories included in backup
BACKUP_DIRS = [
    "decisions", "tasks", "entities", "memory",
    "intelligence", "summaries", "shared", "agents",
]

BACKUP_FILES = ["mind-mem.json", "mind-mem-acl.json"]


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

        target = os.path.join(self.workspace, entry["target"])
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

        Returns number of entries replayed.
        """
        replayed = 0
        if not os.path.isdir(self.wal_dir):
            return 0

        for fname in sorted(os.listdir(self.wal_dir)):
            if not fname.endswith(".json"):
                continue
            entry_path = os.path.join(self.wal_dir, fname)
            try:
                with open(entry_path, "r", encoding="utf-8") as f:
                    entry = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue

            if entry.get("status") == "pending":
                entry_id = entry.get("id", fname[:-5])
                self.rollback(entry_id)
                replayed += 1
                _log.info("wal_replay_rollback", entry_id=entry_id)

        if replayed:
            _log.info("wal_replay_complete", entries=replayed)
        return replayed

    def pending_count(self) -> int:
        """Count pending WAL entries."""
        if not os.path.isdir(self.wal_dir):
            return 0
        count = 0
        for fname in os.listdir(self.wal_dir):
            if fname.endswith(".json"):
                try:
                    with open(os.path.join(self.wal_dir, fname), "r") as f:
                        entry = json.load(f)
                    if entry.get("status") == "pending":
                        count += 1
                except (json.JSONDecodeError, OSError):
                    pass
        return count


# ---------------------------------------------------------------------------
# Backup
# ---------------------------------------------------------------------------

def backup_workspace(workspace: str, output: str) -> str:
    """Create a tar.gz backup of the workspace.

    Args:
        workspace: Path to workspace
        output: Output file path (.tar.gz)

    Returns:
        Path to created backup file
    """
    ws = os.path.abspath(workspace)
    output = os.path.abspath(output)

    with tarfile.open(output, "w:gz") as tar:
        for d in BACKUP_DIRS:
            path = os.path.join(ws, d)
            if os.path.isdir(path):
                tar.add(path, arcname=d)

        for f in BACKUP_FILES:
            path = os.path.join(ws, f)
            if os.path.isfile(path):
                tar.add(path, arcname=f)

    size = os.path.getsize(output)
    _log.info("backup_created", output=output, size_bytes=size)
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
    if os.path.isabs(member.name):
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


def restore_workspace(workspace: str, backup_path: str, force: bool = False) -> dict:
    """Restore a workspace from a tar.gz backup.

    Args:
        workspace: Target workspace path
        backup_path: Path to .tar.gz backup
        force: Overwrite existing files without prompting

    Returns:
        Summary dict
    """
    ws = os.path.abspath(workspace)
    backup_path = os.path.abspath(backup_path)
    result = {"restored": 0, "skipped": 0, "blocked": 0, "conflicts": []}

    with tarfile.open(backup_path, "r:gz") as tar:
        for member in tar.getmembers():
            # Security: validate every member before extraction
            if not _is_safe_tar_member(member, ws):
                _log.warning("tar_member_blocked", member=member.name,
                             reason="path traversal or unsafe member type")
                metrics.inc("restore_workspace_blocked_members")
                result["blocked"] += 1
                continue

            member_path = os.path.join(ws, member.name)
            if os.path.exists(member_path) and not force:
                result["conflicts"].append(member.name)
                result["skipped"] += 1
            else:
                # Extract by streaming content to a file we open ourselves,
                # rather than using tar.extract which follows symlinks.
                if member.isfile():
                    dest = os.path.join(ws, member.name)
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    with tar.extractfile(member) as src, open(dest, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                elif member.isdir():
                    os.makedirs(os.path.join(ws, member.name), exist_ok=True)
                else:
                    # Skip anything else (fifos, etc.)
                    continue
                result["restored"] += 1

    _log.info("restore_complete", restored=result["restored"],
              skipped=result["skipped"], blocked=result["blocked"],
              conflicts=len(result["conflicts"]))
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
        path = backup_workspace(ws, output)
        print(f"Backup created: {path} ({os.path.getsize(path)} bytes)")

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
        pending = wal.pending_count()
        if pending == 0:
            print("No pending WAL entries. Workspace is clean.")
        else:
            replayed = wal.replay()
            print(f"Replayed {replayed} pending WAL entry(ies). Workspace restored to consistent state.")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
