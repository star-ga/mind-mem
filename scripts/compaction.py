#!/usr/bin/env python3
"""mind-mem Compaction & GC Engine. Zero external deps.

Archives completed/canceled blocks, removes expired snapshots,
and compacts append-only files to prevent workspace bloat.

Safety: Never deletes source of truth. Archived blocks are moved to
archive files, not deleted. Snapshots older than retention period are
removed (they can be recreated by restoring from git history).

Usage:
    python3 scripts/compaction.py [workspace_path]
    python3 scripts/compaction.py . --dry-run
    python3 scripts/compaction.py . --archive-days 90
    python3 scripts/compaction.py . --snapshot-days 30
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from block_parser import parse_file
from mind_filelock import FileLock
from observability import get_logger, metrics

_log = get_logger("compaction")


def _load_config(ws: str) -> dict:
    """Load compaction config from mind-mem.json."""
    config_path = os.path.join(ws, "mind-mem.json")
    try:
        with open(config_path) as f:
            cfg = json.load(f)
        return cfg.get("compaction", {})
    except (OSError, json.JSONDecodeError):
        return {}


def archive_completed_blocks(ws: str, days: int = 90, dry_run: bool = False) -> list[str]:
    """Move completed/canceled blocks older than `days` to archive files.

    Blocks are appended to {file}_ARCHIVE.md, then removed from the
    source file. This keeps source files small while preserving history.
    """
    archived = []
    cutoff = datetime.now() - timedelta(days=days)
    cutoff_str = cutoff.strftime("%Y-%m-%d")

    files_to_compact = {
        "tasks/TASKS.md": {"done", "canceled"},
        "decisions/DECISIONS.md": {"superseded", "revoked"},
    }

    for rel_path, archive_statuses in files_to_compact.items():
        path = os.path.join(ws, rel_path)
        if not os.path.isfile(path):
            continue

        blocks = parse_file(path)
        to_archive = []
        to_keep = []

        for block in blocks:
            status = block.get("Status", "")
            date = block.get("Date", "9999-99-99")
            if status in archive_statuses and date < cutoff_str:
                to_archive.append(block)
            else:
                to_keep.append(block)

        if not to_archive:
            continue

        if dry_run:
            for b in to_archive:
                archived.append(f"[dry-run] Would archive {b['_id']} ({b.get('Status')}) from {rel_path}")
            continue

        # Write archived blocks to archive file
        archive_rel = rel_path.replace(".md", "_ARCHIVE.md")
        archive_path = os.path.join(ws, archive_rel)

        with FileLock(path):
            # Re-read to avoid TOCTOU
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

            # Build archive content
            archive_lines = []
            for b in to_archive:
                # Extract the raw text for this block from the file
                block_text = _extract_block_text(content, b["_id"])
                if block_text:
                    archive_lines.append(block_text)

            if archive_lines:
                os.makedirs(os.path.dirname(archive_path), exist_ok=True)
                with open(archive_path, "a", encoding="utf-8") as f:
                    for text in archive_lines:
                        f.write(f"\n{text}\n---\n")

                # Remove archived blocks from source
                new_content = content
                for b in to_archive:
                    block_text = _extract_block_text(new_content, b["_id"])
                    if block_text:
                        new_content = new_content.replace(block_text, "")

                # Clean up excessive blank lines
                new_content = re.sub(r"\n{4,}", "\n\n\n", new_content)

                tmp_path = path + ".tmp"
                with open(tmp_path, "w", encoding="utf-8") as f:
                    f.write(new_content)
                os.replace(tmp_path, path)

                for b in to_archive:
                    archived.append(f"Archived {b['_id']} ({b.get('Status')}) -> {archive_rel}")

    return archived


def _extract_block_text(content: str, block_id: str) -> str | None:
    """Extract raw text of a block from file content."""
    pattern = re.compile(
        rf"^\[{re.escape(block_id)}\].*?(?=^\[[A-Z]+-|^---|\Z)",
        re.MULTILINE | re.DOTALL
    )
    match = pattern.search(content)
    if match:
        return match.group(0).strip()
    return None


def cleanup_snapshots(ws: str, days: int = 30, dry_run: bool = False) -> list[str]:
    """Remove snapshot directories older than `days`.

    Snapshots are in intelligence/applied/<timestamp>/. Each contains
    APPLY_RECEIPT.md and a full workspace copy. Old ones can be large.
    """
    cleaned = []
    applied_dir = os.path.join(ws, "intelligence", "applied")
    if not os.path.isdir(applied_dir):
        return cleaned

    cutoff = datetime.now() - timedelta(days=days)

    for entry in os.listdir(applied_dir):
        snap_path = os.path.join(applied_dir, entry)
        if not os.path.isdir(snap_path):
            continue

        # Parse timestamp from directory name (YYYYMMDD-HHMMSS)
        try:
            snap_dt = datetime.strptime(entry, "%Y%m%d-%H%M%S")
        except ValueError:
            continue

        if snap_dt < cutoff:
            if dry_run:
                cleaned.append(f"[dry-run] Would remove snapshot {entry}")
            else:
                shutil.rmtree(snap_path, ignore_errors=True)
                cleaned.append(f"Removed snapshot {entry}")

    return cleaned


def cleanup_daily_logs(ws: str, days: int = 180, dry_run: bool = False) -> list[str]:
    """Archive daily log files older than `days` into yearly archives."""
    cleaned = []
    memory_dir = os.path.join(ws, "memory")
    if not os.path.isdir(memory_dir):
        return cleaned

    cutoff = datetime.now() - timedelta(days=days)
    cutoff_str = cutoff.strftime("%Y-%m-%d")
    date_re = re.compile(r"^(\d{4}-\d{2}-\d{2})\.md$")

    for fname in sorted(os.listdir(memory_dir)):
        m = date_re.match(fname)
        if not m:
            continue
        date_str = m.group(1)
        if date_str >= cutoff_str:
            continue

        year = date_str[:4]
        archive_path = os.path.join(memory_dir, f"archive-{year}.md")
        log_path = os.path.join(memory_dir, fname)

        if dry_run:
            cleaned.append(f"[dry-run] Would archive {fname} -> archive-{year}.md")
            continue

        # Append to yearly archive
        with open(log_path, "r", encoding="utf-8") as f:
            content = f.read()

        with open(archive_path, "a", encoding="utf-8") as f:
            f.write(f"\n# {date_str}\n\n{content}\n---\n")

        os.remove(log_path)
        cleaned.append(f"Archived {fname} -> archive-{year}.md")

    return cleaned


def compact_signals(ws: str, days: int = 60, dry_run: bool = False) -> list[str]:
    """Remove processed signals older than `days` from SIGNALS.md.

    Only removes signals with Status: resolved or Status: rejected.
    Pending signals are never removed.
    """
    cleaned = []
    signals_path = os.path.join(ws, "intelligence", "SIGNALS.md")
    if not os.path.isfile(signals_path):
        return cleaned

    blocks = parse_file(signals_path)
    cutoff = datetime.now() - timedelta(days=days)
    cutoff_str = cutoff.strftime("%Y-%m-%d")

    removable = [
        b for b in blocks
        if b.get("Status") in ("resolved", "rejected")
        and b.get("Date", "9999-99-99") < cutoff_str
    ]

    if not removable:
        return cleaned

    if dry_run:
        for b in removable:
            cleaned.append(f"[dry-run] Would remove signal {b['_id']} ({b.get('Status')})")
        return cleaned

    with FileLock(signals_path):
        with open(signals_path, "r", encoding="utf-8") as f:
            content = f.read()

        for b in removable:
            block_text = _extract_block_text(content, b["_id"])
            if block_text:
                content = content.replace(block_text, "")
                cleaned.append(f"Removed signal {b['_id']} ({b.get('Status')})")

        content = re.sub(r"\n{4,}", "\n\n\n", content)
        with open(signals_path, "w", encoding="utf-8") as f:
            f.write(content)

    return cleaned


def main():
    parser = argparse.ArgumentParser(description="mind-mem Compaction & GC Engine")
    parser.add_argument("workspace", nargs="?", default=".")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    parser.add_argument("--archive-days", type=int, default=90,
                        help="Archive completed blocks older than N days (default: 90)")
    parser.add_argument("--snapshot-days", type=int, default=30,
                        help="Remove snapshots older than N days (default: 30)")
    parser.add_argument("--log-days", type=int, default=180,
                        help="Archive daily logs older than N days (default: 180)")
    parser.add_argument("--signal-days", type=int, default=60,
                        help="Remove resolved signals older than N days (default: 60)")
    args = parser.parse_args()

    ws = os.path.abspath(args.workspace)

    # Override from config
    cfg = _load_config(ws)
    archive_days = cfg.get("archive_days", args.archive_days)
    snapshot_days = cfg.get("snapshot_days", args.snapshot_days)
    log_days = cfg.get("log_days", args.log_days)
    signal_days = cfg.get("signal_days", args.signal_days)

    print(f"mind-mem compaction: {ws}")
    if args.dry_run:
        print("  (dry-run mode — no changes will be made)")
    print()

    all_actions = []

    # 1. Archive completed blocks
    actions = archive_completed_blocks(ws, archive_days, args.dry_run)
    all_actions.extend(actions)
    if actions:
        print(f"Block archival ({archive_days}d threshold):")
        for a in actions:
            print(f"  {a}")
    else:
        print(f"Block archival: nothing to archive (threshold: {archive_days}d)")

    # 2. Cleanup snapshots
    actions = cleanup_snapshots(ws, snapshot_days, args.dry_run)
    all_actions.extend(actions)
    if actions:
        print(f"\nSnapshot cleanup ({snapshot_days}d threshold):")
        for a in actions:
            print(f"  {a}")
    else:
        print(f"Snapshot cleanup: nothing to remove (threshold: {snapshot_days}d)")

    # 3. Archive daily logs
    actions = cleanup_daily_logs(ws, log_days, args.dry_run)
    all_actions.extend(actions)
    if actions:
        print(f"\nDaily log archival ({log_days}d threshold):")
        for a in actions:
            print(f"  {a}")
    else:
        print(f"Daily log archival: nothing to archive (threshold: {log_days}d)")

    # 4. Compact signals
    actions = compact_signals(ws, signal_days, args.dry_run)
    all_actions.extend(actions)
    if actions:
        print(f"\nSignal compaction ({signal_days}d threshold):")
        for a in actions:
            print(f"  {a}")
    else:
        print(f"Signal compaction: nothing to remove (threshold: {signal_days}d)")

    _log.info("compaction_complete", actions=len(all_actions), dry_run=args.dry_run)
    metrics.inc("compaction_actions", len(all_actions))
    print(f"\nTotal: {len(all_actions)} action(s)")


if __name__ == "__main__":
    main()
