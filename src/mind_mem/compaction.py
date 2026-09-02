#!/usr/bin/env python3
"""mind-mem Compaction & GC Engine. Zero external deps.

Archives completed/canceled blocks, removes expired snapshots,
and compacts append-only files to prevent workspace bloat.

Safety: Never deletes source of truth. Archived blocks are moved to
archive files, not deleted. Snapshots older than retention period are
removed (they can be recreated by restoring from git history).

**Governance.** Two of the sweeps below rewrite the corpus of record, and
both now do it under a governance scope opened before a byte moves:

``compact_signals``           removes blocks. One
                              :meth:`~mind_mem.governance_gate.GovernanceGate.admit_delete_batch`
                              scope per sweep — one authorisation record,
                              one removal record over the frozen id set.
``archive_completed_blocks``  moves blocks between files of record. One
                              :meth:`~mind_mem.governance_gate.GovernanceGate.admit_batch`
                              scope per run, on the carrying tier: nothing
                              dies, so nothing is recorded as a death. The
                              full argument is in that function's docstring.

The two remaining sweeps (:func:`cleanup_snapshots`,
:func:`cleanup_daily_logs`) touch no block of record — a snapshot
directory and a dated log file, neither parsed into blocks nor served by
recall — so neither opens a scope.

Usage:
    python3 -m mind_mem.compaction [workspace_path]
    python3 -m mind_mem.compaction . --dry-run
    python3 -m mind_mem.compaction . --archive-days 90
    python3 -m mind_mem.compaction . --snapshot-days 30
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from datetime import datetime, timedelta

from .block_parser import parse_file
from .enums import IngestTier, TaskStatus
from .mind_filelock import FileLock
from .observability import get_logger, metrics

_log = get_logger("compaction")

#: Rationale recorded for the signals one compaction sweep removes.
#:
#: The other three delete doors name themselves when the caller gave no
#: reason (:data:`mind_mem.http_transport.DEFAULT_DELETE_RATIONALE`,
#: :data:`mind_mem.mcp.tools.memory_ops.DEFAULT_DELETE_RATIONALE`). This
#: one has no caller to ask: a sweep is a scheduled retention policy, not
#: a decision about any particular signal. So the record carries the
#: door's name and the window that selected the set — never a sentence
#: invented here about why someone wanted this content gone.
SIGNAL_SWEEP_RATIONALE = "compaction-signal-sweep"

#: Evidence verb for the archive move.
#:
#: ``MIGRATE`` rather than a new member: ``governance_gate._ACTION_MAP``
#: is an allowlist and adding a verb is a wire change an older reader
#: cannot parse, while the shape here is the one ``MIGRATE`` already
#: names — already-governed blocks moved between files of record, as
#: ``mm migrate-store`` moves them between backends. Emphatically **not**
#: ``DELETE``: see :func:`archive_completed_blocks` for why recording a
#: move as a death would be the worse lie.
ARCHIVE_VERB = "MIGRATE"


def _sweep_batch_id(prefix: str, block_ids: list[str]) -> str:
    """A stable subject id for one compaction decision.

    Derived from the frozen id set, not from a clock, for the reason
    ``http_transport._clear_batch_id`` gives: the evidence record already
    carries its own timestamp, and a subject id that changes between two
    identical runs tells an auditor nothing while making the record
    impossible to reproduce.
    """
    digest = hashlib.sha256("\n".join(block_ids).encode("utf-8")).hexdigest()
    return f"{prefix}-{digest[:16]}"


def _load_config(ws: str) -> dict:
    """Load compaction config from mind-mem.json."""
    config_path = os.path.join(ws, "mind-mem.json")
    try:
        with open(config_path, encoding="utf-8") as f:
            cfg = json.load(f)
        return dict(cfg.get("compaction", {}))
    except (OSError, json.JSONDecodeError):
        return {}


def archive_completed_blocks(ws: str, days: int = 90, dry_run: bool = False) -> list[str]:
    """Move completed/canceled blocks older than `days` to archive files.

    Blocks are appended to {file}_ARCHIVE.md, then removed from the
    source file. This keeps source files small while preserving history.

    **A relocation is recorded, and it is recorded as a move.** The
    thesis is that no content enters, leaves, or dies without a receipt,
    and this sweep rewrites the corpus of record. Measured on a workspace
    built by ``mind-mem-init``, before this change: a ``done`` task left
    ``tasks/TASKS.md``, appeared in ``tasks/TASKS_ARCHIVE.md``, and the
    evidence chain gained **zero** rows. What the same probe also
    measured is why this is not a death and must not be recorded as one:

    * ``BlockStore.get_by_id`` still resolves the block afterwards.
      ``_discover_files`` globs *every* ``.md`` in the corpus dirs, so
      the archive file is inside the store's read surface — the block
      stays readable, stays reachable by ``DELETE /memories/{id}`` and
      ``POST /clear``, and its eventual death still mints a receipt then.
    * It does leave the *recall* surface: ``_recall_constants.CORPUS_FILES``
      has no ``*_ARCHIVE.md`` row, so the indexer, the BM25 corpus walk
      and ``storage.iter_active_blocks`` (scan, drift, export, reindex)
      all stop seeing it.

    So the honest reading is "still governed, no longer served", and both
    halves have to survive into the record. A ``DELETE`` receipt would
    fail the first half: it writes a removal row carrying the content
    hash of something that is still in the workspace, which makes
    "removed" mean two different things and leaves a future auditor
    unable to tell an archive from a destruction. Silence fails the
    second: a block silently stops being recallable with nothing in the
    chain naming who moved it or when, and the splice that moves it is
    the same destructive primitive the delete path uses.

    One :meth:`~mind_mem.governance_gate.GovernanceGate.admit_batch`
    scope per run, on :attr:`~mind_mem.enums.IngestTier.RESTAMP` — the
    carrying tier, which mints no status and cannot raise one. That is
    the shape ``pipeline_hash.reextract_dirty_blocks`` already uses for a
    bulk rewrite of already-admitted blocks, and an archive is exactly
    that plus a change of file. The record names every id moved; a
    ``dry_run`` moves nothing and opens no scope.
    """
    archived: list[str] = []
    cutoff = datetime.now() - timedelta(days=days)
    cutoff_str = cutoff.strftime("%Y-%m-%d")

    files_to_compact = {
        "tasks/TASKS.md": {s.value for s in TaskStatus.closed()},
        "decisions/DECISIONS.md": {"superseded", "revoked"},
    }

    # Plan first, act second. The id set the scope covers is frozen
    # before a byte moves, exactly as ``POST /clear`` freezes the set it
    # wipes: a block written into TASKS.md while DECISIONS.md is being
    # rewritten is outside this run's record, so this run must not move
    # it. Planning both files also makes the run ONE decision with one
    # record, rather than one record per file that happened to have work.
    plan: list[tuple[str, str, str, list[dict]]] = []
    for rel_path, archive_statuses in files_to_compact.items():
        path = os.path.join(ws, rel_path)
        if not os.path.isfile(path):
            continue
        to_archive = [
            block
            for block in parse_file(path)
            if block.get("Status", "") in archive_statuses and block.get("Date", "9999-99-99") < cutoff_str
        ]
        if to_archive:
            plan.append((rel_path, path, rel_path.replace(".md", "_ARCHIVE.md"), to_archive))

    if not plan:
        return archived

    if dry_run:
        for rel_path, _path, _archive_rel, to_archive in plan:
            for b in to_archive:
                archived.append(f"[dry-run] Would archive {b['_id']} ({b.get('Status')}) from {rel_path}")
        return archived

    moved_ids = [str(b["_id"]) for _rel, _path, _arch, to_archive in plan for b in to_archive]

    # Imported here, as the delete doors import it: the governance layer
    # is not a dependency of importing this module, only of running a
    # sweep that rewrites the corpus.
    from .governance_gate import get_gate

    with get_gate(ws).admit_batch(
        action=ARCHIVE_VERB,
        batch_id=_sweep_batch_id("archive", moved_ids),
        block_ids=moved_ids,
        content="\n".join(moved_ids),
        tier=IngestTier.RESTAMP,
        actor="compaction",
        metadata={
            "door": "compaction.archive_completed_blocks",
            "retention_days": days,
            "blocks": len(moved_ids),
            "files": [rel for rel, _p, _a, _t in plan],
        },
    ) as receipt:
        for _rel_path, path, archive_rel, to_archive in plan:
            archived.extend(_archive_one_file(ws, path, archive_rel, to_archive))
        _log.info("compaction_archived", blocks=len(archived), admission=receipt.entry_id)

    return archived


def _archive_one_file(ws: str, path: str, archive_rel: str, to_archive: list[dict]) -> list[str]:
    """Move *to_archive* out of *path* into *archive_rel*; return the action lines.

    Runs inside the caller's open ``admit_batch`` scope. Split out of
    :func:`archive_completed_blocks` so the planning half and the moving
    half are each readable on their own; the body is the pre-5.0.2 move
    verbatim, because the defect was the missing scope and not this.
    """
    moved: list[str] = []
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

        if not archive_lines:
            return moved

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
        moved.append(f"Archived {b['_id']} ({b.get('Status')}) -> {archive_rel}")
    return moved


def _extract_block_text(content: str, block_id: str) -> str | None:
    """Extract raw text of a block from file content."""
    pattern = re.compile(rf"^\[{re.escape(block_id)}\].*?(?=^\[[A-Z]+-|^---|\Z)", re.MULTILINE | re.DOTALL)
    match = pattern.search(content)
    if match:
        return match.group(0).strip()
    return None


def cleanup_snapshots(ws: str, days: int = 30, dry_run: bool = False) -> list[str]:
    """Remove snapshot directories older than `days`.

    Snapshots are in intelligence/applied/<timestamp>/. Each contains
    APPLY_RECEIPT.md and a full workspace copy. Old ones can be large.
    """
    cleaned: list[str] = []
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
    cleaned: list[str] = []
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

        # Append to yearly archive (under lock to prevent races)
        with FileLock(log_path):
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

    **The fourth door that killed content with no record**, found while
    closing the third. Measured on a workspace built by
    ``mind-mem-init``: a sweep returned ``["Removed signal SIG-…
    (resolved)"]``, the blocks left ``intelligence/SIGNALS.md`` — a file
    in ``CORPUS_FILES``, so recall had been serving them — and
    ``memory/evidence_chain.jsonl`` gained **zero** rows while
    ``memory/deleted_blocks.jsonl`` gained no entry either. Same defect
    class as the MCP tool's Markdown leg, for the same reason: this
    function never called a block store, it spliced the corpus file
    itself, so the store's ``delete_block`` gate could not see it.

    **A compaction is a batch.** One
    :meth:`~mind_mem.governance_gate.GovernanceGate.admit_delete_batch`
    scope covers the whole sweep, so a run that removes forty signals
    leaves one authorisation record and one removal record carrying a
    Merkle root over every ``(block_id, content_hash)`` that actually
    went — not forty unlinked records with nothing saying they were one
    decision. Same reasoning as ``POST /clear``.

    The id set is frozen before the file is opened, so a signal written
    *while* the sweep runs is outside the receipt and cannot be taken by
    it. Eligibility is decided before the scope opens for the same reason
    the clear endpoint decides it there: a sweep with nothing to remove
    mints no authorisation, because a receipt covering nothing authorises
    nothing. ``dry_run`` returns on that same path — it destroys nothing,
    so it must not put a delete decision in the chain, and ``memory_health``
    calls it on every dashboard render.

    Raises:
        GovernanceBypassError: The gate refused the sweep (retired gate,
            drifted spec binding). Nothing is removed — a refused
            authorisation must never be reported as a completed
            compaction.
    """
    cleaned: list[str] = []
    signals_rel = os.path.join("intelligence", "SIGNALS.md")
    signals_path = os.path.join(ws, signals_rel)
    if not os.path.isfile(signals_path):
        return cleaned

    blocks = parse_file(signals_path)
    cutoff = datetime.now() - timedelta(days=days)
    cutoff_str = cutoff.strftime("%Y-%m-%d")

    removable = [b for b in blocks if b.get("Status") in ("resolved", "rejected") and b.get("Date", "9999-99-99") < cutoff_str]

    if not removable:
        return cleaned

    if dry_run:
        for b in removable:
            cleaned.append(f"[dry-run] Would remove signal {b['_id']} ({b.get('Status')})")
        return cleaned

    doomed = [str(b["_id"]) for b in removable]

    # Imported here, as ``memory_ops.delete_memory_item`` imports them:
    # the governance layer and the block store are dependencies of
    # *running* a sweep, not of importing this module.
    from .block_store import _record_deletion
    from .governance_gate import get_gate

    removed: list[tuple[str, str]] = []
    with get_gate(ws).admit_delete_batch(
        _sweep_batch_id("signal-sweep", doomed),
        doomed,
        rationale=f"{SIGNAL_SWEEP_RATIONALE}: resolved/rejected signals older than {days}d",
        # Empty lets the gate resolve the caller — the authenticated REST
        # agent when one is in front, else "system". A scheduled sweep has
        # no human identity to claim, and inventing one would put a name
        # in the actor field that nobody stood behind.
        actor="",
        target_file=signals_rel,
        metadata={"door": "compaction.compact_signals", "retention_days": days, "eligible": len(doomed)},
    ) as receipt:
        with FileLock(signals_path):
            with open(signals_path, "r", encoding="utf-8") as f:
                content = f.read()

            for b in removable:
                block_text = _extract_block_text(content, b["_id"])
                if block_text:
                    content = content.replace(block_text, "")
                    # Journal-ahead, inside the lock and before the
                    # rewrite lands, exactly as ``MarkdownBlockStore.
                    # delete_block`` orders it: a crash between the two
                    # must leave a recoverable copy of content that may
                    # be gone, never gone content with no copy.
                    _record_deletion(ws, str(b["_id"]), block_text)
                    removed.append((str(b["_id"]), block_text))
                    cleaned.append(f"Removed signal {b['_id']} ({b.get('Status')})")

            content = re.sub(r"\n{4,}", "\n\n\n", content)
            with open(signals_path, "w", encoding="utf-8") as f:
                f.write(content)

        # The blocks are gone from disk. Report them under the scope that
        # authorised it — after the rewrite and outside the file lock, at
        # the point ``MarkdownBlockStore.delete_block`` reports its own.
        # The journal above is recovery; this is the audit fact, and the
        # gate turns the ledger into the one removal record when the
        # scope closes. A signal whose text was not found removed nothing
        # and is reported as nothing.
        for block_id, block_text in removed:
            receipt.record_removal(block_id, block_text)
        _log.info("compaction_signals_removed", blocks=len(removed), admission=receipt.entry_id)

    return cleaned


def _run_tier_promotion(ws: str) -> int:
    """Invoke TierManager.run_promotion_cycle. Returns promoted count."""
    try:
        from .memory_tiers import TierManager

        db_path = os.path.join(ws, "intelligence", "tiers.db")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        # The one product construction site that HAS the workspace, so it is
        # the one that can turn tier events on. Flag still decides.
        mgr = TierManager(db_path, workspace=ws)
        promotions = mgr.run_promotion_cycle()
        # v3.0.0+ (#502): TTL/LRU decay — demote stale blocks, evict
        # never-used WORKING-tier entries. Runs alongside promotion so
        # the tier distribution actually stays bounded.
        demotions, evicted = mgr.run_decay_cycle()
        if demotions or evicted:
            _log.info(
                "tier_decay",
                demotions=len(demotions),
                evicted=len(evicted),
            )
        return len(promotions)
    except Exception as exc:  # pragma: no cover — best-effort
        _log.warning("tier_promotion_failed", error=str(exc))
        return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="mind-mem Compaction & GC Engine")
    parser.add_argument("workspace", nargs="?", default=".")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    parser.add_argument("--archive-days", type=int, default=90, help="Archive completed blocks older than N days (default: 90)")
    parser.add_argument("--snapshot-days", type=int, default=30, help="Remove snapshots older than N days (default: 30)")
    parser.add_argument("--log-days", type=int, default=180, help="Archive daily logs older than N days (default: 180)")
    parser.add_argument("--signal-days", type=int, default=60, help="Remove resolved signals older than N days (default: 60)")
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

    # 5. Tier promotion cycle — moves blocks through WORKING → SHARED →
    # LONG_TERM → VERIFIED based on access frequency + age. Best-effort
    # so compaction keeps working when the tier DB is unavailable.
    if not args.dry_run:
        promoted = _run_tier_promotion(ws)
        if promoted:
            all_actions.append(f"Promoted {promoted} block(s) through tiers")
            print(f"\nTier promotion: {promoted} block(s) moved up a tier")
        else:
            print("\nTier promotion: no blocks eligible for promotion")

    _log.info("compaction_complete", actions=len(all_actions), dry_run=args.dry_run)
    metrics.inc("compaction_actions", len(all_actions))
    print(f"\nTotal: {len(all_actions)} action(s)")


if __name__ == "__main__":
    main()
