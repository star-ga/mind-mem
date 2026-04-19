"""Central corpus path registry for mind-mem.

All modules should import corpus paths from here instead of hardcoding them.
This is the single source of truth for which directories constitute the corpus.
"""

from __future__ import annotations

# Core corpus directories (order matters for scan priority)
CORPUS_DIRS: tuple[str, ...] = (
    "decisions",
    "tasks",
    "entities",
    "intelligence",
)

# Directories that contain active memory blocks
MEMORY_DIRS: tuple[str, ...] = CORPUS_DIRS

# Directories included in backup snapshots
BACKUP_DIRS: tuple[str, ...] = CORPUS_DIRS + ("memory", "summaries", "shared", "agents")

# Directories included in apply-engine rollback snapshots.
#
# v3.2.0 §2.2 atomicity-scope fix: ``maintenance/tracked/`` joins the
# snapshot so behavioural state files (dedup-state.json, compaction
# checkpoints, anything whose presence/absence changes the next
# apply's outcome) are captured + restored atomically with the
# corpus. Append-only files (validation-report.txt, noisy logs) keep
# their prior home under ``maintenance/append-only/`` and are still
# snapshot-excluded so they don't bloat the archive.
SNAPSHOT_DIRS: tuple[str, ...] = (
    "decisions",
    "tasks",
    "entities",
    "summaries",
    "memory",
    "maintenance/tracked",
)

# ``maintenance/`` directories that are EXPLICITLY excluded from
# snapshots — append-only outputs where rollback would lose real
# signal. Anything under a path listed here is guaranteed to skip
# both the snapshot capture and the restore walk.
SNAPSHOT_EXCLUDE_DIRS: tuple[str, ...] = (
    "maintenance/append-only",
    "intelligence/applied",
)

# Directories used for cross-reference validation
VALIDATE_DIRS: tuple[str, ...] = (
    "decisions",
    "tasks",
    "entities",
    "summaries",
)

# File extensions recognized as block files
BLOCK_EXTENSIONS: tuple[str, ...] = (".md",)
