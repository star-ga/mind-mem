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

# Directories included in apply-engine rollback snapshots
SNAPSHOT_DIRS: tuple[str, ...] = (
    "decisions",
    "tasks",
    "entities",
    "summaries",
    "memory",
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
