"""v3.2.0 §2.2 — one-shot migration helper for ``maintenance/`` subdivision.

Splits the legacy wholesale ``maintenance/`` directory into two
sibling subdirectories so the apply-engine's snapshot scope can
correctly categorise each file:

  maintenance/tracked/      — included in snapshots; rolled back.
  maintenance/append-only/  — excluded from snapshots; rollback keeps
                              whatever was appended during the aborted
                              apply.

Rule of thumb applied by :func:`classify_maintenance_file`:

  - Anything whose presence/content **changes** the next apply's
    behaviour (dedup-state, *-checkpoint, *.state, *.lock) → tracked.
  - Anything that's append-only observability output
    (*-report.txt, *.log, *.ndjson, compaction-*.jsonl) → append-only.
  - Unknown → ``tracked`` (safer to snapshot than to drop state).

Idempotent. Safe to call multiple times. Prints a one-line summary
per migrated file so users can audit what moved.

Invoked automatically by :func:`mind_mem.apply_engine.apply_proposal`
on first-run detection of the old layout (no ``tracked/`` or
``append-only/`` subdir present yet). Can also be run standalone::

    python3 -m mind_mem.maintenance_migrate [workspace_path]

STARGA, Inc. — Apache-2.0.
"""

from __future__ import annotations

import os
import shutil
import sys
from typing import Literal

Category = Literal["tracked", "append-only"]

_APPEND_ONLY_SUFFIXES = (
    "-report.txt",
    ".log",
    ".ndjson",
)
_APPEND_ONLY_PREFIXES = (
    "compaction-",
    "validation-",
    "intel-scan-",
)
_TRACKED_SUFFIXES = (
    "-state.json",
    "-checkpoint.json",
    ".lock",
)


def classify_maintenance_file(basename: str) -> Category:
    """Return the snapshot category for a maintenance-directory file.

    Suffix rules beat prefix rules so ``compaction-checkpoint.json``
    (a state file with a compaction prefix) is correctly classified
    as tracked rather than as append-only compaction output.
    """
    name = basename.lower()
    # Tracked-suffix wins first — behavioural state is the safe side.
    for suf in _TRACKED_SUFFIXES:
        if name.endswith(suf):
            return "tracked"
    for suf in _APPEND_ONLY_SUFFIXES:
        if name.endswith(suf):
            return "append-only"
    for prefix in _APPEND_ONLY_PREFIXES:
        if name.startswith(prefix):
            return "append-only"
    # Unknown — snapshot-inclusive is the safer default.
    return "tracked"


def already_migrated(ws: str) -> bool:
    """True when either subdirectory exists (migration has happened)."""
    base = os.path.join(ws, "maintenance")
    return os.path.isdir(os.path.join(base, "tracked")) or os.path.isdir(os.path.join(base, "append-only"))


def migrate_maintenance(ws: str, *, verbose: bool = True) -> dict[Category, int]:
    """Move files under ``<ws>/maintenance`` into tracked/append-only
    subdirs per :func:`classify_maintenance_file`.

    Returns a per-category file count. When the layout is already
    migrated (``already_migrated`` is True), returns zero counts and
    does nothing — safe to call on every apply.
    """
    counts: dict[Category, int] = {"tracked": 0, "append-only": 0}
    base = os.path.join(ws, "maintenance")
    if not os.path.isdir(base):
        return counts
    if already_migrated(ws):
        return counts

    tracked_dir = os.path.join(base, "tracked")
    append_dir = os.path.join(base, "append-only")
    os.makedirs(tracked_dir, exist_ok=True)
    os.makedirs(append_dir, exist_ok=True)

    # Move every top-level file that ISN'T already a tracked/ or
    # append-only/ subdir entry.
    for entry in sorted(os.listdir(base)):
        src = os.path.join(base, entry)
        if not os.path.isfile(src):
            continue
        cat = classify_maintenance_file(entry)
        dst_dir = tracked_dir if cat == "tracked" else append_dir
        dst = os.path.join(dst_dir, entry)
        if os.path.exists(dst):
            # Don't overwrite — rename the incoming to avoid collision.
            root, ext = os.path.splitext(entry)
            suffix = 1
            while os.path.exists(dst):
                dst = os.path.join(dst_dir, f"{root}.{suffix}{ext}")
                suffix += 1
        shutil.move(src, dst)
        counts[cat] += 1
        if verbose:
            print(
                f"[maintenance-migrate] {entry} → maintenance/{cat}/",
                file=sys.stderr,
            )

    return counts


def main() -> int:
    ws = sys.argv[1] if len(sys.argv) > 1 else "."
    if not os.path.isfile(os.path.join(ws, "mind-mem.json")):
        print(f"error: no mind-mem.json at {ws!r}", file=sys.stderr)
        return 1
    counts = migrate_maintenance(ws, verbose=True)
    print(
        f"migrated: tracked={counts['tracked']} append-only={counts['append-only']}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
