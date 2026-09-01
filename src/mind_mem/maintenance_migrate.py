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

Invoked by :func:`mind_mem.apply_engine.apply_proposal` on first-run
detection of the old layout (no ``tracked/`` or ``append-only/`` subdir
present yet), under the workspace apply lock and before the snapshot —
see :func:`migrate_if_enabled`. That auto-invocation is gated on the
``v4.maintenance_layout`` flag and is OFF by default: with the flag
unset nothing here reads or writes the filesystem, so a workspace that
never opts in keeps the legacy flat layout byte-for-byte.

Can also be run explicitly::

    mm migrate --maintenance
    python3 -m mind_mem.maintenance_migrate [workspace_path]

STARGA, Inc. — Apache-2.0.
"""

from __future__ import annotations

import json
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

#: Suffixes of files that are NOT migrated at all — they stay flat in
#: ``maintenance/``.
#:
#: ``init_workspace.MAINTENANCE_SCRIPTS`` copies the workspace's own
#: tooling (``validate.sh`` and every ``*.py`` helper) into
#: ``maintenance/`` at init time, and every entry in that list ends in
#: ``.py`` or ``.sh``. Those are shipped CODE, not corpus state and not
#: observability output: their paths are quoted to operators (init
#: prints ``bash maintenance/validate.sh <ws>``) and pinned by
#: ``tests/test_maintenance_scripts_ship.py``. ``classify_maintenance_file``
#: would file them under ``tracked`` via the unknown-default and the move
#: would break every one of those references — so the move loop skips
#: them outright rather than the classifier being bent to cover a case
#: that is not about snapshot scope at all.
_SKIP_SUFFIXES = (".py", ".sh")

#: v4 feature flag gating the automatic first-run migration.
FLAG = "maintenance_layout"


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
        if entry.lower().endswith(_SKIP_SUFFIXES):
            # Shipped workspace tooling — see _SKIP_SUFFIXES. Left where
            # init_workspace put it; nothing about snapshot scope applies.
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


def flag_enabled(ws: str) -> bool:
    """``v4.maintenance_layout`` state for *workspace*, ambient config as fallback.

    ``feature_flags.is_enabled`` resolves the config from the process
    environment, which is right for process-wide surfaces and wrong for
    an API whose whole subject is one explicit workspace directory — so
    the workspace's own ``mind-mem.json`` is consulted first. Same shape
    as :func:`mind_mem.lint._flag_enabled`, for the same reason.

    Reads only; never writes. Unset (the default) → ``False``.
    """
    config_path = os.path.join(ws, "mind-mem.json")
    try:
        with open(config_path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        data = None
    if isinstance(data, dict):
        block = data.get("v4")
        if isinstance(block, dict):
            sub = block.get(FLAG)
            if isinstance(sub, dict):
                return sub.get("enabled") is True

    from .v4.feature_flags import is_enabled

    return is_enabled(FLAG)


def migrate_if_enabled(ws: str, *, verbose: bool = False) -> dict[Category, int] | None:
    """Run the first-run layout migration iff ``v4.maintenance_layout`` is ON.

    The gated entry point the apply engine calls. Returns ``None`` when
    the flag is OFF — and in that case touches nothing on disk at all, so
    a workspace that never opts in behaves exactly as it did before this
    call site existed. When the flag is ON it returns the per-category
    move counts, which are ``{"tracked": 0, "append-only": 0}`` on every
    run after the first (``already_migrated`` short-circuits).

    Deterministic: no clock, no randomness, no network. The file order is
    ``sorted(os.listdir(...))``, so a given flat layout always splits the
    same way.
    """
    if not flag_enabled(ws):
        return None
    return migrate_maintenance(ws, verbose=verbose)


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
