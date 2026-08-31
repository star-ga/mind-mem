#!/usr/bin/env python3
"""Mind-Mem Schema Version Migration. Zero external deps.

Detects workspace schema version from mind-mem.json and performs safe,
idempotent migrations to bring older workspaces up to date.

Usage:
    python3 -m mind_mem.schema_version [workspace_path]

As library:
    from .schema_version import get_workspace_version, check_migration_needed, migrate_workspace
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

from .observability import get_logger  # noqa: E402

_log = get_logger("schema_version")

CURRENT_SCHEMA_VERSION = "2.1.0"


class MigrationError(RuntimeError):
    """A migration step could not be applied.

    Raised instead of swallowing the underlying ``OSError`` /
    ``JSONDecodeError``: a read-only workspace, a full disk or one
    corrupt byte of JSON must not be reported as a completed migration.
    """


# Ordered list of migrations. Each entry is (from_version, to_version, description, callable).
# Migrations are applied in order; each must be idempotent.
_MIGRATIONS: list[tuple[str, str, str, Callable[[str], None]]] = []


def _version_tuple(v: str) -> tuple[int, ...]:
    """Convert '1.0.0' -> (1, 0, 0) for comparison."""
    return tuple(int(x) for x in v.split("."))


def get_workspace_version(workspace: str) -> str:
    """Read schema version from mind-mem.json. Returns '1.0.0' if missing or unreadable."""
    config_path = os.path.join(workspace, "mind-mem.json")
    if not os.path.isfile(config_path):
        return "1.0.0"
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        return str(config.get("schema_version", config.get("version", "1.0.0")))
    except (json.JSONDecodeError, OSError):
        return "1.0.0"


def check_migration_needed(workspace: str) -> list[str]:
    """Return list of migration step descriptions needed for this workspace."""
    current = get_workspace_version(workspace)
    current_t = _version_tuple(current)
    target_t = _version_tuple(CURRENT_SCHEMA_VERSION)

    if current_t >= target_t:
        return []

    steps = []
    for from_v, to_v, desc, _fn in _MIGRATIONS:
        from_t = _version_tuple(from_v)
        to_t = _version_tuple(to_v)
        if current_t >= from_t and current_t < to_t:
            steps.append(f"{from_v} -> {to_v}: {desc}")
    return steps


def migrate_workspace(workspace: str) -> dict:
    """Perform safe, idempotent migrations on a workspace.

    Returns dict with keys: migrated (bool — whether any step actually
    applied), from_version, to_version (the version genuinely reached,
    NOT the target, when a step failed), steps (applied step
    descriptions) and failed (descriptions of the step that raised,
    empty on success). Migration stops at the first failing step
    because later steps build on it.
    """
    workspace = os.path.abspath(workspace)
    from_version = get_workspace_version(workspace)
    from_t = _version_tuple(from_version)
    target_t = _version_tuple(CURRENT_SCHEMA_VERSION)

    if from_t >= target_t:
        _log.info("migrate_skip", workspace=workspace, version=from_version, reason="already current")
        return {
            "migrated": False,
            "from_version": from_version,
            "to_version": from_version,
            "steps": [],
            "failed": [],
        }

    applied: list[str] = []
    failed: list[str] = []
    reached = from_version

    for from_v, to_v, desc, fn in _MIGRATIONS:
        to_t = _version_tuple(to_v)
        if from_t < to_t:
            _log.info("migrate_step", workspace=workspace, step=desc)
            try:
                fn(workspace)
            except Exception as exc:
                # Report the step that did not apply instead of
                # appending it to ``steps`` and returning the target
                # version: an unwritable or corrupt workspace stays on
                # its old schema, and a caller told otherwise will read
                # the legacy keys and re-migrate on every run. Later
                # steps build on this one, so stop here.
                _log.error("migrate_step_failed", workspace=workspace, step=desc, error=str(exc))
                failed.append(f"{from_v} -> {to_v}: {desc} — {exc}")
                break
            applied.append(f"{from_v} -> {to_v}: {desc}")
            reached = to_v

    to_version = reached if failed else CURRENT_SCHEMA_VERSION
    _log.info(
        "migrate_done",
        workspace=workspace,
        from_version=from_version,
        to_version=to_version,
        steps=len(applied),
        failed=len(failed),
    )

    return {
        "migrated": bool(applied),
        "from_version": from_version,
        "to_version": to_version,
        "steps": applied,
        "failed": failed,
    }


# ---------------------------------------------------------------------------
# Migration steps (each must be idempotent)
# ---------------------------------------------------------------------------


def _migrate_v1_to_v2(workspace: str) -> None:
    """v1.0 -> v2.0: add intelligence/proposed/, shared/, and schema_version field."""
    # Ensure intelligence/proposed/ exists
    proposed_dir = os.path.join(workspace, "intelligence", "proposed")
    os.makedirs(proposed_dir, exist_ok=True)

    # Ensure shared/ directory exists
    shared_dir = os.path.join(workspace, "shared")
    os.makedirs(shared_dir, exist_ok=True)

    # Add schema_version to mind-mem.json
    config_path = os.path.join(workspace, "mind-mem.json")
    config: dict = {}
    if os.path.isfile(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        except (json.JSONDecodeError, OSError):
            config = {}

    config["schema_version"] = "2.0.0"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        f.write("\n")


def _migrate_v2_to_v21(workspace: str) -> None:
    """v2.0 -> v2.1: rename self_correcting_mode to governance_mode.

    Backfill script for workspaces created before the terminology change.
    Idempotent: skips if governance_mode already present.

    intel-state.json is migrated FIRST because mind-mem.json carries the
    ``schema_version`` bump: if the second half fails, the workspace is
    still recorded as 2.0.0 and the next run retries the whole step,
    instead of being stamped 2.1.0 with half the rename applied.
    """
    # Migrate intel-state.json
    state_path = os.path.join(workspace, "memory", "intel-state.json")
    if os.path.isfile(state_path):
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            if "self_correcting_mode" in state and "governance_mode" not in state:
                state["governance_mode"] = state.pop("self_correcting_mode")
                with open(state_path, "w", encoding="utf-8") as f:
                    json.dump(state, f, indent=2)
                    f.write("\n")
        except (json.JSONDecodeError, OSError) as exc:
            raise MigrationError(f"{state_path}: {exc}") from exc

    # Migrate mind-mem.json (carries the schema_version bump — last)
    config_path = os.path.join(workspace, "mind-mem.json")
    if os.path.isfile(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            changed = False
            if "self_correcting_mode" in config and "governance_mode" not in config:
                config["governance_mode"] = config.pop("self_correcting_mode")
                changed = True
            if config.get("schema_version") != "2.1.0":
                config["schema_version"] = "2.1.0"
                changed = True
            if changed:
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(config, f, indent=2)
                    f.write("\n")
        except (json.JSONDecodeError, OSError) as exc:
            raise MigrationError(f"{config_path}: {exc}") from exc


# Register migrations in order
_MIGRATIONS.append(("1.0.0", "2.0.0", "Add intelligence/proposed/, shared/, and schema_version", _migrate_v1_to_v2))
_MIGRATIONS.append(("2.0.0", "2.1.0", "Rename self_correcting_mode to governance_mode", _migrate_v2_to_v21))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    ws = sys.argv[1] if len(sys.argv) > 1 else "."
    ws = os.path.abspath(ws)

    version = get_workspace_version(ws)
    print(f"Workspace: {ws}")
    print(f"Current version: {version}")
    print(f"Target version:  {CURRENT_SCHEMA_VERSION}")

    needed = check_migration_needed(ws)
    if not needed:
        print("No migration needed.")
        return 0

    print(f"\nMigration steps ({len(needed)}):")
    for step in needed:
        print(f"  - {step}")

    result = migrate_workspace(ws)
    if result["migrated"]:
        print(f"\nMigrated {result['from_version']} -> {result['to_version']}")
        for step in result["steps"]:
            print(f"  + {step}")
    else:
        print("\nNo changes applied.")

    if result["failed"]:
        # Exit non-zero: the workspace is still on ``to_version``, so a
        # script that chained on this must not proceed as if migrated.
        print(f"\nFAILED — workspace is still at {result['to_version']}:")
        for step in result["failed"]:
            print(f"  ! {step}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
