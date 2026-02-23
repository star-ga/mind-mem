#!/usr/bin/env python3
"""mind-mem workspace initializer. Zero external deps.

Scaffolds directory structure and copies templates for a new mind-mem workspace.
Never overwrites existing files.

Usage:
    python3 scripts/init_workspace.py [workspace_path]
    python3 scripts/init_workspace.py /path/to/workspace
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys

_log = logging.getLogger("mind-mem.init_workspace")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLUGIN_ROOT = os.path.dirname(SCRIPT_DIR)
TEMPLATE_DIR = os.path.join(PLUGIN_ROOT, "templates")

DIRS = [
    "decisions",
    "tasks",
    "entities",
    "memory",
    "summaries/weekly",
    "summaries/daily",
    "intelligence",
    "intelligence/proposed",
    "intelligence/applied",
    "intelligence/state/snapshots",
    "maintenance",
    "maintenance/weeklog",
]

TEMPLATE_MAP = {
    "decisions/DECISIONS.md": "DECISIONS.md",
    "tasks/TASKS.md": "TASKS.md",
    "entities/projects.md": "projects.md",
    "entities/people.md": "people.md",
    "entities/tools.md": "tools.md",
    "entities/incidents.md": "incidents.md",
    "MEMORY.md": "MEMORY.md",
    "memory/intel-state.json": "intel-state.json",
    "memory/maint-state.json": "maint-state.json",
    "intelligence/CONTRADICTIONS.md": "CONTRADICTIONS.md",
    "intelligence/DRIFT.md": "DRIFT.md",
    "intelligence/IMPACT.md": "IMPACT.md",
    "intelligence/BRIEFINGS.md": "BRIEFINGS.md",
    "intelligence/AUDIT.md": "AUDIT.md",
    "intelligence/SIGNALS.md": "SIGNALS.md",
    "intelligence/SCAN_LOG.md": "SCAN_LOG.md",
    "intelligence/proposed/DECISIONS_PROPOSED.md": "DECISIONS_PROPOSED.md",
    "intelligence/proposed/TASKS_PROPOSED.md": "TASKS_PROPOSED.md",
    "intelligence/proposed/EDITS_PROPOSED.md": "EDITS_PROPOSED.md",
}

MAINTENANCE_SCRIPTS = [
    "intel_scan.py",
    "apply_engine.py",
    "block_parser.py",
    "recall.py",
    "capture.py",
    "validate.sh",
    "validate_py.py",
    "mind_filelock.py",
    "compaction.py",
    "observability.py",
    "namespaces.py",
    "conflict_resolver.py",
    "backup_restore.py",
    "transcript_capture.py",
]

DEFAULT_CONFIG = {
    "version": "1.4.0",
    "workspace_path": ".",
    "auto_capture": True,
    "auto_recall": True,
    "governance_mode": "detect_only",
    "recall": {
        "backend": "bm25",
    },
    "proposal_budget": {
        "per_run": 3,
        "per_day": 6,
        "backlog_limit": 30,
    },
    "scan_schedule": "daily",
    "mcp_acl": {
        "admin_tools": [
            "write_memory", "apply_proposal", "approve_apply",
            "rollback_proposal", "delete_memory_item", "reindex_vectors",
        ],
        "default_scope": "user",
    },
    "mcp_rate_limit": {
        "max_calls_per_minute": 120,
        "query_timeout_seconds": 30,
    },
    "limits": {
        "max_recall_results": 100,
        "max_similar_results": 50,
        "max_prefetch_results": 20,
        "max_category_results": 10,
        "query_timeout_seconds": 30,
        "rate_limit_calls_per_minute": 120,
    },
}


# Numeric range constraints for recall config: (min, max, default)
_RECALL_RANGES: dict[str, tuple[float, float, float]] = {
    "bm25_k1": (0.5, 3.0, 1.2),
    "bm25_b": (0.0, 1.0, 0.75),
    "limit": (1, 1000, 20),
    "rrf_k": (1, 200, 60),
    "bm25_weight": (0.0, 10.0, 1.0),
    "vector_weight": (0.0, 10.0, 1.0),
    "recency_weight": (0.0, 1.0, 0.3),
    "top_k": (1, 200, 18),
}


def _validate_config(cfg: dict) -> dict:
    """Validate and clamp numeric config values to safe ranges.

    Mutates and returns *cfg* so callers can chain.  Out-of-range values
    are clamped and a warning is logged for each adjustment.
    """
    if not isinstance(cfg, dict):
        return cfg
    recall = cfg.get("recall")
    if not isinstance(recall, dict):
        return cfg

    for key, (lo, hi, default) in _RECALL_RANGES.items():
        if key not in recall:
            continue
        raw = recall[key]
        # Coerce to numeric; fall back to default on bad type
        try:
            val = float(raw)
        except (TypeError, ValueError):
            _log.warning(
                "config_value_invalid: recall.%s=%r is not numeric, using default %s",
                key, raw, default,
            )
            recall[key] = type(default)(default)
            continue

        if val < lo or val > hi:
            clamped = max(lo, min(hi, val))
            # Preserve int type for integer-range keys
            clamped = type(default)(clamped)
            _log.warning(
                "config_value_clamped: recall.%s=%s out of range [%s, %s], clamped to %s",
                key, raw, lo, hi, clamped,
            )
            recall[key] = clamped
        else:
            # Preserve type (int vs float) matching the default
            recall[key] = type(default)(val)

    return cfg


def load_config(ws: str) -> dict:
    """Load mind-mem.json from *ws*, validate numeric ranges, return config dict.

    Returns DEFAULT_CONFIG (shallow copy) if the file is missing or unreadable.
    """
    config_path = os.path.join(os.path.abspath(ws), "mind-mem.json")
    if os.path.isfile(config_path):
        try:
            with open(config_path) as f:
                cfg = json.load(f)
            return _validate_config(cfg)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            _log.warning("config_load_failed: %s (%s)", config_path, exc)
    return dict(DEFAULT_CONFIG)


def init(ws: str) -> tuple[list[str], list[str]]:
    """Initialize a mind-mem workspace."""
    ws = os.path.abspath(ws)

    # Security: reject symlinks as workspace root to prevent writing outside intended location
    if os.path.islink(ws):
        print(f"ERROR: workspace path is a symlink: {ws}", file=sys.stderr)
        sys.exit(1)

    created = []
    skipped = []

    # Create directories
    for d in DIRS:
        path = os.path.join(ws, d)
        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)
            created.append(f"dir:  {d}/")

    # Copy templates (never overwrite)
    for target_rel, template_name in TEMPLATE_MAP.items():
        target = os.path.join(ws, target_rel)
        if os.path.exists(target):
            skipped.append(target_rel)
            continue
        src = os.path.join(TEMPLATE_DIR, template_name)
        if os.path.exists(src):
            os.makedirs(os.path.dirname(target), exist_ok=True)
            shutil.copy2(src, target)
            created.append(f"file: {target_rel}")

    # Copy maintenance scripts (never overwrite)
    for script in MAINTENANCE_SCRIPTS:
        src = os.path.join(SCRIPT_DIR, script)
        dst = os.path.join(ws, "maintenance", script)
        if os.path.exists(dst):
            skipped.append(f"maintenance/{script}")
            continue
        if os.path.exists(src):
            shutil.copy2(src, dst)
            created.append(f"file: maintenance/{script}")

    # Create mind-mem.json config (never overwrite)
    config_path = os.path.join(ws, "mind-mem.json")
    if not os.path.exists(config_path):
        with open(config_path, "w") as f:
            json.dump(DEFAULT_CONFIG, f, indent=2)
            f.write("\n")
        created.append("file: mind-mem.json")
    else:
        skipped.append("mind-mem.json")

    # Create empty weekly summary placeholder
    placeholder = os.path.join(ws, "summaries/weekly/.gitkeep")
    if not os.path.exists(placeholder):
        with open(placeholder, "w") as f:
            pass

    return created, skipped


def main():
    ws = sys.argv[1] if len(sys.argv) > 1 else "."

    print(f"mind-mem init: {os.path.abspath(ws)}")
    print()

    created, skipped = init(ws)

    if created:
        print(f"Created ({len(created)}):")
        for item in created:
            print(f"  + {item}")

    if skipped:
        print(f"\nSkipped (already exist) ({len(skipped)}):")
        for item in skipped:
            print(f"  = {item}")

    print(f"\nDone. Run 'bash maintenance/validate.sh {ws}' to verify.")


if __name__ == "__main__":
    main()
