#!/usr/bin/env python3
"""mind-mem workspace initializer. Zero external deps (Postgres optional).

Scaffolds directory structure and copies templates for a new mind-mem workspace.
Never overwrites existing files.

Usage:
    python3 -m mind_mem.init_workspace [workspace_path]
    python3 -m mind_mem.init_workspace /path/to/workspace

    # Opt-in Postgres backend (default stays SQLite/markdown):
    python3 -m mind_mem.init_workspace /path/to/workspace \\
        --backend postgres --dsn postgresql://user:pw@host/db --schema mind_mem

The Postgres backend may also be selected via the environment so CI /
container entrypoints need no flags:
    MIND_MEM_BACKEND=postgres MIND_MEM_DSN=postgresql://... \\
        python3 -m mind_mem.init_workspace /path/to/workspace
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from typing import Any

from . import __version__ as _pkg_version
from .observability import get_logger

_log = get_logger("init_workspace")

# Backends init_workspace can scaffold. ``markdown`` is the zero-config
# default (SQLite recall cache + local Markdown corpus); ``postgres`` and
# ``encrypted`` are opt-in and write an extra ``block_store`` config
# section. Kept in sync with ``storage._SUPPORTED_BACKENDS``.
SUPPORTED_BACKENDS = ("markdown", "postgres", "encrypted")

# When a non-markdown block_store backend is chosen, recall must read from
# the local SQLite FTS cache (which the store-driven reindexer mirrors from
# the backend) rather than the empty Markdown corpus. ``markdown`` keeps the
# zero-config BM25 scan over the on-disk corpus.
_BACKEND_RECALL = {
    "markdown": "bm25",
    "postgres": "sqlite",
    "encrypted": "bm25",
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# PLUGIN_ROOT: go up two levels from src/mind_mem/ to repo root
PLUGIN_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
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

# DEFAULT_CONFIG.version tracks the package version on every workspace init
# so users can tell at a glance what release wrote their config. Was
# hardcoded "1.7.0" through v3.8.10 (purely informational; nothing validates it).
DEFAULT_CONFIG = {
    "version": _pkg_version,
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
            "write_memory",
            "apply_proposal",
            "approve_apply",
            "rollback_proposal",
            "delete_memory_item",
            "reindex_vectors",
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


def _build_config(
    *,
    backend: str = "markdown",
    dsn: str | None = None,
    schema: str | None = None,
) -> dict[str, Any]:
    """Return a fresh config dict for *backend*.

    For the default ``markdown`` backend this is a deep copy of
    :data:`DEFAULT_CONFIG` with **no** ``block_store`` section — the
    zero-config SQLite/Markdown layout, byte-for-byte unchanged.

    For ``postgres`` (and ``encrypted``) a ``block_store`` section is
    added and ``recall.backend`` is set so the recall cache reads from the
    backend-mirroring SQLite FTS index rather than the empty Markdown
    corpus.

    Args:
        backend: One of :data:`SUPPORTED_BACKENDS`.
        dsn:     Postgres connection string (required for ``postgres``).
        schema:  Postgres schema name (defaults to ``"mind_mem"`` for the
                 ``postgres`` backend; ignored otherwise).

    Returns:
        A new config dict ready to serialise to ``mind-mem.json``.

    Raises:
        ValueError: ``backend`` is unknown, or ``postgres`` is requested
                    without a ``dsn``.
    """
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; choose from {', '.join(SUPPORTED_BACKENDS)}")

    # Deep-ish copy so callers never mutate the module-level template.
    cfg: dict[str, Any] = json.loads(json.dumps(DEFAULT_CONFIG))

    if backend == "markdown":
        # Default path: identical to the historical config — no
        # block_store section, recall.backend = "bm25".
        return cfg

    recall = cfg.setdefault("recall", {})
    recall["backend"] = _BACKEND_RECALL[backend]

    if backend == "postgres":
        if not dsn:
            raise ValueError("the postgres backend requires a --dsn (or MIND_MEM_DSN) connection string")
        cfg["block_store"] = {
            "backend": "postgres",
            "dsn": dsn,
            "schema": schema or "mind_mem",
        }
    elif backend == "encrypted":
        # Encrypted wraps the Markdown corpus; the passphrase is supplied
        # at runtime via MIND_MEM_ENCRYPTION_PASSPHRASE (never written to
        # disk). No dsn/schema apply.
        cfg["block_store"] = {"backend": "encrypted"}

    return cfg


def _ensure_postgres_schema(dsn: str, schema: str) -> tuple[bool, str | None]:
    """Best-effort: create the Postgres schema/tables for a fresh install.

    Returns ``(ok, error)``. Never raises — a missing ``psycopg`` driver or
    an unreachable database degrades to ``(False, <reason>)`` so init still
    writes a usable config and the SQLite-default path is never affected.
    """
    try:
        from .block_store_postgres import PostgresBlockStore
    except ImportError as exc:
        return False, f"psycopg not installed; run 'pip install \"mind-mem[postgres]\"' then mind-mem-init again ({exc})"

    store = None
    try:
        store = PostgresBlockStore(dsn=dsn, schema=schema, workspace=os.getcwd())
        store._ensure_schema()
        return True, None
    except Exception as exc:  # noqa: BLE001 — surface, don't crash init
        return False, str(exc)
    finally:
        if store is not None:
            try:
                store.close()
            except Exception:  # noqa: BLE001
                _log.debug("store.close during cleanup failed", exc_info=True)


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
            _log.warning(f"config_value_invalid: {key}", key=key, raw=raw, default=default)
            recall[key] = type(default)(default)
            continue

        if val < lo or val > hi:
            clamped = max(lo, min(hi, val))
            # Preserve int type for integer-range keys
            clamped = type(default)(clamped)
            _log.warning(f"config_value_clamped: {key}", key=key, raw=raw, lo=lo, hi=hi, clamped=clamped)
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
            _log.warning(f"config_load_failed: {config_path}", path=config_path, error=str(exc))
    return dict(DEFAULT_CONFIG)


def init(
    ws: str,
    *,
    backend: str = "markdown",
    dsn: str | None = None,
    schema: str | None = None,
) -> tuple[list[str], list[str]]:
    """Initialize a mind-mem workspace.

    Args:
        ws:      Workspace root path.
        backend: Block-store backend to scaffold (:data:`SUPPORTED_BACKENDS`).
                 ``"markdown"`` (default) writes the historical zero-config
                 layout; ``"postgres"`` / ``"encrypted"`` add a
                 ``block_store`` section to ``mind-mem.json``.
        dsn:     Postgres connection string (required when ``backend`` is
                 ``"postgres"``).
        schema:  Postgres schema name (``backend="postgres"`` only).

    Returns:
        ``(created, skipped)`` lists of relative-path descriptions.

    Raises:
        ValueError: ``backend`` is unknown or ``postgres`` lacks a ``dsn``.
    """
    ws = os.path.abspath(ws)

    # Validate backend selection up front (before any filesystem writes) so
    # a bad invocation never leaves a half-built workspace.
    config = _build_config(backend=backend, dsn=dsn, schema=schema)

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
            os.makedirs(path, mode=0o700, exist_ok=True)
            created.append(f"dir:  {d}/")

    # Copy templates (never overwrite)
    for target_rel, template_name in TEMPLATE_MAP.items():
        target = os.path.join(ws, target_rel)
        if os.path.exists(target):
            skipped.append(target_rel)
            continue
        src = os.path.join(TEMPLATE_DIR, template_name)
        if os.path.exists(src):
            os.makedirs(os.path.dirname(target), mode=0o700, exist_ok=True)
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

    # Create mind-mem.json config (never overwrite). For the default
    # markdown backend ``config`` equals DEFAULT_CONFIG (no block_store
    # section), so the zero-config path is byte-for-byte unchanged.
    config_path = os.path.join(ws, "mind-mem.json")
    if not os.path.exists(config_path):
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
            f.write("\n")
        os.chmod(config_path, 0o600)
        created.append("file: mind-mem.json")
    else:
        skipped.append("mind-mem.json")

    # Create empty weekly summary placeholder
    placeholder = os.path.join(ws, "summaries/weekly/.gitkeep")
    if not os.path.exists(placeholder):
        with open(placeholder, "w") as f:
            pass

    return created, skipped


def _build_arg_parser() -> argparse.ArgumentParser:
    """Construct the ``mind-mem-init`` argument parser.

    Unknown flags are rejected by argparse (exit 2) instead of being
    silently treated as the workspace path — which previously caused
    ``mind-mem-init --help`` to create a literal ``./--help`` directory.
    """
    parser = argparse.ArgumentParser(
        prog="mind-mem-init",
        description="Initialize a mind-mem workspace (SQLite/Markdown by default; opt-in Postgres).",
    )
    parser.add_argument(
        "workspace",
        nargs="?",
        default=".",
        help="Workspace directory to initialize (default: current directory).",
    )
    parser.add_argument(
        "--backend",
        choices=SUPPORTED_BACKENDS,
        default=os.environ.get("MIND_MEM_BACKEND", "markdown"),
        help="Block-store backend (default: markdown / SQLite, zero-config). Override via MIND_MEM_BACKEND.",
    )
    parser.add_argument(
        "--dsn",
        default=os.environ.get("MIND_MEM_DSN"),
        help="Postgres connection string (required for --backend postgres). Override via MIND_MEM_DSN.",
    )
    parser.add_argument(
        "--schema",
        default=os.environ.get("MIND_MEM_SCHEMA"),
        help="Postgres schema name (default: mind_mem; --backend postgres only).",
    )
    parser.add_argument(
        "--ensure-schema",
        action="store_true",
        help="For --backend postgres, also CREATE the schema/tables now (best-effort; skipped if psycopg/Postgres unavailable).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    ws = args.workspace

    print(f"mind-mem init: {os.path.abspath(ws)} (backend={args.backend})")
    print()

    try:
        created, skipped = init(ws, backend=args.backend, dsn=args.dsn, schema=args.schema)
    except ValueError as exc:
        parser.error(str(exc))  # exits 2 with a usage message

    if created:
        print(f"Created ({len(created)}):")
        for item in created:
            print(f"  + {item}")

    if skipped:
        print(f"\nSkipped (already exist) ({len(skipped)}):")
        for item in skipped:
            print(f"  = {item}")

    # Opt-in Postgres schema provisioning. Best-effort: a failure here
    # never aborts init — the config is already written and the user can
    # provision later (or fix the DSN). The SQLite/markdown default never
    # reaches this branch.
    if args.backend == "postgres" and args.ensure_schema:
        schema = args.schema or "mind_mem"
        ok, err = _ensure_postgres_schema(args.dsn or "", schema)
        if ok:
            print(f"\nPostgres schema ensured: {schema}")
        else:
            print(f"\nWARNING: could not ensure Postgres schema ({schema}): {err}", file=sys.stderr)
            print("The config was still written; provision the DB and re-run with --ensure-schema, ", file=sys.stderr)
            print("or run 'mm doctor --rebuild-cache' once the backend is reachable.", file=sys.stderr)

    print(f"\nDone. Run 'bash maintenance/validate.sh {ws}' to verify.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
