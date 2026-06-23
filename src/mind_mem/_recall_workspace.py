# Copyright 2026 STARGA, Inc.
"""Workspace resolution + emptiness probe for the recall CLI entrypoints.

Both recall CLIs (``python -m mind_mem.recall`` via ``_recall_core.main`` and
``python -m mind_mem.recall_vector`` via ``recall_vector.main``) previously
hardcoded ``--workspace`` default ``"."`` and never honoured
``MIND_MEM_WORKSPACE`` — diverging from every other component
(``mm_cli``, ``mcp.infra.workspace``, ``api.rest``). This module is the single
source of truth that closes that gap.

It adds one genuinely new rung the sibling resolvers lack: an upward walk for
the nearest ``mind-mem.json`` from the current directory, so a recall invoked
from a deep subdirectory auto-locates its workspace.

Pure / immutable: no global state, no side effects. Resolution and the
emptiness probe are independent so callers can resolve first (cheap) and probe
only when results are empty (potentially DB-touching).
"""

from __future__ import annotations

import os

__all__ = [
    "resolve_workspace",
    "probe_block_count",
    "WorkspaceHealth",
    "empty_workspace_warning",
]

# Marker file every backend keeps at the workspace root (see storage layer).
_CONFIG_FILENAME = "mind-mem.json"

# Backends whose blocks of record live on the local Markdown corpus. Kept in
# sync with ``mind_mem.storage._MARKDOWN_BACKENDS`` /
# ``mind_mem.mcp.infra.workspace._MARKDOWN_BACKENDS``.
_MARKDOWN_BACKENDS: frozenset[str] = frozenset({"markdown", "encrypted"})


def _find_config_upward(start: str) -> str | None:
    """Return the nearest ancestor dir containing ``mind-mem.json``, or None.

    Walks parent directories from *start* up to the filesystem root with a
    bounded loop (terminates when ``dirname`` stops changing). No globbing —
    a single ``os.path.dirname`` ascent.
    """
    cur = os.path.abspath(start)
    while True:
        if os.path.isfile(os.path.join(cur, _CONFIG_FILENAME)):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:  # reached filesystem root
            return None
        cur = parent


def resolve_workspace(cli_arg: str | None) -> str:
    """Resolve the recall workspace path to an absolute path.

    Resolution order (first hit wins):

    1. Explicit ``--workspace`` — only when actually passed (the argparse
       default is ``None``, so passing ``"."`` is distinguishable from unset).
    2. ``MIND_MEM_WORKSPACE`` environment variable.
    3. Nearest ``mind-mem.json`` walking upward from the current directory.
    4. The current working directory.

    Args:
        cli_arg: The value of ``--workspace`` if the user passed it, else None.

    Returns:
        An absolute workspace path.
    """
    if cli_arg is not None:
        return os.path.abspath(cli_arg)

    env_ws = os.environ.get("MIND_MEM_WORKSPACE", "").strip()
    if env_ws:
        return os.path.abspath(env_ws)

    discovered = _find_config_upward(os.getcwd())
    if discovered is not None:
        return discovered

    return os.path.abspath(os.getcwd())


class WorkspaceHealth:
    """Result of an emptiness probe over a resolved workspace.

    Attributes:
        workspace:  Absolute resolved workspace path.
        backend:    Detected ``block_store.backend``.
        configured: True if ``<workspace>/mind-mem.json`` exists.
        blocks:     Block count in the store (``-1`` when the probe could not
                    be performed — e.g. an unreachable Postgres).
        probe_error: Short message when the probe degraded, else None.
    """

    __slots__ = ("workspace", "backend", "configured", "blocks", "probe_error")

    def __init__(
        self,
        workspace: str,
        backend: str,
        configured: bool,
        blocks: int,
        probe_error: str | None = None,
    ) -> None:
        self.workspace = workspace
        self.backend = backend
        self.configured = configured
        self.blocks = blocks
        self.probe_error = probe_error

    @property
    def is_empty_or_unbuilt(self) -> bool:
        """True when the workspace is unconfigured or has zero blocks.

        A probe failure (``blocks == -1``) is NOT treated as empty — we must
        not turn an unreachable DB into a false "empty store" warning.
        """
        if not self.configured:
            return True
        return self.blocks == 0


def probe_block_count(workspace: str) -> WorkspaceHealth:
    """Probe how many active blocks the resolved *workspace* store holds.

    Backend-aware, read-only-safe, and never raises:

    * markdown / encrypted / sqlite → ``sqlite_index.index_status`` block count
      (read-only-safe), falling back to the markdown corpus enumeration.
    * any other backend (e.g. postgres) → ``storage.iter_active_blocks`` length
      (already backend-aware; used by scan / governance).

    A slow or unreachable store degrades to ``blocks == -1`` with a
    ``probe_error`` note rather than crashing the recall — so a flaky Postgres
    never turns a recall into a stack trace.
    """
    configured = os.path.isfile(os.path.join(workspace, _CONFIG_FILENAME))

    try:
        from .storage import _backend_name

        backend = _backend_name(workspace)
    except Exception:  # pragma: no cover - defensive: storage import/config
        backend = "markdown"

    if backend in _MARKDOWN_BACKENDS:
        try:
            from .sqlite_index import index_status

            status = index_status(workspace)
            blocks = int(status.get("blocks", 0))
            if blocks > 0 or status.get("schema_built", True) is not False:
                # Trust the index count when the FTS schema is built. When the
                # schema is not built yet the index reports 0 — fall through to
                # a corpus count so a freshly-written-but-unindexed corpus is
                # not mislabelled empty.
                if blocks > 0:
                    return WorkspaceHealth(workspace, backend, configured, blocks)
        except Exception as exc:  # pragma: no cover - defensive
            return WorkspaceHealth(workspace, backend, configured, -1, str(exc))

        # Index reported 0 (or unbuilt) — confirm against the on-disk corpus so
        # an unindexed-but-populated workspace is not falsely flagged empty.
        try:
            from .storage import iter_active_blocks

            return WorkspaceHealth(workspace, backend, configured, len(iter_active_blocks(workspace)))
        except Exception as exc:  # pragma: no cover - defensive
            return WorkspaceHealth(workspace, backend, configured, -1, str(exc))

    # Non-markdown backend: backend-aware block enumeration.
    try:
        from .storage import iter_active_blocks

        return WorkspaceHealth(workspace, backend, configured, len(iter_active_blocks(workspace)))
    except Exception as exc:
        return WorkspaceHealth(workspace, backend, configured, -1, str(exc))


def empty_workspace_warning(health: WorkspaceHealth) -> str:
    """Build the loud stderr warning for an empty / unconfigured workspace.

    Names the resolved absolute path, the detected backend, and the fix hint —
    so the operator can tell "wrong/empty/missing store" apart from a genuine
    0-of-N miss (which keeps the quiet ``No results found.`` on stdout).
    """
    if not health.configured:
        reason = f"no {_CONFIG_FILENAME} found at this workspace"
    else:
        reason = f"{health.backend} store is empty/unbuilt (0 blocks)"
    return (
        f"recall: WARNING — {reason}.\n"
        f"recall:   resolved workspace: {health.workspace}\n"
        f"recall:   backend: {health.backend}\n"
        f"recall:   fix: run `mind-mem-init {health.workspace}` "
        f"(or `mm doctor --rebuild-cache`), "
        f"or set MIND_MEM_WORKSPACE / pass --workspace <path>."
    )
