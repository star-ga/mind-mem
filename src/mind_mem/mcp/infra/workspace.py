"""Workspace resolution + path-safety helpers.

Extracted from ``mcp_server.py`` in the v3.2.0 §1.2 decomposition
(see docs/v3.2.0-mcp-decomposition-plan.md PR-1). These four
functions are the gateway between MCP tool calls and the on-disk
workspace — every tool that reads from or writes to a workspace
path funnels through here.

v3.2.1: workspace resolution respects a per-request ``ContextVar``
override before falling back to the process-wide
``MIND_MEM_WORKSPACE`` environment variable. This lets the REST
layer scope workspace selection to the request task without racing
against other concurrent requests that mutate shared process state.
The env var remains authoritative for the standalone MCP server.
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import os
from collections.abc import Iterator

_workspace_override: contextvars.ContextVar[str | None] = contextvars.ContextVar("mind_mem_workspace_override", default=None)


def _workspace() -> str:
    """Resolve workspace path.

    Resolution order:

    1. Per-request ``ContextVar`` override (set by :func:`use_workspace`).
    2. ``MIND_MEM_WORKSPACE`` environment variable.
    3. Current working directory.
    """
    override = _workspace_override.get()
    if override is not None:
        return os.path.abspath(override)
    ws = os.environ.get("MIND_MEM_WORKSPACE", ".")
    return os.path.abspath(ws)


@contextlib.contextmanager
def use_workspace(workspace: str) -> Iterator[str]:
    """Temporarily set the workspace override for the current context.

    ``contextvars.ContextVar`` is task-local under asyncio and
    thread-local when propagated through Starlette's thread pool,
    so concurrent REST requests cannot race on this value.

    Yields the resolved absolute workspace path.
    """
    resolved = os.path.abspath(workspace)
    token = _workspace_override.set(resolved)
    try:
        yield resolved
    finally:
        _workspace_override.reset(token)


def _check_workspace(ws: str) -> str | None:
    """Validate workspace exists and has expected structure.

    Returns ``None`` if valid, or an error JSON string if invalid.

    Backend-aware (audit bug #2): the legacy check hardcoded the Markdown
    corpus layout (a local ``decisions/`` directory) as the definition of
    a valid workspace, so all ws-gated MCP tools fail-closed on a
    Postgres-backed workspace where the blocks of record live in the DB,
    not in local Markdown files.

    Resolution:

    * The workspace root must exist (every backend keeps ``mind-mem.json``
      and the recall cache on disk).
    * **Markdown / encrypted backend** (the default, zero-config SQLite
      path) — keep the exact legacy behaviour: require the local
      ``decisions/`` directory. This leaves the default path byte-for-byte
      unchanged.
    * **Any other backend** (e.g. ``postgres``) — validate by probing the
      configured block store (``ping()`` when available, else
      ``list_blocks()``) instead of requiring ``decisions/`` on disk.

    Never raises: a store probe failure (unreachable DB, missing extra,
    misconfiguration) is converted to an error JSON string so the calling
    tool returns a structured error rather than crashing the server.
    """
    if not os.path.isdir(ws):
        return json.dumps({"error": "Workspace not found. Run: mind-mem-init <path>"})

    backend = _resolve_backend(ws)
    if backend in _MARKDOWN_BACKENDS:
        decisions_dir = os.path.join(ws, "decisions")
        if not os.path.isdir(decisions_dir):
            return json.dumps({"error": ("Workspace is missing the 'decisions/' directory. Run: mind-mem-init <path>")})
        return None

    return _check_store_backend(ws, backend)


# Backends whose blocks of record live on the local Markdown corpus, so a
# valid workspace must have the ``decisions/`` directory on disk. Every
# other backend (e.g. ``postgres``) keeps its blocks in the store and is
# validated by probing the store instead. Kept in sync with
# ``mind_mem.storage._MARKDOWN_BACKENDS``.
_MARKDOWN_BACKENDS: frozenset[str] = frozenset({"markdown", "encrypted"})


def _resolve_backend(ws: str) -> str:
    """Return the configured ``block_store.backend`` for *ws*.

    Routes through ``mind_mem.storage._backend_name`` (the single source
    of truth for backend detection) via a lazy import to keep this module
    import-cheap and free of import cycles. Degrades to ``"markdown"`` on
    any failure so the default path is never disturbed.
    """
    try:
        from ...storage import _backend_name
    except Exception:  # pragma: no cover - defensive: storage import failure
        return "markdown"
    try:
        return _backend_name(ws)
    except Exception:  # pragma: no cover - defensive: config read failure
        return "markdown"


def _check_store_backend(ws: str, backend: str) -> str | None:
    """Validate a non-Markdown workspace by probing its block store.

    Returns ``None`` when the store is reachable and provisioned, else an
    error JSON string. Prefers the store's ``ping()`` health probe (which
    never raises and reports a structured status); falls back to
    ``list_blocks()`` for stores without ``ping()``.
    """
    try:
        from ...storage import get_block_store

        store = get_block_store(ws)
    except Exception as exc:
        return json.dumps(
            {
                "error": (
                    f"Workspace block store ({backend!r}) could not be initialised: {exc}. Check block_store config in mind-mem.json."
                ),
            }
        )

    ping = getattr(store, "ping", None)
    if callable(ping):
        try:
            status = ping()
        except Exception as exc:  # defensive: ping is documented never to raise
            return json.dumps({"error": (f"Workspace block store ({backend!r}) is unreachable: {exc}.")})
        if isinstance(status, dict) and not status.get("ok", False):
            detail = status.get("error") or "store reported not ok"
            return json.dumps({"error": (f"Workspace block store ({backend!r}) is not ready: {detail}.")})
        return None

    # No ping(): fall back to a lightweight list_blocks() probe.
    try:
        store.list_blocks()
    except Exception as exc:
        return json.dumps({"error": (f"Workspace block store ({backend!r}) is unreachable: {exc}.")})
    return None


def _validate_path(ws: str, rel_path: str) -> str:
    """Validate that rel_path resolves inside workspace. Returns resolved path.

    Raises ValueError if the path escapes the workspace boundary OR
    if any component of the resulting path is a symlink (audit S-10
    — symlinks open a TOCTOU window between validation and open()).

    The walk uses ``lstat``/``islink`` per-component, so even
    workspace-internal symlinks are rejected: the link itself can be
    re-pointed between checks. Callers that need to follow a symlink
    must resolve it explicitly out-of-band.
    """
    if not isinstance(rel_path, str) or "\x00" in rel_path:
        raise ValueError("Invalid path: empty or contains NUL byte")
    ws_real = os.path.realpath(ws)
    joined = os.path.join(ws_real, rel_path)
    abs_joined = os.path.abspath(joined)

    # Walk parent components within the workspace and reject any
    # symlink. We start the walk from ws_real so a symlink that points
    # at the workspace itself is allowed (the workspace root realpath
    # is the authority), but every component *below* it is checked.
    rel_inside = os.path.relpath(abs_joined, ws_real)
    if rel_inside == "." or rel_inside == "":
        # Asking for the workspace itself — no further symlink check.
        return ws_real

    cursor = ws_real
    for component in rel_inside.split(os.sep):
        if component in ("", "."):
            continue
        cursor = os.path.join(cursor, component)
        if os.path.islink(cursor):
            raise ValueError(f"Invalid path: symlink in component {component!r} (audit S-10 — symlinks defeat TOCTOU guard)")
        if not os.path.exists(cursor):
            # Allow non-existent tail (callers may want to create it).
            # We've already verified every existing component is not a
            # symlink, so a non-existent component cannot be subverted.
            break

    path = os.path.realpath(abs_joined)
    if path != ws_real and not path.startswith(ws_real + os.sep):
        raise ValueError("Invalid path: escapes workspace")
    return path


def _read_file(rel_path: str) -> str:
    """Read a file from workspace, return contents or error message."""
    ws = _workspace()
    try:
        path = _validate_path(ws, rel_path)
    except ValueError:
        return "Error: path escapes workspace"
    if not os.path.isfile(path):
        return f"File not found: {rel_path}"
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
