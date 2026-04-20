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

    Returns None if valid, or an error JSON string if invalid.
    """
    if not os.path.isdir(ws):
        return json.dumps({"error": "Workspace not found. Run: mind-mem-init <path>"})
    decisions_dir = os.path.join(ws, "decisions")
    if not os.path.isdir(decisions_dir):
        return json.dumps({"error": ("Workspace is missing the 'decisions/' directory. Run: mind-mem-init <path>")})
    return None


def _validate_path(ws: str, rel_path: str) -> str:
    """Validate that rel_path resolves inside workspace. Returns resolved path.

    Raises ValueError if the path escapes the workspace boundary.
    """
    ws_real = os.path.realpath(ws)
    path = os.path.realpath(os.path.join(ws_real, rel_path))
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
