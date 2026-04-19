"""Workspace resolution + path-safety helpers.

Extracted from ``mcp_server.py`` in the v3.2.0 §1.2 decomposition
(see docs/v3.2.0-mcp-decomposition-plan.md PR-1). These four
functions are the gateway between MCP tool calls and the on-disk
workspace — every tool that reads from or writes to a workspace
path funnels through here.
"""

from __future__ import annotations

import json
import os


def _workspace() -> str:
    """Resolve workspace path from environment."""
    ws = os.environ.get("MIND_MEM_WORKSPACE", ".")
    return os.path.abspath(ws)


def _check_workspace(ws: str) -> str | None:
    """Validate workspace exists and has expected structure.

    Returns None if valid, or an error JSON string if invalid.
    """
    if not os.path.isdir(ws):
        return json.dumps({"error": "Workspace not found. Run: mind-mem-init <path>"})
    decisions_dir = os.path.join(ws, "decisions")
    if not os.path.isdir(decisions_dir):
        return json.dumps(
            {
                "error": (
                    "Workspace is missing the 'decisions/' directory. "
                    "Run: mind-mem-init <path>"
                )
            }
        )
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
