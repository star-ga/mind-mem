# Copyright 2026 STARGA, Inc.
"""OpenCode (V1 and V2) MCP registration writer.

OpenCode reads ``opencode.json`` / ``opencode.jsonc`` from
``~/.config/opencode/`` (global) and from the project tree. Two MCP shapes
exist:

* **V1 / compatible** — servers directly under ``mcp``::

      {"mcp": {"mind-mem": {"type": "local", "command": [...],
                            "environment": {...}, "enabled": true}}}

  OpenCode 1.x reads this, and OpenCode 2.x normalises it in memory
  without rewriting the file, so it is the shape that works on both.

* **V2 native** — servers grouped under ``mcp.servers``::

      {"mcp": {"servers": {"mind-mem": {"type": "local", "command": [...],
                                        "environment": {...}}}}}

  OpenCode 1.x cannot read this shape.

The writer keeps whichever dialect the file already uses: if the user's
config already holds a native ``mcp.servers`` map (the file is V2-only
already), the entry goes there; otherwise the V1-compatible shape is
written so the same file keeps working on either major version.

Safety rules, stricter than the generic JSON writers because OpenCode
configs are commonly hand-edited JSONC:

* A file that is not strict JSON is never rewritten. JSONC comments would
  be lost and an unparseable file would be truncated, so the result
  reports ``skipped`` with a reason and the snippet to paste by hand.
* Before an existing file is changed, a timestamped copy is written next
  to it (``<file>.bak-mind-mem-YYYYmmdd-HHMMSS``).
* Writes are atomic (temp file + ``os.replace``).
* Re-running with an identical entry changes nothing and writes no backup.
"""

from __future__ import annotations

import json
import os
from typing import Any

from mind_mem.client_config_io import atomic_write_text, backup_file

SERVER_NAME = "mind-mem"


def resolve_config_file(path: str) -> str:
    """Prefer an existing ``opencode.jsonc`` when ``opencode.json`` is absent.

    OpenCode accepts either extension in the same directory. Writing a new
    ``.json`` beside a user's ``.jsonc`` would split their config in two, so
    target the file they already have.
    """
    if path.endswith(".json") and not os.path.exists(path):
        jsonc = path + "c"
        if os.path.isfile(jsonc):
            return jsonc
    return path


def _command(srv: dict[str, Any]) -> list[str]:
    return [str(srv["command"]), *[str(a) for a in srv.get("args", [])]]


def legacy_entry(srv: dict[str, Any]) -> dict[str, Any]:
    """V1-compatible entry (read by OpenCode 1.x and 2.x)."""
    return {
        "type": "local",
        "command": _command(srv),
        "environment": dict(srv.get("env", {})),
        "enabled": True,
    }


def native_entry(srv: dict[str, Any]) -> dict[str, Any]:
    """OpenCode 2.x native ``mcp.servers`` entry."""
    return {
        "type": "local",
        "command": _command(srv),
        "environment": dict(srv.get("env", {})),
    }


def _uses_native_servers(mcp: dict[str, Any]) -> bool:
    """True when ``mcp.servers`` is a V2 server map, not a V1 server named
    "servers" (a V1 entry carries its own ``type``)."""
    servers = mcp.get("servers")
    return isinstance(servers, dict) and "type" not in servers


def merge_opencode_mcp(existing: dict[str, Any], srv: dict[str, Any]) -> tuple[dict[str, Any], bool, str]:
    """Return ``(new_config, changed, dialect)``; never mutates *existing*.

    ``dialect`` is ``"v2-native"`` or ``"v1-compatible"``.
    Raises ``ValueError`` when ``mcp`` exists but is not an object.
    """
    out = json.loads(json.dumps(existing))
    mcp = out.setdefault("mcp", {})
    if not isinstance(mcp, dict):
        raise ValueError("existing `mcp` value is not an object")

    if _uses_native_servers(mcp):
        target = native_entry(srv)
        servers = mcp["servers"]
        changed = servers.get(SERVER_NAME) != target
        servers[SERVER_NAME] = target
        # A V1 duplicate beside the native one would make OpenCode 2.x
        # report a conflict for the same name; the native entry wins.
        if SERVER_NAME in mcp:
            del mcp[SERVER_NAME]
            changed = True
        return out, changed, "v2-native"

    target = legacy_entry(srv)
    changed = mcp.get(SERVER_NAME) != target
    mcp[SERVER_NAME] = target
    return out, changed, "v1-compatible"


def manual_snippet(srv: dict[str, Any]) -> str:
    """The stanza a user can paste into a JSONC file by hand."""
    return json.dumps({"mcp": {SERVER_NAME: legacy_entry(srv)}}, indent=2)


def _backup(path: str) -> str:
    return backup_file(path)


def _atomic_write(path: str, text: str) -> None:
    atomic_write_text(path, text)


def install_opencode_mcp(
    path: str,
    srv: dict[str, Any],
    *,
    agent: str = "opencode",
    dry_run: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Register mind-mem in an OpenCode config file.

    Returns the same result keys as ``hook_installer.install_mcp_config``
    plus ``dialect`` and, when a backup was taken, ``backup``. ``force``
    rewrites even when nothing changed but never discards user keys and
    never rewrites a file that is not strict JSON.
    """
    path = resolve_config_file(path)
    exists = os.path.isfile(path)
    existing: dict[str, Any] = {}
    if exists:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                raw = fh.read()
        except OSError as exc:
            return _refused(agent, path, srv, f"cannot read config: {exc}")
        if raw.strip():
            try:
                loaded = json.loads(raw)
            except json.JSONDecodeError:
                return _refused(
                    agent,
                    path,
                    srv,
                    "config is JSONC (comments or trailing commas) or not valid JSON; left untouched — add the snippet under `mcp` by hand",
                )
            if not isinstance(loaded, dict):
                return _refused(agent, path, srv, "config root is not a JSON object; left untouched")
            existing = loaded

    try:
        content, changed, dialect = merge_opencode_mcp(existing, srv)
    except ValueError as exc:
        return _refused(agent, path, srv, f"{exc}; left untouched")

    serialised = json.dumps(content, indent=2) + "\n"
    result: dict[str, Any] = {
        "agent": agent,
        "path": path,
        "written": False,
        "content": serialised,
        "merged": exists and not force,
        "skipped": exists and not changed and not force,
        "dialect": dialect,
    }
    if dry_run or result["skipped"]:
        if result["skipped"]:
            result["merged"] = False
        return result

    # Write through a symlinked config (dotfile managers) instead of
    # replacing the link with a regular file.
    target = os.path.realpath(path)
    if exists:
        result["backup"] = _backup(target)
    _atomic_write(target, serialised)
    result["written"] = True
    return result


def _refused(agent: str, path: str, srv: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "agent": agent,
        "path": path,
        "written": False,
        "merged": False,
        "skipped": True,
        "reason": reason,
        "content": manual_snippet(srv),
    }


__all__ = [
    "SERVER_NAME",
    "install_opencode_mcp",
    "legacy_entry",
    "manual_snippet",
    "merge_opencode_mcp",
    "native_entry",
    "resolve_config_file",
]
