# Copyright 2026 STARGA, Inc.
"""Register (and unregister) mind-mem as an MCP server in each client.

Generic over :data:`mind_mem.client_registry.AGENT_REGISTRY`: the spec says
which file(s) and which dialect, :mod:`mind_mem.client_writers` does the
edit, :mod:`mind_mem.client_config_io` does safe reads and atomic writes.

Rules every write follows:

* only the mind-mem entry changes; other servers, keys and (for TOML and
  YAML) comments are preserved;
* a file that cannot be merged safely (JSONC, invalid, unreadable) is left
  byte-identical and the snippet to paste is returned;
* re-running is idempotent: an unchanged file is not rewritten;
* a changed file is backed up first and replaced atomically;
* nothing is written outside the client's documented location, and guarded
  clients are written only into a directory the client already created.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

from mind_mem.client_config_io import UnsafeConfigError, atomic_write_text, read_json_object, read_text_file, write_config
from mind_mem.client_paths import expand
from mind_mem.client_registry import AGENT_REGISTRY, AgentSpec
from mind_mem.client_restore import is_skeleton, matching_backup_path
from mind_mem.client_writers import JSON_FORMATS, JSON_WRITERS, TEXT_FORMATS, json_entry, remove_json_mcp, remove_opencode_mcp

NO_MCP_REASON = "agent does not support native MCP"


def _mcp_server_path() -> str:
    """Locate the mcp_server.py launcher on this machine.

    Preference order:
    1. ``MIND_MEM_MCP_SERVER`` env var (explicit override)
    2. ``<package_dir>/../mcp_server.py``  (src-layout checkout)
    3. ``<package_dir>/mcp_server.py``     (packaged install)
    """
    override = os.environ.get("MIND_MEM_MCP_SERVER")
    if override and os.path.isfile(override):
        return os.path.abspath(override)
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    for candidate in (
        os.path.normpath(os.path.join(pkg_dir, "..", "..", "mcp_server.py")),
        os.path.normpath(os.path.join(pkg_dir, "..", "mcp_server.py")),
        os.path.join(pkg_dir, "mcp_server.py"),
    ):
        if os.path.isfile(candidate):
            return candidate
    return "mcp_server.py"  # fallback -- caller must ensure PATH


def mcp_server_spec(workspace: str) -> dict[str, Any]:
    """Canonical MCP-server invocation record: ``{command, args, env}``."""
    return {
        "command": sys.executable or "python3",
        "args": [_mcp_server_path()],
        "env": {"MIND_MEM_WORKSPACE": workspace},
    }


def refused_result(agent: str, path: str, snippet: Any, reason: str) -> dict:
    """Nothing written; ``content`` is the stanza to paste, ``reason`` why."""
    text = json.dumps(snippet, indent=2) if isinstance(snippet, dict) else snippet
    return {"agent": agent, "path": path, "written": False, "content": text, "merged": False, "skipped": True, "reason": reason}


def write_result(agent: str, path: str, serialised: str, merged: bool) -> dict:
    """Back up an existing file, write atomically, and describe the write."""
    text = serialised if serialised.endswith("\n") else serialised + "\n"
    backup = write_config(path, text)
    result = {"agent": agent, "path": path, "written": True, "content": serialised, "merged": merged, "skipped": False}
    if backup is not None:
        result["backup"] = backup
    return result


def _snippet(spec: AgentSpec, workspace: str, srv: dict[str, Any]) -> str:
    """What a fresh install would write -- used for refusals and manual steps."""
    fmt = spec.mcp_fmt
    if fmt in JSON_WRITERS:
        return json.dumps(JSON_WRITERS[fmt]({}, workspace, srv)[0], indent=2)
    if fmt in TEXT_FORMATS:
        return TEXT_FORMATS[fmt][0]("", workspace, srv)[0]
    if fmt == "mcp-json-opencode":
        from mind_mem.opencode_config import manual_snippet  # noqa: PLC0415

        return manual_snippet(srv)
    return json.dumps({"mcpServers": {"mind-mem": json_entry("mcp-json-servers", srv)}}, indent=2)


def _targets(spec: AgentSpec, workspace: str) -> list[tuple[str, bool]]:
    """``(path, required)`` pairs: the primary, then extra locations. An
    extra (``required=False``) is written only when its directory exists."""
    primary = spec.expand_mcp_path(workspace)
    out = [(primary, True)]
    for tmpl in spec.mcp_extra_paths:
        path = expand(tmpl, workspace)
        if path != primary and all(path != p for p, _ in out):
            out.append((path, False))
    return out


def _install_one(spec: AgentSpec, path: str, workspace: str, srv: dict[str, Any], *, dry_run: bool, force: bool) -> dict:
    agent, fmt = spec.name, spec.mcp_fmt
    if fmt == "mcp-json-opencode":
        from mind_mem.opencode_config import install_opencode_mcp  # noqa: PLC0415

        return install_opencode_mcp(path, srv, agent=agent, dry_run=dry_run, force=force)
    if fmt in JSON_WRITERS:
        # Always load the existing config, `force` included: `force` means
        # "write even if nothing changed", never "discard every other key".
        try:
            loaded = read_json_object(path)
            merged_doc, changed = JSON_WRITERS[fmt](loaded if loaded is not None else {}, workspace, srv)
        except UnsafeConfigError as exc:
            return refused_result(agent, path, _snippet(spec, workspace, srv), str(exc))
        serialised = json.dumps(merged_doc, indent=2)
    elif fmt in TEXT_FORMATS:
        try:
            serialised, changed = TEXT_FORMATS[fmt][0](read_text_file(path) or "", workspace, srv)
        except UnsafeConfigError as exc:
            return refused_result(agent, path, _snippet(spec, workspace, srv), str(exc))
    else:
        raise ValueError(f"unknown mcp_fmt {fmt!r} for agent {agent!r}")
    merged = os.path.isfile(path) and not force
    skipped = merged and not changed
    if dry_run:
        return {"agent": agent, "path": path, "written": False, "content": serialised, "merged": merged, "skipped": skipped}
    if skipped:
        return {"agent": agent, "path": path, "written": False, "content": serialised, "merged": False, "skipped": True}
    return write_result(agent, path, serialised, merged)


def _guard_blocks(spec: AgentSpec, path: str) -> str:
    guard = spec.mcp_guarded()
    if guard == "dir" and not os.path.isdir(os.path.dirname(path)):
        return f"client config directory {os.path.dirname(path)} not found; install or run the client once, then re-run"
    if guard == "file" and not os.path.isfile(path):
        return f"{path} not found; the client creates it on first run -- add the snippet by hand or re-run afterwards"
    return ""


def install_mcp_config(agent: str, workspace: str, *, dry_run: bool = False, force: bool = False) -> dict:
    """Register mind-mem as an MCP server in *agent*'s native config.

    Returns the result for the primary location; results for further
    documented locations that were present go in ``also``. A client with
    no writable file returns ``skipped`` with the steps in ``content``.
    """
    spec = AGENT_REGISTRY.get(agent)
    if spec is None:
        raise ValueError(f"unknown agent: {agent!r}")
    srv = mcp_server_spec(workspace)
    if not spec.mcp_fmt:
        if spec.manual:
            steps = f"Manual setup for {spec.description}: {spec.manual}\n\n{_snippet(spec, workspace, srv)}"
            return {
                "agent": agent,
                "path": "",
                "written": False,
                "merged": False,
                "skipped": True,
                "reason": "manual setup required",
                "content": steps,
            }
        return {"agent": agent, "skipped": True, "reason": NO_MCP_REASON, "written": False}
    if not spec.supported_here():
        path = expand(spec.mcp_path_tmpl, workspace)
        return refused_result(agent, path, _snippet(spec, workspace, srv), f"{spec.description} is not available on this platform")

    primary: dict | None = None
    also: list[dict] = []
    for path, required in _targets(spec, workspace):
        if not required:
            if os.path.isdir(os.path.dirname(path)):
                also.append(_install_one(spec, path, workspace, srv, dry_run=dry_run, force=force))
            continue
        blocked = _guard_blocks(spec, path)
        primary = (
            refused_result(agent, path, _snippet(spec, workspace, srv), blocked)
            if blocked
            else _install_one(spec, path, workspace, srv, dry_run=dry_run, force=force)
        )
    assert primary is not None
    if also:
        primary["also"] = also
    return primary


def _uninstall_one(spec: AgentSpec, path: str, *, dry_run: bool) -> dict:
    agent, fmt = spec.name, spec.mcp_fmt
    base = {"agent": agent, "path": path, "written": False, "removed": False}
    if not os.path.isfile(path):
        return {**base, "reason": "no config file"}
    try:
        if fmt in JSON_FORMATS or fmt == "mcp-json-opencode":
            loaded = read_json_object(path) or {}
            if fmt == "mcp-json-opencode":
                content, changed = remove_opencode_mcp(loaded)
            else:
                content, changed = remove_json_mcp(JSON_FORMATS[fmt][0], loaded)
            serialised = json.dumps(content, indent=2)
        elif fmt in TEXT_FORMATS:
            serialised, changed = TEXT_FORMATS[fmt][1](read_text_file(path) or "")
        else:
            raise ValueError(f"unknown mcp_fmt {fmt!r} for agent {agent!r}")
    except UnsafeConfigError as exc:
        return {**base, "skipped": True, "reason": f"{exc}; remove the mind-mem entry by hand"}
    if not changed:
        return {**base, "reason": "no mind-mem entry"}
    if dry_run:
        return {**base, "removed": True, "content": serialised}
    # Give back exactly what the user had: their pre-install bytes when a
    # backup holds the same document, or no file when mind-mem created it.
    found = matching_backup_path(fmt, path, serialised)
    if found is not None:
        backup, original = found
        atomic_write_text(os.path.realpath(path), original)  # exact bytes: no newline normalisation
        # The file now holds that backup's exact bytes, so the backup is redundant.
        os.unlink(backup)
        return {**base, "written": True, "removed": True, "restored": True, "content": original, "merged": True, "skipped": False}
    if is_skeleton(fmt, serialised):
        os.unlink(os.path.realpath(path))
        return {**base, "removed": True, "deleted": True, "content": ""}
    return {**write_result(agent, path, serialised, merged=True), "removed": True}


def uninstall_mcp_config(agent: str, workspace: str, *, dry_run: bool = False) -> dict:
    """Remove only the mind-mem entry from *agent*'s MCP config file(s)."""
    spec = AGENT_REGISTRY.get(agent)
    if spec is None:
        raise ValueError(f"unknown agent: {agent!r}")
    if not spec.mcp_fmt:
        return {
            "agent": agent,
            "written": False,
            "removed": False,
            "reason": NO_MCP_REASON if not spec.manual else "manual setup: remove the entry in the client",
        }
    results = [_uninstall_one(spec, path, dry_run=dry_run) for path, _ in _targets(spec, workspace)]
    primary = results[0]
    if len(results) > 1:
        primary["also"] = results[1:]
    return primary


__all__ = [
    "NO_MCP_REASON",
    "install_mcp_config",
    "mcp_server_spec",
    "refused_result",
    "uninstall_mcp_config",
    "write_result",
]
