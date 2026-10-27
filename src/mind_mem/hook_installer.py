# Copyright 2026 STARGA, Inc.
"""Agent hook auto-capture installer + event schema (v2.6.0, v2.7.0).

Ships the hook-event JSON Schema, a privacy filter that strips
secrets before persistence, an observation → block pipeline, and a
cross-agent installer that writes the right config file for each
target CLI.

The hooks themselves are plain shell + JSON — this module is the
Python surface that keeps them in sync between installs and that
gives downstream code a single validated hook-event shape to consume.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from mind_mem.client_config_io import UnsafeConfigError, read_json_object, read_text_file
from mind_mem.client_paths import expand

# ---------------------------------------------------------------------------
# Hook event schema
# ---------------------------------------------------------------------------


HOOK_EVENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["type", "timestamp", "project", "session_id"],
    "properties": {
        "type": {
            "type": "string",
            "enum": [
                "SessionStart",
                "PostToolUse",
                "PreCompact",
                "SessionEnd",
                "UserPromptSubmit",
                "Stop",
                "Notification",
                "PreToolUse",
                "SubagentStop",
                "PostUserMessage",
                "OnError",
                "OnWarning",
            ],
        },
        "timestamp": {"type": "string"},
        "tool": {"type": ["string", "null"]},
        "input_hash": {"type": ["string", "null"]},
        "output_summary": {"type": ["string", "null"]},
        "project": {"type": "string"},
        "session_id": {"type": "string"},
    },
}


@dataclass
class HookEvent:
    type: str
    timestamp: str
    project: str
    session_id: str
    tool: Optional[str] = None
    input_hash: Optional[str] = None
    output_summary: Optional[str] = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "timestamp": self.timestamp,
            "tool": self.tool,
            "input_hash": self.input_hash,
            "output_summary": self.output_summary,
            "project": self.project,
            "session_id": self.session_id,
        }


def validate_event(event: Mapping[str, Any]) -> list[str]:
    """Lightweight schema check — no `jsonschema` dependency."""
    errors: list[str] = []
    required = HOOK_EVENT_SCHEMA["required"]
    for key in required:
        if key not in event:
            errors.append(f"missing required field: {key!r}")
    allowed_types = HOOK_EVENT_SCHEMA["properties"]["type"]["enum"]
    if event.get("type") not in allowed_types:
        errors.append(f"invalid type: {event.get('type')!r}")
    return errors


# ---------------------------------------------------------------------------
# Privacy filter
# ---------------------------------------------------------------------------


_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}"),  # Anthropic API keys
    re.compile(r"xai-[A-Za-z0-9]{20,}"),  # xAI API keys
    re.compile(r"sk-[A-Za-z0-9]{20,}"),  # generic secret-key prefix
    re.compile(r"pypi-[A-Za-z0-9_\-]{20,}"),  # PyPI macaroons
    re.compile(r"AKIA[0-9A-Z]{16}"),  # AWS access key id
    re.compile(r"AIza[0-9A-Za-z_\-]{35}"),  # Google API key
    re.compile(r"ghp_[A-Za-z0-9]{36}"),  # GitHub personal token
    re.compile(r"xox[baprs]-[A-Za-z0-9\-]{10,}"),  # Slack tokens
    re.compile(r"<private>[\s\S]*?</private>"),
)


def privacy_filter(text: str) -> str:
    """Redact common credentials + ``<private>`` blocks before persistence."""
    if not text:
        return text
    out = text
    for pat in _SECRET_PATTERNS:
        out = pat.sub("[REDACTED]", out)
    return out


# ---------------------------------------------------------------------------
# Observation → block pipeline
# ---------------------------------------------------------------------------


def observation_to_block(
    event: Mapping[str, Any],
    *,
    window_seconds: int = 300,
    seen_hashes: Optional[set[str]] = None,
) -> Optional[dict]:
    """Turn a hook event into a structured block dict.

    Performs 5-minute SHA-256 dedup (the ``seen_hashes`` set is the
    caller's responsibility to keep window-scoped) plus privacy
    filtering, and returns None when the event should be dropped
    (duplicate or empty after filter).
    """
    if validate_event(event):
        return None
    raw = event.get("output_summary") or event.get("input_hash") or ""
    filtered = privacy_filter(str(raw))
    if not filtered.strip():
        return None
    digest = hashlib.sha256(filtered.encode("utf-8")).hexdigest()[:16]
    if seen_hashes is not None:
        if digest in seen_hashes:
            return None
        seen_hashes.add(digest)
    return {
        "_id": f"OBS-{event.get('session_id', 'unknown')}-{digest}",
        "type": "observation",
        "tool": event.get("tool"),
        "summary": filtered,
        "timestamp": event.get("timestamp", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
        "project": event.get("project", ""),
        "session_id": event.get("session_id", ""),
        "quality_score": min(100, max(0, 40 + len(filtered) // 16)),
    }


# ---------------------------------------------------------------------------
# Installer — registry-driven for all supported AI clients
# ---------------------------------------------------------------------------
#
# The registry (AgentSpec + AGENT_REGISTRY) lives in client_registry /
# client_registry_data, the MCP dialect writers in client_writers and MCP
# registration in client_mcp. They are re-exported here so existing imports
# keep working.


import glob as _glob
import shutil as _shutil

from mind_mem.client_mcp import (
    NO_MCP_REASON,
    install_mcp_config,
    mcp_server_spec,
    refused_result,
    uninstall_mcp_config,
    write_result,
)
from mind_mem.client_registry import _MM_MARKER, AGENT_REGISTRY, MEMORY_PROTOCOL_SNIPPET, AgentSpec
from mind_mem.client_writers import JSON_WRITERS as _MCP_WRITERS_JSON  # noqa: F401 (re-export)
from mind_mem.client_writers import TEXT_WRITERS as _MCP_WRITERS_TEXT  # noqa: F401 (re-export)

# ---------------------------------------------------------------------------
# Format writers — each (existing_text_or_dict, workspace) → (new, merged,
# skipped). Workspace is passed so format writers can render workspace paths.
# ---------------------------------------------------------------------------


def _merge_claude_hooks(existing: dict, workspace: str) -> tuple[dict, bool]:
    """Idempotent hook merge for Claude Code settings.json.

    Claude Code's hook schema requires each event entry to be
    ``{"matcher": "<pattern>", "hooks": [{"type": "command",
    "command": "..."}]}`` — NOT the bare ``{"command": "..."}``
    shape this installer used prior to v1.9.2. Bare entries would
    silently be accepted into settings.json but Claude Code would
    report them as "protected file can't be edited directly" style
    errors at runtime and continuously block Stop / PostToolUse.
    v1.9.2 writes the nested shape AND detects the legacy flat
    shape so re-running install does not produce duplicates.

    Commands installed (5.0.2):
      SessionStart → ``mm status`` — prints workspace status JSON;
          Claude Code picks it up at session start as context.
      SessionStart → ``mm resume-on-start`` — the active task frame's
          resume brief, data-framed (see :mod:`mind_mem.data_marking`)
          because what it prints goes straight into a system prompt.
      Stop        → ``mm status`` — final-state log on session end.

    The prior ``PostToolUse → mm capture --stdin`` hook has been
    removed: ``mm capture`` is not yet a CLI subcommand (only exists
    in the design doc). PostToolUse fires after every tool call, so
    installing a broken command there produced a cascading error
    loop that blocked further edits. Will be re-added as
    ``mm capture --stdin`` once that subcommand ships.

    The prior ``SessionStart → mm inject --workspace X`` hook had a
    latent bug: ``mm inject`` requires a positional query argument
    that SessionStart cannot provide. It was swapped to ``mm status``
    with a note promising a future query-free verb. That verb now
    exists as ``mm resume-on-start`` and is installed here beside
    ``mm status``, so a session again starts with actual memory in
    front of it rather than a status blob. It takes no arguments and
    exits 0 unconditionally, which is what makes it safe on a hook.
    """
    out = json.loads(json.dumps(existing))
    hooks = out.setdefault("hooks", {})
    wanted = [
        ("SessionStart", "mm status"),
        ("SessionStart", "mm resume-on-start"),
        ("Stop", "mm status"),
    ]
    changed = False
    for event, command in wanted:
        entries = hooks.setdefault(event, [])
        if not isinstance(entries, list):
            continue
        # Idempotency: match either the new nested shape OR the legacy
        # bare-command shape (pre-v1.9.2 installs) so re-running does
        # not duplicate. Also migrate any legacy entries in-place.
        already = False
        for i, e in enumerate(entries):
            if not isinstance(e, dict):
                continue
            # Legacy flat shape — replace in place.
            if "hooks" not in e and e.get("command") == command:
                entries[i] = _nested_hook(command)
                already = True
                changed = True
                break
            # New nested shape — check inner commands.
            for inner in e.get("hooks", []):
                if isinstance(inner, dict) and inner.get("command") == command:
                    already = True
                    break
            if already:
                break
        if not already:
            entries.append(_nested_hook(command))
            changed = True
    return out, changed


def _nested_hook(command: str, matcher: str = "") -> dict:
    """Return the Claude-Code-required nested hook entry shape."""
    return {
        "matcher": matcher,
        "hooks": [{"type": "command", "command": command}],
    }


def _merge_openclaw_hooks(existing: dict, workspace: str) -> tuple[dict, bool]:
    """OpenClaw / Nanoclaw / Nemoclaw hook registry merge.

    All three share the claw-family hooks JSON shape at
    ``.hooks.internal.entries.mind-mem`` so we reuse this merger.
    """
    out = json.loads(json.dumps(existing))
    hooks = out.setdefault("hooks", {}).setdefault("internal", {}).setdefault("entries", {})
    # v1.9.2: `mm capture` and `mm vault status` are not CLI
    # subcommands (only exist in the design doc). Use `mm status`
    # until those ship. `mm inject` still requires a query positional;
    # openclaw passes one when it fires the hook, so the inject entry
    # supplies only the flags.
    #
    # Two flags were wrong and made the hook fail on every fire:
    # `--agent openclaw` is not a KNOWN agent (raises UnknownAgentError;
    # valid = claude-code/codex/gemini/cursor/windsurf/aider/generic) and
    # `mm inject` has no `--workspace` flag (argparse rejects it). The
    # workspace resolves from the `MIND_MEM_WORKSPACE` env openclaw
    # exports to hook children (and the entry `workspace` field), so
    # neither flag is needed — `--agent generic` is the correct tag.
    target = {
        "enabled": True,
        "workspace": workspace,
        "commands": {
            "inject": "mm inject --agent generic",
            "capture": "mm status",
            "status": "mm status",
        },
    }
    changed = hooks.get("mind-mem") != target
    if changed:
        hooks["mind-mem"] = target
    return out, changed


def _merge_gemini(existing: dict, workspace: str) -> tuple[dict, bool]:
    """Gemini settings.json system_instruction injection."""
    instr = f'mind-mem workspace: {workspace}; run `mm inject --agent gemini "<question>"` before answering.'
    out = dict(existing)
    changed = out.get("system_instruction") != instr
    out["system_instruction"] = instr
    return out, changed


def _merge_continue(existing: dict, workspace: str) -> tuple[dict, bool]:
    """Continue.dev config.json: inject a systemMessage."""
    out = json.loads(json.dumps(existing))
    sys_msg = f'mind-mem workspace: {workspace}. Run `mm inject --agent continue "<question>"` before composing responses.'
    if out.get("systemMessage") == sys_msg:
        return out, False
    out["systemMessage"] = sys_msg
    return out, True


def _merge_zed(existing: dict, workspace: str) -> tuple[dict, bool]:
    """Zed settings.json: inject assistant default_model_instructions."""
    out = json.loads(json.dumps(existing))
    assistant = out.setdefault("assistant", {})
    sys_msg = f'mind-mem workspace: {workspace}. Use `mm inject --agent zed "<question>"` for context.'
    if assistant.get("default_system_message") == sys_msg:
        return out, False
    assistant["default_system_message"] = sys_msg
    return out, True


def _merge_generic_json(existing: dict, workspace: str) -> tuple[dict, bool]:
    """Fallback JSON merge — sets {"mind_mem": {...}} key."""
    target = {
        "enabled": True,
        "workspace": workspace,
        "inject_cmd": "mm inject --agent generic",
    }
    out = dict(existing)
    if out.get("mind_mem") == target:
        return out, False
    out["mind_mem"] = target
    return out, True


_JSON_MERGERS: dict[str, Any] = {
    "json-claude-hooks": _merge_claude_hooks,
    "json-openclaw-hooks": _merge_openclaw_hooks,
    "json-gemini": _merge_gemini,
    "json-continue": _merge_continue,
    "json-zed": _merge_zed,
    "json-generic": _merge_generic_json,
}


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def _path_exists(pattern: str, workspace: str) -> bool:
    expanded = expand(pattern, workspace)
    if "*" in expanded:
        return bool(_glob.glob(expanded))
    return os.path.exists(expanded)


def detect_installed_agents(workspace: str) -> list[str]:
    """Return the names of agents whose binary is on PATH or whose
    config dir exists. Always includes agents flagged ``always_offer``
    (typically GitHub Copilot, which is near-universal).

    Auto-detection is non-invasive: no processes spawned, no network
    calls. Just :func:`os.path.exists` + :func:`shutil.which`.
    """
    out: list[str] = []
    for spec in AGENT_REGISTRY.values():
        hit = False
        for bin_name in spec.detect_binaries:
            if _shutil.which(bin_name):
                hit = True
                break
        if not hit:
            for p in spec.detect_paths:
                if _path_exists(p, workspace):
                    hit = True
                    break
        if not hit and spec.always_offer:
            hit = True
        if hit:
            out.append(spec.name)
    return out


# ---------------------------------------------------------------------------
# Install (single + all)
# ---------------------------------------------------------------------------


_refused_result = refused_result
_write_result = write_result


def install_config(
    agent: str,
    workspace: str,
    *,
    dry_run: bool = False,
    force: bool = False,
) -> dict:
    """Generate (and optionally write) the config file *agent* expects.

    Non-destructive by default. Existing files are parsed + merged
    (JSON formats) or appended once under the ``# mind-mem`` marker
    (text/yaml formats). Pass ``force=True`` to overwrite.

    Returns::

        {"agent": ..., "path": ..., "written": bool,
         "content": ..., "merged": bool, "skipped": bool}
    """
    spec = AGENT_REGISTRY.get(agent)
    if spec is None:
        raise ValueError(f"unknown agent: {agent!r}")

    path = spec.expand_path(workspace)
    merged = False
    skipped = False
    fmt = spec.config_fmt

    if fmt in _JSON_MERGERS:
        merger = _JSON_MERGERS[fmt]
        # Always load the existing JSON even when --force is set. `force`
        # semantically means "write even if no changes" — it must not mean
        # "drop every other key in the file". Without this, `mm install-all
        # --force` against a config that holds siblings of the mind-mem
        # entry (openclaw.json has Telegram, Discord, channels, agents,
        # gateway, wizard, etc. all at the same top level) produces a
        # minimal JSON containing only the mind-mem entry, silently
        # truncating the other sections.
        #
        # A file that exists but is not a strict-JSON object (JSONC with
        # comments, a typo, an unreadable file) is never rewritten: treating
        # it as {} is exactly how the user's whole config used to be
        # replaced by the mind-mem entry alone.
        try:
            loaded_obj = read_json_object(path)
        except UnsafeConfigError as exc:
            return _refused_result(agent, path, merger({}, workspace)[0], str(exc))
        existing: dict = loaded_obj if loaded_obj is not None else {}
        content, changed = merger(existing, workspace)
        merged = os.path.isfile(path) and not force
        skipped = merged and not changed
    elif fmt in ("text-block", "yaml-block"):
        block = spec.content_tmpl.format(ws=workspace)
        if os.path.isfile(path) and not force:
            try:
                existing_text = read_text_file(path) or ""
            except UnsafeConfigError as exc:
                return _refused_result(agent, path, block, str(exc))
            if _MM_MARKER in existing_text:
                content = existing_text
                skipped = True
            else:
                content = existing_text.rstrip() + "\n\n" + block
                merged = True
        else:
            content = block
    elif fmt == "none":
        # No instruction file for this client: the MCP registration (or the
        # manual steps) carries the whole integration.
        target = spec.mcp_path_tmpl and spec.expand_mcp_path(workspace)
        note = f"{spec.description}: no instruction file; " + (f"MCP entry goes to {target}" if target else f"manual setup — {spec.manual}")
        return {
            "agent": agent,
            "path": path,
            "written": False,
            "content": note,
            "merged": False,
            "skipped": True,
            "reason": "no instruction file",
        }
    else:
        raise ValueError(f"unknown config_fmt: {fmt!r} for agent {agent!r}")

    serialised = json.dumps(content, indent=2) if isinstance(content, dict) else content
    if dry_run:
        return {
            "agent": agent,
            "path": path,
            "written": False,
            "content": serialised,
            "merged": merged,
            "skipped": skipped,
        }
    if skipped:
        return {
            "agent": agent,
            "path": path,
            "written": False,
            "content": serialised,
            "merged": False,
            "skipped": True,
        }
    return _write_result(agent, path, serialised, merged)


def install_all(
    workspace: str,
    *,
    dry_run: bool = False,
    force: bool = False,
    agents: list[str] | None = None,
    include_mcp: bool = True,
) -> list[dict]:
    """Install mind-mem config for every detected (or specified) agent.

    When ``agents`` is None, auto-detects which clients are installed
    on the current machine via :func:`detect_installed_agents`.
    Otherwise uses the explicit list.

    When ``include_mcp`` is True (the default) and the agent's
    registry entry has an ``mcp_fmt`` set, *also* registers mind-mem
    as a native MCP server in that client's config. The text-hook
    install still runs so users see the ``# mind-mem`` block in a
    human-readable config.

    Returns the per-agent result list. Each agent with MCP support
    produces two result dicts: one for the hook install, one for
    the MCP registration (``{"phase": "hook" | "mcp"}``).
    """
    names = agents if agents is not None else detect_installed_agents(workspace)
    results: list[dict] = []
    for name in names:
        try:
            r = install_config(name, workspace, dry_run=dry_run, force=force)
            r["phase"] = "hook"
            results.append(r)
        except Exception as exc:  # pragma: no cover — per-agent isolation
            results.append({"agent": name, "phase": "hook", "error": str(exc), "written": False})
        if include_mcp:
            try:
                r = install_mcp_config(name, workspace, dry_run=dry_run, force=force)
                r["phase"] = "mcp"
                results.append(r)
            except Exception as exc:  # pragma: no cover
                results.append({"agent": name, "phase": "mcp", "error": str(exc), "written": False})
    return results


def uninstall_all(workspace: str, *, dry_run: bool = False, agents: list[str] | None = None) -> list[dict]:
    """Remove only the mind-mem MCP entry from each named (or detected) client.

    Instruction files are left alone: their ``# mind-mem`` block is plain
    text the user may have edited. Per-agent errors are isolated.
    """
    names = agents if agents is not None else detect_installed_agents(workspace)
    results: list[dict] = []
    for name in names:
        try:
            results.append(uninstall_mcp_config(name, workspace, dry_run=dry_run))
        except Exception as exc:  # pragma: no cover — per-agent isolation
            results.append({"agent": name, "error": str(exc), "written": False, "removed": False})
    return results


__all__ = [
    "HOOK_EVENT_SCHEMA",
    "HookEvent",
    "validate_event",
    "privacy_filter",
    "observation_to_block",
    "install_config",
    "install_mcp_config",
    "install_all",
    "uninstall_mcp_config",
    "uninstall_all",
    "MEMORY_PROTOCOL_SNIPPET",
    "NO_MCP_REASON",
    "detect_installed_agents",
    "mcp_server_spec",
    "AGENT_REGISTRY",
    "AgentSpec",
]
