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
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional


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
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}"),         # Anthropic API keys
    re.compile(r"xai-[A-Za-z0-9]{20,}"),               # xAI API keys
    re.compile(r"sk-[A-Za-z0-9]{20,}"),                # generic secret-key prefix
    re.compile(r"pypi-[A-Za-z0-9_\-]{20,}"),           # PyPI macaroons
    re.compile(r"AKIA[0-9A-Z]{16}"),                   # AWS access key id
    re.compile(r"AIza[0-9A-Za-z_\-]{35}"),             # Google API key
    re.compile(r"ghp_[A-Za-z0-9]{36}"),                # GitHub personal token
    re.compile(r"xox[baprs]-[A-Za-z0-9\-]{10,}"),      # Slack tokens
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
# Installer — registry-driven for all supported AI coding clients
# ---------------------------------------------------------------------------


import shutil as _shutil
from dataclasses import dataclass, field

_MM_MARKER = "# mind-mem"


@dataclass(frozen=True)
class AgentSpec:
    """Declarative registry entry for an AI coding client integration.

    Attributes:
        name:            Canonical agent key used in CLI + API.
        description:     One-line human summary.
        config_fmt:      "json-claude-hooks" | "json-gemini" |
                         "json-openclaw-hooks" | "json-generic" |
                         "text-block" | "yaml-block".
        path_tmpl:       Target config path; ``{ws}`` expands to workspace,
                         ``{home}`` to ``~``.
        content_tmpl:    Text body (for text/yaml formats) with ``{ws}``.
        detect_paths:    Files/dirs whose existence marks "installed".
        detect_binaries: Executables whose presence on PATH marks "installed".
        always_offer:    If True, auto-detect includes this agent even without
                         signals — suitable for near-universal configs like
                         GitHub Copilot's per-workspace instructions file.
    """

    name: str
    description: str
    config_fmt: str
    path_tmpl: str
    content_tmpl: str = ""
    detect_paths: tuple[str, ...] = field(default_factory=tuple)
    detect_binaries: tuple[str, ...] = field(default_factory=tuple)
    always_offer: bool = False

    def expand_path(self, workspace: str) -> str:
        return self.path_tmpl.format(
            ws=workspace,
            home=os.path.expanduser("~"),
        )


# ---------------------------------------------------------------------------
# Format writers — each (existing_text_or_dict, workspace) → (new, merged,
# skipped). Workspace is passed so format writers can render workspace paths.
# ---------------------------------------------------------------------------


def _merge_claude_hooks(existing: dict, workspace: str) -> tuple[dict, bool]:
    """Idempotent hook merge for Claude Code settings.json."""
    out = json.loads(json.dumps(existing))
    hooks = out.setdefault("hooks", {})
    wanted = [
        ("SessionStart", f"mm inject --agent claude-code --workspace {workspace}"),
        ("PostToolUse", "mm capture --stdin"),
        ("Stop", "mm vault status"),
    ]
    changed = False
    for event, command in wanted:
        entries = hooks.setdefault(event, [])
        if not isinstance(entries, list):
            continue
        if not any(
            isinstance(e, dict) and e.get("command") == command for e in entries
        ):
            entries.append({"command": command})
            changed = True
    return out, changed


def _merge_openclaw_hooks(existing: dict, workspace: str) -> tuple[dict, bool]:
    """OpenClaw / Nanoclaw / Nemoclaw hook registry merge.

    All three share the claw-family hooks JSON shape at
    ``.hooks.internal.entries.mind-mem`` so we reuse this merger.
    """
    out = json.loads(json.dumps(existing))
    hooks = out.setdefault("hooks", {}).setdefault("internal", {}).setdefault(
        "entries", {}
    )
    target = {
        "enabled": True,
        "workspace": workspace,
        "commands": {
            "inject": "mm inject --agent openclaw --workspace " + workspace,
            "capture": "mm capture --stdin",
            "status": "mm vault status",
        },
    }
    changed = hooks.get("mind-mem") != target
    if changed:
        hooks["mind-mem"] = target
    return out, changed


def _merge_gemini(existing: dict, workspace: str) -> tuple[dict, bool]:
    """Gemini settings.json system_instruction injection."""
    instr = (
        f"mind-mem workspace: {workspace}; "
        "run `mm inject --agent gemini` before answering."
    )
    out = dict(existing)
    changed = out.get("system_instruction") != instr
    out["system_instruction"] = instr
    return out, changed


def _merge_continue(existing: dict, workspace: str) -> tuple[dict, bool]:
    """Continue.dev config.json: inject a systemMessage."""
    out = json.loads(json.dumps(existing))
    sys_msg = (
        f"mind-mem workspace: {workspace}. "
        "Run `mm inject --agent continue` before composing responses."
    )
    if out.get("systemMessage") == sys_msg:
        return out, False
    out["systemMessage"] = sys_msg
    return out, True


def _merge_zed(existing: dict, workspace: str) -> tuple[dict, bool]:
    """Zed settings.json: inject assistant default_model_instructions."""
    out = json.loads(json.dumps(existing))
    assistant = out.setdefault("assistant", {})
    sys_msg = (
        f"mind-mem workspace: {workspace}. "
        "Use `mm inject --agent zed` for context."
    )
    if assistant.get("default_system_message") == sys_msg:
        return out, False
    assistant["default_system_message"] = sys_msg
    return out, True


def _merge_generic_json(existing: dict, workspace: str) -> tuple[dict, bool]:
    """Fallback JSON merge — sets {"mind_mem": {...}} key."""
    target = {
        "enabled": True,
        "workspace": workspace,
        "inject_cmd": f"mm inject --workspace {workspace}",
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
# Registry — every known AI coding client
# ---------------------------------------------------------------------------


AGENT_REGISTRY: dict[str, AgentSpec] = {
    "claude-code": AgentSpec(
        name="claude-code",
        description="Claude Code CLI (Anthropic)",
        config_fmt="json-claude-hooks",
        path_tmpl="{home}/.claude/settings.json",
        detect_paths=("{home}/.claude", "{home}/.config/claude"),
        detect_binaries=("claude",),
    ),
    "codex": AgentSpec(
        name="codex",
        description="OpenAI Codex CLI",
        config_fmt="text-block",
        path_tmpl="{ws}/AGENTS.md",
        content_tmpl=(
            f"{_MM_MARKER}: agent instructions (auto-written)\n\n"
            "Before every response, run `mm context \"$QUERY\"` and prepend the output.\n"
        ),
        detect_binaries=("codex",),
    ),
    "gemini": AgentSpec(
        name="gemini",
        description="Google Gemini CLI",
        config_fmt="json-gemini",
        path_tmpl="{ws}/.gemini/settings.json",
        detect_paths=("{home}/.gemini",),
        detect_binaries=("gemini",),
    ),
    "cursor": AgentSpec(
        name="cursor",
        description="Cursor editor",
        config_fmt="text-block",
        path_tmpl="{ws}/.cursorrules",
        content_tmpl=(
            f"{_MM_MARKER}\nmind-mem workspace: {{ws}}\n"
            "Use `mm inject --agent cursor` before answering.\n"
        ),
        detect_paths=(
            "{home}/.cursor",
            "{home}/Library/Application Support/Cursor",
            "{home}/AppData/Roaming/Cursor",
        ),
        detect_binaries=("cursor",),
    ),
    "windsurf": AgentSpec(
        name="windsurf",
        description="Windsurf editor (Codeium)",
        config_fmt="text-block",
        path_tmpl="{ws}/.windsurfrules",
        content_tmpl=(
            f"{_MM_MARKER}\nworkspace: {{ws}}\n"
            "prefer `mm inject --agent windsurf`.\n"
        ),
        detect_paths=(
            "{home}/.codeium/windsurf",
            "{home}/.windsurf",
            "{home}/Library/Application Support/Windsurf",
        ),
        detect_binaries=("windsurf",),
    ),
    "aider": AgentSpec(
        name="aider",
        description="aider CLI (paul-gauthier)",
        config_fmt="yaml-block",
        path_tmpl="{ws}/.aider.conf.yml",
        content_tmpl=(
            f"{_MM_MARKER} auto-config\n"
            "read: [\"{ws}/CLAUDE.md\"]\n"
        ),
        detect_binaries=("aider",),
    ),
    "openclaw": AgentSpec(
        name="openclaw",
        description="OpenClaw (STARGA cognitive assistant)",
        config_fmt="json-openclaw-hooks",
        path_tmpl="{home}/.openclaw/openclaw.json",
        detect_paths=("{home}/.openclaw",),
        detect_binaries=("openclaw",),
    ),
    "nanoclaw": AgentSpec(
        name="nanoclaw",
        description="NanoClaw (compact claw variant)",
        config_fmt="json-openclaw-hooks",
        path_tmpl="{home}/.nanoclaw/nanoclaw.json",
        detect_paths=("{home}/.nanoclaw",),
        detect_binaries=("nanoclaw",),
    ),
    "nemoclaw": AgentSpec(
        name="nemoclaw",
        description="NemoClaw (memory-focused claw variant)",
        config_fmt="json-openclaw-hooks",
        path_tmpl="{home}/.nemoclaw/nemoclaw.json",
        detect_paths=("{home}/.nemoclaw",),
        detect_binaries=("nemoclaw",),
    ),
    "continue": AgentSpec(
        name="continue",
        description="Continue.dev (VS Code / JetBrains extension)",
        config_fmt="json-continue",
        path_tmpl="{home}/.continue/config.json",
        detect_paths=("{home}/.continue",),
    ),
    "cline": AgentSpec(
        name="cline",
        description="Cline (VS Code extension)",
        config_fmt="text-block",
        path_tmpl="{ws}/.clinerules",
        content_tmpl=(
            f"{_MM_MARKER}\nmind-mem workspace: {{ws}}\n"
            "Run `mm inject --agent cline` before tool use.\n"
        ),
        detect_paths=(
            "{home}/.vscode/extensions",
            "{home}/.vscode-server/extensions",
        ),
    ),
    "roo": AgentSpec(
        name="roo",
        description="Roo Code (VS Code fork / extension)",
        config_fmt="text-block",
        path_tmpl="{ws}/.roo/system-prompt.md",
        content_tmpl=(
            f"{_MM_MARKER}\nmind-mem workspace: {{ws}}\n"
            "Use `mm inject --agent roo` before answering.\n"
        ),
        detect_paths=(
            "{home}/.roo",
            "{home}/.vscode/extensions",
        ),
    ),
    "zed": AgentSpec(
        name="zed",
        description="Zed editor AI assistant",
        config_fmt="json-zed",
        path_tmpl="{home}/.config/zed/settings.json",
        detect_paths=(
            "{home}/.config/zed",
            "{home}/Library/Application Support/Zed",
        ),
        detect_binaries=("zed", "zeditor"),
    ),
    "copilot": AgentSpec(
        name="copilot",
        description="GitHub Copilot (workspace instructions)",
        config_fmt="text-block",
        path_tmpl="{ws}/.github/copilot-instructions.md",
        content_tmpl=(
            f"{_MM_MARKER}: GitHub Copilot workspace instructions\n\n"
            "This repository uses mind-mem for persistent memory. "
            "Before answering, consult memory with `mm inject --agent copilot`. "
            "Respect ADR / DECISION blocks; route new decisions through "
            "`propose_update` rather than modifying them directly.\n"
        ),
        always_offer=True,  # virtually every dev machine has Copilot potential
    ),
    "cody": AgentSpec(
        name="cody",
        description="Sourcegraph Cody",
        config_fmt="json-generic",
        path_tmpl="{ws}/.cody/config.json",
        detect_paths=("{home}/.config/cody",),
        detect_binaries=("cody",),
    ),
    "qodo": AgentSpec(
        name="qodo",
        description="Qodo Gen (formerly CodiumAI)",
        config_fmt="text-block",
        path_tmpl="{ws}/.codium/ai-rules.md",
        content_tmpl=(
            f"{_MM_MARKER}\nmind-mem workspace: {{ws}}\n"
            "Consult mind-mem memory before proposing changes.\n"
        ),
        detect_paths=("{home}/.codium", "{home}/.qodo"),
    ),
}


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def _path_exists(pattern: str, workspace: str) -> bool:
    expanded = pattern.format(ws=workspace, home=os.path.expanduser("~"))
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
        existing: dict = {}
        if os.path.isfile(path) and not force:
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    loaded = json.load(fh)
                if isinstance(loaded, dict):
                    existing = loaded
            except (OSError, json.JSONDecodeError):
                existing = {}
        content, changed = merger(existing, workspace)
        merged = os.path.isfile(path) and not force
        skipped = merged and not changed
    elif fmt in ("text-block", "yaml-block"):
        block = spec.content_tmpl.format(ws=workspace)
        if os.path.isfile(path) and not force:
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    existing_text = fh.read()
            except OSError:
                existing_text = ""
            if _MM_MARKER in existing_text:
                content = existing_text
                skipped = True
            else:
                content = existing_text.rstrip() + "\n\n" + block
                merged = True
        else:
            content = block
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
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if skipped:
        return {
            "agent": agent,
            "path": path,
            "written": False,
            "content": serialised,
            "merged": False,
            "skipped": True,
        }
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(serialised if serialised.endswith("\n") else serialised + "\n")
    return {
        "agent": agent,
        "path": path,
        "written": True,
        "content": serialised,
        "merged": merged,
        "skipped": False,
    }


def install_all(
    workspace: str,
    *,
    dry_run: bool = False,
    force: bool = False,
    agents: list[str] | None = None,
) -> list[dict]:
    """Install mind-mem config for every detected (or specified) agent.

    When ``agents`` is None, auto-detects which clients are installed
    on the current machine via :func:`detect_installed_agents`.
    Otherwise uses the explicit list.

    Returns the per-agent result list so callers can surface which
    files were written, merged, or skipped.
    """
    names = agents if agents is not None else detect_installed_agents(workspace)
    results: list[dict] = []
    for name in names:
        try:
            results.append(
                install_config(name, workspace, dry_run=dry_run, force=force)
            )
        except Exception as exc:  # pragma: no cover — per-agent isolation
            results.append(
                {"agent": name, "error": str(exc), "written": False}
            )
    return results


__all__ = [
    "HOOK_EVENT_SCHEMA",
    "HookEvent",
    "validate_event",
    "privacy_filter",
    "observation_to_block",
    "install_config",
    "install_all",
    "detect_installed_agents",
    "AGENT_REGISTRY",
    "AgentSpec",
]
