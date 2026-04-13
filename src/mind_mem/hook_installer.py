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
# Installer
# ---------------------------------------------------------------------------


_MM_MARKER = "# mind-mem"


def _merge_claude_hooks(existing: dict, workspace: str) -> tuple[dict, bool]:
    """Merge mind-mem hook entries into a Claude Code settings dict.

    Returns (new_dict, changed). Idempotent: re-running doesn't
    duplicate command entries.
    """
    out = json.loads(json.dumps(existing))  # deep copy
    hooks = out.setdefault("hooks", {})
    wanted: list[tuple[str, str]] = [
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


def install_config(
    agent: str,
    workspace: str,
    *,
    dry_run: bool = False,
    force: bool = False,
) -> dict:
    """Generate (and optionally write) the config file each agent expects.

    Non-destructive by default. If the target file already exists:

    - For JSON configs (claude-code, gemini) the file is parsed and
      the mind-mem keys are merged in. Existing user keys are kept.
    - For text configs (codex AGENTS.md, cursor, windsurf, aider)
      the mind-mem block is appended once. Idempotent — re-running
      does not duplicate blocks.
    - ``force=True`` restores the legacy destructive behaviour
      (overwrite the file). Use sparingly.

    The returned dict has::

        {"agent": ..., "path": ..., "written": bool,
         "content": ..., "merged": bool, "skipped": bool}

    and is safe to pretty-print for CLI use.
    """
    merged = False
    skipped = False

    if agent == "claude-code":
        path = os.path.expanduser("~/.claude/settings.json")
        base_content: dict = {
            "hooks": {
                "SessionStart": [
                    {"command": f"mm inject --agent claude-code --workspace {workspace}"}
                ],
                "PostToolUse": [{"command": "mm capture --stdin"}],
                "Stop": [{"command": "mm vault status"}],
            }
        }
        content: Any = base_content
        if os.path.isfile(path) and not force:
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    existing = json.load(fh)
            except (OSError, json.JSONDecodeError):
                existing = None
            if isinstance(existing, dict):
                content, changed = _merge_claude_hooks(existing, workspace)
                merged = True
                skipped = not changed
    elif agent == "codex":
        path = os.path.join(workspace, "AGENTS.md")
        block = (
            f"{_MM_MARKER}: agent instructions (auto-written)\n\n"
            "Before every response, run `mm context \"$QUERY\"` and prepend the output.\n"
        )
        content = block
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
    elif agent == "gemini":
        path = os.path.join(workspace, ".gemini", "settings.json")
        mind_mem_instruction = (
            f"mind-mem workspace: {workspace}; "
            "run `mm inject --agent gemini` for context."
        )
        content = {"system_instruction": mind_mem_instruction}
        if os.path.isfile(path) and not force:
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    existing = json.load(fh)
            except (OSError, json.JSONDecodeError):
                existing = None
            if isinstance(existing, dict):
                out = dict(existing)
                if out.get("system_instruction") != mind_mem_instruction:
                    out["system_instruction"] = mind_mem_instruction
                    merged = True
                else:
                    skipped = True
                content = out
    elif agent == "cursor":
        path = os.path.join(workspace, ".cursorrules")
        block = (
            f"{_MM_MARKER}\nmind-mem workspace: {workspace}\n"
            "Use `mm inject --agent cursor` before answering.\n"
        )
        content = block
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
    elif agent == "windsurf":
        path = os.path.join(workspace, ".windsurfrules")
        block = (
            f"{_MM_MARKER}\nworkspace: {workspace}\n"
            "prefer `mm inject --agent windsurf`.\n"
        )
        content = block
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
    elif agent == "aider":
        path = os.path.join(workspace, ".aider.conf.yml")
        block = (
            f"{_MM_MARKER} auto-config\n"
            f"read: [\"{workspace}/CLAUDE.md\"]\n"
        )
        content = block
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
        raise ValueError(f"unknown agent: {agent!r}")

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


__all__ = [
    "HOOK_EVENT_SCHEMA",
    "HookEvent",
    "validate_event",
    "privacy_filter",
    "observation_to_block",
    "install_config",
]
