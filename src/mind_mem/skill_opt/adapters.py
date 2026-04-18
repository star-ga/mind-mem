# Copyright 2026 STARGA, Inc.
"""SkillAdapter protocol and concrete adapters for each agent system."""

from __future__ import annotations

import os
import re
from typing import Any, Protocol, runtime_checkable

from ._types import SkillSpec


def _parse_yaml_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Extract YAML frontmatter and body from a markdown file.

    Returns (metadata_dict, body_text). If no frontmatter, returns ({}, full text).
    Uses a lightweight parser to avoid a PyYAML dependency.
    """
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    raw = text[4:end].strip()
    body = text[end + 4 :].lstrip("\n")
    meta: dict[str, Any] = {}
    for line in raw.splitlines():
        if ":" not in line:
            continue
        key, _, raw_val = line.partition(":")
        key = key.strip()
        raw_val = raw_val.strip()
        parsed: Any
        if raw_val.startswith('"') and raw_val.endswith('"'):
            parsed = raw_val[1:-1]
        elif raw_val.startswith("'") and raw_val.endswith("'"):
            parsed = raw_val[1:-1]
        elif raw_val.startswith("[") and raw_val.endswith("]"):
            parsed = _parse_inline_list(raw_val)
        elif raw_val.startswith("{"):
            import json

            try:
                parsed = json.loads(raw_val)
            except (json.JSONDecodeError, ValueError):
                parsed = raw_val
        elif raw_val.lower() in ("true", "false"):
            parsed = raw_val.lower() == "true"
        elif re.match(r"^-?\d+$", raw_val):
            parsed = int(raw_val)
        else:
            parsed = raw_val
        meta[key] = parsed
    return meta, body


def _parse_inline_list(val: str) -> list[str]:
    """Parse a YAML-style inline list like ["Read", "Grep", "Glob"]."""
    inner = val[1:-1].strip()
    if not inner:
        return []
    items = []
    for item in inner.split(","):
        item = item.strip().strip("\"'")
        if item:
            items.append(item)
    return items


@runtime_checkable
class SkillAdapter(Protocol):
    """Parse and serialize skill files in a specific format."""

    format_id: str

    def can_handle(self, path: str) -> bool: ...
    def parse(self, path: str) -> SkillSpec: ...
    def serialize(self, spec: SkillSpec, content: str) -> str: ...
    def discover(self, root: str) -> list[str]: ...


class OpenClawSkillAdapter:
    """Adapter for OpenClaw-style SKILL.md files."""

    format_id = "skill-md"

    def can_handle(self, path: str) -> bool:
        return os.path.basename(path).upper() == "SKILL.MD"

    def parse(self, path: str) -> SkillSpec:
        with open(path, encoding="utf-8") as f:
            text = f.read()
        meta, body = _parse_yaml_frontmatter(text)
        name = str(meta.get("name", os.path.basename(os.path.dirname(path))))
        parent = os.path.basename(os.path.dirname(path))
        return SkillSpec(
            skill_id=f"openclaw:{parent}",
            system="openclaw",
            source_path=os.path.realpath(path),
            format=self.format_id,
            name=name,
            description=str(meta.get("description", "")),
            content=text,
            metadata=meta,
        )

    def serialize(self, spec: SkillSpec, content: str) -> str:
        return content

    def discover(self, root: str) -> list[str]:
        root = os.path.expanduser(root)
        if not os.path.isdir(root):
            return []
        paths: list[str] = []
        for entry in sorted(os.listdir(root)):
            skill_path = os.path.join(root, entry, "SKILL.md")
            if os.path.isfile(skill_path):
                paths.append(skill_path)
        return paths


class ClaudeAgentAdapter:
    """Adapter for Claude Code agent .md files (~/.claude/agents/)."""

    format_id = "agent-md"

    def can_handle(self, path: str) -> bool:
        return path.endswith(".md") and "agents" in path

    def parse(self, path: str) -> SkillSpec:
        with open(path, encoding="utf-8") as f:
            text = f.read()
        meta, body = _parse_yaml_frontmatter(text)
        stem = os.path.splitext(os.path.basename(path))[0]
        return SkillSpec(
            skill_id=f"claude:{stem}",
            system="claude-code",
            source_path=os.path.realpath(path),
            format=self.format_id,
            name=str(meta.get("name", stem)),
            description=str(meta.get("description", "")),
            content=text,
            metadata=meta,
        )

    def serialize(self, spec: SkillSpec, content: str) -> str:
        return content

    def discover(self, root: str) -> list[str]:
        root = os.path.expanduser(root)
        if not os.path.isdir(root):
            return []
        return sorted(os.path.join(root, f) for f in os.listdir(root) if f.endswith(".md") and not f.startswith("."))


class CodexSkillAdapter:
    """Adapter for Codex CLI skill files (~/.codex/skills/*/SKILL.md)."""

    format_id = "codex-skill-md"

    def can_handle(self, path: str) -> bool:
        return os.path.basename(path).upper() == "SKILL.MD" and ".codex" in path

    def parse(self, path: str) -> SkillSpec:
        with open(path, encoding="utf-8") as f:
            text = f.read()
        meta, body = _parse_yaml_frontmatter(text)
        parent = os.path.basename(os.path.dirname(path))
        return SkillSpec(
            skill_id=f"codex:{parent}",
            system="codex",
            source_path=os.path.realpath(path),
            format=self.format_id,
            name=str(meta.get("name", parent)),
            description=str(meta.get("description", "")),
            content=text,
            metadata=meta,
        )

    def serialize(self, spec: SkillSpec, content: str) -> str:
        return content

    def discover(self, root: str) -> list[str]:
        root = os.path.expanduser(root)
        if not os.path.isdir(root):
            return []
        paths: list[str] = []
        for entry in sorted(os.listdir(root)):
            skill_path = os.path.join(root, entry, "SKILL.md")
            if os.path.isfile(skill_path):
                paths.append(skill_path)
        return paths


class GeminiInstructionAdapter:
    """Adapter for Gemini system instruction files."""

    format_id = "gemini-instruction"

    def can_handle(self, path: str) -> bool:
        return ".gemini" in path and path.endswith(".md")

    def parse(self, path: str) -> SkillSpec:
        with open(path, encoding="utf-8") as f:
            text = f.read()
        stem = os.path.splitext(os.path.basename(path))[0]
        return SkillSpec(
            skill_id=f"gemini:{stem}",
            system="gemini",
            source_path=os.path.realpath(path),
            format=self.format_id,
            name=stem,
            description=f"Gemini system instruction: {stem}",
            content=text,
            metadata={},
        )

    def serialize(self, spec: SkillSpec, content: str) -> str:
        return content

    def discover(self, root: str) -> list[str]:
        root = os.path.expanduser(root)
        if not os.path.isdir(root):
            return []
        return sorted(os.path.join(root, f) for f in os.listdir(root) if f.endswith(".md") and not f.startswith("."))


ADAPTER_REGISTRY: dict[str, SkillAdapter] = {
    "skill-md": OpenClawSkillAdapter(),  # type: ignore[assignment]
    "agent-md": ClaudeAgentAdapter(),  # type: ignore[assignment]
    "codex-skill-md": CodexSkillAdapter(),  # type: ignore[assignment]
    "gemini-instruction": GeminiInstructionAdapter(),  # type: ignore[assignment]
}


def discover_all(sources: dict[str, str]) -> list[SkillSpec]:
    """Discover and parse all skills across all configured sources."""
    specs: list[SkillSpec] = []
    source_to_adapter: dict[str, SkillAdapter] = {
        "openclaw": ADAPTER_REGISTRY["skill-md"],
        "openclaw_agent": ADAPTER_REGISTRY["skill-md"],
        "claude_agents": ADAPTER_REGISTRY["agent-md"],
        "codex_skills": ADAPTER_REGISTRY["codex-skill-md"],
        "codex_memories": ADAPTER_REGISTRY["codex-skill-md"],
        "gemini": ADAPTER_REGISTRY["gemini-instruction"],
    }
    for source_key, root in sources.items():
        adapter = source_to_adapter.get(source_key)
        if adapter is None:
            continue
        expanded = os.path.expanduser(root)
        for path in adapter.discover(expanded):
            try:
                specs.append(adapter.parse(path))
            except (OSError, ValueError):
                continue
    return specs


def adapter_for_spec(spec: SkillSpec) -> SkillAdapter:
    """Return the adapter that can handle a given SkillSpec's format."""
    adapter = ADAPTER_REGISTRY.get(spec.format)
    if adapter is None:
        raise ValueError(f"No adapter registered for format: {spec.format}")
    return adapter
