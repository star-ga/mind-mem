# Copyright 2026 STARGA, Inc.
"""Universal agent bridge (v2.7.0).

Provides primitives + a thin formatter layer so non-MCP CLI agents
(codex, gemini CLI, Cursor, Windsurf, Aider, plain shell) can share
the same workspace memory through agent-specific text injection.

Two surfaces:

1. :class:`AgentFormatter` — ``inject(query, context_blocks)`` returns
   a pre-formatted text snippet ready for the agent's expected
   convention (CLAUDE.md, AGENTS.md, GEMINI.md, .cursorrules,
   .windsurfrules, .aider.conf.yml repo-map, generic stdout).
2. :class:`VaultBridge` — bidirectional sync between mind-mem's block
   model and an Obsidian-style markdown vault. Forward sync reads
   ``.md`` files into structured records; reverse sync writes records
   back to vault files with frontmatter.

Pure-Python stdlib. Filesystem watcher and the per-agent hook
installer remain deferred (need watchdog and per-agent setup
scripts that aren't unit-testable in this codebase).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Optional

# ---------------------------------------------------------------------------
# AgentFormatter
# ---------------------------------------------------------------------------


class UnknownAgentError(Exception):
    """Raised when an unsupported agent name is requested."""


# Recognised agent identifiers — extended to cover the roadmap's named
# CLIs. Adding a new agent is a one-line change here plus a method on
# :class:`AgentFormatter`.
KNOWN_AGENTS: tuple[str, ...] = (
    "claude-code",
    "codex",
    "gemini",
    "cursor",
    "windsurf",
    "aider",
    "generic",
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _normalise_block(block: Mapping[str, Any]) -> dict:
    """Coerce a result dict into the {id, type, file, text} shape we need."""
    text = ""
    for field_name in ("excerpt", "statement", "text", "content"):
        value = block.get(field_name)
        if isinstance(value, str) and value.strip():
            text = value.strip()
            break
    block_id = block.get("_id") or block.get("id") or block.get("block_id") or "?"
    return {
        "id": str(block_id),
        "type": str(block.get("type", "block")),
        "file": str(block.get("file", "")),
        "text": text,
    }


@dataclass
class AgentFormatter:
    """Render context blocks into agent-native injection snippets."""

    max_blocks: int = 20

    def inject(
        self,
        agent: str,
        query: str,
        blocks: Iterable[Mapping[str, Any]],
    ) -> str:
        """Return a text snippet ready to paste into *agent*'s prompt."""
        if agent not in KNOWN_AGENTS:
            raise UnknownAgentError(f"unknown agent {agent!r}. Known: {', '.join(KNOWN_AGENTS)}")
        normalised = [_normalise_block(b) for b in blocks][: self.max_blocks]
        method = {
            "claude-code": self._claude,
            "codex": self._codex,
            "gemini": self._gemini,
            "cursor": self._cursor,
            "windsurf": self._windsurf,
            "aider": self._aider,
            "generic": self._generic,
        }[agent]
        return method(query, normalised)

    # ------------------------------------------------------------------
    # Per-agent renderers
    # ------------------------------------------------------------------

    def _claude(self, query: str, blocks: list[dict]) -> str:
        # Designed to drop into a CLAUDE.md fenced "Recent Memory"
        # block. Claude Code reads CLAUDE.md verbatim into the system
        # prompt so we keep markdown-friendly headings.
        out = ["# mind-mem context", "", f"**Query:** {query}", ""]
        for b in blocks:
            out.append(f"## {b['type']} — {b['id']}")
            if b["file"]:
                out.append(f"*Source: {b['file']}*")
            out.append("")
            out.append(b["text"] or "_(no excerpt)_")
            out.append("")
        return "\n".join(out).rstrip() + "\n"

    def _codex(self, query: str, blocks: list[dict]) -> str:
        # codex picks up AGENTS.md / codex.md. Headings preserve
        # information structure without forcing a specific format.
        out = [
            f"# Context for: {query}",
            "",
            "Use the references below before answering.",
            "",
        ]
        for b in blocks:
            out.append(f"- **{b['type']}** [{b['id']}]: {b['text']}")
        return "\n".join(out).rstrip() + "\n"

    def _gemini(self, query: str, blocks: list[dict]) -> str:
        # GEMINI.md uses a similar markdown convention; the leading
        # "system" tag is honoured by `gemini --system-instruction`.
        lines = [f"system: relevant memory for query {query!r}", ""]
        for b in blocks:
            lines.append(f"- [{b['id']}] ({b['type']}) {b['text']}")
        return "\n".join(lines).rstrip() + "\n"

    def _cursor(self, query: str, blocks: list[dict]) -> str:
        # .cursorrules is plain text. Bullet form keeps it scannable.
        lines = [
            f"# Workspace memory for: {query}",
            "",
            "When answering, prefer these memory blocks:",
        ]
        for b in blocks:
            lines.append(f"  - [{b['id']}] {b['text']}")
        return "\n".join(lines).rstrip() + "\n"

    def _windsurf(self, query: str, blocks: list[dict]) -> str:
        # .windsurfrules is similar to Cursor; differentiate slightly
        # so a future format-divergence is non-disruptive.
        lines = [f"# mind-mem ({query})", ""]
        for b in blocks:
            lines.append(f"- {b['type']} {b['id']} :: {b['text']}")
        return "\n".join(lines).rstrip() + "\n"

    def _aider(self, query: str, blocks: list[dict]) -> str:
        # aider's repo-map is YAML-ish. We stay close to that.
        lines = ["repo_map:", "  memory:"]
        for b in blocks:
            lines.append(f"    - id: {b['id']}")
            lines.append(f"      type: {b['type']}")
            text_quoted = b["text"].replace('"', "'")
            lines.append(f'      summary: "{text_quoted}"')
        return "\n".join(lines).rstrip() + "\n"

    def _generic(self, query: str, blocks: list[dict]) -> str:
        # Plain text: pipeable into anything. Useful default for
        # `mm inject --agent generic | <some-cli>` integrations.
        lines = [f"Query: {query}", "Context:"]
        for b in blocks:
            lines.append(f"- [{b['id']}] {b['text']}")
        return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# VaultBridge — bidirectional sync between mind-mem and a markdown vault
# ---------------------------------------------------------------------------


_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Split ``---YAML---`` frontmatter from the body. Stdlib-only.

    A real YAML parser is overkill here — vault frontmatter is almost
    always flat ``key: value`` pairs. Anything more complex is left as
    a single string so round-tripping doesn't lose data.
    """
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    raw = m.group(1)
    body = text[m.end() :]
    out: dict[str, str] = {}
    for line in raw.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        out[key.strip()] = value.strip().strip('"').strip("'")
    return out, body


def _serialise_frontmatter(fields: Mapping[str, Any]) -> str:
    if not fields:
        return ""
    lines = ["---"]
    for k, v in sorted(fields.items()):
        if v is None:
            continue
        if isinstance(v, (list, tuple)):
            lines.append(f"{k}: [{', '.join(str(x) for x in v)}]")
        else:
            text = str(v).replace("\n", " ")
            lines.append(f"{k}: {text}")
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


@dataclass(frozen=True)
class VaultBlock:
    """Round-trippable representation of one vault note."""

    relative_path: str
    block_id: str
    block_type: str
    title: str
    body: str
    frontmatter: dict = field(default_factory=dict)
    modified_at: Optional[str] = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "block_id": self.block_id,
            "block_type": self.block_type,
            "title": self.title,
            "body": self.body,
            "frontmatter": self.frontmatter,
            "modified_at": self.modified_at,
        }


def _id_from_filename(path: str) -> str:
    name = os.path.basename(path)
    if name.endswith(".md"):
        name = name[:-3]
    return name or "block"


@dataclass
class VaultBridge:
    """Forward (vault → mind-mem) + reverse (mind-mem → vault) sync."""

    vault_root: str
    excludes: tuple[str, ...] = (".obsidian", ".trash", "templates")

    # ------------------------------------------------------------------
    # Forward sync: vault → mind-mem
    # ------------------------------------------------------------------

    def scan(
        self,
        *,
        sync_dirs: Optional[Iterable[str]] = None,
    ) -> list[VaultBlock]:
        """Walk the vault and return :class:`VaultBlock` records.

        Args:
            sync_dirs: Optional list of vault subdirectories to include.
                When None, the entire vault (minus :attr:`excludes`) is
                scanned.
        """
        root = os.path.realpath(self.vault_root)
        if not os.path.isdir(root):
            raise FileNotFoundError(f"vault root not found: {root}")

        roots: list[str] = []
        if sync_dirs:
            for d in sync_dirs:
                full = os.path.realpath(os.path.join(root, d))
                if not full.startswith(root + os.sep) and full != root:
                    raise ValueError(f"sync_dir {d!r} escapes vault root")
                if os.path.isdir(full):
                    roots.append(full)
        else:
            roots = [root]

        out: list[VaultBlock] = []
        for sub_root in roots:
            for dirpath, dirnames, filenames in os.walk(sub_root):
                # Pruning excludes saves an awful lot of stat() calls
                # in vaults with large .obsidian or .trash dirs.
                dirnames[:] = [d for d in dirnames if d not in self.excludes]
                for fn in filenames:
                    if not fn.endswith(".md"):
                        continue
                    full = os.path.join(dirpath, fn)
                    rel = os.path.relpath(full, root)
                    try:
                        text = open(full, "r", encoding="utf-8", errors="replace").read()
                    except OSError:
                        continue
                    fm, body = _parse_frontmatter(text)
                    block_id = fm.get("id") or _id_from_filename(full)
                    block_type = fm.get("type") or "note"
                    title = fm.get("title") or _id_from_filename(full)
                    try:
                        mtime = os.path.getmtime(full)
                        modified_at = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat().replace("+00:00", "Z")
                    except OSError:
                        modified_at = None
                    out.append(
                        VaultBlock(
                            relative_path=rel,
                            block_id=block_id,
                            block_type=block_type,
                            title=title,
                            body=body.strip(),
                            frontmatter=fm,
                            modified_at=modified_at,
                        )
                    )
        return out

    # ------------------------------------------------------------------
    # Reverse sync: mind-mem → vault
    # ------------------------------------------------------------------

    def write(
        self,
        block: VaultBlock,
        *,
        overwrite: bool = False,
    ) -> str:
        """Write *block* into the vault, returning the absolute path.

        Refuses to write if the path escapes the vault root or if the
        target exists and ``overwrite`` is False. The frontmatter
        always includes ``id``, ``type``, and ``title`` so a future
        round-trip parses cleanly.
        """
        root = os.path.realpath(self.vault_root)
        if not os.path.isdir(root):
            raise FileNotFoundError(f"vault root not found: {root}")
        if not block.relative_path or os.path.isabs(block.relative_path):
            raise ValueError("relative_path must be a non-empty relative path")

        target = os.path.realpath(os.path.join(root, block.relative_path))
        if not target.startswith(root + os.sep):
            raise ValueError(f"relative_path escapes vault root: {block.relative_path!r}")
        if os.path.exists(target) and not overwrite:
            raise FileExistsError(target)

        os.makedirs(os.path.dirname(target), exist_ok=True)
        fm = dict(block.frontmatter)
        fm.update(
            id=block.block_id,
            type=block.block_type,
            title=block.title,
            updated=_now_iso(),
        )
        text = _serialise_frontmatter(fm) + (block.body.rstrip() + "\n")
        # Atomic write: write to .tmp then rename.
        tmp = target + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
        return target


__all__ = [
    "AgentFormatter",
    "KNOWN_AGENTS",
    "UnknownAgentError",
    "VaultBlock",
    "VaultBridge",
]
