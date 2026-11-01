# Copyright 2026 STARGA, Inc.
"""Spec type, instruction text, and path helpers for client entries.

The registry module and the client tables import this module.
This module does not import another project module.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any


def current_platform() -> str:
    """``"linux"``, ``"darwin"`` or ``"win32"`` (other POSIX maps to linux)."""
    if sys.platform.startswith("win"):
        return "win32"
    if sys.platform == "darwin":
        return "darwin"
    return "linux"


def template_vars(workspace: str) -> dict[str, str]:
    """Return the placeholder values for the current platform."""
    home = os.path.expanduser("~")
    plat = current_platform()
    config = os.path.join(home, ".config")
    appdata = os.environ.get("APPDATA") or os.path.join(home, "AppData", "Roaming")
    if plat == "darwin":
        appsupport = os.path.join(home, "Library", "Application Support")
        data = appsupport
    elif plat == "win32":
        appsupport = appdata
        data = appdata
    else:
        appsupport = config
        data = os.path.join(home, ".local", "share")
    return {
        "ws": workspace,
        "home": home,
        "config": config,
        "appdata": appdata,
        "appsupport": appsupport,
        "vscode_user": os.path.join(appsupport, "Code", "User"),
        "data": data,
    }


def expand(template: str, workspace: str) -> str:
    """Expand *template* for the current platform."""
    return os.path.normpath(template.format(**template_vars(workspace))) if template else template


def env_override(override: tuple[str, str] | tuple[()], workspace: str) -> str | None:
    """Resolve an ``(ENV_VAR, suffix)`` override.

    When the variable is set, the target is ``$ENV_VAR/suffix`` (or the
    variable's value itself when *suffix* is empty, for variables that name
    a file). Returns ``None`` when the variable is unset or empty.
    """
    if not override:
        return None
    var, suffix = override
    value = os.environ.get(var, "").strip()
    if not value:
        return None
    value = os.path.expanduser(value)
    return os.path.normpath(os.path.join(value, suffix)) if suffix else os.path.normpath(value)


_MM_MARKER = "# mind-mem"

# Canonical Memory Protocol snippet -- written into every detected CLI's
# system-prompt file by ``mm install-all``. The companion document is
# ``docs/agent-memory-protocol.md``; both must be kept in sync.
#
# Wiring at the protocol level (MCP entries) is necessary but not
# sufficient -- without an explicit instruction the LLM in each CLI does
# not reliably call the MCP tools. This snippet is the instruction half.
MEMORY_PROTOCOL_SNIPPET: str = (
    f"{_MM_MARKER}: agent instructions (auto-written by `mm install-all`)\n"
    "\n"
    "## Memory Protocol (mind-mem MCP — USE CONSTANTLY)\n"
    "\n"
    "The mind-mem MCP server is wired into this CLI. Before answering ANY\n"
    "question that touches prior work, decisions, project state, people,\n"
    'companies, technical choices, or "remember when…" — call mind-mem\n'
    "first. The governed blocks in mind-mem's Postgres backend are the\n"
    "source of truth.\n"
    "\n"
    "### Always recall before answering\n"
    'Run `mcp__mind-mem__recall` (or `mm recall "$query"` via bash) with\n'
    "relevant keywords FIRST. If recall returns results, cite them. If\n"
    'not, say "no record found" — do not guess.\n'
    "\n"
    "### Always propose_update after learning new facts\n"
    "When the user shares a new decision, project status, relationship,\n"
    "contact, or technical finding worth preserving — run\n"
    "`mcp__mind-mem__propose_update` to store it. Don't wait to be asked.\n"
    "\n"
    "### Use hybrid_search for complex queries\n"
    "For multi-faceted questions, use `mcp__mind-mem__hybrid_search`\n"
    "(BM25 + vector + RRF fusion) for best recall.\n"
    "\n"
    "### Hallucination guardrail\n"
    "If you find yourself writing about prior projects, repos, or\n"
    "decisions without having called recall in this session, you are\n"
    "about to hallucinate. Stop and call recall first.\n"
)


@dataclass(frozen=True)
class AgentSpec:
    """Declarative registry entry for an AI client integration.

    Attributes:
        name:            Canonical agent key used in CLI + API.
        description:     One-line human summary.
        config_fmt:      Instruction/hook dialect: "json-claude-hooks" |
                         "json-gemini" | "json-openclaw-hooks" | "json-generic"
                         | "json-continue" | "json-zed" | "text-block" |
                         "yaml-block" | "none" (no instruction file).
        path_tmpl:       Instruction/hook target (see :mod:`client_paths`).
        content_tmpl:    Text body (for text/yaml formats) with ``{ws}``.
        detect_paths:    Files/dirs whose existence marks "installed"; a
                         pattern with ``*`` is globbed (editor extensions).
        detect_binaries: Executables whose presence on PATH marks "installed".
        always_offer:    Auto-detect includes this agent even without signals.
        vendor:          Who ships the client (docs tables).
        mcp_fmt:         MCP dialect ("" = no automatic MCP registration).
        mcp_path_tmpl:   Primary MCP config file.
        mcp_platform_paths: ``(platform, template)`` overrides of the primary.
        mcp_path_alts:   Existing files preferred over the primary, in order.
        mcp_env_override: ``(ENV_VAR, suffix)``: the client's own relocation
                         variable; when set it replaces the primary.
        mcp_extra_paths: Further documented locations, written only when
                         their directory already exists.
        mcp_guard:       "dir": write the primary only when its directory
                         exists; "file": only when the file exists.
        mcp_guard_platforms: Platforms the guard applies to (empty = all).
        platforms:       Platforms the MCP integration exists on (empty = all).
        manual:          Manual MCP steps for clients with no writable file.
    """

    name: str
    description: str
    config_fmt: str
    path_tmpl: str
    content_tmpl: str = ""
    detect_paths: tuple[str, ...] = field(default_factory=tuple)
    detect_binaries: tuple[str, ...] = field(default_factory=tuple)
    always_offer: bool = False
    mcp_fmt: str = ""
    mcp_path_tmpl: str = ""
    vendor: str = ""
    mcp_platform_paths: tuple[tuple[str, str], ...] = ()
    mcp_path_alts: tuple[str, ...] = ()
    mcp_env_override: tuple[str, str] | tuple[()] = ()
    mcp_extra_paths: tuple[str, ...] = ()
    mcp_guard: str = ""
    mcp_guard_platforms: tuple[str, ...] = ()
    platforms: tuple[str, ...] = ()
    manual: str = ""

    def expand_path(self, workspace: str) -> str:
        return expand(self.path_tmpl, workspace)

    def expand_mcp_path(self, workspace: str) -> str:
        """The primary MCP config file for this machine."""
        override = env_override(self.mcp_env_override, workspace)
        if override:
            return override
        plat = current_platform()
        tmpl = dict(self.mcp_platform_paths).get(plat, self.mcp_path_tmpl)
        for alt in self.mcp_path_alts:
            path = expand(alt, workspace)
            if os.path.isfile(path):
                return path
        return expand(tmpl, workspace)

    def mcp_guarded(self) -> str:
        """The guard in force on this platform ("" when none)."""
        if self.mcp_guard and (not self.mcp_guard_platforms or current_platform() in self.mcp_guard_platforms):
            return self.mcp_guard
        return ""

    def supported_here(self) -> bool:
        return not self.platforms or current_platform() in self.platforms


_AGENTS_MD = "{ws}/AGENTS.md"


def _client(
    name: str,
    description: str,
    vendor: str,
    *,
    mcp_fmt: str = "",
    mcp: str = "",
    hook: str = _AGENTS_MD,
    detect: tuple[str, ...] = (),
    bins: tuple[str, ...] = (),
    **extra: Any,
) -> AgentSpec:
    """Compact constructor for the clients added after the original 20.

    ``hook=""`` means the client has no instruction file mind-mem writes;
    the hook phase then reports the MCP file (or manual steps) instead.
    """
    if hook:
        return AgentSpec(
            name=name,
            description=description,
            vendor=vendor,
            config_fmt="text-block",
            path_tmpl=hook,
            content_tmpl=MEMORY_PROTOCOL_SNIPPET,
            detect_paths=detect,
            detect_binaries=bins,
            mcp_fmt=mcp_fmt,
            mcp_path_tmpl=mcp,
            **extra,
        )
    return AgentSpec(
        name=name,
        description=description,
        vendor=vendor,
        config_fmt="none",
        path_tmpl=mcp or "(manual setup)",
        detect_paths=detect,
        detect_binaries=bins,
        mcp_fmt=mcp_fmt,
        mcp_path_tmpl=mcp,
        **extra,  # type: ignore[arg-type]
    )


def _manual(name: str, description: str, vendor: str, steps: str, **extra: Any) -> AgentSpec:
    """A client with MCP support but no per-user file mind-mem can merge into."""
    return _client(name, description, vendor, hook="", manual=steps, **extra)
