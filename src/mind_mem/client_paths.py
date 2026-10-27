# Copyright 2026 STARGA, Inc.
"""Per-platform path expansion for AI-client config locations.

Registry templates (:mod:`mind_mem.client_registry`) use these
placeholders:

``{home}``       the user's home directory (``~``)
``{ws}``         the mind-mem workspace
``{config}``     ``~/.config`` (the XDG default; many CLIs hard-code it
                 on every OS, so ``$XDG_CONFIG_HOME`` is honoured only by
                 clients that document it, through ``mcp_env_override``)
``{appdata}``    ``%APPDATA%`` on Windows (``~/AppData/Roaming`` fallback)
``{appsupport}`` the per-OS application-data root: ``~/Library/Application
                 Support`` on macOS, ``%APPDATA%`` on Windows, ``~/.config``
                 on Linux
``{vscode_user}`` VS Code's ``User`` directory under ``{appsupport}``
``{data}``       ``~/.local/share`` on Linux, ``{appsupport}`` elsewhere

Expansion only reads environment variables and never touches the disk.
"""

from __future__ import annotations

import os
import sys


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


__all__ = ["current_platform", "env_override", "expand", "template_vars"]
