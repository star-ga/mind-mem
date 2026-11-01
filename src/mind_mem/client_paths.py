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

from mind_mem.client_registry_schema import current_platform, env_override, expand, template_vars

__all__ = ["current_platform", "env_override", "expand", "template_vars"]
