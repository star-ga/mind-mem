# Copyright 2026 STARGA, Inc.
"""Registry data: one entry per supported AI client.

Kept apart from :mod:`mind_mem.client_registry` (the schema) so the list
can grow without the logic file growing. Paths use the placeholders of
:mod:`mind_mem.client_paths`. Every path is the client's own documented
location; where upstream documentation is incomplete the entry is guarded
(``mcp_guard="dir"``) so nothing is created for a client that is absent.
"""

from __future__ import annotations

from mind_mem.client_registry import _MM_MARKER, MEMORY_PROTOCOL_SNIPPET, AgentSpec, _client, _manual

_VSIX = "{home}/.vscode/extensions/"
_CLINE_SETTINGS = "globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json"
_ROO_SETTINGS = "globalStorage/rooveterinaryinc.roo-cline/settings/mcp_settings.json"

# ---------------------------------------------------------------------------
# The original 20 clients
# ---------------------------------------------------------------------------

CORE_AGENTS: tuple[AgentSpec, ...] = (
    AgentSpec(
        name="claude-code",
        description="Claude Code CLI (Anthropic)",
        vendor="Anthropic",
        config_fmt="json-claude-hooks",
        path_tmpl="{home}/.claude/settings.json",
        detect_paths=("{home}/.claude", "{home}/.config/claude"),
        detect_binaries=("claude",),
        # User-scope MCP servers live in ~/.claude.json (top-level
        # mcpServers), not in ~/.claude/settings.json.
        mcp_fmt="mcp-json-claude",
        mcp_path_tmpl="{home}/.claude.json",
    ),
    AgentSpec(
        name="codex",
        description="OpenAI Codex CLI",
        vendor="OpenAI",
        config_fmt="text-block",
        path_tmpl="{ws}/AGENTS.md",
        content_tmpl=MEMORY_PROTOCOL_SNIPPET,
        detect_binaries=("codex",),
        mcp_fmt="mcp-toml-codex",
        mcp_path_tmpl="{home}/.codex/config.toml",
    ),
    AgentSpec(
        name="grok-build",
        description="xAI Grok Build CLI",
        vendor="xAI",
        config_fmt="text-block",
        path_tmpl="{ws}/AGENTS.md",
        content_tmpl=MEMORY_PROTOCOL_SNIPPET,
        detect_binaries=("grok",),
        mcp_fmt="mcp-toml-codex",
        mcp_path_tmpl="{home}/.grok/config.toml",
    ),
    AgentSpec(
        name="vibe",
        description="Mistral Vibe CLI",
        vendor="Mistral AI",
        config_fmt="text-block",
        path_tmpl="{ws}/AGENTS.md",
        content_tmpl=MEMORY_PROTOCOL_SNIPPET,
        detect_paths=("{home}/.vibe",),
        detect_binaries=("vibe",),
        mcp_fmt="mcp-toml-vibe",
        mcp_path_tmpl="{home}/.vibe/config.toml",
        mcp_env_override=("VIBE_HOME", "config.toml"),
    ),
    # OpenCode (1.x and 2.x): instructions in the GLOBAL AGENTS.md (2.x reads
    # AGENTS.md only); MCP in the global opencode.json(c). ~/.opencode is
    # where the 2.x installer puts its binary.
    AgentSpec(
        name="opencode",
        description="OpenCode CLI (1.x and 2.x)",
        vendor="OpenCode (open source)",
        config_fmt="text-block",
        path_tmpl="{home}/.config/opencode/AGENTS.md",
        content_tmpl=MEMORY_PROTOCOL_SNIPPET,
        detect_paths=("{home}/.config/opencode", "{home}/.opencode"),
        detect_binaries=("opencode",),
        mcp_fmt="mcp-json-opencode",
        mcp_path_tmpl="{home}/.config/opencode/opencode.json",
    ),
    AgentSpec(
        name="gemini",
        description="Google Gemini CLI and Gemini Code Assist",
        vendor="Google",
        config_fmt="json-gemini",
        path_tmpl="{ws}/.gemini/settings.json",
        detect_paths=("{home}/.gemini", _VSIX + "google.geminicodeassist-*"),
        detect_binaries=("gemini",),
        mcp_fmt="mcp-json-servers",
        mcp_path_tmpl="{home}/.gemini/settings.json",
    ),
    AgentSpec(
        name="cursor",
        description="Cursor editor and cursor-agent CLI",
        vendor="Anysphere",
        config_fmt="text-block",
        path_tmpl="{ws}/.cursorrules",
        content_tmpl=MEMORY_PROTOCOL_SNIPPET,
        detect_paths=(
            "{home}/.cursor",
            "{home}/Library/Application Support/Cursor",
            "{home}/AppData/Roaming/Cursor",
        ),
        detect_binaries=("cursor", "cursor-agent"),
        mcp_fmt="mcp-json-cursor",
        mcp_path_tmpl="{home}/.cursor/mcp.json",
    ),
    # Windsurf's MCP file; Devin Desktop (its successor) reads the same JSON
    # shape from ~/.config/devin, written when that directory exists.
    AgentSpec(
        name="windsurf",
        description="Windsurf editor / Devin Desktop (Cognition)",
        vendor="Cognition (formerly Codeium)",
        config_fmt="text-block",
        path_tmpl="{ws}/.windsurfrules",
        content_tmpl=MEMORY_PROTOCOL_SNIPPET,
        detect_paths=(
            "{home}/.codeium/windsurf",
            "{home}/.windsurf",
            "{home}/Library/Application Support/Windsurf",
            "{config}/devin",
        ),
        detect_binaries=("windsurf", "devin"),
        mcp_fmt="mcp-json-windsurf",
        mcp_path_tmpl="{home}/.codeium/windsurf/mcp_config.json",
        mcp_extra_paths=("{config}/devin/mcp_config.json", "{appdata}/devin/mcp_config.json"),
    ),
    AgentSpec(
        name="aider",
        description="aider CLI (Aider-AI, open source; created by Paul Gauthier; no native MCP)",
        vendor="Aider-AI (open source)",
        config_fmt="yaml-block",
        path_tmpl="{ws}/.aider.conf.yml",
        content_tmpl=(f'{_MM_MARKER} auto-config\nread: ["{{ws}}/CLAUDE.md"]\n'),
        detect_binaries=("aider",),
    ),
    # OpenClaw: hooks under hooks.internal.entries, MCP servers under the
    # nested mcp.servers map of the same file.
    AgentSpec(
        name="openclaw",
        description="OpenClaw (open-source AI assistant)",
        vendor="OpenClaw Foundation (open source)",
        config_fmt="json-openclaw-hooks",
        path_tmpl="{home}/.openclaw/openclaw.json",
        detect_paths=("{home}/.openclaw",),
        detect_binaries=("openclaw",),
        mcp_fmt="mcp-json-openclaw",
        mcp_path_tmpl="{home}/.openclaw/openclaw.json",
    ),
    AgentSpec(
        name="nanoclaw",
        description="NanoClaw (open-source agent by Gavriel Cohen / Qwibit)",
        vendor="NanoClaw / Qwibit (open source)",
        config_fmt="json-openclaw-hooks",
        path_tmpl="{home}/.nanoclaw/nanoclaw.json",
        detect_paths=("{home}/.nanoclaw",),
        detect_binaries=("nanoclaw",),
    ),
    AgentSpec(
        name="nemoclaw",
        description="NVIDIA NemoClaw (open-source stack that runs OpenClaw)",
        vendor="NVIDIA",
        config_fmt="json-openclaw-hooks",
        path_tmpl="{home}/.nemoclaw/nemoclaw.json",
        detect_paths=("{home}/.nemoclaw",),
        detect_binaries=("nemoclaw",),
    ),
    # Continue: config.yaml is the current format (config.json is legacy).
    # The MCP list item is merged only into a config.yaml Continue already
    # created -- writing a new one would shadow a legacy config.json.
    AgentSpec(
        name="continue",
        description="Continue (VS Code / JetBrains extension)",
        vendor="Continue.dev",
        config_fmt="json-continue",
        path_tmpl="{home}/.continue/config.json",
        detect_paths=("{home}/.continue", _VSIX + "continue.continue-*"),
        mcp_fmt="mcp-yaml-continue",
        mcp_path_tmpl="{home}/.continue/config.yaml",
        mcp_guard="file",
    ),
    AgentSpec(
        name="cline",
        description="Cline (VS Code extension and CLI)",
        vendor="Cline",
        config_fmt="text-block",
        path_tmpl="{ws}/.clinerules",
        content_tmpl=MEMORY_PROTOCOL_SNIPPET,
        detect_paths=(
            _VSIX + "saoudrizwan.claude-dev-*",
            "{home}/.vscode-server/extensions/saoudrizwan.claude-dev-*",
            "{home}/.cline",
        ),
        detect_binaries=("cline",),
        mcp_fmt="mcp-json-cline",
        mcp_path_tmpl="{vscode_user}/" + _CLINE_SETTINGS,
        mcp_env_override=("CLINE_MCP_SETTINGS_PATH", ""),
        mcp_extra_paths=(
            "{home}/.vscode-server/data/User/" + _CLINE_SETTINGS,
            "{home}/.cline/data/settings/cline_mcp_settings.json",
        ),
    ),
    AgentSpec(
        name="roo",
        description="Roo Code (VS Code extension)",
        vendor="Roo Code",
        config_fmt="text-block",
        path_tmpl="{ws}/.roo/system-prompt.md",
        content_tmpl=MEMORY_PROTOCOL_SNIPPET,
        detect_paths=(
            "{home}/.roo",
            _VSIX + "rooveterinaryinc.roo-cline-*",
            "{home}/.vscode-server/extensions/rooveterinaryinc.roo-cline-*",
        ),
        mcp_fmt="mcp-json-servers",
        mcp_path_tmpl="{vscode_user}/" + _ROO_SETTINGS,
        mcp_extra_paths=("{home}/.vscode-server/data/User/" + _ROO_SETTINGS,),
    ),
    AgentSpec(
        name="zed",
        description="Zed editor AI assistant",
        vendor="Zed Industries",
        config_fmt="json-zed",
        path_tmpl="{home}/.config/zed/settings.json",
        detect_paths=(
            "{home}/.config/zed",
            "{home}/Library/Application Support/Zed",
            "{appdata}/Zed",
        ),
        detect_binaries=("zed", "zeditor", "zed-editor"),
        mcp_fmt="mcp-json-zed",
        mcp_path_tmpl="{home}/.config/zed/settings.json",
        mcp_platform_paths=(("win32", "{appdata}/Zed/settings.json"),),
    ),
    AgentSpec(
        name="copilot",
        description="GitHub Copilot (workspace instructions; VS Code MCP is the `vscode` entry)",
        vendor="GitHub / Microsoft",
        config_fmt="text-block",
        path_tmpl="{ws}/.github/copilot-instructions.md",
        content_tmpl=(
            f"{_MM_MARKER}: GitHub Copilot workspace instructions\n\n"
            "This repository uses mind-mem for persistent memory. "
            'Before answering, consult memory with `mm inject --agent copilot "<question>"`. '
            "Respect ADR / DECISION blocks; route new decisions through "
            "`propose_update` rather than modifying them directly.\n"
        ),
        always_offer=True,  # virtually every dev machine has Copilot potential
    ),
    # GitHub Copilot CLI (`@github/copilot` npm) -- its own ~/.copilot tree.
    AgentSpec(
        name="copilot-cli",
        description="GitHub Copilot CLI (@github/copilot)",
        vendor="GitHub / Microsoft",
        config_fmt="text-block",
        path_tmpl="{ws}/AGENTS.md",
        content_tmpl=MEMORY_PROTOCOL_SNIPPET,
        detect_paths=("{home}/.copilot",),
        detect_binaries=("copilot",),
        mcp_fmt="mcp-json-copilot",
        mcp_path_tmpl="{home}/.copilot/mcp-config.json",
        mcp_env_override=("COPILOT_HOME", "mcp-config.json"),
    ),
    AgentSpec(
        name="cody",
        description="Sourcegraph Cody",
        vendor="Sourcegraph",
        config_fmt="json-generic",
        path_tmpl="{ws}/.cody/config.json",
        detect_paths=("{home}/.config/cody",),
        detect_binaries=("cody",),
    ),
    AgentSpec(
        name="qodo",
        description="Qodo Gen (formerly CodiumAI)",
        vendor="Qodo",
        config_fmt="text-block",
        path_tmpl="{ws}/.codium/ai-rules.md",
        content_tmpl=(f"{_MM_MARKER}\nmind-mem workspace: {{ws}}\nConsult mind-mem memory before proposing changes.\n"),
        detect_paths=("{home}/.codium", "{home}/.qodo"),
    ),
)

# ---------------------------------------------------------------------------
# Clients with a documented per-user MCP file
# ---------------------------------------------------------------------------

MORE_AGENTS: tuple[AgentSpec, ...] = (
    _client(
        "claude-desktop",
        "Claude Desktop",
        "Anthropic",
        mcp_fmt="mcp-json-servers",
        mcp="{appsupport}/Claude/claude_desktop_config.json",
        hook="",
        detect=("{appsupport}/Claude",),
        # Linux builds are community packages: write only into an existing dir.
        mcp_guard="dir",
        mcp_guard_platforms=("linux",),
    ),
    _client(
        "vscode",
        "VS Code native MCP (Copilot agent mode)",
        "Microsoft",
        mcp_fmt="mcp-json-vscode",
        mcp="{vscode_user}/mcp.json",
        hook="",
        detect=("{vscode_user}",),
        bins=("code", "code-insiders", "codium"),
        mcp_guard="dir",
    ),
    _client(
        "visual-studio",
        "Visual Studio 2022 17.14+ / 2026 (Windows)",
        "Microsoft",
        mcp_fmt="mcp-json-visualstudio",
        mcp="{home}/.mcp.json",
        hook="",
        bins=("devenv.exe", "vswhere.exe"),
        platforms=("win32",),
    ),
    _client(
        "qwen-code",
        "Qwen Code CLI",
        "Alibaba",
        mcp_fmt="mcp-json-servers",
        mcp="{home}/.qwen/settings.json",
        hook="{home}/.qwen/QWEN.md",
        detect=("{home}/.qwen",),
        bins=("qwen",),
    ),
    _client(
        "kimi-code",
        "Kimi Code CLI",
        "Moonshot AI",
        mcp_fmt="mcp-json-servers",
        mcp="{home}/.kimi-code/mcp.json",
        detect=("{home}/.kimi-code", "{home}/.kimi"),
        bins=("kimi",),
        mcp_env_override=("KIMI_CODE_HOME", "mcp.json"),
        mcp_extra_paths=("{home}/.kimi/mcp.json",),
    ),
    _client(
        "amp",
        "Amp CLI and editor extension",
        "Amp",
        mcp_fmt="mcp-json-amp",
        mcp="{config}/amp/settings.json",
        hook="{config}/amp/AGENTS.md",
        detect=("{config}/amp",),
        bins=("amp",),
        mcp_env_override=("AMP_SETTINGS_FILE", ""),
    ),
    _client(
        "goose",
        "Goose agent",
        "Agentic AI Foundation (created by Block)",
        mcp_fmt="mcp-yaml-goose",
        mcp="{config}/goose/config.yaml",
        hook="{config}/goose/.goosehints",
        detect=("{config}/goose",),
        bins=("goose",),
        mcp_platform_paths=(("win32", "{appdata}/Block/goose/config/config.yaml"),),
    ),
    _client(
        "crush",
        "Crush CLI",
        "Charm",
        mcp_fmt="mcp-json-crush",
        mcp="{config}/crush/crush.json",
        detect=("{config}/crush",),
        bins=("crush",),
    ),
    _client(
        "factory-droid",
        "Factory Droid CLI",
        "Factory",
        mcp_fmt="mcp-json-factory",
        mcp="{home}/.factory/mcp.json",
        detect=("{home}/.factory",),
        bins=("droid",),
    ),
    _client(
        "kiro",
        "Kiro IDE and kiro-cli",
        "AWS",
        mcp_fmt="mcp-json-cline",
        mcp="{home}/.kiro/settings/mcp.json",
        hook="{home}/.kiro/steering/mind-mem.md",
        detect=("{home}/.kiro",),
        bins=("kiro", "kiro-cli"),
    ),
    _client(
        "auggie",
        "Augment Code (Auggie CLI)",
        "Augment Code",
        mcp_fmt="mcp-json-servers",
        mcp="{home}/.augment/settings.json",
        detect=("{home}/.augment",),
        bins=("auggie",),
    ),
    _client(
        "openhands",
        "OpenHands CLI",
        "All Hands AI",
        mcp_fmt="mcp-json-openhands",
        mcp="{home}/.openhands/mcp.json",
        detect=("{home}/.openhands",),
        bins=("openhands",),
    ),
    _client(
        "mimo-code",
        "MiMo Code CLI",
        "Xiaomi",
        mcp_fmt="mcp-json-opencode",
        mcp="{config}/mimocode/mimocode.jsonc",
        detect=("{config}/mimocode",),
        bins=("mimo", "mimocode"),
        mcp_path_alts=("{config}/mimocode/mimocode.jsonc", "{config}/mimocode/mimocode.json"),
    ),
    _client(
        "junie",
        "JetBrains Junie (IDE and CLI)",
        "JetBrains",
        mcp_fmt="mcp-json-servers",
        mcp="{home}/.junie/mcp/mcp.json",
        detect=("{home}/.junie",),
        bins=("junie",),
    ),
    _client(
        "kilo",
        "Kilo Code (CLI and extension)",
        "Kilo Code",
        mcp_fmt="mcp-json-opencode",
        mcp="{config}/kilo/kilo.jsonc",
        detect=("{config}/kilo", _VSIX + "kilocode.kilo-code-*"),
        bins=("kilo",),
        mcp_path_alts=("{config}/kilo/kilo.jsonc", "{config}/kilo/kilo.json", "{config}/kilo/config.json"),
    ),
    _client(
        "antigravity",
        "Google Antigravity IDE and agy CLI",
        "Google",
        mcp_fmt="mcp-json-servers",
        mcp="{home}/.gemini/config/mcp_config.json",
        detect=("{home}/.gemini/config", "{home}/.gemini/antigravity"),
        bins=("agy", "antigravity"),
    ),
    _client(
        "tabnine",
        "Tabnine agent",
        "Tabnine",
        mcp_fmt="mcp-json-servers",
        mcp="{home}/.tabnine/mcp_servers.json",
        hook="",
        detect=("{home}/.tabnine", _VSIX + "tabnine.tabnine-vscode-*"),
    ),
    _client(
        "hermes-agent",
        "Hermes Agent",
        "Nous Research",
        mcp_fmt="mcp-yaml-hermes",
        mcp="{home}/.hermes/config.yaml",
        hook="",
        detect=("{home}/.hermes",),
        bins=("hermes",),
    ),
    _client(
        "anythingllm",
        "AnythingLLM Desktop",
        "Mintplex Labs",
        mcp_fmt="mcp-json-anythingllm",
        # docs.anythingllm.com: <storage>/plugins/anythingllm_mcp_servers.json;
        # the desktop app's storage dir is <appdata>/anythingllm-desktop/storage.
        mcp="{appsupport}/anythingllm-desktop/storage/plugins/anythingllm_mcp_servers.json",
        hook="",
        detect=("{appsupport}/anythingllm-desktop",),
        mcp_guard="dir",
    ),
    _client(
        "warp",
        "Warp terminal agent",
        "Warp",
        mcp_fmt="mcp-json-servers",
        mcp="{home}/.warp/.mcp.json",
        detect=("{home}/.warp",),
        bins=("warp", "warp-terminal"),
        mcp_guard="dir",
    ),
    _client(
        "open-interpreter",
        "Open Interpreter",
        "Open Interpreter",
        mcp_fmt="mcp-toml-codex",
        mcp="{home}/.openinterpreter/config.toml",
        hook="",
        detect=("{home}/.openinterpreter",),
        bins=("interpreter",),
        mcp_guard="dir",
    ),
    # Guarded: MCP support is documented but the path, a key or the product
    # line is only partly so -- written only into an existing directory.
    _client(
        "amazon-q",
        "Amazon Q Developer (IDE and legacy q CLI)",
        "AWS",
        mcp_fmt="mcp-json-servers",
        mcp="{home}/.aws/amazonq/mcp.json",
        hook="",
        detect=("{home}/.aws/amazonq",),
        mcp_guard="dir",
    ),
    _client(
        "lm-studio",
        "LM Studio",
        "Element Labs",
        mcp_fmt="mcp-json-servers",
        mcp="{home}/.lmstudio/mcp.json",
        hook="",
        detect=("{home}/.lmstudio",),
        bins=("lms",),
        mcp_guard="dir",
    ),
    _client(
        "jan",
        "Jan desktop app",
        "Menlo Research",
        mcp_fmt="mcp-json-jan",
        mcp="{data}/Jan/data/mcp_config.json",
        hook="",
        detect=("{data}/Jan",),
        mcp_env_override=("JAN_DATA_FOLDER", "mcp_config.json"),
        mcp_guard="dir",
    ),
    _client(
        "boltai",
        "BoltAI (macOS)",
        "BoltAI",
        mcp_fmt="mcp-json-servers",
        mcp="{home}/.boltai/mcp.json",
        hook="",
        detect=("{home}/.boltai",),
        platforms=("darwin",),
        mcp_guard="dir",
    ),
    _client(
        "iflow",
        "iFlow CLI",
        "iFlow",
        mcp_fmt="mcp-json-servers",
        mcp="{home}/.iflow/settings.json",
        detect=("{home}/.iflow",),
        bins=("iflow",),
        mcp_guard="dir",
    ),
    _client(
        "neovim-mcphub",
        "Neovim via mcphub.nvim (avante, CodeCompanion, CopilotChat)",
        "mcphub.nvim (open source)",
        mcp_fmt="mcp-json-servers",
        mcp="{config}/mcphub/servers.json",
        hook="",
        detect=("{config}/mcphub",),
        mcp_guard="dir",
    ),
)

# ---------------------------------------------------------------------------
# Clients with MCP but no per-user file to merge into: printed steps
# ---------------------------------------------------------------------------

_PASTE = 'paste the mcpServers JSON below (server "mind-mem")'
_HTTP = "start `mind-mem-mcp --transport http` (bearer token required) and register its URL"

MANUAL_AGENTS: tuple[AgentSpec, ...] = (
    _manual(
        "trae",
        "Trae IDE",
        "ByteDance",
        f"Settings > MCP > Add > Raw config, or enable workspace MCP and add it to .trae/mcp.json: {_PASTE}",
    ),
    _manual(
        "android-studio",
        "Android Studio (Gemini agent)",
        "Google",
        f"Settings > Tools > AI > MCP Servers accepts HTTP servers only: {_HTTP} as `httpUrl`",
    ),
    _manual(
        "jetbrains-ai-assistant",
        "JetBrains AI Assistant",
        "JetBrains",
        "Settings > Tools > AI Assistant > Model Context Protocol: use 'Import from Claude' after "
        f"`mm install-all --agent claude-desktop`, or {_PASTE}",
    ),
    _manual(
        "librechat",
        "LibreChat",
        "LibreChat",
        "add `mcpServers.mind-mem` (type: stdio, command, args, env) to your deployment's librechat.yaml and restart",
    ),
    _manual(
        "eclipse-copilot",
        "Eclipse (GitHub Copilot plugin)",
        "GitHub / Microsoft",
        f"Preferences > GitHub Copilot > MCP: {_PASTE} (under `servers`)",
    ),
    _manual(
        "xcode",
        "Xcode 26.3+ agents",
        "Apple",
        "Xcode's agents use the Claude Code and Codex registrations: run `mm install-all --agent claude-code --agent codex`",
    ),
    _manual("witsy", "Witsy", "Witsy (open source)", f"Settings > MCP > Add server: {_PASTE}"),
    _manual(
        "emacs-mcp-el",
        "Emacs (mcp.el)",
        "mcp.el (open source)",
        'add ("mind-mem" :command "<command>" :args (...)) to `mcp-hub-servers` in your init file',
    ),
    _manual("open-webui", "Open WebUI", "Open WebUI (open source)", f"Admin > Settings > Tools: {_HTTP} (or expose it through mcpo)"),
    _manual(
        "docker-mcp-toolkit",
        "Docker MCP Toolkit",
        "Docker",
        "the toolkit is a gateway for containerised servers; register mind-mem in each client directly with `mm install-all`",
    ),
    _manual("cherry-studio", "Cherry Studio", "CherryHQ", f"Settings > MCP Servers > Add > Import from JSON: {_PASTE}"),
    _manual("msty", "Msty", "Msty", f"Settings > Tools > Add MCP server (JSON): {_PASTE}"),
    _manual("raycast", "Raycast", "Raycast", f"Raycast AI > Manage MCP Servers > Install Server: {_PASTE}"),
    _manual("5ire", "5ire", "5ire (open source)", f"Tools > New: {_PASTE}"),
    _manual(
        "perplexity-desktop", "Perplexity desktop (macOS)", "Perplexity", f"Settings > Connectors > Add connector (advanced): {_PASTE}"
    ),
    _manual(
        "chatgpt-desktop",
        "ChatGPT desktop",
        "OpenAI",
        f"remote HTTPS connectors only: {_HTTP} behind HTTPS and add it under Settings > Connectors",
    ),
    _manual("n8n", "n8n (MCP Client node)", "n8n", f"use the MCP Client Tool node: {_HTTP}"),
    _manual("dify", "Dify", "LangGenius", f"Tools > MCP > Add MCP server (HTTP): {_HTTP}"),
    _manual("flowise", "Flowise", "FlowiseAI", "add a Custom MCP tool node with the stdio command and MIND_MEM_WORKSPACE env shown below"),
    _manual("blackbox", "Blackbox AI (VS Code extension)", "Blackbox", f"open the extension's MCP settings and {_PASTE}"),
    # Manual because no user-level file could be verified: fast-agent finds
    # fast-agent.yaml from the working directory (fast-agent.ai/mcp), and the
    # mcp-yaml-fast-agent writer stays available for a project-level target.
    _manual(
        "fast-agent",
        "fast-agent",
        "fast-agent (open source)",
        "add the server under `mcp.servers.mind-mem` (transport: stdio, command, args, env) in your project's fast-agent.yaml",
        detect=("{home}/.fast-agent",),
        bins=("fast-agent",),
    ),
    # Manual because the user-level path is unverified: the project's docs do
    # not document one, and third-party guides disagree (~/.void vs ~/.void-editor).
    _manual(
        "void",
        "Void editor",
        "Void (open source)",
        f"open Void's MCP settings (Settings > MCP) and {_PASTE}",
        detect=("{home}/.void-editor", "{home}/.void"),
        bins=("void",),
    ),
)


__all__ = ["CORE_AGENTS", "MANUAL_AGENTS", "MORE_AGENTS"]
