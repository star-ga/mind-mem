# Client Integrations

MIND-Mem works with **69 AI coding clients** out of the box. Every
client reads and writes to the same shared workspace (default
`~/.openclaw/workspace/`), so a fact captured in one tool is
immediately visible to every other.

The fastest path to configure any client is:

```bash
mm detect        # list every AI coding client mind-mem recognises on this machine
mm install-all   # auto-configure each detected client: hook + native MCP registration
```

**Since v3.1.0, `mm install-all` writes TWO things per MCP-aware client:**
1. The **text-hook** (SessionStart/PostToolUse/Stop) for visibility and auto-capture.
2. A **native MCP server entry** so the client gets the full 107-tool surface.

Pass `--no-mcp` to skip the MCP registration phase (hook-only, useful
when you want to register a custom MCP endpoint yourself or for
clients behind firewalls).

### Supported clients

These are the entries of `mind_mem.hook_installer.AGENT_REGISTRY`:

| Client | Vendor | `mm install` id | Install mode |
|--------|--------|-----------------|--------------|
| Claude Code | Anthropic | `claude-code` | auto MCP |
| Codex CLI | OpenAI | `codex` | auto MCP |
| Grok Build CLI | xAI | `grok-build` | auto MCP |
| Vibe (Mistral CLI) | Mistral AI | `vibe` | auto MCP |
| OpenCode (1.x / 2.x) | OpenCode (open source) | `opencode` | auto MCP |
| Gemini CLI | Google | `gemini` | auto MCP |
| Cursor | Anysphere | `cursor` | auto MCP |
| Windsurf | Cognition (formerly Codeium) | `windsurf` | auto MCP |
| aider | Aider-AI (open source) | `aider` | none |
| OpenClaw | OpenClaw Foundation (open source) | `openclaw` | auto MCP |
| NanoClaw | NanoClaw / Qwibit (open source) | `nanoclaw` | none |
| NemoClaw | NVIDIA | `nemoclaw` | none |
| Continue | Continue.dev | `continue` | auto MCP |
| Cline | Cline | `cline` | auto MCP |
| Roo Code | Roo Code | `roo` | auto MCP |
| Zed | Zed Industries | `zed` | auto MCP |
| GitHub Copilot (workspace instructions) | GitHub / Microsoft | `copilot` | none |
| GitHub Copilot CLI | GitHub / Microsoft | `copilot-cli` | auto MCP |
| Cody | Sourcegraph | `cody` | none |
| Qodo Gen | Qodo | `qodo` | none |
| Claude Desktop | Anthropic | `claude-desktop` | auto MCP |
| VS Code (native MCP) | Microsoft | `vscode` | auto MCP |
| Visual Studio (Windows) | Microsoft | `visual-studio` | auto MCP |
| Qwen Code | Alibaba | `qwen-code` | auto MCP |
| Kimi Code CLI | Moonshot AI | `kimi-code` | auto MCP |
| Amp | Amp | `amp` | auto MCP |
| goose | Agentic AI Foundation (created by Block) | `goose` | auto MCP |
| Crush | Charm | `crush` | auto MCP |
| Factory Droid | Factory | `factory-droid` | auto MCP |
| Kiro | AWS | `kiro` | auto MCP |
| Auggie (Augment Code) | Augment Code | `auggie` | auto MCP |
| OpenHands CLI | All Hands AI | `openhands` | auto MCP |
| MiMo Code | Xiaomi | `mimo-code` | auto MCP |
| Junie | JetBrains | `junie` | auto MCP |
| Kilo Code | Kilo Code | `kilo` | auto MCP |
| Antigravity | Google | `antigravity` | auto MCP |
| Tabnine agent | Tabnine | `tabnine` | auto MCP |
| Hermes Agent | Nous Research | `hermes-agent` | auto MCP |
| AnythingLLM Desktop | Mintplex Labs | `anythingllm` | auto MCP |
| Warp | Warp | `warp` | auto MCP |
| Open Interpreter | Open Interpreter | `open-interpreter` | auto MCP |
| Amazon Q Developer | AWS | `amazon-q` | auto MCP |
| LM Studio | Element Labs | `lm-studio` | auto MCP |
| Jan | Menlo Research | `jan` | auto MCP |
| BoltAI (macOS) | BoltAI | `boltai` | auto MCP |
| iFlow CLI | iFlow | `iflow` | auto MCP |
| Neovim (mcphub.nvim) | mcphub.nvim (open source) | `neovim-mcphub` | auto MCP |
| Trae | ByteDance | `trae` | manual MCP |
| Android Studio (Gemini) | Google | `android-studio` | manual MCP |
| JetBrains AI Assistant | JetBrains | `jetbrains-ai-assistant` | manual MCP |
| LibreChat | LibreChat | `librechat` | manual MCP |
| Eclipse (GitHub Copilot) | GitHub / Microsoft | `eclipse-copilot` | manual MCP |
| Xcode 26.3+ | Apple | `xcode` | manual MCP |
| Witsy | Witsy (open source) | `witsy` | manual MCP |
| Emacs (mcp.el) | mcp.el (open source) | `emacs-mcp-el` | manual MCP |
| Open WebUI | Open WebUI (open source) | `open-webui` | manual MCP |
| Docker MCP Toolkit | Docker | `docker-mcp-toolkit` | manual MCP |
| Cherry Studio | CherryHQ | `cherry-studio` | manual MCP |
| Msty Studio | Msty | `msty` | manual MCP |
| Raycast | Raycast | `raycast` | manual MCP |
| 5ire | 5ire (open source) | `5ire` | manual MCP |
| Perplexity (macOS app) | Perplexity | `perplexity-desktop` | manual MCP |
| ChatGPT | OpenAI | `chatgpt-desktop` | manual MCP |
| n8n | n8n | `n8n` | manual MCP |
| Dify | LangGenius | `dify` | manual MCP |
| Flowise | FlowiseAI | `flowise` | manual MCP |
| Blackbox AI | Blackbox | `blackbox` | manual MCP |
| fast-agent | fast-agent (open source) | `fast-agent` | manual MCP |
| Void | Void (open source) | `void` | manual MCP |

- **auto MCP** (41) -- the mind-mem server entry is merged into the client's
  own config file (paths below).
- **manual MCP** (22) -- the client supports MCP but keeps its server list in
  the app UI, a per-deployment file, or accepts HTTP servers only, so there is
  no per-user file to merge into; `mm install-all --agent <id>` writes nothing
  and prints the exact steps and snippet.
- **none** (6) -- no native MCP; an instruction or hook file routes memory
  access through the `mm` CLI.

Use `mm install <agent>` to configure a single client; use
`mm install-all --agent <A> --agent <B>` to target an explicit subset.

Every writer is **non-destructive**: existing config files are parsed
and the MIND-Mem stanza is merged in under the `# mind-mem` marker.
Re-running the command is idempotent. Pass `--force` to overwrite a
hand-rolled config you want replaced.

### Native MCP formats per client

`mm install-all` writes format-specific MCP entries for 41 MCP-aware clients.
Every write merges only the `mind-mem` entry: other servers, settings and (in
TOML and YAML files) comments are kept, a changed file is backed up first
(`<file>.bak-mind-mem-<timestamp>`) and replaced atomically, and a second run
leaves the file byte-identical. A file that cannot be merged safely (JSONC
with comments, invalid JSON, YAML flow style) is left untouched and the
snippet to paste is printed instead.

Modes:

- **auto** -- the entry is merged into the file, creating it if needed.
- **guarded** -- the entry is written only when the client's config directory
  already exists (the client is absent otherwise, or the location is only
  partly documented upstream).
- **existing file only** -- Continue: merged only into a `config.yaml`
  Continue has already created (writing a new one would shadow a legacy
  `config.json`).
- Paths shown are Linux; macOS and Windows locations are resolved per platform
  (`~/Library/Application Support/...`, `%APPDATA%\...`). A client's own
  relocation variable is honoured: `VIBE_HOME`, `COPILOT_HOME`,
  `CLINE_MCP_SETTINGS_PATH`, `AMP_SETTINGS_FILE`, `KIMI_CODE_HOME`,
  `JAN_DATA_FOLDER`.

| Client | `mm install` id | Config path | Format | Stanza | Mode |
| --- | --- | --- | --- | --- | --- |
| Claude Code | `claude-code` | `~/.claude.json` | JSON | `mcpServers.mind-mem` (type stdio) | auto |
| Codex CLI | `codex` | `~/.codex/config.toml` | TOML | `[mcp_servers.mind-mem]` | auto |
| Grok Build CLI | `grok-build` | `~/.grok/config.toml` | TOML | `[mcp_servers.mind-mem]` | auto |
| Vibe (Mistral CLI) | `vibe` | `~/.vibe/config.toml` | TOML | `[[mcp_servers]] name = "mind-mem"` | auto |
| OpenCode (1.x / 2.x) | `opencode` | `~/.config/opencode/opencode.json` | JSON | `mcp.mind-mem` (type local) | auto |
| Gemini CLI | `gemini` | `~/.gemini/settings.json` | JSON | `mcpServers.mind-mem` | auto |
| Cursor | `cursor` | `~/.cursor/mcp.json` | JSON | `mcpServers.mind-mem` (type stdio) | auto |
| Windsurf | `windsurf` | `~/.codeium/windsurf/mcp_config.json` | JSON | `mcpServers.mind-mem` | also `~/.config/devin/mcp_config.json` if present; also `%APPDATA%/devin/mcp_config.json` if present |
| OpenClaw | `openclaw` | `~/.openclaw/openclaw.json` | JSON | `mcp.servers.mind-mem` | auto |
| Continue | `continue` | `~/.continue/config.yaml` | YAML | `mcpServers: - name: mind-mem` | existing file only |
| Cline | `cline` | `~/.config/Code/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json` | JSON | `mcpServers.mind-mem` (disabled false, autoApprove []) | also `~/.vscode-server/data/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json` if present; also `~/.cline/data/settings/cline_mcp_settings.json` if present |
| Roo Code | `roo` | `~/.config/Code/User/globalStorage/rooveterinaryinc.roo-cline/settings/mcp_settings.json` | JSON | `mcpServers.mind-mem` | also `~/.vscode-server/data/User/globalStorage/rooveterinaryinc.roo-cline/settings/mcp_settings.json` if present |
| Zed | `zed` | `~/.config/zed/settings.json` | JSON | `context_servers.mind-mem` | auto |
| GitHub Copilot CLI | `copilot-cli` | `~/.copilot/mcp-config.json` | JSON | `mcpServers.mind-mem` (type local, tools ["*"]) | auto |
| Claude Desktop | `claude-desktop` | `~/.config/Claude/claude_desktop_config.json` | JSON | `mcpServers.mind-mem` | guarded on linux |
| VS Code (native MCP) | `vscode` | `~/.config/Code/User/mcp.json` | JSON | `servers.mind-mem` (type stdio) | guarded |
| Visual Studio (Windows) | `visual-studio` | `~/.mcp.json` | JSON | `servers.mind-mem` | Windows only |
| Qwen Code | `qwen-code` | `~/.qwen/settings.json` | JSON | `mcpServers.mind-mem` | auto |
| Kimi Code CLI | `kimi-code` | `~/.kimi-code/mcp.json` | JSON | `mcpServers.mind-mem` | also `~/.kimi/mcp.json` if present |
| Amp | `amp` | `~/.config/amp/settings.json` | JSON | `"amp.mcpServers".mind-mem` | auto |
| goose | `goose` | `~/.config/goose/config.yaml` | YAML | `extensions.mind-mem` | auto |
| Crush | `crush` | `~/.config/crush/crush.json` | JSON | `mcp.mind-mem` (type stdio) | auto |
| Factory Droid | `factory-droid` | `~/.factory/mcp.json` | JSON | `mcpServers.mind-mem` (type stdio, disabled false) | auto |
| Kiro | `kiro` | `~/.kiro/settings/mcp.json` | JSON | `mcpServers.mind-mem` (disabled false, autoApprove []) | auto |
| Auggie (Augment Code) | `auggie` | `~/.augment/settings.json` | JSON | `mcpServers.mind-mem` | auto |
| OpenHands CLI | `openhands` | `~/.openhands/mcp.json` | JSON | `mcpServers.mind-mem` (transport stdio) | auto |
| MiMo Code | `mimo-code` | `~/.config/mimocode/mimocode.jsonc` | JSON | `mcp.mind-mem` (type local) | auto |
| Junie | `junie` | `~/.junie/mcp/mcp.json` | JSON | `mcpServers.mind-mem` | auto |
| Kilo Code | `kilo` | `~/.config/kilo/kilo.jsonc` | JSON | `mcp.mind-mem` (type local) | auto |
| Antigravity | `antigravity` | `~/.gemini/config/mcp_config.json` | JSON | `mcpServers.mind-mem` | auto |
| Tabnine agent | `tabnine` | `~/.tabnine/mcp_servers.json` | JSON | `mcpServers.mind-mem` | auto |
| Hermes Agent | `hermes-agent` | `~/.hermes/config.yaml` | YAML | `mcp_servers.mind-mem` | auto |
| AnythingLLM Desktop | `anythingllm` | `~/.config/anythingllm-desktop/storage/plugins/anythingllm_mcp_servers.json` | JSON | `mcpServers.mind-mem` (anythingllm.autoStart) | guarded |
| Warp | `warp` | `~/.warp/.mcp.json` | JSON | `mcpServers.mind-mem` | guarded |
| Open Interpreter | `open-interpreter` | `~/.openinterpreter/config.toml` | TOML | `[mcp_servers.mind-mem]` | guarded |
| Amazon Q Developer | `amazon-q` | `~/.aws/amazonq/mcp.json` | JSON | `mcpServers.mind-mem` | guarded |
| LM Studio | `lm-studio` | `~/.lmstudio/mcp.json` | JSON | `mcpServers.mind-mem` | guarded |
| Jan | `jan` | `~/.local/share/Jan/data/mcp_config.json` | JSON | `mcpServers.mind-mem` (active true) | guarded |
| BoltAI (macOS) | `boltai` | `~/.boltai/mcp.json` | JSON | `mcpServers.mind-mem` | guarded; macOS only |
| iFlow CLI | `iflow` | `~/.iflow/settings.json` | JSON | `mcpServers.mind-mem` | guarded |
| Neovim (mcphub.nvim) | `neovim-mcphub` | `~/.config/mcphub/servers.json` | JSON | `mcpServers.mind-mem` | guarded |

All entries point at `<mind-mem-install>/mcp_server.py` with
`MIND_MEM_WORKSPACE` set to the shared workspace. Clients not on this
list still get the text-hook fallback, which routes tool calls
through the `mm` CLI.

### Clients with manual MCP setup

| Client | `mm install` id | Steps |
| --- | --- | --- |
| Trae | `trae` | Settings > MCP > Add > Raw config, or enable workspace MCP and add it to .trae/mcp.json: paste the mcpServers JSON below (server "mind-mem") |
| Android Studio (Gemini) | `android-studio` | Settings > Tools > AI > MCP Servers accepts HTTP servers only: start `mind-mem-mcp --transport http` (bearer token required) and register its URL as `httpUrl` |
| JetBrains AI Assistant | `jetbrains-ai-assistant` | Settings > Tools > AI Assistant > Model Context Protocol: use 'Import from Claude' after `mm install-all --agent claude-desktop`, or paste the mcpServers JSON below (server "mind-mem") |
| LibreChat | `librechat` | add `mcpServers.mind-mem` (type: stdio, command, args, env) to your deployment's librechat.yaml and restart |
| Eclipse (GitHub Copilot) | `eclipse-copilot` | Preferences > GitHub Copilot > MCP: paste the mcpServers JSON below (server "mind-mem") (under `servers`) |
| Xcode 26.3+ | `xcode` | Xcode's agents use the Claude Code and Codex registrations: run `mm install-all --agent claude-code --agent codex` |
| Witsy | `witsy` | Settings > MCP > Add server: paste the mcpServers JSON below (server "mind-mem") |
| Emacs (mcp.el) | `emacs-mcp-el` | add ("mind-mem" :command "<command>" :args (...)) to `mcp-hub-servers` in your init file |
| Open WebUI | `open-webui` | Admin > Settings > Tools: start `mind-mem-mcp --transport http` (bearer token required) and register its URL (or expose it through mcpo) |
| Docker MCP Toolkit | `docker-mcp-toolkit` | the toolkit is a gateway for containerised servers; register mind-mem in each client directly with `mm install-all` |
| Cherry Studio | `cherry-studio` | Settings > MCP Servers > Add > Import from JSON: paste the mcpServers JSON below (server "mind-mem") |
| Msty Studio | `msty` | Settings > Tools > Add MCP server (JSON): paste the mcpServers JSON below (server "mind-mem") |
| Raycast | `raycast` | Raycast AI > Manage MCP Servers > Install Server: paste the mcpServers JSON below (server "mind-mem") |
| 5ire | `5ire` | Tools > New: paste the mcpServers JSON below (server "mind-mem") |
| Perplexity (macOS app) | `perplexity-desktop` | Settings > Connectors > Add connector (advanced): paste the mcpServers JSON below (server "mind-mem") |
| ChatGPT | `chatgpt-desktop` | remote HTTPS connectors only: start `mind-mem-mcp --transport http` (bearer token required) and register its URL behind HTTPS and add it under Settings > Connectors |
| n8n | `n8n` | use the MCP Client Tool node: start `mind-mem-mcp --transport http` (bearer token required) and register its URL |
| Dify | `dify` | Tools > MCP > Add MCP server (HTTP): start `mind-mem-mcp --transport http` (bearer token required) and register its URL |
| Flowise | `flowise` | add a Custom MCP tool node with the stdio command and MIND_MEM_WORKSPACE env shown below |
| Blackbox AI | `blackbox` | open the extension's MCP settings and paste the mcpServers JSON below (server "mind-mem") |
| fast-agent | `fast-agent` | add the server under `mcp.servers.mind-mem` (transport: stdio, command, args, env) in your project's fast-agent.yaml |
| Void | `void` | open Void's MCP settings (Settings > MCP) and paste the mcpServers JSON below (server "mind-mem") |

### Clients without native MCP

These keep their instruction-file integration, which routes memory access
through the `mm` CLI:

- aider (`aider`) -- `<workspace>/.aider.conf.yml`
- NanoClaw (`nanoclaw`) -- `~/.nanoclaw/nanoclaw.json`
- NemoClaw (`nemoclaw`) -- `~/.nemoclaw/nemoclaw.json`
- GitHub Copilot (workspace instructions) (`copilot`) -- `<workspace>/.github/copilot-instructions.md`
- Cody (`cody`) -- `<workspace>/.cody/config.json`
- Qodo Gen (`qodo`) -- `<workspace>/.codium/ai-rules.md`

### Agent frameworks and SDKs

CrewAI, OpenAI Agents SDK, Mastra, VoltAgent, LangChain MCP adapters,
smolagents, Pydantic AI, mcp-agent and Google ADK are libraries rather than
installed clients, so there is no file to write. Point their stdio MCP client
at the same command `mm install-all --dry-run` prints:

```python
# Python frameworks: one stdio server definition
command = "mind-mem-mcp"           # or: python3 /path/to/mcp_server.py
args = []
env = {"MIND_MEM_WORKSPACE": "/path/to/workspace"}
```

### Removing mind-mem from clients

```bash
mm uninstall-all --dry-run            # show which entries would be removed
mm uninstall-all                      # every detected client
mm uninstall-all --agent goose        # one client
```

Only the `mind-mem` entry is removed; other servers and settings stay. When
the file existed before install, uninstall writes back its original bytes
(from the `<file>.bak-mind-mem-*` backup install took, used only when it holds
the same document as the post-removal content, so later edits are never
lost). When mind-mem created the file, uninstall deletes it. Instruction-file
blocks (`# mind-mem`) are left in place.

### Verified sources

Each registry entry was checked against the client's current documentation
(October 2026): the config path, the file format and the MCP key for every
**auto MCP** client, and the setup route for every **manual MCP** client. A
client whose user-level file could not be verified is listed as manual, never
guessed. Linux paths above; the source states the per-OS locations.

| Client | `mm install` id | Install mode | Source |
| --- | --- | --- | --- |
| Claude Code | `claude-code` | auto MCP | <https://code.claude.com/docs/en/mcp> |
| Codex CLI | `codex` | auto MCP | <https://developers.openai.com/codex/mcp> |
| Grok Build CLI | `grok-build` | auto MCP | <https://docs.x.ai/build/features/mcp-servers> |
| Vibe (Mistral CLI) | `vibe` | auto MCP | <https://github.com/mistralai/mistral-vibe> |
| OpenCode (1.x / 2.x) | `opencode` | auto MCP | <https://opencode.ai/docs/mcp-servers> |
| Gemini CLI | `gemini` | auto MCP | <https://github.com/google-gemini/gemini-cli/blob/main/docs/tools/mcp-server.md> |
| Cursor | `cursor` | auto MCP | <https://cursor.com/docs/context/mcp> |
| Windsurf | `windsurf` | auto MCP | <https://docs.devin.ai/desktop/cascade/mcp> — Windsurf is now Devin Desktop; its doc lists `~/.config/devin/mcp_config.json`, written too when that directory exists |
| aider | `aider` | none | <https://aider.chat/docs/config/aider_conf.html> — no native MCP; instruction file only |
| OpenClaw | `openclaw` | auto MCP | <https://docs.openclaw.ai/tools/mcp> |
| NanoClaw | `nanoclaw` | none | <https://nanoclaw.dev> — no native MCP writer; hook file only |
| NemoClaw | `nemoclaw` | none | <https://docs.nvidia.com/nemoclaw> — no native MCP writer; hook file only |
| Continue | `continue` | auto MCP | <https://docs.continue.dev/customize/deep-dives/mcp> — global `~/.continue/config.yaml`, `mcpServers` list; merged only into an existing file |
| Cline | `cline` | auto MCP | <https://docs.cline.bot/mcp/configuring-mcp-servers> |
| Roo Code | `roo` | auto MCP | <https://docs.roocode.com/features/mcp/using-mcp-in-roo> |
| Zed | `zed` | auto MCP | <https://zed.dev/docs/ai/mcp> |
| GitHub Copilot (workspace instructions) | `copilot` | none | <https://docs.github.com/en/copilot/how-tos/configure-custom-instructions/add-repository-instructions> — workspace instructions; VS Code MCP is the `vscode` entry |
| GitHub Copilot CLI | `copilot-cli` | auto MCP | <https://docs.github.com/copilot/how-tos/copilot-cli/customize-copilot/add-mcp-servers> — `~/.copilot/mcp-config.json` per the Copilot CLI overview; entry shape `type`, `tools` |
| Cody | `cody` | none | <https://sourcegraph.com/docs/cody> — no native MCP writer |
| Qodo Gen | `qodo` | none | <https://docs.qodo.ai> — no native MCP writer |
| Claude Desktop | `claude-desktop` | auto MCP | <https://modelcontextprotocol.io/docs/develop/connect-local-servers> — official builds are macOS and Windows; on Linux the file is written only into an existing directory |
| VS Code (native MCP) | `vscode` | auto MCP | <https://code.visualstudio.com/docs/copilot/customization/mcp-servers> |
| Visual Studio (Windows) | `visual-studio` | auto MCP | <https://learn.microsoft.com/en-us/visualstudio/ide/mcp-servers> |
| Qwen Code | `qwen-code` | auto MCP | <https://qwenlm.github.io/qwen-code-docs/en/users/features/mcp> |
| Kimi Code CLI | `kimi-code` | auto MCP | <https://moonshotai.github.io/kimi-code/en/configuration/data-locations.html> — `$KIMI_CODE_HOME/mcp.json` (default `~/.kimi-code/`); the older kimi-cli `~/.kimi/mcp.json` is written when present |
| Amp | `amp` | auto MCP | <https://ampcode.com/docs/customize/mcp> |
| goose | `goose` | auto MCP | <https://github.com/aaif-goose/goose/blob/main/documentation/docs/guides/config-files.md> — Windows: `%APPDATA%\Block\goose\config\config.yaml` |
| Crush | `crush` | auto MCP | <https://github.com/charmbracelet/crush> |
| Factory Droid | `factory-droid` | auto MCP | <https://docs.factory.ai/cli/configuration/mcp> |
| Kiro | `kiro` | auto MCP | <https://kiro.dev/docs/mcp/configuration> |
| Auggie (Augment Code) | `auggie` | auto MCP | <https://docs.augmentcode.com/cli/integrations> |
| OpenHands CLI | `openhands` | auto MCP | <https://docs.openhands.dev/openhands/usage/cli/mcp-servers> |
| MiMo Code | `mimo-code` | auto MCP | <https://mimo.xiaomi.com/mimocode/mcp-servers> |
| Junie | `junie` | auto MCP | <https://junie.jetbrains.com/docs/junie-cli-mcp-configuration.html> |
| Kilo Code | `kilo` | auto MCP | <https://kilo.ai/docs/automate/mcp/using-in-cli> — doc lists `kilo.json`, `kilo.jsonc` and `config.json`; an existing one is preferred |
| Antigravity | `antigravity` | auto MCP | <https://antigravity.google/docs/cli/mcp> — CLI doc: global servers in `~/.gemini/config/mcp_config.json` |
| Tabnine agent | `tabnine` | auto MCP | <https://docs.tabnine.com/main/getting-started/tabnine-agent/mcp-intro-and-setup> |
| Hermes Agent | `hermes-agent` | auto MCP | <https://hermes-agent.nousresearch.com/docs/user-guide/features/mcp> |
| AnythingLLM Desktop | `anythingllm` | auto MCP | <https://docs.anythingllm.com/mcp-compatibility/desktop> — `<storage>/plugins/anythingllm_mcp_servers.json`; desktop storage is `anythingllm-desktop/storage` under the OS app-data dir |
| Warp | `warp` | auto MCP | <https://docs.warp.dev/agents/capabilities/mcp/> |
| Open Interpreter | `open-interpreter` | auto MCP | <https://openinterpreter.com/docs/terminal/mcp> |
| Amazon Q Developer | `amazon-q` | auto MCP | <https://docs.aws.amazon.com/amazonq/latest/qdeveloper-ug/mcp-ide.html> — legacy `mcp.json`, still read while `useLegacyMcpJson` is true; written only into an existing directory |
| LM Studio | `lm-studio` | auto MCP | <https://lmstudio.ai/docs/app/mcp> |
| Jan | `jan` | auto MCP | <https://www.jan.ai/docs/desktop/integrations/mcp-servers> — doc confirms `mcp_config.json` in the Jan data folder; the `mcpServers` map and `active` flag are from Jan's source (janhq/jan `src-tauri/src/core/mcp`) |
| BoltAI (macOS) | `boltai` | auto MCP | <https://docs.boltai.com/docs/plugins/mcp-servers> |
| iFlow CLI | `iflow` | auto MCP | <https://platform.iflow.cn/cli/examples/mcp> |
| Neovim (mcphub.nvim) | `neovim-mcphub` | auto MCP | <https://ravitemer.github.io/mcphub.nvim/mcp/servers_json.html> |
| Trae | `trae` | manual MCP | <https://docs.trae.ai/ide/model-context-protocol> — per-project `.trae/mcp.json` or the Settings UI |
| Android Studio (Gemini) | `android-studio` | manual MCP | <https://developer.android.com/studio/gemini/add-mcp-server> — HTTP servers only (stdio unsupported per the doc) |
| JetBrains AI Assistant | `jetbrains-ai-assistant` | manual MCP | <https://www.jetbrains.com/help/ai-assistant/mcp.html> |
| LibreChat | `librechat` | manual MCP | <https://librechat.ai/docs/configuration/librechat_yaml/object_structure/mcp_servers> |
| Eclipse (GitHub Copilot) | `eclipse-copilot` | manual MCP | <https://docs.github.com/en/copilot/how-tos/provide-context/use-mcp/extend-copilot-chat-with-mcp> |
| Xcode 26.3+ | `xcode` | manual MCP | <https://developer.apple.com/xcode/> — Xcode 26.3 hosts Claude Agent and Codex, which read the Claude Code and Codex registrations |
| Witsy | `witsy` | manual MCP | <https://github.com/nbonamy/witsy> |
| Emacs (mcp.el) | `emacs-mcp-el` | manual MCP | <https://github.com/lizqwerscott/mcp.el> |
| Open WebUI | `open-webui` | manual MCP | <https://docs.openwebui.com/features/extensibility/plugin/tools/openapi-servers/mcp> |
| Docker MCP Toolkit | `docker-mcp-toolkit` | manual MCP | <https://docs.docker.com/ai/mcp-catalog-and-toolkit/toolkit> |
| Cherry Studio | `cherry-studio` | manual MCP | <https://docs.cherry-ai.com> |
| Msty Studio | `msty` | manual MCP | <https://docs.msty.ai/studio/toolbox/tools> |
| Raycast | `raycast` | manual MCP | <https://manual.raycast.com/ai/model-context-protocol> |
| 5ire | `5ire` | manual MCP | <https://github.com/nanbingxyz/5ire> |
| Perplexity (macOS app) | `perplexity-desktop` | manual MCP | <https://perplexity.ai/help-center/en/articles/11502712-local-and-remote-mcps-for-perplexity> |
| ChatGPT | `chatgpt-desktop` | manual MCP | <https://help.openai.com/en/articles/11487775-connectors-in-chatgpt> |
| n8n | `n8n` | manual MCP | <https://docs.n8n.io/integrations/builtin/cluster-nodes/sub-nodes/n8n-nodes-langchain.toolmcp/> |
| Dify | `dify` | manual MCP | <https://docs.dify.ai/en/cloud/use-dify/build/mcp> |
| Flowise | `flowise` | manual MCP | <https://docs.flowiseai.com/tutorials/tools-and-mcp> |
| Blackbox AI | `blackbox` | manual MCP | <https://www.blackbox.ai> — vendor MCP documentation could not be verified; manual steps only |
| fast-agent | `fast-agent` | manual MCP | <https://fast-agent.ai/mcp/> — config is `fast-agent.yaml` found from the working directory; no per-user file is documented, so manual |
| Void | `void` | manual MCP | <https://github.com/voideditor/void> — no maintained documentation of a user-level MCP file (guides disagree on `~/.void` vs `~/.void-editor`), so manual |

## Claude Code (Anthropic)

| | |
|---|---|
| Config path | `~/.claude/settings.json` |
| Format | JSON `hooks` section |
| MCP path | `~/.claude.json` (top-level `mcpServers`, user scope, `type: stdio`) |
| Install | `mm install claude-code` (hooks) / `mm install-all --agent claude-code` (hooks + MCP) |

Configures two hooks (v3.1.1+):
- `SessionStart` → `mm status` — report workspace + memory status at
  session start. (Pre-3.1.1 used `mm inject --agent claude-code
  --workspace <ws>`, which silently failed because `mm inject`
  requires a positional query the hook cannot supply. Re-running
  `mm install claude-code` on any pre-3.1.1 install auto-migrates the
  hook shape in place.)
- `Stop` → `mm status` — status check on session exit. (Pre-3.1.1
  used `mm vault status`, which is not a shipped subcommand. Same
  auto-migration applies.)

A dedicated `mm inject-on-start` subcommand suitable for the
`SessionStart` event is planned for a future release.

Also add the `Memory Protocol` block to `~/.claude/CLAUDE.md` so the
model *must* call `mcp__mind-mem__recall` before answering memory-
relevant questions. See [the MCP integration guide](mcp-integration.md#claude-code).

## Codex CLI (OpenAI)

| | |
|---|---|
| Config path | `<ws>/AGENTS.md` |
| Format | Markdown block |
| Install | `mm install codex` |

Appends an `AGENTS.md` block telling the codex CLI to run
`mm context "$QUERY"` before every response.

## OpenCode (1.x and 2.x)

| | |
|---|---|
| Instructions | `~/.config/opencode/AGENTS.md` (global, loaded in every session) |
| MCP config | `~/.config/opencode/opencode.json` (or an existing `opencode.jsonc`) |
| Stanza | `mcp.mind-mem` (works on 1.x and 2.x), or `mcp.servers.mind-mem` when the file already uses the 2.x-native `mcp.servers` map |
| Install | `mm install-all --agent opencode` (`mm install opencode` writes the instructions only) |

Detected by the `opencode` binary on `PATH`, `~/.config/opencode/`, or
`~/.opencode/` (where the 2.x installer puts its binary).

The MCP entry is a local stdio server:

```json
{
  "mcp": {
    "mind-mem": {
      "type": "local",
      "command": ["/path/to/python3", "/path/to/mind-mem/mcp_server.py"],
      "environment": { "MIND_MEM_WORKSPACE": "/path/to/workspace" },
      "enabled": true
    }
  }
}
```

OpenCode 1.x reads this shape directly; OpenCode 2.x normalizes it in
memory without rewriting your file, so one config serves both. If your
file already uses the 2.x-native `mcp.servers` map, the entry is written
there instead (and any `mcp.mind-mem` duplicate is removed), since that
file is already 2.x-only.

Safety: the writer merges into the existing file and keeps every other
key and server. Before changing an existing file it saves a copy as
`opencode.json.bak-mind-mem-<timestamp>`. Re-running is a no-op. A file
that is not strict JSON (JSONC with comments or trailing commas) is
**never rewritten**, because comments would be lost; the command reports
`skipped` with the snippet to paste under `mcp` by hand.

The instructions half is the same Memory Protocol block other CLIs get,
appended once to the global `AGENTS.md` (OpenCode 2.x reads `AGENTS.md`
only, not `CLAUDE.md`). Existing text in that file is preserved.

If the OpenCode 2.x background server is already running, run
`opencode reload` (or start a new session) so it picks up the new
server, then check it with `opencode mcp list`. OpenCode names the tools
`mind-mem_<tool>` (for example `mind-mem_recall`); under 2.x Code Mode
they are grouped as `tools.mind_mem.<tool>`.

Limits: no OpenCode plugin ships yet, so there is no automatic
session-start recall or capture hook like Claude Code's; the agent calls
the MCP tools as the `AGENTS.md` protocol instructs. Project-level
`opencode.json` files are left alone; only the global config is written.

## Gemini CLI (Google)

| | |
|---|---|
| Config path | `<ws>/.gemini/settings.json` |
| Format | JSON `system_instruction` |
| Install | `mm install gemini` |

Injects a system instruction pointing at the MIND-Mem workspace. The
instruction survives `gemini --yolo` and interactive mode.

## Qwen Code (Alibaba)

| | |
|---|---|
| Instructions | `~/.qwen/QWEN.md` |
| MCP config | `~/.qwen/settings.json` (user) or per-project `.mcp.json` |
| Format | JSON `mcpServers.mind-mem` (stdio) |
| Install | `mm install-all --agent qwen-code` |

`mm install-all --agent qwen-code` merges the `mcpServers.mind-mem` entry into
`~/.qwen/settings.json`. The notes below cover setting it up by hand.

Qwen Code is a Gemini-CLI descendant, so it reads MCP servers from
two places and merges them, with the **project `.mcp.json` entry
winning** on a name clash:

1. `~/.qwen/settings.json` → `mcpServers` — user scope, applies to
   every project.
2. `<project>/.mcp.json` → `mcpServers` — project scope. This file
   is shared with Claude Code and other Claude-compatible tools.

Both the **pip install** (`pip install "mind-mem[mcp]"`) and the
**repo checkout** (`./install.sh`, an editable install) provide the
canonical `mind-mem-mcp` console script (stdio transport by default,
107 tools), so the stanza is identical either way:

```json
{
  "mcpServers": {
    "mind-mem": {
      "command": "mind-mem-mcp",
      "args": [],
      "env": {
        "MIND_MEM_WORKSPACE": "/absolute/path/to/workspace",
        "MIND_MEM_SCOPE": "admin"
      },
      "timeout": 90000,
      "trust": true
    }
  }
}
```

Use an absolute `command` path (e.g. `~/.local/bin/mind-mem-mcp`
expanded) if Qwen Code's spawn PATH does not include the pip bin dir.
Then restart Qwen Code and check `/mcp` — `mind-mem` should show
connected with 107 tools. Verify independently with `mm doctor`.

### Qwen Code gotchas (learned the hard way)

- **`timeout` is milliseconds.** Qwen Code reads
  `mcpServers.<name>.timeout` as **milliseconds** (default 600000).
  Claude Code's `.mcp.json` uses **seconds**. A shared `.mcp.json`
  carrying `"timeout": 30` gives Claude Code 30 s but Qwen Code
  30 ms — every tool call then times out and startup discovery
  flaps. In shared `.mcp.json` files, omit `timeout` entirely (each
  CLI applies its own sane default) or keep per-CLI overrides in each
  CLI's own user-level config file.
- **Approval hash-binding.** Project `.mcp.json` servers are gated:
  Qwen Code stores an approval keyed by a SHA-256 hash of the server
  config in `~/.qwen/mcpApprovals.json`. Any edit to the entry resets
  it to *pending* — re-approve via the `/mcp` UI after changing it.
- **Bogus OAuth dialog.** If the connection fails at startup, the
  recovery path may surface
  `Cannot perform dynamic registration without authorization URL`.
  For a stdio server this is a red herring — check `timeout` units
  and command resolution first; stdio servers never use OAuth.
- `"trust": true` in `~/.qwen/settings.json` skips per-tool approval
  prompts.

## Cursor

| | |
|---|---|
| Config path | `<ws>/.cursorrules` |
| Format | Text block |
| Install | `mm install cursor` |

Appends a `.cursorrules` block. Cursor picks it up automatically on
next file open.

## Windsurf (Cognition, formerly Codeium)

| | |
|---|---|
| Config path | `<ws>/.windsurfrules` |
| Format | Text block |
| Install | `mm install windsurf` |

Same pattern as Cursor. Windsurf reads it on every workspace open.

## aider (Aider-AI)

| | |
|---|---|
| Config path | `<ws>/.aider.conf.yml` |
| Format | YAML block |
| Install | `mm install aider` |

Adds `read: ["<ws>/CLAUDE.md"]` so aider always loads the MIND-Mem
context file on startup.

## OpenClaw (OpenClaw Foundation)

| | |
|---|---|
| Config path | `~/.openclaw/openclaw.json` |
| Format | JSON `hooks.internal.entries.mind-mem` |
| Install | `mm install openclaw` |

Registers the OpenClaw hook entry pointing at the MIND-Mem workspace.
OpenClaw shares the same `~/.openclaw/workspace/` directory as
MIND-Mem so no separate storage — writes flow both ways.

## NanoClaw (Gavriel Cohen / Qwibit, open source)

| | |
|---|---|
| Config path | `~/.nanoclaw/nanoclaw.json` |
| Format | Same JSON hook shape the OpenClaw writer uses |
| Install | `mm install nanoclaw` |

[NanoClaw](https://nanoclaw.dev/) is a separate open-source agent project,
written as a lighter alternative to OpenClaw; it is not an OpenClaw variant.
`mm install nanoclaw` writes the `hooks.internal.entries.mind-mem` entry to
`~/.nanoclaw/nanoclaw.json` idempotently. We have not verified that current
NanoClaw releases read that file.

## NemoClaw (NVIDIA)

| | |
|---|---|
| Config path | `~/.nemoclaw/nemoclaw.json` |
| Format | Same JSON hook shape the OpenClaw writer uses |
| Install | `mm install nemoclaw` |

[NVIDIA NemoClaw](https://docs.nvidia.com/nemoclaw/) is NVIDIA's open-source
stack for running OpenClaw ([announcement](https://nvidianews.nvidia.com/news/nvidia-announces-nemoclaw)).
`mm install nemoclaw` writes the `hooks.internal.entries.mind-mem` entry to
`~/.nemoclaw/nemoclaw.json` idempotently. We have not verified that current
NemoClaw releases read that file.

## Continue.dev

| | |
|---|---|
| Config path | `~/.continue/config.json` |
| Format | JSON `systemMessage` |
| MCP config | `~/.continue/config.yaml` (`mcpServers` list), merged only into an existing file |
| Install | `mm install continue` |

Injects a `systemMessage` into the continue config. Applies to both
the VS Code and JetBrains editions.

## Cline (VS Code extension)

| | |
|---|---|
| Config path | `<ws>/.clinerules` |
| Format | Text block |
| MCP config | `~/.config/Code/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json` (also the VS Code Server and `~/.cline` CLI copies when present) |
| Install | `mm install cline` |

Drops a `.clinerules` file at the workspace root. Cline reads it on
every session start.

## Roo Code (VS Code extension)

| | |
|---|---|
| Config path | `<ws>/.roo/system-prompt.md` |
| Format | Markdown block |
| MCP config | `~/.config/Code/User/globalStorage/rooveterinaryinc.roo-cline/settings/mcp_settings.json` (also the VS Code Server copy when present) |
| Install | `mm install roo` |

Writes a system-prompt override under `.roo/`. The prompt is
mind-mem-aware and routes Roo's tool calls through the MCP layer.

## Zed

| | |
|---|---|
| Config path | `~/.config/zed/settings.json` (the writer uses this path on every OS) |
| Format | JSON `assistant.default_system_message` |
| Install | `mm install zed` |

Adds a default system message to Zed's built-in AI assistant.

## GitHub Copilot

| | |
|---|---|
| Config path | `<ws>/.github/copilot-instructions.md` |
| Format | Markdown file |
| Install | `mm install copilot` |

Creates (or appends to) the repository's Copilot workspace
instructions file. `mm install-all` **always** runs this one because
Copilot is near-universal — no way to detect whether a given user
actually has Copilot enabled, so we assume they might and drop the
file in. Harmless if Copilot is absent.

## GitHub Copilot CLI

| | |
|---|---|
| Config path | `<ws>/AGENTS.md` (workspace memory protocol) |
| MCP path | `~/.copilot/mcp-config.json` |
| Format | Markdown + `{"mcpServers": {...}}` JSON |
| Detection | `copilot` binary on PATH or `~/.copilot/` directory |
| Install | `mm install copilot-cli` |

Wires the `@github/copilot` CLI (a separate product from the VS Code
Copilot extension above). The workspace `AGENTS.md` carries the same
memory protocol snippet Codex and Vibe use, while the MCP entry at
`~/.copilot/mcp-config.json` exposes the `mind-mem` MCP server inside
each Copilot session.

## Cody (Sourcegraph)

| | |
|---|---|
| Config path | `<ws>/.cody/config.json` |
| Format | Generic JSON with `mind_mem` key |
| Install | `mm install cody` |

Injects a `mind_mem` stanza Cody can pick up via its custom context
providers.

## Qodo Gen (formerly CodiumAI)

| | |
|---|---|
| Config path | `<ws>/.codium/ai-rules.md` |
| Format | Markdown block |
| Install | `mm install qodo` |

Same pattern as Cursor / Windsurf. Qodo picks it up when the plugin
parses workspace rules.

## Vibe (Mistral CLI, Mistral AI)

| | |
|---|---|
| Config path | `<ws>/AGENTS.md` |
| Format | Markdown block |
| MCP config | `~/.vibe/config.toml` (a top-level `mcp_servers = [...]` array) |
| Detected by | the `vibe` binary on `PATH` or `~/.vibe/` |
| Install | `mm install vibe` |

Writes the memory-protocol block into the workspace `AGENTS.md` and adds a
`mind-mem` entry to the `mcp_servers` array in `~/.vibe/config.toml`, keeping
any other entries.

## Grok Build CLI (xAI)

| | |
|---|---|
| Config path | `<ws>/AGENTS.md` |
| Format | Markdown block |
| MCP config | `~/.grok/config.toml` (`mcp-toml-codex` shape, same as Codex CLI) |
| Detected by | the `grok` binary on `PATH` |
| Install | `mm install grok-build` |

Writes the memory-protocol block into the workspace `AGENTS.md` and registers
mind-mem as a native MCP server in `~/.grok/config.toml`. Because the file is
shared with other tools that read `AGENTS.md`, the block is fenced by the
`# mind-mem` marker and rewritten in place rather than appended.

## More clients

### Claude Desktop (Anthropic)

| | |
|---|---|
| MCP config | `~/.config/Claude/claude_desktop_config.json` (JSON, `mcpServers.mind-mem`) |
| Mode | guarded on linux |
| Install | `mm install-all --agent claude-desktop` |
| Detected by | `~/.config/Claude` |
| Source | <https://modelcontextprotocol.io/docs/develop/connect-local-servers> |

### VS Code (native MCP) (Microsoft)

| | |
|---|---|
| MCP config | `~/.config/Code/User/mcp.json` (JSON, `servers.mind-mem` (type stdio)) |
| Mode | guarded |
| Install | `mm install-all --agent vscode` |
| Detected by | `code` on `PATH`, `code-insiders` on `PATH`, `codium` on `PATH`, `~/.config/Code/User` |
| Source | <https://code.visualstudio.com/docs/copilot/customization/mcp-servers> |

### Visual Studio (Windows) (Microsoft)

| | |
|---|---|
| MCP config | `~/.mcp.json` (JSON, `servers.mind-mem`) |
| Mode | Windows only |
| Install | `mm install-all --agent visual-studio` |
| Detected by | `devenv.exe` on `PATH`, `vswhere.exe` on `PATH` |
| Source | <https://learn.microsoft.com/en-us/visualstudio/ide/mcp-servers> |

### Kimi Code CLI (Moonshot AI)

| | |
|---|---|
| Instructions | `<workspace>/AGENTS.md` (Markdown block) |
| MCP config | `~/.kimi-code/mcp.json` (JSON, `mcpServers.mind-mem`) |
| Mode | also `~/.kimi/mcp.json` if present |
| Install | `mm install-all --agent kimi-code` |
| Detected by | `kimi` on `PATH`, `~/.kimi-code`, `~/.kimi` |
| Source | <https://moonshotai.github.io/kimi-code/en/configuration/data-locations.html> |

### Amp (Amp)

| | |
|---|---|
| Instructions | `~/.config/amp/AGENTS.md` (Markdown block) |
| MCP config | `~/.config/amp/settings.json` (JSON, `"amp.mcpServers".mind-mem`) |
| Mode | auto |
| Install | `mm install-all --agent amp` |
| Detected by | `amp` on `PATH`, `~/.config/amp` |
| Source | <https://ampcode.com/docs/customize/mcp> |

### goose (Agentic AI Foundation (created by Block))

| | |
|---|---|
| Instructions | `~/.config/goose/.goosehints` (Markdown block) |
| MCP config | `~/.config/goose/config.yaml` (YAML, `extensions.mind-mem`) |
| Mode | auto |
| Install | `mm install-all --agent goose` |
| Detected by | `goose` on `PATH`, `~/.config/goose` |
| Source | <https://github.com/aaif-goose/goose/blob/main/documentation/docs/guides/config-files.md> |

### Crush (Charm)

| | |
|---|---|
| Instructions | `<workspace>/AGENTS.md` (Markdown block) |
| MCP config | `~/.config/crush/crush.json` (JSON, `mcp.mind-mem` (type stdio)) |
| Mode | auto |
| Install | `mm install-all --agent crush` |
| Detected by | `crush` on `PATH`, `~/.config/crush` |
| Source | <https://github.com/charmbracelet/crush> |

### Factory Droid (Factory)

| | |
|---|---|
| Instructions | `<workspace>/AGENTS.md` (Markdown block) |
| MCP config | `~/.factory/mcp.json` (JSON, `mcpServers.mind-mem` (type stdio, disabled false)) |
| Mode | auto |
| Install | `mm install-all --agent factory-droid` |
| Detected by | `droid` on `PATH`, `~/.factory` |
| Source | <https://docs.factory.ai/cli/configuration/mcp> |

### Kiro (AWS)

| | |
|---|---|
| Instructions | `~/.kiro/steering/mind-mem.md` (Markdown block) |
| MCP config | `~/.kiro/settings/mcp.json` (JSON, `mcpServers.mind-mem` (disabled false, autoApprove [])) |
| Mode | auto |
| Install | `mm install-all --agent kiro` |
| Detected by | `kiro` on `PATH`, `kiro-cli` on `PATH`, `~/.kiro` |
| Source | <https://kiro.dev/docs/mcp/configuration> |

### Auggie (Augment Code) (Augment Code)

| | |
|---|---|
| Instructions | `<workspace>/AGENTS.md` (Markdown block) |
| MCP config | `~/.augment/settings.json` (JSON, `mcpServers.mind-mem`) |
| Mode | auto |
| Install | `mm install-all --agent auggie` |
| Detected by | `auggie` on `PATH`, `~/.augment` |
| Source | <https://docs.augmentcode.com/cli/integrations> |

### OpenHands CLI (All Hands AI)

| | |
|---|---|
| Instructions | `<workspace>/AGENTS.md` (Markdown block) |
| MCP config | `~/.openhands/mcp.json` (JSON, `mcpServers.mind-mem` (transport stdio)) |
| Mode | auto |
| Install | `mm install-all --agent openhands` |
| Detected by | `openhands` on `PATH`, `~/.openhands` |
| Source | <https://docs.openhands.dev/openhands/usage/cli/mcp-servers> |

### MiMo Code (Xiaomi)

| | |
|---|---|
| Instructions | `<workspace>/AGENTS.md` (Markdown block) |
| MCP config | `~/.config/mimocode/mimocode.jsonc` (JSON, `mcp.mind-mem` (type local)) |
| Mode | auto |
| Install | `mm install-all --agent mimo-code` |
| Detected by | `mimo` on `PATH`, `mimocode` on `PATH`, `~/.config/mimocode` |
| Source | <https://mimo.xiaomi.com/mimocode/mcp-servers> |

### Junie (JetBrains)

| | |
|---|---|
| Instructions | `<workspace>/AGENTS.md` (Markdown block) |
| MCP config | `~/.junie/mcp/mcp.json` (JSON, `mcpServers.mind-mem`) |
| Mode | auto |
| Install | `mm install-all --agent junie` |
| Detected by | `junie` on `PATH`, `~/.junie` |
| Source | <https://junie.jetbrains.com/docs/junie-cli-mcp-configuration.html> |

### Kilo Code (Kilo Code)

| | |
|---|---|
| Instructions | `<workspace>/AGENTS.md` (Markdown block) |
| MCP config | `~/.config/kilo/kilo.jsonc` (JSON, `mcp.mind-mem` (type local)) |
| Mode | auto |
| Install | `mm install-all --agent kilo` |
| Detected by | `kilo` on `PATH`, `~/.config/kilo`, `~/.vscode/extensions/kilocode.kilo-code-*` |
| Source | <https://kilo.ai/docs/automate/mcp/using-in-cli> |

### Antigravity (Google)

| | |
|---|---|
| Instructions | `<workspace>/AGENTS.md` (Markdown block) |
| MCP config | `~/.gemini/config/mcp_config.json` (JSON, `mcpServers.mind-mem`) |
| Mode | auto |
| Install | `mm install-all --agent antigravity` |
| Detected by | `agy` on `PATH`, `antigravity` on `PATH`, `~/.gemini/config`, `~/.gemini/antigravity` |
| Source | <https://antigravity.google/docs/cli/mcp> |

### Tabnine agent (Tabnine)

| | |
|---|---|
| MCP config | `~/.tabnine/mcp_servers.json` (JSON, `mcpServers.mind-mem`) |
| Mode | auto |
| Install | `mm install-all --agent tabnine` |
| Detected by | `~/.tabnine`, `~/.vscode/extensions/tabnine.tabnine-vscode-*` |
| Source | <https://docs.tabnine.com/main/getting-started/tabnine-agent/mcp-intro-and-setup> |

### Hermes Agent (Nous Research)

| | |
|---|---|
| MCP config | `~/.hermes/config.yaml` (YAML, `mcp_servers.mind-mem`) |
| Mode | auto |
| Install | `mm install-all --agent hermes-agent` |
| Detected by | `hermes` on `PATH`, `~/.hermes` |
| Source | <https://hermes-agent.nousresearch.com/docs/user-guide/features/mcp> |

### AnythingLLM Desktop (Mintplex Labs)

| | |
|---|---|
| MCP config | `~/.config/anythingllm-desktop/storage/plugins/anythingllm_mcp_servers.json` (JSON, `mcpServers.mind-mem` (anythingllm.autoStart)) |
| Mode | guarded |
| Install | `mm install-all --agent anythingllm` |
| Detected by | `~/.config/anythingllm-desktop` |
| Source | <https://docs.anythingllm.com/mcp-compatibility/desktop> |

### Warp (Warp)

| | |
|---|---|
| Instructions | `<workspace>/AGENTS.md` (Markdown block) |
| MCP config | `~/.warp/.mcp.json` (JSON, `mcpServers.mind-mem`) |
| Mode | guarded |
| Install | `mm install-all --agent warp` |
| Detected by | `warp` on `PATH`, `warp-terminal` on `PATH`, `~/.warp` |
| Source | <https://docs.warp.dev/agents/capabilities/mcp/> |

### Open Interpreter (Open Interpreter)

| | |
|---|---|
| MCP config | `~/.openinterpreter/config.toml` (TOML, `[mcp_servers.mind-mem]`) |
| Mode | guarded |
| Install | `mm install-all --agent open-interpreter` |
| Detected by | `interpreter` on `PATH`, `~/.openinterpreter` |
| Source | <https://openinterpreter.com/docs/terminal/mcp> |

### Amazon Q Developer (AWS)

| | |
|---|---|
| MCP config | `~/.aws/amazonq/mcp.json` (JSON, `mcpServers.mind-mem`) |
| Mode | guarded |
| Install | `mm install-all --agent amazon-q` |
| Detected by | `~/.aws/amazonq` |
| Source | <https://docs.aws.amazon.com/amazonq/latest/qdeveloper-ug/mcp-ide.html> |

### LM Studio (Element Labs)

| | |
|---|---|
| MCP config | `~/.lmstudio/mcp.json` (JSON, `mcpServers.mind-mem`) |
| Mode | guarded |
| Install | `mm install-all --agent lm-studio` |
| Detected by | `lms` on `PATH`, `~/.lmstudio` |
| Source | <https://lmstudio.ai/docs/app/mcp> |

### Jan (Menlo Research)

| | |
|---|---|
| MCP config | `~/.local/share/Jan/data/mcp_config.json` (JSON, `mcpServers.mind-mem` (active true)) |
| Mode | guarded |
| Install | `mm install-all --agent jan` |
| Detected by | `~/.local/share/Jan` |
| Source | <https://www.jan.ai/docs/desktop/integrations/mcp-servers> |

### BoltAI (macOS) (BoltAI)

| | |
|---|---|
| MCP config | `~/.boltai/mcp.json` (JSON, `mcpServers.mind-mem`) |
| Mode | guarded; macOS only |
| Install | `mm install-all --agent boltai` |
| Detected by | `~/.boltai` |
| Source | <https://docs.boltai.com/docs/plugins/mcp-servers> |

### iFlow CLI (iFlow)

| | |
|---|---|
| Instructions | `<workspace>/AGENTS.md` (Markdown block) |
| MCP config | `~/.iflow/settings.json` (JSON, `mcpServers.mind-mem`) |
| Mode | guarded |
| Install | `mm install-all --agent iflow` |
| Detected by | `iflow` on `PATH`, `~/.iflow` |
| Source | <https://platform.iflow.cn/cli/examples/mcp> |

### Neovim (mcphub.nvim) (mcphub.nvim (open source))

| | |
|---|---|
| MCP config | `~/.config/mcphub/servers.json` (JSON, `mcpServers.mind-mem`) |
| Mode | guarded |
| Install | `mm install-all --agent neovim-mcphub` |
| Detected by | `~/.config/mcphub` |
| Source | <https://ravitemer.github.io/mcphub.nvim/mcp/servers_json.html> |

### Trae (ByteDance)

| | |
|---|---|
| MCP | manual: Settings > MCP > Add > Raw config, or enable workspace MCP and add it to .trae/mcp.json: paste the mcpServers JSON below (server "mind-mem") |
| Install | `mm install-all --agent trae` (prints the steps and the JSON snippet; writes nothing) |
| Detected by | - |
| Source | <https://docs.trae.ai/ide/model-context-protocol> |

### Android Studio (Gemini) (Google)

| | |
|---|---|
| MCP | manual: Settings > Tools > AI > MCP Servers accepts HTTP servers only: start `mind-mem-mcp --transport http` (bearer token required) and register its URL as `httpUrl` |
| Install | `mm install-all --agent android-studio` (prints the steps and the JSON snippet; writes nothing) |
| Detected by | - |
| Source | <https://developer.android.com/studio/gemini/add-mcp-server> |

### JetBrains AI Assistant (JetBrains)

| | |
|---|---|
| MCP | manual: Settings > Tools > AI Assistant > Model Context Protocol: use 'Import from Claude' after `mm install-all --agent claude-desktop`, or paste the mcpServers JSON below (server "mind-mem") |
| Install | `mm install-all --agent jetbrains-ai-assistant` (prints the steps and the JSON snippet; writes nothing) |
| Detected by | - |
| Source | <https://www.jetbrains.com/help/ai-assistant/mcp.html> |

### LibreChat (LibreChat)

| | |
|---|---|
| MCP | manual: add `mcpServers.mind-mem` (type: stdio, command, args, env) to your deployment's librechat.yaml and restart |
| Install | `mm install-all --agent librechat` (prints the steps and the JSON snippet; writes nothing) |
| Detected by | - |
| Source | <https://librechat.ai/docs/configuration/librechat_yaml/object_structure/mcp_servers> |

### Eclipse (GitHub Copilot) (GitHub / Microsoft)

| | |
|---|---|
| MCP | manual: Preferences > GitHub Copilot > MCP: paste the mcpServers JSON below (server "mind-mem") (under `servers`) |
| Install | `mm install-all --agent eclipse-copilot` (prints the steps and the JSON snippet; writes nothing) |
| Detected by | - |
| Source | <https://docs.github.com/en/copilot/how-tos/provide-context/use-mcp/extend-copilot-chat-with-mcp> |

### Xcode 26.3+ (Apple)

| | |
|---|---|
| MCP | manual: Xcode's agents use the Claude Code and Codex registrations: run `mm install-all --agent claude-code --agent codex` |
| Install | `mm install-all --agent xcode` (prints the steps and the JSON snippet; writes nothing) |
| Detected by | - |
| Source | <https://developer.apple.com/xcode/> |

### Witsy (Witsy (open source))

| | |
|---|---|
| MCP | manual: Settings > MCP > Add server: paste the mcpServers JSON below (server "mind-mem") |
| Install | `mm install-all --agent witsy` (prints the steps and the JSON snippet; writes nothing) |
| Detected by | - |
| Source | <https://github.com/nbonamy/witsy> |

### Emacs (mcp.el) (mcp.el (open source))

| | |
|---|---|
| MCP | manual: add ("mind-mem" :command "<command>" :args (...)) to `mcp-hub-servers` in your init file |
| Install | `mm install-all --agent emacs-mcp-el` (prints the steps and the JSON snippet; writes nothing) |
| Detected by | - |
| Source | <https://github.com/lizqwerscott/mcp.el> |

### Open WebUI (Open WebUI (open source))

| | |
|---|---|
| MCP | manual: Admin > Settings > Tools: start `mind-mem-mcp --transport http` (bearer token required) and register its URL (or expose it through mcpo) |
| Install | `mm install-all --agent open-webui` (prints the steps and the JSON snippet; writes nothing) |
| Detected by | - |
| Source | <https://docs.openwebui.com/features/extensibility/plugin/tools/openapi-servers/mcp> |

### Docker MCP Toolkit (Docker)

| | |
|---|---|
| MCP | manual: the toolkit is a gateway for containerised servers; register mind-mem in each client directly with `mm install-all` |
| Install | `mm install-all --agent docker-mcp-toolkit` (prints the steps and the JSON snippet; writes nothing) |
| Detected by | - |
| Source | <https://docs.docker.com/ai/mcp-catalog-and-toolkit/toolkit> |

### Cherry Studio (CherryHQ)

| | |
|---|---|
| MCP | manual: Settings > MCP Servers > Add > Import from JSON: paste the mcpServers JSON below (server "mind-mem") |
| Install | `mm install-all --agent cherry-studio` (prints the steps and the JSON snippet; writes nothing) |
| Detected by | - |
| Source | <https://docs.cherry-ai.com> |

### Msty Studio (Msty)

| | |
|---|---|
| MCP | manual: Settings > Tools > Add MCP server (JSON): paste the mcpServers JSON below (server "mind-mem") |
| Install | `mm install-all --agent msty` (prints the steps and the JSON snippet; writes nothing) |
| Detected by | - |
| Source | <https://docs.msty.ai/studio/toolbox/tools> |

### Raycast (Raycast)

| | |
|---|---|
| MCP | manual: Raycast AI > Manage MCP Servers > Install Server: paste the mcpServers JSON below (server "mind-mem") |
| Install | `mm install-all --agent raycast` (prints the steps and the JSON snippet; writes nothing) |
| Detected by | - |
| Source | <https://manual.raycast.com/ai/model-context-protocol> |

### 5ire (5ire (open source))

| | |
|---|---|
| MCP | manual: Tools > New: paste the mcpServers JSON below (server "mind-mem") |
| Install | `mm install-all --agent 5ire` (prints the steps and the JSON snippet; writes nothing) |
| Detected by | - |
| Source | <https://github.com/nanbingxyz/5ire> |

### Perplexity (macOS app) (Perplexity)

| | |
|---|---|
| MCP | manual: Settings > Connectors > Add connector (advanced): paste the mcpServers JSON below (server "mind-mem") |
| Install | `mm install-all --agent perplexity-desktop` (prints the steps and the JSON snippet; writes nothing) |
| Detected by | - |
| Source | <https://perplexity.ai/help-center/en/articles/11502712-local-and-remote-mcps-for-perplexity> |

### ChatGPT (OpenAI)

| | |
|---|---|
| MCP | manual: remote HTTPS connectors only: start `mind-mem-mcp --transport http` (bearer token required) and register its URL behind HTTPS and add it under Settings > Connectors |
| Install | `mm install-all --agent chatgpt-desktop` (prints the steps and the JSON snippet; writes nothing) |
| Detected by | - |
| Source | <https://help.openai.com/en/articles/11487775-connectors-in-chatgpt> |

### n8n (n8n)

| | |
|---|---|
| MCP | manual: use the MCP Client Tool node: start `mind-mem-mcp --transport http` (bearer token required) and register its URL |
| Install | `mm install-all --agent n8n` (prints the steps and the JSON snippet; writes nothing) |
| Detected by | - |
| Source | <https://docs.n8n.io/integrations/builtin/cluster-nodes/sub-nodes/n8n-nodes-langchain.toolmcp/> |

### Dify (LangGenius)

| | |
|---|---|
| MCP | manual: Tools > MCP > Add MCP server (HTTP): start `mind-mem-mcp --transport http` (bearer token required) and register its URL |
| Install | `mm install-all --agent dify` (prints the steps and the JSON snippet; writes nothing) |
| Detected by | - |
| Source | <https://docs.dify.ai/en/cloud/use-dify/build/mcp> |

### Flowise (FlowiseAI)

| | |
|---|---|
| MCP | manual: add a Custom MCP tool node with the stdio command and MIND_MEM_WORKSPACE env shown below |
| Install | `mm install-all --agent flowise` (prints the steps and the JSON snippet; writes nothing) |
| Detected by | - |
| Source | <https://docs.flowiseai.com/tutorials/tools-and-mcp> |

### Blackbox AI (Blackbox)

| | |
|---|---|
| MCP | manual: open the extension's MCP settings and paste the mcpServers JSON below (server "mind-mem") |
| Install | `mm install-all --agent blackbox` (prints the steps and the JSON snippet; writes nothing) |
| Detected by | - |
| Source | <https://www.blackbox.ai> |

### fast-agent (fast-agent (open source))

| | |
|---|---|
| MCP | manual: add the server under `mcp.servers.mind-mem` (transport: stdio, command, args, env) in your project's fast-agent.yaml |
| Install | `mm install-all --agent fast-agent` (prints the steps and the JSON snippet; writes nothing) |
| Detected by | `fast-agent` on `PATH`, `~/.fast-agent` |
| Source | <https://fast-agent.ai/mcp/> |

### Void (Void (open source))

| | |
|---|---|
| MCP | manual: open Void's MCP settings (Settings > MCP) and paste the mcpServers JSON below (server "mind-mem") |
| Install | `mm install-all --agent void` (prints the steps and the JSON snippet; writes nothing) |
| Detected by | `void` on `PATH`, `~/.void-editor`, `~/.void` |
| Source | <https://github.com/voideditor/void> |

## Running `install-all` end-to-end

The typical first-time flow after installing mind-mem:

```bash
# Point mm at the workspace you want every client to share.
export MIND_MEM_WORKSPACE="$HOME/.openclaw/workspace"
mind-mem-init "$MIND_MEM_WORKSPACE"

# See what's installed on this box.
mm detect

# Configure every detected client.
mm install-all

# Verify — prints per-client status, files written / merged / skipped.
mm install-all --dry-run
```

After this, every supported client on your machine reads from and
writes into the same MIND-Mem workspace. Add a tool to the same
box later, and `mm install-all` picks it up on the next run — it's
safe to re-run.

## Manual / scripted hooks

If you want to drive MIND-Mem from a bespoke tool that isn't in the
registry, shell out to the CLI directly:

- `mm recall <query> --limit 10` — JSON result list, ideal for
  piping into a tool-call.
- `mm context "<query>" --max-tokens 1500` — pre-packed context
  snippet ready to prepend to a prompt.
- `mm inject "<query>" --agent generic` — returns the MIND-Mem
  context in a format the agent's system prompt can consume.
- `mm status` — print workspace health as JSON.

Every CLI subcommand shares the same `--workspace` semantics:
either the `MIND_MEM_WORKSPACE` env var or a `--workspace` flag,
falling back to the current directory.

## Shared workspace policy

All clients write to one workspace by default so memory is portable.
To give a single client its own isolated namespace, pass
`MIND_MEM_WORKSPACE` in its environment or (for JSON-config clients)
edit the installed stanza to point elsewhere. The `NamespaceManager`
then partitions queries by agent-id so concurrent writes from
different clients don't cross-contaminate.

## Troubleshooting

- **`mm install <client>` says "unknown agent"** — run `mm detect`
  to see the exact names; the CLI accepts only keys from
  `AGENT_REGISTRY`.
- **MCP tools time out in Qwen Code but work in Claude Code** —
  `timeout` in a shared `.mcp.json` is seconds for Claude Code but
  **milliseconds** for Qwen Code (a Gemini-CLI descendant). Remove
  `timeout` from the shared file or set a millisecond value in Qwen
  Code's own `~/.qwen/settings.json`. See
  [Qwen Code](#qwen-code-alibaba).
- **Re-running `mm install-all` shows every client as `skipped`** —
  that's correct. The installers are idempotent; the `# mind-mem`
  marker prevents duplicate blocks. Use `--force` to rewrite.
- **Config file was hand-edited and won't merge** — for JSON configs
  MIND-Mem does a structured merge (your keys survive). For text
  configs MIND-Mem only appends once. If the hand-edit broke the
  marker, run `mm install <client> --force` and re-apply your hand
  edits on top.
