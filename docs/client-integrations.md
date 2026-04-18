# Client Integrations

mind-mem works with **16 AI coding clients** out of the box. Every
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
2. A **native MCP server entry** so the client gets the full 57-tool surface.

Pass `--no-mcp` to skip the MCP registration phase (hook-only, useful
when you want to register a custom MCP endpoint yourself or for
clients behind firewalls).

Use `mm install <agent>` to configure a single client; use
`mm install-all --agent <A> --agent <B>` to target an explicit subset.

Every writer is **non-destructive**: existing config files are parsed
and the mind-mem stanza is merged in under the `# mind-mem` marker.
Re-running the command is idempotent. Pass `--force` to overwrite a
hand-rolled config you want replaced.

### Native MCP formats per client

v3.1.0 added format-specific MCP writers for 8 MCP-aware clients:

| Client | Config path | Format | Stanza |
| --- | --- | --- | --- |
| Codex | `~/.codex/config.toml` | TOML | `[mcp_servers.mind-mem]` |
| Gemini | `~/.gemini/settings.json` | JSON | `mcpServers.mind-mem` |
| Cursor | `~/.cursor/mcp.json` | JSON | `mcpServers.mind-mem` |
| Windsurf | `~/.codeium/windsurf/mcp_config.json` | JSON | `mcpServers.mind-mem` |
| Continue | `~/.continue/config.json` | JSON | `mcpServers.mind-mem` |
| Cline | `<vscode-user>/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json` | JSON | `mcpServers.mind-mem` |
| Roo | `<vscode-user>/globalStorage/rooveterinaryinc.roo-cline/settings/mcp_settings.json` | JSON | `mcpServers.mind-mem` |
| Zed | `~/.config/zed/settings.json` | JSON | `context_servers.mind-mem` |

All entries point at `<mind-mem-install>/mcp_server.py` with
`MIND_MEM_WORKSPACE` set to the shared workspace. Clients not on this
list still get the text-hook fallback, which routes tool calls
through the `mm` CLI.

## Claude Code (Anthropic)

| | |
|---|---|
| Config path | `~/.claude/settings.json` |
| Format | JSON `hooks` section |
| Install | `mm install claude-code` |

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

## Gemini CLI (Google)

| | |
|---|---|
| Config path | `<ws>/.gemini/settings.json` |
| Format | JSON `system_instruction` |
| Install | `mm install gemini` |

Injects a system instruction pointing at the mind-mem workspace. The
instruction survives `gemini --yolo` and interactive mode.

## Cursor

| | |
|---|---|
| Config path | `<ws>/.cursorrules` |
| Format | Text block |
| Install | `mm install cursor` |

Appends a `.cursorrules` block. Cursor picks it up automatically on
next file open.

## Windsurf (Codeium)

| | |
|---|---|
| Config path | `<ws>/.windsurfrules` |
| Format | Text block |
| Install | `mm install windsurf` |

Same pattern as Cursor. Windsurf reads it on every workspace open.

## aider (paul-gauthier)

| | |
|---|---|
| Config path | `<ws>/.aider.conf.yml` |
| Format | YAML block |
| Install | `mm install aider` |

Adds `read: ["<ws>/CLAUDE.md"]` so aider always loads the mind-mem
context file on startup.

## OpenClaw (STARGA cognitive assistant)

| | |
|---|---|
| Config path | `~/.openclaw/openclaw.json` |
| Format | JSON `hooks.internal.entries.mind-mem` |
| Install | `mm install openclaw` |

Registers the OpenClaw hook entry pointing at the mind-mem workspace.
OpenClaw shares the same `~/.openclaw/workspace/` directory as
mind-mem so no separate storage — writes flow both ways.

## NanoClaw (compact claw variant)

| | |
|---|---|
| Config path | `~/.nanoclaw/nanoclaw.json` |
| Format | Same JSON shape as OpenClaw |
| Install | `mm install nanoclaw` |

NanoClaw is the compact OpenClaw variant. Same hook registry shape;
reuses the shared workspace. Install writes the config idempotently
under the `hooks.internal.entries.mind-mem` path.

## NemoClaw (memory-focused claw variant)

| | |
|---|---|
| Config path | `~/.nemoclaw/nemoclaw.json` |
| Format | Same JSON shape as OpenClaw |
| Install | `mm install nemoclaw` |

NemoClaw emphasises long-horizon memory. Its hook loop leans on
mind-mem's `hybrid_search` path more aggressively than the generic
OpenClaw preset. Install writes the same shape; differences are
purely runtime behaviour on the NemoClaw side.

## Continue.dev

| | |
|---|---|
| Config path | `~/.continue/config.json` |
| Format | JSON `systemMessage` |
| Install | `mm install continue` |

Injects a `systemMessage` into the continue config. Applies to both
the VS Code and JetBrains editions.

## Cline (VS Code extension)

| | |
|---|---|
| Config path | `<ws>/.clinerules` |
| Format | Text block |
| Install | `mm install cline` |

Drops a `.clinerules` file at the workspace root. Cline reads it on
every session start.

## Roo Code (VS Code fork)

| | |
|---|---|
| Config path | `<ws>/.roo/system-prompt.md` |
| Format | Markdown block |
| Install | `mm install roo` |

Writes a system-prompt override under `.roo/`. The prompt is
mind-mem-aware and routes Roo's tool calls through the MCP layer.

## Zed

| | |
|---|---|
| Config path | `~/.config/zed/settings.json` (macOS: `~/Library/Application Support/Zed/settings.json`) |
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
writes into the same mind-mem workspace. Add a tool to the same
box later, and `mm install-all` picks it up on the next run — it's
safe to re-run.

## Manual / scripted hooks

If you want to drive mind-mem from a bespoke tool that isn't in the
registry, shell out to the CLI directly:

- `mm recall <query> --limit 10` — JSON result list, ideal for
  piping into a tool-call.
- `mm context "<query>" --max-tokens 1500` — pre-packed context
  snippet ready to prepend to a prompt.
- `mm inject "<query>" --agent generic` — returns the mind-mem
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
- **Re-running `mm install-all` shows every client as `skipped`** —
  that's correct. The installers are idempotent; the `# mind-mem`
  marker prevents duplicate blocks. Use `--force` to rewrite.
- **Config file was hand-edited and won't merge** — for JSON configs
  mind-mem does a structured merge (your keys survive). For text
  configs mind-mem only appends once. If the hand-edit broke the
  marker, run `mm install <client> --force` and re-apply your hand
  edits on top.
