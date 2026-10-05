# MCP server or CLI?

Both talk to the same workspace and the same engine. Pick by how your agent
runs tools.

| | CLI (`mm`) | MCP server (`mind-mem-mcp`) |
| --- | --- | --- |
| Needs | `pip install mind-mem` | `pip install "mind-mem[mcp]"` and a client that speaks MCP |
| Agent calls | shell commands | typed tool calls |
| Output | JSON / text on stdout | structured tool results |
| Writes | governed paths only (capture, scan, review) | adds `propose_update`, which queues a proposal for review |
| Best for | shell-first agents, scripts, CI, hooks | clients with native MCP support |

Use the CLI when the agent already lives in a terminal, when MCP is not
available, or when you want results you can pipe and diff. Use MCP when the
client supports it and you want the agent to propose new memory directly.
Many setups use both: `mm install-all` writes hook and instruction files and
registers the MCP server for clients that support it (`--no-mcp` skips the
registration).

## Starting the MCP server

```bash
mind-mem-mcp                                   # stdio (what clients launch)
mind-mem-mcp --transport http --port 8765      # HTTP, loopback by default
mind-mem-mcp --watch                           # reindex when workspace .md files change
```

The server reads `MIND_MEM_WORKSPACE` like the CLI. Admin tools (approvals,
applies) require `MIND_MEM_SCOPE=admin` on stdio, or an admin token over
HTTP. The HTTP transport refuses to start without a token unless
`--allow-unauthenticated-localhost` is given on a loopback bind.

## Tool-to-command map

| MCP tool | CLI equivalent |
| --- | --- |
| `recall` | `mm recall "<query>"` |
| `hybrid_search` | `mm recall "<query>"` with `recall.backend` / `recall.vector_enabled` configured for hybrid retrieval |
| `pack_recall_budget` | `mm context "<query>" --max-tokens 1500` |
| `get_block` | `mm inspect <block-id>` |
| `retrieval_diagnostics` | `mm explain "<query>"` |
| `resume_brief` | `mm resume` |
| `check_dead_ends` | `mm dead-ends --tool ... --command ...` |
| `scan` | `mind-mem-scan "$MIND_MEM_WORKSPACE"` |
| `reindex` | `python3 -m mind_mem.sqlite_index build --workspace "$MIND_MEM_WORKSPACE"` |
| `approve_apply`, `reject_proposal` | `mm review --approve <id>` / `mm review --reject <id> --reason "..."` (admin scope) |
| `lint` | `mm lint` (requires `v4.lint`) |
| `vault_scan` | `mm vault scan <vault_root>` |
| `memory_verify`, `verify_chain` | `mind-mem-verify "$MIND_MEM_WORKSPACE"`, `mm chain survey "$MIND_MEM_WORKSPACE"` |
| `propose_update` | No direct CLI command. From the shell, write to the daily log and run `mind-mem-capture` (see [../SKILL.md](../SKILL.md#writing-memory-governed)). |

The MCP server exposes more tools than the CLI has commands; the CLI covers
the day-to-day read, review and maintenance paths. `mm trace` shows recent MCP
tool calls from the structured log when you need to see what an MCP client did.
