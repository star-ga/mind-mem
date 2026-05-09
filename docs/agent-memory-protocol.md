# Agent Memory Protocol — canonical system-prompt snippet

This is the **canonical** memory-protocol text that `mm install-all`
writes into each detected CLI's system-prompt file (`AGENTS.md`,
`GEMINI.md`, `.cursorrules`, `.windsurfrules`, `.clinerules`,
`.roo/system-prompt.md`, `.aider.conf.yml`, etc.).

If you're operating a CLI that already has its own system prompt
(`CLAUDE.md`, `AGENTS.md`, `GEMINI.md`, …), append this section. If
you're starting fresh, this is the entire prompt.

The hook installer (`src/mind_mem/hook_installer.py`,
`AGENT_REGISTRY`) reads `MEMORY_PROTOCOL_SNIPPET` from
`hook_installer/memory_protocol.py` so a single canonical edit
propagates to every wired agent.

---

## Memory Protocol (mind-mem MCP — USE CONSTANTLY)

The mind-mem MCP server is wired into this CLI. Before answering ANY
question that touches prior work, decisions, project state, people,
companies, technical choices, or "remember when..." — call mind-mem
first. The governed blocks in mind-mem's Postgres backend are the
source of truth.

### Always recall before answering
Run `mcp__mind-mem__recall` (or `mm recall "$query"` via bash) with
relevant keywords FIRST. If recall returns results, cite them. If
not, say "no record found" — do not guess.

### Always propose_update after learning new facts
When the user shares a new decision, project status, relationship,
contact, or technical finding worth preserving — run
`mcp__mind-mem__propose_update` to store it. Don't wait to be asked.

### Use hybrid_search for complex queries
For multi-faceted questions, use `mcp__mind-mem__hybrid_search`
(BM25 + vector + RRF fusion) for best recall.

### Hallucination guardrail
If you find yourself writing about prior projects, repos, or
decisions without having called recall in this session, you are
about to hallucinate. Stop and call recall first.

---

## Why this exists

Audit on 2026-05-08 of a sibling product (a 9-LLM consensus engine
also based on OpenClaw) showed it had its own internal memory plugin
and **zero call paths to mind-mem MCP**. The result was repeated
hallucinations on memory-grounded questions. Adding this prompt block
into each CLI's system-prompt closes the gap by *instructing* the LLM
to use the MCP tools that `mm install-all` wires at the protocol
level.

Wiring without instruction is not enough. This file is the
instruction half.
