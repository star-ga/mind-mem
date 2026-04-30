# Integrations

> Honest positioning. Every claim below is verifiable from the source
> tree, the public PyPI artifact, or a published benchmark file in
> `benchmarks/`. Nothing here is a customer-relationship claim about
> any AI vendor — the integrations described are *software-level*
> (the named tool talks to mind-mem via the Model Context Protocol),
> not commercial.

## What mind-mem actually ships

### Native MCP integration with 17 AI development tools

mind-mem speaks the [Model Context Protocol](https://modelcontextprotocol.io/).
Any MCP-compatible client connects with one command:

```bash
pip install mind-mem
mm install-all
```

`mm install-all` auto-detects every supported client on your machine
and writes the appropriate config file for each.

Currently supported:

| Client | Vendor | Config |
|--------|--------|--------|
| Claude Code | Anthropic | `~/.claude/settings.json` |
| Claude Desktop | Anthropic | `~/Library/Application Support/Claude/` |
| Codex CLI | OpenAI | `~/.codex/config.toml` |
| Gemini CLI | Google | `~/.gemini/settings.json` |
| Vibe (Mistral CLI) | Mistral | `~/.vibe/config.toml` |
| Cursor | Anysphere | `~/.cursor/mcp.json` |
| Windsurf | Codeium | `~/.codeium/windsurf/mcp_config.json` |
| Zed | Zed Industries | `~/.config/zed/settings.json` |
| Continue | Continue.dev | `.continue/config.json` |
| Cline | Cline.bot | VS Code extension settings |
| Roo | Roo Code | VS Code extension settings |
| GitHub Copilot | GitHub / Microsoft | VS Code extension settings |
| Cody | Sourcegraph | `~/.sourcegraph/cody.json` |
| Qodo | Qodo | `~/.qodo/config.json` |
| aider | aider-chat | `.aider.conf.yml` |
| OpenClaw | STARGA | `~/.openclaw/openclaw.json` |
| NanoClaw / NemoClaw | STARGA | `~/.openclaw/openclaw.json` |

**What this means**: each of these tools can call mind-mem's 71 MCP
tools (recall, propose_update, scan, hybrid_search, etc.) the same
way it calls any other MCP server.

**What this does *not* mean**: none of these vendors are commercial
customers, paying users, partners, or have endorsed mind-mem. The
integration is at the protocol layer — *their* software talks to
*our* software. Compatibility is open and unilateral.

## Open-source distribution

```bash
pip install mind-mem
```

- License: Apache-2.0
- PyPI: [`mind-mem`](https://pypi.org/project/mind-mem/)
- Source: [`star-ga/mind-mem`](https://github.com/star-ga/mind-mem)
- Local model: `mind-mem:4b` (Qwen-3.5-4B fine-tune) ships via
  Ollama — no cloud API required for the extraction model

## Compatible with major LLM providers

mind-mem's recall pipeline is provider-agnostic. Tested against:

- Anthropic Claude (3.5 Sonnet, 4.x family)
- OpenAI GPT (4o, 5.4)
- Google Gemini (2.0 Flash, 3.1 Pro)
- Mistral Large
- Local: Ollama, vLLM, llama.cpp endpoints

The "compatibility" claim is at the API contract level — the same
mind-mem server returns the same answers regardless of which LLM is
asking. We do not use any provider's commercial relationship as a
positioning artefact.

## Reproducible benchmarks

All numbers below are reproducible from `benchmarks/` and the
matching pipeline configs in the README.

| Benchmark | Score | Methodology |
|-----------|-------|-------------|
| **NIAH** (Needle In A Haystack) | **250 / 250** (100%) | Hybrid BM25 + BAAI/bge-large-en-v1.5 + RRF (k=60) + sqlite-vec. See `benchmarks/NIAH.md`. |
| **LoCoMo** (Mistral Large judge, 10-conv, 1986 questions) | **73.8% Acc≥50, mean 70.5** | BM25 + RM3 query expansion → top-18 evidence → observation compression → answer → judge. Full 10-conv benchmark. |
| **LoCoMo** (Mistral Large judge, conv-0, 199 questions, hybrid pipeline) | **92.5% Acc≥50, mean 76.7** | Hybrid: BM25 + Qwen3-Embedding-8B (4096d) → RRF fusion → top-18 → compression → answer → judge. |
| **LoCoMo Adversarial subset** | **97.9% Acc≥50** | Subset of the conv-0 hybrid run; tests retrieval against intentionally-misleading distractor turns. |

> Comparisons: published numbers for Mem0 (66.88), Zep (65.99),
> Letta (74.0), Memobase (75.8), LangMem (58.10) on the same
> benchmark. mind-mem surpasses Mem0 and Letta on the same 10-conv
> LoCoMo benchmark with **zero cloud infrastructure** and
> **local-only retrieval** — no graph DB, no vector DB service, no
> LLM in the retrieval loop unless the operator opts in.

## Production usage at STARGA

mind-mem is the daily-driver memory layer for STARGA's six active
projects: `mind`, `mind-runtime`, `mindlang.dev`, `mind-inference`,
`mind-fleet`, `arch-mind`. Used internally for cross-session recall,
contradiction detection, and audit-grade rationale chains during
agent-driven development.

This is a STARGA-internal usage statement — first-party, verifiable
in our own commit history. We do not extrapolate it into a third-party
"trusted by" claim.

## What we do not claim

- "OpenAI is a customer" — false. OpenAI runs its own memory systems.
  Codex CLI integration is software-level (MCP), not a commercial
  relationship.
- "Microsoft is a customer" — false. Copilot integration is via the
  VS Code extension MCP surface, not a Microsoft Inc. commercial
  relationship.
- "Anthropic is a customer" — false. Claude Code is built on the MCP
  spec; mind-mem implements that spec; Anthropic has not endorsed,
  contracted with, or partnered with STARGA.
- "Used by N production teams outside STARGA" — we have no telemetry.
  PyPI download counts measure installs, not active use, and we do
  not turn install counts into traction claims.

If a future integration becomes a real commercial relationship
(signed contract, NDA, paid pilot), it will appear in the press
release and on this page — not before.

## Surface allow-list (where this section may be reused)

Verbatim copy of the section above is approved for:

- README "Integrations" section
- mindlang.dev / mind-mem product pages
- Investor decks and one-pagers
- Cold outreach emails
- LinkedIn / X marketing
- Press releases

Do not paraphrase in a way that drops the "via MCP integration"
qualifier next to vendor names. The qualifier is what keeps the
claim defensible.
