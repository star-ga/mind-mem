# Integrations

> Honest positioning. Every claim below is verifiable from the source
> tree, the public PyPI artifact, or a published benchmark file in
> `benchmarks/`. Nothing here is a customer-relationship claim about
> any AI vendor — the integrations described are *software-level*
> (clients use their configured MCP connection or local integration),
> not commercial.

## What MIND-Mem actually ships

### Native integration with 20 clients (12 MCP-aware clients)

MIND-Mem speaks the [Model Context Protocol](https://modelcontextprotocol.io/).
Any MCP-compatible client connects with one command:

```bash
pip install mind-mem
mm install-all
```

`mm install-all` auto-detects every supported client on your machine
and writes the appropriate config file for each.

Currently supported:

| Client | Vendor | `mm install` id | MCP config written |
|--------|--------|-----------------|--------------------|
| Claude Code | Anthropic | `claude-code` | No (instructions/hooks only) |
| Codex CLI | OpenAI | `codex` | Yes |
| Grok Build CLI | xAI | `grok-build` | Yes |
| Vibe (Mistral CLI) | Mistral AI | `vibe` | Yes |
| OpenCode (1.x / 2.x) | OpenCode (open source) | `opencode` | Yes |
| Gemini CLI | Google | `gemini` | Yes |
| Cursor | Anysphere | `cursor` | Yes |
| Windsurf | Cognition (formerly Codeium) | `windsurf` | Yes |
| aider | Aider-AI (open source) | `aider` | No (instructions/hooks only) |
| OpenClaw | OpenClaw Foundation (open source) | `openclaw` | No (instructions/hooks only) |
| NanoClaw | NanoClaw / Qwibit (open source) | `nanoclaw` | No (instructions/hooks only) |
| NemoClaw | NVIDIA | `nemoclaw` | No (instructions/hooks only) |
| Continue | Continue.dev | `continue` | Yes |
| Cline | Cline | `cline` | Yes |
| Roo Code | Roo Code | `roo` | Yes |
| Zed | Zed Industries | `zed` | Yes |
| GitHub Copilot (workspace instructions) | GitHub / Microsoft | `copilot` | No (instructions/hooks only) |
| GitHub Copilot CLI | GitHub / Microsoft | `copilot-cli` | Yes |
| Cody | Sourcegraph | `cody` | No (instructions/hooks only) |
| Qodo Gen | Qodo | `qodo` | No (instructions/hooks only) |

Every row is an entry in `mind_mem.hook_installer.AGENT_REGISTRY`; the
per-client config paths are in
[`docs/client-integrations.md`](client-integrations.md). Claude Desktop
(Anthropic) is not in the registry: `./install.sh --claude-desktop`
configures it, or see [`docs/claude-desktop-setup.md`](claude-desktop-setup.md).

**What this means**: each client marked "Yes" above can call MIND-Mem's
107 MCP tools (recall, propose_update, scan, hybrid_search,
mic_convert_tool, mic_inspect_tool, etc.) the same way it calls any other
MCP server; the others get instructions or hooks that route through the
`mm` CLI.

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
- Local model: `mind-mem:4b` (fully trained) ships via
  Ollama — no cloud API required for the extraction model

## Compatible with major LLM providers

MIND-Mem's recall pipeline is provider-agnostic: any MCP-capable client or
OpenAI-compatible endpoint can use the same server interface. The provider
adapters (Anthropic; OpenAI-compatible, which Mistral and other OpenAI-style
APIs route through; Ollama; vLLM; llama.cpp) are covered by mocked contract
tests, which make no live call. The only model-in-the-loop benchmark run so
far is LoCoMo, with `mistral-large-latest` as answerer and judge. We do not
claim live testing against specific versions of other vendors' models. Replay also requires the same query, admitted corpus,
configuration, scoring instant, execution providers and dependencies.
Different client models can generate different queries. Provider compatibility
does not imply a commercial relationship.

## Reproducible benchmarks

All numbers below are reproducible from `benchmarks/` and the
matching pipeline configs in the README.

| Benchmark | Score | Methodology |
|-----------|-------|-------------|
| **NIAH** (Needle In A Haystack) | **250 / 250** (100%) | Hybrid BM25 + all-MiniLM-L6-v2 + RRF (k=60) + sqlite-vec. See `benchmarks/NIAH.md`. |
| **LoCoMo** (external LLM judge, 10-conv, 1986 questions) | **73.8% Acc≥50, mean 70.5** | BM25 + RM3 query expansion → top-18 evidence → observation compression → answer → judge. Full 10-conv benchmark. |
| **LoCoMo** (external LLM judge, conv-0, 199 questions, hybrid pipeline) | **92.5% Acc≥50, mean 76.7** | Hybrid: BM25 + Qwen3-Embedding-8B (4096d) → RRF fusion → top-18 → compression → answer → judge. |
| **LoCoMo Adversarial subset** | **97.9% Acc≥50** | Subset of the conv-0 hybrid run; tests retrieval against intentionally-misleading distractor turns. |

> Comparisons: third-party self-reported numbers for Mem0 (66.88),
> Zep (65.99), Letta (74.0), Memobase (75.8) and LangMem (58.10) on
> LoCoMo; none were re-run by MIND-Mem or measured under a shared
> contract. MIND-Mem's 73.8% is above Mem0's paper figure and below
> Letta's and Memobase's self-reported figures, and it gets there with
> **zero cloud infrastructure** and **local-only retrieval** — no graph
> DB, no vector DB service, no LLM in the retrieval loop unless the
> operator opts in.

## Production usage at STARGA

MIND-Mem is the daily-driver memory layer across STARGA's active
projects, including `mind`, `mindlang.dev`, `mind-inference`, and
`arch-mind`. Used internally for cross-session recall,
contradiction detection, and audit-grade rationale chains during
agent-driven development.

This is a STARGA-internal usage statement — first-party, verifiable
in our own commit history. We do not extrapolate it into a third-party
"trusted by" claim.

## What we do not claim

- "OpenAI is a customer" — false. OpenAI runs its own memory systems.
  Codex CLI integration is software-level (MCP), not a commercial
  relationship.
- "Microsoft is a customer" — false. The GitHub Copilot integration is
  a workspace instructions file (plus an MCP entry for Copilot CLI),
  not a Microsoft Inc. commercial relationship.
- "Anthropic is a customer" — false. Claude Code is built on the MCP
  spec; MIND-Mem implements that spec; Anthropic has not endorsed,
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
- mindlang.dev / MIND-Mem product pages
- Investor decks and one-pagers
- Cold outreach emails
- LinkedIn / X marketing
- Press releases

Do not paraphrase in a way that drops the "via MCP integration"
qualifier next to vendor names. The qualifier is what keeps the
claim defensible.
