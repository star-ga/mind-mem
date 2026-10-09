<h1 align="center">
  <img src="https://raw.githubusercontent.com/star-ga/mind-mem/main/assets/logo.png" alt="MIND-Mem logo" width="140"><br>
  MIND-Mem
</h1>
<p align="center">
  <strong>Replayable memory for AI agents. Governed recall with canonical, hash-anchored audit evidence.</strong>
</p>
<p align="center">
  The governed memory layer for multi-agent and regulated use: agents propose, reviewers approve, every change is audited.
</p>
<p align="center">
  Built on the MIND substrate &bull; Governed-write &bull; Deterministic recall &bull; 107 MCP tools<br>
  <sub>MIND Language Profile: <code>default</code> (full tensor stdlib + Q16.16 + heap) &mdash; see <a href="https://github.com/star-ga/mind/blob/main/docs/roadmap.md#phase-106--library-output--c-abi-mindc-026--030">Phase 10.6</a></sub><!-- mind-profile: default -->
</p>
<p align="center">
  <a href="https://pypi.org/project/mind-mem/"><img src="https://img.shields.io/pypi/v/mind-mem?style=flat-square&color=blue&label=PyPI" alt="PyPI"></a>
  <a href="https://pypi.org/project/mind-mem/"><img src="https://img.shields.io/pypi/pyversions/mind-mem?style=flat-square" alt="Python Versions"></a>
  <a href="https://github.com/star-ga/mind-mem/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/mind-mem?style=flat-square" alt="License"></a>
  <a href="https://github.com/star-ga/mind-mem/releases"><img src="https://img.shields.io/github/v/release/star-ga/mind-mem?style=flat-square&color=green&label=Release" alt="Release"></a>
  <img src="https://img.shields.io/badge/MIND-substrate-orange?style=flat-square" alt="MIND Substrate">
  <img src="https://img.shields.io/badge/deterministic-byte--identical-brightgreen?style=flat-square" alt="Byte-identical Determinism">
  <img src="https://img.shields.io/badge/governed--write-propose→apply-purple?style=flat-square" alt="Governed Write">
  <img src="https://img.shields.io/badge/MCP-compatible-blueviolet?style=flat-square" alt="MCP Compatible">
  <img src="https://img.shields.io/badge/core_deps-zero-brightgreen?style=flat-square" alt="Zero Core Dependencies">
  <a href="https://github.com/star-ga/mind-mem/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/star-ga/mind-mem/ci.yml?branch=main&style=flat-square&label=CI" alt="CI"></a>
  <a href="https://github.com/star-ga/mind-mem/actions/workflows/release.yml"><img src="https://img.shields.io/github/actions/workflow/status/star-ga/mind-mem/release.yml?style=flat-square&label=Release" alt="Release"></a>
  <img src="https://img.shields.io/badge/test_functions-12%2C624-brightgreen?style=flat-square" alt="Test functions: 12,624">
  <img src="https://img.shields.io/badge/MCP_tools-107-blue?style=flat-square" alt="MCP Tools: 107">
  <img src="https://img.shields.io/badge/clients-20-blueviolet?style=flat-square" alt="AI Clients: 20">
  <img src="https://img.shields.io/badge/backends-markdown_%7C_postgres_%7C_encrypted-teal?style=flat-square" alt="Storage: Markdown + Postgres + Encrypted">
  <img src="https://img.shields.io/badge/audit-cross--model_%2B_SAST_%2B_SoW-darkgreen?style=flat-square" alt="Cross-model consensus audit + SAST (CodeQL/bandit/trivy) + external-audit SoW published">
</p>

<p align="center"><sub>
  <strong>Current release:</strong> <code>v5.0.4</code> (candidate; publication pending) &mdash; corrects lifecycle ranking, graph admission and release gates, and adds source-only training preparation &mdash;
  <a href="CHANGELOG.md">see CHANGELOG</a>
  (single source of truth; per-version detail tables below may lag the changelog)
</sub></p>

---

MIND-Mem is a deterministic AI memory system: recall is defined by the **query, admitted corpus, configuration, scoring instant, and execution providers**. With those inputs held constant, its canonical audit/evidence encoding is byte-identical across replay; ranking scores themselves remain standard floating-point. The Q16.16 fixed-point audit chain is embedded in every applied decision.

`scoring_instant` is a UTC date and is the honest part of that claim: recency ranking is load-bearing for a coding agent, so it is not deleted, it is *named*. Omit it and it resolves to today in UTC — the one clock read on the whole path, taken once at the boundary, never inside the scoring loop. Its resolved value is bound into the recall attestation, so any attested run replays exactly by passing that date back.

Built on the MIND substrate. Governed-write (`propose → review → approve_apply`). 107 MCP tools as the surface — but the differentiator is the substrate underneath. On the same workspace, recall uses the query, admitted corpus, configuration, `scoring_instant`, and execution providers. With those inputs held constant, the canonical audit/evidence encoding is byte-identical across replay; ranking scores remain standard floating-point, so this does not promise universal cross-provider result identity.

Most memory layers ship tools. That is table-stakes. MIND-Mem ships a substrate: Q16.16 fixed-point encoding in the audit-hash preimage, a governance pipeline that rejects every unreviewed write, and an audit chain where every applied proposal is hash-anchored. The scoring path itself is pure Python (`mind_kernels.py`): the wheel ships MIND-language kernel *sources* under `mind/` and no compiled kernel, and the optional native `libmindmem.so` is built from `lib/kernels.c` (C99). The substrate claim is the encoding, the gate and the chain — not the kernels, which are not compiled yet. The same query on the same workspace with the same admitted corpus, configuration, scoring instant and execution providers produces repeatable ranked recall; that recall's canonical audit/replay encoding is byte-identical under those held-constant conditions. That property is what makes MIND-Mem suitable as a canonical memory layer across heterogeneous agent stacks.

> **If your agent runs for weeks, it will drift. MIND-Mem prevents silent drift.**
>
> MIND-Mem powers the Memory Plane of the [MIND Cognitive Kernel](https://mindlang.dev/docs/cognitive-kernel) — the deterministic AI runtime architecture.

### 30-Second Demo

```bash
pip install mind-mem
mind-mem-init ~/my-workspace        # Create workspace
mind-mem-recall -q "API decisions" --workspace ~/my-workspace  # Hybrid BM25F search
mind-mem-scan ~/my-workspace        # Detect drift & contradictions
```

Output:
```
[1.204] D-20260215-001 (decision) — Use async/await for all API endpoints
        decisions/DECISIONS.md:11
[1.094] D-20260210-003 (decision) — REST over GraphQL for public API
        decisions/DECISIONS.md:20
```

<sub>Current release: **v5.0.4** (candidate; publication pending) — corrects lifecycle ranking and graph admission, strengthens release validation, and adds source-only training preparation. See [CHANGELOG.md](CHANGELOG.md) for candidate changes and published release history.</sub>

### Substrate Properties

| Property                | What it means                                                                     |
| ----------------------- | --------------------------------------------------------------------------------- |
| **Byte-identical replay** | Replay fixes the query, admitted corpus, configuration, `scoring_instant`, execution providers and dependencies. Canonical Q16.16 audit encoding produces identical bytes and hashes for identical preimages. Ranking uses floating-point scores; provider behavior, access-state updates and receipt metadata can change the inputs and results. |
| **Governed-write**      | Nothing reaches the source of truth without `propose → review → approve_apply`. No silent mutations. Ever. |
| **Auditable**           | Every apply logged with timestamp, receipt, and DIFF. Full traceability from signal to decision. |
| **Deterministic**       | No ML in the retrieval core. Q16.16 fixed-point encoding in the audit-hash preimage. The same preimage produces the same hash. |
| **Local-first**         | The default retrieval path stores data locally. External storage and model providers are optional and must be configured. |
| **No vendor lock-in**   | Plain Markdown files. Move to any system, any time.                               |
| **Zero infrastructure** | Core requires only Python 3.10+ stdlib. Postgres, Redis, Docker, and GPU are opt-in extras. |
| **100% NIAH**           | 250/250 Needle In A Haystack retrieval, every needle/depth/size — full-matrix repro package committed, first-party verified; no independent reproduction yet ([EVIDENCE.md](EVIDENCE.md) row 1). |

---

## Table of Contents

- [Why MIND-Mem](#why-mind-mem)
- [Features](#features)
- [Integrations are the substrate working](#integrations-are-the-substrate-working)
- [Benchmark Results](#benchmark-results)
- [Quick Start](#quick-start)
- [Health Summary](#health-summary)
- [Commands](#commands)
- [Architecture](#architecture)
- [How It Compares](#how-it-compares)
- [Companion Tools](#companion-tools)
- [Recall](#recall)
- [MIND Kernels](#mind-kernels)
- [Auto-Capture](#auto-capture)
- [Multi-Agent Memory](#multi-agent-memory)
- [Governance Modes](#governance-modes)
- [Block Format](#block-format)
- [Configuration](#configuration)
- [MCP Server](#mcp-server)
- [Security](#security)
- [Troubleshooting](#troubleshooting)
- [MIND language sources](#mind-language-sources)
- [Contributing](#contributing)
- [License](#license)

### Deep-dive docs

- [`docs/setup.md`](docs/setup.md) — install, configure, wire MCP, opt in to MIND native kernels
- [`docs/usage.md`](docs/usage.md) — every surface (MCP tools by category, `mm` CLI, `mind-mem-verify`, Python library) with worked examples
- [`docs/client-integrations.md`](docs/client-integrations.md) — **20 AI client integrations** (Claude Code, Codex, OpenCode, Grok Build, Vibe, Gemini, Cursor, Windsurf, aider, OpenClaw, NanoClaw, NemoClaw, Continue, Cline, Roo, Zed, Copilot, Copilot CLI, Cody, Qodo) with `mm install-all` auto-detection
- [`docs/task-frames.md`](docs/task-frames.md) — **task frames + the dead-end registry**: `[TF-...]` multi-session continuity (`resume_brief`, `mm resume`) and `[DE-...]` negative action-space memory, matched by a deterministic declarative overlap that warns and never blocks
- [`docs/review.md`](docs/review.md) — **`mm review`**: batch approval for the HITL queue — pending proposals with their pre-apply diff, provenance, chain status and staleness inline, approved or rejected many at once through the governed `approve_apply` path, with no auto-approve at any risk level
- [`docs/mind-mem-4b-setup.md`](docs/mind-mem-4b-setup.md) — download + run the `star-ga/mind-mem-4b` full-FT model locally (transformers, exllamav2, vLLM, llama.cpp, Ollama, **MindLLM**)
- [`docs/companion-tools.md`](docs/companion-tools.md) — **companion tools** that complement (not compete with) mind-mem: MindLLM (STARGA, commercial) for deterministic + evidence-chained inference, [GitNexus](https://github.com/abhigyanpatwari/GitNexus) for code knowledge-graph
- [`ROADMAP.md`](ROADMAP.md) — feature roadmap (genuinely-open items at the top; bulk of v3.2.0→v4.0.0 shipped)
- [`docs/specs/retrieval-receipt-contract.md`](docs/specs/retrieval-receipt-contract.md) — **draft** portable retrieval-evidence contract and acceptance gates; optional billing and settlement remain demand-gated
- [`CHANGELOG.md`](CHANGELOG.md) — release notes for every published version

---

## Why MIND-Mem

Most memory plugins **store and retrieve**. That's table stakes.

MIND-Mem also **detects when your memory is wrong** — contradictions between decisions, drift from informal choices never formalized, dead decisions nobody references, orphan tasks pointing at nothing — and offers a safe path to fix it.

| Problem                  | Without MIND-Mem                  | With MIND-Mem                             |
| ------------------------ | --------------------------------- | ----------------------------------------- |
| Contradicting decisions  | Follows whichever seen last       | Flags, links both, proposes fix           |
| Informal chat decision   | Lost after session ends           | Auto-captured, proposed to formalize      |
| Stale decision           | Zombie confuses future sessions   | Detected as dead, flagged                 |
| Orphan task reference    | Silent breakage                   | Caught in integrity scan                  |
| Scattered recall quality | Single-mode search misses context | Hybrid BM25+Vector+RRF fusion finds it    |
| Ambiguous query intent   | One-size-fits-all retrieval       | 9-type intent router optimizes parameters |

### Novel Contributions

MIND-Mem introduces several techniques not found in existing memory systems:

| Technique | What's new | Why it matters |
|-----------|-----------|----------------|
| **Co-retrieval graph** | PageRank-like score propagation across blocks frequently retrieved together | Surfaces structurally relevant blocks with zero lexical overlap (+2.0pp accuracy) |
| **Fact card sub-block indexing** | Atomic fact extraction → small-to-big retrieval with parent score blending | Catches fine-grained facts that full-block BM25 misses (+2.6pp accuracy) |
| **Adaptive knee cutoff** | Score-drop-based truncation instead of fixed top-K | Eliminates noise that hurts LLM judges — returns 3-15 results adaptively |
| **Hard negative mining** | Logs BM25-high / cross-encoder-low blocks as misleading, penalizes in future queries | Self-improving retrieval: precision increases over time without retraining |
| **Deterministic abstention** | Pre-LLM confidence gate using 5-signal scoring (entity, BM25, speaker, evidence, negation) | Prevents hallucinated answers to unanswerable questions — no ML required |
| **Governance pipeline** | Contradiction detection + drift analysis + safe apply with audit trail | Only memory system that detects when stored knowledge is wrong |
| **Agent-agnostic shared memory** | Single MCP workspace shared across Claude Code, Codex, Gemini, Cursor, Windsurf, Zed | Memory compounds across tools instead of fragmenting |

---

## Features

### Hybrid BM25+Vector Search with RRF Fusion
Thread-parallel BM25 and vector search with Reciprocal Rank Fusion (k=60). Configurable weights per signal. Vector is optional — works with just BM25 out of the box.

### RM3 Dynamic Query Expansion
Pseudo-relevance feedback using JM-smoothed language models. Expands queries with top terms from initial result set. Falls back to static synonyms for adversarial queries. Zero dependencies.

### 9-Type Intent Router
Classifies queries into WHY, WHEN, ENTITY, WHAT, HOW, LIST, VERIFY, COMPARE, or TRACE. Each intent type maps to optimized retrieval parameters (limits, expansion settings, graph traversal depth).

### A-MEM Metadata Evolution
Auto-maintained per-block metadata: access counts, importance scores (clamped to [0.8, 1.5] reranking boost), keyword evolution, and co-occurrence tracking. Importance decays with exponential recency.

### Deterministic Reranking
Four-signal reranking pipeline: negation awareness (penalizes contradicting results), date proximity (Gaussian decay), 20-category taxonomy matching, and recency boosting. No ML required.

### Optional Cross-Encoder
Drop-in ms-marco-MiniLM-L-6-v2 cross-encoder (80MB). Blends 0.6 * CE + 0.4 * original score. Falls back gracefully when unavailable. Enabled via config.

### MIND Kernel Sources and Configuration

The `mind/` directory contains 26 `.mind` files: 18 INI-style pipeline
configurations and eight MIND-language tensor-source prototypes. The
configuration files are parsed by `mind_ffi.py`; the source prototypes are
migration work and are not a native serving backend. See
[`docs/MIND_CONFIG_VS_MIND_LANG.md`](docs/MIND_CONFIG_VS_MIND_LANG.md) for the
verified split. The pure-Python scoring logic in
`src/mind_mem/mind_kernels.py` remains authoritative. An optional C library
implements the existing native scoring ABI when a compatible library is
provided.

### MIC/MAP — MIND IR graph serialization
Pure-Python codec for the STARGA wire formats: **mic@2** (line-oriented
text, LLM-readable, git-friendly) and **MIC-B** (varint binary, ~4×
smaller). Both encode typed dataflow graphs (symbols + types + values +
output) with byte-identical round-trip. Streaming parser for bounded peak
memory; optional Cython accelerator via `mind-mem[accelerated]`
(+16/+20/+36 % on parse). Two MCP tools (`mic_convert`, `mic_inspect`) and
a `mm mic` CLI surface it for agents and operators. See
[`docs/mic-map.md`](docs/mic-map.md). Note that the canonical IR per
RFC 0021 is `mic@1` text + `mic@3` binary (see
[mindlang.dev/docs/mic](https://mindlang.dev/docs/mic)); the `mic@2`/MIC-B
codec mind-mem ships is the back-compat lineage.

### BM25F Hybrid Recall
BM25F field-weighted scoring (k1=1.2, b=0.75) with per-field weighting (Statement: 3x, Title: 2.5x, Name: 2x, Summary: 1.5x), Porter stemming, bigram phrase matching (25% boost per hit), overlapping sentence chunking (3-sentence windows with 1-sentence overlap), domain-aware query expansion, and optional 2-hop graph-based cross-reference neighbor boosting. Zero dependencies. Fast and deterministic.

### Graph-Based Recall
2-hop cross-reference neighbor boosting — when a keyword match is found, blocks that reference or are referenced by the match get boosted (1-hop: 0.3x decay, 2-hop: 0.1x decay). Surfaces related decisions, tasks, and entities that share no keywords but are structurally connected. Auto-enabled for multi-hop queries.

### Vector Recall (optional)
Pluggable embedding backend — local ONNX (all-MiniLM-L6-v2, no server needed) or cloud (Pinecone). Falls back to BM25 when unavailable.

### Persistent Memory
Structured, validated, append-only decisions / tasks / entities / incidents with provenance and supersede chains. Plain Markdown files — readable by humans, parseable by machines.

### Immune System
Continuous integrity checking: contradictions, drift, dead decisions, orphan tasks, coverage scoring, regression detection. Structural validation via `validate.sh` / `validate_py` (17 checks on a fresh workspace; the count grows with the blocks present).

### Safe Governance
All changes flow through graduated modes: `detect_only` → `propose` → `enforce`. Apply engine with snapshot, receipt, DIFF, and automatic rollback on validation failure.

### Adversarial Abstention Classifier
Deterministic pre-LLM confidence gate for adversarial/verification queries. Computes confidence from entity overlap, BM25 score, speaker coverage, evidence density, and negation asymmetry. Below threshold → forces abstention without calling the LLM, preventing hallucinated answers to unanswerable questions.

### Auto-Capture with Structured Extraction
Session-end hook detects decision/task language (27 patterns with confidence classification), extracts structured metadata (subject, object, tags), and writes to `SIGNALS.md` only. Never touches source of truth directly. All signals go through `/apply`.

### Tool-Output Offload (v4.2.0)
A single `cargo test` / `pytest` / build run dumps 10k–50k lines into an agent's context — the biggest single token sink for coding agents. `mm tool-run -- <cmd>` stores the **full output out-of-context** (a `tool_outputs` sibling table; SQLite by default, reuses the Postgres connection with no new DB) and returns only a compact `{handle, summary}`; `mm tool-recall <handle>` returns the full text on demand. The summary is **bounded** regardless of input (a 10 MB line or 100k error lines can't blow it up), **fail-safe** (the full text is always stored and every truncation is explicit and counted — a failure line is never silently dropped), and **deterministic** (pure pattern extraction, no LLM; versioned config). See [docs/tool-output-architecture.md](docs/tool-output-architecture.md).

### Concurrency Safety
Cross-platform advisory file locking (`fcntl`/`msvcrt`/atomic create) protects all concurrent write paths. Stale lock detection with PID-based cleanup. Zero dependencies.

### Compaction & GC
Automated workspace maintenance: archive completed blocks, clean up old snapshots, compact resolved signals, archive daily logs into yearly files. Configurable thresholds with dry-run mode.

### Observability
Structured JSON logging (via stdlib), in-process metrics counters, and timing context managers. All scripts emit machine-parseable events. Controlled via `MIND_MEM_LOG_LEVEL` env var.

### Multi-Agent Namespaces & ACL
Workspace-level + per-agent private namespaces with JSON-based ACL. fnmatch pattern matching for agent policies. Shared fact ledger for cross-agent propagation with dedup and review gate.

### Automated Conflict Resolution
Graduated resolution pipeline: timestamp priority, confidence priority, scope specificity, manual fallback. Generates supersede proposals with integrity hashes. Human veto loop — never auto-applies without review.

### Write-Ahead Log (WAL) + Backup/Restore
Crash-safe writes via journal-based WAL. Full workspace backup (tar.gz), git-friendly JSONL export, selective restore with conflict detection and path traversal protection.

### Transcript JSONL Capture
Scans Claude Code transcript files for user corrections, convention discoveries, bug fix insights, and architectural decisions. 16 transcript-specific patterns with role filtering and confidence classification.

### MCP Server (107 tools, 8 resources)
Full [Model Context Protocol](https://modelcontextprotocol.io/) server with 107 distinct tools and 8 read-only resources (6 static + 2 templated). Works with Claude Code, Claude Desktop, Cursor, Windsurf, and any MCP-compatible client. HTTP and stdio transports; HTTP requires bearer-token auth (fail-closed) — see [Token Auth (HTTP)](#token-auth-http). v3.8.11 added `mic_convert_tool` / `mic_inspect_tool` (MIC/MAP wire format); v3.9.0 added `compile_truth_walkthrough`, `recall_with_persona`, `pipeline_status`, and `reindex_dirty`; v3.11.0 added `validate_block`, `block_lineage`, and `add_block_edge` (deterministic quality gates + typed lineage edges).

### Structural Validation + 12,624 Test Functions
`validate.sh` checks schemas, cross-references, ID formats, status values, supersede chains, ConstraintSignatures, and more. The repository contains 12,624 test functions across the core and optional surfaces; collected case counts also depend on parametrization, optional dependencies and test selectors.

### Audit Trail
Every applied proposal logged with timestamp, receipt, and DIFF. Full traceability from signal → proposal → decision.

### Calibration Feedback Loop
Per-block quality tracking with Bayesian weight computation. When users provide feedback (thumbs up/down) via `calibration_feedback`, the system maintains a rolling quality score per block over a 30-day window. Bayesian smoothing constrains calibration weights to the 0.5-1.5 range, preventing any single block from dominating or being silenced. Calibration weights integrate directly into the BM25 + FTS5 retrieval pipeline — high-quality blocks rank higher, low-quality blocks are naturally demoted. Use `calibration_stats` to inspect per-block quality distributions and global calibration health. With `v4.llm_noise_profile` enabled (default off), it also carries an `llm_reliability` section: a per-provider, per-domain reliability EMA fed by `report_outcome` and persisted to `intelligence/llm_profiles.json`. Reliability is evidence for an operator reading the record — nothing on the retrieval scoring path reads it.

### LLM-Guided Multi-Query Expansion
Generates semantically diverse query reformulations before search — synonym expansion, specificity shifts, temporal rephrasing, and negation variants. Combines all reformulated queries with Reciprocal Rank Fusion for broader recall without sacrificing precision. Runs locally with zero API calls.

### 4-Layer Search Deduplication
Post-retrieval dedup pipeline: best-chunk-per-source (keeps highest-scoring chunk from each file), cosine similarity dedup (>0.85 threshold), type diversity capping (max 3 results per block type), and per-source chunk limiting. Eliminates redundant results that waste LLM context.

### LLM-Guided Smart Chunking
Content-aware chunking that splits at semantic boundaries (headers, paragraph breaks, list items, code blocks) instead of fixed character counts. Produces variable-size chunks with overlap for continuity. Supports markdown, code, and prose with format-specific splitting rules.

### Compiled Truth Pages
Per-entity knowledge compilation: current-best-understanding on top, timestamped evidence trail below. Contradiction detection across evidence entries with automatic flagging. Entities accumulate knowledge from all sessions — each new evidence entry is checked against existing facts.

### Dream Cycle (Autonomous Memory Enrichment)
Scheduled background enrichment: scans recent memory for missing cross-references, broken citations, orphan entities, and consolidation opportunities. Generates repair proposals for stale links, detects implicit entities not yet formalized, and compacts redundant entries. Runs during idle periods with configurable depth.

### Feature Completeness Matrix

Cells were checked against each project's public source or, where the engine is closed (Supermemory's engine, Graphlit), its official docs, in October 2026. Several retrieval legs exist in other projects only behind an optional backend; see the Full Feature Matrix below.

**Columns:** **MM** MIND-Mem · **M0** [Mem0](https://github.com/mem0ai/mem0) · **SM** [Supermemory](https://supermemory.ai) · **CM** [claude-mem](https://github.com/thedotmack/claude-mem) · **Le** [Letta](https://www.letta.com) · **Zep** [Zep](https://www.getzep.com) · **LM** [LangMem](https://github.com/langchain-ai/langmem) · **Co** [Cognee](https://www.cognee.ai) · **GL** [Graphlit](https://www.graphlit.com) · **CW** [ClawMem](https://github.com/yoloshii/ClawMem) · **MU** [MemU](https://github.com/NevaMind-AI/memU) · **En** [Engram](https://github.com/Gentleman-Programming/engram) · **BM** [Basic Memory](https://github.com/basicmachines-co/basic-memory)  
**Cells:** ✓ present · ◐ partial or a different mechanism · opt present but off by default · — looked for and not found · n/v not verified (closed engine or undocumented) · N/A not applicable · ↓<sup>n</sup> see note *n* under the table. Small notes under each table carry the qualifiers.

**Recall**

| Capability | MM | M0 | SM | CM | Le | Zep | LM | Co | GL | CW | MU | En | BM |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| BM25 lexical | ✓ | ✓ | ◐ | ✓ | ◐ | ✓ | — | ✓ | ◐ | ✓ | — | ✓ | ✓ |
| Vector semantic | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — | ✓ |
| Hybrid fusion<sup>1</sup> | ✓ | ◐ | ◐ | ◐ | ◐ | ✓ | — | ◐ | ◐ | ✓ | — | — | ◐ |
| Cross-encoder | ✓ | opt | opt | — | ◐ | ✓ | — | — | ✓ | ✓ | — | — | opt |
| Intent routing<sup>2</sup> | ✓ | — | — | — | — | — | — | ◐ | — | ✓ | — | — | — |
| Query expansion<sup>3</sup> | ✓ | — | ◐ | — | — | — | ◐ | ◐ | ◐ | ◐ | — | — | — |
| Graph boost<sup>4</sup> | ✓ | ◐ | ◐ | — | — | ◐ | — | ✓ | ◐ | ✓ | — | — | ◐ |
| Fact sub-blocks | ✓ | ◐ | ✓ | ✓ | — | ✓ | — | ◐ | ◐ | ✓ | ◐ | — | ✓ |
| Hard negatives | ✓ | — | — | — | — | — | — | — | — | — | — | — | — |
| Knee cutoff | ✓ | ◐ | ◐ | — | — | ◐ | — | — | — | ◐ | — | — | ◐ |
| Multi-query + RRF | ✓ | — | ◐ | — | — | — | ◐ | ◐ | — | ✓ | — | — | — |
| Search dedup<sup>5</sup> | ✓ | ◐ | n/v | ◐ | — | ◐ | ◐ | ◐ | n/v | ✓ | — | ◐ | ◐ |
| Semantic chunking | ✓ | — | ✓ | ◐ | ◐ | ◐ | — | ◐ | ◐ | ✓ | — | — | ✓ |

<sub><sup>1</sup> MIND-Mem: BM25+vector+RRF<br><sup>2</sup> MIND-Mem: 9 types<br><sup>3</sup> MIND-Mem: RM3<br><sup>4</sup> MIND-Mem: co-retrieval PageRank<br><sup>5</sup> MIND-Mem: 4 layers</sub>

**Persistence**

| Capability | MM | M0 | SM | CM | Le | Zep | LM | Co | GL | CW | MU | En | BM |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Truth pages<sup>1</sup> | ✓ | — | ◐ | ◐ | — | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | — | — |
| Background enrichment<sup>2</sup> | ✓ | ◐ | ✓ | ◐ | ✓ | ◐ | ◐ | ✓ | ◐ | opt | ✓ | — | — |
| Backup/restore<sup>3</sup> | ✓ | ◐ | — | ◐ | ◐ | — | — | — | — | — | — | ◐ | ◐ |

<sub><sup>1</sup> compiled, per entity<br><sup>2</sup> MIND-Mem: dream cycle<br><sup>3</sup> with zip-slip protection</sub>

**Integrity**

| Capability | MM | M0 | SM | CM | Le | Zep | LM | Co | GL | CW | MU | En | BM |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Contradictions | ✓ | ◐ | ✓ | — | ◐ | ✓ | ◐ | ◐ | — | ◐ | — | ◐ | — |
| Drift analysis | ✓ | — | — | — | — | — | — | ◐ | — | — | — | ◐ | ◐ |

**Governance**

| Capability | MM | M0 | SM | CM | Le | Zep | LM | Co | GL | CW | MU | En | BM |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Propose/apply<sup>1</sup> | ✓ | — | ◐ | — | — | — | — | ◐ | ◐ | — | — | — | — |
| Shared memory<sup>2</sup> | ✓ | ◐ | ✓ | ✓ | ✓ | ✓ | ◐ | ✓ | ✓ | ✓ | ◐ | ✓ | ◐ |

<sub><sup>1</sup> governance pipeline<br><sup>2</sup> multi-agent, via MCP or API</sub>

**Operations**

| Capability | MM | M0 | SM | CM | Le | Zep | LM | Co | GL | CW | MU | En | BM |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Zero core deps | ✓ | — | n/v | — | — | — | — | — | — | — | — | ✓ | — |
| Local-only<sup>1</sup> | ✓ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ✓ | — | ✓ | ◐ | ✓ | ✓ |
| Native C scorer<sup>2</sup> | ✓ | — | — | — | — | — | — | — | — | ◐ | ◐ | — | ◐ |

<sub><sup>1</sup> no cloud required<br><sup>2</sup> optional backend</sub>

---

## Integrations are the substrate working

MIND-Mem provides integrations for 20 supported clients, including 12 MCP-aware clients. They can share a governed memory workspace. Reproducing a recall result requires the same request, corpus, configuration, scoring instant and execution dependencies; the client count alone does not establish cross-client output identity.

> Honest positioning: the integrations below are *software-level* —
> clients use their configured MCP connection or local integration.
> They are **not** commercial-customer relationships with any vendor.
> Full positioning policy: [`docs/integrations.md`](docs/integrations.md).

### Native integration with 20 clients (12 MCP-aware clients)

```bash
pip install mind-mem
mm install-all
```

`mm install-all` auto-detects every supported client on your machine
and writes the appropriate config file for each. MIND-Mem speaks the
[Model Context Protocol](https://modelcontextprotocol.io/) — any
MCP-compatible client connects with one command.

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

### Compatible with major LLM providers

MIND-Mem's recall pipeline is provider-agnostic: any MCP-capable client or
OpenAI-compatible endpoint can use the same server interface. The provider
adapters (Anthropic; OpenAI-compatible, which Mistral and other OpenAI-style
APIs route through; Ollama; vLLM; llama.cpp) are covered by mocked contract
tests, which make no live call. The
only model-in-the-loop benchmark run so far is LoCoMo, with
`mistral-large-latest` as answerer and judge. We do not claim live testing
against specific versions of other vendors' models. Replay also requires the
same query, admitted corpus, configuration, scoring instant, execution
providers and dependencies; different client models can generate different
queries.

### Production usage at STARGA

MIND-Mem is the daily-driver memory layer across STARGA's active
projects, including `mind`, `mindlang.dev`, `mind-inference`, and
`arch-mind`. First-party, verifiable in our own commit history.

### What we do not claim

- ❌ "OpenAI / Microsoft / Anthropic / Google is a customer" — false.
  These are software-level MCP integrations, not commercial
  relationships.
- ❌ "Used by N production teams outside STARGA" — we have no
  telemetry. PyPI download counts measure installs, not active use.

If a future integration becomes a real commercial relationship
(signed contract, paid pilot, named reference), it will appear in
the press release first — not in the README.

---

## Benchmark Results

MIND-Mem's recall engine evaluated on standard long-term memory benchmarks using multiple configurations — from pure BM25 to full hybrid retrieval with neural reranking.

### Needle In A Haystack (NIAH)

**250/250 — 100% retrieval** across all haystack sizes, burial depths, and needle types.

> **Provenance.** The full-matrix repro package is committed at
> `benchmarks/repro/niah/` (raw per-case rows, recomputed metrics, and a
> manifest pinning the commit, config, seeds and hardware — produced on a
> clean tree, `repo_tracked_files_dirty_at_run: false`). `make repro-verify`
> recomputes the 250/250 headline from those raw rows rather than trusting
> the manifest's summary. This is **first-party** evidence: nobody outside
> STARGA has re-run it yet and reported the same
> `metrics.determinism.decision_fingerprint` — see [EVIDENCE.md](EVIDENCE.md)
> row 1 for exactly what "verified" does and does not mean here.

A single fact is planted at a controlled depth within a haystack of semantically diverse filler blocks. The system must retrieve the needle in its top-5 results using only a natural-language query.

| Haystack Size | Depths Tested | Needles | Passed | Rate |
|---------------|---------------|---------|--------|------|
| 10 blocks | 0/25/50/75/100% | 10 | 50/50 | 100% |
| 50 blocks | 0/25/50/75/100% | 10 | 50/50 | 100% |
| 100 blocks | 0/25/50/75/100% | 10 | 50/50 | 100% |
| 200 blocks | 0/25/50/75/100% | 10 | 50/50 | 100% |
| 500 blocks | 0/25/50/75/100% | 10 | 50/50 | 100% |

**Config:** Hybrid BM25 + all-MiniLM-L6-v2 + RRF (k=60) + sqlite-vec. Full details: [benchmarks/NIAH.md](benchmarks/NIAH.md)

### LoCoMo LLM-as-Judge

Same pipeline as Mem0 and Letta evaluations: retrieve context, generate answer with LLM, score against gold reference with judge LLM. Directly comparable methodology.

> **Canonical flagship number:** the full 10-conversation (1986-question) BM25
> run directly below (Overall Acc>=50 **73.8%**, Mean **70.5**) — see
> [`docs/benchmarks.md`](docs/benchmarks.md#locomo-benchmark--canonical-number)
> for scope, evidence, and reproduction. All other LoCoMo tables on this page
> are smaller historical subsamples, kept for their per-category detail — do
> not treat any of them as the headline number.

**v1.0.7 — Hybrid + top_k=18** (external LLM answerer + judge, conv-0 subsample, 199 of 1986 questions — 10% of the full set; no raw per-question artifact is checked into this repo for this run):

| Category        |      N | Acc (>=50) | Mean Score |
| --------------- | -----: | ---------: | ---------: |
| **Overall**     | **199**|  **92.5%** |   **76.7** |
| Adversarial     |     47 |      97.9% |       89.8 |
| Multi-hop       |     37 |      91.9% |       74.3 |
| Open-domain     |     70 |      92.9% |       72.7 |
| Temporal        |     13 |      92.3% |       76.2 |
| Single-hop      |     32 |      84.4% |       68.9 |

> **Pipeline:** BM25 + Qwen3-Embedding-8B (4096d) vector search → RRF fusion (k=60) → top-18 evidence blocks → observation compression → answer → judge. A/B validated: +2.8 mean vs top_k=10 baseline.

**v1.1.1 — BM25 + top_k=18 (canonical — full dataset)** (external LLM answerer + judge, 10 conversations, 1986 questions):

| Category        |        N | Acc (>=50) | Mean Score |
| --------------- | -------: | ---------: | ---------: |
| **Overall**     | **1986** |  **73.8%** |   **70.5** |
| Adversarial     |      446 |      92.4% |       87.2 |
| Single-hop      |      282 |      80.9% |       68.7 |
| Open-domain     |      841 |      71.2% |       70.3 |
| Temporal        |       96 |      66.7% |       65.9 |
| Multi-hop       |      321 |      50.5% |       51.1 |

> **Pipeline:** BM25 + RM3 query expansion → top-18 evidence blocks → observation compression → answer → judge. Full 10-conversation benchmark with the same external LLM as both answerer and judge.

**v1.0.0 — BM25-only baseline** (external LLM answerer + judge, 10 conversations):

| Category    |        N | Acc (>=50) | Mean Score |
| ----------- | -------: | ---------: | ---------: |
| **Overall** | **1986** |  **67.3%** |   **61.4** |
| Open-domain |      841 |      86.6% |       78.3 |
| Temporal    |       96 |      78.1% |       65.7 |
| Single-hop  |      282 |      68.8% |       59.1 |
| Multi-hop   |      321 |      55.5% |       48.4 |
| Adversarial |      446 |      36.3% |       39.5 |

> **Key improvements since v1.0.0:** Adversarial accuracy tripled from 36.3% to 92.4% via abstention classifier + hybrid retrieval. Overall Acc≥50 improved from 67.3% to 73.8% (+6.5pp).

### Competitive Landscape (LoCoMo)

Full 10-conversation (1986-question) LoCoMo, Acc>=50 — apples-to-apples scope
and metric. Canonical MIND-Mem table + evidence:
[`docs/benchmarks.md`](docs/benchmarks.md#locomo-benchmark--canonical-number).
On this metric **MIND-Mem is not the top score** — Memobase and Letta report
slightly higher, both on cloud infrastructure with embedding + vector-DB
dependencies. MIND-Mem's differentiator is not "wins every cell"; it's being
the only **local-only, zero-core-dependency, governed** system in the table —
governance (contradiction detection, drift analysis, proposal/review/apply
audit trail, byte-identical replay) is a property no other row has, and it
is not measured by this benchmark at all.

| System | LoCoMo Acc>=50 (full 10-conv, 1986Q) | Infrastructure | Dependencies |
| --- | ---: | --- | --- |
| Memobase¹ | 75.8% | Cloud + GPU | embeddings + vector DB |
| Letta¹ | 74.0% | Cloud | embeddings + vector DB |
| **MIND-Mem** (BM25) | **73.8%** | **Local-only** | **Zero core** |
| Full-context¹ | 72.9% | N/A | LLM context window |
| Mem0 (own LoCoMo paper)² | 66.9% | Cloud (managed) | graph DB + embeddings |

¹ Third-party **self-reported** numbers (Letta's August 2025 analysis — see
"Why Plain Files Outperform Fancy Retrieval" below). **Not re-run by MIND-Mem, and
not measured under a shared contract:** different hardware, different judge
configuration, and each system's own harness. Rows in this table are therefore
*indicative of scope*, not a head-to-head result, and none of them — including
ours — has been reproduced by the others. A genuine comparison requires every
system run under one adapter contract on one box with a pinned judge and >=2
reps; that run does not exist yet, and until it does no ordering in this table
should be read as a ranking.

² `66.88` is Mem0's own published LoCoMo-paper number. Mem0's separate 2026
managed platform self-reports **91.6** on LoCoMo — a different setup/judge
(hosted product, not the open-paper config), not apples-to-apples with this
table. Surfaced rather than omitted, per policy: never publish a comparison
a skeptic could catch as cherry-picked.

### LongMemEval-S

> The previous headline (`R@5 = 85.3`) is **retracted**, not held — it had no committed
> artifact, was not reproducible after two attempts, and its own per-category rows summed
> to 376 under a stated N=470. It is replaced by the measurement below, which ships with
> per-question NDJSON so anyone can recompute it. See
> [`benchmarks/STATUS.md`](benchmarks/STATUS.md) and
> [`benchmarks/REPORT.md`](benchmarks/REPORT.md).

Full eligible set (470 of 500; 30 abstention questions excluded), two reps identical per
question, artifacts under `docs/benchmarks/2026-09-03-longmemeval-s-full-*`.
**Configuration: BM25F/SQLite with the vector leg OFF** — one leg of the product, not the
shipped hybrid. The hybrid number does not exist yet and nothing here may be read as one.

| adapter | recall_any@5 | recall_all@5 (official) | MRR |
| --- | ---: | ---: | ---: |
| `mind_mem` (BM25F/SQLite, vector off) | 0.9404 | 0.8170 | 0.8776 |
| `bm25_baseline` (zero-dependency, in-memory) | 0.9702 | 0.8298 | 0.9081 |

Paired over the same 470 question ids: on the **official strict protocol
(`recall_all@5`) the two are statistically indistinguishable** (McNemar exact, p=0.4799);
the zero-dependency baseline is better on the lenient protocol (`recall_any@5`, p=0.0043)
and on MRR (p=0.0013). An independent audit refuted every artefact explanation — the
recall caps are genuinely off in the pinned config, there is no ingest truncation, and
index fragment ids never reach recall — so the deficit is **ordering quality, not
candidate recall**. That is the work in flight; we publish the number that exists rather
than the one we want.

### Performance (Latency & Throughput)

Measured on a single developer workstation (commodity x86-64, warm cache, single process) against a 65-block workspace (typical personal workspace) with the SQLite FTS5 backend. Absolute latencies are hardware-dependent — the portable claim is the O(log N) scaling noted below, not the millisecond figures. These figures come from a single-workstation run; no committed artifact backs them yet and `make repro-verify` does not cover them:

| Operation | Metric | Value |
|-----------|--------|-------|
| **Query** (FTS5 + rerank) | p50 latency | **2.1 ms** |
| **Query** (FTS5 + rerank) | p95 latency | **4.9 ms** |
| **Query** (FTS5 + rerank) | mean latency | **2.6 ms** |
| **Incremental reindex** | elapsed | **32 ms** (13 blocks indexed) |
| **Full index build** | elapsed | **48 ms** (65 blocks) |
| **MCP tool overhead** | stdio round-trip | **< 15 ms** |
| **Memory footprint** | RSS (idle MCP server) | **~28 MB** |

Query latency scales as O(log N) with SQLite FTS5 (vs O(corpus) for scan backend). The co-retrieval graph adds < 1ms per query. Knee cutoff and fact aggregation add negligible overhead (< 0.5ms).

### Feedback-Quality -> Downstream-Success (synthetic, deterministic)

Downstream-success prediction (synthetic, deterministic): starved 0.00 -> sufficient 1.00 at matched budget. 48-episode regression gate over the v4.7.0 per-hit feedback-quality credit + v4.8.0 recall-sufficiency score; see [`benchmarks/REPORT.md`](benchmarks/REPORT.md#feedback-quality---downstream-success-group-i-item-3-synthetic-deterministic) and [`benchmarks/feedback_success_bench.py`](benchmarks/feedback_success_bench.py).

### Run Benchmarks Yourself

```bash
# Retrieval-only (R@K metrics)
python3 benchmarks/locomo_harness.py
python3 benchmarks/longmemeval_harness.py

# LLM-as-judge (accuracy metrics, requires API key)
python3 benchmarks/locomo_judge.py --dry-run
python3 benchmarks/locomo_judge.py --answerer-model <your-answerer-model> --output results.json

# Hybrid retrieval with any model pair (BM25 + vector + cross-encoder)
python3 benchmarks/locomo_judge.py --hybrid --compress --answerer-model <your-answerer-model> --judge-model <your-judge-model> --output results.json

# Selective conversations
python3 benchmarks/locomo_harness.py --conv-ids 4,7,8
```

---

## Install in 3 commands

```bash
pip install mind-mem
mm install-all --force      # auto-wires every detected AI CLI
mm install-model            # downloads mind-mem-4b GGUF + imports to Ollama
```

Full options + Postgres setup + troubleshooting:
[**docs/install-guide.md**](docs/install-guide.md)

---

## Quick Start

### One-line install (recommended)

```bash
pipx install "mind-mem[mcp]"
mind-mem-mcp --help          # smoke-test
```

`pipx` keeps MIND-Mem in its own venv, exposes the `mind-mem-mcp` console
script on `PATH`, and avoids polluting your system Python. If you don't have
pipx, `pip install --user "mind-mem[mcp]"` works too.

Then wire it into every AI coding client on your machine:

```bash
git clone https://github.com/star-ga/mind-mem.git
cd mind-mem
./install.sh --all --no-install   # Already installed via pipx, just wire clients
```

Or do both in one shot (the installer will auto-pick pipx if available, else
fall back to pip):

```bash
git clone https://github.com/star-ga/mind-mem.git
cd mind-mem
./install.sh --all
```

`./install.sh` wires the MCP server into a fixed set of eight clients:
Claude Code, Claude Desktop, Codex CLI, Gemini CLI, Cursor, Windsurf, Zed and
OpenClaw. Each client launches the same `mind-mem-mcp` binary, so all agents
share one workspace.

For the full set of 20 clients (12 of them get an MCP config), use
`mm install-all` after installing the package; it auto-detects what is on
your machine and writes these files:

| Client | Config Location | Format |
| ------ | --------------- | ------ |
| **Claude Code** (`claude-code`) | `~/.claude/settings.json` | JSON (hooks) |
| **Codex CLI** (`codex`) | `<workspace>/AGENTS.md` + MCP `~/.codex/config.toml` | Markdown block + TOML |
| **Grok Build CLI** (`grok-build`) | `<workspace>/AGENTS.md` + MCP `~/.grok/config.toml` | Markdown block + TOML |
| **Vibe (Mistral CLI)** (`vibe`) | `<workspace>/AGENTS.md` + MCP `~/.vibe/config.toml` | Markdown block + TOML |
| **OpenCode (1.x / 2.x)** (`opencode`) | `~/.config/opencode/AGENTS.md` + MCP `~/.config/opencode/opencode.json` | Markdown block + JSON |
| **Gemini CLI** (`gemini`) | `<workspace>/.gemini/settings.json` + MCP `~/.gemini/settings.json` | JSON |
| **Cursor** (`cursor`) | `<workspace>/.cursorrules` + MCP `~/.cursor/mcp.json` | Markdown block + JSON |
| **Windsurf** (`windsurf`) | `<workspace>/.windsurfrules` + MCP `~/.codeium/windsurf/mcp_config.json` | Markdown block + JSON |
| **aider** (`aider`) | `<workspace>/.aider.conf.yml` | YAML |
| **OpenClaw** (`openclaw`) | `~/.openclaw/openclaw.json` | JSON (hooks) |
| **NanoClaw** (`nanoclaw`) | `~/.nanoclaw/nanoclaw.json` | JSON (hooks) |
| **NemoClaw** (`nemoclaw`) | `~/.nemoclaw/nemoclaw.json` | JSON (hooks) |
| **Continue** (`continue`) | `~/.continue/config.json` (instructions + MCP) | JSON |
| **Cline** (`cline`) | `<workspace>/.clinerules` + MCP `~/.vscode-server/data/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json` | Markdown block + JSON |
| **Roo Code** (`roo`) | `<workspace>/.roo/system-prompt.md` + MCP `~/.vscode-server/data/User/globalStorage/rooveterinaryinc.roo-cline/settings/mcp_settings.json` | Markdown block + JSON |
| **Zed** (`zed`) | `~/.config/zed/settings.json` (instructions + MCP) | JSON |
| **GitHub Copilot (workspace instructions)** (`copilot`) | `<workspace>/.github/copilot-instructions.md` | Markdown block |
| **GitHub Copilot CLI** (`copilot-cli`) | `<workspace>/AGENTS.md` + MCP `~/.copilot/mcp-config.json` | Markdown block + JSON |
| **Cody** (`cody`) | `<workspace>/.cody/config.json` | JSON |
| **Qodo Gen** (`qodo`) | `<workspace>/.codium/ai-rules.md` | Markdown block |

> **Cline and Roo Code:** the installer writes the VS Code Server (remote / WSL)
> settings tree, `~/.vscode-server/data/User/`. Desktop VS Code keeps its user
> settings elsewhere (for example `~/.config/Code/User/` on Linux), and those
> paths are not written yet; add the MCP entry there by hand for now.

Selective install:

```bash
./install.sh --claude-code --codex --gemini         # Only specific clients
./install.sh --all --workspace ~/my-project/memory  # Custom workspace path
```

Uninstall:

```bash
./uninstall.sh          # Remove from all clients (keeps workspace data)
./uninstall.sh --purge  # Remove everything including workspace data
```

### Manual Setup

For manual or per-project setup:

**1. Clone into your project**

```bash
cd /path/to/your/project
git clone https://github.com/star-ga/mind-mem.git .mind-mem
```

**2. Initialize workspace**

```bash
PYTHONPATH=.mind-mem/src python3 -m mind_mem.init_workspace .
# or, after `pip install -e .mind-mem`:  mind-mem-init .
```

Creates the directory tree, the template files and the `mind-mem.json` config. **Never overwrites existing files.**

**3. Validate**

```bash
bash .mind-mem/src/mind_mem/validate.sh .
# or cross-platform:
PYTHONPATH=.mind-mem/src python3 -m mind_mem.validate_py .
```

Expected: `0 issues` (17 checks on a fresh workspace; warnings for empty sections are normal).

**4. First scan**

```bash
PYTHONPATH=.mind-mem/src python3 -m mind_mem.intel_scan .
```

Expected: `0 critical | 0 warnings` on a fresh workspace.

**5. Verify recall + capture**

```bash
PYTHONPATH=.mind-mem/src python3 -m mind_mem.recall --query "test" --workspace .
# → No results found. (empty workspace — correct)

PYTHONPATH=.mind-mem/src python3 -m mind_mem.capture .
# → capture: no daily log for YYYY-MM-DD, nothing to scan (correct)
```

**6. Add hooks (optional)**

**Option A: Claude Code hooks** (recommended)

Merge into your `.claude/hooks.json`:

```json
{
  "hooks": [
    {
      "event": "SessionStart",
      "command": "bash .mind-mem/hooks/session-start.sh"
    },
    {
      "event": "Stop",
      "command": "bash .mind-mem/hooks/session-end.sh"
    }
  ]
}
```

**Option B: OpenClaw hooks** (for OpenClaw 2026.2+)

```bash
cp -r .mind-mem/hooks/openclaw/mind-mem ~/.openclaw/hooks/mind-mem
openclaw hooks enable mind-mem
```

**7. Smoke Test (optional)**

```bash
bash .mind-mem/src/mind_mem/smoke_test.sh
```

Creates a temp workspace, runs init → validate → scan → recall → capture → pytest, then cleans up.

---

## Health Summary

After setup, this is what a healthy workspace looks like:

```
$ python3 -m mind_mem.intel_scan .

mind-mem Intelligence Scan Report v2.0
Mode: detect_only

=== 1. CONTRADICTION DETECTION ===
  OK: No contradictions found among 25 signatures.

=== 2. DRIFT ANALYSIS ===
  OK: All active decisions referenced or exempt.
  INFO: Metrics: active_decisions=17, active_tasks=7, blocked=0,
        dead_decisions=0, incidents=3, decision_coverage=100%

=== 3. DECISION IMPACT GRAPH ===
  OK: Built impact graph: 11 decision(s) with edges.

=== 4. STATE SNAPSHOT ===
  OK: Snapshot saved.

=== 5. WEEKLY BRIEFING ===
  OK: Briefing generated.

TOTAL: 0 critical | 0 warnings | 16 info
```

---

## Commands

| Command           | What it does                                                                                    |
| ----------------- | ----------------------------------------------------------------------------------------------- |
| `/scan`           | Run integrity scan — contradictions, drift, dead decisions, impact graph, snapshot, briefing    |
| `/apply`          | Review and apply proposals from scan results (dry-run first, then apply)                        |
| `/recall <query>` | Search across all memory files with ranked results (add `--graph` for cross-reference boosting) |

The three skills above live in [`skills/`](skills/).

### Agent skill: CLI user manual

Agents that prefer the shell over MCP can install a `mind-mem` skill that works
as a user manual for the `mm` CLI. Its [`SKILL.md`](skills/mind-mem/SKILL.md)
is a table of contents — what mind-mem is, which command to run for which job,
and how governed writes work — and it points to reference files the agent opens
only when it needs them:

| Reference | Covers |
| --- | --- |
| [`install.md`](skills/mind-mem/references/install.md) | Install, workspaces, wiring clients, upgrading, removing |
| [`cli.md`](skills/mind-mem/references/cli.md) | Every `mm` subcommand with its flags and an example (generated from the parser) |
| [`configuration.md`](skills/mind-mem/references/configuration.md) | `mind-mem.json` keys, `mm config set`, environment variables |
| [`troubleshooting.md`](skills/mind-mem/references/troubleshooting.md) | `mm doctor`, `mind-mem-verify`, common errors and their fixes |
| [`mcp-vs-cli.md`](skills/mind-mem/references/mcp-vs-cli.md) | When to use which, and the MCP-tool-to-command map |
| [`faq.md`](skills/mind-mem/references/faq.md) | Basic questions about the product |

```bash
mm skill install                             # -> ~/.claude/skills/mind-mem
mm skill install --target ~/.codex/skills    # any agent that reads SKILL.md folders
```

The skill ships inside the package. `tests/test_skill_manual.py` fails the build
if a documented command, flag, environment variable or config key stops
existing, and `python3 scripts/gen_skill_cli_reference.py` regenerates the CLI
reference after the parser changes.

---

## Architecture

```
your-workspace/
├── mcp_server.py            # MCP server (FastMCP, 107 tools, 8 resources)
├── mind-mem.json             # Config
├── MEMORY.md                # Protocol rules
│
├── mind/                    # 26 .mind files: 18 INI config + 8 MIND sources
│   ├── README.md            # Source/config inventory and migration status
│   ├── bm25.mind            # MIND-language source prototype
│   ├── rrf.mind             # MIND-language source prototype
│   ├── ranking.mind         # MIND-language source prototype
│   └── recall.mind          # INI pipeline configuration example
│
├── lib/                     # Optional native C scoring backend
│   └── libmindmem.so        # Locally built from lib/kernels.c; not bundled
│
├── decisions/
│   └── DECISIONS.md         # Formal decisions [D-YYYYMMDD-###]
├── tasks/
│   └── TASKS.md             # Tasks [T-YYYYMMDD-###]
├── entities/
│   ├── projects.md          # [PRJ-###]
│   ├── people.md            # [PER-###]
│   ├── tools.md             # [TOOL-###]
│   └── incidents.md         # [INC-###]
│
├── memory/
│   ├── YYYY-MM-DD.md        # Daily logs (append-only)
│   ├── intel-state.json     # Scanner state + metrics
│   └── maint-state.json     # Maintenance state
│
├── summaries/
│   ├── weekly/              # Weekly summaries
│   └── daily/               # Daily summaries
│
├── intelligence/
│   ├── CONTRADICTIONS.md    # Detected contradictions
│   ├── DRIFT.md             # Drift detections
│   ├── SIGNALS.md           # Auto-captured signals
│   ├── IMPACT.md            # Decision impact graph
│   ├── BRIEFINGS.md         # Weekly briefings
│   ├── AUDIT.md             # Applied proposal audit trail
│   ├── SCAN_LOG.md          # Scan history
│   ├── proposed/            # Staged proposals + resolution proposals
│   │   ├── DECISIONS_PROPOSED.md
│   │   ├── TASKS_PROPOSED.md
│   │   ├── EDITS_PROPOSED.md
│   │   └── RESOLUTIONS_PROPOSED.md
│   ├── applied/             # Snapshot archives (rollback)
│   └── state/snapshots/     # State snapshots
│
├── shared/                  # Multi-agent shared namespace
│   ├── decisions/
│   ├── tasks/
│   ├── entities/
│   └── intelligence/
│       └── LEDGER.md        # Cross-agent fact ledger
│
├── agents/                  # Per-agent private namespaces
│   └── <agent-id>/
│       ├── decisions/
│       ├── tasks/
│       └── memory/
│
├── mind-mem-acl.json        # Multi-agent access control
├── .mind-mem-wal/           # Write-ahead log (crash recovery)
│
└── src/mind_mem/
    ├── mind_ffi.py          # MIND FFI bridge (ctypes)
    ├── hybrid_recall.py     # Hybrid BM25+Vector+RRF orchestrator
    ├── block_metadata.py    # A-MEM metadata evolution
    ├── cross_encoder_reranker.py  # Optional cross-encoder
    ├── intent_router.py     # 9-type intent classification (adaptive)
    ├── recall.py            # BM25F + RM3 + graph scoring engine
    ├── recall_vector.py     # Vector/embedding backends
    ├── sqlite_index.py      # FTS5 + vector + metadata index
    ├── connection_manager.py # SQLite connection pool (WAL read/write separation)
    ├── block_store.py       # BlockStore protocol + MarkdownBlockStore
    ├── corpus_registry.py   # Central corpus path registry
    ├── abstention_classifier.py  # Adversarial abstention
    ├── evidence_packer.py   # Evidence assembly and ranking
    ├── intel_scan.py        # Integrity scanner
    ├── apply_engine.py      # Proposal apply engine (delta-based snapshots)
    ├── block_parser.py      # Markdown block parser (typed)
    ├── capture.py           # Auto-capture (27 patterns)
    ├── compaction.py        # Compaction/GC/archival
    ├── mind_filelock.py     # Cross-platform advisory file locking
    ├── observability.py     # Structured JSON logging + metrics
    ├── namespaces.py        # Multi-agent namespace & ACL
    ├── conflict_resolver.py # Automated conflict resolution
    ├── backup_restore.py    # WAL + backup/restore + JSONL export
    ├── transcript_capture.py  # Transcript JSONL signal extraction
    ├── validate.sh          # Structural validator (shell)
    └── validate_py.py       # Structural validator (Python, cross-platform)
```

---

## How It Compares

### Quick Comparison

| Feature | MIND-Mem | Mem0 | Letta | Zep/Graphiti | Engram | Basic Memory |
|---|---|---|---|---|---|---|
| Local-only | Yes | Part<sup>a</sup> | Self-host<sup>b</sup> | Self-host<sup>c</sup> | Yes<sup>d</sup> | Yes<sup>e</sup> |
| Zero infrastructure | Yes | Part<sup>f</sup> | No | No | Yes<sup>g</sup> | Yes |
| Hybrid retrieval | BM25F + vector + RRF | Semantic + BM25 + entity, additive<sup>h</sup> | Vector<sup>i</sup> | Graph + BM25 + vector, RRF / cross-encoder | FTS5 only | FTS5 + vector, score fusion |
| Governance (propose/review/apply) | Yes | No | No | No | No | No |
| Contradiction detection | Yes | Platform only<sup>j</sup> | LLM prompt only | Yes<sup>k</sup> | Agent-judged | No |
| Test functions | 12,624 test functions | - | - | - | - | - |
| LoCoMo benchmark (full 10-conv, Acc>=50)¹ | 73.8% | 66.9%² | 74.0% | - | - | - |
| MCP tools | 107 distinct<sup>l</sup> | Hosted<sup>m</sup> | Client only | 13 | 23 | 27 |
| Core dependencies | 0 | Many | Many | Many | 0<sup>n</sup> | Many |

<sub><sup>a</sup> Mem0, Local-only: OSS can run locally; defaults to a cloud LLM<br><sup>b</sup> Letta, Local-only: heavy runtime<br><sup>c</sup> Zep/Graphiti, Local-only: graph DB + LLM<br><sup>d</sup> Engram, Local-only: cloud optional<br><sup>e</sup> Basic Memory, Local-only: cloud optional<br><sup>f</sup> Mem0, Zero infrastructure: library mode; needs LLM + embedder<br><sup>g</sup> Engram, Zero infrastructure: single Go binary<br><sup>h</sup> Mem0, Hybrid retrieval: no RRF<br><sup>i</sup> Letta, Hybrid retrieval: BM25 + RRF only with Turbopuffer<br><sup>j</sup> Mem0, Contradiction detection: Dream<br><sup>k</sup> Zep/Graphiti, Contradiction detection: LLM-based<br><sup>l</sup> MIND-Mem, MCP tools: `mcp.tool` registrations; `recall` dispatcher shadows base `recall`<br><sup>m</sup> Mem0, MCP tools: Platform<br><sup>n</sup> Engram, Core dependencies: single binary</sub>

¹ Canonical MIND-Mem LoCoMo number — see
[`docs/benchmarks.md`](docs/benchmarks.md#locomo-benchmark--canonical-number)
for scope/evidence. On this apples-to-apples metric MIND-Mem is not the top
score of every system evaluated (see the Competitive Landscape table above);
its differentiator is being the only local-only, zero-dependency, governed
option.

² Mem0's own published LoCoMo-paper number. Mem0's separate 2026 managed
platform self-reports 91.6 on a different setup/judge — not apples-to-apples
with this row.

### At a Glance

| Tool | Strength | Trade-off |
| ---- | -------- | --------- |
| [**Mem0**](https://github.com/mem0ai/mem0) | Managed platform plus self-hostable OSS; semantic + BM25 + entity retrieval; user/agent/run scoping | Graph memory, Dream and hosted MCP are platform-only; OSS writes go straight in with no review |
| [**Supermemory**](https://supermemory.ai) | Knowledge graph with supersession, optional reranking, auto-ingestion from Drive/Notion | Engine is closed-source; cloud-first (local binary lacks connectors and MCP); only inferred memories are reviewed |
| [**claude-mem**](https://github.com/thedotmack/claude-mem) | Purpose-built for Claude Code, Chroma vectors + SQLite FTS5, MCP server | Needs Bun, uv, Chroma and a worker service; LLM observer; no contradiction detection or write governance |
| [**Letta**](https://www.letta.com) | Self-editing memory blocks, sleep-time agents, git-tracked memory (letta-code) | Full agent runtime (heavy), not just memory; hybrid search needs Turbopuffer |
| [**Zep**](https://www.getzep.com) | Temporal knowledge graph (Graphiti, Apache-2.0), bi-temporal model, BM25 + vector + RRF, MCP server | Needs a graph database and an LLM; Zep itself is the managed service |
| [**LangMem**](https://github.com/langchain-ai/langmem) | Native LangChain/LangGraph integration, background extraction | Tied to LangChain ecosystem; retrieval delegated to the LangGraph store |
| [**Cognee**](https://www.cognee.ai) | Graph + vector memory, local-first defaults, auto-improve loop, MCP server | LLM-driven graph build; heavy dependency tree; no lexical+vector fusion or write governance |
| [**Graphlit**](https://www.graphlit.com) | Multimodal ingestion, semantic search, managed platform | Cloud-only, managed service |
| [**ClawMem**](https://github.com/yoloshii/ClawMem) | Full ML pipeline (cross-encoder + QMD + beam search), 33 MCP tools | Bun/TypeScript with local GGUF models (~4 GB); no propose/review/apply pipeline |
| [**MemU**](https://github.com/NevaMind-AI/memU) | Markdown wiki/skills, host-agent-driven capture, LLM-free vector retrieval | Needs an embedding API key; no lexical or hybrid search; no MCP server |
| [**Engram**](https://github.com/Gentleman-Programming/engram) | Single Go binary, FTS5 search, agent-judged conflict/supersession relations, Git sync | Lexical-only recall; no vector search or propose/review/apply pipeline |
| [**Basic Memory**](https://github.com/basicmachines-co/basic-memory) | Markdown knowledge graph, FTS + vector hybrid search, valid-time filters | Python with many dependencies; no contradiction detection or propose/review/apply pipeline |
| **MIND-Mem** | Integrity + governance + zero core deps + hybrid search + MIND kernels + 107 MCP tools (incl. MIC/MAP, walkthrough, persona, pipeline-hash) + cross-model consensus audits (published for v3.11 and v3.12; see `audits/`) | Lexical recall by default (vector/CE optional) |

### Full Feature Matrix

Compared against every major memory solution for AI agents, checked against each project's public source or official docs in October 2026. Column codes and cell symbols are the same as in the [Feature Completeness Matrix](#feature-completeness-matrix).

**Recall**

| Feature | MM | M0 | SM | CM | Le | Zep | LM | Co | GL | CW | MU | En | BM |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Vector | **opt** | ✓ | ✓ | <sub>Chroma</sub> | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — | ↓<sup>1</sup> |
| Lexical | <sub>**BM25F**</sub> | <sub>BM25</sub> | <sub>FTS</sub> | <sub>FTS5</sub> | ◐ | <sub>BM25</sub> | — | ◐ | <sub>Keyword</sub> | <sub>BM25</sub> | — | <sub>FTS5</sub> | <sub>FTS5</sub> |
| Graph | <sub>**2-hop**</sub> | ◐ | ✓ | — | — | ✓ | — | ✓ | ✓ | ↓<sup>2</sup> | — | — | ✓ |
| Hybrid + RRF | **✓** | ◐<sup>3</sup> | ◐ | ◐ | ◐ | ✓ | — | ◐ | ◐ | **✓** | — | — | ◐<sup>4</sup> |
| Cross-encoder | **↓**<sup>5</sup> | opt | opt | — | ◐ | ✓ | — | — | ↓<sup>6</sup> | ↓<sup>7</sup> | — | — | opt |
| Intent routing | <sub>**9 types**</sub> | — | — | — | — | — | — | ◐ | — | ✓ | — | — | — |
| Query expansion | **↓**<sup>8</sup> | — | ↓<sup>9</sup> | — | — | — | ◐ | ◐ | ◐ | ↓<sup>10</sup> | — | — | — |

<sub><sup>1</sup> Basic Memory, Vector: FastEmbed<br><sup>2</sup> ClawMem, Graph: Beam + MPFP<br><sup>3</sup> Mem0, Hybrid + RRF: no RRF<br><sup>4</sup> Basic Memory, Hybrid + RRF: score fusion<br><sup>5</sup> MIND-Mem, Cross-encoder: MiniLM 80MB<br><sup>6</sup> Graphlit, Cross-encoder: Reranker<br><sup>7</sup> ClawMem, Cross-encoder: qwen3 0.6B<br><sup>8</sup> MIND-Mem, Query expansion: RM3 (zero-dep)<br><sup>9</sup> Supermemory, Query expansion: LLM rewrite<br><sup>10</sup> ClawMem, Query expansion: QMD 1.7B</sub>

**Persistence**

| Feature | MM | M0 | SM | CM | Le | Zep | LM | Co | GL | CW | MU | En | BM |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Structured | <sub>**Markdown**</sub> | <sub>JSON</sub> | <sub>Grph</sub> | <sub>SQL</sub> | <sub>Blk</sub> | <sub>Grph</sub> | <sub>KV</sub> | <sub>Grph</sub> | <sub>Grph</sub> | <sub>SQL</sub> | ↓<sup>1</sup> | <sub>SQL</sub> | ↓<sup>2</sup> |
| Entities | **✓** | ✓ | ✓ | — | ◐ | ✓ | ◐ | ✓ | ✓ | ✓ | — | — | ✓ |
| Temporal | **✓** | ◐ | ✓ | ◐ | ◐ | ✓ | ◐ | ✓ | ✓ | ✓ | ◐ | ◐ | ✓ |
| Supersede | **✓** | ◐ | ✓ | — | ◐ | ✓ | ◐ | ◐ | ◐ | ✓ | ◐ | ✓ | — |
| Append-only | **✓** | ◐ | — | — | — | ◐ | — | ◐ | — | — | — | ◐ | — |
| A-MEM metadata | **✓** | — | — | ◐ | — | — | — | ◐ | ◐ | ✓ | — | ◐ | — |

<sub><sup>1</sup> MemU, Structured: SQL + Markdown<br><sup>2</sup> Basic Memory, Structured: Markdown</sub>

**Integrity**

| Feature | MM | M0 | SM | CM | Le | Zep | LM | Co | GL | CW | MU | En | BM |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Contradictions | **✓** | ◐ | ✓ | — | ◐ | ✓ | ◐ | ◐ | — | ◐ | — | ↓<sup>1</sup> | — |
| Drift detection | **✓** | — | — | — | — | — | — | — | — | — | — | ↓<sup>2</sup> | ↓<sup>3</sup> |
| Validation | <sub>**Structural**</sub> | — | — | — | — | n/v | ◐ | ✓ | — | ◐ | — | ◐ | <sub>Schema</sub> |
| Impact graph | **✓** | — | — | — | — | — | — | — | — | ◐ | — | — | — |
| Coverage | **✓** | — | — | — | — | — | — | — | — | — | — | — | — |
| Multi-agent | <sub>**ACL-based**</sub> | ◐ | ✓ | ◐ | ✓ | ◐ | ◐ | ✓ | ✓ | ✓ | ◐ | ✓ | ◐ |
| Conflict res. | <sub>**Automatic**</sub> | ◐ | ◐ | — | — | ◐ | ◐ | ◐ | — | ◐ | — | ↓<sup>4</sup> | ◐ |
| WAL/crash | **✓** | — | n/v | ◐ | — | n/v | — | ◐ | n/v | ✓ | ◐ | ✓ | ↓<sup>5</sup> |
| Backup/restore | **✓** | ◐ | — | ◐ | ◐ | — | — | — | — | — | — | <sub>JSON</sub> | <sub>Cloud</sub> |
| Abstention | **✓** | — | — | — | — | — | — | — | — | ✓ | — | — | — |

<sub><sup>1</sup> Engram, Contradictions: Agent-judged<br><sup>2</sup> Engram, Drift detection: Staleness<br><sup>3</sup> Basic Memory, Drift detection: Schema drift<br><sup>4</sup> Engram, Conflict res.: Agent-judged<br><sup>5</sup> Basic Memory, WAL/crash: Index only</sub>

**Governance**

| Feature | MM | M0 | SM | CM | Le | Zep | LM | Co | GL | CW | MU | En | BM |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Auto-capture | <sub>**Propose**</sub> | <sub>Auto</sub> | <sub>Auto</sub> | <sub>Auto</sub> | <sub>Self</sub> | <sub>Ext</sub> | <sub>Auto</sub> | <sub>Auto</sub> | <sub>Ing</sub> | <sub>Auto</sub> | <sub>Auto</sub> | ↓<sup>1</sup> | ↓<sup>2</sup> |
| Proposal queue | **✓** | — | ◐ | — | — | — | — | ◐ | — | — | — | — | — |
| Rollback | **✓** | — | ◐ | — | ◐ | ◐ | — | — | — | ◐ | — | — | <sub>Cloud</sub> |
| Mode governance | <sub>**3 modes**</sub> | — | — | — | — | — | — | — | — | — | — | — | — |
| Audit trail | <sub>**Full**</sub> | ◐ | ◐ | — | ◐ | ◐ | — | opt | n/v | ◐ | ◐ | ◐ | — |

<sub><sup>1</sup> Engram, Auto-capture: Agent + passive<br><sup>2</sup> Basic Memory, Auto-capture: Plugin hooks</sub>

**Operations**

| Feature | MM | M0 | SM | CM | Le | Zep | LM | Co | GL | CW | MU | En | BM |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Local-only | **✓** | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ✓ | — | ✓ | ◐ | ✓ | ✓ |
| Zero core deps | **✓** | — | n/v | — | — | — | — | — | — | — | — | ✓ | — |
| No daemon | **✓** | ✓ | — | — | — | ◐ | ✓ | ✓ | N/A | ◐ | ✓ | ◐ | ✓ |
| GPU required | <sub>**No**</sub> | <sub>No</sub> | <sub>No</sub> | <sub>No</sub> | <sub>No</sub> | <sub>No</sub> | <sub>No</sub> | <sub>No</sub> | <sub>No</sub> | No<sup>1</sup> | <sub>No</sub> | <sub>No</sub> | <sub>No</sub> |
| Git-friendly | **✓** | — | — | — | ✓ | — | — | — | — | ◐ | ◐ | ✓ | ✓ |
| MCP server | <sub>**107 tools**</sub> | <sub>Hosted</sub> | ✓ | ✓ | — | ✓ | — | ✓ | ✓ | ✓ | — | <sub>23 tools</sub> | <sub>27 tools</sub> |
| MIND `.mind` files<sup>2</sup> | <sub>**26 files**</sub> | — | — | — | — | — | — | — | — | — | — | — | — |

<sub><sup>1</sup> ClawMem, GPU required: optional<br><sup>2</sup> 18 config + 8 source</sub>

### The Gap MIND-Mem Fills

Most tools above focus on **storage + retrieval**. Where contradiction handling exists, it is LLM-driven at write time; none of them answers all of these:

- "Do any of my decisions contradict each other?"
- "Which decisions are active but nobody references anymore?"
- "Did I make a decision in chat that was never formalized?"
- "What's the downstream impact if I change this decision?"
- "Is my memory state structurally valid right now?"

**MIND-Mem focuses on memory governance and integrity — the critical layer most memory systems ignore entirely.**

### Why Plain Files Outperform Fancy Retrieval

Letta's August 2025 analysis showed that a plain-file baseline (full conversations stored as files + agent filesystem tools) scored **74.0% on LoCoMo** with gpt-4o-mini — beating Mem0's top graph variant at 68.5%. Key reasons:

- **LLMs excel at tool-based retrieval.** Agents can iteratively query/refine file searches better than single-shot vector retrieval that might miss subtle connections.
- **Benchmarks reward recall + reasoning over storage sophistication.** Strong judge LLMs handle the rest once relevant chunks are loaded.
- **Overhead hurts.** Specialized pipelines introduce failure modes (bad embeddings, chunking errors, stale indexes) that simple file access avoids.
- **For text-heavy agentic use cases, "how well the agent manages context" > "how smart the retrieval index is."**

MIND-Mem's deterministic retrieval pipeline validates these findings: **73.8% on the full 10-conversation LoCoMo suite** (Acc≥50, canonical run below) with zero dependencies, no embeddings, and no vector database — **5.3pp above Mem0's top graph variant (68.5%)**. (68.5% is the Mem0 graph-variant figure quoted in Letta's analysis; the tables above use Mem0's own LoCoMo-paper number, 66.9%. Both are third-party self-reported and were not re-run by MIND-Mem.) The key insight: treating retrieval as a reasoning pipeline (wide candidate pool → deterministic rerank → context packing) matches embedding+vector systems without any ML infrastructure. Unlike plain-file baselines, MIND-Mem adds integrity checking, governance, and agent-agnostic shared memory via MCP.

---

## Companion Tools

External tools that solve an adjacent problem MIND-Mem deliberately does
not solve. They are listed as **complements, not competitors** — MIND-Mem
**does not depend on any of them**. License, scope, and
substrate-of-record concerns make co-existence the right pattern: each one
is a separate process you run alongside MIND-Mem, never a package
dependency.

| Tool | Solves | Relationship to MIND-Mem |
| ---- | ------ | ------------------------ |
| **MindLLM** (STARGA, commercial; available to customers, contact info@star.ga) | Deterministic, evidence-chained local inference behind OpenAI-compatible endpoints | Optional LLM backend — `"extraction": {"backend": "mindllm"}` in `mind-mem.json`, default endpoint `http://localhost:8080/v1` (override with `MIND_MEM_MINDLLM_URL`). `"backend": "auto"` probes it before vLLM. |
| [**GitNexus**](https://github.com/abhigyanpatwari/GitNexus) (third-party) | Code knowledge-graph indexer — parses repo structure (call graphs, dependencies, clusters) and serves architectural-awareness tools to coding agents over MCP | Sibling MCP server, no integration code. Its license is **PolyForm Noncommercial**, incompatible with MIND-Mem's Apache-2.0 as a programmatic dependency — so co-installation, never a dependency. |

### GitNexus answers a different question

| Question | Tool |
| -------- | ---- |
| "What does the code do at this point in time?" | **GitNexus** |
| "What did we decide, and why, over time?" | **MIND-Mem** |

Code structure *now* versus governed decision history — orthogonal, and
usefully so. Install both and each shows up in your MCP client's tool list
answering its own question domain, with no wiring between them:

```bash
# GitNexus — follow its own README for install + MCP registration
git clone https://github.com/abhigyanpatwari/GitNexus

# MIND-Mem (Apache-2.0, this repo)
pip install "mind-mem[all]"
mm install-all   # auto-wires MCP for Claude Code, Cursor, Windsurf, ...
```

Full positioning, the MindLLM quick start, and the license reasoning:
[`docs/companion-tools.md`](docs/companion-tools.md).

---

## Recall

### Default: BM25 Hybrid

```bash
python3 -m mind_mem.recall --query "authentication" --workspace .
python3 -m mind_mem.recall --query "auth" --json --limit 5 --workspace .
python3 -m mind_mem.recall --query "deadline" --active-only --workspace .
```

BM25F scoring (k1=1.2, b=0.75) with per-field weighting, bigram phrase matching, overlapping sentence chunking, and query-type-aware parameter tuning. Searches across all structured files.

**BM25F field weighting:** Terms in `Statement` fields score 3x higher than terms in `Context` (0.5x). This naturally prioritizes core content over auxiliary metadata.

**RM3 query expansion:** Pseudo-relevance feedback from top-k initial results. JM-smoothed language model extracts expansion terms, interpolated with the original query at configurable alpha. Falls back to static synonyms for adversarial queries.

**Adversarial abstention:** Deterministic pre-LLM confidence gate. Computes confidence from entity overlap, BM25 score, speaker coverage, evidence density, and negation asymmetry. Below threshold → forces abstention.

**Stemming:** "queries" matches "query", "deployed" matches "deployment". Simplified Porter stemmer with zero dependencies.

### Hybrid Search (BM25 + Vector + RRF)

```json
{
  "recall": {
    "backend": "hybrid",
    "vector_enabled": true,
    "rrf_k": 60,
    "bm25_weight": 1.0,
    "vector_weight": 1.0
  }
}
```

Thread-parallel BM25 and vector retrieval fused via RRF: `score(doc) = bm25_w / (k + bm25_rank) + vec_w / (k + vec_rank)`. Deduplicates by block ID. Falls back to BM25-only when vector backend is unavailable.

### Graph-Based (2-hop cross-reference boost)

```bash
python3 -m mind_mem.recall --query "database" --graph --workspace .
```

2-hop graph traversal: 1-hop neighbors get 0.3x score boost, 2-hop get 0.1x (tagged `[graph]`). Surfaces structurally connected blocks via `AlignsWith`, `Dependencies`, `Supersedes`, `Sources`, and ConstraintSignature scopes. Auto-enabled for multi-hop queries.

### Vector (pluggable)

```json
{
  "recall": {
    "backend": "vector",
    "vector_enabled": true,
    "vector_model": "all-MiniLM-L6-v2",
    "onnx_backend": true
  }
}
```

Supports ONNX inference (local, no server) or cloud embeddings. Falls back to BM25 automatically if unavailable.

---

## MIND Kernels

MIND-Mem ships **26 `.mind` files** under `mind/`: 18 INI-style pipeline
configuration files and eight MIND-language tensor sources. Configuration is
parsed by `load_kernel_config()` in `src/mind_mem/mind_ffi.py`; compiler sources
are migration prototypes. See the [file inventory](mind/README.md) and
[format distinction](docs/MIND_CONFIG_VS_MIND_LANG.md).

### Native migration status

The Python implementation remains available without `mindc`. An optional C
library implements the existing native scoring ABI. A MIND-emitted replacement
still needs compiler support, consumer ABI compatibility, numerical parity and
performance validation. Source verification alone does not establish those gates.
See [compiler development and native bridge status](mind/README.md) for the
current boundary. The 26 files ship in the wheel under
`<sys.prefix>/share/mind-mem/kernels/`; packaging them does not execute the sources.

### Compiler Source Index

| Source | Role |
| --- | --- |
| `abstention.mind`, `bm25.mind`, `category.mind`, `importance.mind` | MIND-language scoring prototypes |
| `prefetch.mind`, `ranking.mind`, `reranker.mind`, `rrf.mind` | MIND-language scoring prototypes |

The other 18 files are INI-style configurations and do not define compiler
functions. See the [source/configuration inventory](mind/README.md).

### Performance

<details>
<summary>Optional C scoring ABI vs pure Python — 9 core functions (200 iterations, <code>perf_counter</code>)</summary>

&nbsp;

| Function           |     N=100 |   N=1,000 |   N=5,000 |
| ------------------ | --------: | --------: | --------: |
| `rrf_fuse`         | **10.8x** | **69.0x** | **72.5x** |
| `bm25f_batch`      | **13.2x** | **113.8x** | **193.1x** |
| `negation_penalty` |  **3.3x** |  **7.0x** | **18.4x** |
| `date_proximity`   | **10.7x** | **15.3x** | **26.9x** |
| `category_boost`   |  **3.3x** | **19.8x** | **17.7x** |
| `importance_batch` | **22.3x** | **46.2x** | **48.6x** |
| `confidence_score` |  **0.9x** |  **0.8x** |  **0.9x** |
| `top_k_mask`       |  **3.1x** |  **8.1x** | **11.8x** |
| `weighted_rank`    |  **5.1x** | **26.6x** | **121.8x** |
| **Overall**        |           |           | **49.0x** |

> Previously reported local figures, retained for reference: **49x** aggregate speedup at N=5,000 and up to **193x** for an individual function. The harness sums kernel medians and excludes native array marshaling; this is not an end-to-end retrieval measurement. These C ABI figures lack a current source/artifact/hardware receipt here and do not establish MIND emission or numerical parity. The earlier 14-layer runtime protection claim is also unverified by the current C source.

</details>

### FFI Bridge

The compiled `.so` exposes a C99-compatible ABI. Python calls via `ctypes` through `src/mind_mem/mind_ffi.py`:

```python
from mind_ffi import get_kernel, is_available, is_protected

if is_available():
    kernel = get_kernel()
    scores = kernel.rrf_fuse_py(bm25_ranks, vec_ranks, k=60.0)
    print(f"Protected: {is_protected()}")  # True with the hardened build
```

### Without MIND

If `lib/libmindmem.so` is not present, MIND-Mem uses the supported pure-Python
implementations. The optional native C library is a performance path; no
MIND-emitted replacement or cross-backend parity claim follows from its absence.

---

## Auto-Capture

```
Session end
    ↓
capture.py scans daily log (or --scan-all for batch)
    ↓
Detects decision/task language (27 patterns, 3 confidence levels)
    ↓
Extracts structured metadata (subject, object, tags)
    ↓
Classifies confidence (high/medium/low → P1/P2/P3)
    ↓
Writes to intelligence/SIGNALS.md ONLY
    ↓
User reviews signals
    ↓
/apply promotes to DECISIONS.md or TASKS.md
```

**Batch scanning:** `python3 -m mind_mem.capture . --scan-all` scans the last 7 days of daily logs.

**Safety guarantee:** `capture.py` never writes to `decisions/` or `tasks/` directly. All signals must go through the apply engine.

---

## Multi-Agent Memory

### Namespace Setup

```bash
python3 -m mind_mem.namespaces workspace/ --init coder-1 reviewer-1
```

Creates `shared/` (visible to all) and `agents/coder-1/`, `agents/reviewer-1/` (private) directories with ACL config.

### Access Control

```json
{
  "default_policy": "read",
  "agents": {
    "coder-1": {"namespaces": ["shared", "agents/coder-1"], "write": ["agents/coder-1"], "read": ["shared"]},
    "reviewer-*": {"namespaces": ["shared"], "write": [], "read": ["shared"]},
    "*": {"namespaces": ["shared"], "write": [], "read": ["shared"]}
  }
}
```

### Shared Fact Ledger

High-confidence facts proposed to `shared/intelligence/LEDGER.md` become visible to all agents after review. Append-only with dedup and file locking.

### Conflict Resolution

```bash
python3 -m mind_mem.conflict_resolver workspace/ --analyze
python3 -m mind_mem.conflict_resolver workspace/ --propose
```

Graduated resolution: confidence priority > scope specificity > timestamp priority > manual fallback.

### Transcript Capture

```bash
python3 -m mind_mem.transcript_capture workspace/ --transcript path/to/session.jsonl
python3 -m mind_mem.transcript_capture workspace/ --scan-recent --days 3
```

Scans Claude Code JSONL transcripts for user corrections, convention discoveries, and architectural decisions. 16 patterns with confidence classification.

### Backup & Restore

```bash
python3 -m mind_mem.backup_restore backup workspace/ --output backup.tar.gz
python3 -m mind_mem.backup_restore export workspace/ --output export.jsonl
python3 -m mind_mem.backup_restore restore workspace/ --input backup.tar.gz
python3 -m mind_mem.backup_restore wal-replay workspace/
```

---

## Governance Modes

| Mode          | What it does                                             | When to use                                               |
| ------------- | -------------------------------------------------------- | --------------------------------------------------------- |
| `detect_only` | Scan + validate + report only                            | **Start here.** First week after install.                 |
| `propose`     | Report + generate fix proposals in `proposed/`           | After a clean observation week with zero critical issues. |
| `enforce`     | Bounded auto-supersede + self-healing within constraints | Production mode. Requires explicit opt-in.                |

**Recommended rollout:**
1. Install → run in `detect_only` for 7 days
2. Review scan logs → if clean, switch to `propose`
3. Triage proposals for 2-3 weeks → if confident, enable `enforce`

---

## Block Format

All structured data uses a simple, parseable markdown format:

```markdown
[D-20260213-001]
Date: 2026-02-13
Status: active
Statement: Use PostgreSQL for the user database
Tags: database, infrastructure
Rationale: Better JSON support than MySQL for our use case
ConstraintSignatures:
- id: CS-db-engine
  domain: infrastructure
  subject: database
  predicate: engine
  object: postgresql
  modality: must
  priority: 9
  scope: {projects: [PRJ-myapp]}
  evidence: Benchmarked JSON performance
  axis:
    key: database.engine
  relation: standalone
  enforcement: structural
```

Blocks are parsed by `block_parser.py` — a zero-dependency markdown parser that extracts `[ID]` headers and `Key: Value` fields into structured dicts.

---

## Configuration

All settings in `mind-mem.json` (created by `init_workspace.py`):

```json
{
  "version": "5.0.4",
  "auto_capture": true,
  "auto_recall": true,
  "governance_mode": "detect_only",
  "recall": {
    "backend": "bm25",
    "rrf_k": 60,
    "bm25_weight": 1.0,
    "vector_weight": 1.0,
    "vector_model": "all-MiniLM-L6-v2",
    "vector_enabled": false,
    "onnx_backend": false
  },
  "proposal_budget": {
    "per_run": 3,
    "per_day": 6,
    "backlog_limit": 30
  },
  "compaction": {
    "archive_days": 90,
    "snapshot_days": 30,
    "log_days": 180,
    "signal_days": 60
  }
}
```

| Key                             | Default              | Description                                                  |
| ------------------------------- | -------------------- | ------------------------------------------------------------ |
| `version`                       | installed package version | Config file version (written by `init_workspace`)       |
| `auto_capture`                  | `true`               | Run capture engine on session end (`hooks/session-end.sh`)   |
| `auto_recall`                   | `true`               | Show health/recall context on session start (`hooks/session-start.sh`) |
| `governance_mode`               | `"detect_only"`      | Governance mode (`detect_only`, `propose`, `enforce`)        |
| `recall.backend`                | `"bm25"`             | `"bm25"` (BM25), `"hybrid"` (BM25+Vector+RRF), or `"vector"` |
| `recall.rrf_k`                  | `60`                 | RRF fusion parameter k                                       |
| `recall.bm25_weight`            | `1.0`                | BM25 weight in RRF fusion                                    |
| `recall.vector_weight`          | `1.0`                | Vector weight in RRF fusion                                  |
| `recall.vector_model`           | `"all-MiniLM-L6-v2"` | Embedding model for vector search                            |
| `recall.vector_enabled`         | `false`              | Enable vector search backend                                 |
| `recall.onnx_backend`           | `false`              | Use ONNX for local embeddings (no server needed)             |
| `proposal_budget.per_run`       | `3`                  | Max proposals generated per scan                             |
| `proposal_budget.per_day`       | `6`                  | Max proposals per day                                        |
| `proposal_budget.backlog_limit` | `30`                 | Max pending proposals before pausing                         |
| `compaction.archive_days`       | `90`                 | Archive completed blocks older than N days                   |
| `compaction.snapshot_days`      | `30`                 | Remove apply snapshots older than N days                     |
| `compaction.log_days`           | `180`                | Archive daily logs older than N days                         |
| `compaction.signal_days`        | `60`                 | Remove resolved/rejected signals older than N days           |

---

## MCP Server

MIND-Mem ships with a [Model Context Protocol](https://modelcontextprotocol.io/) server that exposes memory as resources and tools to any MCP-compatible client.

> **Pair with [mind-nerve](https://pypi.org/project/mind-nerve/) for token-cheap routing.** When your agent host loads many skills/tools/MCP servers, mind-nerve sits in front and returns only the top-K relevant to each request — typically a 95%+ reduction in skill-listing tokens. Apache-2.0 wheel, `pip install mind-nerve`. See [`star-ga/mind-nerve`](https://github.com/star-ga/mind-nerve).

### Install

```bash
pipx install "mind-mem[mcp]"   # preferred — isolated venv with mind-mem-mcp on PATH
# or
pip install --user "mind-mem[mcp]"
```

The `[mcp]` extra pulls `fastmcp>=3.2.0` (the version line declared in
pyproject.toml) and registers the `mind-mem-mcp` console script.

### Automatic Setup (Recommended)

```bash
./install.sh --all
```

Configures all detected clients automatically. See [Quick Start](#quick-start).

### Manual Setup

For the JSON-configured MCP clients below (and Claude Desktop), add to the respective JSON config under `mcpServers`:

```json
{
  "mcpServers": {
    "mind-mem": {
      "command": "mind-mem-mcp",
      "args": [],
      "env": {"MIND_MEM_WORKSPACE": "/path/to/your/workspace"}
    }
  }
}
```

`mind-mem-mcp` is the console script registered by `pipx install
"MIND-Mem[mcp]"` (or `pip install --user "MIND-Mem[mcp]"`). If you're running
out of a source checkout instead, replace `"command": "mind-mem-mcp"` with
`"command": "python3", "args": ["/path/to/mind-mem/mcp_server.py"]`.

| Client | Config File |
| ------ | ----------- |
| **Claude Code** (`claude-code`) | `~/.claude.json`, written by `claude mcp add --scope user mind-mem -e MIND_MEM_WORKSPACE=/path/to/your/workspace -- mind-mem-mcp`; `mm install claude-code` itself writes hooks to `~/.claude/settings.json` |
| **Codex CLI** (`codex`) | `~/.codex/config.toml` (TOML) |
| **Grok Build CLI** (`grok-build`) | `~/.grok/config.toml` (TOML) |
| **Vibe (Mistral CLI)** (`vibe`) | `~/.vibe/config.toml` (TOML) |
| **OpenCode (1.x / 2.x)** (`opencode`) | `~/.config/opencode/opencode.json` (JSON) |
| **Gemini CLI** (`gemini`) | `~/.gemini/settings.json` (JSON) |
| **Cursor** (`cursor`) | `~/.cursor/mcp.json` (JSON) |
| **Windsurf** (`windsurf`) | `~/.codeium/windsurf/mcp_config.json` (JSON) |
| **aider** (`aider`) | no MCP writer — `mm install aider` writes `<workspace>/.aider.conf.yml` |
| **OpenClaw** (`openclaw`) | no MCP writer — `mm install openclaw` writes `~/.openclaw/openclaw.json` |
| **NanoClaw** (`nanoclaw`) | no MCP writer — `mm install nanoclaw` writes `~/.nanoclaw/nanoclaw.json` |
| **NemoClaw** (`nemoclaw`) | no MCP writer — `mm install nemoclaw` writes `~/.nemoclaw/nemoclaw.json` |
| **Continue** (`continue`) | `~/.continue/config.json` (JSON) |
| **Cline** (`cline`) | `~/.vscode-server/data/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json` (JSON) |
| **Roo Code** (`roo`) | `~/.vscode-server/data/User/globalStorage/rooveterinaryinc.roo-cline/settings/mcp_settings.json` (JSON) |
| **Zed** (`zed`) | `~/.config/zed/settings.json` (JSON) |
| **GitHub Copilot (workspace instructions)** (`copilot`) | no MCP writer — `mm install copilot` writes `<workspace>/.github/copilot-instructions.md` |
| **GitHub Copilot CLI** (`copilot-cli`) | `~/.copilot/mcp-config.json` (JSON) |
| **Cody** (`cody`) | no MCP writer — `mm install cody` writes `<workspace>/.cody/config.json` |
| **Qodo Gen** (`qodo`) | no MCP writer — `mm install qodo` writes `<workspace>/.codium/ai-rules.md` |

> **Cline and Roo Code:** the installer writes the VS Code Server (remote / WSL)
> settings tree, `~/.vscode-server/data/User/`. Desktop VS Code keeps its user
> settings elsewhere (for example `~/.config/Code/User/` on Linux), and those
> paths are not written yet; add the MCP entry there by hand for now.

Claude Desktop is not in the `mm install-all` registry. Configure it with
`./install.sh --claude-desktop` or by hand in
`~/.config/Claude/claude_desktop_config.json` (macOS:
`~/Library/Application Support/Claude/claude_desktop_config.json`).

For **Codex CLI** (TOML format), add to `~/.codex/config.toml`:

```toml
[mcp_servers.mind-mem]
command = "mind-mem-mcp"
args = []

[mcp_servers.mind-mem.env]
MIND_MEM_WORKSPACE = "/path/to/your/workspace"
```

For **Zed**, add to `~/.config/zed/settings.json` under `context_servers`:

```json
{
  "context_servers": {
    "mind-mem": {
      "command": {
        "path": "mind-mem-mcp",
        "args": [],
        "env": {"MIND_MEM_WORKSPACE": "/path/to/your/workspace"}
      }
    }
  }
}
```

### Direct (stdio / HTTP)

```bash
# stdio transport (default)
MIND_MEM_WORKSPACE=/path/to/workspace mind-mem-mcp

# HTTP transport (multi-client / remote) — requires MIND_MEM_TOKEN per v3.7.0 fail-closed contract
MIND_MEM_WORKSPACE=/path/to/workspace MIND_MEM_TOKEN=$(openssl rand -hex 32) \
  mind-mem-mcp --transport http --host 127.0.0.1 --port 8765
```

### Resources (read-only)

| URI                          | Description                                   |
| ---------------------------- | --------------------------------------------- |
| `mind-mem://decisions`       | Active decisions                              |
| `mind-mem://tasks`           | All tasks                                     |
| `mind-mem://entities/{type}` | Entities (projects, people, tools, incidents) |
| `mind-mem://signals`         | Signals that PASSED review + `withheld_count` |
| `mind-mem://contradictions`  | Detected contradictions                       |
| `mind-mem://health`          | Workspace health summary                      |
| `mind-mem://recall/{query}`  | BM25 recall search results                    |
| `mind-mem://ledger`          | Shared fact ledger (multi-agent)              |

### Core tools (23 of 107; full list in [`docs/api-reference.md`](docs/api-reference.md))

| Tool                  | Description                                                    |
| --------------------- | -------------------------------------------------------------- |
| `recall`              | Search memory with BM25 (query, limit, active_only)            |
| `propose_update`      | Propose a decision/task — writes to SIGNALS.md only            |
| `approve_apply`       | Apply a staged proposal (dry_run=True by default)              |
| `rollback_proposal`   | Rollback an applied proposal by receipt timestamp              |
| `scan`                | Run integrity scan (contradictions, drift, signals)            |
| `list_contradictions` | List contradictions with auto-resolution analysis              |
| `hybrid_search`       | Hybrid BM25+Vector search with RRF fusion                      |
| `find_similar`        | Find blocks similar to a given block                           |
| `intent_classify`     | Classify query intent (9 types with parameter recommendations) |
| `index_stats`         | Index statistics, MIND kernel availability, block counts       |
| `retrieval_diagnostics` | Pipeline rejection rates, intent histogram, hard negatives     |
| `reindex`             | Rebuild FTS5 index (optionally including vectors)              |
| `memory_evolution`    | View/trigger A-MEM metadata evolution for a block              |
| `list_mind_kernels`   | List available MIND kernel configurations                      |
| `get_mind_kernel`     | Read a specific MIND kernel configuration as JSON              |
| `category_summary`    | Category summaries relevant to a given topic                   |
| `prefetch`            | Pre-assemble context from recent conversation signals          |
| `delete_memory_item`  | Delete a memory block by ID (admin-scope)                      |
| `export_memory`       | Export workspace as JSONL (user-scope)                         |
| `calibration_feedback` | Submit quality feedback for a retrieved block (thumbs up/down) |
| `calibration_stats`   | View per-block and global calibration statistics               |
| `report_outcome`      | Report whether acting on recalled blocks actually worked       |
| `outcome_stats`       | Query recorded outcomes — which memories earned their keep     |

### Token Auth (HTTP)

```bash
MIND_MEM_TOKEN=your-secret mind-mem-mcp --transport http --port 8765
```

**As of v3.7.0, HTTP authentication fails CLOSED.** If neither
`MIND_MEM_TOKEN` nor `MIND_MEM_ADMIN_TOKEN` is set, the server refuses
to start. For local development you can opt back into the legacy
behaviour, but only on a loopback bind:

```bash
MIND_MEM_ALLOW_UNAUTHENTICATED_LOCALHOST=1 \
  mind-mem-mcp --transport http --host 127.0.0.1 --port 8765 \
               --allow-unauthenticated-localhost
```

The flag is a no-op if the bind host isn't `127.0.0.1` / `::1` /
`localhost` — the server still refuses to start. Production
deployments should always set a token.

The standalone `mm http-serve` adapter also enforces route privileges when
`MIND_MEM_ADMIN_TOKEN` is configured. Send either credential through
`X-MindMem-Token`: the user token can access user routes, while the admin
token also authenticates and may access admin routes. A user request to an
admin route receives HTTP 404. Setting the admin variable to an empty or
comma-only value keeps admin routes closed; leaving it unset preserves
legacy single-token full access. Authentication and route authorization
share one credential snapshot per request, refreshed for each request on
a persistent connection, so a token rotation takes effect on the next request.

### Safety Guarantees

- **`propose_update` never writes to DECISIONS.md or TASKS.md.** All proposals go to SIGNALS.md.
- **`approve_apply` defaults to dry_run=True.** Creates a snapshot before applying for rollback.
- **All resources are read-only.** No MCP client can mutate source of truth through resources.
- **Namespace-aware.** Multi-agent workspaces scope resources by agent ACL.

---

## Security

### Threat Model

| What we protect       | How                                                                  |
| --------------------- | -------------------------------------------------------------------- |
| Memory integrity      | Structural validator, ConstraintSignature validation                 |
| Accidental overwrites | Proposal-based mutations only (never direct writes)                  |
| Rollback safety       | Snapshot before every apply, atomic `os.replace()`                   |
| Symlink attacks       | Symlink detection in restore paths                                   |
| Path traversal        | All paths resolved via `os.path.realpath()`, workspace-relative only |

| What we do NOT protect against | Why                                                            |
| ------------------------------ | -------------------------------------------------------------- |
| Malicious local user           | Single-user CLI tool — filesystem access = data access         |
| Network attacks                | No network calls, no listening ports, no telemetry             |
| Encrypted storage              | Files are plaintext Markdown — use disk encryption if needed   |

### No Network Calls

MIND-Mem makes **zero network calls** from its core. No telemetry, no phoning home, no cloud dependencies. Optional features (vector embeddings, cross-encoder) may download models on first use.

---

## Requirements

- **Python 3.10+**
- **No external packages** — stdlib only for core functionality

### Optional Dependencies

| Package                      | Purpose                 | Install                               |
| ---------------------------- | ----------------------- | ------------------------------------- |
| `fastmcp`                    | MCP server              | `pip install mind-mem[mcp]`           |
| `onnxruntime` + `tokenizers` | Local vector embeddings | `pip install mind-mem[embeddings]`    |
| `sentence-transformers`      | Cross-encoder reranking | `pip install mind-mem[cross-encoder]` |
| `ollama`                     | LLM extraction (local)  | `pip install ollama`                  |

### mind-mem:4b — Purpose-Trained LLM

> **⏳ Coming: a retrained `mind-mem-4b`.** The weights published today are a full
> fine-tune of **Qwen3.5-4B**, trained against an earlier state of this repo. A full
> retrain on a **newer base model** is planned, generated from the current tool
> surface — several upcoming MIND-Mem features depend on it, because the shipped
> weights predate the surfaces those features expose. Until it lands, the current
> model remains the recommended one and everything below applies to it. The
> throughput figures in this section were measured on the **current** weights and
> will be restated when the new model ships. Not released yet; no date promised.

For best LLM extraction quality, use **[mind-mem:4b](https://huggingface.co/star-ga/mind-mem-4b)** — a full fine-tune of Qwen3.5-4B on MIND-Mem's 8 extraction tasks (entity extraction, fact extraction, observation compression, contradiction detection, governance analysis, intent classification, axis-aware retrieval, LLM reranking). Empirical on RTX 3080 (Q4_K_M, 2.6GB VRAM): **104 tok/s generation, 1585 tok/s prefill** (single-workstation Ollama measurement; no committed artifact).

**Ollama (recommended):**
```bash
# Download the GGUF from HuggingFace
wget https://huggingface.co/star-ga/mind-mem-4b/resolve/main/mind-mem-4b-Q4_K_M.gguf

# Create Ollama model
cat > Modelfile << 'EOF'
FROM ./mind-mem-4b-Q4_K_M.gguf
SYSTEM "You are mind-mem, a governance-aware memory extraction assistant."
PARAMETER temperature 0.1
PARAMETER num_ctx 8192
PARAMETER num_predict 1024
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"
EOF
ollama create mind-mem:4b -f Modelfile
```

Then set in `mind-mem.json`:
```json
{
  "extraction": {
    "enabled": true,
    "model": "mind-mem:4b",
    "backend": "ollama"
  }
}
```

**Full fine-tune (transformers, no adapter):**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("star-ga/mind-mem-4b", device_map="auto", torch_dtype="bfloat16")
tokenizer = AutoTokenizer.from_pretrained("star-ga/mind-mem-4b")
```

| Resource | Link |
|----------|------|
| Model (GGUF + bf16 safetensors) | [star-ga/mind-mem-4b](https://huggingface.co/star-ga/mind-mem-4b) |
| Base model | Qwen/Qwen3.5-4B |
| Training | Full fine-tune on Runpod H200 SXM (141 GB HBM3e), v4.0.0 corpus plus r3/r4 addendums (r4 includes 8 KernelKind anchor examples), bf16, paged-AdamW-8bit, batch 2 × accum 16, max_length 2048, LR 1.5e-5 cosine + 3% warmup |
| Eval (current published v4.1.1 weights) | **133/133 = 100%** — 111 main probes plus 22 held-out paraphrases, as reported in the [HF model card](https://huggingface.co/star-ga/mind-mem-4b). |

Two held-out probes use documented inference-time anchors. This is the
published checkpoint's reported result, not a new independent evaluation or
coverage of the full runtime surface. The weights were trained on 83 MCP tools.
The current server exposes 107 MCP tools.

### Platform Support

| Platform               | Status      | Notes                                   |
| ---------------------- | ----------- | --------------------------------------- |
| Linux                  | Full        | Primary target                          |
| macOS                  | Full        | POSIX-compliant shell scripts           |
| Windows (WSL/Git Bash) | Full        | Use WSL2 or Git Bash for shell hooks    |
| Windows (native)       | Python only | Use `validate_py.py`; hooks require WSL |

---

## Troubleshooting

| Problem                                     | Solution                                                                                                          |
| ------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `validate.sh` says "No mind-mem.json found" | Run in a workspace, not the repo root. Run `init_workspace.py` first.                                             |
| `recall` returns no results                 | Workspace is empty. Add decisions/tasks first.                                                                    |
| `capture` says "no daily log"               | No `memory/YYYY-MM-DD.md` for today. Write something first.                                                       |
| `intel_scan` finds 0 contradictions         | Good — no conflicting decisions.                                                                                  |
| Tests fail on Windows                       | Use `validate_py.py` instead of `validate.sh`. Hooks require WSL.                                                 |
| MIND kernel not loading                     | Expected — no compiled kernel ships in the wheel. Of the 26 `mind/*.mind` files, 18 are INI-style config read at runtime and 8 are MIND-language tensor source that is inert until compiled; the optional native `libmindmem.so` is built from `lib/kernels.c`, with optional version reporting. The current C source has no version symbol, so its version compatibility is unknown; a reported version mismatch does not automatically refuse loading. Pure-Python scoring (in `mind_kernels.py`) is the authoritative path. See [`docs/MIND_CONFIG_VS_MIND_LANG.md`](docs/MIND_CONFIG_VS_MIND_LANG.md). |

### FAQ

**No results from recall?**
Check that the workspace path is correct and points to an initialized workspace
containing decisions, tasks, or entities. If the FTS5 index is stale or missing,
run the `reindex` MCP tool to rebuild it.

**MCP connection failed?**
Verify that `fastmcp` is installed (`pip install fastmcp`). Check the transport
configuration in your client's MCP config (stdio vs HTTP). Ensure the
`MIND_MEM_WORKSPACE` environment variable points to a valid workspace directory.

**MIND sources or native kernels not loading?**
The eight MIND-language files are migration prototypes and are not required by
the supported Python path. The optional native backend is the C implementation
in `lib/kernels.c`; it is not bundled in the wheel. See
[`mind/README.md`](mind/README.md) for the current source and ABI boundary.

**Index corrupt?**
Run the `reindex` MCP tool, or from the command line:
`python3 -m mind_mem.sqlite_index --rebuild --workspace /path/to/workspace`.
This drops and recreates the FTS5 index from all workspace files.

---

## Specification

For the formal grammar, invariant rules, state machine, and atomicity guarantees, see **[SPEC.md](SPEC.md)**.

---

## MIND language sources

Eight files in `mind/` contain MIND-language scoring prototypes; 18 additional
`.mind` files are INI-style runtime configuration. The prototypes have not
established a complete native backend, consumer ABI compatibility, numerical
parity, or performance parity. The existing optional native implementation is
the C library in `lib/kernels.c`, while the supported Python implementation
remains available without a compiler or shared library.

The MIND language compiler is at [github.com/star-ga/mind](https://github.com/star-ga/mind). The formal specification is at [github.com/star-ga/mind-spec](https://github.com/star-ga/mind-spec). The agent CLI being built on the same substrate is at [github.com/star-ga/mind](https://github.com/star-ga/mind) (RFC 0013, in development). Visit [mindlang.dev](https://mindlang.dev) to see the substrate that makes byte-identical replay possible.

The compiler and specification links above are the references for MIND-language
syntax and semantics. A readable source prototype or compiler verification does
not by itself establish native execution or byte-identical scoring output.

---

## Contributing

Contributions welcome. Please open an issue first to discuss what you'd like to change.

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

[Apache 2.0](LICENSE) — Copyright 2026 STARGA Inc and contributors.
