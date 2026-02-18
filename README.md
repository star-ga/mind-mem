<p align="center">
  <h1 align="center">mind-mem</h1>
  <p align="center">
    <strong>Drop-in memory for Claude Code, OpenClaw, and any MCP-compatible agent.</strong>
  </p>
  <p align="center">
    Local-first &bull; Zero-infrastructure &bull; Governance-aware &bull; MIND-accelerated
  </p>
  <p align="center">
    <a href="https://github.com/star-ga/mind-mem/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/star-ga/mind-mem/ci.yml?branch=main&style=flat-square&label=CI" alt="CI"></a>
    <a href="https://github.com/star-ga/mind-mem/blob/main/LICENSE"><img src="https://img.shields.io/github/license/star-ga/mind-mem?style=flat-square&color=blue" alt="License"></a>
    <a href="https://github.com/star-ga/mind-mem/releases"><img src="https://img.shields.io/github/v/release/star-ga/mind-mem?style=flat-square&color=green" alt="Release"></a>
    <img src="https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square&logo=python&logoColor=white" alt="Python 3.10+">
    <img src="https://img.shields.io/badge/dependencies-zero-brightgreen?style=flat-square" alt="Zero Dependencies">
    <img src="https://img.shields.io/badge/MCP-compatible-purple?style=flat-square" alt="MCP Compatible">
    <img src="https://img.shields.io/badge/MIND-accelerated-orange?style=flat-square" alt="MIND Accelerated">
  </p>
</p>

---

Drop-in memory layer for AI coding agents — Claude Code, Claude Desktop, Cursor, Windsurf, OpenClaw, or any MCP-compatible client. Upgrades your agent from "chat history + notes" to a governed **Memory OS** with hybrid search, RRF fusion, intent routing, optional MIND kernels, structured persistence, contradiction detection, drift analysis, safe governance, and full audit trail.

> **If your agent runs for weeks, it will drift. mind-mem prevents silent drift.**

### Trust Signals

| Principle | What it means |
|---|---|
| **Deterministic** | Same input, same output. No ML in the core, no probabilistic mutations. |
| **Auditable** | Every apply logged with timestamp, receipt, and DIFF. Full traceability. |
| **Local-first** | All data stays on disk. No cloud calls, no telemetry, no phoning home. |
| **No vendor lock-in** | Plain Markdown files. Move to any system, any time. |
| **Zero magic** | Every check is a grep, every mutation is a file write. Read the source in 30 min. |
| **No silent mutation** | Nothing writes to source of truth without explicit `/apply`. Ever. |
| **Zero infrastructure** | No Redis, no Postgres, no vector DB, no GPU. Python 3.10+ is all you need. |

---

## Table of Contents

- [Why mind-mem](#why-mind-mem)
- [Features](#features)
- [Benchmark Results](#benchmark-results)
- [Quick Start](#quick-start)
- [Health Summary](#health-summary)
- [Commands](#commands)
- [Architecture](#architecture)
- [How It Compares](#how-it-compares)
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
- [Contributing](#contributing)
- [License](#license)

---

## Why mind-mem

Most memory plugins **store and retrieve**. That's table stakes.

mind-mem also **detects when your memory is wrong** — contradictions between decisions, drift from informal choices never formalized, dead decisions nobody references, orphan tasks pointing at nothing — and offers a safe path to fix it.

| Problem | Without mind-mem | With mind-mem |
|---|---|---|
| Contradicting decisions | Follows whichever seen last | Flags, links both, proposes fix |
| Informal chat decision | Lost after session ends | Auto-captured, proposed to formalize |
| Stale decision | Zombie confuses future sessions | Detected as dead, flagged |
| Orphan task reference | Silent breakage | Caught in integrity scan |
| Scattered recall quality | Single-mode search misses context | Hybrid BM25+Vector+RRF fusion finds it |
| Ambiguous query intent | One-size-fits-all retrieval | 9-type intent router optimizes parameters |

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

### MIND Kernels (Optional, Native Speed)
6 compiled MIND kernels for BM25F scoring, RRF fusion, reranking, abstention, ranking, and importance scoring. Compiles to native `.so` via the [MIND compiler](https://mindlang.dev). Pure Python fallback always available — no functionality is lost without compilation.

### BM25F Hybrid Recall
BM25F field-weighted scoring (k1=1.2, b=0.75) with per-field weighting (Statement: 3x, Title: 2.5x, Name: 2x, Summary: 1.5x), Porter stemming, bigram phrase matching (25% boost per hit), overlapping sentence chunking (3-sentence windows with 1-sentence overlap), domain-aware query expansion, and optional 2-hop graph-based cross-reference neighbor boosting. Zero dependencies. Fast and deterministic.

### Graph-Based Recall
2-hop cross-reference neighbor boosting — when a keyword match is found, blocks that reference or are referenced by the match get boosted (1-hop: 0.3x decay, 2-hop: 0.1x decay). Surfaces related decisions, tasks, and entities that share no keywords but are structurally connected. Auto-enabled for multi-hop queries.

### Vector Recall (optional)
Pluggable embedding backend — local ONNX (all-MiniLM-L6-v2, no server needed) or cloud (Pinecone). Falls back to BM25 when unavailable.

### Persistent Memory
Structured, validated, append-only decisions / tasks / entities / incidents with provenance and supersede chains. Plain Markdown files — readable by humans, parseable by machines.

### Immune System
Continuous integrity checking: contradictions, drift, dead decisions, orphan tasks, coverage scoring, regression detection. 74+ structural validation rules.

### Safe Governance
All changes flow through graduated modes: `detect_only` → `propose` → `enforce`. Apply engine with snapshot, receipt, DIFF, and automatic rollback on validation failure.

### Adversarial Abstention Classifier
Deterministic pre-LLM confidence gate for adversarial/verification queries. Computes confidence from entity overlap, BM25 score, speaker coverage, evidence density, and negation asymmetry. Below threshold → forces abstention without calling the LLM, preventing hallucinated answers to unanswerable questions.

### Auto-Capture with Structured Extraction
Session-end hook detects decision/task language (26 patterns with confidence classification), extracts structured metadata (subject, object, tags), and writes to `SIGNALS.md` only. Never touches source of truth directly. All signals go through `/apply`.

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

### MCP Server (14 tools, 8 resources)
Full [Model Context Protocol](https://modelcontextprotocol.io/) server with 14 tools and 8 read-only resources. Works with Claude Code, Claude Desktop, Cursor, Windsurf, and any MCP-compatible client. HTTP and stdio transports with optional bearer token auth.

### 74+ Structural Checks + 676 Unit Tests
`validate.sh` checks schemas, cross-references, ID formats, status values, supersede chains, ConstraintSignatures, and more. Backed by 676 pytest unit tests covering all core modules.

### Audit Trail
Every applied proposal logged with timestamp, receipt, and DIFF. Full traceability from signal → proposal → decision.

---

## Benchmark Results

mind-mem's recall engine evaluated on two standard long-term memory benchmarks. Zero dependencies, pure deterministic retrieval — no embeddings, no vector DB, no cloud calls.

### LoCoMo LLM-as-Judge

Same pipeline as Mem0 and Letta evaluations: retrieve context, generate answer with LLM, score against gold reference with judge LLM. Directly comparable methodology.

| Category | N | Acc (>=50) | Mean Score |
|---|--:|--:|--:|
| **Overall** | **1986** | **67.3%** | **61.4** |
| Open-domain | 841 | 86.6% | 78.3 |
| Temporal | 96 | 78.1% | 65.7 |
| Single-hop | 282 | 68.8% | 59.1 |
| Multi-hop | 321 | 55.5% | 48.4 |
| Adversarial | 446 | 36.3% | 39.5 |

> **Judge:** `gpt-4o-mini` (answerer + judge) | **N:** 1986 questions, 10 conversations | See [`benchmarks/REPORT.md`](benchmarks/REPORT.md) for full methodology and reproduction steps.

### Competitive Landscape

| System | Score | Approach |
|---|--:|---|
| Memobase | 75.8% | Specialized extraction |
| **Letta** | 74.0% | Files + agent tool use |
| **Mem0** | 68.5% | Graph + LLM extraction |
| **mind-mem** | **67.3%** | Deterministic BM25 + rule-based packing |

> mind-mem reaches **98%** of Mem0's score with pure deterministic retrieval — no embeddings, no vector DB, no cloud calls, no LLM in the retrieval loop. mind-mem's unique value is **governance** (contradiction detection, drift analysis, audit trails) and **agent-agnostic shared memory** via MCP — areas these benchmarks don't measure.

### LongMemEval (ICLR 2025, 470 questions)

| Category | N | R@1 | R@5 | R@10 | MRR |
|---|--:|--:|--:|--:|--:|
| **Overall** | **470** | **73.2** | **85.3** | **88.1** | **.784** |
| Multi-session | 121 | 83.5 | 95.9 | 95.9 | .885 |
| Temporal | 127 | 76.4 | 91.3 | 92.9 | .826 |
| Knowledge update | 72 | 80.6 | 88.9 | 91.7 | .844 |
| Single-session | 56 | 82.1 | 89.3 | 89.3 | .847 |

### Run Benchmarks Yourself

```bash
# Retrieval-only (R@K metrics)
python3 benchmarks/locomo_harness.py
python3 benchmarks/longmemeval_harness.py

# LLM-as-judge (accuracy metrics, requires API key)
python3 benchmarks/locomo_judge.py --dry-run
python3 benchmarks/locomo_judge.py --answerer-model gpt-4o-mini --output results.json

# Selective conversations
python3 benchmarks/locomo_harness.py --conv-ids 4,7,8
```

---

## Quick Start

Get from zero to validated workspace in under 3 minutes.

### 1. Clone

```bash
cd /path/to/your/project
git clone https://github.com/star-ga/mind-mem.git .mind-mem
```

### 2. Initialize workspace

```bash
python3 .mind-mem/scripts/init_workspace.py .
```

Creates 12 directories, 19 template files, and `mind-mem.json` config. **Never overwrites existing files.**

### 3. Validate

```bash
bash .mind-mem/scripts/validate.sh .
# or cross-platform:
python3 .mind-mem/scripts/validate_py.py .
```

Expected: `74 checks | 74 passed | 0 issues`.

### 4. First scan

```bash
python3 .mind-mem/scripts/intel_scan.py .
```

Expected: `0 critical | 0 warnings` on a fresh workspace.

### 5. Verify recall + capture

```bash
python3 .mind-mem/scripts/recall.py --query "test" --workspace .
# → No results found. (empty workspace — correct)

python3 .mind-mem/scripts/capture.py .
# → capture: no daily log for YYYY-MM-DD, nothing to scan (correct)
```

### 6. Add skills (optional)

```bash
cp -r .mind-mem/skills/* .claude/skills/ 2>/dev/null || true
```

Gives you `/scan`, `/apply`, and `/recall` slash commands in Claude Code.

### 7. Add hooks (optional)

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

Configure workspace path in `~/.openclaw/openclaw.json`:

```json
{
  "hooks": {
    "internal": {
      "entries": {
        "mind-mem": {
          "enabled": true,
          "env": { "MIND_MEM_WORKSPACE": "/path/to/your/workspace" }
        }
      }
    }
  }
}
```

You're live. Start in `detect_only` for one week, then move to `propose`.

### 8. Smoke Test (optional)

```bash
bash .mind-mem/scripts/smoke_test.sh
```

Creates a temp workspace, runs init → validate → scan → recall → capture → pytest, then cleans up. All 11 checks should pass.

---

## Health Summary

After setup, this is what a healthy workspace looks like:

```
$ python3 scripts/intel_scan.py .

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

| Command | What it does |
|---|---|
| `/scan` | Run integrity scan — contradictions, drift, dead decisions, impact graph, snapshot, briefing |
| `/apply` | Review and apply proposals from scan results (dry-run first, then apply) |
| `/recall <query>` | Search across all memory files with ranked results (add `--graph` for cross-reference boosting) |

---

## Architecture

```
your-workspace/
├── mcp_server.py            # MCP server (FastMCP, 14 tools, 8 resources)
├── mind-mem.json             # Config
├── MEMORY.md                # Protocol rules
│
├── mind/                    # MIND source files (.mind)
│   ├── bm25.mind           # BM25F scoring kernel
│   ├── rrf.mind            # Reciprocal Rank Fusion kernel
│   ├── reranker.mind        # Deterministic reranking
│   ├── abstention.mind      # Confidence gating
│   ├── ranking.mind         # Evidence ranking
│   └── importance.mind      # A-MEM importance scoring
│
├── lib/                     # Compiled MIND kernels (optional)
│   └── libmindmem.so       # mindc output — not required for operation
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
└── scripts/
    ├── mind_ffi.py          # MIND FFI bridge (ctypes)
    ├── hybrid_recall.py     # Hybrid BM25+Vector+RRF orchestrator
    ├── block_metadata.py    # A-MEM metadata evolution
    ├── cross_encoder_reranker.py  # Optional cross-encoder
    ├── intent_router.py     # 9-type intent classification
    ├── recall.py            # BM25F + RM3 + graph scoring engine
    ├── recall_vector.py     # Vector/embedding backends
    ├── sqlite_index.py      # FTS5 + vector + metadata index
    ├── abstention_classifier.py  # Adversarial abstention
    ├── evidence_packer.py   # Evidence assembly and ranking
    ├── intel_scan.py        # Integrity scanner
    ├── apply_engine.py      # Proposal apply engine
    ├── block_parser.py      # Markdown block parser (typed)
    ├── capture.py           # Auto-capture (26 patterns)
    ├── compaction.py        # Compaction/GC/archival
    ├── filelock.py          # Cross-platform advisory file locking
    ├── observability.py     # Structured JSON logging + metrics
    ├── namespaces.py        # Multi-agent namespace & ACL
    ├── conflict_resolver.py # Automated conflict resolution
    ├── backup_restore.py    # WAL + backup/restore + JSONL export
    ├── transcript_capture.py  # Transcript JSONL signal extraction
    ├── validate.sh          # Structural validator (74+ checks)
    └── validate_py.py       # Structural validator (Python, cross-platform)
```

---

## How It Compares

### Full Feature Matrix

Compared against every major memory solution for AI agents (as of 2026):

| | [Mem0](https://github.com/mem0ai/mem0) | [SMem](https://supermemory.ai) | [c-mem](https://github.com/thedotmack/claude-mem) | [Letta](https://www.letta.com) | [Zep](https://www.getzep.com) | [LMem](https://github.com/langchain-ai) | [Cognee](https://www.cognee.ai) | [Gphlt](https://www.graphlit.com) | [ClawMem](https://github.com/yoloshii/ClawMem) | **mind-mem** |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| **Recall** | | | | | | | | | | |
| Vector | Cloud | Cloud | Chroma | Yes | Yes | Yes | Yes | Yes | Yes | **Optional** |
| Lexical | Filter | — | — | — | — | — | — | — | BM25 | **BM25F** |
| Graph | Yes | — | — | — | Yes | — | Yes | Yes | Beam | **2-hop** |
| Hybrid + RRF | Part | — | — | — | Yes | — | Yes | Yes | **Yes** | **Yes** |
| Cross-encoder | — | — | — | — | — | — | — | — | qwen3 0.6B | **MiniLM 80MB** |
| Intent routing | — | — | — | — | — | — | — | — | Yes | **9 types** |
| Query expansion | — | — | — | — | — | — | — | — | QMD 1.7B | **RM3 (zero-dep)** |
| **Persistence** | | | | | | | | | | |
| Structured | JSON | JSON | SQL | Blk | Grph | KV | Grph | Grph | SQL | **Markdown** |
| Entities | Yes | Yes | — | Yes | Yes | Yes | Yes | Yes | — | **Yes** |
| Temporal | — | — | — | — | Yes | — | — | — | — | **Yes** |
| Supersede | — | — | — | Yes | Yes | — | — | — | — | **Yes** |
| Append-only | — | — | — | — | — | — | — | — | — | **Yes** |
| A-MEM metadata | — | — | — | — | — | — | — | — | Yes | **Yes** |
| **Integrity** | | | | | | | | | | |
| Contradictions | — | — | — | — | — | — | — | — | — | **Yes** |
| Drift detection | — | — | — | — | — | — | — | — | — | **Yes** |
| Validation | — | — | — | — | — | — | — | — | — | **74+ rules** |
| Impact graph | — | — | — | — | — | — | — | — | — | **Yes** |
| Coverage | — | — | — | — | — | — | — | — | — | **Yes** |
| Multi-agent | — | — | — | Yes | — | — | — | — | — | **ACL-based** |
| Conflict res. | — | — | — | — | — | — | — | — | — | **Automatic** |
| WAL/crash | — | — | — | — | — | — | — | — | — | **Yes** |
| Backup/restore | — | — | — | — | — | — | — | — | — | **Yes** |
| Abstention | — | — | — | — | — | — | — | — | — | **Yes** |
| **Governance** | | | | | | | | | | |
| Auto-capture | Auto | Auto | Auto | Self | Ext | Ext | Ext | Ing | Auto | **Propose** |
| Proposal queue | — | — | — | — | — | — | — | — | — | **Yes** |
| Rollback | — | — | — | — | — | — | — | — | — | **Yes** |
| Mode governance | — | — | — | — | — | — | — | — | — | **3 modes** |
| Audit trail | — | Part | — | — | — | — | — | — | — | **Full** |
| **Operations** | | | | | | | | | | |
| Local-only | — | — | Yes | — | — | — | — | — | Yes | **Yes** |
| Zero deps | — | — | — | — | — | — | — | — | — | **Yes** |
| No daemon | — | — | — | — | — | Yes | — | — | — | **Yes** |
| GPU required | — | — | — | — | — | — | — | — | **4.5GB** | **No** |
| Git-friendly | — | — | — | Part | — | — | — | — | — | **Yes** |
| MCP server | — | — | — | — | — | — | — | — | — | **14 tools** |
| MIND kernels | — | — | — | — | — | — | — | — | — | **6 kernels** |

### mind-mem vs ClawMem (Head-to-Head)

ClawMem requires 4.5GB VRAM across 3 llama-server instances (qwen3-reranker-0.6B + QMD 1.7B GGUF + embedding model). mind-mem closes every feature gap with CPU-friendly alternatives that require zero infrastructure:

| Capability | ClawMem | mind-mem | Advantage |
|---|---|---|---|
| Hybrid search | BM25 + vector + RRF | BM25F + vector + RRF | mind-mem: field-weighted BM25F |
| Cross-encoder | qwen3-reranker 0.6B (GPU) | ms-marco-MiniLM 80MB (CPU) | mind-mem: 7x smaller, no GPU |
| Query expansion | QMD 1.7B GGUF (GPU) | RM3 pseudo-relevance (zero-dep) | mind-mem: no model needed |
| Intent routing | Yes | 9 types with parameter mapping | mind-mem: more granular |
| A-MEM metadata | Yes | Yes (importance, keywords, co-occurrence) | Equivalent |
| Graph search | Multi-graph beam | 2-hop cross-reference | Different approach, both effective |
| Governance | None | Contradiction detection, drift, audit | **mind-mem only** |
| Infrastructure | 3x llama-server, 4.5GB VRAM | Python 3.10+ | **mind-mem: zero** |
| MIND kernels | None | 6 compiled kernels (optional) | **mind-mem only** |
| MCP server | None | 14 tools, 8 resources | **mind-mem only** |

### What Each Tool Does Best

| Tool | Strength | Trade-off |
|---|---|---|
| **Mem0** | Fast managed service, graph memory, multi-user scoping | Cloud-dependent, no integrity checking |
| **Supermemory** | Fastest retrieval (ms), auto-ingestion from Drive/Notion | Cloud-dependent, auto-writes without review |
| **claude-mem** | Purpose-built for Claude Code, ChromaDB vectors | Requires ChromaDB + Express worker, no integrity |
| **Letta** | Self-editing memory blocks, sleep-time compute, 74% LoCoMo | Full agent runtime (heavy), not just memory |
| **Zep** | Temporal knowledge graph, bi-temporal model, sub-second at scale | Cloud service, complex architecture |
| **LangMem** | Native LangChain/LangGraph integration | Tied to LangChain ecosystem |
| **Cognee** | Advanced chunking, web content bridging | Research-oriented, complex setup |
| **Graphlit** | Multimodal ingestion, semantic search, managed platform | Cloud-only, managed service |
| **ClawMem** | Full ML pipeline (cross-encoder + QMD + beam search) | 4.5GB VRAM, 3 GPU processes required |
| **mind-mem** | Integrity + governance + zero deps + hybrid search + MIND kernels + 14 MCP tools | Lexical recall by default (vector/CE optional) |

### The Gap mind-mem Fills

Every tool above does **storage + retrieval**. None of them answer:

- "Do any of my decisions contradict each other?"
- "Which decisions are active but nobody references anymore?"
- "Did I make a decision in chat that was never formalized?"
- "What's the downstream impact if I change this decision?"
- "Is my memory state structurally valid right now?"

**mind-mem focuses on memory governance and integrity — the critical layer most memory systems ignore entirely.**

---

## Recall

### Default: BM25 Hybrid

```bash
python3 scripts/recall.py --query "authentication" --workspace .
python3 scripts/recall.py --query "auth" --json --limit 5 --workspace .
python3 scripts/recall.py --query "deadline" --active-only --workspace .
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
python3 scripts/recall.py --query "database" --graph --workspace .
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

mind-mem includes 6 `.mind` kernel source files — numerical hot paths written in the [MIND programming language](https://mindlang.dev). The MIND kernel is **optional**. mind-mem works identically without it (pure Python fallback). With it, scoring runs at native speed with compile-time tensor shape verification.

### Compilation

Requires the MIND compiler (`mindc`). See [mindlang.dev](https://mindlang.dev) for installation.

```bash
# Compile all kernels to a single shared library
mindc mind/bm25.mind mind/rrf.mind mind/reranker.mind mind/abstention.mind \
      mind/ranking.mind mind/importance.mind \
      --emit=shared -o lib/libmindmem.so

# Or compile individually for testing
mindc mind/bm25.mind --emit=shared -o lib/libbm25.so
```

### Kernel Index

| File | Functions | Purpose |
|------|-----------|---------|
| `bm25.mind` | `bm25f_doc`, `bm25f_batch`, `apply_recency`, `apply_graph_boost` | BM25F scoring with field boosts |
| `rrf.mind` | `rrf_fuse`, `rrf_fuse_three` | Reciprocal Rank Fusion |
| `reranker.mind` | `date_proximity_score`, `category_boost`, `negation_penalty`, `rerank_deterministic` | Deterministic reranking |
| `abstention.mind` | `entity_overlap`, `confidence_score` | Confidence gating |
| `ranking.mind` | `weighted_rank`, `top_k_mask` | Evidence ranking |
| `importance.mind` | `importance_score` | A-MEM importance scoring |

### FFI Bridge

The compiled `.so` exposes a C99-compatible ABI. Python calls via `ctypes` through `scripts/mind_ffi.py`:

```python
from mind_ffi import get_kernel, is_available

if is_available():
    kernel = get_kernel()
    scores = kernel.rrf_fuse_py(bm25_ranks, vec_ranks, k=60.0)
```

### Without MIND

If `lib/libmindmem.so` is not present, mind-mem uses pure Python implementations. The Python fallback produces identical results (within f32 epsilon). No functionality is lost — MIND is a performance optimization, not a requirement.

---

## Auto-Capture

```
Session end
    ↓
capture.py scans daily log (or --scan-all for batch)
    ↓
Detects decision/task language (26 patterns, 3 confidence levels)
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

**Batch scanning:** `python3 scripts/capture.py . --scan-all` scans the last 7 days of daily logs.

**Safety guarantee:** `capture.py` never writes to `decisions/` or `tasks/` directly. All signals must go through the apply engine.

---

## Multi-Agent Memory

### Namespace Setup

```bash
python3 scripts/namespaces.py workspace/ --init coder-1 reviewer-1
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
python3 scripts/conflict_resolver.py workspace/ --analyze
python3 scripts/conflict_resolver.py workspace/ --propose
```

Graduated resolution: confidence priority > scope specificity > timestamp priority > manual fallback.

### Transcript Capture

```bash
python3 scripts/transcript_capture.py workspace/ --transcript path/to/session.jsonl
python3 scripts/transcript_capture.py workspace/ --scan-recent --days 3
```

Scans Claude Code JSONL transcripts for user corrections, convention discoveries, and architectural decisions. 16 patterns with confidence classification.

### Backup & Restore

```bash
python3 scripts/backup_restore.py backup workspace/ --output backup.tar.gz
python3 scripts/backup_restore.py export workspace/ --output export.jsonl
python3 scripts/backup_restore.py restore workspace/ --input backup.tar.gz
python3 scripts/backup_restore.py wal-replay workspace/
```

---

## Governance Modes

| Mode | What it does | When to use |
|---|---|---|
| `detect_only` | Scan + validate + report only | **Start here.** First week after install. |
| `propose` | Report + generate fix proposals in `proposed/` | After a clean observation week with zero critical issues. |
| `enforce` | Bounded auto-supersede + self-healing within constraints | Production mode. Requires explicit opt-in. |

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
  "version": "1.0.0",
  "workspace_path": ".",
  "auto_capture": true,
  "auto_recall": true,
  "governance_mode": "detect_only",
  "recall": {
    "backend": "scan",
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
  },
  "scan_schedule": "daily"
}
```

| Key | Default | Description |
|---|---|---|
| `version` | `"1.0.0"` | Config schema version |
| `auto_capture` | `true` | Run capture engine on session end |
| `auto_recall` | `true` | Show recall context on session start |
| `governance_mode` | `"detect_only"` | Governance mode (`detect_only`, `propose`, `enforce`) |
| `recall.backend` | `"scan"` | `"scan"` (BM25), `"hybrid"` (BM25+Vector+RRF), or `"vector"` |
| `recall.rrf_k` | `60` | RRF fusion parameter k |
| `recall.bm25_weight` | `1.0` | BM25 weight in RRF fusion |
| `recall.vector_weight` | `1.0` | Vector weight in RRF fusion |
| `recall.vector_model` | `"all-MiniLM-L6-v2"` | Embedding model for vector search |
| `recall.vector_enabled` | `false` | Enable vector search backend |
| `recall.onnx_backend` | `false` | Use ONNX for local embeddings (no server needed) |
| `proposal_budget.per_run` | `3` | Max proposals generated per scan |
| `proposal_budget.per_day` | `6` | Max proposals per day |
| `proposal_budget.backlog_limit` | `30` | Max pending proposals before pausing |
| `compaction.archive_days` | `90` | Archive completed blocks older than N days |
| `compaction.snapshot_days` | `30` | Remove apply snapshots older than N days |
| `compaction.log_days` | `180` | Archive daily logs older than N days |
| `compaction.signal_days` | `60` | Remove resolved/rejected signals older than N days |
| `scan_schedule` | `"daily"` | `"daily"` or `"manual"` |

---

## MCP Server

mind-mem ships with a [Model Context Protocol](https://modelcontextprotocol.io/) server that exposes memory as resources and tools to any MCP-compatible client.

### Install

```bash
pip install fastmcp
```

### Claude Code

Add to `~/.claude/mcp.json` (global) or `.mcp.json` (per-project):

```json
{
  "mcpServers": {
    "mind-mem": {
      "command": "python3",
      "args": ["/path/to/mind-mem/mcp_server.py"],
      "env": {"MIND_MEM_WORKSPACE": "/path/to/your/workspace"}
    }
  }
}
```

### Claude Desktop

Add to `~/.claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "mind-mem": {
      "command": "python3",
      "args": ["/path/to/mind-mem/mcp_server.py"],
      "env": {"MIND_MEM_WORKSPACE": "/path/to/your/workspace"}
    }
  }
}
```

### Cursor / Windsurf

Add to your MCP config (`.cursor/mcp.json` or equivalent):

```json
{
  "mcpServers": {
    "mind-mem": {
      "command": "python3",
      "args": ["/path/to/mind-mem/mcp_server.py"],
      "env": {"MIND_MEM_WORKSPACE": "."}
    }
  }
}
```

### Direct (stdio / HTTP)

```bash
# stdio transport (default)
MIND_MEM_WORKSPACE=/path/to/workspace python3 mcp_server.py

# HTTP transport (multi-client / remote)
MIND_MEM_WORKSPACE=/path/to/workspace python3 mcp_server.py --transport http --port 8765
```

### Resources (read-only)

| URI | Description |
|---|---|
| `mind-mem://decisions` | Active decisions |
| `mind-mem://tasks` | All tasks |
| `mind-mem://entities/{type}` | Entities (projects, people, tools, incidents) |
| `mind-mem://signals` | Auto-captured signals pending review |
| `mind-mem://contradictions` | Detected contradictions |
| `mind-mem://health` | Workspace health summary |
| `mind-mem://recall/{query}` | BM25 recall search results |
| `mind-mem://ledger` | Shared fact ledger (multi-agent) |

### Tools (14 total)

| Tool | Description |
|---|---|
| `recall` | Search memory with BM25 (query, limit, active_only) |
| `propose_update` | Propose a decision/task — writes to SIGNALS.md only |
| `approve_apply` | Apply a staged proposal (dry_run=True by default) |
| `rollback_proposal` | Rollback an applied proposal by receipt timestamp |
| `scan` | Run integrity scan (contradictions, drift, signals) |
| `list_contradictions` | List contradictions with auto-resolution analysis |
| `hybrid_search` | Hybrid BM25+Vector search with RRF fusion |
| `find_similar` | Find blocks similar to a given block |
| `intent_classify` | Classify query intent (9 types with parameter recommendations) |
| `index_stats` | Index statistics, MIND kernel availability, block counts |
| `reindex` | Rebuild FTS5 index (optionally including vectors) |
| `memory_evolution` | View/trigger A-MEM metadata evolution for a block |
| `list_mind_kernels` | List available MIND kernel configurations |
| `get_mind_kernel` | Read a specific MIND kernel configuration as JSON |

### Token Auth (HTTP)

```bash
MIND_MEM_TOKEN=your-secret python3 mcp_server.py --transport http --port 8765
```

### Safety Guarantees

- **`propose_update` never writes to DECISIONS.md or TASKS.md.** All proposals go to SIGNALS.md.
- **`approve_apply` defaults to dry_run=True.** Creates a snapshot before applying for rollback.
- **All resources are read-only.** No MCP client can mutate source of truth through resources.
- **Namespace-aware.** Multi-agent workspaces scope resources by agent ACL.

---

## Security

### Threat Model

| What we protect | How |
|---|---|
| Memory integrity | 74+ structural checks, ConstraintSignature validation |
| Accidental overwrites | Proposal-based mutations only (never direct writes) |
| Rollback safety | Snapshot before every apply, atomic `os.replace()` |
| Symlink attacks | Symlink detection in restore paths |
| Path traversal | All paths resolved via `os.path.realpath()`, workspace-relative only |

| What we do NOT protect against | Why |
|---|---|
| Malicious local user | Single-user CLI tool — filesystem access = data access |
| Network attacks | No network calls, no listening ports, no telemetry |
| Encrypted storage | Files are plaintext Markdown — use disk encryption if needed |

### No Network Calls

mind-mem makes **zero network calls** from its core. No telemetry, no phoning home, no cloud dependencies. Optional features (vector embeddings, cross-encoder) may download models on first use.

---

## Requirements

- **Python 3.10+**
- **No external packages** — stdlib only for core functionality

### Optional Dependencies

| Package | Purpose | Install |
|---|---|---|
| `fastmcp` | MCP server | `pip install mind-mem[mcp]` |
| `onnxruntime` + `tokenizers` | Local vector embeddings | `pip install mind-mem[embeddings]` |
| `sentence-transformers` | Cross-encoder reranking | `pip install mind-mem[cross-encoder]` |

### Platform Support

| Platform | Status | Notes |
|----------|--------|-------|
| Linux | Full | Primary target |
| macOS | Full | POSIX-compliant shell scripts |
| Windows (WSL/Git Bash) | Full | Use WSL2 or Git Bash for shell hooks |
| Windows (native) | Python only | Use `validate_py.py`; hooks require WSL |

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `validate.sh` says "No mind-mem.json found" | Run in a workspace, not the repo root. Run `init_workspace.py` first. |
| `recall` returns no results | Workspace is empty. Add decisions/tasks first. |
| `capture` says "no daily log" | No `memory/YYYY-MM-DD.md` for today. Write something first. |
| `intel_scan` finds 0 contradictions | Good — no conflicting decisions. |
| Tests fail on Windows | Use `validate_py.py` instead of `validate.sh`. Hooks require WSL. |
| MIND kernel not loading | Compile with `mindc mind/*.mind --emit=shared -o lib/libmindmem.so`. Or ignore — pure Python works identically. |

---

## Specification

For the formal grammar, invariant rules, state machine, and atomicity guarantees, see **[SPEC.md](SPEC.md)**.

---

## Contributing

Contributions welcome. Please open an issue first to discuss what you'd like to change.

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

[MIT](LICENSE) — Copyright 2026 STARGA Inc.
