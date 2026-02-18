# Architecture

## Overview

mind-mem is a hybrid MIND/Python memory system organized in three layers:

```
┌─────────────────────────────────────────────┐
│              MCP Server (14 tools)           │  mcp_server.py
├─────────────────────────────────────────────┤
│           Python Application Layer           │  scripts/*.py
│  ┌──────────┐ ┌──────────┐ ┌──────────────┐ │
│  │  Recall   │ │  Govern  │ │  Capture     │ │
│  │  Engine   │ │  Engine  │ │  Pipeline    │ │
│  └─────┬────┘ └──────────┘ └──────────────┘ │
│        │                                     │
│  ┌─────┴────┐ ┌──────────┐ ┌──────────────┐ │
│  │  Hybrid   │ │  Intent  │ │  Block       │ │
│  │  Backend  │ │  Router  │ │  Metadata    │ │
│  └──────────┘ └──────────┘ └──────────────┘ │
├─────────────────────────────────────────────┤
│         MIND Kernel Layer (optional)         │  mind/*.mind → lib/*.so
│  ┌──────┐ ┌─────┐ ┌────────┐ ┌───────────┐ │
│  │ BM25F│ │ RRF │ │Reranker│ │ Importance│ │
│  └──────┘ └─────┘ └────────┘ └───────────┘ │
├─────────────────────────────────────────────┤
│              Storage Layer                   │
│  ┌──────────┐ ┌──────────┐ ┌──────────────┐ │
│  │ Markdown  │ │ SQLite   │ │   WAL        │ │
│  │ Files     │ │ FTS5     │ │   Journal    │ │
│  └──────────┘ └──────────┘ └──────────────┘ │
└─────────────────────────────────────────────┘
```

## Module Inventory

### Core Modules

| Module               | Lines | Purpose                                                                         |
| -------------------- | ----- | ------------------------------------------------------------------------------- |
| `recall.py`          | ~2100 | BM25F + RM3 + graph scoring, stemming, query expansion, field boosts, reranking |
| `intel_scan.py`      | ~1250 | Integrity scanning, contradiction detection, drift analysis                     |
| `apply_engine.py`    | ~1200 | Proposal application, dry-run, rollback, audit trail                            |
| `sqlite_index.py`    | ~780  | FTS5 index + block_vectors + block_meta tables                                  |
| `evidence_packer.py` | ~250  | Structured evidence assembly for LLM context                                    |
| `block_parser.py`    | ~520  | Markdown block grammar parsing (schema v1/v2)                                   |

### Search & Ranking Modules

| Module                      | Lines | Purpose                                               |
| --------------------------- | ----- | ----------------------------------------------------- |
| `hybrid_recall.py`          | ~310  | HybridBackend: thread-parallel BM25+Vector+RRF        |
| `intent_router.py`          | ~160  | 9-type intent classification with parameter mapping   |
| `block_metadata.py`         | ~200  | A-MEM: access tracking, importance, keyword evolution |
| `cross_encoder_reranker.py` | ~80   | Optional ms-marco-MiniLM cross-encoder                |
| `mind_ffi.py`               | ~255  | ctypes FFI bridge to compiled MIND .so                |
| `abstention_classifier.py`  | ~275  | Adversarial abstention (5-feature confidence gate)    |
| `recall_vector.py`          | ~700  | Vector/embedding recall backends                      |

### Governance & Support Modules

| Module                  | Lines | Purpose                                    |
| ----------------------- | ----- | ------------------------------------------ |
| `capture.py`            | ~330  | Auto-capture from daily logs (26 patterns) |
| `compaction.py`         | ~340  | GC, archival, retention policies           |
| `conflict_resolver.py`  | ~310  | Graduated conflict resolution pipeline     |
| `backup_restore.py`     | ~410  | WAL, backup, restore, JSONL export         |
| `namespaces.py`         | ~380  | Multi-agent namespace + ACL management     |
| `filelock.py`           | ~150  | Cross-platform advisory file locking       |
| `observability.py`      | ~180  | Structured JSON logging + metrics          |
| `transcript_capture.py` | ~200  | JSONL transcript signal extraction         |

## Data Flow

### Recall Pipeline

```
Query
  ↓
Intent Router (9 types) → parameters
  ↓
RM3 Expansion (if enabled) → expanded query
  ↓
┌─────────────────────────────────────┐
│        Hybrid Backend (parallel)     │
│  ┌──────────┐    ┌────────────────┐ │
│  │ BM25F    │    │ Vector Search  │ │
│  │ (FTS5)   │    │ (ONNX/cloud)  │ │
│  └────┬─────┘    └───────┬────────┘ │
│       └──────┬───────────┘           │
│              ↓                       │
│         RRF Fusion (k=60)            │
└──────────────┬──────────────────────┘
               ↓
  Deterministic Reranking
  (negation + date proximity + category + recency)
               ↓
  Optional Cross-Encoder
               ↓
  A-MEM Importance Boost
               ↓
  Context Packing (adjacency + diversity + pronoun rescue)
               ↓
  Evidence Packing (speaker-attributed, category-ordered)
               ↓
  Results
```

### Governance Pipeline

```
Session End
  ↓
capture.py → SIGNALS.md (never source of truth)
  ↓
User reviews signals
  ↓
/apply → apply_engine.py
  ↓
Snapshot → Validate → Write → Audit → Receipt
  ↓
intel_scan.py
  ↓
Contradictions? → CONTRADICTIONS.md + resolution proposals
Drift? → DRIFT.md
Dead decisions? → flagged
Orphan tasks? → flagged
  ↓
Impact graph → IMPACT.md
Weekly briefing → BRIEFINGS.md
```

## Storage Model

### Markdown Files (Source of Truth)

All structured data lives in plain Markdown files using the block format:

```
[D-YYYYMMDD-NNN]
Date: ...
Status: active|superseded|deprecated
Statement: ...
Tags: ...
```

Benefits:
- Human-readable without tools
- Git-diffable
- No database dependency
- Portable to any system

### SQLite FTS5 (Search Index)

`sqlite_index.py` maintains a full-text search index. The index is ephemeral — it can always be rebuilt from the source Markdown.

Tables:
- `blocks` — FTS5 full-text index of all block content
- `block_vectors` — Optional embedding vectors (BLOB)
- `block_meta` — A-MEM metadata (access counts, importance, keywords)

### WAL Journal (Crash Safety)

Write operations go through a journal-based Write-Ahead Log. If the process crashes mid-write, `wal-replay` recovers to a consistent state.

## MIND Kernel Architecture

MIND kernels are compiled numerical hot paths that optionally accelerate scoring:

```
mind/bm25.mind     → mindc → lib/libmindmem.so → ctypes → Python
mind/rrf.mind      →
mind/reranker.mind →        (C99 ABI, flat float arrays)
mind/abstention.mind→
mind/ranking.mind  →
mind/importance.mind→
```

The FFI bridge (`mind_ffi.py`) loads the `.so` via ctypes and exposes Python-callable wrappers. If the `.so` is missing, all functions fall back to pure Python implementations that produce identical results.

## Multi-Agent Architecture

```
shared/                    ← All agents can read
├── decisions/
├── tasks/
├── entities/
└── intelligence/
    └── LEDGER.md         ← Cross-agent facts

agents/
├── coder-1/              ← Private to coder-1
│   ├── decisions/
│   ├── tasks/
│   └── memory/
└── reviewer-1/           ← Private to reviewer-1
    ├── decisions/
    └── memory/

mind-mem-acl.json         ← Access control policies
```

Each agent sees `shared/` + its own namespace. ACL uses fnmatch patterns for flexible policy definition.
