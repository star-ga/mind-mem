# Architecture

## Overview

mind-mem is a persistent, auditable, contradiction-safe memory system for coding agents. It provides BM25F-based retrieval with graph boost, fact indexing, and adaptive cutoff.

## System Architecture

```mermaid
graph TB
    subgraph MCP["MCP Server (57 tools)"]
        direction LR
        recall[recall]
        propose[propose_update]
        scan[scan]
        hybrid[hybrid_search]
        dream[dream_cycle]
        truth[compiled_truth]
        expand[expand_query]
        chunk[smart_chunk]
        dedup_tool[deduplicate]
        snapshot[snapshots]
        reindex[reindex]
        briefing[briefing]
    end

    subgraph Engine["Core Engines"]
        direction TB
        RE["Recall Engine<br/>BM25F + graph boost + knee cutoff"]
        HE["Hybrid Engine<br/>BM25 + vector + RRF fusion"]
        QE["Query Expansion<br/>synonym + specificity + temporal"]
        DD["4-Layer Dedup<br/>best-per-source → cosine → type cap → chunk cap"]
        SC["Smart Chunker<br/>semantic boundary splitting"]
        DC["Dream Cycle<br/>enrichment + repair + consolidation"]
        CT["Compiled Truth<br/>per-entity knowledge compilation"]
    end

    subgraph Scoring["Scoring & Reranking"]
        BM["BM25F Scoring<br/>field weights + stemming"]
        GR["Graph Boost<br/>cross-reference scoring"]
        RR["Reranker<br/>feature-based reranking"]
        XR["Cross-Encoder<br/>opt-in neural reranking"]
        MIND["MIND FFI Kernels<br/>compiled .mind scoring"]
    end

    subgraph Storage["Storage Layer"]
        BP["Block Parser<br/>markdown → structured blocks"]
        CM["ConnectionManager<br/>thread-safe SQLite pool"]
        BS["BlockStore<br/>decoupled block access"]
        VEC["sqlite-vec<br/>vector embeddings"]
    end

    subgraph FS["Workspace Filesystem"]
        decisions["decisions/"]
        tasks["tasks/"]
        entities["entities/"]
        memory["memory/"]
        intelligence["intelligence/"]
    end

    MCP --> Engine
    Engine --> Scoring
    Scoring --> Storage
    Storage --> FS
```

## Query Pipeline

```mermaid
flowchart LR
    Q["Query"] --> ID["Intent Detection<br/>WHAT/WHEN/WHO/HOW/WHY"]
    ID --> QX["Query Expansion<br/>multi-query + RM3"]
    QX --> BM["BM25F Scoring<br/>field weights"]
    BM --> GB["Graph Boost<br/>+ entity boost"]
    GB --> RR["Reranking"]
    RR --> DD["4-Layer Dedup"]
    DD --> KC["Knee Cutoff"]
    KC --> CP["Context Pack"]
    CP --> R["Results"]

    style Q fill:#2d5a27,stroke:#4a8c3f,color:#fff
    style R fill:#2d5a27,stroke:#4a8c3f,color:#fff
```

## Dream Cycle (Nightly Enrichment)

```mermaid
flowchart TB
    trigger["Heartbeat Trigger<br/>(after 23:00)"] --> scan_phase["Scan Phase"]

    subgraph scan_phase["Phase 1: Scan"]
        orphans["Find orphan entities"]
        broken["Detect broken citations"]
        stale["Flag stale blocks"]
        missing["Discover missing cross-refs"]
    end

    scan_phase --> repair["Phase 2: Repair"]

    subgraph repair["Phase 2: Repair"]
        fix_cite["Fix citations"]
        link_entities["Link entities"]
        merge_dupes["Merge duplicates"]
    end

    repair --> consolidate["Phase 3: Consolidate"]

    subgraph consolidate["Phase 3: Consolidate"]
        promote["Promote to compiled truth"]
        compact["Compact redundant entries"]
        report["Generate dream report"]
    end

    consolidate --> log["memory/dream-cycle-*.md"]
```

## Compiled Truth Pipeline

```
┌─────────────────────────────────────────────┐
│                 MCP Server                   │
│              (mcp_server.py)                 │
├──────────┬──────────┬──────────┬────────────┤
│  recall  │ propose  │  scan    │  hybrid    │
│  engine  │ update   │  engine  │  search    │
├──────────┴──────────┴──────────┴────────────┤
│              Block Parser                    │
│           (block_parser.py)                  │
├─────────────────────────────────────────────┤
│              Workspace FS                    │
│    decisions/ tasks/ entities/ memory/       │
└─────────────────────────────────────────────┘
```

## Core Modules

### Recall Engine (`_recall_core.py`)
Main BM25F pipeline. Loads blocks from workspace, tokenizes query, scores candidates, applies boosts, reranks, and returns top-K results with adaptive knee cutoff.

### Block Parser (`block_parser.py`)
Parses markdown files into structured blocks. Each block has an ID, type, statement, and optional metadata fields.

### Tokenization (`_recall_tokenization.py`)
Handles text tokenization with stemming, stopword removal, and Unicode normalization.

### Query Detection (`_recall_detection.py`)
Classifies query intent (WHAT/WHEN/WHO/HOW/WHY), detects skeptical queries, extracts field tokens, and handles query decomposition.

### Scoring (`_recall_scoring.py`)
BM25F scoring with field weights, cross-reference graph building, date scoring, and weighted term frequency computation.

### Reranking (`_recall_reranking.py`)
Deterministic reranking of BM25 candidates using feature-based scoring.

### Context Packing (`_recall_context.py`)
Formats retrieved blocks into context strings for LLM consumption.

### Query Expansion (`_recall_expansion.py`)
Expands queries with synonyms, month name variants, and RM3 pseudo-relevance feedback.

### Temporal Filtering (`_recall_temporal.py`)
Resolves time references ("today", "last week") and applies temporal filters to results.

### MIND FFI (`mind_ffi.py`)
Interface to MIND scoring kernels for customizable BM25 parameter overrides.

## Data Flow

1. Query arrives via MCP tool call
2. Query type detected and expanded
3. Blocks loaded from workspace files
4. BM25F scoring applied with field weights
5. Graph boost, entity boost, and other boosters applied
6. Reranking refines candidate ordering
7. Knee cutoff determines final result count
8. Context packed and returned to caller

## Storage

All data is stored as plain markdown files in the workspace directory. No external database required (zero dependencies).
