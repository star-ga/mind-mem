# ADR-001: Zero External Dependencies in Core

## Status
Accepted

## Context
mind-mem needs to work as a memory system for AI coding agents. These agents run in diverse environments (containers, VMs, CI pipelines, local machines) where installing dependencies may be restricted or fail.

## Decision
The core mind-mem package (scripts/) will have zero external dependencies. All functionality must be implementable using Python stdlib only.

Optional features (vector search, cross-encoder reranking, LLM integration) may use external packages but must be clearly marked as optional and gracefully degrade when unavailable.

## Consequences

### Positive
- Works everywhere Python 3.10+ is installed
- No dependency conflicts with host projects
- Fast installation (nothing to download)
- No supply chain attack surface in core

### Negative
- Must implement BM25, FTS5 integration, file locking manually
- Cannot use popular libraries (numpy, pandas) in core
- Vector search requires optional sentence-transformers install

## Alternatives Considered
- **Minimal dependencies (numpy only):** Rejected — numpy is 20MB+ and triggers compilation on some platforms
- **Full dependency stack:** Rejected — conflicts with agent host environments are common
