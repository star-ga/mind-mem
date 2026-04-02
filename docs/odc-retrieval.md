# Observer-Dependent Cognition in mind-mem

**Version:** v2.0-alpha.2 target
**Date:** 2026-04-01

## Overview

mind-mem's retrieval pipeline already implements multi-axis observation through hybrid search (BM25 + vector + RRF fusion). ODC formalizes this: every recall explicitly declares its observation axes, results carry axis metadata, and the system can rotate axes for higher-confidence results.

## Current State (implicit axes)

| Retrieval Method | Implicit Axis |
|---|---|
| BM25F | Lexical (term frequency, field weights) |
| Vector search | Semantic (embedding similarity) |
| RRF fusion | Multi-axis collapse (rank reciprocal) |
| Recency decay | Temporal |
| Cross-encoder rerank | Contextual relevance |

## ODC Enhancement (explicit axes)

### observation_axis field
Added to RecallRequest — declares which axes are active:
- `lexical`, `semantic`, `temporal`, `entity-graph`, `contradiction`, `adversarial`

### Axis metadata on results
Every recall result tagged with:
- Which axes produced it
- Per-axis confidence scores
- Whether axis rotation was triggered

### Adversarial axis injection
Deliberately query from an opposing observation basis to surface contradictions.

## References
- [ODC Specification v1.0](specs/observer-dependent-cognition.md)
