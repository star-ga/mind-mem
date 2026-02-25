# ADR-002: BM25F as Primary Scoring Algorithm

## Status
Accepted

## Context
Memory retrieval needs a fast, accurate text search algorithm that works without external services or GPU acceleration.

## Decision
Use BM25F (BM25 with field boosting) as the primary scoring algorithm. Implementation uses SQLite FTS5 for tokenization and our own BM25F scoring logic with named constants.

## Consequences

### Positive
- CPU-only, no GPU required
- Deterministic scoring (same query always returns same results)
- Field boosting allows title/tag matches to rank higher
- Well-understood algorithm with decades of research

### Negative
- No semantic understanding (handled by optional vector search layer)
- Requires tuning K1/B parameters for optimal results
- Performance degrades with very large corpora (>100K blocks)
