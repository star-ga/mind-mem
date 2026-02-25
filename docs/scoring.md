# Scoring System

mind-mem uses BM25F (BM25 with field weights) as its primary scoring algorithm, enhanced with several boosting mechanisms.

## BM25F Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| k1 | 1.2 | Term frequency saturation |
| b | 0.75 | Document length normalization |

## Field Weights

Different block fields receive different weights during scoring:

- **Statement**: Primary content field (highest weight)
- **Type**: Block type classification
- **Tags**: Associated tags and labels
- **Context**: Surrounding context
- **Source**: Source file information

## Boosting Mechanisms

### Co-retrieval Graph Boost
Blocks frequently returned together in past queries get linked. Querying one surfaces its neighbors via PageRank-like score propagation.

### Fact Card Indexing
Atomic fact cards (10-20 word sub-blocks) are indexed from Statement fields. Individual fact scores aggregate to parent blocks (small-to-big retrieval pattern).

### Entity Boost
Named entities mentioned in the query receive additional scoring weight when found in block fields.

### Priority Boost
Blocks marked as high-priority receive a configurable score multiplier.

### Status Boost
Active decisions receive higher boost than WIP or archived blocks.

### Bigram Boost
Consecutive query terms found together in block fields receive additional score.

### Date Score
Recent blocks receive a recency boost based on their creation or modification date.

## Knee Score Cutoff

Instead of a fixed top-K limit, mind-mem uses adaptive truncation at the steepest score drop (the "knee" of the score curve). This ensures results above the natural quality threshold are included while filtering noise.

## Hard Negative Mining

Blocks that score high on BM25 but low on cross-encoder reranking are tracked as hard negatives and receive a 30% demotion in future queries.

## MIND Kernel Overrides

Scoring parameters can be customized via `.mind/recall.mind` kernel configuration files. See [MIND Kernels](mcp-tool-examples.md) for details.
