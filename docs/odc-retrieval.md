# Observer-Dependent Cognition in mind-mem

**Version:** v2.0.0a3 (released 2026-04-13)
**Modules:** `mind_mem.observation_axis`, `mind_mem.axis_recall`
**MCP tool:** `recall_with_axis` (user scope)

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

## Usage

```python
from mind_mem.axis_recall import recall_with_axis
from mind_mem.observation_axis import AxisWeights, ObservationAxis

# Default: lexical + semantic, matching pre-ODC behaviour
result = recall_with_axis(workspace, "JWT auth decision")

# Explicit: emphasise temporal + entity graph for time-anchored queries
weights = AxisWeights.uniform([ObservationAxis.TEMPORAL, ObservationAxis.ENTITY_GRAPH])
result = recall_with_axis(workspace, "who decided auth", weights=weights)

# Adversarial: probe the opposing basis for contradictions
result = recall_with_axis(workspace, "JWT is secure", adversarial=True)

for block in result["results"]:
    print(block["_id"], block["_axis_score"], block["observation"]["axes"])
```

From the MCP surface:

```
recall_with_axis(
  query="JWT auth",
  axes="lexical,semantic,temporal",
  weights="lexical=1.0,semantic=2.0,temporal=0.5",
  limit=10,
  adversarial=false,
  allow_rotation=true,
)
```

## Confidence and rotation

Per-axis confidence is `1 / (1 + (rank - 1))` (rank 1 → 1.0, rank 2 → 0.5, rank 3 → 0.33…). When the top result's best axis confidence falls below `DEFAULT_ROTATION_THRESHOLD = 0.35`, the pipeline rotates to up to two orthogonal axes and re-runs. Rotation is marked on the rotation-contributing results (`observation.rotated=True`); primary-only results keep `rotated=False`.

## Adversarial axis

The `ADVERSARIAL` axis rewrites the query as `NOT "<phrase>"` (double-quoting the phrase so the FTS5 parser negates the whole expression, not just the first token). Empty queries skip the axis outright. The adversarial pair map sends LEXICAL / SEMANTIC / TEMPORAL / ENTITY_GRAPH queries through CONTRADICTION as the opposing basis.

## References
- Source: [`src/mind_mem/observation_axis.py`](../src/mind_mem/observation_axis.py)
- Orchestrator: [`src/mind_mem/axis_recall.py`](../src/mind_mem/axis_recall.py)
- Tests: [`tests/test_observation_axis.py`](../tests/test_observation_axis.py), [`tests/test_axis_recall.py`](../tests/test_axis_recall.py)
