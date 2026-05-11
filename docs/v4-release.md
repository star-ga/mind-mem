# v4.0.0 Release Notes

Released 2026-05-10. Audience: existing mind-mem v3.x users.

All v4 surfaces are **flag-gated**. Nothing activates unless you add
the corresponding key to `mind-mem.json`. Your existing workspace,
schema, and MCP config require no changes to upgrade.

```bash
pip install --upgrade mind-mem
```

---

## What is new

v4 adds four layers on top of the v3.x substrate:

1. **Cognition** — tiered memory + cognitive kernel (pluggable retrieval
   strategies + surprise-weighted promotion)
2. **Knowledge graph** — multi-label block kinds, tag/TTL metadata,
   per-kind summaries, pluggable embedder
3. **Resilience / governance** — eviction policies, federation (VClock),
   self-editing, vector quantization (PQ), HNSW kind index, circuit
   breaker, backpressure controller, health probes
4. **Observability** — metrics primitives, structured logging context

---

## Cognition

### `tier_memory.py`

Flag: `v4.tier_memory`

Adds a `block_recall_tier` table with hot / warm / cold tiers.
Tier writes use compare-and-swap (CAS) via a `block_version` integer
column. Stale concurrent writes raise `StaleVersionError` instead of
silently overwriting.

```python
from mind_mem.tier_memory import write_tier, StaleVersionError

try:
    write_tier(workspace, block_id, tier="hot", expected_version=3)
except StaleVersionError:
    # another writer promoted this block first — re-read and retry
    ...
```

This closes the unanimous blind spot identified in the 4-LLM
architecture audit: read-after-write consistency during concurrent tier
promotions.

### `cognitive_kernel.py`

Flag: `v4.cognitive_kernel`

Exposes retrieval strategy as a first-class composable parameter via
`KernelKind`:

| Kernel | Behaviour |
|--------|-----------|
| `DEFAULT` | Standard hybrid BM25+vector+RRF (existing behaviour) |
| `SURPRISE_WEIGHTED` | Boosts blocks with high surprise score |
| `LINEAGE_FIRST` | Walks the typed-edge graph before scoring |
| `RECENT_FIRST` | Decays older blocks more aggressively |
| `CONTRADICTS_FIRST` | Surfaces contradicting blocks at the top |
| `GRAPH_WALK` | Pure graph traversal, no lexical scoring |

```python
from mind_mem.cognitive_kernel import mind_recall, KernelKind, register_kernel

# built-in kernel
results = mind_recall(workspace, "OAuth migration", kernel=KernelKind.LINEAGE_FIRST)

# custom kernel
register_kernel("my_kernel", my_retriever_fn)
results = mind_recall(workspace, "OAuth migration", kernel="my_kernel")
```

`is_kernel_registered(name)` returns `bool`. Default callers using
`recall(query)` are unaffected — `DEFAULT` kernel is used transparently.

### `surprise_retrieval.py`

Flag: `v4.surprise_retrieval` (activated automatically by `v4.cognitive_kernel`)

`compute_surprise(block, context_embedding)` returns a `float` in
`[0.0, 1.0]` representing semantic distance from the rolling recall
context. High surprise = unexpectedly relevant.

`FallbackPolicy` controls what happens when the embedder fails:

| Policy | Action |
|--------|--------|
| `NEUTRAL` | Treat surprise as `0.5` (default) |
| `PROMOTE` | Treat surprise as `1.0` — keep the block |
| `DEMOTE` | Treat surprise as `0.0` — deprioritise the block |
| `RAISE` | Re-raise `EmbeddingFailureError` |

```python
from mind_mem.surprise_retrieval import compute_surprise, FallbackPolicy

score = compute_surprise(block, ctx_embedding, fallback=FallbackPolicy.NEUTRAL)
```

---

## Knowledge graph

### `block_kinds.py`

Flag: `v4.block_kinds`

Adds a `block_kind_tags(block_id, kind, PRIMARY KEY(block_id, kind))`
junction table. Blocks can now carry multiple kinds simultaneously
(e.g. a block that is both `entity` and `code`).

```python
from mind_mem.block_kinds import add_kind_tag, get_kind_tags

add_kind_tag(workspace, block_id, "entity")
add_kind_tag(workspace, block_id, "code")
print(get_kind_tags(workspace, block_id))  # ["entity", "code"]
```

Additive — existing blocks have no tags until you add them. The
junction table is never populated by default behaviour.

### `block_metadata.py`

Flag: `v4.block_metadata`

ChromaDB-style key-value tag storage, per-block TTL, and Weaviate-style
schema validators.

```python
from mind_mem.block_metadata import (
    set_block_metadata, get_block_metadata, list_blocks_by_tag,
    register_schema_validator, validate_block, SchemaValidationResult,
)

set_block_metadata(workspace, block_id, {"project": "mind-mem", "env": "prod"})
blocks = list_blocks_by_tag(workspace, tag="project", value="mind-mem")

register_schema_validator("my_schema", my_validator_fn)
result: SchemaValidationResult = validate_block(workspace, block_id, schema="my_schema")
# result.valid, result.errors
```

TTL is set via the `ttl_seconds` key in metadata. Expired blocks are
not deleted automatically — they are flagged by the eviction planner.

### `kind_summaries.py`

Flag: `v4.kind_summaries`

Precomputes a per-kind global summary on write, following the GraphRAG
pattern. Useful for agents that need a high-level map before diving
into individual blocks.

```python
from mind_mem.kind_summaries import refresh_summary, get_summary

refresh_summary(workspace, kind="entity")
summary = get_summary(workspace, kind="entity")
```

Call `refresh_summary` after a batch of writes; it is not triggered
automatically.

### `embedding_pipeline.py`

Flag: `v4.embedding_pipeline`

Pluggable embedder interface. Default implementation uses hashed 3-grams
(zero external dependencies). Swap in any embedding function at runtime.

```python
from mind_mem.embedding_pipeline import register_embedder, embed

def my_embedder(text: str) -> list[float]:
    ...  # call your model

register_embedder("openai", my_embedder)
vec = embed("what did Alice say about OAuth?", backend="openai")
```

The `surprise_retrieval` module uses the active embedder automatically.

### `consolidation_worker.py`

Flag: `v4.consolidation`

`plan_consolidation(workspace)` is a pure function — it reads the
workspace and returns a `ConsolidationPlan` describing which blocks
should be merged, split, or promoted. It never writes. Apply the plan
explicitly after review.

```python
from mind_mem.consolidation_worker import plan_consolidation

plan = plan_consolidation(workspace)
for action in plan.actions:
    print(action)  # inspect before applying
plan.apply(workspace)
```

---

## Resilience / governance

### `eviction.py`

Flag: `v4.eviction`

Four eviction policies following the Redis CONFIG SET pattern.

| Policy | Evicts |
|--------|--------|
| `LRU` | Least recently accessed blocks |
| `LOW_SURPRISE` | Blocks with consistently low surprise scores |
| `AGE` | Oldest blocks by creation timestamp |
| `COMPOSITE` | Weighted combination of LRU + LOW_SURPRISE + AGE |

```python
from mind_mem.eviction import set_active_policy, active_policy, EvictionPlan

set_active_policy("COMPOSITE")
print(active_policy())  # "COMPOSITE"

plan = EvictionPlan.build(workspace, target_bytes=500_000_000)
print(plan.debug_plan())  # human-readable candidate list
plan.apply(workspace)
```

`is_policy_registered(name)` validates custom policies before setting.

### `federation.py`

Flag: `v4.federation`

Foundation for multi-host memory merges. Adds two tables:
- `block_tier_vclock(block_id, node_id, clock)` — vector clock per
  node for conflict detection
- `tier_conflict_log(block_id, node_a, node_b, detected_at)` — log of
  detected divergences

`MergeStrategy` enum (`LAST_WRITE_WINS`, `HIGHEST_SURPRISE_WINS`,
`MANUAL`) controls resolution behaviour. Automatic resolution is applied
only for `LAST_WRITE_WINS` and `HIGHEST_SURPRISE_WINS`. `MANUAL`
writes to `tier_conflict_log` and waits for operator action.

This module ships the data model and conflict detection; active sync
transport is out of scope for v4.0.0.

### `self_editing.py`

Flag: `v4.self_editing`

Adds a `block_edits` table. All edits are proposed, not directly
applied — they go through the same governance pipeline as
`propose_update → approve_apply`.

```python
from mind_mem.self_editing import propose_edit, approve_edit, reject_edit

edit_id = propose_edit(workspace, block_id, field="content", new_value="...")
approve_edit(workspace, edit_id)
# or
reject_edit(workspace, edit_id, reason="factually incorrect")
```

No direct mutation path exists. This enforces the same audit-trail
guarantee for self-edits that `propose_update` provides for new blocks.

### `pq.py`

Flag: `v4.pq`

Product Quantization codec. `M=32` sub-spaces, `K=256` centroids per
sub-space. 96× compression vs. raw `float32` vectors.

```python
from mind_mem.pq import PQCodec

codec = PQCodec.train(vectors)       # train on existing embeddings
codes = codec.encode(new_vectors)    # uint8 codes, 96x smaller
approx = codec.decode(codes)         # approximate reconstruction
codec.save(workspace / "pq.bin")
codec = PQCodec.load(workspace / "pq.bin")
```

Used automatically by `hnsw_kind_index` when `v4.pq` is enabled.

### `hnsw_kind_index.py`

Flag: `v4.hnsw_kind_index`

HNSW index on the `kind` column (`M=16`, `efc=200`). Accelerates
graph-walk kernel queries that filter by kind before scoring.

At startup, detects whether `sqlite-vec` is installed and uses its HNSW
implementation. Falls back to brute-force cosine scan when `sqlite-vec`
is absent. Behaviour is identical in both paths; only throughput differs.

```python
from mind_mem.hnsw_kind_index import build_kind_index, query_kind_index

build_kind_index(workspace)  # call once after bulk inserts
results = query_kind_index(workspace, kind="entity", query_vec=vec, top_k=10)
```

### `circuit_breaker.py`

Flag: `v4.circuit_breaker`

```python
from mind_mem.circuit_breaker import CircuitBreaker, CircuitState, circuit_breaker, default_breaker

# singleton for the default workspace
default_breaker.call(my_fn, *args)

# custom breaker
cb = CircuitBreaker(failure_threshold=5, recovery_timeout=30, half_open_probes=2)

@circuit_breaker(cb)
def call_external_embedder(text: str) -> list[float]:
    ...
```

States: `CLOSED` (normal), `OPEN` (rejecting calls), `HALF_OPEN`
(probing recovery). Transitions are thread-safe.

### `backpressure.py`

Flag: `v4.backpressure`

Hysteresis-gated overload detection. When the workspace write queue
exceeds the high-water mark, the controller recommends a pause before
the next write.

```python
from mind_mem.backpressure import controller

pause_ms = controller.recommended_pause()
if pause_ms > 0:
    time.sleep(pause_ms / 1000)
controller.record_write()
```

`current_pause` is the active hysteresis value. `recommended_pause` is
the caller-facing hint (may be 0 when load is below threshold).

### `health.py`

Flag: none — always available once installed.

```python
from mind_mem.health import health_check, register_health_probe

result = health_check(workspace)
# result.ok: bool
# result.probes: dict[str, ProbeResult]
# result.disabled_count: int
```

7 built-in probes: `db_connection`, `schema_version`, `wal_mode`,
`block_count`, `index_freshness`, `encryption_status`, `vector_backend`.

`register_health_probe(name, fn)` adds custom probes. `health_check`
is `BaseException`-safe — it never propagates an exception, even if a
probe crashes.

---

## Observability

### `observability.py`

Flag: `v4.observability`

```python
from mind_mem.observability import counter, gauge, histogram, timed, set_exporter

counter("recall.calls").inc()
gauge("workspace.block_count").set(n)
histogram("recall.latency_ms").observe(42.3)

@timed("propose_update.duration_ms")
def propose_update(...):
    ...

# plug in Prometheus, OTLP, or a custom exporter
set_exporter(my_exporter)
```

`MAX_CARDINALITY=10000` per metric. Labels that would exceed this limit
replace the offending label value with the sentinel `"__overflow__"` so
the metric keeps recording without unbounded memory growth.

### `logging_context.py`

Flag: `v4.logging_context`

Contextvar-backed key-value stack. Values propagate across `await`
boundaries automatically.

```python
from mind_mem.logging_context import with_context, with_correlation_id, StructuredLogFilter
import logging

logging.getLogger().addFilter(StructuredLogFilter())

with with_correlation_id("req-abc-123"):
    with with_context(user_id="u_456", op="recall"):
        logger.info("retrieving blocks")
        # log record includes: correlation_id, user_id, op
```

`StructuredLogFilter` injects the current context stack into every
`LogRecord` as a `context` dict field.

---

## Foundation

### `feature_flags.py`

All 35 v4 flags live here. An unknown flag name in `mind-mem.json` is
rejected at startup with a clear error.

```python
from mind_mem.feature_flags import is_enabled, require_enabled, FeatureDisabledError, flag_config

if is_enabled("v4.cognitive_kernel"):
    ...

require_enabled("v4.tier_memory")  # raises FeatureDisabledError if off

cfg = flag_config("v4.eviction")   # returns dict of flag-level config keys
```

---

## Eval and model notes

The v4.0.0 retrain clears the un-softened harness at **109/109 = 100%**.
The two probes intentionally softened in v3.12.1 (`qg.escape_hatch` and
`lin.cites`) are reverted and pass cleanly:

- `qg.escape_hatch` — required tokens restored to `["force", "strict"]`.
  Corpus contradictions about `force=True` vs `mode="off"` resolved;
  canonical answer is `force=True` on `validate_block`.
- `lin.cites` — required tokens restored to `["cites", "0.8"]`.
  `KIND_DECAY['cites']` is `0.8`; the v3.12.1 `0.4` confabulation is
  corrected via the per-kind reinforcement block in
  `train/build_corpus.py`.

14 new `V4_SURFACES` probes cover: tier promotion and CAS, kernel
dispatch, surprise score range, FallbackPolicy variants, multi-label
kind tags, schema validator registration, PQ encode/decode, HNSW
fallback, circuit breaker state transitions, backpressure hysteresis,
health probe registration, feature flag enforcement, observability
cardinality guard, and structured log context propagation.

---

## Migration

No migration required. All v4 tables are created on first use of the
corresponding feature flag. Existing blocks, schemas, and MCP tool names
are unchanged.

To opt in incrementally:

```json
{
  "features": {
    "v4.tier_memory": true,
    "v4.cognitive_kernel": true,
    "v4.block_kinds": true,
    "v4.block_metadata": true,
    "v4.eviction": true,
    "v4.circuit_breaker": true,
    "v4.observability": true,
    "v4.logging_context": true
  }
}
```

Enable `v4.pq` and `v4.hnsw_kind_index` together — PQ codes are consumed
by the HNSW index. Enabling one without the other is valid but suboptimal.

---

## Full module list

| Module | Flag | Category |
|--------|------|----------|
| `tier_memory.py` | `v4.tier_memory` | Cognition |
| `cognitive_kernel.py` | `v4.cognitive_kernel` | Cognition |
| `surprise_retrieval.py` | `v4.surprise_retrieval` | Cognition |
| `block_kinds.py` | `v4.block_kinds` | Knowledge graph |
| `block_metadata.py` | `v4.block_metadata` | Knowledge graph |
| `kind_summaries.py` | `v4.kind_summaries` | Knowledge graph |
| `embedding_pipeline.py` | `v4.embedding_pipeline` | Knowledge graph |
| `consolidation_worker.py` | `v4.consolidation` | Knowledge graph |
| `eviction.py` | `v4.eviction` | Resilience |
| `federation.py` | `v4.federation` | Resilience |
| `self_editing.py` | `v4.self_editing` | Resilience |
| `pq.py` | `v4.pq` | Resilience |
| `hnsw_kind_index.py` | `v4.hnsw_kind_index` | Resilience |
| `circuit_breaker.py` | `v4.circuit_breaker` | Resilience |
| `backpressure.py` | `v4.backpressure` | Resilience |
| `health.py` | (always on) | Resilience |
| `observability.py` | `v4.observability` | Observability |
| `logging_context.py` | `v4.logging_context` | Observability |
| `feature_flags.py` | (always on) | Foundation |
