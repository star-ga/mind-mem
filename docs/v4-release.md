# v4.0.0 Release Notes

> **These surfaces were removed in 5.0.0 and RESTORED for 5.0.1.**
> The 5.0.0 reachability sweep deleted them because no product code imported
> them. That was overturned: "nothing imports it" is evidence about WIRING, not
> about worth, and the sweep removed 14,711 lines of working capability on that
> basis. Two of the modules were not even unreachable — `session_summarizer`
> had a shell caller in `hooks/session-end.sh` and a Python importer in
> `bootstrap_corpus`.
>
> They are back and being wired module by module, flag-gated and default-OFF,
> with flag-off behaviour byte-identical to 5.0.0. Import examples below are
> valid again. Where a restored module duplicates a live one, the resolution is
> substitution or merge — not deletion. See CHANGELOG 5.0.1.


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

This closes the unanimous blind spot identified in the cross-model
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

The codec is standalone. Nothing else in v4 consumes PQ codes yet —
in particular `hnsw_kind_index` does not, in either flag combination.

### `hnsw_kind_index.py`

Flag: `v4.hnsw_kind_index`

Kind-filtered kNN over a `block_kind_embeddings` table (`block_id`,
`kind`, packed float32 `payload`, `dim`), with a `kind` index so a query
touches only its own partition.

**There is no ANN backend.** Despite the module name, `knn_by_kind` runs
a brute-force cosine scan over the kind partition on every install — the
correct answer, at O(n). `sqlite-vec` is *not* used to serve queries;
`backend_status(workspace)` reports `backend: "brute_force"` always, and
reports `sqlite_vec_available` separately as a readiness signal for the
ANN work that is still to be built. The module docstring records what
that work needs (a per-dimension `vec0` table populated on the write
path, a sync watermark, and an equivalence gate against the brute-force
result) and why a partial version would be a faster wrong answer.

```python
from mind_mem.v4.hnsw_kind_index import (
    backend_status,
    knn_by_kind,
    register_block_embedding,
)

register_block_embedding(workspace, block_id, kind="entity", embedding=vec)
# [(block_id, cosine_distance)], ascending distance, at most k
results = knn_by_kind(workspace, kind="entity", query=vec, k=10)
backend_status(workspace)["backend"]  # -> "brute_force"
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

Hysteresis-gated overload detection for producer loops. A producer
reports how deep its backlog is and learns, from the same call, whether
to stop adding work:

```python
from mind_mem.v4.backpressure import PRODUCER_INBOX, batch_limit, report_depth

overloaded = report_depth(PRODUCER_INBOX, len(pending))   # True/False
limit = batch_limit(PRODUCER_INBOX, len(pending))         # int cap, or None
```

Entering the overloaded state needs `depth >= high_watermark` (default
1000); leaving it needs `depth <= low_watermark` (default 200). The gap
is the point: a queue oscillating around 600 stays in the state it is
in rather than flapping on every tick.

Each producer gets its **own** controller, keyed by name.
`set_depth` is last-writer-wins, so one shared controller would let the
inbox backlog overwrite the change-stream queue depth and both readings
would be fiction. `any_overloaded()` is the aggregate for a loop that
only wants to know whether the *process* is behind.

**It opens no door.** Backpressure writes nothing, reads no block and
touches no store; it paces loops that already own their governed write
path. It sheds RATE, never DATA — a throttled tick defers work to the
next tick and no input is dropped.

*(The pre-5.0.1 snippet here imported `mind_mem.backpressure`, called
`controller` as if it were an instance, invoked a `record_write()` that
does not exist, and divided the result by 1000. `recommended_pause`
returns SECONDS. Corrected above.)*

`current_pause()` peeks at the pause hint; `recommended_pause()`
returns it AND advances the exponential backoff, so observability code
must use the former or watching the system would change it.

**Wired producers (5.0.1)**

| Producer | Loop | Behaviour while overloaded |
|---|---|---|
| `inbox` | `InboxWatcher._loop` (`inbox.py`) | the scheduled tick takes `low_watermark` files and yields; the rest stay in the inbox for the next pass. `process_once` reports depth but is never capped — one-shot mode is an explicit request for the whole backlog. |
| `change_stream` | `ChangeStream.publish` | reports `queue_depth`; `ChangeStream.is_overloaded()` lets a producer ask before publishing. The bus never changes what it delivers, queues or sheds — backpressure is advice to producers, not policy the bus enforces. |
| `daemon` | `Daemon._tick` | defers the tick via `any_overloaded()` and does **not** stamp `last_run`; the next tick runs normally. Fires only when a reporting producer shares the process. |
| `webhook` | *not wired* | `PRODUCER_WEBHOOK` is reserved for the ingestion webhook drain, which does not exist yet. The drain author wires one `batch_limit(PRODUCER_WEBHOOK, depth)` call rather than inventing a fourth spelling. |

Per-producer watermarks override the workspace defaults — 50 files and
5000 events are not the same kind of "deep":

```json
{"v4": {"backpressure": {"enabled": true,
                         "high_watermark": 5000,
                         "low_watermark": 500,
                         "producers": {"inbox": {"high_watermark": 20,
                                                 "low_watermark": 5}}}}}
```

With the flag enabled, the `stream_status` MCP tool grows a
`backpressure` object carrying per-producer depth, watermarks and
overload state. The key is **absent** when the flag is off, so a client
can tell "nothing is measuring" from "measuring, and fine". No new tool
and no new ACL row: it is queue telemetry about a bus that tool already
reports on, and it carries counters only — never a block id, never
block content.

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

Wired in 5.0.1: with the flag on, `mind_mem.observability` installs
`StructuredLogFilter` on the handler its `StructuredLogger` owns, and the
MCP tool decorator (`mcp_tool_observe`) runs every tool call inside a fresh
`with_correlation_id` scope. Two concurrent recalls therefore tag every log
line they emit with their own id.

```python
from mind_mem.v4.logging_context import with_context, with_correlation_id

@with_correlation_id           # decorator: fresh uuid4 per call, inherited
def handle_request(...):       # by nested calls that already have one
    with with_context(user_id="u_456", op="recall"):
        log.info("retrieving blocks")
        # emitted JSON gains: "ctx": {"correlation_id": ..., "user_id": ..., "op": ...}
```

`StructuredLogFilter` injects the current context stack into every
`LogRecord` as a `ctx` dict attribute. Install it on the **handler**, never
on the root logger: `StructuredLogger` sets `propagate = False`, so a
root-logger install never sees a mind-mem record.

### `granularity_align.py`

Flag: `v4.granularity_align`

Named merge operation for the duplicate-memory pain: as a workspace grows,
two blocks end up making the same claim at different levels of abstraction
("use Q16.16 for determinism" / "all scoring must use fixed-point
arithmetic"). Neither is wrong, both pollute ranking.

Wired in 5.0.1: with the flag on, the `plan_consolidation` MCP tool carries
an extra `granularity_align` section listing the merge candidates
`find_merge_candidates` detects, each with the merged block `merge_blocks`
would produce.

```json
{
  "v4": {
    "granularity_align": {
      "enabled": true,
      "min_similarity": 0.75,
      "max_candidates": 20,
      "max_blocks": 400
    }
  }
}
```

`min_similarity` is the term-frequency cosine threshold (default 0.75);
`max_candidates` caps the returned pairs (0 means no cap, matching the
module's own contract); `max_blocks` caps how many top-level blocks are
compared, because candidate detection is O(n²) in that count. The flag is
read from the *workspace's* `mind-mem.json` first and the ambient config
second, the same resolution `maintenance_migrate` uses.

**Proposal-only.** The section is data: `applied` is always `false` and
`route` names the only path to the corpus — `propose_update`, then
`approve_apply`. `plan_consolidation` opens the index `mode=ro` and writes
nothing, so a merge reaches a block only after a human approves it. With the
flag off the section is absent and the tool's JSON is byte-identical to what
it was before the wiring existed.

### `multi_modal.py`

Flag: `v4.multi_modal`

The image / audio block schema shipped with no caller: `ImageBlock`,
`AudioBlock`, `thumbnail_hash` and `modal_token_cost` were tested in
isolation and invoked by nothing, while the two places they belonged sat a
few lines apart in the same package.

Wired in 5.0.1, in both places:

* `inbox._ingest_image` / `_ingest_audio` no longer raise. A drop is
  accepted when the operator supplies a **sidecar** — `board.png` beside
  `board.png.txt` (or `.md`, or a JSON object with `transcript` /
  `duration_seconds` / `speakers` / `dimensions`). Nothing reads pixels or
  samples: the media file is hashed for a stable `ThumbnailHash` and the
  sidecar text becomes the block. The watcher treats the pair as one drop
  and stages both files; an orphan sidecar stays an ordinary text drop.
* `pack_recall_budget` prices results by modality. `pack_to_budget` gained an
  optional `cost_fn`, defaulting to `None` — the char-count estimator that
  has always priced it — and the tool passes `multi_modal.pack_cost` only
  when the flag is on. An image costs its 85-token tile cost instead of the
  length of its caption; a text result costs exactly what it did before, so
  a text-only budget cannot move by a token.

**The door is an untrusted door.** The sidecar goes through the same
codepoint sanitizer as the text ingest and one shared writer opens
`admit_block` under `IngestTier.EXTERNAL_INGEST`, so a media drop lands
`Status: quarantined` and recall cannot see it until a governance proposal
releases it. The red-team suite carries it as Door 3, with the same
positive-control pairing as the text and agent-message doors, plus a proof
that the flag-off build refuses and writes nothing.

Two things the door will not do: it does not accept an `embedding` from a
sidecar (a vector is a scoring input, and a drop does not get to steer
retrieval), and it does not read image dimensions from the file — unstated
means unknown rather than parsed from a header for one format only.

Two seams had to be fixed for the wiring to be real rather than nominal, and
both were pre-existing:

* `block_parser` only reads field keys matching `[A-Z][A-Za-z]+:`, so the
  media blocks use `Type` / `ThumbnailHash` / `SourcePath` /
  `DurationSeconds`. (The text door's `type: INBOX_DOCUMENT` line has always
  been dropped on read-back. Left alone here: changing it would change bytes
  on a path this flag does not gate.)
* a result's `type` comes only from `_recall_detection.get_block_type`, an id
  prefix table — so `INBOX-IMG-` / `INBOX-AUD-` were added to it. Without
  that an image reached the packer as `"unknown"` and was priced by its
  caption: the cost function would have been wired and inert.

---

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

`v4.pq` and `v4.hnsw_kind_index` are independent: the kind index stores
raw float32 vectors and never reads PQ codes. Enable whichever you need.

---

## Full module list

| Module | Flag | Category |
|--------|------|----------|
| `tier_memory.py` | `v4.tier_memory` | Cognition |
| `cognitive_kernel.py` | `v4.cognitive_kernel` | Cognition |
| `surprise_retrieval.py` | `v4.surprise_retrieval` | Cognition |
| `llm_noise_profile.py` | `v4.llm_noise_profile` | Cognition |
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
