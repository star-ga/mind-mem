# Performance Tuning

Guide to the two things that actually decide mind-mem's recall latency: which
leg serves the query, and how much per-query work the serving leg does.

> **Every number below was measured**, on synthetic Markdown corpora of
> uniformly-shaped decision blocks, `nice -n 15` on a shared 12-core i7-5930K,
> Python 3.12, five queries per sample, median of the per-sample totals. They
> are not budgets, targets, or estimates. Reproduce them against your own
> corpus before quoting them — block shape moves them, and a machine under
> different load moves them further.
>
> This file previously carried a table of latency *guidelines* that was wrong
> by 3.7x–8.7x against the measurements below, and named three functions that
> do not exist in the package (`rebuild_fts_index`, `build_vector_index`, and
> a `--older-than` flag). Every symbol named here is checked against the
> source.

## The one decision that matters: scan or index

Recall has two lexical legs.

* **BM25F scan** (`mind_mem._recall_core`) parses the Markdown corpus and
  scores every block in memory. It is **O(corpus)** per query.
* **SQLite FTS5** (`mind_mem.sqlite_index.query_index`) queries a prebuilt
  index. It is O(matching rows) per query.

Measured, same corpus, same five queries, per query:

| Blocks | Corpus on disk | BM25F scan | FTS5 index | Scan / index |
|--------|----------------|------------|------------|--------------|
| 500    | 2.5 MB         | 185.7 ms   | 14.9 ms    | 12.5x        |
| 2,000  | 9.5 MB         | 705.7 ms   | 23.0 ms    | 30.6x        |
| 5,000  | 24 MB          | 1,741.6 ms | 36.1 ms    | 48.2x        |

The ratio grows with the corpus, because only one side of it does.

**A default (`markdown` backend) install takes the scan.** `mind-mem-init`
writes `recall.backend = "bm25"` and nothing in the package builds an FTS
index, so an untouched workspace pays the left-hand column on every query.

### Building the index

```bash
python3 -m mind_mem.sqlite_index build -w /path/to/workspace          # incremental
python3 -m mind_mem.sqlite_index build -w /path/to/workspace --full   # rebuild
python3 -m mind_mem.sqlite_index status -w /path/to/workspace         # what is actually indexed
```

The workspace is `-w` / `--workspace`, not a positional argument.

or from Python — `mind_mem.sqlite_index.build_index`, the real symbol:

```python
from mind_mem.sqlite_index import build_index
build_index(workspace_path, incremental=False)   # incremental=True by default
```

The MCP `reindex` tool does the same thing over the server.

### Building it on first recall (opt-in)

```json
{ "recall": { "auto_build_index": true } }
```

`mind_mem.sqlite_index.ensure_index` then builds the index the first time a
recall finds none, which is what this document has always promised and what
the package did not do.

**It is off by default, deliberately.** Building the index does not merely
make the existing path faster — it changes *which leg serves*, because both
the hybrid BM25 arm and the MCP `recall` tool dispatch on whether a usable
index exists. In-memory BM25F and FTS5 `bm25()` are different rankers, so
turning this on changes result ORDER on a default install. Latency wins are
free; ranking changes are not, and this one is gated on a paired
retrieval-quality scorecard rather than on an argument. Turn it on when you
have measured that the ranking change is one you want.

Note that `.mind-mem-index/recall.db` **existing is not evidence of an
index**: `CalibrationManager` shares that file and creates it on the recall
path with a `calibration_feedback` table and nothing else. Ask
`mind_mem.sqlite_index.index_status(workspace)` instead, which reports the
schema and the row counts.

## Per-query work on the serving leg

Two costs used to scale with the corpus regardless of which leg served. Both
were removed in 5.0.2; the numbers are here so a regression is visible.

### Staleness checking (`is_stale`)

`is_stale` runs on every indexed query. It compares each corpus file's size
and nanosecond mtime against the recorded state — and used to follow that
with a full SHA-256 of every file whose metadata matched, i.e. of the entire
corpus whenever nothing had changed.

| | 5,000 blocks / 24 MB |
|---|---|
| Full-hash check (pre-5.0.2 behaviour, still available as `is_stale(ws, verify_hash=True)`) | 11.4 ms |
| Metadata-only check (the default since 5.0.2) | 1.0 ms |

That is ~20% of an indexed query, removed, and the remaining cost is O(files)
rather than O(corpus bytes). The full hash is still run unconditionally by
`build_index`, so an in-place edit that preserves both size and `mtime_ns` is
still picked up by the next build — the query path just no longer pays to
look for it on every request.

### Per-candidate weight lookups

The calibration weight and the A-MEM importance boost were each resolved with
one SQLite statement per candidate, inside the scoring loop. Both now resolve
in one batched statement per 900 candidates
(`CalibrationManager.get_block_weights`,
`BlockMetadataManager.get_importance_boosts`).

| Candidates | Per-candidate | Batched |
|------------|---------------|---------|
| 200 (typical FTS candidate list) | 2.2 ms | 0.08 ms |
| 5,000 (a scan over a mid-size corpus) | 58.3 ms | 1.9 ms |

End to end at 5,000 blocks, measured as a paired A/B on one workspace:

| Leg | Per-candidate | Batched | Delta |
|-----|---------------|---------|-------|
| FTS5 | 41.3 ms/query | 35.6 ms/query | −13.8% |
| BM25F scan | 1,924.6 ms/query | 1,741.6 ms/query | −9.5% |

The served list — ids **and** scores — is byte-identical across the two arms.
The batch and per-candidate forms read the same rows through the same
weighting function; this is a latency change and nothing else.

## Corpus cap and the truncation marker

The scan leg scores at most `MAX_BLOCKS_PER_QUERY` (50,000,
`mind_mem._recall_constants`) blocks. Past that it keeps an **arbitrary
prefix** — so on a workspace larger than the cap, the answer is computed from
part of the corpus.

When that fires, the result carries an in-band degradation marker:

```json
{"leg": "bm25", "reason": "corpus_truncated",
 "blocks_total": "63000", "blocks_scored": "50000"}
```

Read it as `results.degraded`. On the MCP `recall` tool's default `auto`
dispatch (and on an explicit `backend="hybrid"`), the marker is lifted into
the response envelope's `degraded` field with a matching warning, so a
truncated answer is not indistinguishable from a complete one.

Known gap, stated rather than implied: an explicit `backend="bm25"` request
to that tool takes a branch that does not read the marker off the result
object, so the envelope omits it there. The library-level marker is present
either way. Past the cap, use the FTS index — it has no such cap.

## Memory

Peak RSS added by one scan-leg recall (the whole corpus is parsed into
memory), measured with `resource.getrusage`:

| Blocks | Corpus | Added RSS | Process total |
|--------|--------|-----------|---------------|
| 500    | 2.5 MB | 11 MB     | 39 MB         |
| 2,000  | 9.5 MB | 25 MB     | 53 MB         |
| 5,000  | 24 MB  | 53 MB     | 81 MB         |

Roughly 2.2x the corpus size, and it is transient. The FTS leg does not load
the corpus, so it does not pay this at all — a second reason to index a large
workspace rather than tune the scan.

## BM25 parameters

Defined in `src/mind_mem/_recall_constants.py`. The names below are the real
ones; earlier revisions of this file named `K1`, `B`, `BOOST_TITLE` and
`BOOST_TAGS`, none of which exist.

| Symbol | Default | Description |
|--------|---------|-------------|
| `BM25_K1` | 1.2 | Term-frequency saturation |
| `BM25_B` | 0.75 | Document-length normalisation |
| `FIELD_WEIGHTS["Statement"]` | 3.0 | Statement field multiplier |
| `FIELD_WEIGHTS["Title"]` | 2.5 | Title field multiplier |
| `FIELD_WEIGHTS["Tags"]` | 0.8 | Tags field multiplier |

`FIELD_WEIGHTS` is a dict over every scored field, not two boost constants.

### When to tune

- **Raise `BM25_K1`** (1.5–2.0) when recall misses relevant blocks whose match
  is a single rare term.
- **Lower `BM25_B`** (0.3–0.5) when short blocks are being unfairly penalised.
- **Raise `FIELD_WEIGHTS["Title"]`** when block titles should dominate.

Change one at a time and measure retrieval quality, not latency: these move
ranking, and nothing here tells you whether the move was an improvement.

## Vector recall

Optional, opt-in, and it is a *second* leg fused with BM25 — not a
replacement. Enable in `mind-mem.json`:

```json
{ "recall": { "vector_enabled": true, "provider": "ollama",
              "ollama_embed_model": "mxbai-embed-large" } }
```

Rebuild the vector index with the real symbol, `recall_vector.rebuild_index`:

```python
from mind_mem.recall_vector import rebuild_index
rebuild_index(workspace_path)
```

Bound the leg so a cold or absent embedder cannot stall a request —
`recall.embed_timeout_seconds` and `recall.vector_deadline_seconds`. On
timeout the result degrades to BM25-only and says so in `results.degraded`.

## Cross-encoder reranking depth

The reranker is the most expensive optional stage in recall, and since 5.0.2
it runs over `recall.rerank_depth` fused candidates rather than over the
`limit` blocks already being served. That is what makes reranking able to
change recall@k at all — before, every block it could promote was in the
response already — and it is also the knob that decides what the stage costs.

Depth defaults to `min(50, 5 * limit)`, capped at `MAX_RERANK_CANDIDATES` and
then floored at `limit`, so at the common `limit: 10` the default depth is 50.

Measured on this box — reranker call only, median of 3, `OMP_NUM_THREADS=2`,
`nice -n 15`:

| `rerank_depth` | reranker latency |
|---------------:|-----------------:|
| 10             | 48 ms            |
| 25             | 111 ms           |
| 50             | 212 ms           |
| 100            | 433 ms           |
| 200            | 839 ms           |

Roughly linear in depth, ~4.2 ms per candidate. At the default depth of 50
that is about **212 ms** added to a `limit: 10` recall. These are one machine's
numbers on one model — treat the *shape* as the transferable part and re-measure
the constant on your own hardware before budgeting against it.

Set it explicitly when the default trade is wrong for you:

```json
{ "recall": { "rerank_depth": 25 } }
```

Two operating notes:

- Widening only happens when a reranker will actually run. With the
  cross-encoder off, the legs are not widened and the depth costs nothing.
- The value is **not** part of the recall cache key (nor is any other config
  value), so editing it inside the cache TTL can be served from an entry the
  old pipeline produced. Clear the cache, or wait out the TTL, before
  measuring a change.

## Compaction

Reduce corpus size by archiving old blocks. The real flags are per-category
day counts; there is no `--older-than`:

```bash
python3 -m mind_mem.compaction /path/to/workspace --dry-run
python3 -m mind_mem.compaction /path/to/workspace --archive-days 90 \
    --snapshot-days 30 --log-days 180 --signal-days 60
```

Run `--dry-run` first. On the scan leg, compaction is a direct latency lever
because latency is linear in corpus size; on the FTS leg it mostly is not.

## Metrics that exist

Counters on `mind_mem.observability.metrics`, verified against the source.
There is no `recall_latency_ms`, no `fts_cache_hits` / `fts_cache_misses`, and
no `index_staleness_s` — earlier revisions of this file invented all four.

| Counter | Meaning |
|---------|---------|
| `recall_queries` / `recall_results` | Queries served, hits returned |
| `index_queries` | Queries served by the FTS leg |
| `index_builds` / `index_blocks_indexed` | Index builds, blocks written |
| `index_stale_queries` | Queries served against a stale index |
| `recall_corpus_truncated` | Queries that hit `MAX_BLOCKS_PER_QUERY` |
| `recall_index_auto_built` | `auto_build_index` builds performed |
| `recall_cache_hits_total` / `recall_cache_misses_total` / `recall_cache_evictions_total` | MCP recall response cache |
| `hybrid_bm25_leg_index_empty` | BM25 arm structurally empty while the store has blocks |
| `recall_withheld_candidates` | Candidates withheld by admission |

`scan_duration_ms` is the one observed distribution.

### Structured logging

```python
import logging
logging.getLogger("mind-mem").setLevel(logging.DEBUG)
```

`index_stale`, `blocks_capped`, `auto_build_index`, `hybrid_bm25_leg_degraded`
and `query_complete` are the events worth watching on a latency
investigation.
