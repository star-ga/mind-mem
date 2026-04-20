# Recall Cache Effectiveness Benchmark

v3.2.0 shipped a two-tier recall cache (in-process LRU L1 + optional
Redis L2). NIAH uses unique needle queries and so can't measure cache
value. This bench does: a realistic repeat-query workload through
`_recall_impl` (the cached path that REST and MCP tools take).

## Methodology

- **Seed:** 500 decision blocks (20 topics × 5 domains).
- **Queries:** 1,000 total, drawn from a pool of 40 unique queries.
  First 40 are guaranteed unique (pure misses); remaining 960 are
  sampled uniformly from the pool (≈96% hit rate, matching real
  agent workloads where queries repeat within a session).
- **Measurement:** `time.perf_counter` wraps each `_recall_impl`
  call. First encounter of a query → miss bucket; any subsequent
  encounter → hit bucket.
- **Configs:**
  - `no_cache` — `cache.enabled: false`
  - `lru_only` — in-process LRU only, no Redis
  - `redis` — L1 LRU + L2 Redis (`redis://localhost:6379/0`)
- **Isolation:** `reset_singleton()` + `redis-cli flushdb` between configs.
- **Platform:** i7-5930K, NVMe SSD, redis-server 7.2 localhost.

## Results

| Config      | Hit%  | p50 miss | p50 hit  | p95 hit  | Speedup  |
|-------------|-------|----------|----------|----------|----------|
| no_cache    | 96.0% | 6.08ms   | 6.09ms   | 7.36ms   | 1×       |
| lru_only    | 96.0% | 9.01ms   | 0.056ms  | 0.066ms  | **161×** |
| redis (L1+L2) | 96.0% | 9.59ms   | 0.056ms  | 0.073ms  | **171×** |

Raw JSON: [`cache_effectiveness_v3.2.1.json`](cache_effectiveness_v3.2.1.json)

## What the numbers mean

- **Hit% = 96.0% in every config** — that's the request shape, not
  the cache. `no_cache` processes every repeat as a fresh query
  (0% effective cache), so `p50 hit` ≈ `p50 miss`. `lru_only` and
  `redis` cut repeat latency to sub-100μs.
- **161-171× speedup on hits** — hybrid retrieval is ~6-10ms (BM25
  index lookup + score + serialize). Cache lookup is ~50μs (dict +
  timestamp check). That's two orders of magnitude headroom when
  agents re-ask questions.
- **Cache miss costs ~3ms extra vs no_cache** — the cache wrapper
  still has to check L1 (and L2 in the Redis config) before falling
  through to the inner implementation, then write the result back.
  Worth it on any workload with >30% repeat rate.
- **Redis adds ~0.6ms on miss vs LRU-only** — the L2 network round-
  trip. Hits stay fast because the `RecallCache` fills L1 from L2
  on the first miss; subsequent hits never touch Redis in the same
  process.

## When to use Redis

Redis **is not faster than pure LRU** in a single-process benchmark
— both tiers land in the same in-memory dict once warm. Redis is
load-bearing for **cross-worker sharing**:

- `uvicorn --workers 4` — four independent LRU caches with no
  coordination. Redis gives all four workers the same hit set.
- Horizontal scaling — multiple REST pods behind a load balancer
  see each other's cached results.
- Cache warm-up — a new worker hitting cold LRU can still see
  cached entries if Redis is shared.

On a single-worker deployment, `lru_only` is the right default.

## Reproducing

```bash
cd mind-mem
pip install "mind-mem[all]"

# Start redis (or point at an existing instance)
redis-server --daemonize yes

# Run the bench
python benchmarks/cache_effectiveness.py \
    --n 500 --queries 1000 --pool 40 \
    --output benchmarks/cache_effectiveness_v3.2.1.json
```

Parameters:
- `--n` — number of seed blocks (default 500)
- `--queries` — total queries (default 1000)
- `--pool` — unique queries in the sampling pool (default 40)
- `--redis-url` — Redis URL (default `redis://localhost:6379/0`)
- `--output` — write raw JSON results to this path

## Relationship to NIAH

| Benchmark | Measures | Why |
|-----------|----------|-----|
| NIAH | Retrieval correctness across matrix | Unique needle queries |
| Cache effectiveness | Cache latency + hit rate | Repeat queries through `_recall_impl` |

Both run in the v3.2.1 suite. NIAH reports `250/250 (100%) in 18:50`;
cache bench reports `161-171× speedup on 96% hits`. Neither
replaces the other.
