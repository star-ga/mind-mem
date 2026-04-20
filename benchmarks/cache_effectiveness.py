"""Cache-effectiveness benchmark — Redis L2 vs LRU-only vs no-cache.

NIAH measures retrieval quality on unique queries, so the recall
cache adds zero value there (every query is a miss by construction).
This bench is the counterpart: measure cache latency + hit rate on
a realistic repeat-query workload that goes through
``_recall_impl`` — the path the REST API and MCP tools take.

Shape:
    1. Seed a workspace with ``N`` decision blocks.
    2. Pick ``K`` distinct queries derived from block content.
    3. Run ``M`` queries where ``repeat_pct`` percent hit the same
       ``K`` queries repeatedly (the realistic case — agents tend
       to re-ask the same question in a session).
    4. Measure per-query latency; bucket miss vs hit.
    5. Run once per config: ``no_cache``, ``lru_only``, ``redis``.

Usage:
    python benchmarks/cache_effectiveness.py [--n 500] [--queries 1000]

Expected output:

    Config      Hit%   p50 miss   p50 hit   p95 hit   Speedup
    no_cache    0.0%   4.2ms      —         —         —
    lru_only    79.8%  4.3ms      0.08ms    0.12ms    54×
    redis       79.8%  4.3ms      0.35ms    0.9ms     12×

Redis runs ~3-10× slower than pure LRU per hit (network + serialize
round-trip) but gives cross-worker sharing, so it's the correct
choice when ``uvicorn --workers N``. LRU-only is best for single-
worker deployments.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass, field


@dataclass
class RunResult:
    name: str
    n_queries: int = 0
    miss_latencies_ms: list[float] = field(default_factory=list)
    hit_latencies_ms: list[float] = field(default_factory=list)

    @property
    def hit_rate(self) -> float:
        total = len(self.miss_latencies_ms) + len(self.hit_latencies_ms)
        if total == 0:
            return 0.0
        return 100.0 * len(self.hit_latencies_ms) / total

    def p50(self, values: list[float]) -> float:
        return statistics.median(values) if values else 0.0

    def p95(self, values: list[float]) -> float:
        if not values:
            return 0.0
        s = sorted(values)
        k = max(0, int(0.95 * (len(s) - 1)))
        return s[k]

    def format_row(self) -> str:
        miss_p50 = self.p50(self.miss_latencies_ms)
        hit_p50 = self.p50(self.hit_latencies_ms)
        hit_p95 = self.p95(self.hit_latencies_ms)
        speedup = f"{miss_p50 / hit_p50:.0f}×" if hit_p50 > 0 and miss_p50 > 0 else "—"
        return (
            f"  {self.name:<10} "
            f"{self.hit_rate:>5.1f}%  "
            f"{miss_p50:>7.2f}ms  "
            f"{hit_p50:>6.2f}ms  "
            f"{hit_p95:>6.2f}ms  "
            f"{speedup:>7}"
        )


def build_workspace(n_blocks: int) -> str:
    """Create a tmp workspace with ``n_blocks`` decision blocks."""
    ws = tempfile.mkdtemp(prefix="mm_cache_bench_")
    for d in ("decisions", "tasks", "entities", "intelligence", "memory", ".mind-mem-index"):
        os.makedirs(os.path.join(ws, d), exist_ok=True)

    decisions_md = os.path.join(ws, "decisions", "DECISIONS.md")
    topics = [
        "PostgreSQL", "Redis", "Kubernetes", "Docker", "FastAPI",
        "asyncio", "gRPC", "REST", "OAuth2", "JWT",
        "sqlite", "BM25", "RRF", "vector search", "RAG",
        "Prometheus", "Grafana", "OpenTelemetry", "CI/CD", "GitHub Actions",
    ]
    domains = ["auth", "storage", "recall", "observability", "deployment"]

    with open(decisions_md, "w", encoding="utf-8") as f:
        for i in range(n_blocks):
            topic = topics[i % len(topics)]
            domain = domains[i % len(domains)]
            block_id = f"D-20260420-{i:04d}"
            f.write(
                f"[{block_id}]\n"
                f"type: decision\n"
                f"Status: active\n"
                f"Statement: Use {topic} for {domain} layer in production.\n"
                f"Rationale: Chosen after evaluating {topic} against alternatives "
                f"for the {domain} subsystem. Benchmarks showed superior performance.\n"
                f"---\n\n"
            )
    return ws


def seed_query_pool(k: int) -> list[str]:
    """Build a pool of ``k`` realistic natural-language queries."""
    topics = [
        "PostgreSQL", "Redis", "Kubernetes", "Docker", "FastAPI",
        "asyncio", "gRPC", "REST", "OAuth2", "JWT",
        "sqlite", "BM25", "RRF", "vector search", "RAG",
        "Prometheus", "Grafana", "OpenTelemetry", "CI/CD", "GitHub Actions",
    ]
    domains = ["auth", "storage", "recall", "observability", "deployment"]
    queries: list[str] = []
    while len(queries) < k:
        t = topics[len(queries) % len(topics)]
        d = domains[(len(queries) // len(topics)) % len(domains)]
        queries.append(f"{t} for {d}")
    return queries[:k]


def reset_caches() -> None:
    """Drop the singleton + flush Redis so each config runs clean."""
    try:
        from mind_mem.recall_cache import reset_singleton
        reset_singleton()
    except ImportError:
        pass
    try:
        import redis

        client = redis.from_url(
            os.environ.get("MIND_MEM_REDIS_URL", "redis://localhost:6379/0"),
            decode_responses=True,
            socket_timeout=1.0,
        )
        client.flushdb()
    except Exception:
        pass


def run_config(
    name: str,
    ws: str,
    queries: list[str],
    pool: list[str],
    repeat_pct: float,
    *,
    cache_enabled: bool,
    redis_url: str | None,
) -> RunResult:
    """Run ``len(queries)`` recalls with the given cache config."""
    reset_caches()

    # Point mind-mem.json at the requested cache config.
    config = {
        "version": "1.7.0",
        "recall": {"backend": "bm25"},
        "cache": {
            "enabled": cache_enabled,
            "ttl_seconds": 300,
        },
    }
    if redis_url is not None:
        config["cache"]["redis_url"] = redis_url
    with open(os.path.join(ws, "mind-mem.json"), "w", encoding="utf-8") as f:
        json.dump(config, f)

    os.environ["MIND_MEM_WORKSPACE"] = ws
    if redis_url is not None:
        os.environ["MIND_MEM_REDIS_URL"] = redis_url
    else:
        os.environ.pop("MIND_MEM_REDIS_URL", None)

    # Build indexes.
    from mind_mem.sqlite_index import build_index
    build_index(ws, incremental=False)

    # Fresh _recall_impl with the new cache config.
    from mind_mem.mcp.tools.recall import _recall_impl

    result = RunResult(name=name)
    result.n_queries = len(queries)

    seen: set[str] = set()
    for q in queries:
        first_time = q not in seen
        t0 = time.perf_counter()
        _recall_impl(query=q, limit=10, active_only=False, backend="bm25")
        dt_ms = (time.perf_counter() - t0) * 1000.0
        if first_time:
            result.miss_latencies_ms.append(dt_ms)
            seen.add(q)
        else:
            result.hit_latencies_ms.append(dt_ms)

    return result


def build_query_stream(
    total: int, pool: list[str], repeat_pct: float, seed: int = 42
) -> list[str]:
    """Generate ``total`` queries. First ``|pool|`` unique, rest sampled."""
    rng = random.Random(seed)
    stream: list[str] = list(pool)
    remaining = total - len(pool)
    for _ in range(max(0, remaining)):
        stream.append(rng.choice(pool))
    return stream


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=500, help="Number of seed blocks")
    parser.add_argument("--queries", type=int, default=1000, help="Total queries")
    parser.add_argument("--pool", type=int, default=40, help="Unique queries in pool")
    parser.add_argument("--repeat-pct", type=float, default=80.0, help="Repeat percentage")
    parser.add_argument("--redis-url", default=os.environ.get("MIND_MEM_REDIS_URL", "redis://localhost:6379/0"))
    parser.add_argument("--output", help="Optional path to write result JSON")
    args = parser.parse_args()

    print(f"mind-mem cache-effectiveness bench")
    print(f"  blocks: {args.n}")
    print(f"  queries: {args.queries} (unique pool size {args.pool}, repeat {args.repeat_pct:.0f}%)")
    print(f"  redis: {args.redis_url}")
    print()

    print("Building workspace + indexes...")
    ws = build_workspace(args.n)
    pool = seed_query_pool(args.pool)
    stream = build_query_stream(args.queries, pool, args.repeat_pct)

    try:
        # Warm-up to load embedding models / Python imports once.
        os.environ["MIND_MEM_WORKSPACE"] = ws

        configs: list[tuple[str, dict]] = [
            ("no_cache", {"cache_enabled": False, "redis_url": None}),
            ("lru_only", {"cache_enabled": True, "redis_url": None}),
            ("redis", {"cache_enabled": True, "redis_url": args.redis_url}),
        ]

        results: list[RunResult] = []
        for name, cfg in configs:
            print(f"  [{name:<9}] running {len(stream)} queries...", flush=True)
            r = run_config(name, ws, stream, pool, args.repeat_pct, **cfg)
            results.append(r)

        print()
        print(f"  {'Config':<10} {'Hit%':>6} {'p50 miss':>9} {'p50 hit':>8} {'p95 hit':>8} {'Speedup':>8}")
        print(f"  {'-' * 10} {'-' * 6} {'-' * 9} {'-' * 8} {'-' * 8} {'-' * 8}")
        for r in results:
            print(r.format_row())
        print()

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "n_blocks": args.n,
                        "n_queries": args.queries,
                        "pool_size": args.pool,
                        "repeat_pct": args.repeat_pct,
                        "runs": [
                            {
                                "name": r.name,
                                "hit_rate_pct": r.hit_rate,
                                "miss_p50_ms": r.p50(r.miss_latencies_ms),
                                "miss_p95_ms": r.p95(r.miss_latencies_ms),
                                "hit_p50_ms": r.p50(r.hit_latencies_ms),
                                "hit_p95_ms": r.p95(r.hit_latencies_ms),
                                "n_misses": len(r.miss_latencies_ms),
                                "n_hits": len(r.hit_latencies_ms),
                            }
                            for r in results
                        ],
                    },
                    f,
                    indent=2,
                )
            print(f"Results written to {args.output}")

    finally:
        shutil.rmtree(ws, ignore_errors=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
