#!/usr/bin/env python3
"""Needle In A Haystack (NIAH) benchmark for mind-mem recall.

Tests that mind-mem's hybrid BM25+Vector search can reliably retrieve
a specific "needle" fact planted at various depths within a haystack
of filler memory blocks. Targets 100% retrieval across all combinations
of haystack size, depth, and needle type.

Matrix: 5 sizes × 5 depths × 10 needles = 250 test cases.
Target: needle in top-5 results for ALL cases.

Copyright (c) STARGA Inc. All rights reserved.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
import time

import pytest

# ---------------------------------------------------------------------------
# Needles — 10 diverse domain facts with queries and expected keywords
# ---------------------------------------------------------------------------

NEEDLES = [
    {
        "needle": "The secret project codenamed 'Velvet Hammer' was approved on February 14th, 2019, with a budget of exactly $7,432,891.",
        "query": "What was the budget for project Velvet Hammer?",
        "expected_keywords": ["7432891", "Velvet Hammer"],
    },
    {
        "needle": "Dr. Elena Vasquez discovered that mixing 3.7ml of compound XR-42 with titanium oxide produces a luminescent blue crystal at exactly 127.3°C.",
        "query": "What temperature produces luminescent crystals with compound XR-42?",
        "expected_keywords": ["127.3", "XR-42"],
    },
    {
        "needle": "Server rack 17B in the Portland datacenter has a faulty DIMM in slot J3 that causes memory errors every 72 hours, ticket INC-2024-8847.",
        "query": "Which server rack has a faulty DIMM?",
        "expected_keywords": ["17B", "J3"],
    },
    {
        "needle": "The Fibonacci sequence applied to Picotronix pod timing offsets produces optimal autoconfiguration convergence at the 13th term (233 microseconds).",
        "query": "What Fibonacci term optimizes Picotronix pod timing?",
        "expected_keywords": ["13th", "233"],
    },
    {
        "needle": "During the 1987 Chess Olympiad in Dubai, Sergei Volkov played a queen sacrifice on move 23 later proven optimal by Stockfish 16.",
        "query": "Who played the queen sacrifice at the 1987 Dubai Chess Olympiad?",
        "expected_keywords": ["Sergei Volkov", "23"],
    },
    {
        "needle": "The annual migration of Pyralidae moths through the Zanskar Valley peaks on August 17th, with an average count of 2.4 million individuals.",
        "query": "When do moths migrate through the Zanskar Valley?",
        "expected_keywords": ["August 17", "2.4 million"],
    },
    {
        "needle": "CEO Margaux Fontaine signed the Helix-9 acquisition deal for $312M on March 3rd, 2023, contingent on FTC regulatory approval within 180 days.",
        "query": "How much did the Helix-9 acquisition cost?",
        "expected_keywords": ["312M", "Margaux Fontaine"],
    },
    {
        "needle": "The deepest point of Lake Karachay measures 1,847 meters and contains tritium concentrations of 4.2 GBq per liter.",
        "query": "What is the depth and tritium level of Lake Karachay?",
        "expected_keywords": ["1847", "4.2 GBq"],
    },
    {
        "needle": "Algorithm Z-Prime reduces transformer inference latency by 42.7% on NVIDIA H100 GPUs when batch size exceeds 128 tokens.",
        "query": "What speedup does algorithm Z-Prime achieve on H100 GPUs?",
        "expected_keywords": ["42.7", "Z-Prime"],
    },
    {
        "needle": "The Treaty of Novgorod-Seversky signed December 5th, 1618, granted exclusive amber trading rights to the Hanseatic League for 99 years.",
        "query": "What did the Treaty of Novgorod-Seversky grant?",
        "expected_keywords": ["amber trading", "Hanseatic League"],
    },
]

# Haystack sizes and insertion depth percentages
HAYSTACK_SIZES = [10, 50, 100, 200, 500]
DEPTH_PERCENTAGES = [0, 25, 50, 75, 100]

TOP_K = 5  # Needle must appear in top-K results

# Embedding model config — use lighter model to avoid OOM on large matrices
_RECALL_CONFIG = {
    "backend": "hybrid",
    "rrf_k": 60,
    "bm25_weight": 1.0,
    "vector_weight": 1.0,
    "model": "all-MiniLM-L6-v2",
    "vector_enabled": True,
    "onnx_backend": True,
    "provider": "sqlite_vec",
    "knee_cutoff": False,
}


# ---------------------------------------------------------------------------
# Filler block generators — diverse content to make haystack realistic
# ---------------------------------------------------------------------------

_FILLER_TEMPLATES_DECISIONS = [
    "Adopted {tech} as primary {domain} framework after evaluating {count} alternatives. Migration deadline: Q{q} {year}.",
    "Policy updated: all {domain} deployments must pass {tech} compliance checks before production release.",
    "Budget allocated: ${amount} for {tech} infrastructure upgrade across {count} regional datacenters.",
    "Deprecated {tech} v{major}.{minor} in favor of {alt_tech}. Sunset date: {year}-{month:02d}-15.",
    "Security audit for {domain} systems completed. {count} critical vulnerabilities patched within 72 hours.",
    "Vendor contract with {company} renewed for {count} years at ${amount}/year for {domain} services.",
    "Performance benchmark established: {tech} achieves {throughput} ops/sec on standard workload profile.",
    "Data retention policy for {domain} logs extended from {count} days to {count2} days per compliance.",
    "Cross-team review of {tech} integration approved. {count} engineering teams affected by migration.",
    "Architecture decision: {tech} selected for {domain} microservices. Expected {throughput}% latency reduction.",
    "Hiring plan approved for {count} senior {domain} engineers. Target start date: {year}-{month:02d}-01.",
    "Risk assessment completed for {tech} dependency. Mitigation plan includes {count} fallback providers.",
    "Release cadence changed to bi-weekly for {domain} services. Next release: {year}-{month:02d}-{day:02d}.",
    "Observability stack upgraded: {tech} replaces legacy monitoring for {count} production services.",
    "Training program launched: {count} engineers enrolled in {tech} certification by end of Q{q}.",
]

_FILLER_TEMPLATES_TASKS = [
    "Migrate {service} from {tech} to {alt_tech} by {year}-{month:02d}-{day:02d}. Owner: {person}.",
    "Investigate {throughput}ms latency spike in {service} reported on {year}-{month:02d}-{day:02d}.",
    "Update {tech} SDK to version {major}.{minor}.{patch} across all {domain} microservices.",
    "Write runbook for {service} failover procedure. Due: {year}-{month:02d}-{day:02d}.",
    "Conduct load test for {service} at {throughput}x current traffic. Target: {count} concurrent users.",
    "Review and merge PR #{pr_num} for {tech} configuration changes in {service}.",
    "Set up automated {domain} regression tests for {service}. Target coverage: {throughput}%.",
    "Deploy {tech} hotfix {major}.{minor}.{patch} to {count} production nodes.",
    "Document API changes for {service} v{major}.{minor}. Deadline: {year}-{month:02d}-{day:02d}.",
    "Rotate TLS certificates for {count} {domain} endpoints before {year}-{month:02d}-{day:02d}.",
]

_FILLER_TEMPLATES_INCIDENTS = [
    "Outage in {service}: {tech} cluster lost {count} nodes due to {domain} failure. MTTR: {throughput} minutes.",
    "{service} returned HTTP 5xx errors for {throughput} minutes. Root cause: {tech} misconfiguration in {domain} layer.",
    "Data inconsistency detected in {service} after {tech} upgrade. {count} records affected, rollback initiated.",
    "Memory leak in {service} caused OOM kills on {count} pods. Fix: {tech} garbage collection tuning.",
    "Network partition between {service} and {alt_tech} backend lasted {throughput} seconds. Auto-healed.",
]

_TECHS = [
    "Kubernetes",
    "PostgreSQL",
    "Redis",
    "Kafka",
    "Elasticsearch",
    "gRPC",
    "GraphQL",
    "Terraform",
    "Docker",
    "Prometheus",
    "Envoy",
    "NATS",
    "RabbitMQ",
    "Consul",
    "Vault",
    "ArgoCD",
    "Istio",
    "Linkerd",
]
_DOMAINS = [
    "infrastructure",
    "security",
    "observability",
    "networking",
    "storage",
    "authentication",
    "billing",
    "analytics",
    "data-pipeline",
    "ML-ops",
]
_SERVICES = [
    "auth-service",
    "payment-gateway",
    "user-api",
    "notification-hub",
    "search-indexer",
    "order-processor",
    "inventory-sync",
    "email-relay",
    "cdn-edge",
    "rate-limiter",
    "session-store",
    "config-server",
]
_COMPANIES = ["Cloudflare", "Datadog", "PagerDuty", "Snowflake", "Confluent", "HashiCorp", "Elastic", "MongoDB", "CockroachDB", "Temporal"]
_PERSONS = [
    "Alice Chen",
    "Bob Martinez",
    "Carol Wu",
    "David Park",
    "Eva Johansson",
    "Frank Okafor",
    "Grace Kim",
    "Hiro Tanaka",
    "Isla Reeves",
    "Jamal Washington",
]


def _deterministic_hash(seed: int, salt: str = "") -> int:
    h = hashlib.sha256(f"{seed}:{salt}".encode()).hexdigest()
    return int(h[:8], 16)


def _pick(lst: list, seed: int, salt: str = "") -> str:
    return lst[_deterministic_hash(seed, salt) % len(lst)]


def _generate_filler_block(index: int, block_type: str = "decision") -> dict:
    s = index
    tech = _pick(_TECHS, s, "tech")
    alt_tech = _pick(_TECHS, s + 1000, "alt")
    domain = _pick(_DOMAINS, s, "domain")
    service = _pick(_SERVICES, s, "service")
    company = _pick(_COMPANIES, s, "company")
    person = _pick(_PERSONS, s, "person")
    year = 2020 + (s % 6)
    month = 1 + (s % 12)
    day = 1 + (s % 28)
    q = 1 + (s % 4)
    count = 2 + (s % 50)
    count2 = count * 3
    amount = 10000 + (s * 137) % 900000
    throughput = 10 + (s * 31) % 990
    major = 1 + (s % 5)
    minor = (s * 7) % 20
    patch = (s * 13) % 50
    pr_num = 1000 + s

    if block_type == "decision":
        templates = _FILLER_TEMPLATES_DECISIONS
    elif block_type == "task":
        templates = _FILLER_TEMPLATES_TASKS
    else:
        templates = _FILLER_TEMPLATES_INCIDENTS

    template = templates[s % len(templates)]
    try:
        statement = template.format(
            tech=tech,
            alt_tech=alt_tech,
            domain=domain,
            service=service,
            company=company,
            person=person,
            year=year,
            month=month,
            day=day,
            q=q,
            count=count,
            count2=count2,
            amount=amount,
            throughput=throughput,
            major=major,
            minor=minor,
            patch=patch,
            pr_num=pr_num,
        )
    except (KeyError, IndexError):
        statement = f"Filler block #{index}: {tech} configuration for {domain} in {service}."

    return {
        "index": index,
        "type": block_type,
        "statement": statement,
        "date": f"{year}-{month:02d}-{day:02d}",
        "tags": f"{tech}, {domain}, {service}",
    }


def _build_block_md(block_id: str, block: dict) -> str:
    lines = [f"[{block_id}]"]
    if block["type"] == "decision":
        lines.append("Type: Decision")
        lines.append("Status: active")
    elif block["type"] == "task":
        lines.append("Type: Task")
        lines.append("Status: active")
    else:
        lines.append("Type: Incident")
        lines.append("Status: active")
    lines.append(f"Date: {block['date']}")
    lines.append(f"Tags: {block['tags']}")
    lines.append(f"Statement: {block['statement']}")
    lines.append("")
    return "\n".join(lines)


def _build_needle_block_md(needle_id: str, needle_text: str) -> str:
    return (
        f"[{needle_id}]\n"
        f"Type: Decision\n"
        f"Status: active\n"
        f"Date: 2025-06-15\n"
        f"Tags: classified, needle, special-record\n"
        f"Statement: {needle_text}\n\n"
    )


# ---------------------------------------------------------------------------
# Workspace builder
# ---------------------------------------------------------------------------


def _build_workspace(
    haystack_size: int,
    needle_text: str,
    needle_id: str,
    depth_pct: int,
) -> str:
    """Build a temporary mind-mem workspace with filler blocks and one needle.

    Returns workspace path (caller must clean up with shutil.rmtree).
    """
    ws = tempfile.mkdtemp(prefix="niah_")

    for d in ["decisions", "tasks", "entities", "intelligence", ".mind-mem-index"]:
        os.makedirs(os.path.join(ws, d), exist_ok=True)

    config = {"version": "1.7.0", "recall": _RECALL_CONFIG.copy()}
    with open(os.path.join(ws, "mind-mem.json"), "w") as f:
        json.dump(config, f)

    n_decisions = max(1, int(haystack_size * 0.50))
    n_tasks = max(1, int(haystack_size * 0.30))
    n_other = max(1, haystack_size - n_decisions - n_tasks)

    needle_pos = max(0, min(n_decisions, int(depth_pct / 100.0 * n_decisions)))

    decisions_content = ""
    filler_idx = 0
    for i in range(n_decisions + 1):
        if i == needle_pos:
            decisions_content += _build_needle_block_md(needle_id, needle_text)
        else:
            block = _generate_filler_block(filler_idx, "decision")
            block_id = f"D-{20250101 + filler_idx}-{(filler_idx % 999) + 1:03d}"
            decisions_content += _build_block_md(block_id, block)
            filler_idx += 1

    with open(os.path.join(ws, "decisions", "DECISIONS.md"), "w") as f:
        f.write(decisions_content)

    tasks_content = ""
    for i in range(n_tasks):
        block = _generate_filler_block(filler_idx + i, "task")
        block_id = f"T-{20250101 + filler_idx + i}-{((filler_idx + i) % 999) + 1:03d}"
        tasks_content += _build_block_md(block_id, block)
    with open(os.path.join(ws, "tasks", "TASKS.md"), "w") as f:
        f.write(tasks_content)

    incidents_content = ""
    for i in range(n_other):
        block = _generate_filler_block(filler_idx + n_tasks + i, "incident")
        block_id = f"INC-{20250101 + filler_idx + n_tasks + i}-x{i:03d}"
        incidents_content += _build_block_md(block_id, block)
    with open(os.path.join(ws, "entities", "incidents.md"), "w") as f:
        f.write(incidents_content)

    for name in ["projects.md", "people.md", "tools.md"]:
        with open(os.path.join(ws, "entities", name), "w") as f:
            f.write("")
    for name in ["CONTRADICTIONS.md", "DRIFT.md", "SIGNALS.md"]:
        with open(os.path.join(ws, "intelligence", name), "w") as f:
            f.write("")

    return ws


# ---------------------------------------------------------------------------
# Search functions
# ---------------------------------------------------------------------------


def _build_indexes(workspace: str) -> None:
    """Build FTS5 + vector indexes."""
    from mind_mem.sqlite_index import build_index

    build_index(workspace, incremental=False)

    from mind_mem.recall_vector import VectorBackend

    vb = VectorBackend(_RECALL_CONFIG)
    vb.index(workspace)


def _hybrid_search(workspace: str, query: str, limit: int = 10) -> list[dict]:
    from mind_mem.hybrid_recall import HybridBackend

    hb = HybridBackend(_RECALL_CONFIG)
    return hb.search(query, workspace, limit=limit, active_only=False, rerank=False)


def _check_needle_found(results: list[dict], needle_id: str, expected_keywords: list[str]) -> bool:
    for r in results:
        rid = r.get("_id", "")
        if rid == needle_id:
            return True
        excerpt = r.get("excerpt", "")
        if all(kw.lower() in excerpt.lower() for kw in expected_keywords):
            return True
    return False


# ---------------------------------------------------------------------------
# Pytest parametrized test — 250 cases, build-test-cleanup per case
# ---------------------------------------------------------------------------


def _make_test_id(size, depth, needle_idx):
    return f"sz{size}_d{depth}_n{needle_idx}"


_TEST_PARAMS = []
for size in HAYSTACK_SIZES:
    for depth in DEPTH_PERCENTAGES:
        for needle_idx in range(len(NEEDLES)):
            _TEST_PARAMS.append((size, depth, needle_idx))


@pytest.mark.parametrize(
    "haystack_size,depth_pct,needle_idx",
    _TEST_PARAMS,
    ids=[_make_test_id(s, d, n) for s, d, n in _TEST_PARAMS],
)
def test_niah_hybrid(haystack_size: int, depth_pct: int, needle_idx: int):
    """NIAH: needle must be found in top-K via hybrid BM25+Vector search."""
    needle = NEEDLES[needle_idx]
    needle_id = f"NEEDLE-{needle_idx + 1:03d}"

    ws = _build_workspace(haystack_size, needle["needle"], needle_id, depth_pct)
    try:
        _build_indexes(ws)
        results = _hybrid_search(ws, needle["query"], limit=TOP_K)
        found = _check_needle_found(results, needle_id, needle["expected_keywords"])

        if not found:
            result_ids = [r.get("_id", "?") for r in results]
            result_excerpts = [r.get("excerpt", "")[:60] for r in results]
            pytest.fail(
                f"NIAH MISS: needle={needle_id} not in top-{TOP_K}\n"
                f"  haystack={haystack_size}, depth={depth_pct}%\n"
                f"  query: {needle['query']}\n"
                f"  expected: {needle['expected_keywords']}\n"
                f"  got ids: {result_ids}\n"
                f"  excerpts: {result_excerpts}"
            )
    finally:
        shutil.rmtree(ws, ignore_errors=True)


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------


def run_niah_standalone():
    """Run NIAH benchmark standalone with progress reporting."""
    print("=" * 70)
    print("NEEDLE IN A HAYSTACK BENCHMARK — mind-mem hybrid recall")
    print("=" * 70)

    total = len(_TEST_PARAMS)
    passed = 0
    failed = 0
    failures = []

    print(f"\nMatrix: {len(HAYSTACK_SIZES)} sizes × {len(DEPTH_PERCENTAGES)} depths × {len(NEEDLES)} needles = {total} cases")
    print(f"Sizes: {HAYSTACK_SIZES}")
    print(f"Depths: {DEPTH_PERCENTAGES}%")
    print(f"Top-K: {TOP_K}")
    print(f"Model: {_RECALL_CONFIG['model']}")
    print()

    start = time.time()
    for i, (size, depth, needle_idx) in enumerate(_TEST_PARAMS):
        needle = NEEDLES[needle_idx]
        needle_id = f"NEEDLE-{needle_idx + 1:03d}"

        ws = _build_workspace(size, needle["needle"], needle_id, depth)
        try:
            _build_indexes(ws)
            results = _hybrid_search(ws, needle["query"], limit=TOP_K)
            found = _check_needle_found(results, needle_id, needle["expected_keywords"])
        finally:
            shutil.rmtree(ws, ignore_errors=True)

        if found:
            passed += 1
        else:
            failed += 1
            failures.append(
                {
                    "size": size,
                    "depth": depth,
                    "needle_idx": needle_idx,
                    "query": needle["query"],
                    "got": [r.get("_id", "?") for r in results],
                }
            )

        if (i + 1) % 25 == 0 or (i + 1) == total:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (total - i - 1) / rate if rate > 0 else 0
            print(f"  [{i + 1:3d}/{total}] pass={passed} fail={failed} ({elapsed:.0f}s, ETA {eta:.0f}s)")

    elapsed = time.time() - start
    print(f"\n{'=' * 70}")
    print(f"RESULTS: {passed}/{total} passed ({100 * passed / total:.1f}%) in {elapsed:.0f}s")
    if failures:
        print(f"\nFAILURES ({len(failures)}):")
        for f in failures[:30]:
            print(f"  sz={f['size']} d={f['depth']}% n={f['needle_idx']}: {f['query'][:50]}")
            print(f"    got: {f['got']}")
    else:
        print("ALL 250 CASES PASSED ✓")
    print("=" * 70)

    return passed == total


if __name__ == "__main__":
    success = run_niah_standalone()
    sys.exit(0 if success else 1)
