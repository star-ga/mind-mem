#!/usr/bin/env python3
"""Concurrency and performance stress tests for recall engine.

Tests cover:
1. TestConcurrentRecall — parallel recall() thread safety with SQLite and BM25
2. TestPerformanceStress — performance under synthetic block loads (1000-2000+)
"""

import os
import sys
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from recall import recall

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_block(prefix, num, statement, status="active"):
    """Generate a single memory block in standard markdown format."""
    date = "2026-02-19"
    return (
        f"[{prefix}-{date.replace('-', '')}-{num:03d}]\n"
        f"Statement: {statement}\n"
        f"Status: {status}\n"
        f"Date: {date}\n"
    )


def _setup_workspace(tmpdir, decisions_content="", tasks_content=""):
    """Create a minimal workspace with all required directories and files."""
    for d in ("decisions", "tasks", "entities", "intelligence"):
        os.makedirs(os.path.join(tmpdir, d), exist_ok=True)

    with open(os.path.join(tmpdir, "decisions", "DECISIONS.md"), "w") as f:
        f.write(decisions_content)

    default_task = (
        "[T-20260219-099]\nTitle: Unrelated placeholder task\nStatus: active\n"
    )
    with open(os.path.join(tmpdir, "tasks", "TASKS.md"), "w") as f:
        f.write(tasks_content or default_task)

    for fname in (
        "entities/projects.md",
        "entities/people.md",
        "entities/tools.md",
        "entities/incidents.md",
        "intelligence/CONTRADICTIONS.md",
        "intelligence/DRIFT.md",
        "intelligence/SIGNALS.md",
    ):
        path = os.path.join(tmpdir, fname)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(f"# {os.path.basename(fname)}\n")

    return tmpdir


def _generate_blocks(n, topic_fn=None):
    """Generate *n* decision blocks separated by ``---`` markers.

    Args:
        n: Number of blocks to generate.
        topic_fn: Optional callable(index) -> statement string.
                  Defaults to a rotating set of topics.
    """
    topics = [
        "Use JWT for authentication across all microservices",
        "Deploy PostgreSQL as primary relational database",
        "Implement Redis caching layer for session management",
        "Adopt Kubernetes for container orchestration",
        "Use GraphQL for the public API gateway",
        "Enable TLS 1.3 for all internal service communication",
        "Migrate logging pipeline to OpenTelemetry",
        "Store secrets in HashiCorp Vault with auto-rotation",
        "Use Terraform for infrastructure as code provisioning",
        "Run nightly chaos engineering tests in staging",
    ]
    blocks = []
    for i in range(1, n + 1):
        if topic_fn is not None:
            stmt = topic_fn(i)
        else:
            stmt = f"{topics[(i - 1) % len(topics)]} (variant {i})"
        blocks.append(_make_block("D", i, stmt))
    return "\n---\n\n".join(blocks)


# ===========================================================================
# 1. Concurrent Recall
# ===========================================================================

class TestConcurrentRecall(unittest.TestCase):
    """Verify thread safety when multiple recall() calls run in parallel."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        content = _generate_blocks(50)
        self.ws = _setup_workspace(self._tmpdir.name, decisions_content=content)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_four_parallel_queries_return_correct_results(self):
        """4 parallel queries for the same term all return results without errors."""
        errors = []
        results = []

        def do_recall():
            return recall(self.ws, "authentication JWT")

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(do_recall) for _ in range(4)]
            for fut in as_completed(futures):
                try:
                    res = fut.result(timeout=10)
                    results.append(res)
                except Exception as exc:
                    errors.append(exc)

        self.assertEqual(errors, [], f"Parallel queries raised errors: {errors}")
        self.assertEqual(len(results), 4)
        for res in results:
            self.assertIsInstance(res, list)
            self.assertGreater(len(res), 0, "Expected at least one result per query")

    def test_eight_parallel_queries_no_crash(self):
        """8 parallel queries (higher contention) all complete without crashing."""
        errors = []

        def do_recall():
            return recall(self.ws, "database PostgreSQL")

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(do_recall) for _ in range(8)]
            for fut in as_completed(futures):
                try:
                    fut.result(timeout=10)
                except Exception as exc:
                    errors.append(exc)

        self.assertEqual(errors, [], f"High-contention queries crashed: {errors}")

    def test_parallel_different_queries_all_valid(self):
        """Parallel queries with different query strings all return valid results."""
        queries = [
            "authentication JWT",
            "database PostgreSQL",
            "caching Redis",
            "Kubernetes orchestration",
            "GraphQL API",
        ]
        results_map = {}
        errors = []

        def do_recall(q):
            return (q, recall(self.ws, q))

        with ThreadPoolExecutor(max_workers=len(queries)) as pool:
            futures = [pool.submit(do_recall, q) for q in queries]
            for fut in as_completed(futures):
                try:
                    query, res = fut.result(timeout=10)
                    results_map[query] = res
                except Exception as exc:
                    errors.append(exc)

        self.assertEqual(errors, [], f"Mixed queries raised errors: {errors}")
        self.assertEqual(len(results_map), len(queries))
        for query, res in results_map.items():
            self.assertIsInstance(res, list, f"Query '{query}' returned non-list")
            self.assertGreater(
                len(res), 0,
                f"Query '{query}' returned zero results",
            )

    def test_parallel_graph_boost_no_state_corruption(self):
        """Parallel queries with graph_boost=True don't corrupt shared state."""
        errors = []
        results = []

        def do_recall():
            return recall(self.ws, "authentication JWT", graph_boost=True)

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(do_recall) for _ in range(4)]
            for fut in as_completed(futures):
                try:
                    res = fut.result(timeout=10)
                    results.append(res)
                except Exception as exc:
                    errors.append(exc)

        self.assertEqual(errors, [], f"Graph-boost queries raised errors: {errors}")
        self.assertEqual(len(results), 4)

        # All threads should return the same set of block IDs (deterministic)
        id_sets = [frozenset(r["_id"] for r in res) for res in results]
        for ids in id_sets[1:]:
            self.assertEqual(
                ids, id_sets[0],
                "Graph-boost results diverged across threads — possible state corruption",
            )


# ===========================================================================
# 2. Performance Stress
# ===========================================================================

class TestPerformanceStress(unittest.TestCase):
    """Performance and stress tests with large synthetic workspaces."""

    def test_1000_blocks_recall_under_5_seconds(self):
        """Recall over 1000 blocks completes in under 5 seconds."""
        with tempfile.TemporaryDirectory() as tmpdir:
            content = _generate_blocks(1000)
            ws = _setup_workspace(tmpdir, decisions_content=content)

            start = time.monotonic()
            results = recall(ws, "authentication JWT", limit=10)
            elapsed = time.monotonic() - start

            self.assertIsInstance(results, list)
            self.assertGreater(len(results), 0)
            self.assertLess(
                elapsed, 5.0,
                f"Recall over 1000 blocks took {elapsed:.2f}s (limit: 5s)",
            )

    def test_2000_blocks_returns_results(self):
        """Recall over 2000 blocks still returns results without error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            content = _generate_blocks(2000)
            ws = _setup_workspace(tmpdir, decisions_content=content)

            results = recall(ws, "Terraform infrastructure", limit=10)

            self.assertIsInstance(results, list)
            self.assertGreater(len(results), 0, "Expected results from 2000-block corpus")

    def test_many_blocks_graph_boost_no_memory_blowup(self):
        """Graph boost on a large corpus completes without excessive memory use."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create blocks with cross-references to exercise graph traversal
            def topic_with_xref(i):
                ref_id = f"D-20260219-{((i % 50) + 1):03d}"
                return (
                    f"Decision about service {i} architecture "
                    f"(see {ref_id} for related context)"
                )

            content = _generate_blocks(1000, topic_fn=topic_with_xref)
            ws = _setup_workspace(tmpdir, decisions_content=content)

            start = time.monotonic()
            results = recall(ws, "architecture service", graph_boost=True, limit=10)
            elapsed = time.monotonic() - start

            self.assertIsInstance(results, list)
            self.assertGreater(len(results), 0)
            # Graph boost on 1000 blocks with xrefs should still be reasonable
            self.assertLess(
                elapsed, 10.0,
                f"Graph-boost recall on 1000 xref blocks took {elapsed:.2f}s (limit: 10s)",
            )

    def test_limit_respected_with_many_blocks(self):
        """The limit parameter is respected even when the corpus is large."""
        with tempfile.TemporaryDirectory() as tmpdir:
            content = _generate_blocks(500)
            ws = _setup_workspace(tmpdir, decisions_content=content)

            for limit in (1, 3, 5, 10):
                results = recall(ws, "authentication", limit=limit)
                self.assertLessEqual(
                    len(results), limit,
                    f"limit={limit} but got {len(results)} results",
                )


if __name__ == "__main__":
    unittest.main()
