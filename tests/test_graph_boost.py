#!/usr/bin/env python3
"""Tests for graph boost, context packing, config validation, and block cap.

Covers:
    1. In-memory graph boost path (recall with graph_boost=True)
    2. context_pack() function (dialog adjacency, empty inputs, no-augment)
    3. _load_backend() config validation (unknown keys)
    4. MAX_BLOCKS_PER_QUERY cap enforcement
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from recall import (
    MAX_BLOCKS_PER_QUERY,
    _load_backend,
    build_xref_graph,
    context_pack,
    recall,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_workspace(tmpdir, decisions="", tasks=""):
    """Create a minimal workspace directory structure for recall tests."""
    for d in ("decisions", "tasks", "entities", "intelligence"):
        os.makedirs(os.path.join(tmpdir, d), exist_ok=True)

    with open(os.path.join(tmpdir, "decisions", "DECISIONS.md"), "w") as f:
        f.write(decisions)

    default_task = "[T-20260101-099]\nTitle: Unrelated placeholder task\nStatus: active\n"
    with open(os.path.join(tmpdir, "tasks", "TASKS.md"), "w") as f:
        f.write(tasks or default_task)

    # Stub out the remaining corpus files so recall does not skip them
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


def _block(block_id, statement, status="active", date="2026-01-01", **extra):
    """Format a single block as markdown text."""
    lines = [
        f"[{block_id}]",
        f"Statement: {statement}",
        f"Status: {status}",
        f"Date: {date}",
    ]
    for key, val in extra.items():
        lines.append(f"{key}: {val}")
    return "\n".join(lines)


def _dia_block(block_id, dia_id, speaker, text, status="active", **extra):
    """Format a dialog turn block as markdown text."""
    lines = [
        f"[{block_id}]",
        f"Statement: [{speaker}] {text}",
        f"Status: {status}",
        f"Tags: session-1, {speaker}",
        f"DiaID: {dia_id}",
    ]
    for key, val in extra.items():
        lines.append(f"{key}: {val}")
    return "\n".join(lines)


def _dia_result(block_id, dia_id, speaker, text, score=10.0):
    """Create a result dict that mimics recall output for a dialog turn."""
    return {
        "_id": block_id,
        "type": "unknown",
        "score": score,
        "excerpt": f"[{speaker}] {text}",
        "speaker": speaker,
        "tags": f"session-1, {speaker}",
        "file": "decisions/DECISIONS.md",
        "line": 10,
        "status": "active",
        "DiaID": dia_id,
    }


def _parsed_dia_block(block_id, dia_id, speaker, text, status="active"):
    """Create a parsed block dict (as returned by parse_file)."""
    return {
        "_id": block_id,
        "Statement": f"[{speaker}] {text}",
        "Status": status,
        "Tags": f"session-1, {speaker}",
        "DiaID": dia_id,
        "_source_file": "decisions/DECISIONS.md",
        "_line": 10,
    }


# ---------------------------------------------------------------------------
# 1. Graph boost via cross-references
# ---------------------------------------------------------------------------

class TestGraphBoostCrossRef(unittest.TestCase):
    """Verify that blocks cross-referencing each other get boosted with
    via_graph=True when recall() is invoked with graph_boost=True."""

    def test_xref_field_creates_graph_edge(self):
        """XRefs field should cause build_xref_graph to create edges."""
        blocks = [
            {"_id": "D-20260101-001", "Statement": "Main decision about auth",
             "Context": "Related to D-20260101-002"},
            {"_id": "D-20260101-002", "Statement": "Secondary decision"},
        ]
        graph = build_xref_graph(blocks)
        self.assertIn("D-20260101-002", graph["D-20260101-001"])
        self.assertIn("D-20260101-001", graph["D-20260101-002"])

    def test_explicit_xref_text_creates_edge(self):
        """A block referencing another in its Statement field yields an edge."""
        blocks = [
            {"_id": "D-20260101-001",
             "Statement": "Implements the requirement from D-20260101-002"},
            {"_id": "D-20260101-002",
             "Statement": "Original requirement for authentication"},
        ]
        graph = build_xref_graph(blocks)
        self.assertIn("D-20260101-002", graph["D-20260101-001"])
        self.assertIn("D-20260101-001", graph["D-20260101-002"])

    def test_graph_boost_surfaces_referenced_block(self):
        """recall(graph_boost=True) should surface a block that is referenced
        by a matching block but does not itself match the query."""
        with tempfile.TemporaryDirectory() as td:
            decisions = "\n\n---\n\n".join([
                _block("D-20260101-001",
                       "Use PostgreSQL for the database layer",
                       Context="See D-20260101-002 for migration plan"),
                _block("D-20260101-002",
                       "Migration plan for schema upgrades"),
            ])
            ws = _make_workspace(td, decisions=decisions)

            # Without graph: only the keyword-matching block appears
            results_no_graph = recall(ws, "PostgreSQL database", graph_boost=False)
            ids_no_graph = {r["_id"] for r in results_no_graph}
            self.assertIn("D-20260101-001", ids_no_graph)

            # With graph: the referenced block should appear via graph
            results_graph = recall(ws, "PostgreSQL database", graph_boost=True)
            ids_graph = {r["_id"] for r in results_graph}
            self.assertIn("D-20260101-001", ids_graph)
            self.assertIn("D-20260101-002", ids_graph)

    def test_graph_boost_sets_via_graph_flag(self):
        """Blocks discovered via graph traversal should have via_graph=True."""
        with tempfile.TemporaryDirectory() as td:
            decisions = "\n\n---\n\n".join([
                _block("D-20260101-001",
                       "Deploy Redis caching layer",
                       Context="Related to D-20260101-002"),
                _block("D-20260101-002",
                       "Cache invalidation strategy"),
            ])
            ws = _make_workspace(td, decisions=decisions)

            results = recall(ws, "Redis caching", graph_boost=True)
            graph_hits = [r for r in results if r.get("via_graph")]
            self.assertGreater(len(graph_hits), 0,
                               "At least one result should be flagged via_graph")

    def test_graph_boost_increases_score(self):
        """A block that both matches the query AND is referenced by another
        matching block should have a higher score with graph boost."""
        with tempfile.TemporaryDirectory() as td:
            # Both blocks mention "encryption" so both get BM25 scores,
            # and they cross-reference each other so graph boost adds more.
            decisions = "\n\n---\n\n".join([
                _block("D-20260101-001",
                       "Enable encryption at rest for all databases",
                       Context="Depends on D-20260101-002"),
                _block("D-20260101-002",
                       "Encryption key management policy",
                       Context="Required by D-20260101-001"),
            ])
            ws = _make_workspace(td, decisions=decisions)

            results_plain = recall(ws, "encryption", graph_boost=False)
            results_graph = recall(ws, "encryption", graph_boost=True)

            score_plain = {r["_id"]: r["score"] for r in results_plain}
            score_graph = {r["_id"]: r["score"] for r in results_graph}

            # With graph boost, scores should be >= plain scores
            for bid in ("D-20260101-001", "D-20260101-002"):
                if bid in score_plain and bid in score_graph:
                    self.assertGreaterEqual(score_graph[bid], score_plain[bid])

    def test_graph_boost_no_self_edges(self):
        """A block mentioning its own ID should not create a self-edge."""
        blocks = [
            {"_id": "D-20260101-001",
             "Statement": "This is D-20260101-001, a self-referential block"},
        ]
        graph = build_xref_graph(blocks)
        self.assertNotIn("D-20260101-001", graph.get("D-20260101-001", set()))

    def test_graph_boost_unknown_refs_ignored(self):
        """References to non-existent block IDs should not appear in graph."""
        blocks = [
            {"_id": "D-20260101-001",
             "Statement": "References D-20260101-999 which does not exist"},
        ]
        graph = build_xref_graph(blocks)
        self.assertEqual(len(graph.get("D-20260101-001", set())), 0)

    def test_graph_boost_chain_traversal(self):
        """Graph traversal should follow chains: A->B->C discovers C from A."""
        with tempfile.TemporaryDirectory() as td:
            decisions = "\n\n---\n\n".join([
                _block("D-20260101-001",
                       "Adopt Kubernetes for container orchestration",
                       Context="See D-20260101-002"),
                _block("D-20260101-002",
                       "Helm charts for deployment automation",
                       Context="Feeds into D-20260101-003"),
                _block("D-20260101-003",
                       "Monitoring dashboard for cluster health"),
            ])
            ws = _make_workspace(td, decisions=decisions)

            results = recall(ws, "Kubernetes orchestration", graph_boost=True)
            ids = {r["_id"] for r in results}
            # The primary match is D-001. D-002 is 1 hop, D-003 is 2 hops.
            self.assertIn("D-20260101-001", ids)
            self.assertIn("D-20260101-002", ids)
            # D-003 may or may not appear depending on hop decay threshold,
            # but the 2-hop traversal should reach it.
            # We check it is at least reachable in the graph.
            blocks_parsed = [
                {"_id": "D-20260101-001", "Context": "See D-20260101-002"},
                {"_id": "D-20260101-002", "Context": "Feeds into D-20260101-003"},
                {"_id": "D-20260101-003", "Statement": "Monitoring"},
            ]
            graph = build_xref_graph(blocks_parsed)
            self.assertIn("D-20260101-002", graph["D-20260101-001"])
            self.assertIn("D-20260101-003", graph["D-20260101-002"])


# ---------------------------------------------------------------------------
# 2. context_pack() tests
# ---------------------------------------------------------------------------

class TestContextPackDialogAdjacency(unittest.TestCase):
    """Test dialog adjacency expansion in context_pack()."""

    def _three_turn_dialog(self):
        """3-turn dialog: question -> answer -> followup."""
        return [
            _parsed_dia_block("DIA-D1-1", "D1:1", "Alice",
                              "What's the deployment process?"),
            _parsed_dia_block("DIA-D1-2", "D1:2", "Bob",
                              "We use a blue-green deployment strategy."),
            _parsed_dia_block("DIA-D1-3", "D1:3", "Alice",
                              "That makes sense, thanks for explaining."),
        ]

    def test_question_pulls_adjacent_answers(self):
        """A question turn in top_results should pull next 1-2 turns."""
        blocks = self._three_turn_dialog()
        top = [_dia_result("DIA-D1-1", "D1:1", "Alice",
                           "What's the deployment process?", score=20.0)]
        result = context_pack("deployment process", top, blocks, top, limit=10)
        dias = {r["DiaID"] for r in result}
        self.assertIn("D1:2", dias, "Answer turn should be pulled via adjacency")

    def test_adjacency_pulls_up_to_two_turns(self):
        """Adjacency expansion should pull up to 2 subsequent turns."""
        blocks = self._three_turn_dialog()
        top = [_dia_result("DIA-D1-1", "D1:1", "Alice",
                           "What's the deployment process?", score=20.0)]
        result = context_pack("deployment process", top, blocks, top, limit=10)
        dias = {r["DiaID"] for r in result}
        self.assertIn("D1:2", dias)
        self.assertIn("D1:3", dias)

    def test_adjacency_marks_via_adjacency(self):
        """Adjacency-added results should have via_adjacency=True."""
        blocks = self._three_turn_dialog()
        top = [_dia_result("DIA-D1-1", "D1:1", "Alice",
                           "What's the deployment process?", score=20.0)]
        result = context_pack("deployment process", top, blocks, top, limit=10)
        adj_hits = [r for r in result if r.get("via_adjacency")]
        self.assertGreater(len(adj_hits), 0)

    def test_non_question_no_adjacency(self):
        """A statement (not a question) should NOT trigger adjacency."""
        blocks = self._three_turn_dialog()
        top = [_dia_result("DIA-D1-2", "D1:2", "Bob",
                           "We use a blue-green deployment strategy.", score=20.0)]
        result = context_pack("deployment", top, blocks, top, limit=10)
        # Only the original result should be present
        self.assertEqual(len(result), 1)

    def test_fact_card_not_expanded(self):
        """Fact cards (non-DIA- prefix) should not trigger adjacency."""
        blocks = self._three_turn_dialog()
        fact = _dia_result("FACT-001", "D1:1", "Alice",
                           "What's the process?", score=20.0)
        result = context_pack("process", [fact], blocks, [fact], limit=10)
        self.assertEqual(len(result), 1,
                         "Fact cards must not trigger dialog adjacency")


class TestContextPackEmptyInputs(unittest.TestCase):
    """Test context_pack with empty or degenerate inputs."""

    def test_empty_top_results(self):
        """Empty top_results should return empty list."""
        result = context_pack("test query", [], [], [], limit=10)
        self.assertEqual(result, [])

    def test_empty_blocks_with_results(self):
        """Empty all_blocks should return top_results unchanged."""
        top = [_dia_result("DIA-D1-1", "D1:1", "Alice", "some text", score=10.0)]
        result = context_pack("test", top, [], top, limit=10)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["_id"], "DIA-D1-1")

    def test_empty_query_still_returns_results(self):
        """Even with empty query, top_results should be returned."""
        blocks = [_parsed_dia_block("DIA-D1-1", "D1:1", "Alice", "hello")]
        top = [_dia_result("DIA-D1-1", "D1:1", "Alice", "hello", score=5.0)]
        result = context_pack("", top, blocks, top, limit=10)
        self.assertEqual(len(result), 1)


class TestContextPackNoAugmentation(unittest.TestCase):
    """Test that context_pack returns results unchanged when no rules apply."""

    def test_non_dialog_blocks_unchanged(self):
        """Non-dialog results (D- prefix) should pass through without augmentation."""
        block = {
            "_id": "D-20260101-001",
            "Statement": "Use PostgreSQL for persistence",
            "Status": "active",
            "_source_file": "decisions/DECISIONS.md",
            "_line": 5,
        }
        result_dict = {
            "_id": "D-20260101-001",
            "type": "decision",
            "score": 15.0,
            "excerpt": "Use PostgreSQL for persistence",
            "speaker": "",
            "tags": "",
            "file": "decisions/DECISIONS.md",
            "line": 5,
            "status": "active",
            "DiaID": "",
        }
        result = context_pack("PostgreSQL", [result_dict], [block],
                              [result_dict], limit=10)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["_id"], "D-20260101-001")
        # No augmentation flags should be set
        self.assertFalse(result[0].get("via_adjacency", False))
        self.assertFalse(result[0].get("via_diversity", False))
        self.assertFalse(result[0].get("via_pronoun_rescue", False))

    def test_statement_turn_returns_unchanged(self):
        """A dialog statement (not question) with no pronouns returns as-is."""
        blocks = [
            _parsed_dia_block("DIA-D1-1", "D1:1", "Alice",
                              "The database migration completed successfully."),
        ]
        top = [_dia_result("DIA-D1-1", "D1:1", "Alice",
                           "The database migration completed successfully.",
                           score=18.0)]
        result = context_pack("database migration", top, blocks, top, limit=10)
        self.assertEqual(len(result), 1)


# ---------------------------------------------------------------------------
# 3. Config validation: _load_backend warns on unknown keys
# ---------------------------------------------------------------------------

class TestLoadBackendConfigValidation(unittest.TestCase):
    """Test that _load_backend() handles unknown config keys gracefully."""

    def test_unknown_key_returns_none(self):
        """_load_backend with unknown recall config key should return None
        (scan fallback) and not crash."""
        with tempfile.TemporaryDirectory() as td:
            config = {
                "recall": {
                    "backend": "scan",
                    "bogus_key": True,
                    "another_unknown": 42,
                }
            }
            with open(os.path.join(td, "mind-mem.json"), "w") as f:
                json.dump(config, f)

            result = _load_backend(td)
            self.assertIsNone(result,
                              "_load_backend should return None for 'scan' backend")

    def test_valid_config_returns_none_for_scan(self):
        """_load_backend with valid keys and backend=scan returns None."""
        with tempfile.TemporaryDirectory() as td:
            config = {
                "recall": {
                    "backend": "scan",
                    "limit": 10,
                }
            }
            with open(os.path.join(td, "mind-mem.json"), "w") as f:
                json.dump(config, f)

            result = _load_backend(td)
            self.assertIsNone(result)

    def test_missing_config_returns_none(self):
        """_load_backend with no config file returns None (scan fallback)."""
        with tempfile.TemporaryDirectory() as td:
            result = _load_backend(td)
            self.assertIsNone(result)

    def test_malformed_json_returns_none(self):
        """_load_backend with malformed JSON returns None without crashing."""
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "mind-mem.json"), "w") as f:
                f.write("{invalid json content!!!")

            result = _load_backend(td)
            self.assertIsNone(result)

    def test_empty_recall_section_returns_none(self):
        """_load_backend with empty recall section returns None."""
        with tempfile.TemporaryDirectory() as td:
            config = {"recall": {}}
            with open(os.path.join(td, "mind-mem.json"), "w") as f:
                json.dump(config, f)

            result = _load_backend(td)
            self.assertIsNone(result)


# ---------------------------------------------------------------------------
# 4. MAX_BLOCKS_PER_QUERY cap enforcement
# ---------------------------------------------------------------------------

class TestBlockCap(unittest.TestCase):
    """Verify that recall() caps the number of blocks it processes."""

    def test_max_blocks_constant_is_positive(self):
        """MAX_BLOCKS_PER_QUERY should be a positive integer."""
        self.assertIsInstance(MAX_BLOCKS_PER_QUERY, int)
        self.assertGreater(MAX_BLOCKS_PER_QUERY, 0)

    def test_recall_handles_many_blocks(self):
        """recall() should not crash when workspace has many blocks.

        We generate a workspace with more blocks than typical and verify
        recall returns results without error. We do not exceed the actual
        50000 cap (that would be too slow for a unit test), but we verify
        the code path handles a moderately large block count gracefully.
        """
        with tempfile.TemporaryDirectory() as td:
            # Generate 500 blocks -- enough to exercise the iteration path
            # without making the test slow.
            parts = []
            for i in range(1, 501):
                parts.append(
                    f"[D-20260101-{i:03d}]\n"
                    f"Statement: Authentication decision number {i} about tokens\n"
                    f"Status: active\nDate: 2026-01-01\n"
                )
            decisions = "\n---\n\n".join(parts)
            ws = _make_workspace(td, decisions=decisions)

            results = recall(ws, "authentication tokens", limit=10)
            self.assertGreater(len(results), 0)
            self.assertLessEqual(len(results), 10)

    def test_recall_returns_within_limit(self):
        """Even with many blocks, recall should respect the limit parameter."""
        with tempfile.TemporaryDirectory() as td:
            parts = []
            for i in range(1, 101):
                parts.append(
                    f"[D-20260101-{i:03d}]\n"
                    f"Statement: Database migration step {i}\n"
                    f"Status: active\nDate: 2026-01-01\n"
                )
            decisions = "\n---\n\n".join(parts)
            ws = _make_workspace(td, decisions=decisions)

            for limit in (1, 3, 5):
                results = recall(ws, "database migration", limit=limit)
                self.assertLessEqual(
                    len(results), limit,
                    f"Expected at most {limit} results, got {len(results)}")

    def test_cap_value_is_documented(self):
        """MAX_BLOCKS_PER_QUERY should be 50000 as documented."""
        self.assertEqual(MAX_BLOCKS_PER_QUERY, 50000)


# ---------------------------------------------------------------------------
# Integration: graph boost + context_pack combined
# ---------------------------------------------------------------------------

class TestGraphBoostIntegration(unittest.TestCase):
    """End-to-end integration of graph boost with real recall()."""

    def test_bidirectional_xref_both_blocks_found(self):
        """Two blocks that reference each other should both appear."""
        with tempfile.TemporaryDirectory() as td:
            decisions = "\n\n---\n\n".join([
                _block("D-20260101-001",
                       "Adopt GraphQL for the API gateway",
                       Context="Complementary to D-20260101-002"),
                _block("D-20260101-002",
                       "REST fallback for legacy clients",
                       Context="Fallback for D-20260101-001"),
            ])
            ws = _make_workspace(td, decisions=decisions)

            results = recall(ws, "GraphQL API gateway", graph_boost=True)
            ids = {r["_id"] for r in results}
            self.assertIn("D-20260101-001", ids)
            self.assertIn("D-20260101-002", ids)

    def test_graph_boost_with_task_references(self):
        """Cross-references between decisions and tasks should create edges."""
        with tempfile.TemporaryDirectory() as td:
            decisions = _block(
                "D-20260101-001",
                "Implement rate limiting on all endpoints",
                Context="Tracked in T-20260101-001",
            )
            tasks = "\n\n---\n\n".join([
                "[T-20260101-001]\n"
                "Title: Rate limiter implementation\n"
                "Status: active\n"
                "AlignsWith: D-20260101-001\n",
                "[T-20260101-099]\nTitle: Placeholder\nStatus: active\n",
            ])
            ws = _make_workspace(td, decisions=decisions, tasks=tasks)

            results = recall(ws, "rate limiting endpoints", graph_boost=True)
            ids = {r["_id"] for r in results}
            self.assertIn("D-20260101-001", ids)
            self.assertIn("T-20260101-001", ids)


if __name__ == "__main__":
    unittest.main()
