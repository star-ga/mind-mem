#!/usr/bin/env python3
"""Tests for recall.py — zero external deps (stdlib unittest)."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from recall import (
    _rm3_language_model,
    build_xref_graph,
    extract_text,
    get_block_type,
    get_excerpt,
    recall,
    rm3_expand,
    tokenize,
)


class TestTokenize(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(tokenize("Hello World"), ["hello", "world"])

    def test_special_chars(self):
        self.assertEqual(tokenize("auth/JWT-token"), ["auth", "jwt", "token"])

    def test_empty(self):
        self.assertEqual(tokenize(""), [])

    def test_numbers(self):
        self.assertIn("42", tokenize("answer is 42"))

    def test_stopwords_filtered(self):
        tokens = tokenize("the database is being used for all queries")
        self.assertNotIn("the", tokens)
        self.assertNotIn("is", tokens)
        self.assertNotIn("for", tokens)
        self.assertIn("database", tokens)
        self.assertIn("query", tokens)  # stemmer reduces "queries" -> "query"

    def test_single_char_filtered(self):
        tokens = tokenize("a b c database")
        self.assertNotIn("a", tokens)
        self.assertNotIn("b", tokens)
        self.assertEqual(tokens, ["database"])


class TestExtractText(unittest.TestCase):
    def test_extracts_search_fields(self):
        block = {"Statement": "Use JWT", "Tags": "auth, security"}
        text = extract_text(block)
        self.assertIn("Use JWT", text)
        self.assertIn("auth, security", text)

    def test_extracts_constraint_sigs(self):
        block = {
            "ConstraintSignatures": [
                {"subject": "we", "predicate": "must_use", "object": "JWT", "domain": "auth"}
            ]
        }
        text = extract_text(block)
        self.assertIn("we", text)
        self.assertIn("must_use", text)

    def test_includes_block_id(self):
        block = {"_id": "D-20260214-001", "Statement": "Use JWT"}
        text = extract_text(block)
        self.assertIn("D-20260214-001", text)

    def test_empty_block(self):
        self.assertEqual(extract_text({}).strip(), "")


class TestGetBlockType(unittest.TestCase):
    def test_known_prefixes(self):
        self.assertEqual(get_block_type("D-20260213-001"), "decision")
        self.assertEqual(get_block_type("T-20260213-001"), "task")
        self.assertEqual(get_block_type("PRJ-001"), "project")
        self.assertEqual(get_block_type("SIG-20260213-001"), "signal")

    def test_unknown(self):
        self.assertEqual(get_block_type("X-001"), "unknown")


class TestGetExcerpt(unittest.TestCase):
    def test_returns_statement(self):
        block = {"Statement": "Use JWT for authentication"}
        self.assertEqual(get_excerpt(block), "Use JWT for authentication")

    def test_fallback_to_id(self):
        block = {"_id": "D-001"}
        self.assertEqual(get_excerpt(block), "D-001")

    def test_truncation(self):
        block = {"Statement": "x" * 400}
        self.assertEqual(len(get_excerpt(block)), 300)


class TestRecall(unittest.TestCase):
    def _setup_workspace(self, tmpdir, decisions_content=""):
        """Create minimal workspace with decisions file and a dummy task for IDF diversity."""
        for d in ["decisions", "tasks", "entities", "intelligence"]:
            os.makedirs(os.path.join(tmpdir, d), exist_ok=True)
        with open(os.path.join(tmpdir, "decisions", "DECISIONS.md"), "w") as f:
            f.write(decisions_content)
        # Create a dummy task block so TF-IDF has >1 document (IDF needs document diversity)
        with open(os.path.join(tmpdir, "tasks", "TASKS.md"), "w") as f:
            f.write("[T-20260213-099]\nTitle: Unrelated placeholder task\nStatus: active\n")
        for fname in ["entities/projects.md", "entities/people.md",
                       "entities/tools.md", "entities/incidents.md",
                       "intelligence/CONTRADICTIONS.md", "intelligence/DRIFT.md",
                       "intelligence/SIGNALS.md"]:
            path = os.path.join(tmpdir, fname)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write(f"# {os.path.basename(fname)}\n")
        return tmpdir

    def test_empty_query(self):
        with tempfile.TemporaryDirectory() as td:
            ws = self._setup_workspace(td)
            results = recall(ws, "")
            self.assertEqual(results, [])

    def test_no_results(self):
        with tempfile.TemporaryDirectory() as td:
            ws = self._setup_workspace(td, "# Decisions\n")
            results = recall(ws, "xyznonexistent")
            self.assertEqual(results, [])

    def test_finds_matching_block(self):
        with tempfile.TemporaryDirectory() as td:
            content = (
                "[D-20260213-001]\nStatement: Use JWT for authentication\n"
                "Status: active\nDate: 2026-02-13\n"
            )
            ws = self._setup_workspace(td, content)
            results = recall(ws, "JWT authentication")
            self.assertGreater(len(results), 0)
            self.assertEqual(results[0]["_id"], "D-20260213-001")

    def test_limit(self):
        with tempfile.TemporaryDirectory() as td:
            content = ""
            for i in range(1, 6):
                content += f"[D-20260213-{i:03d}]\nStatement: Auth decision {i}\nStatus: active\n\n---\n\n"
            ws = self._setup_workspace(td, content)
            results = recall(ws, "auth", limit=2)
            self.assertLessEqual(len(results), 2)

    def test_active_only_filter(self):
        with tempfile.TemporaryDirectory() as td:
            content = (
                "[D-20260213-001]\nStatement: JWT auth\nStatus: active\n\n---\n\n"
                "[D-20260213-002]\nStatement: JWT superseded\nStatus: superseded\n"
            )
            ws = self._setup_workspace(td, content)
            results = recall(ws, "JWT", active_only=True)
            ids = [r["_id"] for r in results]
            self.assertIn("D-20260213-001", ids)
            self.assertNotIn("D-20260213-002", ids)

    def test_boosts_active_status(self):
        """Active blocks should score higher than non-active ones."""
        with tempfile.TemporaryDirectory() as td:
            content = (
                "[D-20260213-001]\nStatement: JWT token auth\nStatus: superseded\nDate: 2026-02-13\n\n---\n\n"
                "[D-20260213-002]\nStatement: JWT token auth\nStatus: active\nDate: 2026-02-13\n"
            )
            ws = self._setup_workspace(td, content)
            results = recall(ws, "JWT token")
            self.assertEqual(len(results), 2)
            # Active block should rank higher
            self.assertEqual(results[0]["_id"], "D-20260213-002")


class TestBuildXrefGraph(unittest.TestCase):
    """Tests for cross-reference graph construction."""

    def test_bidirectional_edges(self):
        blocks = [
            {"_id": "D-20260213-001", "Statement": "Use JWT", "Context": "See T-20260213-001"},
            {"_id": "T-20260213-001", "Title": "Implement auth"},
        ]
        graph = build_xref_graph(blocks)
        self.assertIn("T-20260213-001", graph["D-20260213-001"])
        self.assertIn("D-20260213-001", graph["T-20260213-001"])

    def test_no_self_edges(self):
        blocks = [
            {"_id": "D-20260213-001", "Statement": "This is D-20260213-001"},
        ]
        graph = build_xref_graph(blocks)
        self.assertNotIn("D-20260213-001", graph["D-20260213-001"])

    def test_unknown_ids_ignored(self):
        blocks = [
            {"_id": "D-20260213-001", "Statement": "Ref D-20260213-999"},
        ]
        graph = build_xref_graph(blocks)
        self.assertEqual(len(graph["D-20260213-001"]), 0)


class TestGraphRecall(unittest.TestCase):
    """Tests for graph-boosted recall."""

    def _setup_workspace(self, tmpdir, decisions_content="", tasks_content=""):
        for d in ["decisions", "tasks", "entities", "intelligence"]:
            os.makedirs(os.path.join(tmpdir, d), exist_ok=True)
        with open(os.path.join(tmpdir, "decisions", "DECISIONS.md"), "w") as f:
            f.write(decisions_content)
        with open(os.path.join(tmpdir, "tasks", "TASKS.md"), "w") as f:
            f.write(tasks_content or "[T-20260213-099]\nTitle: Unrelated placeholder\nStatus: active\n")
        for fname in ["entities/projects.md", "entities/people.md",
                       "entities/tools.md", "entities/incidents.md",
                       "intelligence/CONTRADICTIONS.md", "intelligence/DRIFT.md",
                       "intelligence/SIGNALS.md"]:
            path = os.path.join(tmpdir, fname)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write(f"# {os.path.basename(fname)}\n")
        return tmpdir

    def test_graph_discovers_neighbor(self):
        """Graph recall should surface blocks connected to keyword matches."""
        with tempfile.TemporaryDirectory() as td:
            decisions = (
                "[D-20260213-001]\nStatement: Use PostgreSQL database\n"
                "Status: active\nDate: 2026-02-13\n"
            )
            tasks = (
                "[T-20260213-001]\nTitle: Set up database migration\n"
                "Status: active\nDate: 2026-02-13\n"
                "AlignsWith: D-20260213-001\n"
                "\n---\n\n"
                "[T-20260213-099]\nTitle: Unrelated placeholder\nStatus: active\n"
            )
            ws = self._setup_workspace(td, decisions, tasks)

            # Without graph: "PostgreSQL" only matches the decision
            results_no_graph = recall(ws, "PostgreSQL", graph_boost=False)
            ids_no_graph = [r["_id"] for r in results_no_graph]
            self.assertIn("D-20260213-001", ids_no_graph)
            self.assertNotIn("T-20260213-001", ids_no_graph)

            # With graph: task should appear via neighbor boost
            results_graph = recall(ws, "PostgreSQL", graph_boost=True)
            ids_graph = [r["_id"] for r in results_graph]
            self.assertIn("D-20260213-001", ids_graph)
            self.assertIn("T-20260213-001", ids_graph)

    def test_graph_boost_marks_results(self):
        """Graph-discovered results should have via_graph flag."""
        with tempfile.TemporaryDirectory() as td:
            decisions = (
                "[D-20260213-001]\nStatement: Use PostgreSQL database\n"
                "Status: active\nDate: 2026-02-13\n"
            )
            tasks = (
                "[T-20260213-001]\nTitle: Setup migration\n"
                "Status: active\nAlignsWith: D-20260213-001\n"
                "\n---\n\n"
                "[T-20260213-099]\nTitle: Unrelated placeholder\nStatus: active\n"
            )
            ws = self._setup_workspace(td, decisions, tasks)
            results = recall(ws, "PostgreSQL", graph_boost=True)
            graph_results = [r for r in results if r.get("via_graph")]
            self.assertGreater(len(graph_results), 0)


class TestStemmer(unittest.TestCase):
    """Tests for the simplified Porter stemmer."""

    def setUp(self):
        from recall import _stem
        self.stem = _stem

    def test_ing_suffix(self):
        self.assertEqual(self.stem("running"), "runn")
        self.assertEqual(self.stem("testing"), "test")

    def test_ed_suffix(self):
        self.assertEqual(self.stem("decided"), "decid")
        # "used" is only 4 chars — below len>4 threshold for -ed rule
        self.assertEqual(self.stem("used"), "used")

    def test_tion_suffix(self):
        # -tion rule: strip "tion", add "t" → "authenticat"
        self.assertEqual(self.stem("authentication"), "authenticat")

    def test_ies_suffix(self):
        self.assertEqual(self.stem("queries"), "query")
        self.assertEqual(self.stem("entries"), "entry")

    def test_ly_suffix(self):
        self.assertEqual(self.stem("quickly"), "quick")

    def test_ment_suffix(self):
        self.assertEqual(self.stem("deployment"), "deploy")

    def test_short_words_unchanged(self):
        self.assertEqual(self.stem("go"), "go")
        self.assertEqual(self.stem("db"), "db")
        self.assertEqual(self.stem("api"), "api")

    def test_already_stemmed(self):
        # Words that don't match any suffix rule
        self.assertEqual(self.stem("auth"), "auth")
        self.assertEqual(self.stem("jwt"), "jwt")


class TestExpandQuery(unittest.TestCase):
    """Tests for domain-aware query expansion."""

    def setUp(self):
        from recall import expand_query
        self.expand = expand_query

    def test_auth_expands(self):
        expanded = self.expand(["auth"])
        self.assertIn("auth", expanded)
        self.assertGreater(len(expanded), 1)

    def test_db_expands(self):
        expanded = self.expand(["db"])
        self.assertIn("db", expanded)
        self.assertGreater(len(expanded), 1)

    def test_unknown_doesnt_expand(self):
        expanded = self.expand(["xyzfoo"])
        self.assertEqual(expanded, ["xyzfoo"])

    def test_max_expansions_respected(self):
        expanded = self.expand(["auth"], max_expansions=1)
        self.assertLessEqual(len(expanded), 2)  # original + 1

    def test_no_duplicates(self):
        expanded = self.expand(["auth", "authentication"])
        self.assertEqual(len(expanded), len(set(expanded)))


class TestRM3Expansion(unittest.TestCase):
    """Tests for RM3 pseudo-relevance feedback."""

    def test_rm3_no_expansion_at_alpha_1(self):
        """alpha=1.0 means no expansion (original query only)."""
        result = rm3_expand(["cat", "dog"], [], {}, 100, alpha=1.0)
        self.assertEqual(set(result.keys()), {"cat", "dog"})
        for v in result.values():
            self.assertAlmostEqual(v, 1.0)

    def test_rm3_expands_terms(self):
        """RM3 should add expansion terms from feedback docs."""
        docs = [
            (["cat", "feline", "whisker", "pet"], 0.9),
            (["cat", "kitten", "purr", "pet"], 0.8),
        ]
        cf = {"cat": 10, "feline": 2, "whisker": 1, "pet": 8, "kitten": 3, "purr": 1}
        result = rm3_expand(["cat"], docs, cf, 100, alpha=0.6, fb_terms=3, fb_docs=2)
        # Should have original + some expansion terms
        self.assertIn("cat", result)
        self.assertGreater(len(result), 1)

    def test_rm3_not_activated_for_adversarial(self):
        """RM3 should not be used for adversarial queries (handled by caller)."""
        # This is checked by the caller, not rm3_expand itself
        result = rm3_expand(["cat"], [], {}, 100, alpha=0.6)
        self.assertIn("cat", result)  # Still returns original tokens

    def test_rm3_empty_feedback(self):
        """No feedback docs -> returns original query."""
        result = rm3_expand(["test"], [], {}, 100, alpha=0.6)
        self.assertEqual(set(result.keys()), {"test"})

    def test_rm3_language_model(self):
        """JM-smoothed language model produces valid probabilities."""
        probs = _rm3_language_model(
            ["cat", "cat", "dog"],
            {"cat": 10, "dog": 5, "bird": 3},
            100,
        )
        self.assertGreater(probs["cat"], probs["dog"])  # cat has higher tf
        self.assertGreater(probs["bird"], 0)  # smoothing gives non-zero


if __name__ == "__main__":
    unittest.main()
