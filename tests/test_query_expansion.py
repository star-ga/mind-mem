#!/usr/bin/env python3
"""Tests for query_expansion.py -- multi-query expansion for improved recall."""

import unittest

from mind_mem.query_expansion import (
    LLMQueryExpander,
    NLPQueryExpander,
    QueryExpander,
    _normalize_for_dedup,
    create_expander,
    expand_queries,
)

# ---------------------------------------------------------------------------
# NLPQueryExpander tests
# ---------------------------------------------------------------------------


class TestNLPQueryExpander(unittest.TestCase):
    """Test NLP-based query expansion."""

    def setUp(self):
        self.expander = NLPQueryExpander()

    def test_original_always_first(self):
        """The original query should always be the first result."""
        result = self.expander.expand("find database errors")
        self.assertEqual(result[0], "find database errors")

    def test_returns_list(self):
        """expand() should always return a list of strings."""
        result = self.expander.expand("test query")
        self.assertIsInstance(result, list)
        for item in result:
            self.assertIsInstance(item, str)

    def test_max_expansions_respected(self):
        """Should not return more than max_expansions results."""
        result = self.expander.expand("find database errors", max_expansions=2)
        self.assertLessEqual(len(result), 2)

    def test_max_expansions_one_returns_original(self):
        """max_expansions=1 should return only the original."""
        result = self.expander.expand("find database errors", max_expansions=1)
        self.assertEqual(result, ["find database errors"])

    def test_empty_query(self):
        """Empty query should return a single-element list."""
        result = self.expander.expand("")
        self.assertEqual(len(result), 1)

    def test_whitespace_query(self):
        """Whitespace-only query should be handled gracefully."""
        result = self.expander.expand("   ")
        self.assertEqual(len(result), 1)

    def test_synonym_substitution(self):
        """Should generate a synonym-based alternative."""
        result = self.expander.expand("fix the error", max_expansions=3)
        self.assertGreater(len(result), 1)
        # The synonym alternative should differ from the original
        alternatives = result[1:]
        self.assertTrue(
            all(alt != result[0] for alt in alternatives),
            "Alternatives should differ from original",
        )

    def test_synonym_known_word(self):
        """A query with a known synonym should produce a substitution."""
        result = self.expander.expand("remove the database", max_expansions=2)
        self.assertEqual(len(result), 2)
        # "remove" should be substituted with "delete" or "drop"
        alt_lower = result[1].lower()
        self.assertTrue(
            "delete" in alt_lower or "drop" in alt_lower,
            f"Expected synonym substitution, got: {result[1]}",
        )

    def test_question_rewrite(self):
        """Question-form queries should get a declarative alternative."""
        result = self.expander.expand("how to deploy the application?", max_expansions=3)
        # Should have at least 2 variants (original + rewrite)
        self.assertGreater(len(result), 1)

    def test_what_is_rewrite(self):
        """'what is X' should rewrite to 'X definition'."""
        result = self.expander.expand("what is authentication?", max_expansions=3)
        has_definition = any("definition" in r.lower() for r in result)
        self.assertTrue(has_definition, f"Expected 'definition' in results: {result}")

    def test_keyword_extraction(self):
        """Multi-word queries with stopwords should extract keywords."""
        result = self.expander.expand(
            "the database is being very slow today",
            max_expansions=3,
        )
        self.assertGreater(len(result), 1)

    def test_no_duplicates(self):
        """Expanded queries should not contain duplicates."""
        result = self.expander.expand("check the database", max_expansions=5)
        normalized = [_normalize_for_dedup(r) for r in result]
        self.assertEqual(len(normalized), len(set(normalized)))

    def test_single_word_query(self):
        """Single-word queries should still work."""
        result = self.expander.expand("error")
        self.assertGreaterEqual(len(result), 1)
        self.assertEqual(result[0], "error")

    def test_unknown_words_no_expansion(self):
        """Queries with no known synonyms should still return original."""
        result = self.expander.expand("xyzzy foobar baz", max_expansions=2)
        self.assertGreaterEqual(len(result), 1)
        self.assertEqual(result[0], "xyzzy foobar baz")

    def test_preserves_casing(self):
        """Synonym substitution should attempt to preserve case."""
        result = self.expander.expand("Fix the problem", max_expansions=2)
        if len(result) > 1:
            # First word should be capitalized if original was
            self.assertTrue(result[1][0].isupper())


class TestNLPSynonymSubstitute(unittest.TestCase):
    """Test the synonym substitution strategy in isolation."""

    def test_first_match_replaced(self):
        expander = NLPQueryExpander()
        alt = expander._synonym_substitute("add a new user")
        self.assertIsNotNone(alt)
        self.assertNotEqual(alt, "add a new user")
        # "add" -> "create" or "insert"
        self.assertTrue(
            alt.lower().startswith("create") or alt.lower().startswith("insert"),
            f"Expected 'create' or 'insert', got: {alt}",
        )

    def test_no_match_returns_none(self):
        expander = NLPQueryExpander()
        alt = expander._synonym_substitute("xyzzy foobar")
        self.assertIsNone(alt)

    def test_trailing_punctuation_preserved(self):
        expander = NLPQueryExpander()
        alt = expander._synonym_substitute("fix? something")
        if alt is not None:
            # The replacement for "fix" should keep the "?"
            words = alt.split()
            self.assertTrue(words[0].endswith("?"), f"Expected trailing '?', got: {words[0]}")


class TestNLPQuestionRewrite(unittest.TestCase):
    """Test question rewriting strategy."""

    def test_how_to(self):
        expander = NLPQueryExpander()
        result = expander._question_rewrite("how to configure SSL?")
        self.assertIsNotNone(result)
        self.assertIn("steps to", result.lower())

    def test_why_does(self):
        expander = NLPQueryExpander()
        result = expander._question_rewrite("why does the server crash?")
        self.assertIsNotNone(result)
        self.assertIn("reason for", result.lower())

    def test_non_question(self):
        expander = NLPQueryExpander()
        result = expander._question_rewrite("database performance")
        self.assertIsNone(result)


class TestNLPKeywordExtraction(unittest.TestCase):
    """Test keyword extraction strategy."""

    def test_removes_stopwords(self):
        expander = NLPQueryExpander()
        result = expander._extract_keywords("the system is not working properly")
        self.assertIsNotNone(result)
        self.assertNotIn("the", result.split())
        self.assertNotIn("is", result.split())

    def test_too_few_keywords_returns_none(self):
        expander = NLPQueryExpander()
        # Single content word -> not enough for meaningful keyword extraction
        result = expander._extract_keywords("the database")
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# LLMQueryExpander tests
# ---------------------------------------------------------------------------


class TestLLMQueryExpander(unittest.TestCase):
    """Test LLM-backed query expansion (without actual API calls)."""

    def test_fallback_on_no_api_key(self):
        """Should fall back to NLP expansion when API key is missing."""
        import os

        # Ensure the env var is not set
        env_key = "ANTHROPIC_API_KEY_TEST_DUMMY_12345"
        old_val = os.environ.pop(env_key, None)
        try:
            expander = LLMQueryExpander(
                config={
                    "api_key_env": env_key,
                    "provider": "anthropic",
                }
            )
            result = expander.expand("find errors", max_expansions=3)
            # Should fall back gracefully and return at least the original
            self.assertGreaterEqual(len(result), 1)
            self.assertEqual(result[0], "find errors")
        finally:
            if old_val is not None:
                os.environ[env_key] = old_val

    def test_empty_query(self):
        """Empty query should be handled by LLM expander."""
        expander = LLMQueryExpander()
        result = expander.expand("")
        self.assertEqual(len(result), 1)

    def test_unsupported_provider(self):
        """Unsupported provider should fall back to NLP."""
        import os

        os.environ["TEST_DUMMY_KEY_QE_99"] = "test-key"
        try:
            expander = LLMQueryExpander(
                config={
                    "provider": "unsupported_provider",
                    "api_key_env": "TEST_DUMMY_KEY_QE_99",
                }
            )
            result = expander.expand("test query", max_expansions=2)
            # Should fall back to NLP
            self.assertGreaterEqual(len(result), 1)
            self.assertEqual(result[0], "test query")
        finally:
            os.environ.pop("TEST_DUMMY_KEY_QE_99", None)


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance(unittest.TestCase):
    """Verify that expanders conform to the QueryExpander protocol."""

    def test_nlp_is_query_expander(self):
        self.assertIsInstance(NLPQueryExpander(), QueryExpander)

    def test_llm_is_query_expander(self):
        self.assertIsInstance(LLMQueryExpander(), QueryExpander)


# ---------------------------------------------------------------------------
# Factory / convenience API
# ---------------------------------------------------------------------------


class TestCreateExpander(unittest.TestCase):
    """Test create_expander factory function."""

    def test_default_returns_nlp(self):
        expander = create_expander()
        self.assertIsInstance(expander, NLPQueryExpander)

    def test_none_config_returns_nlp(self):
        expander = create_expander(None)
        self.assertIsInstance(expander, NLPQueryExpander)

    def test_llm_disabled_returns_nlp(self):
        expander = create_expander({"llm": {"enabled": False}})
        self.assertIsInstance(expander, NLPQueryExpander)

    def test_llm_enabled_returns_llm(self):
        expander = create_expander({"llm": {"enabled": True}})
        self.assertIsInstance(expander, LLMQueryExpander)

    def test_no_llm_key_returns_nlp(self):
        expander = create_expander({"max_expansions": 5})
        self.assertIsInstance(expander, NLPQueryExpander)


class TestExpandQueries(unittest.TestCase):
    """Test expand_queries convenience function."""

    def test_default_expansion(self):
        result = expand_queries("fix the error")
        self.assertIsInstance(result, list)
        self.assertGreaterEqual(len(result), 1)
        self.assertEqual(result[0], "fix the error")

    def test_config_max_expansions(self):
        result = expand_queries("find bugs", config={"max_expansions": 2})
        self.assertLessEqual(len(result), 2)

    def test_config_none(self):
        result = expand_queries("test query", config=None)
        self.assertGreaterEqual(len(result), 1)


# ---------------------------------------------------------------------------
# Normalization helper
# ---------------------------------------------------------------------------


class TestNormalizeForDedup(unittest.TestCase):
    """Test the dedup normalization helper."""

    def test_case_insensitive(self):
        self.assertEqual(
            _normalize_for_dedup("Hello World"),
            _normalize_for_dedup("hello world"),
        )

    def test_whitespace_normalized(self):
        self.assertEqual(
            _normalize_for_dedup("hello   world"),
            _normalize_for_dedup("hello world"),
        )

    def test_strips_edges(self):
        self.assertEqual(
            _normalize_for_dedup("  hello  "),
            _normalize_for_dedup("hello"),
        )


# ---------------------------------------------------------------------------
# Integration with HybridBackend
# ---------------------------------------------------------------------------


class TestHybridQueryExpansion(unittest.TestCase):
    """Test query expansion integration in HybridBackend."""

    def test_expansion_disabled_by_default(self):
        """Query expansion should be off when not configured."""
        from mind_mem.hybrid_recall import HybridBackend

        hb = HybridBackend()
        self.assertFalse(hb._query_expansion_enabled)

    def test_expansion_enabled_via_config(self):
        """Query expansion should activate when config enables it."""
        from mind_mem.hybrid_recall import HybridBackend

        hb = HybridBackend(
            config={
                "query_expansion": {"enabled": True, "max_expansions": 3},
            }
        )
        self.assertTrue(hb._query_expansion_enabled)

    def test_expansion_config_not_dict_fallback(self):
        """Non-dict query_expansion config should be handled gracefully."""
        from mind_mem.hybrid_recall import HybridBackend

        hb = HybridBackend(config={"query_expansion": "invalid"})
        self.assertFalse(hb._query_expansion_enabled)

    def test_search_expanded_fuses_results(self):
        """_search_expanded should fuse results from multiple query variants."""
        import threading

        from mind_mem.hybrid_recall import HybridBackend

        hb = HybridBackend(config={"vector_enabled": False})
        call_log: list[str] = []
        lock = threading.Lock()

        def mock_bm25(query, workspace, **kw):
            with lock:
                call_log.append(query)
            # Return different results per query to verify fusion
            if "error" in query.lower():
                return [{"_id": "err-1", "score": 5.0}, {"_id": "shared", "score": 3.0}]
            return [{"_id": "alt-1", "score": 4.0}, {"_id": "shared", "score": 2.0}]

        hb._bm25_search = mock_bm25

        result = hb._search_expanded(
            queries=["find error", "locate exception"],
            workspace="/tmp/test",
            limit=10,
        )

        # Both queries should have been searched
        self.assertEqual(len(call_log), 2)
        # "shared" should rank high (appears in both lists)
        ids = [r["_id"] for r in result]
        self.assertIn("shared", ids)
        # All unique IDs should be present
        self.assertIn("err-1", ids)
        self.assertIn("alt-1", ids)
        # Results should have RRF scores
        for r in result:
            self.assertIn("rrf_score", r)


if __name__ == "__main__":
    unittest.main()
