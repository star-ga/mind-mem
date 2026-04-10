#!/usr/bin/env python3
"""Tests for dedup.py -- 4-layer deduplication filter."""

import unittest

from mind_mem.dedup import (
    DedupConfig,
    _cosine_similarity,
    _extract_source_key,
    _get_result_text,
    _get_result_type,
    _term_vector,
    _text_tokens,
    deduplicate_results,
    layer_best_per_source,
    layer_cosine_dedup,
    layer_source_chunk_cap,
    layer_type_diversity_cap,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(
    id_: str = "D-001",
    score: float = 1.0,
    excerpt: str = "test content",
    file: str = "memory.md",
    type_: str = "decision",
    tags: str = "",
    line: int = 1,
) -> dict:
    """Create a minimal result dict matching the recall output shape."""
    return {
        "_id": id_,
        "score": score,
        "excerpt": excerpt,
        "file": file,
        "type": type_,
        "tags": tags,
        "line": line,
        "status": "active",
    }


# ---------------------------------------------------------------------------
# DedupConfig tests
# ---------------------------------------------------------------------------


class TestDedupConfig(unittest.TestCase):
    """Test DedupConfig initialization and validation."""

    def test_defaults(self):
        cfg = DedupConfig()
        self.assertTrue(cfg.enabled)
        self.assertTrue(cfg.best_per_source)
        self.assertTrue(cfg.cosine_enabled)
        self.assertAlmostEqual(cfg.cosine_threshold, 0.85)
        self.assertTrue(cfg.type_cap_enabled)
        self.assertEqual(cfg.type_cap, 3)
        self.assertTrue(cfg.source_cap_enabled)
        self.assertEqual(cfg.source_cap, 5)

    def test_custom_values(self):
        cfg = DedupConfig({
            "enabled": False,
            "best_per_source": False,
            "cosine_enabled": False,
            "cosine_threshold": 0.9,
            "type_cap_enabled": False,
            "type_cap": 5,
            "source_cap_enabled": False,
            "source_cap": 10,
        })
        self.assertFalse(cfg.enabled)
        self.assertFalse(cfg.best_per_source)
        self.assertFalse(cfg.cosine_enabled)
        self.assertAlmostEqual(cfg.cosine_threshold, 0.9)
        self.assertEqual(cfg.type_cap, 5)
        self.assertEqual(cfg.source_cap, 10)

    def test_threshold_clamping(self):
        cfg = DedupConfig({"cosine_threshold": 1.5})
        self.assertAlmostEqual(cfg.cosine_threshold, 1.0)

        cfg = DedupConfig({"cosine_threshold": -0.5})
        self.assertAlmostEqual(cfg.cosine_threshold, 0.0)

    def test_cap_minimum(self):
        cfg = DedupConfig({"type_cap": 0, "source_cap": -1})
        self.assertEqual(cfg.type_cap, 1)
        self.assertEqual(cfg.source_cap, 1)

    def test_none_config(self):
        cfg = DedupConfig(None)
        self.assertTrue(cfg.enabled)

    def test_from_recall_config(self):
        recall_cfg = {"dedup": {"cosine_threshold": 0.7, "type_cap": 4}}
        cfg = DedupConfig.from_recall_config(recall_cfg)
        self.assertAlmostEqual(cfg.cosine_threshold, 0.7)
        self.assertEqual(cfg.type_cap, 4)

    def test_from_recall_config_missing(self):
        cfg = DedupConfig.from_recall_config({})
        self.assertTrue(cfg.enabled)  # defaults

    def test_from_recall_config_non_dict(self):
        cfg = DedupConfig.from_recall_config({"dedup": "invalid"})
        self.assertTrue(cfg.enabled)  # defaults


# ---------------------------------------------------------------------------
# Text tokenization tests
# ---------------------------------------------------------------------------


class TestTextTokens(unittest.TestCase):
    """Test text tokenization helper."""

    def test_basic_tokenization(self):
        tokens = _text_tokens("The quick brown fox jumps over the lazy dog")
        self.assertIn("quick", tokens)
        self.assertIn("brown", tokens)
        self.assertNotIn("the", tokens)  # stop word
        self.assertNotIn("a", tokens)  # stop word

    def test_empty_string(self):
        self.assertEqual(_text_tokens(""), [])

    def test_single_char_filtered(self):
        tokens = _text_tokens("a b c dd ee")
        self.assertNotIn("a", tokens)
        self.assertNotIn("b", tokens)
        self.assertIn("dd", tokens)
        self.assertIn("ee", tokens)


# ---------------------------------------------------------------------------
# Layer 1: Best chunk per source
# ---------------------------------------------------------------------------


class TestLayerBestPerSource(unittest.TestCase):
    """Test layer 1: best chunk per source block."""

    def test_keeps_best_chunk(self):
        results = [
            _make_result(id_="D-001.0", score=5.0),
            _make_result(id_="D-001.1", score=8.0),
            _make_result(id_="D-001.2", score=3.0),
        ]
        filtered = layer_best_per_source(results)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["_id"], "D-001.1")

    def test_different_sources_kept(self):
        results = [
            _make_result(id_="D-001.0", score=5.0),
            _make_result(id_="D-002.0", score=4.0),
            _make_result(id_="D-003", score=3.0),
        ]
        filtered = layer_best_per_source(results)
        self.assertEqual(len(filtered), 3)

    def test_non_chunked_ids_preserved(self):
        results = [
            _make_result(id_="FACT-001", score=5.0),
            _make_result(id_="FACT-002", score=4.0),
        ]
        filtered = layer_best_per_source(results)
        self.assertEqual(len(filtered), 2)

    def test_empty_input(self):
        self.assertEqual(layer_best_per_source([]), [])

    def test_preserves_score_order(self):
        results = [
            _make_result(id_="D-001.0", score=10.0),
            _make_result(id_="D-002.0", score=8.0),
            _make_result(id_="D-001.1", score=6.0),  # lower than D-001.0
            _make_result(id_="D-002.1", score=4.0),  # lower than D-002.0
        ]
        filtered = layer_best_per_source(results)
        self.assertEqual(len(filtered), 2)
        self.assertEqual(filtered[0]["_id"], "D-001.0")
        self.assertEqual(filtered[1]["_id"], "D-002.0")


class TestExtractSourceKey(unittest.TestCase):
    """Test source key extraction."""

    def test_chunked_id(self):
        self.assertEqual(_extract_source_key({"_id": "D-001.0"}), "D-001")
        self.assertEqual(_extract_source_key({"_id": "D-001.5"}), "D-001")

    def test_non_chunked_id(self):
        self.assertEqual(_extract_source_key({"_id": "FACT-001"}), "FACT-001")
        self.assertEqual(_extract_source_key({"_id": "D-20260222-001"}), "D-20260222-001")

    def test_multi_dot_id(self):
        # Only the last .N is stripped
        self.assertEqual(_extract_source_key({"_id": "D-1.2.3"}), "D-1.2")

    def test_dot_non_numeric_preserved(self):
        self.assertEqual(_extract_source_key({"_id": "file.md"}), "file.md")

    def test_empty_id(self):
        self.assertEqual(_extract_source_key({"_id": ""}), "")
        self.assertEqual(_extract_source_key({}), "")


# ---------------------------------------------------------------------------
# Layer 2: Cosine similarity dedup
# ---------------------------------------------------------------------------


class TestLayerCosineDedup(unittest.TestCase):
    """Test layer 2: cosine similarity dedup."""

    def test_identical_texts_deduped(self):
        results = [
            _make_result(id_="A", score=5.0, excerpt="the quick brown fox jumps over"),
            _make_result(id_="B", score=4.0, excerpt="the quick brown fox jumps over"),
        ]
        filtered = layer_cosine_dedup(results, threshold=0.85)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["_id"], "A")  # higher score kept

    def test_different_texts_kept(self):
        results = [
            _make_result(id_="A", score=5.0, excerpt="machine learning algorithms for classification"),
            _make_result(id_="B", score=4.0, excerpt="database schema migration and versioning"),
        ]
        filtered = layer_cosine_dedup(results, threshold=0.85)
        self.assertEqual(len(filtered), 2)

    def test_near_duplicate_threshold(self):
        # Texts with high overlap
        results = [
            _make_result(id_="A", score=5.0, excerpt="authentication token expires after twenty four hours"),
            _make_result(id_="B", score=4.0, excerpt="authentication token expires after twelve hours period"),
        ]
        filtered_strict = layer_cosine_dedup(results, threshold=0.5)
        filtered_loose = layer_cosine_dedup(results, threshold=0.99)
        # Strict threshold should remove more
        self.assertLessEqual(len(filtered_strict), len(filtered_loose))

    def test_zero_threshold_keeps_all(self):
        results = [
            _make_result(id_="A", score=5.0, excerpt="same text here"),
            _make_result(id_="B", score=4.0, excerpt="same text here"),
        ]
        filtered = layer_cosine_dedup(results, threshold=0.0)
        self.assertEqual(len(filtered), 2)

    def test_over_one_threshold_keeps_all(self):
        results = [
            _make_result(id_="A", score=5.0, excerpt="same text here"),
            _make_result(id_="B", score=4.0, excerpt="same text here"),
        ]
        filtered = layer_cosine_dedup(results, threshold=1.5)
        self.assertEqual(len(filtered), 2)

    def test_empty_input(self):
        self.assertEqual(layer_cosine_dedup([], threshold=0.85), [])

    def test_single_item(self):
        results = [_make_result(id_="A", score=5.0)]
        self.assertEqual(len(layer_cosine_dedup(results)), 1)


class TestCosineSimilarity(unittest.TestCase):
    """Test the cosine similarity computation."""

    def test_identical_vectors(self):
        vec = {"hello": 2, "world": 1}
        self.assertAlmostEqual(_cosine_similarity(vec, vec), 1.0, places=5)

    def test_orthogonal_vectors(self):
        vec_a = {"hello": 1}
        vec_b = {"world": 1}
        self.assertAlmostEqual(_cosine_similarity(vec_a, vec_b), 0.0)

    def test_empty_vectors(self):
        self.assertAlmostEqual(_cosine_similarity({}, {"hello": 1}), 0.0)
        self.assertAlmostEqual(_cosine_similarity({}, {}), 0.0)

    def test_partial_overlap(self):
        vec_a = {"hello": 1, "world": 1}
        vec_b = {"hello": 1, "foo": 1}
        sim = _cosine_similarity(vec_a, vec_b)
        self.assertGreater(sim, 0.0)
        self.assertLess(sim, 1.0)

    def test_symmetry(self):
        vec_a = {"a": 3, "b": 2, "c": 1}
        vec_b = {"a": 1, "c": 4, "d": 2}
        self.assertAlmostEqual(
            _cosine_similarity(vec_a, vec_b),
            _cosine_similarity(vec_b, vec_a),
            places=10,
        )


class TestTermVector(unittest.TestCase):
    """Test term frequency vector builder."""

    def test_basic(self):
        vec = _term_vector(["hello", "world", "hello"])
        self.assertEqual(vec["hello"], 2)
        self.assertEqual(vec["world"], 1)

    def test_empty(self):
        self.assertEqual(_term_vector([]), {})


# ---------------------------------------------------------------------------
# Layer 3: Type diversity cap
# ---------------------------------------------------------------------------


class TestLayerTypeDiversityCap(unittest.TestCase):
    """Test layer 3: type diversity cap."""

    def test_caps_per_type(self):
        results = [
            _make_result(id_=f"DIA-{i}", score=10.0 - i, type_="dialog")
            for i in range(10)
        ]
        filtered = layer_type_diversity_cap(results, cap=3)
        self.assertEqual(len(filtered), 3)

    def test_different_types_unaffected(self):
        results = [
            _make_result(id_="DIA-1", score=10.0, type_="dialog"),
            _make_result(id_="FACT-1", score=9.0, type_="fact"),
            _make_result(id_="DEC-1", score=8.0, type_="decision"),
        ]
        filtered = layer_type_diversity_cap(results, cap=3)
        self.assertEqual(len(filtered), 3)

    def test_mixed_types_capped(self):
        results = [
            _make_result(id_="DIA-1", score=10.0, type_="dialog"),
            _make_result(id_="DIA-2", score=9.0, type_="dialog"),
            _make_result(id_="DIA-3", score=8.0, type_="dialog"),
            _make_result(id_="DIA-4", score=7.0, type_="dialog"),
            _make_result(id_="FACT-1", score=6.0, type_="fact"),
            _make_result(id_="FACT-2", score=5.0, type_="fact"),
        ]
        filtered = layer_type_diversity_cap(results, cap=2)
        dialog_count = sum(1 for r in filtered if r["type"] == "dialog")
        fact_count = sum(1 for r in filtered if r["type"] == "fact")
        self.assertEqual(dialog_count, 2)
        self.assertEqual(fact_count, 2)

    def test_cap_of_one(self):
        results = [
            _make_result(id_="DIA-1", score=10.0, type_="dialog"),
            _make_result(id_="DIA-2", score=9.0, type_="dialog"),
        ]
        filtered = layer_type_diversity_cap(results, cap=1)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["_id"], "DIA-1")

    def test_cap_zero_coerced_to_one(self):
        results = [
            _make_result(id_="DIA-1", score=10.0, type_="dialog"),
            _make_result(id_="DIA-2", score=9.0, type_="dialog"),
        ]
        filtered = layer_type_diversity_cap(results, cap=0)
        self.assertEqual(len(filtered), 1)

    def test_empty_input(self):
        self.assertEqual(layer_type_diversity_cap([], cap=3), [])

    def test_preserves_order(self):
        results = [
            _make_result(id_="DIA-1", score=10.0, type_="dialog"),
            _make_result(id_="FACT-1", score=9.0, type_="fact"),
            _make_result(id_="DIA-2", score=8.0, type_="dialog"),
        ]
        filtered = layer_type_diversity_cap(results, cap=2)
        self.assertEqual(len(filtered), 3)
        self.assertEqual(filtered[0]["_id"], "DIA-1")
        self.assertEqual(filtered[1]["_id"], "FACT-1")
        self.assertEqual(filtered[2]["_id"], "DIA-2")


class TestGetResultType(unittest.TestCase):
    """Test type extraction helper."""

    def test_explicit_type(self):
        self.assertEqual(_get_result_type({"type": "dialog"}), "dialog")

    def test_from_id_prefix(self):
        self.assertEqual(_get_result_type({"_id": "DIA-001", "type": ""}), "DIA")
        self.assertEqual(_get_result_type({"_id": "FACT-001", "type": ""}), "FACT")

    def test_unknown_fallback(self):
        self.assertEqual(_get_result_type({"_id": "something", "type": ""}), "unknown")

    def test_empty(self):
        self.assertEqual(_get_result_type({}), "unknown")


# ---------------------------------------------------------------------------
# Layer 4: Per-source chunk cap
# ---------------------------------------------------------------------------


class TestLayerSourceChunkCap(unittest.TestCase):
    """Test layer 4: per-source file chunk cap."""

    def test_caps_per_file(self):
        results = [
            _make_result(id_=f"D-{i}", score=10.0 - i, file="memory.md")
            for i in range(10)
        ]
        filtered = layer_source_chunk_cap(results, cap=3)
        self.assertEqual(len(filtered), 3)

    def test_different_files_unaffected(self):
        results = [
            _make_result(id_="D-1", score=10.0, file="a.md"),
            _make_result(id_="D-2", score=9.0, file="b.md"),
            _make_result(id_="D-3", score=8.0, file="c.md"),
        ]
        filtered = layer_source_chunk_cap(results, cap=1)
        self.assertEqual(len(filtered), 3)

    def test_mixed_files_capped(self):
        results = [
            _make_result(id_="D-1", score=10.0, file="a.md"),
            _make_result(id_="D-2", score=9.0, file="a.md"),
            _make_result(id_="D-3", score=8.0, file="a.md"),
            _make_result(id_="D-4", score=7.0, file="b.md"),
            _make_result(id_="D-5", score=6.0, file="b.md"),
        ]
        filtered = layer_source_chunk_cap(results, cap=2)
        a_count = sum(1 for r in filtered if r["file"] == "a.md")
        b_count = sum(1 for r in filtered if r["file"] == "b.md")
        self.assertEqual(a_count, 2)
        self.assertEqual(b_count, 2)

    def test_empty_input(self):
        self.assertEqual(layer_source_chunk_cap([], cap=5), [])

    def test_cap_of_one(self):
        results = [
            _make_result(id_="D-1", score=10.0, file="a.md"),
            _make_result(id_="D-2", score=9.0, file="a.md"),
        ]
        filtered = layer_source_chunk_cap(results, cap=1)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["_id"], "D-1")

    def test_preserves_order(self):
        results = [
            _make_result(id_="D-1", score=10.0, file="a.md"),
            _make_result(id_="D-2", score=9.0, file="b.md"),
            _make_result(id_="D-3", score=8.0, file="a.md"),
        ]
        filtered = layer_source_chunk_cap(results, cap=5)
        self.assertEqual(filtered[0]["_id"], "D-1")
        self.assertEqual(filtered[1]["_id"], "D-2")
        self.assertEqual(filtered[2]["_id"], "D-3")


# ---------------------------------------------------------------------------
# Full pipeline tests
# ---------------------------------------------------------------------------


class TestDeduplicateResults(unittest.TestCase):
    """Test the full 4-layer pipeline."""

    def test_empty_input(self):
        self.assertEqual(deduplicate_results([]), [])

    def test_disabled_returns_unchanged(self):
        cfg = DedupConfig({"enabled": False})
        results = [
            _make_result(id_="D-001.0", score=5.0),
            _make_result(id_="D-001.1", score=4.0),
        ]
        filtered = deduplicate_results(results, config=cfg)
        self.assertEqual(len(filtered), 2)

    def test_all_layers_applied(self):
        """Build a result set that each layer should reduce."""
        results = [
            # Chunk dedup: D-001.0 and D-001.1 should collapse to D-001.0
            _make_result(id_="D-001.0", score=10.0, excerpt="unique text alpha", file="a.md", type_="decision"),
            _make_result(id_="D-001.1", score=9.0, excerpt="unique text beta", file="a.md", type_="decision"),
            # Cosine dedup: nearly identical texts
            _make_result(id_="D-002", score=8.0, excerpt="database migration schema versioning tool", file="b.md", type_="decision"),
            _make_result(id_="D-003", score=7.0, excerpt="database migration schema versioning tool update", file="c.md", type_="decision"),
            # Different type to test diversity cap
            _make_result(id_="FACT-001", score=6.0, excerpt="completely different content here about testing", file="d.md", type_="fact"),
        ]
        cfg = DedupConfig({
            "cosine_threshold": 0.85,
            "type_cap": 3,
            "source_cap": 5,
        })
        filtered = deduplicate_results(results, config=cfg)
        # Layer 1 removes D-001.1 (chunk dedup)
        # Layer 2 may remove D-003 (cosine sim with D-002)
        # Total should be reduced
        self.assertLess(len(filtered), len(results))

    def test_default_config(self):
        results = [_make_result(id_="D-001", score=5.0)]
        filtered = deduplicate_results(results)
        self.assertEqual(len(filtered), 1)

    def test_individual_layers_toggled(self):
        results = [
            _make_result(id_="D-001.0", score=10.0, excerpt="hello world"),
            _make_result(id_="D-001.1", score=9.0, excerpt="hello world"),
        ]

        # Only best_per_source enabled
        cfg = DedupConfig({
            "best_per_source": True,
            "cosine_enabled": False,
            "type_cap_enabled": False,
            "source_cap_enabled": False,
        })
        filtered = deduplicate_results(results, config=cfg)
        self.assertEqual(len(filtered), 1)

        # Only cosine enabled
        cfg = DedupConfig({
            "best_per_source": False,
            "cosine_enabled": True,
            "cosine_threshold": 0.85,
            "type_cap_enabled": False,
            "source_cap_enabled": False,
        })
        filtered = deduplicate_results(results, config=cfg)
        self.assertEqual(len(filtered), 1)

    def test_pipeline_preserves_score_ordering(self):
        results = [
            _make_result(id_="A", score=10.0, excerpt="alpha content unique", file="a.md", type_="fact"),
            _make_result(id_="B", score=8.0, excerpt="beta content different", file="b.md", type_="dialog"),
            _make_result(id_="C", score=6.0, excerpt="gamma content another", file="c.md", type_="decision"),
        ]
        filtered = deduplicate_results(results)
        for i in range(len(filtered) - 1):
            self.assertGreaterEqual(filtered[i]["score"], filtered[i + 1]["score"])

    def test_large_result_set(self):
        """Pipeline handles 200+ items without error."""
        results = [
            _make_result(
                id_=f"D-{i}",
                score=200.0 - i,
                excerpt=f"document number {i} with unique content about topic {i % 20}",
                file=f"file{i % 10}.md",
                type_="dialog" if i % 2 == 0 else "fact",
            )
            for i in range(200)
        ]
        cfg = DedupConfig({"type_cap": 50, "source_cap": 30})
        filtered = deduplicate_results(results, config=cfg)
        self.assertGreater(len(filtered), 0)
        self.assertLessEqual(len(filtered), len(results))


# ---------------------------------------------------------------------------
# GetResultText helper
# ---------------------------------------------------------------------------


class TestGetResultText(unittest.TestCase):
    """Test text extraction from result dicts."""

    def test_excerpt_only(self):
        text = _get_result_text({"excerpt": "hello world"})
        self.assertIn("hello world", text)

    def test_multiple_fields(self):
        text = _get_result_text({
            "excerpt": "primary",
            "content": "secondary",
            "tags": "tag1,tag2",
        })
        self.assertIn("primary", text)
        self.assertIn("secondary", text)
        self.assertIn("tag1", text)

    def test_empty_result(self):
        text = _get_result_text({})
        self.assertEqual(text, "")


if __name__ == "__main__":
    unittest.main()
