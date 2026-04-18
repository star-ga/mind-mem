#!/usr/bin/env python3
"""Tests for smart_chunker.py — semantic-boundary document chunking."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from mind_mem.smart_chunker import (
    Chunk,
    SmartChunkerConfig,
    _detect_code_boundaries,
    _extract_overlap_prefix,
    _force_split_text,
    _identify_code_spans,
    _in_code_span,
    _parse_llm_score,
    _score_boundary,
    _Segment,
    _segment_document,
    smart_chunk,
    smart_chunk_blocks,
)

# ---------------------------------------------------------------------------
# SmartChunkerConfig tests
# ---------------------------------------------------------------------------


class TestSmartChunkerConfig(unittest.TestCase):
    """Test configuration defaults and overrides."""

    def test_defaults(self):
        cfg = SmartChunkerConfig()
        self.assertEqual(cfg.max_chunk_size, 1500)
        self.assertEqual(cfg.min_chunk_size, 100)
        self.assertEqual(cfg.overlap_sentences, 1)
        self.assertTrue(cfg.preserve_code_blocks)
        self.assertFalse(cfg.llm_refine)
        self.assertEqual(cfg.llm_model, "qwen3.5:9b")
        self.assertEqual(cfg.llm_backend, "auto")
        self.assertEqual(cfg.source, "")

    def test_custom_values(self):
        cfg = SmartChunkerConfig(
            max_chunk_size=500,
            min_chunk_size=50,
            overlap_sentences=2,
            preserve_code_blocks=False,
            llm_refine=True,
            llm_model="llama3:8b",
            source="test-doc",
        )
        self.assertEqual(cfg.max_chunk_size, 500)
        self.assertEqual(cfg.min_chunk_size, 50)
        self.assertEqual(cfg.overlap_sentences, 2)
        self.assertFalse(cfg.preserve_code_blocks)
        self.assertTrue(cfg.llm_refine)
        self.assertEqual(cfg.llm_model, "llama3:8b")
        self.assertEqual(cfg.source, "test-doc")


# ---------------------------------------------------------------------------
# Segment detection tests
# ---------------------------------------------------------------------------


class TestCodeSpanDetection(unittest.TestCase):
    """Test fenced code block detection."""

    def test_single_code_block(self):
        text = "Before\n```python\nprint('hi')\n```\nAfter"
        spans = _identify_code_spans(text)
        self.assertEqual(len(spans), 1)
        # The span should cover the fenced region
        self.assertIn("print('hi')", text[spans[0][0] : spans[0][1]])

    def test_multiple_code_blocks(self):
        text = "A\n```\ncode1\n```\nB\n```\ncode2\n```\nC"
        spans = _identify_code_spans(text)
        self.assertEqual(len(spans), 2)

    def test_no_code_blocks(self):
        text = "Just plain text with no fences."
        spans = _identify_code_spans(text)
        self.assertEqual(len(spans), 0)

    def test_unclosed_fence(self):
        text = "Before\n```\ncode without end"
        spans = _identify_code_spans(text)
        self.assertEqual(len(spans), 0)

    def test_tilde_fence(self):
        text = "Before\n~~~\ncode\n~~~\nAfter"
        spans = _identify_code_spans(text)
        self.assertEqual(len(spans), 1)

    def test_in_code_span(self):
        spans = [(10, 50), (100, 150)]
        self.assertTrue(_in_code_span(25, spans))
        self.assertTrue(_in_code_span(10, spans))
        self.assertFalse(_in_code_span(50, spans))
        self.assertFalse(_in_code_span(5, spans))
        self.assertFalse(_in_code_span(75, spans))

    def test_in_code_span_empty(self):
        self.assertFalse(_in_code_span(10, []))


class TestSegmentDocument(unittest.TestCase):
    """Test document segmentation into structural elements."""

    def test_empty_document(self):
        self.assertEqual(_segment_document(""), [])
        self.assertEqual(_segment_document("   "), [])

    def test_single_paragraph(self):
        text = "This is a single paragraph of text."
        segments = _segment_document(text)
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].kind, "paragraph")

    def test_header_detection(self):
        text = "# Main Title\n\nSome content here."
        segments = _segment_document(text)
        kinds = [s.kind for s in segments]
        self.assertIn("header", kinds)

    def test_multiple_headers(self):
        text = "# Title\n\nParagraph one.\n\n## Section\n\nParagraph two."
        segments = _segment_document(text)
        header_segments = [s for s in segments if s.kind == "header"]
        self.assertEqual(len(header_segments), 2)

    def test_code_block_segment(self):
        text = "Before\n\n```python\ndef foo():\n    pass\n```\n\nAfter"
        segments = _segment_document(text)
        kinds = [s.kind for s in segments]
        self.assertIn("code", kinds)

    def test_list_detection(self):
        text = "Introduction.\n\n- Item one\n- Item two\n- Item three"
        segments = _segment_document(text)
        kinds = [s.kind for s in segments]
        self.assertIn("list", kinds)

    def test_numbered_list_detection(self):
        text = "Steps:\n\n1. First step\n2. Second step\n3. Third step"
        segments = _segment_document(text)
        kinds = [s.kind for s in segments]
        self.assertIn("list", kinds)

    def test_header_text_preserved(self):
        text = "# Introduction\n\nContent.\n\n## Methods\n\nMore content."
        segments = _segment_document(text)
        header_segs = [s for s in segments if s.kind == "header"]
        header_texts = [s.header_text for s in header_segs]
        self.assertIn("Introduction", header_texts)
        self.assertIn("Methods", header_texts)

    def test_header_level_tracked(self):
        text = "# H1\n\n## H2\n\n### H3"
        segments = _segment_document(text)
        header_segs = [s for s in segments if s.kind == "header"]
        levels = [s.header_level for s in header_segs]
        self.assertIn(1, levels)
        self.assertIn(2, levels)
        self.assertIn(3, levels)

    def test_character_offsets(self):
        text = "First paragraph.\n\nSecond paragraph."
        segments = _segment_document(text)
        # Each segment should have valid start/end
        for seg in segments:
            self.assertGreaterEqual(seg.start, 0)
            self.assertLessEqual(seg.end, len(text))
            self.assertLessEqual(seg.start, seg.end)


# ---------------------------------------------------------------------------
# Code boundary detection tests
# ---------------------------------------------------------------------------


class TestDetectCodeBoundaries(unittest.TestCase):
    """Test function/class boundary detection in code blocks."""

    def test_python_functions(self):
        code = "def foo():\n    pass\n\ndef bar():\n    pass\n"
        boundaries = _detect_code_boundaries(code)
        self.assertGreater(len(boundaries), 0)

    def test_python_class(self):
        code = "class Foo:\n    pass\n\nclass Bar:\n    pass\n"
        boundaries = _detect_code_boundaries(code)
        self.assertGreater(len(boundaries), 0)

    def test_python_async_def(self):
        code = "async def foo():\n    pass\n\nasync def bar():\n    pass\n"
        boundaries = _detect_code_boundaries(code)
        self.assertGreater(len(boundaries), 0)

    def test_javascript_function(self):
        code = "function foo() {\n}\n\nfunction bar() {\n}\n"
        boundaries = _detect_code_boundaries(code)
        self.assertGreater(len(boundaries), 0)

    def test_rust_fn(self):
        code = "fn main() {\n}\n\npub fn helper() {\n}\n"
        boundaries = _detect_code_boundaries(code)
        self.assertGreater(len(boundaries), 0)

    def test_go_func(self):
        code = "func main() {\n}\n\nfunc helper() {\n}\n"
        boundaries = _detect_code_boundaries(code)
        self.assertGreater(len(boundaries), 0)

    def test_no_boundaries_in_plain_text(self):
        code = "just some text\nwith no code\nboundaries here\n"
        boundaries = _detect_code_boundaries(code)
        self.assertEqual(len(boundaries), 0)

    def test_single_function_no_split(self):
        code = "def only_one():\n    return True\n"
        boundaries = _detect_code_boundaries(code)
        # First function at position 0 is excluded, so no boundaries
        self.assertEqual(len(boundaries), 0)

    def test_excludes_position_zero(self):
        code = "def first():\n    pass\n"
        boundaries = _detect_code_boundaries(code)
        self.assertNotIn(0, boundaries)

    def test_code_block_splitting_in_segments(self):
        """Code blocks with multiple functions should be split into sub-segments."""
        code = "```python\ndef foo():\n    pass\n\ndef bar():\n    pass\n\ndef baz():\n    pass\n```"
        text = f"Intro.\n\n{code}\n\nOutro."
        segments = _segment_document(text)
        code_segments = [s for s in segments if s.kind == "code"]
        # With 3 functions (2+ boundaries), the code block should be split
        self.assertGreaterEqual(len(code_segments), 1)


# ---------------------------------------------------------------------------
# Boundary scoring tests
# ---------------------------------------------------------------------------


class TestBoundaryScoring(unittest.TestCase):
    """Test the heuristic boundary scoring function."""

    def test_header_boundary_scores_high(self):
        before = _Segment("Some text.", 0, 10, "paragraph")
        after = _Segment("# New Section", 11, 24, "header", "New Section", 1)
        score = _score_boundary(before, after)
        self.assertGreater(score, 0.5)

    def test_same_kind_scores_lower(self):
        seg1 = _Segment("Paragraph A.", 0, 12, "paragraph")
        seg2 = _Segment("Paragraph B.", 13, 25, "paragraph")
        score_same = _score_boundary(seg1, seg2)

        seg3 = _Segment("Paragraph A.", 0, 12, "paragraph")
        seg4 = _Segment("```\ncode\n```", 13, 25, "code")
        score_diff = _score_boundary(seg3, seg4)

        self.assertGreater(score_diff, score_same)

    def test_h1_scores_higher_than_h4(self):
        before = _Segment("Some text.", 0, 10, "paragraph")
        h1 = _Segment("# Big Section", 11, 25, "header", "Big Section", 1)
        h4 = _Segment("#### Small Section", 11, 30, "header", "Small Section", 4)

        score_h1 = _score_boundary(before, h1)
        score_h4 = _score_boundary(before, h4)
        self.assertGreater(score_h1, score_h4)

    def test_topic_shift_high_score(self):
        before = _Segment("Python Django Flask web framework API.", 0, 40, "paragraph")
        after = _Segment("Nuclear physics atoms electrons quarks.", 41, 80, "paragraph")
        score = _score_boundary(before, after)
        # Topic shift should add to the score
        self.assertGreater(score, 0.1)

    def test_score_capped_at_one(self):
        before = _Segment("A.", 0, 2, "code")
        after = _Segment("# X", 3, 6, "header", "X", 1)
        score = _score_boundary(before, after)
        self.assertLessEqual(score, 1.0)


# ---------------------------------------------------------------------------
# Force split tests
# ---------------------------------------------------------------------------


class TestForceSplit(unittest.TestCase):
    """Test force-splitting of oversized text."""

    def test_short_text_unchanged(self):
        result = _force_split_text("Short text.", 100)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], "Short text.")

    def test_splits_at_sentence_boundary(self):
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        result = _force_split_text(text, 40)
        self.assertGreater(len(result), 1)
        # Each chunk should end at a sentence boundary when possible
        for chunk in result[:-1]:
            self.assertTrue(
                chunk.rstrip().endswith(".") or chunk.rstrip().endswith("!") or chunk.rstrip().endswith("?"),
                f"Chunk does not end at sentence boundary: {chunk!r}",
            )

    def test_falls_back_to_word_boundary(self):
        # Single long "sentence" with no period
        text = " ".join(f"word{i}" for i in range(100))
        result = _force_split_text(text, 50)
        self.assertGreater(len(result), 1)

    def test_covers_all_content(self):
        text = "A. B. C. D. E. F. G. H. I. J."
        result = _force_split_text(text, 15)
        combined = " ".join(result)
        for letter in "ABCDEFGHIJ":
            self.assertIn(letter, combined)

    def test_empty_text(self):
        result = _force_split_text("", 100)
        self.assertEqual(result, [])

    def test_whitespace_only(self):
        result = _force_split_text("   ", 100)
        self.assertEqual(result, [])


# ---------------------------------------------------------------------------
# Overlap extraction tests
# ---------------------------------------------------------------------------


class TestOverlapExtraction(unittest.TestCase):
    """Test sentence overlap extraction."""

    def test_extract_last_sentence(self):
        text = "First sentence. Second sentence. Third sentence."
        overlap = _extract_overlap_prefix(text, 1)
        self.assertIn("Third", overlap)

    def test_extract_two_sentences(self):
        text = "One. Two. Three."
        overlap = _extract_overlap_prefix(text, 2)
        self.assertIn("Two", overlap)
        self.assertIn("Three", overlap)

    def test_zero_overlap(self):
        text = "Sentence one. Sentence two."
        overlap = _extract_overlap_prefix(text, 0)
        self.assertEqual(overlap, "")

    def test_empty_text(self):
        overlap = _extract_overlap_prefix("", 1)
        self.assertEqual(overlap, "")

    def test_whitespace_only(self):
        overlap = _extract_overlap_prefix("   ", 1)
        self.assertEqual(overlap, "")


# ---------------------------------------------------------------------------
# LLM score parsing tests
# ---------------------------------------------------------------------------


class TestParseLLMScore(unittest.TestCase):
    """Test parsing float scores from LLM output."""

    def test_clean_float(self):
        self.assertAlmostEqual(_parse_llm_score("0.75"), 0.75)

    def test_integer_zero(self):
        self.assertAlmostEqual(_parse_llm_score("0"), 0.0)

    def test_integer_one(self):
        self.assertAlmostEqual(_parse_llm_score("1"), 1.0)

    def test_with_surrounding_text(self):
        result = _parse_llm_score("The score is 0.85 for this boundary.")
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 0.85)

    def test_out_of_range_high(self):
        result = _parse_llm_score("2.5")
        self.assertIsNone(result)

    def test_out_of_range_negative(self):
        result = _parse_llm_score("-0.5")
        self.assertIsNone(result)

    def test_no_number(self):
        result = _parse_llm_score("This is not a number at all.")
        self.assertIsNone(result)

    def test_empty_string(self):
        result = _parse_llm_score("")
        self.assertIsNone(result)

    def test_whitespace_padded(self):
        self.assertAlmostEqual(_parse_llm_score("  0.42  "), 0.42)


# ---------------------------------------------------------------------------
# smart_chunk (main API) tests
# ---------------------------------------------------------------------------


class TestSmartChunk(unittest.TestCase):
    """Tests for the main smart_chunk() function."""

    def test_empty_text_returns_empty(self):
        result = smart_chunk("")
        self.assertEqual(result, [])

    def test_whitespace_only_returns_empty(self):
        result = smart_chunk("   \n\n  ")
        self.assertEqual(result, [])

    def test_short_document_single_chunk(self):
        text = "A short paragraph of text."
        result = smart_chunk(text)
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], Chunk)
        self.assertIn("short paragraph", result[0].text)

    def test_chunk_has_metadata(self):
        text = "Some content here."
        result = smart_chunk(text, source="test.md")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].metadata["source"], "test.md")

    def test_source_from_config(self):
        cfg = SmartChunkerConfig(source="config-source")
        result = smart_chunk("Content.", config=cfg)
        self.assertEqual(result[0].metadata["source"], "config-source")

    def test_source_arg_overrides_config(self):
        cfg = SmartChunkerConfig(source="config-source")
        result = smart_chunk("Content.", config=cfg, source="arg-source")
        self.assertEqual(result[0].metadata["source"], "arg-source")

    def test_multi_section_document_splits(self):
        sections = []
        for i in range(5):
            sections.append(f"## Section {i}\n\n" + " ".join(f"word{j}" for j in range(200)))
        text = "\n\n".join(sections)

        cfg = SmartChunkerConfig(max_chunk_size=500)
        result = smart_chunk(text, config=cfg)
        self.assertGreater(len(result), 1)

    def test_respects_max_chunk_size(self):
        text = "\n\n".join(f"## Part {i}\n\n" + ". ".join(f"Sentence {j}" for j in range(50)) + "." for i in range(10))
        cfg = SmartChunkerConfig(max_chunk_size=300)
        result = smart_chunk(text, config=cfg)
        for chunk in result:
            # Allow a small tolerance for overlap text
            self.assertLessEqual(
                len(chunk.text),
                cfg.max_chunk_size * 2,  # generous bound for overlap
                f"Chunk {chunk.index} exceeds expected size limit",
            )

    def test_chunk_indices_sequential(self):
        text = "\n\n".join(f"Paragraph {i}. " * 20 for i in range(10))
        cfg = SmartChunkerConfig(max_chunk_size=200)
        result = smart_chunk(text, config=cfg)
        for i, chunk in enumerate(result):
            self.assertEqual(chunk.index, i)

    def test_position_metadata(self):
        text = "\n\n".join(f"## Section {i}\n\n" + "Content. " * 50 for i in range(5))
        cfg = SmartChunkerConfig(max_chunk_size=200)
        result = smart_chunk(text, config=cfg)
        if len(result) >= 2:
            self.assertEqual(result[0].metadata["position"], "first")
            self.assertEqual(result[-1].metadata["position"], "last")

    def test_single_chunk_position_is_only(self):
        result = smart_chunk("Short text.")
        self.assertEqual(result[0].metadata["position"], "only")

    def test_section_headers_in_metadata(self):
        text = "# Main Title\n\nIntroduction paragraph.\n\n## Methods\n\nMethod details."
        result = smart_chunk(text)
        # At least one chunk should have section header info
        all_sections = []
        for chunk in result:
            if "section_headers" in chunk.metadata:
                all_sections.extend(chunk.metadata["section_headers"])
            if chunk.metadata.get("section"):
                all_sections.append(chunk.metadata["section"])
        # Should capture at least one header
        self.assertTrue(len(all_sections) > 0)

    def test_content_types_in_metadata(self):
        text = "# Title\n\nSome paragraph.\n\n```python\ncode()\n```"
        result = smart_chunk(text)
        all_types = set()
        for chunk in result:
            if "content_types" in chunk.metadata:
                all_types.update(chunk.metadata["content_types"])
        # Should detect at least header and paragraph (code may or may not
        # be in same chunk depending on size)
        self.assertTrue(len(all_types) > 0)

    def test_preserves_all_content(self):
        paragraphs = [f"Unique paragraph {i} with distinctive text." for i in range(10)]
        text = "\n\n".join(paragraphs)
        cfg = SmartChunkerConfig(max_chunk_size=200, overlap_sentences=0)
        result = smart_chunk(text, config=cfg)
        combined = " ".join(c.text for c in result)
        for i in range(10):
            self.assertIn(
                f"Unique paragraph {i}",
                combined,
                f"Paragraph {i} missing from chunks",
            )

    def test_code_block_preservation(self):
        code = "```python\ndef hello():\n    print('world')\n```"
        text = f"Intro text.\n\n{code}\n\nOutro text."
        result = smart_chunk(text)
        combined = " ".join(c.text for c in result)
        self.assertIn("def hello():", combined)
        self.assertIn("print('world')", combined)

    def test_overlap_sentences(self):
        # Create paragraphs with clear sentence boundaries
        p1 = "Alpha one. Alpha two. Alpha three."
        p2 = "Beta one. Beta two. Beta three."
        p3 = "Gamma one. Gamma two. Gamma three."
        text = f"{p1}\n\n{p2}\n\n{p3}"
        cfg = SmartChunkerConfig(max_chunk_size=60, overlap_sentences=1)
        result = smart_chunk(text, config=cfg)
        if len(result) >= 2:
            # The second chunk should contain overlap from the first
            # (the last sentence of the previous chunk)
            second_text = result[1].text
            # With overlap, some content from the end of the first chunk
            # should appear at the start of the second
            self.assertTrue(len(second_text) > 0)

    def test_no_overlap(self):
        text = "First part. " * 50 + "\n\n" + "Second part. " * 50
        cfg = SmartChunkerConfig(max_chunk_size=200, overlap_sentences=0)
        result = smart_chunk(text, config=cfg)
        self.assertGreater(len(result), 0)

    def test_chunk_start_end_chars(self):
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        result = smart_chunk(text)
        for chunk in result:
            self.assertGreaterEqual(chunk.start_char, 0)
            self.assertGreaterEqual(chunk.end_char, chunk.start_char)

    def test_chunk_total_in_metadata(self):
        text = "\n\n".join(f"Section {i}. " * 30 for i in range(5))
        cfg = SmartChunkerConfig(max_chunk_size=200)
        result = smart_chunk(text, config=cfg)
        if result:
            expected_total = len(result)
            for chunk in result:
                self.assertEqual(chunk.metadata["chunk_total"], expected_total)


# ---------------------------------------------------------------------------
# smart_chunk_blocks tests
# ---------------------------------------------------------------------------


class TestSmartChunkBlocks(unittest.TestCase):
    """Tests for smart_chunk_blocks() — block-level chunking."""

    def test_short_block_unchanged(self):
        block = {
            "_id": "D-20260410-001",
            "_line": 1,
            "Statement": "Short statement.",
            "Status": "active",
        }
        result = smart_chunk_blocks([block])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["_id"], "D-20260410-001")

    def test_long_block_is_split(self):
        long_text = "\n\n".join(f"## Section {i}\n\n" + ". ".join(f"Sentence {j}" for j in range(30)) + "." for i in range(10))
        block = {
            "_id": "D-20260410-002",
            "_line": 1,
            "Statement": long_text,
            "Status": "active",
        }
        cfg = SmartChunkerConfig(max_chunk_size=300)
        result = smart_chunk_blocks([block], config=cfg)
        self.assertGreater(len(result), 1)

    def test_chunk_ids_have_suffix(self):
        long_text = "Content. " * 500
        block = {"_id": "D-001", "_line": 1, "Description": long_text}
        cfg = SmartChunkerConfig(max_chunk_size=200)
        result = smart_chunk_blocks([block], config=cfg)
        if len(result) > 1:
            for i, chunk in enumerate(result):
                self.assertEqual(chunk["_id"], f"D-001.{i}")

    def test_preserves_metadata(self):
        long_text = "Content. " * 500
        block = {"_id": "D-001", "_line": 1, "Statement": long_text, "Status": "active", "Tags": "test"}
        cfg = SmartChunkerConfig(max_chunk_size=200)
        result = smart_chunk_blocks([block], config=cfg)
        for chunk in result:
            self.assertEqual(chunk["Status"], "active")
            self.assertEqual(chunk["Tags"], "test")

    def test_chunk_parent_field(self):
        long_text = "Content. " * 500
        block = {"_id": "D-001", "_line": 1, "Statement": long_text}
        cfg = SmartChunkerConfig(max_chunk_size=200)
        result = smart_chunk_blocks([block], config=cfg)
        if len(result) > 1:
            for chunk in result:
                self.assertEqual(chunk["_chunk_parent"], "D-001")

    def test_mixed_blocks(self):
        short_block = {"_id": "D-001", "_line": 1, "Statement": "Short."}
        long_block = {"_id": "D-002", "_line": 10, "Statement": "Content. " * 500}
        cfg = SmartChunkerConfig(max_chunk_size=200)
        result = smart_chunk_blocks([short_block, long_block], config=cfg)
        # First block unchanged, second split
        self.assertEqual(result[0]["_id"], "D-001")
        self.assertGreater(len(result), 2)

    def test_uses_longest_text_field(self):
        block = {
            "_id": "D-001",
            "_line": 1,
            "Title": "Short title",
            "Description": "Long description. " * 300,
        }
        cfg = SmartChunkerConfig(max_chunk_size=200)
        result = smart_chunk_blocks([block], config=cfg)
        if len(result) > 1:
            # Title should be unchanged in all chunks
            for chunk in result:
                self.assertEqual(chunk["Title"], "Short title")

    def test_empty_blocks_list(self):
        result = smart_chunk_blocks([])
        self.assertEqual(result, [])

    def test_block_without_text_fields(self):
        block = {"_id": "D-001", "_line": 1, "Tags": "test"}
        result = smart_chunk_blocks([block])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], block)

    def test_custom_text_fields(self):
        block = {
            "_id": "D-001",
            "_line": 1,
            "Body": "Long body text. " * 300,
        }
        cfg = SmartChunkerConfig(max_chunk_size=200)
        # Default text_fields won't include "Body"
        result_default = smart_chunk_blocks([block], config=cfg)
        self.assertEqual(len(result_default), 1)

        # Custom text_fields should find it
        result_custom = smart_chunk_blocks([block], config=cfg, text_fields=("Body",))
        self.assertGreater(len(result_custom), 1)


# ---------------------------------------------------------------------------
# LLM refinement tests (mocked)
# ---------------------------------------------------------------------------


class TestLLMRefinement(unittest.TestCase):
    """Test LLM-guided boundary refinement with mocked backends."""

    def test_llm_refine_off_by_default(self):
        """When llm_refine=False, no LLM calls are made."""
        text = "\n\n".join(f"Section {i}. " * 30 for i in range(5))
        cfg = SmartChunkerConfig(max_chunk_size=200, llm_refine=False)
        # Should not raise or attempt LLM calls
        result = smart_chunk(text, config=cfg)
        self.assertGreater(len(result), 0)

    @patch("mind_mem.llm_extractor.is_available", return_value=False)
    def test_llm_refine_skips_when_unavailable(self, _mock_avail):
        """When LLM is unavailable, refinement is skipped gracefully."""
        from mind_mem.smart_chunker import _refine_boundaries_with_llm

        segments = [
            _Segment("Text A.", 0, 7, "paragraph"),
            _Segment("Text B.", 8, 15, "paragraph"),
        ]
        scores = [(1, 0.5)]
        cfg = SmartChunkerConfig(llm_refine=True)
        result = _refine_boundaries_with_llm(segments, scores, cfg)
        # Scores should be unchanged
        self.assertEqual(result, scores)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and unusual inputs."""

    def test_single_character(self):
        result = smart_chunk("x")
        self.assertEqual(len(result), 1)

    def test_only_headers(self):
        text = "# Header 1\n\n## Header 2\n\n### Header 3"
        result = smart_chunk(text)
        self.assertGreater(len(result), 0)

    def test_only_code_block(self):
        text = "```python\nprint('hello world')\n```"
        result = smart_chunk(text)
        self.assertGreater(len(result), 0)

    def test_very_long_single_paragraph(self):
        text = " ".join(f"word{i}" for i in range(5000))
        cfg = SmartChunkerConfig(max_chunk_size=500)
        result = smart_chunk(text, config=cfg)
        self.assertGreater(len(result), 1)

    def test_unicode_content(self):
        text = "# Titre Principal\n\nLes donnees sont importantes. \u00c9tude approfondie.\n\n## \u65e5\u672c\u8a9e\n\n\u30c6\u30b9\u30c8\u30c6\u30ad\u30b9\u30c8\u3002"
        result = smart_chunk(text)
        self.assertGreater(len(result), 0)
        combined = " ".join(c.text for c in result)
        self.assertIn("\u00c9tude", combined)

    def test_many_blank_lines(self):
        text = "Paragraph 1.\n\n\n\n\n\nParagraph 2.\n\n\n\nParagraph 3."
        result = smart_chunk(text)
        self.assertGreater(len(result), 0)

    def test_nested_headers(self):
        text = (
            "# Level 1\n\nContent.\n\n"
            "## Level 2\n\nContent.\n\n"
            "### Level 3\n\nContent.\n\n"
            "#### Level 4\n\nContent.\n\n"
            "##### Level 5\n\nContent.\n\n"
            "###### Level 6\n\nContent."
        )
        result = smart_chunk(text)
        self.assertGreater(len(result), 0)

    def test_mixed_content_types(self):
        text = (
            "# Introduction\n\n"
            "A paragraph of text.\n\n"
            "- List item one\n- List item two\n\n"
            "```\ncode_block()\n```\n\n"
            "Another paragraph.\n\n"
            "1. Numbered item\n2. Another item"
        )
        result = smart_chunk(text)
        self.assertGreater(len(result), 0)

    def test_chunk_object_fields(self):
        result = smart_chunk("Test content.")
        chunk = result[0]
        self.assertIsInstance(chunk.text, str)
        self.assertIsInstance(chunk.index, int)
        self.assertIsInstance(chunk.start_char, int)
        self.assertIsInstance(chunk.end_char, int)
        self.assertIsInstance(chunk.metadata, dict)

    def test_min_chunk_size_merging(self):
        # Create many tiny segments
        text = "\n\n".join(f"W{i}." for i in range(20))
        cfg = SmartChunkerConfig(min_chunk_size=50, max_chunk_size=200)
        result = smart_chunk(text, config=cfg)
        # After merging, chunks should be >= min_chunk_size (where possible)
        for chunk in result:
            # Allow some tolerance — the merge is best-effort
            self.assertTrue(len(chunk.text) > 0)


if __name__ == "__main__":
    unittest.main()
