#!/usr/bin/env python3
"""Tests for block_parser.py — overlapping chunk splitting + dedup."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from block_parser import chunk_block, deduplicate_chunks


class TestChunkBlock(unittest.TestCase):
    """Tests for chunk_block() — overlapping window splitting."""

    def _make_block(self, statement_words=100, id_="D-20260222-001"):
        """Create a block with N words in Statement."""
        text = " ".join(f"word{i}" for i in range(statement_words))
        return {
            "_id": id_,
            "_line": 1,
            "Statement": text,
            "Status": "active",
            "Tags": "test",
        }

    def test_short_block_not_chunked(self):
        block = self._make_block(50)
        chunks = chunk_block(block, max_tokens=400, overlap=50)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["_id"], "D-20260222-001")

    def test_long_block_is_chunked(self):
        block = self._make_block(500)
        chunks = chunk_block(block, max_tokens=200, overlap=50)
        self.assertGreater(len(chunks), 1)

    def test_chunk_ids_have_suffix(self):
        block = self._make_block(500)
        chunks = chunk_block(block, max_tokens=200, overlap=50)
        for i, chunk in enumerate(chunks):
            self.assertEqual(chunk["_id"], f"D-20260222-001.{i}")

    def test_chunk_preserves_metadata(self):
        block = self._make_block(500)
        chunks = chunk_block(block, max_tokens=200, overlap=50)
        for chunk in chunks:
            self.assertEqual(chunk["Status"], "active")
            self.assertEqual(chunk["Tags"], "test")
            self.assertEqual(chunk["_line"], 1)

    def test_chunk_parent_field(self):
        block = self._make_block(500)
        chunks = chunk_block(block, max_tokens=200, overlap=50)
        for chunk in chunks:
            self.assertEqual(chunk["_chunk_parent"], "D-20260222-001")

    def test_chunks_overlap(self):
        block = self._make_block(500)
        chunks = chunk_block(block, max_tokens=200, overlap=50)
        if len(chunks) >= 2:
            words0 = set(chunks[0]["Statement"].split())
            words1 = set(chunks[1]["Statement"].split())
            overlap = words0 & words1
            self.assertGreater(len(overlap), 0)

    def test_all_words_covered(self):
        block = self._make_block(500)
        original_words = set(block["Statement"].split())
        chunks = chunk_block(block, max_tokens=200, overlap=50)
        chunked_words = set()
        for c in chunks:
            chunked_words.update(c["Statement"].split())
        self.assertEqual(original_words, chunked_words)

    def test_no_overlap_zero_value(self):
        block = self._make_block(500)
        chunks = chunk_block(block, max_tokens=200, overlap=0)
        self.assertGreater(len(chunks), 1)
        # With 0 overlap, chunks should be non-overlapping
        for i in range(len(chunks) - 1):
            words_i = chunks[i]["Statement"].split()
            words_next = chunks[i + 1]["Statement"].split()
            # Last word of chunk i should not be first word of chunk i+1
            if words_i and words_next:
                self.assertNotEqual(words_i[-1], words_next[0])

    def test_block_without_text_field(self):
        block = {"_id": "D-001", "_line": 1, "Tags": "test"}
        chunks = chunk_block(block, max_tokens=200, overlap=50)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["_id"], "D-001")

    def test_exact_boundary(self):
        """Block with exactly max_tokens words should not be chunked."""
        block = self._make_block(400)
        chunks = chunk_block(block, max_tokens=400, overlap=50)
        self.assertEqual(len(chunks), 1)

    def test_max_tokens_plus_one(self):
        """Block with max_tokens+1 words should be chunked into 2."""
        block = self._make_block(401)
        chunks = chunk_block(block, max_tokens=400, overlap=50)
        self.assertEqual(len(chunks), 2)

    def test_uses_longest_text_field(self):
        """Should chunk the longest text field, not shorter ones."""
        block = {
            "_id": "D-001",
            "_line": 1,
            "Title": "short title",
            "Description": " ".join(f"w{i}" for i in range(500)),
        }
        chunks = chunk_block(block, max_tokens=200, overlap=50)
        self.assertGreater(len(chunks), 1)
        # Title should be preserved unchanged
        for chunk in chunks:
            self.assertEqual(chunk["Title"], "short title")


class TestDeduplicateChunks(unittest.TestCase):
    """Tests for deduplicate_chunks() — merging by base block ID."""

    def test_no_chunks_passthrough(self):
        results = [
            {"_id": "D-001", "score": 5.0},
            {"_id": "D-002", "score": 3.0},
        ]
        deduped = deduplicate_chunks(results)
        self.assertEqual(len(deduped), 2)

    def test_chunks_merged_highest_score(self):
        results = [
            {"_id": "D-001.0", "score": 3.0},
            {"_id": "D-001.1", "score": 5.0},
            {"_id": "D-002", "score": 4.0},
        ]
        deduped = deduplicate_chunks(results)
        self.assertEqual(len(deduped), 2)
        d001 = next(r for r in deduped if "D-001" in r["_id"])
        self.assertEqual(d001["score"], 5.0)

    def test_preserves_non_chunked_ids(self):
        results = [
            {"_id": "D-001", "score": 5.0},
            {"_id": "D-002", "score": 3.0},
        ]
        deduped = deduplicate_chunks(results)
        ids = [r["_id"] for r in deduped]
        self.assertIn("D-001", ids)
        self.assertIn("D-002", ids)

    def test_complex_id_not_treated_as_chunk(self):
        """IDs with dots that aren't chunk suffixes should not be merged."""
        results = [
            {"_id": "D-001.abc", "score": 5.0},
            {"_id": "D-001.def", "score": 3.0},
        ]
        deduped = deduplicate_chunks(results)
        # "abc" and "def" are not digits, so these are not chunk IDs
        self.assertEqual(len(deduped), 2)

    def test_empty_input(self):
        self.assertEqual(deduplicate_chunks([]), [])

    def test_ordering_preserved(self):
        results = [
            {"_id": "D-002", "score": 6.0},
            {"_id": "D-001.1", "score": 5.0},
            {"_id": "D-001.0", "score": 3.0},
        ]
        deduped = deduplicate_chunks(results)
        # D-002 should come first (appeared first), D-001.1 second (highest score)
        self.assertEqual(deduped[0]["_id"], "D-002")
        self.assertEqual(deduped[1]["_id"], "D-001.1")


if __name__ == "__main__":
    unittest.main()
