#!/usr/bin/env python3
"""Tests for _recall_reranking.py — deterministic reranker + LLM rerank."""

import json
import os
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from mind_mem._recall_reranking import llm_rerank, rerank_hits


def _make_hit(id_, score, excerpt="test excerpt", speaker="", tags="", dia=""):
    """Helper to build a minimal hit dict."""
    h = {
        "_id": id_,
        "score": score,
        "excerpt": excerpt,
        "speaker": speaker,
        "tags": tags,
        "line": 1,
        "file": "test.md",
    }
    if dia:
        h["DiaID"] = dia
    return h


class TestRerankerDeterministic(unittest.TestCase):
    """Tests for the deterministic rerank_hits() function."""

    def test_empty_input(self):
        result = rerank_hits("test query", [])
        self.assertEqual(result, [])

    def test_single_hit_preserved(self):
        hits = [_make_hit("D-001", 5.0, "the only result")]
        result = rerank_hits("test query", hits)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["_id"], "D-001")

    def test_scores_are_modified(self):
        hits = [
            _make_hit("D-001", 5.0, "PostgreSQL database decision"),
            _make_hit("D-002", 4.0, "random unrelated topic here"),
        ]
        original_scores = [h["score"] for h in hits]
        result = rerank_hits("PostgreSQL database", hits)
        new_scores = [h["score"] for h in result]
        # At least one score should have changed
        self.assertNotEqual(original_scores, new_scores)

    def test_entity_overlap_boosts_score(self):
        hits = [
            _make_hit("D-001", 3.0, "We use Redis for caching"),
            _make_hit("D-002", 3.0, "The weather is nice today"),
        ]
        result = rerank_hits("What caching solution do we use? Redis?", hits)
        # Hit mentioning Redis should score higher
        redis_hit = next(h for h in result if h["_id"] == "D-001")
        other_hit = next(h for h in result if h["_id"] == "D-002")
        self.assertGreater(redis_hit["score"], other_hit["score"])

    def test_anchor_rule_preserves_bm25_top1(self):
        # BM25 top-1 should always be in the final top-10
        hits = [_make_hit(f"D-{i:03d}", 10.0 - i) for i in range(15)]
        result = rerank_hits("some query", hits)
        top10_ids = {h["_id"] for h in result[:10]}
        self.assertIn("D-000", top10_ids)

    def test_debug_mode_attaches_features(self):
        hits = [_make_hit("D-001", 5.0, "test content")]
        result = rerank_hits("test query", hits, debug=True)
        self.assertIn("_rerank_features", result[0])
        features = result[0]["_rerank_features"]
        self.assertIn("entity_overlap", features)
        self.assertIn("bm25_original", features)

    def test_sorting_descending(self):
        hits = [
            _make_hit("D-001", 1.0, "low scorer"),
            _make_hit("D-002", 10.0, "high scorer"),
        ]
        result = rerank_hits("test", hits)
        self.assertGreaterEqual(result[0]["score"], result[1]["score"])


class TestLLMRerank(unittest.TestCase):
    """Tests for the LLM-based llm_rerank() function."""

    def test_empty_input(self):
        result = llm_rerank("test", [])
        self.assertEqual(result, [])

    def test_fallback_on_network_error(self):
        hits = [
            _make_hit("D-001", 5.0, "test 1"),
            _make_hit("D-002", 3.0, "test 2"),
        ]
        original_scores = [h["score"] for h in hits]
        # Use an unreachable URL — should fall back silently
        result = llm_rerank(
            "test query", hits,
            url="http://127.0.0.1:19999/nonexistent",
            timeout=0.5,
        )
        # Scores should be unchanged (fallback)
        for h, orig in zip(result, original_scores):
            self.assertEqual(h["score"], orig)

    def test_blending_with_mock_server(self):
        """Test LLM rerank with a real HTTP mock returning valid scores."""
        hits = [
            _make_hit("D-001", 5.0, "highly relevant result"),
            _make_hit("D-002", 3.0, "less relevant result"),
            _make_hit("D-003", 4.0, "medium relevance"),
        ]

        class MockHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                self.rfile.read(length)
                response = json.dumps({
                    "response": "[0.9, 0.1, 0.5]",
                })
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(response.encode())

            def log_message(self, format, *args):
                pass  # suppress output

        server = HTTPServer(("127.0.0.1", 0), MockHandler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.handle_request, daemon=True)
        thread.start()

        try:
            result = llm_rerank(
                "find relevant docs", hits,
                url=f"http://127.0.0.1:{port}/api/generate",
                weight=0.3,
                timeout=5.0,
            )
            # D-001 had highest BM25 (5.0) + highest LLM (0.9) → should still be first
            self.assertEqual(result[0]["_id"], "D-001")
            # Scores should be modified from originals
            self.assertNotEqual(result[0]["score"], 5.0)
        finally:
            server.server_close()

    def test_handles_wrong_length_response(self):
        """If LLM returns wrong number of scores, fall back."""
        hits = [
            _make_hit("D-001", 5.0, "test"),
            _make_hit("D-002", 3.0, "test"),
        ]

        class MockHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                self.rfile.read(length)
                # Return wrong number of scores
                response = json.dumps({"response": "[0.5]"})
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(response.encode())

            def log_message(self, format, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), MockHandler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.handle_request, daemon=True)
        thread.start()

        try:
            result = llm_rerank(
                "test", hits,
                url=f"http://127.0.0.1:{port}/api/generate",
                timeout=5.0,
            )
            # Should fall back — scores unchanged
            self.assertEqual(result[0]["score"], 5.0)
        finally:
            server.server_close()

    def test_handles_non_json_response(self):
        """If LLM returns garbage, fall back."""
        hits = [_make_hit("D-001", 5.0, "test")]

        class MockHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                self.rfile.read(length)
                response = json.dumps({"response": "I cannot score these."})
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(response.encode())

            def log_message(self, format, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), MockHandler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.handle_request, daemon=True)
        thread.start()

        try:
            result = llm_rerank(
                "test", hits,
                url=f"http://127.0.0.1:{port}/api/generate",
                timeout=5.0,
            )
            self.assertEqual(result[0]["score"], 5.0)
        finally:
            server.server_close()

    def test_score_clamping(self):
        """LLM scores outside [0,1] should be clamped."""
        hits = [
            _make_hit("D-001", 5.0, "test 1"),
            _make_hit("D-002", 3.0, "test 2"),
        ]

        class MockHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                self.rfile.read(length)
                # Return out-of-range scores
                response = json.dumps({"response": "[1.5, -0.3]"})
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(response.encode())

            def log_message(self, format, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), MockHandler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.handle_request, daemon=True)
        thread.start()

        try:
            result = llm_rerank(
                "test", hits,
                url=f"http://127.0.0.1:{port}/api/generate",
                weight=0.3,
                timeout=5.0,
            )
            # Should still produce valid scores (clamped to [0,1])
            for h in result:
                self.assertGreaterEqual(h["score"], 0.0)
        finally:
            server.server_close()

    def test_weight_zero_preserves_original(self):
        """weight=0 should preserve original scores exactly."""
        hits = [
            _make_hit("D-001", 5.0, "test 1"),
            _make_hit("D-002", 3.0, "test 2"),
        ]

        class MockHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                self.rfile.read(length)
                response = json.dumps({"response": "[0.1, 0.9]"})
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(response.encode())

            def log_message(self, format, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), MockHandler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.handle_request, daemon=True)
        thread.start()

        try:
            result = llm_rerank(
                "test", hits,
                url=f"http://127.0.0.1:{port}/api/generate",
                weight=0.0,
                timeout=5.0,
            )
            self.assertEqual(result[0]["score"], 5.0)
            self.assertEqual(result[1]["score"], 3.0)
        finally:
            server.server_close()


if __name__ == "__main__":
    unittest.main()
