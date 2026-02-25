#!/usr/bin/env python3
"""Error path and edge-case tests for mind-mem — malformed inputs, missing files, bad configs."""

import json
import os
import sys
import tempfile
import unittest

from mind_mem.block_parser import parse_blocks, parse_file  # noqa: E402
from mind_mem.init_workspace import _validate_config, load_config  # noqa: E402
from mind_mem.mind_ffi import (  # noqa: E402
    MindMemKernel,
    get_mind_dir,
    list_kernels,
    load_kernel,
    load_kernel_config,
)
from mind_mem.recall import recall, tokenize  # noqa: E402

# ---------------------------------------------------------------------------
# 1. Malformed config (mind-mem.json)
# ---------------------------------------------------------------------------


class TestMalformedConfig(unittest.TestCase):
    """Verify graceful handling of broken or missing mind-mem.json files."""

    def test_json_syntax_error(self):
        """load_config returns defaults when JSON has syntax errors."""
        with tempfile.TemporaryDirectory() as ws:
            config_path = os.path.join(ws, "mind-mem.json")
            with open(config_path, "w") as f:
                f.write("{invalid json,,}")
            cfg = load_config(ws)
            self.assertIsInstance(cfg, dict)

    def test_empty_file(self):
        """load_config returns defaults when config file is empty."""
        with tempfile.TemporaryDirectory() as ws:
            config_path = os.path.join(ws, "mind-mem.json")
            with open(config_path, "w"):
                pass  # empty file
            cfg = load_config(ws)
            self.assertIsInstance(cfg, dict)

    def test_missing_config_file(self):
        """load_config returns defaults when mind-mem.json doesn't exist."""
        with tempfile.TemporaryDirectory() as ws:
            cfg = load_config(ws)
            self.assertIsInstance(cfg, dict)

    def test_config_is_array_not_object(self):
        """load_config handles JSON array instead of object."""
        with tempfile.TemporaryDirectory() as ws:
            config_path = os.path.join(ws, "mind-mem.json")
            with open(config_path, "w") as f:
                json.dump([1, 2, 3], f)
            cfg = load_config(ws)
            self.assertIsInstance(cfg, (dict, list))

    def test_config_is_null(self):
        """load_config handles JSON null without crashing."""
        with tempfile.TemporaryDirectory() as ws:
            config_path = os.path.join(ws, "mind-mem.json")
            with open(config_path, "w") as f:
                f.write("null")
            # Should not raise — may return None since json.load("null") -> None
            # and _validate_config passes it through
            cfg = load_config(ws)
            self.assertIsInstance(cfg, (dict, type(None)))

    def test_config_with_binary_garbage(self):
        """load_config returns defaults when config contains binary data."""
        with tempfile.TemporaryDirectory() as ws:
            config_path = os.path.join(ws, "mind-mem.json")
            with open(config_path, "wb") as f:
                f.write(b"\x00\x01\x02\xff\xfe\xfd")
            cfg = load_config(ws)
            self.assertIsInstance(cfg, dict)


# ---------------------------------------------------------------------------
# 2. Corrupted block files
# ---------------------------------------------------------------------------


class TestCorruptedBlockFile(unittest.TestCase):
    """Verify block_parser handles corrupt/invalid markdown gracefully."""

    def test_binary_content(self):
        """parse_blocks with binary-like content should not crash."""
        text = "\x00\x01[FAKE-ID]\x02\x03Statement: binary\xff\n"
        # Should not raise, may return 0 or more blocks
        result = parse_blocks(text)
        self.assertIsInstance(result, list)

    def test_only_separators(self):
        """File with only --- separators produces zero blocks."""
        text = "---\n---\n---\n---\n"
        blocks = parse_blocks(text)
        self.assertEqual(blocks, [])

    def test_field_without_block_header(self):
        """Fields without a preceding block ID are ignored."""
        text = "Statement: orphan field\nStatus: active\nDate: 2026-01-01\n"
        blocks = parse_blocks(text)
        self.assertEqual(blocks, [])

    def test_nested_brackets_in_id(self):
        """Nested brackets in ID line should not parse as block."""
        text = "[[D-20260215-001]]\nStatement: test\n"
        blocks = parse_blocks(text)
        # The regex requires exact [ID] format at start of line
        self.assertEqual(blocks, [])

    def test_truncated_file(self):
        """File that ends mid-field should still parse partial block."""
        text = "[D-20260215-001]\nStatement: this is"
        blocks = parse_blocks(text)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["_id"], "D-20260215-001")

    def test_parse_file_nonexistent(self):
        """parse_file raises on nonexistent file."""
        with self.assertRaises((OSError, FileNotFoundError)):
            parse_file("/nonexistent/path/to/file.md")

    @unittest.skipIf(sys.platform == "win32", "chmod 000 not enforced on Windows")
    def test_parse_file_permission_denied(self):
        """parse_file raises on unreadable file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("[D-20260215-001]\nStatement: test\n")
            fpath = f.name
        try:
            os.chmod(fpath, 0o000)
            with self.assertRaises((OSError, PermissionError)):
                parse_file(fpath)
        finally:
            os.chmod(fpath, 0o644)
            os.unlink(fpath)


# ---------------------------------------------------------------------------
# 3. Invalid block IDs
# ---------------------------------------------------------------------------


class TestInvalidBlockIDs(unittest.TestCase):
    """Verify handling of block IDs with special characters and path traversal."""

    def test_id_with_path_traversal(self):
        """Block IDs with path traversal chars are accepted as data (not used for file access)."""
        text = "[D-../../etc/passwd]\nStatement: malicious\n"
        blocks = parse_blocks(text)
        # Parser accepts any [UPPERCASE-...] pattern; IDs are data, not file paths
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["_id"], "D-../../etc/passwd")

    def test_id_with_null_bytes(self):
        """Block IDs with null bytes parse (regex matches any non-bracket char)."""
        text = "[D-20260215\x00-001]\nStatement: null byte\n"
        blocks = parse_blocks(text)
        # The regex [^\]]+ matches null bytes — parser accepts it
        self.assertEqual(len(blocks), 1)

    def test_id_with_newline_injection(self):
        """Block IDs spanning lines should not parse."""
        text = "[D-20260215\n-001]\nStatement: newline in id\n"
        blocks = parse_blocks(text)
        # Newline breaks the ID
        self.assertEqual(blocks, [])

    def test_id_with_sql_injection(self):
        """Block IDs with SQL injection chars are parsed as data (not used in SQL)."""
        text = "[D-20260215'; DROP TABLE--001]\nStatement: sql injection\n"
        blocks = parse_blocks(text)
        # Parser accepts the ID as plain text data — no SQL execution risk
        self.assertEqual(len(blocks), 1)

    def test_very_long_id(self):
        """Block IDs with extremely long names should parse but not crash."""
        long_id = "D-" + "9" * 5000
        text = f"[{long_id}]\nStatement: long id\n"
        blocks = parse_blocks(text)
        # Should parse since it matches the regex pattern
        self.assertEqual(len(blocks), 1)

    def test_id_with_unicode_injection(self):
        """Block IDs with unicode null are accepted by the permissive regex."""
        text = "[D-2026\u0000215-001]\nStatement: unicode null\n"
        blocks = parse_blocks(text)
        # The regex [^\]]+ matches any non-bracket character including \u0000
        self.assertEqual(len(blocks), 1)


# ---------------------------------------------------------------------------
# 4. Empty workspace recall
# ---------------------------------------------------------------------------


class TestEmptyWorkspaceRecall(unittest.TestCase):
    """Verify recall on a workspace with no memory files."""

    def test_recall_empty_workspace(self):
        """Recall on empty workspace returns empty list, no crash."""
        with tempfile.TemporaryDirectory() as ws:
            results = recall(ws, "test query")
            self.assertEqual(results, [])

    def test_recall_workspace_with_empty_corpus_files(self):
        """Recall when corpus files exist but are empty returns empty list."""
        with tempfile.TemporaryDirectory() as ws:
            os.makedirs(os.path.join(ws, "decisions"), exist_ok=True)
            with open(os.path.join(ws, "decisions", "DECISIONS.md"), "w") as f:
                f.write("")
            results = recall(ws, "test query")
            self.assertEqual(results, [])

    def test_recall_nonexistent_workspace(self):
        """Recall on nonexistent workspace path returns empty list."""
        results = recall("/nonexistent/workspace/path", "test query")
        self.assertEqual(results, [])


# ---------------------------------------------------------------------------
# 5. Missing .so graceful fallback (mind_ffi)
# ---------------------------------------------------------------------------


class TestMissingSOFallback(unittest.TestCase):
    """Verify mind_ffi degrades gracefully when .so library is missing."""

    def test_kernel_explicit_bad_path(self):
        """MindMemKernel with explicit bad path raises OSError."""
        with self.assertRaises(OSError):
            MindMemKernel("/tmp/nonexistent_libmindmem.so")

    def test_kernel_bad_file(self):
        """MindMemKernel with non-shared-library file raises OSError."""
        with tempfile.NamedTemporaryFile(suffix=".so", delete=False) as f:
            f.write(b"not a shared library")
            fpath = f.name
        try:
            with self.assertRaises(OSError):
                MindMemKernel(fpath)
        finally:
            os.unlink(fpath)

    def test_list_kernels_nonexistent_dir(self):
        """list_kernels returns [] for nonexistent directory."""
        result = list_kernels("/nonexistent/directory")
        self.assertEqual(result, [])

    def test_list_kernels_empty_dir(self):
        """list_kernels returns [] for directory with no .mind files."""
        with tempfile.TemporaryDirectory() as d:
            result = list_kernels(d)
            self.assertEqual(result, [])

    def test_load_kernel_nonexistent_file(self):
        """load_kernel returns {} for nonexistent path."""
        result = load_kernel("/nonexistent/kernel.mind")
        self.assertEqual(result, {})

    def test_load_kernel_config_nonexistent(self):
        """load_kernel_config returns {} for nonexistent path."""
        result = load_kernel_config("/nonexistent/config.mind")
        self.assertEqual(result, {})

    def test_load_kernel_config_binary_file(self):
        """load_kernel_config returns {} for binary file."""
        with tempfile.NamedTemporaryFile(suffix=".mind", delete=False) as f:
            f.write(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")
            fpath = f.name
        try:
            result = load_kernel_config(fpath)
            self.assertIsInstance(result, dict)
        finally:
            os.unlink(fpath)

    def test_get_mind_dir_empty_workspace(self):
        """get_mind_dir with nonexistent workspace falls back."""
        with tempfile.TemporaryDirectory() as ws:
            result = get_mind_dir(ws)
            self.assertIsInstance(result, str)


# ---------------------------------------------------------------------------
# 6. Invalid config values
# ---------------------------------------------------------------------------


class TestInvalidConfigValues(unittest.TestCase):
    """Verify _validate_config handles invalid numeric values."""

    def test_negative_limit(self):
        """Negative limit is clamped to minimum (1)."""
        cfg = {"recall": {"limit": -5}}
        result = _validate_config(cfg)
        self.assertGreaterEqual(result["recall"]["limit"], 1)

    def test_zero_limit(self):
        """Zero limit is clamped to minimum (1)."""
        cfg = {"recall": {"limit": 0}}
        result = _validate_config(cfg)
        self.assertGreaterEqual(result["recall"]["limit"], 1)

    def test_extreme_bm25_k1(self):
        """bm25_k1 above max is clamped to 3.0."""
        cfg = {"recall": {"bm25_k1": 999.0}}
        result = _validate_config(cfg)
        self.assertLessEqual(result["recall"]["bm25_k1"], 3.0)

    def test_negative_bm25_b(self):
        """Negative bm25_b is clamped to 0.0."""
        cfg = {"recall": {"bm25_b": -1.0}}
        result = _validate_config(cfg)
        self.assertGreaterEqual(result["recall"]["bm25_b"], 0.0)

    def test_string_as_numeric_value(self):
        """String value for numeric field falls back to default."""
        cfg = {"recall": {"limit": "not_a_number"}}
        result = _validate_config(cfg)
        # Should use default value, which is 20
        self.assertIsInstance(result["recall"]["limit"], int)
        self.assertEqual(result["recall"]["limit"], 20)

    def test_none_as_numeric_value(self):
        """None value for numeric field falls back to default."""
        cfg = {"recall": {"limit": None}}
        result = _validate_config(cfg)
        self.assertIsInstance(result["recall"]["limit"], int)

    def test_empty_recall_section(self):
        """Empty recall section is accepted without error."""
        cfg = {"recall": {}}
        result = _validate_config(cfg)
        self.assertEqual(result["recall"], {})

    def test_missing_recall_section(self):
        """Config without recall section is passed through."""
        cfg = {"other": "data"}
        result = _validate_config(cfg)
        self.assertEqual(result, {"other": "data"})

    def test_recall_not_dict(self):
        """Non-dict recall value is passed through."""
        cfg = {"recall": "not a dict"}
        result = _validate_config(cfg)
        self.assertEqual(result["recall"], "not a dict")

    def test_extreme_rrf_k(self):
        """rrf_k above max is clamped to 200."""
        cfg = {"recall": {"rrf_k": 99999}}
        result = _validate_config(cfg)
        self.assertLessEqual(result["recall"]["rrf_k"], 200)

    def test_negative_vector_weight(self):
        """Negative vector_weight is clamped to 0.0."""
        cfg = {"recall": {"vector_weight": -10.0}}
        result = _validate_config(cfg)
        self.assertGreaterEqual(result["recall"]["vector_weight"], 0.0)

    def test_float_infinity(self):
        """Float infinity for config value is clamped."""
        cfg = {"recall": {"bm25_k1": float("inf")}}
        result = _validate_config(cfg)
        self.assertLessEqual(result["recall"]["bm25_k1"], 3.0)


# ---------------------------------------------------------------------------
# 7. Large limit values in recall
# ---------------------------------------------------------------------------


class TestLargeLimitValues(unittest.TestCase):
    """Verify recall handles extreme limit values."""

    def test_recall_limit_999999(self):
        """Recall with limit=999999 should work without OOM on empty workspace."""
        with tempfile.TemporaryDirectory() as ws:
            results = recall(ws, "test query", limit=999999)
            self.assertEqual(results, [])

    def test_recall_limit_zero(self):
        """Recall with limit=0 returns empty list."""
        with tempfile.TemporaryDirectory() as ws:
            results = recall(ws, "test query", limit=0)
            self.assertEqual(results, [])

    def test_recall_limit_one(self):
        """Recall with limit=1 returns at most one result."""
        with tempfile.TemporaryDirectory() as ws:
            # Create a corpus file with a block
            os.makedirs(os.path.join(ws, "decisions"), exist_ok=True)
            with open(os.path.join(ws, "decisions", "DECISIONS.md"), "w") as f:
                f.write("[D-20260215-001]\nStatement: test query subject\nStatus: active\n")
            results = recall(ws, "test query", limit=1)
            self.assertLessEqual(len(results), 1)


# ---------------------------------------------------------------------------
# 8. Bad query types in recall
# ---------------------------------------------------------------------------


class TestBadQueryTypes(unittest.TestCase):
    """Verify recall handles empty, None, and oversized queries."""

    def test_recall_empty_query(self):
        """Recall with empty string query returns empty list."""
        with tempfile.TemporaryDirectory() as ws:
            results = recall(ws, "")
            self.assertEqual(results, [])

    def test_recall_whitespace_query(self):
        """Recall with whitespace-only query returns empty list."""
        with tempfile.TemporaryDirectory() as ws:
            results = recall(ws, "   \t\n  ")
            self.assertEqual(results, [])

    def test_recall_stopwords_only_query(self):
        """Recall with only stopwords returns empty list."""
        with tempfile.TemporaryDirectory() as ws:
            results = recall(ws, "the a an is are was for")
            self.assertEqual(results, [])

    def test_recall_very_long_query(self):
        """Recall with >10KB query should not crash."""
        with tempfile.TemporaryDirectory() as ws:
            long_query = "database " * 2000  # ~16KB
            results = recall(ws, long_query)
            self.assertIsInstance(results, list)

    def test_recall_special_chars_query(self):
        """Recall with special characters in query should not crash."""
        with tempfile.TemporaryDirectory() as ws:
            results = recall(ws, "!@#$%^&*()[]{}|\\/<>?")
            self.assertIsInstance(results, list)

    def test_recall_unicode_query(self):
        """Recall with unicode query should not crash."""
        with tempfile.TemporaryDirectory() as ws:
            results = recall(ws, "\u4f60\u597d\u4e16\u754c \u3053\u3093\u306b\u3061\u306f \ud83d\ude80")
            self.assertIsInstance(results, list)

    def test_recall_numeric_query(self):
        """Recall with purely numeric query works."""
        with tempfile.TemporaryDirectory() as ws:
            results = recall(ws, "12345678")
            self.assertIsInstance(results, list)

    def test_tokenize_none_input(self):
        """tokenize with non-string input should raise."""
        with self.assertRaises((TypeError, AttributeError)):
            tokenize(None)

    def test_tokenize_very_long_input(self):
        """tokenize with very long input should not hang."""
        long_text = "word " * 50000  # 250K chars
        result = tokenize(long_text)
        self.assertIsInstance(result, list)


# ---------------------------------------------------------------------------
# 9. Config with mind-mem.json in recall path
# ---------------------------------------------------------------------------


class TestRecallWithBadConfig(unittest.TestCase):
    """Recall continues working even when mind-mem.json is corrupt."""

    def test_recall_with_corrupt_config(self):
        """Recall falls back gracefully when mind-mem.json is corrupt."""
        with tempfile.TemporaryDirectory() as ws:
            config_path = os.path.join(ws, "mind-mem.json")
            with open(config_path, "w") as f:
                f.write("{broken json")
            # Should not raise — recall ignores bad config
            results = recall(ws, "test query")
            self.assertEqual(results, [])

    def test_recall_with_config_wrong_type(self):
        """Recall handles config where recall section is wrong type."""
        with tempfile.TemporaryDirectory() as ws:
            config_path = os.path.join(ws, "mind-mem.json")
            with open(config_path, "w") as f:
                json.dump({"recall": "not_a_dict"}, f)
            results = recall(ws, "test query")
            self.assertIsInstance(results, list)

    def test_recall_with_config_unknown_backend(self):
        """Recall handles config with unknown backend name."""
        with tempfile.TemporaryDirectory() as ws:
            config_path = os.path.join(ws, "mind-mem.json")
            with open(config_path, "w") as f:
                json.dump({"recall": {"backend": "nonexistent_backend"}}, f)
            results = recall(ws, "test query")
            self.assertIsInstance(results, list)


# ---------------------------------------------------------------------------
# 10. load_kernel_config with malformed .mind files
# ---------------------------------------------------------------------------


class TestLoadKernelConfigEdgeCases(unittest.TestCase):
    """Verify load_kernel_config handles malformed .mind INI files."""

    def test_empty_mind_file(self):
        """Empty .mind file returns empty dict."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mind", delete=False) as f:
            fpath = f.name
        try:
            result = load_kernel_config(fpath)
            self.assertEqual(result, {})
        finally:
            os.unlink(fpath)

    def test_mind_file_no_sections(self):
        """key=value lines without [section] are ignored."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mind", delete=False) as f:
            f.write("k1 = 1.2\nb = 0.75\n")
            fpath = f.name
        try:
            result = load_kernel_config(fpath)
            self.assertEqual(result, {})
        finally:
            os.unlink(fpath)

    def test_mind_file_comments_only(self):
        """.mind file with only comments returns empty dict."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mind", delete=False) as f:
            f.write("# comment line 1\n# comment line 2\n")
            fpath = f.name
        try:
            result = load_kernel_config(fpath)
            self.assertEqual(result, {})
        finally:
            os.unlink(fpath)

    def test_mind_file_malformed_section(self):
        """Malformed section headers are skipped."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mind", delete=False) as f:
            f.write("[invalid section!]\nk1 = 1.2\n[bm25]\nk1 = 1.5\n")
            fpath = f.name
        try:
            result = load_kernel_config(fpath)
            # Only [bm25] should parse; [invalid section!] is skipped
            self.assertIn("bm25", result)
            self.assertEqual(result["bm25"]["k1"], 1.5)
        finally:
            os.unlink(fpath)


if __name__ == "__main__":
    unittest.main()
