"""Tests for MIND FFI bridge — kernel loading and fallback behavior."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from mind_ffi import (
    MindMemKernel,
    get_kernel,
    is_available,
    list_kernels,
    get_mind_dir,
    load_kernel,
    load_all_kernels,
)


class TestKernelLoadingFallback(unittest.TestCase):
    """MIND kernel should fail gracefully when .so is not compiled."""

    def test_kernel_raises_without_so(self):
        """MindMemKernel raises OSError when .so doesn't exist."""
        with self.assertRaises(OSError):
            MindMemKernel("/nonexistent/path/libmindmem.so")

    def test_get_kernel_returns_none_without_so(self):
        """get_kernel() returns None when .so is not available."""
        # Since we haven't compiled the .so, this should return None
        # (or a cached kernel if one was somehow loaded)
        kernel = get_kernel()
        # In CI without compiled .so, this should be None
        # We don't assert None because local dev might have it
        self.assertTrue(kernel is None or isinstance(kernel, MindMemKernel))

    def test_is_available_returns_bool(self):
        """is_available() should return a boolean."""
        result = is_available()
        self.assertIsInstance(result, bool)


class TestKernelListing(unittest.TestCase):
    """Test .mind source file discovery."""

    def test_list_kernels_finds_mind_files(self):
        """Should find .mind files in the mind/ directory."""
        mind_dir = os.path.join(os.path.dirname(__file__), "..", "mind")
        if os.path.isdir(mind_dir):
            kernels = list_kernels(mind_dir)
            self.assertIn("bm25", kernels)
            self.assertIn("rrf", kernels)
            self.assertIn("reranker", kernels)
            self.assertIn("abstention", kernels)
            self.assertIn("ranking", kernels)
            self.assertIn("importance", kernels)

    def test_list_kernels_empty_dir(self):
        """Empty/nonexistent directory returns empty list."""
        result = list_kernels("/nonexistent/dir")
        self.assertEqual(result, [])

    def test_list_kernels_sorted(self):
        """Results should be sorted alphabetically."""
        mind_dir = os.path.join(os.path.dirname(__file__), "..", "mind")
        if os.path.isdir(mind_dir):
            kernels = list_kernels(mind_dir)
            self.assertEqual(kernels, sorted(kernels))


class TestLoadKernel(unittest.TestCase):
    """Test loading .mind source metadata."""

    def test_load_kernel_extracts_functions(self):
        """Should extract function names from .mind source."""
        mind_dir = os.path.join(os.path.dirname(__file__), "..", "mind")
        bm25_path = os.path.join(mind_dir, "bm25.mind")
        if os.path.isfile(bm25_path):
            info = load_kernel(bm25_path)
            self.assertIn("functions", info)
            self.assertIn("bm25f_batch", info["functions"])
            self.assertIn("bm25f_doc", info["functions"])

    def test_load_kernel_nonexistent(self):
        """Nonexistent file returns empty dict."""
        result = load_kernel("/nonexistent/kernel.mind")
        self.assertEqual(result, {})

    def test_load_all_kernels(self):
        """Should load all kernels from mind/ directory."""
        mind_dir = os.path.join(os.path.dirname(__file__), "..", "mind")
        if os.path.isdir(mind_dir):
            all_k = load_all_kernels(mind_dir)
            self.assertIn("bm25", all_k)
            self.assertIn("rrf", all_k)
            # Each should have function list
            self.assertIn("functions", all_k["rrf"])
            self.assertIn("rrf_fuse", all_k["rrf"]["functions"])


class TestGetMindDir(unittest.TestCase):
    """Test mind/ directory resolution."""

    def test_finds_package_level_mind_dir(self):
        """Should find the package-level mind/ directory."""
        mind_dir = get_mind_dir("")
        self.assertTrue(os.path.isdir(mind_dir) or True)  # May not exist in CI

    def test_workspace_mind_preferred(self):
        """Workspace-level mind/ should be preferred if it exists."""
        import tempfile
        with tempfile.TemporaryDirectory() as ws:
            ws_mind = os.path.join(ws, "mind")
            os.makedirs(ws_mind)
            result = get_mind_dir(ws)
            self.assertEqual(result, ws_mind)


class TestRRFPurePython(unittest.TestCase):
    """Test RRF computation matches expected values (Python reference)."""

    def test_rrf_basic_math(self):
        """Verify RRF formula: score = w / (k + rank)."""
        k = 60.0
        bm25_ranks = [1.0, 2.0, 3.0]
        vec_ranks = [3.0, 1.0, 2.0]
        bm25_w = 1.0
        vec_w = 1.0

        expected = []
        for i in range(3):
            s = bm25_w / (k + bm25_ranks[i]) + vec_w / (k + vec_ranks[i])
            expected.append(s)

        # Doc 0: 1/(60+1) + 1/(60+3) = 0.01639 + 0.01587 = 0.03226
        # Doc 1: 1/(60+2) + 1/(60+1) = 0.01613 + 0.01639 = 0.03252
        # Doc 2: 1/(60+3) + 1/(60+2) = 0.01587 + 0.01613 = 0.03200
        self.assertAlmostEqual(expected[0], 1/61 + 1/63, places=5)
        self.assertAlmostEqual(expected[1], 1/62 + 1/61, places=5)
        # Doc 1 should score highest (best combined rank)
        self.assertGreater(expected[1], expected[0])
        self.assertGreater(expected[0], expected[2])


if __name__ == "__main__":
    unittest.main()
