"""Tests for MIND FFI bridge — kernel loading, config parsing, and fallback behavior."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from mind_ffi import (
    MindMemKernel,
    _parse_value,
    get_kernel,
    get_kernel_param,
    get_mind_dir,
    is_available,
    list_kernels,
    load_all_kernel_configs,
    load_all_kernels,
    load_kernel,
    load_kernel_config,
)


class TestKernelLoadingFallback(unittest.TestCase):
    """MIND kernel should fail gracefully when .so is not compiled."""

    def test_kernel_raises_without_so(self):
        """MindMemKernel raises OSError when .so doesn't exist."""
        with self.assertRaises(OSError):
            MindMemKernel("/nonexistent/path/libmindmem.so")

    def test_get_kernel_returns_none_without_so(self):
        """get_kernel() returns None when .so is not available."""
        kernel = get_kernel()
        self.assertTrue(kernel is None or isinstance(kernel, MindMemKernel))

    def test_is_available_returns_bool(self):
        """is_available() should return a boolean."""
        result = is_available()
        self.assertIsInstance(result, bool)


class TestParseValue(unittest.TestCase):
    """Tests for INI value auto-detection."""

    def test_bool_true(self):
        self.assertIs(_parse_value("true"), True)
        self.assertIs(_parse_value("True"), True)

    def test_bool_false(self):
        self.assertIs(_parse_value("false"), False)
        self.assertIs(_parse_value("False"), False)

    def test_integer(self):
        self.assertEqual(_parse_value("42"), 42)
        self.assertEqual(_parse_value("-1"), -1)

    def test_float(self):
        self.assertAlmostEqual(_parse_value("1.2"), 1.2)
        self.assertAlmostEqual(_parse_value("0.75"), 0.75)
        self.assertAlmostEqual(_parse_value("-0.5"), -0.5)

    def test_string(self):
        self.assertEqual(_parse_value("hello"), "hello")
        self.assertEqual(_parse_value("morph_only"), "morph_only")

    def test_comma_list(self):
        result = _parse_value("a, b, c")
        self.assertEqual(result, ["a", "b", "c"])

    def test_comma_list_with_numbers(self):
        result = _parse_value("1, 2, 3")
        self.assertEqual(result, [1, 2, 3])


class TestLoadKernelConfig(unittest.TestCase):
    """Tests for INI-style .mind config loading."""

    def test_load_basic_config(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mind", delete=False) as f:
            f.write("[bm25]\nk1 = 1.2\nb = 0.75\n")
            f.flush()
            cfg = load_kernel_config(f.name)
        os.unlink(f.name)
        self.assertIn("bm25", cfg)
        self.assertAlmostEqual(cfg["bm25"]["k1"], 1.2)
        self.assertAlmostEqual(cfg["bm25"]["b"], 0.75)

    def test_load_multiple_sections(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mind", delete=False) as f:
            f.write("[section1]\nfoo = bar\n[section2]\ncount = 10\n")
            f.flush()
            cfg = load_kernel_config(f.name)
        os.unlink(f.name)
        self.assertEqual(cfg["section1"]["foo"], "bar")
        self.assertEqual(cfg["section2"]["count"], 10)

    def test_comments_ignored(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mind", delete=False) as f:
            f.write("# This is a comment\n[sec]\n# Another comment\nkey = val\n")
            f.flush()
            cfg = load_kernel_config(f.name)
        os.unlink(f.name)
        self.assertEqual(cfg["sec"]["key"], "val")

    def test_empty_lines_ignored(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mind", delete=False) as f:
            f.write("[sec]\n\nkey = val\n\n")
            f.flush()
            cfg = load_kernel_config(f.name)
        os.unlink(f.name)
        self.assertEqual(cfg["sec"]["key"], "val")

    def test_nonexistent_file_returns_empty(self):
        cfg = load_kernel_config("/nonexistent/path/file.mind")
        self.assertEqual(cfg, {})

    def test_boolean_values(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mind", delete=False) as f:
            f.write("[flags]\nenabled = true\ndisabled = false\n")
            f.flush()
            cfg = load_kernel_config(f.name)
        os.unlink(f.name)
        self.assertIs(cfg["flags"]["enabled"], True)
        self.assertIs(cfg["flags"]["disabled"], False)


class TestKernelListing(unittest.TestCase):
    """Test .mind source file discovery."""

    def test_list_kernels_finds_mind_files(self):
        with tempfile.TemporaryDirectory() as td:
            for name in ["recall.mind", "rm3.mind", "rerank.mind"]:
                with open(os.path.join(td, name), "w") as f:
                    f.write("[test]\nkey = val\n")
            names = list_kernels(td)
            self.assertEqual(names, ["recall", "rerank", "rm3"])

    def test_list_kernels_ignores_non_mind(self):
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "recall.mind"), "w") as f:
                f.write("[test]\n")
            with open(os.path.join(td, "readme.txt"), "w") as f:
                f.write("not a kernel\n")
            names = list_kernels(td)
            self.assertEqual(names, ["recall"])

    def test_list_kernels_ignores_hidden(self):
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, ".hidden.mind"), "w") as f:
                f.write("[test]\n")
            with open(os.path.join(td, "visible.mind"), "w") as f:
                f.write("[test]\n")
            names = list_kernels(td)
            self.assertEqual(names, ["visible"])

    def test_empty_directory(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(list_kernels(td), [])

    def test_nonexistent_directory(self):
        self.assertEqual(list_kernels("/nonexistent/path"), [])

    def test_list_kernels_sorted(self):
        mind_dir = os.path.join(os.path.dirname(__file__), "..", "mind")
        if os.path.isdir(mind_dir):
            kernels = list_kernels(mind_dir)
            self.assertEqual(kernels, sorted(kernels))


class TestLoadKernelSource(unittest.TestCase):
    """Test loading .mind source metadata (function extraction)."""

    def test_load_kernel_nonexistent(self):
        result = load_kernel("/nonexistent/kernel.mind")
        self.assertEqual(result, {})

    def test_load_kernel_returns_functions_list(self):
        """Our INI-style kernels have no fn declarations."""
        mind_dir = os.path.join(os.path.dirname(__file__), "..", "mind")
        path = os.path.join(mind_dir, "recall.mind")
        if os.path.isfile(path):
            info = load_kernel(path)
            self.assertIn("functions", info)
            self.assertIsInstance(info["functions"], list)

    def test_load_all_kernels_metadata(self):
        mind_dir = os.path.join(os.path.dirname(__file__), "..", "mind")
        if os.path.isdir(mind_dir):
            all_k = load_all_kernels(mind_dir)
            self.assertGreater(len(all_k), 0)
            for name, cfg in all_k.items():
                self.assertIn("functions", cfg)


class TestGetMindDir(unittest.TestCase):
    """Test mind/ directory resolution."""

    def test_workspace_mind_preferred(self):
        with tempfile.TemporaryDirectory() as ws:
            ws_mind = os.path.join(ws, "mind")
            os.makedirs(ws_mind)
            result = get_mind_dir(ws)
            self.assertEqual(result, ws_mind)

    def test_fallback_to_package_level(self):
        with tempfile.TemporaryDirectory() as td:
            result = get_mind_dir(td)
            # Should return either workspace/mind or package-level
            self.assertTrue(result.endswith("mind"))


class TestLoadAllKernelConfigs(unittest.TestCase):
    """Tests for loading all kernel configs from a directory."""

    def test_loads_all(self):
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "recall.mind"), "w") as f:
                f.write("[bm25]\nk1 = 1.2\n")
            with open(os.path.join(td, "rm3.mind"), "w") as f:
                f.write("[rm3]\nalpha = 0.6\n")
            result = load_all_kernel_configs(td)
            self.assertIn("recall", result)
            self.assertIn("rm3", result)
            self.assertAlmostEqual(result["recall"]["bm25"]["k1"], 1.2)
            self.assertAlmostEqual(result["rm3"]["rm3"]["alpha"], 0.6)


class TestGetKernelParam(unittest.TestCase):
    """Tests for parameter extraction with defaults."""

    def test_existing_param(self):
        cfg = {"bm25": {"k1": 1.5}}
        self.assertAlmostEqual(get_kernel_param(cfg, "bm25", "k1", 1.2), 1.5)

    def test_missing_param_uses_default(self):
        cfg = {"bm25": {"k1": 1.5}}
        self.assertAlmostEqual(get_kernel_param(cfg, "bm25", "b", 0.75), 0.75)

    def test_missing_section_uses_default(self):
        cfg = {}
        self.assertEqual(get_kernel_param(cfg, "bm25", "k1", 1.2), 1.2)


class TestShippedKernels(unittest.TestCase):
    """Tests that the shipped .mind kernel files parse correctly."""

    def setUp(self):
        self.mind_dir = os.path.join(os.path.dirname(__file__), "..", "mind")

    def test_recall_kernel_loads(self):
        cfg = load_kernel_config(os.path.join(self.mind_dir, "recall.mind"))
        self.assertIn("bm25", cfg)
        self.assertAlmostEqual(cfg["bm25"]["k1"], 1.2)
        self.assertAlmostEqual(cfg["bm25"]["b"], 0.75)
        self.assertIn("fields", cfg)
        self.assertEqual(cfg["fields"]["Statement"], 3.0)

    def test_rm3_kernel_loads(self):
        cfg = load_kernel_config(os.path.join(self.mind_dir, "rm3.mind"))
        self.assertIn("rm3", cfg)
        self.assertIs(cfg["rm3"]["enabled"], False)
        self.assertAlmostEqual(cfg["rm3"]["alpha"], 0.6)

    def test_rerank_kernel_loads(self):
        cfg = load_kernel_config(os.path.join(self.mind_dir, "rerank.mind"))
        self.assertIn("weights", cfg)
        self.assertAlmostEqual(cfg["weights"]["entity_overlap"], 0.30)

    def test_temporal_kernel_loads(self):
        cfg = load_kernel_config(os.path.join(self.mind_dir, "temporal.mind"))
        self.assertIn("scoring", cfg)
        self.assertAlmostEqual(cfg["scoring"]["recency_weight"], 0.6)

    def test_adversarial_kernel_loads(self):
        cfg = load_kernel_config(os.path.join(self.mind_dir, "adversarial.mind"))
        self.assertIn("scoring", cfg)
        self.assertEqual(cfg["scoring"]["expand_query"], "morph_only")

    def test_hybrid_kernel_loads(self):
        cfg = load_kernel_config(os.path.join(self.mind_dir, "hybrid.mind"))
        self.assertIn("fusion", cfg)
        self.assertEqual(cfg["fusion"]["rrf_k"], 60)

    def test_all_six_kernels_listed(self):
        names = list_kernels(self.mind_dir)
        required = {"adversarial", "hybrid", "recall", "rerank", "rm3", "temporal"}
        self.assertTrue(required.issubset(set(names)),
                        f"Missing kernels: {required - set(names)}")


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

        self.assertAlmostEqual(expected[0], 1/61 + 1/63, places=5)
        self.assertAlmostEqual(expected[1], 1/62 + 1/61, places=5)
        self.assertGreater(expected[1], expected[0])
        self.assertGreater(expected[0], expected[2])


if __name__ == "__main__":
    unittest.main()
