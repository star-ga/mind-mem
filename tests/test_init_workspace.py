"""Tests for init_workspace — config validation and workspace scaffolding."""

import contextlib
import io
import json
import logging
import os
import tempfile
import unittest
import unittest.mock

from mind_mem.init_workspace import (  # noqa: E402
    _RECALL_RANGES,
    DEFAULT_CONFIG,
    SUPPORTED_BACKENDS,
    _build_config,
    _validate_config,
    init,
    load_config,
    main,
)


class TestValidateConfigClamping(unittest.TestCase):
    """_validate_config clamps out-of-range recall values."""

    def test_clamp_bm25_k1_too_high(self):
        cfg = {"recall": {"bm25_k1": 99.0}}
        result = _validate_config(cfg)
        self.assertAlmostEqual(result["recall"]["bm25_k1"], 3.0)

    def test_clamp_bm25_k1_too_low(self):
        cfg = {"recall": {"bm25_k1": -1.0}}
        result = _validate_config(cfg)
        self.assertAlmostEqual(result["recall"]["bm25_k1"], 0.5)

    def test_clamp_bm25_b_too_high(self):
        cfg = {"recall": {"bm25_b": 2.0}}
        result = _validate_config(cfg)
        self.assertAlmostEqual(result["recall"]["bm25_b"], 1.0)

    def test_clamp_limit_too_low(self):
        cfg = {"recall": {"limit": 0}}
        result = _validate_config(cfg)
        self.assertEqual(result["recall"]["limit"], 1)

    def test_clamp_limit_too_high(self):
        cfg = {"recall": {"limit": 5000}}
        result = _validate_config(cfg)
        self.assertEqual(result["recall"]["limit"], 1000)

    def test_clamp_rrf_k_too_low(self):
        cfg = {"recall": {"rrf_k": 0}}
        result = _validate_config(cfg)
        self.assertEqual(result["recall"]["rrf_k"], 1)

    def test_clamp_rrf_k_too_high(self):
        cfg = {"recall": {"rrf_k": 999}}
        result = _validate_config(cfg)
        self.assertEqual(result["recall"]["rrf_k"], 200)

    def test_clamp_bm25_weight_too_high(self):
        cfg = {"recall": {"bm25_weight": 50.0}}
        result = _validate_config(cfg)
        self.assertAlmostEqual(result["recall"]["bm25_weight"], 10.0)

    def test_clamp_vector_weight_negative(self):
        cfg = {"recall": {"vector_weight": -1.0}}
        result = _validate_config(cfg)
        self.assertAlmostEqual(result["recall"]["vector_weight"], 0.0)

    def test_clamp_recency_weight_too_high(self):
        cfg = {"recall": {"recency_weight": 5.0}}
        result = _validate_config(cfg)
        self.assertAlmostEqual(result["recall"]["recency_weight"], 1.0)

    def test_clamp_top_k_too_low(self):
        cfg = {"recall": {"top_k": 0}}
        result = _validate_config(cfg)
        self.assertEqual(result["recall"]["top_k"], 1)

    def test_clamp_top_k_too_high(self):
        cfg = {"recall": {"top_k": 500}}
        result = _validate_config(cfg)
        self.assertEqual(result["recall"]["top_k"], 200)


class TestValidateConfigPassthrough(unittest.TestCase):
    """Values within range pass through unchanged."""

    def test_valid_bm25_k1(self):
        cfg = {"recall": {"bm25_k1": 1.5}}
        result = _validate_config(cfg)
        self.assertAlmostEqual(result["recall"]["bm25_k1"], 1.5)

    def test_valid_limit(self):
        cfg = {"recall": {"limit": 50}}
        result = _validate_config(cfg)
        self.assertEqual(result["recall"]["limit"], 50)

    def test_valid_top_k(self):
        cfg = {"recall": {"top_k": 18}}
        result = _validate_config(cfg)
        self.assertEqual(result["recall"]["top_k"], 18)

    def test_boundary_values_accepted(self):
        """Exact boundary values should not be clamped."""
        for key, (lo, hi, _default) in _RECALL_RANGES.items():
            cfg = {"recall": {key: lo}}
            result = _validate_config(cfg)
            self.assertAlmostEqual(result["recall"][key], lo, msg=f"{key} lower bound")
            cfg = {"recall": {key: hi}}
            result = _validate_config(cfg)
            self.assertAlmostEqual(result["recall"][key], hi, msg=f"{key} upper bound")


class TestValidateConfigEdgeCases(unittest.TestCase):
    """Edge cases: missing recall, non-numeric values, no recall keys."""

    def test_no_recall_section(self):
        cfg = {"version": "1.2.0"}
        result = _validate_config(cfg)
        self.assertEqual(result, cfg)

    def test_recall_not_dict(self):
        cfg = {"recall": "invalid"}
        result = _validate_config(cfg)
        self.assertEqual(result["recall"], "invalid")

    def test_non_numeric_value_replaced_with_default(self):
        cfg = {"recall": {"bm25_k1": "not_a_number"}}
        result = _validate_config(cfg)
        self.assertAlmostEqual(result["recall"]["bm25_k1"], 1.2)

    def test_unrelated_keys_preserved(self):
        cfg = {"recall": {"backend": "sqlite", "bm25_k1": 1.5}}
        result = _validate_config(cfg)
        self.assertEqual(result["recall"]["backend"], "sqlite")
        self.assertAlmostEqual(result["recall"]["bm25_k1"], 1.5)

    def test_empty_recall(self):
        cfg = {"recall": {}}
        result = _validate_config(cfg)
        self.assertEqual(result["recall"], {})

    def test_mutates_in_place(self):
        cfg = {"recall": {"limit": 5000}}
        result = _validate_config(cfg)
        self.assertIs(result, cfg)
        self.assertEqual(cfg["recall"]["limit"], 1000)


class TestValidateConfigLogging(unittest.TestCase):
    """Verify that warnings are logged for clamped values."""

    def test_clamped_value_logs_warning(self):
        cfg = {"recall": {"bm25_k1": 99.0}}
        with self.assertLogs("mind-mem.init_workspace", level="WARNING") as cm:
            _validate_config(cfg)
        self.assertTrue(any("config_value_clamped" in msg for msg in cm.output))
        self.assertTrue(any("bm25_k1" in msg for msg in cm.output))

    def test_non_numeric_logs_warning(self):
        cfg = {"recall": {"limit": "banana"}}
        with self.assertLogs("mind-mem.init_workspace", level="WARNING") as cm:
            _validate_config(cfg)
        self.assertTrue(any("config_value_invalid" in msg for msg in cm.output))

    def test_valid_value_no_warning(self):
        cfg = {"recall": {"bm25_k1": 1.2}}
        logger = logging.getLogger("mind-mem.init_workspace")
        with unittest.mock.patch.object(logger, "warning") as mock_warn:
            _validate_config(cfg)
            mock_warn.assert_not_called()


class TestValidateConfigIntTypes(unittest.TestCase):
    """Integer-range keys should produce int values."""

    def test_limit_stays_int(self):
        cfg = {"recall": {"limit": 50}}
        result = _validate_config(cfg)
        self.assertIsInstance(result["recall"]["limit"], int)

    def test_rrf_k_stays_int(self):
        cfg = {"recall": {"rrf_k": 60}}
        result = _validate_config(cfg)
        self.assertIsInstance(result["recall"]["rrf_k"], int)

    def test_top_k_stays_int(self):
        cfg = {"recall": {"top_k": 18}}
        result = _validate_config(cfg)
        self.assertIsInstance(result["recall"]["top_k"], int)

    def test_bm25_k1_stays_float(self):
        cfg = {"recall": {"bm25_k1": 1.2}}
        result = _validate_config(cfg)
        self.assertIsInstance(result["recall"]["bm25_k1"], float)


class TestLoadConfig(unittest.TestCase):
    """Tests for load_config() which loads + validates mind-mem.json."""

    def test_loads_and_validates(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as ws:
            cfg = {"recall": {"bm25_k1": 99.0, "backend": "bm25"}}
            with open(os.path.join(ws, "mind-mem.json"), "w") as f:
                json.dump(cfg, f)
            result = load_config(ws)
            self.assertAlmostEqual(result["recall"]["bm25_k1"], 3.0)

    def test_missing_file_returns_default(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as ws:
            result = load_config(ws)
            self.assertEqual(result["version"], DEFAULT_CONFIG["version"])

    def test_malformed_json_returns_default(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as ws:
            with open(os.path.join(ws, "mind-mem.json"), "w") as f:
                f.write("{bad json")
            result = load_config(ws)
            self.assertEqual(result["version"], DEFAULT_CONFIG["version"])


class TestInitWorkspace(unittest.TestCase):
    """Test that init() scaffolds a workspace correctly."""

    def test_creates_config(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as ws:
            created, skipped = init(ws)
            config_path = os.path.join(ws, "mind-mem.json")
            self.assertTrue(os.path.exists(config_path))
            with open(config_path) as f:
                cfg = json.load(f)
            self.assertEqual(cfg["version"], DEFAULT_CONFIG["version"])

    def test_does_not_overwrite(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as ws:
            config_path = os.path.join(ws, "mind-mem.json")
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump({"custom": True}, f)
            init(ws)
            with open(config_path) as f:
                cfg = json.load(f)
            self.assertTrue(cfg.get("custom"))


class TestBuildConfig(unittest.TestCase):
    """_build_config produces a backend-correct config (no DB required)."""

    def test_markdown_has_no_block_store(self):
        """The default backend must not add a block_store section."""
        cfg = _build_config()
        self.assertNotIn("block_store", cfg)
        self.assertEqual(cfg["recall"]["backend"], "bm25")

    def test_markdown_equals_default_config(self):
        """The default config must be byte-identical to DEFAULT_CONFIG."""
        self.assertEqual(_build_config(backend="markdown"), DEFAULT_CONFIG)

    def test_does_not_mutate_default_config(self):
        """Building a postgres config must not mutate the module template."""
        before = json.dumps(DEFAULT_CONFIG, sort_keys=True)
        _build_config(backend="postgres", dsn="postgresql://u:p@h/db")
        after = json.dumps(DEFAULT_CONFIG, sort_keys=True)
        self.assertEqual(before, after)
        self.assertNotIn("block_store", DEFAULT_CONFIG)

    def test_postgres_writes_block_store(self):
        cfg = _build_config(backend="postgres", dsn="postgresql://u:p@h:5432/db", schema="myschema")
        self.assertEqual(
            cfg["block_store"],
            {"backend": "postgres", "dsn": "postgresql://u:p@h:5432/db", "schema": "myschema"},
        )
        # recall must point at the SQLite FTS cache (mirrors the PG store),
        # not the empty markdown corpus.
        self.assertEqual(cfg["recall"]["backend"], "sqlite")

    def test_postgres_default_schema(self):
        cfg = _build_config(backend="postgres", dsn="postgresql://u:p@h/db")
        self.assertEqual(cfg["block_store"]["schema"], "mind_mem")

    def test_postgres_requires_dsn(self):
        with self.assertRaises(ValueError):
            _build_config(backend="postgres")

    def test_encrypted_block_store(self):
        cfg = _build_config(backend="encrypted")
        self.assertEqual(cfg["block_store"], {"backend": "encrypted"})

    def test_unknown_backend_rejected(self):
        with self.assertRaises(ValueError):
            _build_config(backend="nosuch")

    def test_supported_backends_constant(self):
        self.assertIn("markdown", SUPPORTED_BACKENDS)
        self.assertIn("postgres", SUPPORTED_BACKENDS)


class TestInitBackendAware(unittest.TestCase):
    """init() honours the backend kwarg while keeping the default path intact."""

    def test_default_init_omits_block_store(self):
        """Default markdown init writes a config with no block_store (SQLite path)."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as ws:
            init(ws)
            with open(os.path.join(ws, "mind-mem.json")) as f:
                cfg = json.load(f)
            self.assertNotIn("block_store", cfg)

    def test_postgres_init_writes_block_store(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as ws:
            init(ws, backend="postgres", dsn="postgresql://u:p@h/db", schema="s1")
            with open(os.path.join(ws, "mind-mem.json")) as f:
                cfg = json.load(f)
            self.assertEqual(cfg["block_store"]["backend"], "postgres")
            self.assertEqual(cfg["block_store"]["dsn"], "postgresql://u:p@h/db")
            self.assertEqual(cfg["block_store"]["schema"], "s1")
            self.assertEqual(cfg["recall"]["backend"], "sqlite")

    def test_postgres_init_without_dsn_raises_before_writing(self):
        """A bad backend selection must not leave a half-built workspace."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as ws:
            target = os.path.join(ws, "fresh")
            with self.assertRaises(ValueError):
                init(target, backend="postgres")
            # No directories/config should have been created.
            self.assertFalse(os.path.exists(os.path.join(target, "mind-mem.json")))
            self.assertFalse(os.path.isdir(os.path.join(target, "decisions")))


class TestMainCli(unittest.TestCase):
    """main() argument handling: argparse rejects junk, env defaults work."""

    def _run_main(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                rc = main(argv)
            except SystemExit as exc:
                rc = exc.code
        return rc, out.getvalue(), err.getvalue()

    def test_default_workspace_markdown(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as ws:
            rc, _out, _err = self._run_main([ws])
            self.assertEqual(rc, 0)
            with open(os.path.join(ws, "mind-mem.json")) as f:
                cfg = json.load(f)
            self.assertNotIn("block_store", cfg)

    def test_help_does_not_create_junk_dir(self):
        """`--help` must exit cleanly, not create a './--help' directory."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as cwd:
            prev = os.getcwd()
            os.chdir(cwd)
            try:
                rc, _out, _err = self._run_main(["--help"])
            finally:
                os.chdir(prev)
            self.assertEqual(rc, 0)
            self.assertFalse(os.path.exists(os.path.join(cwd, "--help")))

    def test_unknown_flag_rejected(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as ws:
            rc, _out, _err = self._run_main([ws, "--bogus"])
            self.assertEqual(rc, 2)

    def test_postgres_flag_without_dsn_errors(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as ws:
            target = os.path.join(ws, "fresh")
            rc, _out, _err = self._run_main([target, "--backend", "postgres"])
            self.assertEqual(rc, 2)
            self.assertFalse(os.path.exists(os.path.join(target, "mind-mem.json")))

    def test_postgres_via_cli_flags(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as ws:
            rc, _out, _err = self._run_main([ws, "--backend", "postgres", "--dsn", "postgresql://u:p@h/db", "--schema", "sch"])
            self.assertEqual(rc, 0)
            with open(os.path.join(ws, "mind-mem.json")) as f:
                cfg = json.load(f)
            self.assertEqual(cfg["block_store"]["backend"], "postgres")
            self.assertEqual(cfg["block_store"]["schema"], "sch")

    def test_postgres_via_env(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as ws:
            env = {
                "MIND_MEM_BACKEND": "postgres",
                "MIND_MEM_DSN": "postgresql://u:p@h/db",
                "MIND_MEM_SCHEMA": "envsch",
            }
            with unittest.mock.patch.dict(os.environ, env):
                rc, _out, _err = self._run_main([ws])
            self.assertEqual(rc, 0)
            with open(os.path.join(ws, "mind-mem.json")) as f:
                cfg = json.load(f)
            self.assertEqual(cfg["block_store"]["backend"], "postgres")
            self.assertEqual(cfg["block_store"]["schema"], "envsch")


if __name__ == "__main__":
    unittest.main()
