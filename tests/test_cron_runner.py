#!/usr/bin/env python3
"""Tests for cron_runner.py — periodic job orchestration, config loading, subprocess dispatch."""

import json
import os
import subprocess
import tempfile
import unittest
from unittest import mock

from mind_mem.cron_runner import (
    ALL_JOBS,
    JOB_DEFS,
    is_job_enabled,
    load_config,
    main,
    print_cron_instructions,
    run_job,
)


class TestLoadConfig(unittest.TestCase):
    """Tests for load_config — workspace config loading."""

    def test_valid_config(self):
        """Valid mind-mem.json is parsed correctly."""
        with tempfile.TemporaryDirectory() as ws:
            cfg = {"auto_ingest": {"enabled": True, "transcript_scan": False}}
            with open(os.path.join(ws, "mind-mem.json"), "w") as f:
                json.dump(cfg, f)
            result = load_config(ws)
            self.assertEqual(result, cfg)

    def test_missing_file_returns_empty(self):
        """Missing config file returns empty dict (graceful default)."""
        with tempfile.TemporaryDirectory() as ws:
            result = load_config(ws)
            self.assertEqual(result, {})

    def test_malformed_json_returns_empty(self):
        """Malformed JSON returns empty dict without raising."""
        with tempfile.TemporaryDirectory() as ws:
            with open(os.path.join(ws, "mind-mem.json"), "w") as f:
                f.write("{not valid json")
            result = load_config(ws)
            self.assertEqual(result, {})

    def test_empty_config_file(self):
        """Empty JSON object in config is returned as-is."""
        with tempfile.TemporaryDirectory() as ws:
            with open(os.path.join(ws, "mind-mem.json"), "w") as f:
                json.dump({}, f)
            result = load_config(ws)
            self.assertEqual(result, {})

    def test_nonexistent_workspace(self):
        """Non-existent workspace path returns empty dict."""
        result = load_config("/nonexistent/workspace/path/xyz")
        self.assertEqual(result, {})


class TestIsJobEnabled(unittest.TestCase):
    """Tests for is_job_enabled — feature-toggle logic."""

    def test_all_enabled_by_default(self):
        """With empty config, all jobs default to enabled."""
        for job_name in ALL_JOBS:
            self.assertTrue(is_job_enabled({}, job_name))

    def test_auto_ingest_disabled_globally(self):
        """Setting auto_ingest.enabled=false disables ALL jobs."""
        config = {"auto_ingest": {"enabled": False}}
        for job_name in ALL_JOBS:
            self.assertFalse(is_job_enabled(config, job_name))

    def test_individual_job_disabled(self):
        """Disabling a single job leaves the others enabled."""
        config = {"auto_ingest": {"enabled": True, "transcript_scan": False}}
        self.assertFalse(is_job_enabled(config, "transcript_scan"))
        self.assertTrue(is_job_enabled(config, "entity_ingest"))
        self.assertTrue(is_job_enabled(config, "intel_scan"))

    def test_global_enabled_individual_disabled(self):
        """Global enabled=true, but individual toggle set to false."""
        config = {"auto_ingest": {"enabled": True, "intel_scan": False}}
        self.assertFalse(is_job_enabled(config, "intel_scan"))
        self.assertTrue(is_job_enabled(config, "transcript_scan"))

    def test_unknown_job_defaults_enabled(self):
        """Unknown job name defaults to enabled (no key = True)."""
        self.assertTrue(is_job_enabled({}, "unknown_future_job"))


class TestRunJob(unittest.TestCase):
    """Tests for run_job — subprocess dispatch with timeout handling."""

    @mock.patch("mind_mem.cron_runner.subprocess.run")
    @mock.patch("mind_mem.cron_runner.os.path.isfile", return_value=True)
    def test_success(self, _mock_isfile, mock_run):
        """Successful job returns status='ok' with duration."""
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="done", stderr="")
        result = run_job("transcript_scan", "/tmp/ws")
        self.assertEqual(result["job"], "transcript_scan")
        self.assertEqual(result["status"], "ok")
        self.assertIn("duration_ms", result)

    @mock.patch("mind_mem.cron_runner.subprocess.run")
    @mock.patch("mind_mem.cron_runner.os.path.isfile", return_value=True)
    def test_failure(self, _mock_isfile, mock_run):
        """Failed job (nonzero exit) returns status='failed' with stderr."""
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="error: out of memory"
        )
        result = run_job("entity_ingest", "/tmp/ws")
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["returncode"], 1)
        self.assertIn("out of memory", result["stderr"])

    @mock.patch("mind_mem.cron_runner.subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="x", timeout=120))
    @mock.patch("mind_mem.cron_runner.os.path.isfile", return_value=True)
    def test_timeout(self, _mock_isfile, _mock_run):
        """Timed-out job returns status='timeout'."""
        result = run_job("intel_scan", "/tmp/ws")
        self.assertEqual(result["status"], "timeout")
        self.assertEqual(result["job"], "intel_scan")

    def test_missing_script_file(self):
        """Non-existent script path returns status='error'."""
        with mock.patch("mind_mem.cron_runner.SCRIPTS_DIR", "/nonexistent/dir"):
            result = run_job("transcript_scan", "/tmp/ws")
            self.assertEqual(result["status"], "error")
            self.assertIn("script not found", result["error"])

    @mock.patch("mind_mem.cron_runner.subprocess.run", side_effect=OSError("permission denied"))
    @mock.patch("mind_mem.cron_runner.os.path.isfile", return_value=True)
    def test_os_error(self, _mock_isfile, _mock_run):
        """OSError during subprocess launch returns status='error'."""
        result = run_job("transcript_scan", "/tmp/ws")
        self.assertEqual(result["status"], "error")
        self.assertIn("permission denied", result["error"])


class TestPrintCronInstructions(unittest.TestCase):
    """Tests for print_cron_instructions — crontab output validation."""

    def test_output_contains_paths_and_schedule(self):
        """Output includes workspace path, script path, and cron schedules."""
        ws_input = "/home/user/workspace"
        with mock.patch("builtins.print") as mock_print:
            print_cron_instructions(ws_input)
        calls = [str(c) for c in mock_print.call_args_list]
        combined = "\n".join(calls)

        # Verify workspace appears in output (path may be normalized on Windows)
        ws_normalized = os.path.abspath(ws_input)
        self.assertIn(ws_normalized, combined)
        # Verify cron schedule patterns
        self.assertIn("0 */6 * * *", combined)  # transcript_scan: every 6h
        self.assertIn("0 3 * * *", combined)  # entity_ingest + intel_scan: 3AM daily
        # Verify script reference
        self.assertIn("cron_runner.py", combined)
        # Verify logger pipe
        self.assertIn("logger -t mind-mem", combined)

    def test_output_contains_crontab_header(self):
        """Output starts with a crontab usage comment."""
        with mock.patch("builtins.print") as mock_print:
            print_cron_instructions("/tmp/ws")
        first_call = str(mock_print.call_args_list[0])
        self.assertIn("crontab -e", first_call)


class TestMain(unittest.TestCase):
    """Tests for main() — CLI entry-point integration."""

    @mock.patch("sys.argv", ["cron_runner", "/tmp/ws", "--install-cron"])
    @mock.patch("builtins.print")
    def test_install_cron_flag(self, mock_print):
        """--install-cron prints instructions and returns 0."""
        ret = main()
        self.assertEqual(ret, 0)
        calls = "\n".join(str(c) for c in mock_print.call_args_list)
        self.assertIn("crontab", calls)

    @mock.patch("sys.argv", ["cron_runner", "/tmp/ws", "--job", "transcript_scan"])
    @mock.patch("mind_mem.cron_runner.run_job")
    @mock.patch("mind_mem.cron_runner.load_config", return_value={})
    def test_single_job(self, _mock_cfg, mock_run):
        """--job transcript_scan runs only that one job."""
        mock_run.return_value = {"job": "transcript_scan", "status": "ok", "duration_ms": 50}
        ret = main()
        self.assertEqual(ret, 0)
        mock_run.assert_called_once_with("transcript_scan", os.path.abspath("/tmp/ws"))

    @mock.patch("sys.argv", ["cron_runner", "/tmp/ws", "--job", "all"])
    @mock.patch("mind_mem.cron_runner.run_job")
    @mock.patch("mind_mem.cron_runner.load_config", return_value={})
    def test_all_jobs(self, _mock_cfg, mock_run):
        """--job all dispatches every defined job."""
        mock_run.return_value = {"job": "x", "status": "ok", "duration_ms": 10}
        ret = main()
        self.assertEqual(ret, 0)
        self.assertEqual(mock_run.call_count, len(ALL_JOBS))

    @mock.patch("sys.argv", ["cron_runner", "/tmp/ws", "--job", "all"])
    @mock.patch("mind_mem.cron_runner.run_job")
    @mock.patch("mind_mem.cron_runner.load_config", return_value={"auto_ingest": {"enabled": False}})
    def test_all_jobs_disabled(self, _mock_cfg, mock_run):
        """With auto_ingest disabled, no jobs are dispatched."""
        ret = main()
        self.assertEqual(ret, 0)
        mock_run.assert_not_called()

    @mock.patch("sys.argv", ["cron_runner", "/tmp/ws", "--job", "all"])
    @mock.patch("mind_mem.cron_runner.run_job")
    @mock.patch("mind_mem.cron_runner.load_config", return_value={})
    def test_failed_job_returns_nonzero(self, _mock_cfg, mock_run):
        """If any job fails, main() returns 1."""
        mock_run.return_value = {"job": "x", "status": "failed", "returncode": 1, "stderr": "err"}
        ret = main()
        self.assertEqual(ret, 1)


class TestJobDefinitions(unittest.TestCase):
    """Smoke tests for job definition constants."""

    def test_all_jobs_have_defs(self):
        """ALL_JOBS matches keys in JOB_DEFS."""
        self.assertEqual(set(ALL_JOBS), set(JOB_DEFS.keys()))

    def test_job_defs_structure(self):
        """Each job def is a 3-tuple (script, args, toggle)."""
        for name, definition in JOB_DEFS.items():
            self.assertEqual(len(definition), 3, f"Bad tuple length for {name}")
            script, args, toggle = definition
            self.assertTrue(script.endswith(".py"), f"{name}: script must be .py")
            self.assertIsInstance(args, list, f"{name}: args must be a list")
            self.assertIsInstance(toggle, str, f"{name}: toggle must be a str")


if __name__ == "__main__":
    unittest.main()
