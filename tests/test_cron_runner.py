#!/usr/bin/env python3
"""Tests for cron_runner.py — periodic job orchestration, config loading, subprocess dispatch."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
import uuid
from unittest import mock

import pytest

from mind_mem import init_workspace
from mind_mem.cron_runner import (
    ALL_JOBS,
    JOB_DEFS,
    PACKAGE,
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
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as ws:
            cfg = {"auto_ingest": {"enabled": True, "transcript_scan": False}}
            with open(os.path.join(ws, "mind-mem.json"), "w") as f:
                json.dump(cfg, f)
            result = load_config(ws)
            self.assertEqual(result, cfg)

    def test_missing_file_returns_empty(self):
        """Missing config file returns empty dict (graceful default)."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as ws:
            result = load_config(ws)
            self.assertEqual(result, {})

    def test_malformed_json_returns_empty(self):
        """Malformed JSON returns empty dict without raising."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as ws:
            with open(os.path.join(ws, "mind-mem.json"), "w") as f:
                f.write("{not valid json")
            result = load_config(ws)
            self.assertEqual(result, {})

    def test_empty_config_file(self):
        """Empty JSON object in config is returned as-is."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as ws:
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
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="error: out of memory")
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
        # Extract actual string args (not repr) to avoid backslash-escaping issues
        combined = "\n".join(str(a) for c in mock_print.call_args_list for a in c.args)

        # Verify workspace appears in output (path may be normalized on Windows)
        ws_normalized = os.path.abspath(ws_input)
        self.assertIn(ws_normalized, combined)
        # Verify cron schedule patterns
        self.assertIn("0 */6 * * *", combined)  # transcript_scan: every 6h
        self.assertIn("0 3 * * *", combined)  # entity_ingest + intel_scan: 3AM daily
        # Verify the runner is invoked as a module (-m), not a bare script path,
        # so the package context (and relative imports) resolve.
        self.assertIn("-m mind_mem.cron_runner", combined)
        # A bare ``cron_runner.py`` script path must NOT appear — running the
        # package module as a top-level script breaks relative imports.
        self.assertNotIn("python3 /", combined)
        # Verify logger pipe
        self.assertIn("logger -t mind-mem", combined)

    def test_output_contains_crontab_header(self):
        """Output starts with a crontab usage comment."""
        with mock.patch("builtins.print") as mock_print:
            print_cron_instructions("/tmp/ws")
        first_arg = str(mock_print.call_args_list[0].args[0])
        self.assertIn("crontab -e", first_arg)


class TestMain(unittest.TestCase):
    """Tests for main() — CLI entry-point integration."""

    @mock.patch("sys.argv", ["cron_runner", "/tmp/ws", "--install-cron"])
    @mock.patch("builtins.print")
    def test_install_cron_flag(self, mock_print):
        """--install-cron prints instructions and returns 0."""
        ret = main()
        self.assertEqual(ret, 0)
        combined = "\n".join(str(a) for c in mock_print.call_args_list for a in c.args)
        self.assertIn("crontab", combined)

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
        """Each job def is a 3-tuple (module, args, toggle).

        ``module`` is a package-relative module name with NO ``.py`` suffix and
        NO path separators — jobs are run via ``python -m mind_mem.<module>``.
        """
        for name, definition in JOB_DEFS.items():
            self.assertEqual(len(definition), 3, f"Bad tuple length for {name}")
            module, args, toggle = definition
            self.assertIsInstance(module, str, f"{name}: module must be a str")
            self.assertFalse(module.endswith(".py"), f"{name}: module must NOT carry a .py suffix")
            self.assertNotIn(os.sep, module, f"{name}: module must be a bare module name, not a path")
            self.assertNotIn("/", module, f"{name}: module must be a bare module name, not a path")
            self.assertIsInstance(args, list, f"{name}: args must be a list")
            self.assertIsInstance(toggle, str, f"{name}: toggle must be a str")


class TestRunJobModuleInvocation(unittest.TestCase):
    """Regression for bug 7: daemon jobs must be invoked as ``python -m mind_mem.<module>``.

    The job scripts (transcript_capture / entity_ingest / intel_scan) are package
    modules that use relative imports (``from .block_parser import ...``). Running
    them as bare script paths raised ``ImportError: attempted relative import with
    no known parent package``, silently breaking every daemon tick on ALL backends.
    """

    @mock.patch("mind_mem.cron_runner.subprocess.run")
    @mock.patch("mind_mem.cron_runner.os.path.isfile", return_value=True)
    def test_cmd_uses_module_runner_not_bare_script(self, _mock_isfile, mock_run):
        """run_job must build ``[python, -m, mind_mem.<module>, ws, *args]`` — never a bare path."""
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
        for job_name, (module, extra_args, _toggle) in JOB_DEFS.items():
            mock_run.reset_mock()
            run_job(job_name, "/tmp/ws")
            cmd = mock_run.call_args.args[0]
            self.assertEqual(cmd[0], sys.executable, f"{job_name}: must use sys.executable")
            self.assertEqual(cmd[1], "-m", f"{job_name}: must invoke via -m (module runner)")
            self.assertEqual(
                cmd[2],
                f"{PACKAGE}.{module}",
                f"{job_name}: must run the package module, not a script path",
            )
            self.assertEqual(cmd[3], "/tmp/ws", f"{job_name}: workspace must follow the module")
            self.assertEqual(cmd[4:], extra_args, f"{job_name}: extra args must follow the workspace")
            # No element may be a bare ``.py`` file path — that's the bug.
            for part in cmd:
                self.assertFalse(
                    str(part).endswith(".py"),
                    f"{job_name}: cmd must not contain a bare .py script path: {part!r}",
                )


def _init_real_workspace(ws: str, *, backend: str = "markdown", dsn=None, schema=None) -> None:
    """Init a real workspace (markdown or postgres) for the end-to-end daemon test."""
    init_workspace.init(ws, backend=backend, dsn=dsn, schema=schema)


def _assert_jobs_import_cleanly(ws: str) -> None:
    """Run every real daemon job via run_job and assert none crash on import.

    This exercises the actual subprocess path (NO mock): a relative-import
    failure surfaces as a non-zero exit with the ImportError in stderr.
    """
    from mind_mem.cron_runner import run_job as _run_job

    for job_name in ALL_JOBS:
        result = _run_job(job_name, ws)
        # The job may legitimately do nothing (empty corpus) but it MUST start,
        # i.e. it must not die at import time. Guard specifically against the
        # relative-import regression and any non-clean status.
        stderr = result.get("stderr", "") or ""
        assert "attempted relative import" not in stderr, f"{job_name}: relative-import regression — {stderr}"
        assert "ImportError" not in stderr, f"{job_name}: import failure — {stderr}"
        assert result["status"] == "ok", f"{job_name}: expected status 'ok', got {result['status']} (stderr: {stderr!r})"


def test_daemon_jobs_run_on_sqlite_backend(tmp_path):
    """E2E: all daemon jobs start cleanly on the default SQLite/markdown workspace."""
    ws = str(tmp_path / "ws-sqlite")
    _init_real_workspace(ws, backend="markdown")
    _assert_jobs_import_cleanly(ws)


# ─── Postgres parity (optional; skips gracefully without psycopg / DSN) ─────────

_PG_DSN_ENV = "MIND_MEM_TEST_PG_DSN"


def _pg_dsn():
    return os.environ.get(_PG_DSN_ENV)


def test_daemon_jobs_run_on_postgres_backend(tmp_path):
    """E2E: all daemon jobs start cleanly on a Postgres-backed workspace.

    Mirrors the tests/test_postgres_*.py pattern — uses a uniquely-named
    scratch schema and tears it down. Skips gracefully when psycopg or a live
    Postgres DSN is unavailable so SQLite-only CI stays green.
    """
    pytest.importorskip("psycopg", reason="psycopg not installed; skipping Postgres parity test")
    dsn = _pg_dsn()
    if not dsn:
        pytest.skip(f"{_PG_DSN_ENV} not set — no live Postgres available")

    from mind_mem.block_store_postgres import PostgresBlockStore

    schema = f"mm_cron_{uuid.uuid4().hex[:12]}"
    ws = str(tmp_path / "ws-pg")
    _init_real_workspace(ws, backend="postgres", dsn=dsn, schema=schema)

    store = PostgresBlockStore(dsn=dsn, schema=schema, workspace=ws)
    try:
        store._ensure_schema()
        # The fix is backend-independent (it is about how the daemon invokes the
        # job modules), so the jobs must import + run cleanly here too.
        _assert_jobs_import_cleanly(ws)
    finally:
        # Drop the scratch schema; never touch the production mind_mem schema.
        try:
            with psycopg_connect(dsn) as conn:  # type: ignore[name-defined]
                with conn.cursor() as cur:
                    cur.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
                conn.commit()
        except Exception:  # pragma: no cover — best-effort teardown
            pass
        try:
            store.close()
        except Exception:  # pragma: no cover
            pass


def psycopg_connect(dsn):
    """Thin import-time-deferred psycopg.connect wrapper for scratch-schema teardown."""
    import psycopg

    return psycopg.connect(dsn)


if __name__ == "__main__":
    unittest.main()
