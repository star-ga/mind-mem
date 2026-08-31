"""An unreadable toggle file must mean "run nothing", never "run everything".

``load_config`` collapses two different states into the same ``{}``: the config
is absent (defaults apply, everything runs) and the config exists but does not
parse. ``is_job_enabled`` then reads ``{}.get("enabled", True)`` and dispatches
every job — so an operator who had disabled ``auto_ingest`` and later left a
trailing comma in ``mind-mem.json`` got every 120-second job spawned on the next
tick, with a ``3 ok`` summary and exit 0.

The load-time warning is not the fix; the wrong action still happened. The
runner now asks :func:`config_read_error` first and fails closed.
"""

from __future__ import annotations

import json
import os
import tempfile
from unittest import mock

import pytest

from mind_mem.cron_runner import ALL_JOBS, config_read_error, main


@pytest.fixture
def workspace():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as ws:
        yield ws


def _write_config(ws: str, raw: str) -> None:
    with open(os.path.join(ws, "mind-mem.json"), "w", encoding="utf-8") as fh:
        fh.write(raw)


class TestConfigReadError:
    """Absent is fine; present-and-unreadable is not."""

    def test_absent_config_is_not_an_error(self, workspace) -> None:
        assert config_read_error(workspace) is None

    def test_nonexistent_workspace_is_not_an_error(self) -> None:
        assert config_read_error("/nonexistent/workspace/path/xyz") is None

    def test_valid_config_is_not_an_error(self, workspace) -> None:
        _write_config(workspace, json.dumps({"auto_ingest": {"enabled": False}}))
        assert config_read_error(workspace) is None

    def test_trailing_comma_is_an_error(self, workspace) -> None:
        _write_config(workspace, '{"auto_ingest": {"enabled": false},}')
        error = config_read_error(workspace)
        assert error is not None
        assert "JSONDecodeError" in error

    def test_non_object_top_level_is_an_error(self, workspace) -> None:
        """``dict(json.load(f))`` on a scalar raises out of ``load_config``."""
        _write_config(workspace, "42")
        error = config_read_error(workspace)
        assert error is not None
        assert "expected an object" in error


class TestRunnerFailsClosed:
    """The dispatch decision, not just a log line."""

    def test_unparseable_config_dispatches_nothing(self, workspace, capsys) -> None:
        """Before the fix this ran every job in ALL_JOBS and returned 0."""
        _write_config(workspace, '{"auto_ingest": {"enabled": false},}')
        with mock.patch("sys.argv", ["cron_runner", workspace, "--job", "all"]):
            with mock.patch("mind_mem.cron_runner.run_job") as run_job:
                ret = main()
        run_job.assert_not_called()
        assert ret == 2
        assert "refusing to run" in capsys.readouterr().out

    def test_unparseable_config_blocks_a_single_named_job_too(self, workspace) -> None:
        _write_config(workspace, "{not valid json")
        with mock.patch("sys.argv", ["cron_runner", workspace, "--job", "intel_scan"]):
            with mock.patch("mind_mem.cron_runner.run_job") as run_job:
                ret = main()
        run_job.assert_not_called()
        assert ret == 2

    def test_valid_config_still_dispatches(self, workspace) -> None:
        """The gate must not turn every tick into a refusal."""
        _write_config(workspace, json.dumps({"auto_ingest": {"enabled": True}}))
        with mock.patch("sys.argv", ["cron_runner", workspace, "--job", "all"]):
            with mock.patch("mind_mem.cron_runner.run_job") as run_job:
                run_job.return_value = {"job": "x", "status": "ok", "duration_ms": 1}
                ret = main()
        assert run_job.call_count == len(ALL_JOBS)
        assert ret == 0

    def test_absent_config_still_dispatches(self, workspace) -> None:
        with mock.patch("sys.argv", ["cron_runner", workspace, "--job", "all"]):
            with mock.patch("mind_mem.cron_runner.run_job") as run_job:
                run_job.return_value = {"job": "x", "status": "ok", "duration_ms": 1}
                ret = main()
        assert run_job.call_count == len(ALL_JOBS)
        assert ret == 0

    def test_disabled_config_is_still_honoured(self, workspace) -> None:
        _write_config(workspace, json.dumps({"auto_ingest": {"enabled": False}}))
        with mock.patch("sys.argv", ["cron_runner", workspace, "--job", "all"]):
            with mock.patch("mind_mem.cron_runner.run_job") as run_job:
                ret = main()
        run_job.assert_not_called()
        assert ret == 0
