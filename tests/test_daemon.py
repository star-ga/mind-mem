"""Tests for the v3.9 background daemon (`mm daemon`)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

from mind_mem.daemon import (
    DEFAULT_INTERVALS,
    Daemon,
    TaskConfig,
    load_daemon_config,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_config(workspace: Path, daemon_block: dict[str, Any]) -> None:
    config = {
        "version": "3.9.0",
        "workspace_path": str(workspace),
        "daemon": daemon_block,
    }
    (workspace / "mind-mem.json").write_text(json.dumps(config))


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    ws.mkdir()
    return ws


# ---------------------------------------------------------------------------
# load_daemon_config
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def test_missing_config_returns_defaults(self, tmp_path: Path) -> None:
        enabled, tasks = load_daemon_config(str(tmp_path))
        assert enabled is True  # daemon block absent => global default True
        assert {t.name for t in tasks} == set(DEFAULT_INTERVALS.keys())
        # All defaults are 0 (disabled).
        assert all(t.interval_seconds == 0 for t in tasks)

    def test_explicit_disabled(self, workspace: Path) -> None:
        _write_config(workspace, {"enabled": False})
        enabled, _ = load_daemon_config(str(workspace))
        assert enabled is False

    def test_per_task_interval(self, workspace: Path) -> None:
        _write_config(
            workspace,
            {
                "dream_cycle": {"auto_interval_seconds": 1800, "dry_run": True},
                "intel_scan": {"auto_interval_seconds": 21600},
            },
        )
        _, tasks = load_daemon_config(str(workspace))
        by_name = {t.name: t for t in tasks}
        assert by_name["dream_cycle"].interval_seconds == 1800
        assert by_name["dream_cycle"].extras == {"dry_run": True}
        assert by_name["dream_cycle"].enabled is True
        assert by_name["intel_scan"].interval_seconds == 21600
        assert by_name["entity_ingest"].interval_seconds == 0
        assert by_name["entity_ingest"].enabled is False

    def test_negative_interval_clamped_to_zero(self, workspace: Path) -> None:
        _write_config(workspace, {"dream_cycle": {"auto_interval_seconds": -5}})
        _, tasks = load_daemon_config(str(workspace))
        assert next(t for t in tasks if t.name == "dream_cycle").interval_seconds == 0

    def test_too_small_interval_clamped_to_minimum(self, workspace: Path) -> None:
        _write_config(workspace, {"dream_cycle": {"auto_interval_seconds": 5}})
        _, tasks = load_daemon_config(str(workspace))
        # Foot-gun guard: 0 < interval < 60 is bumped to 60.
        assert next(t for t in tasks if t.name == "dream_cycle").interval_seconds == 60

    def test_garbage_interval_falls_back_to_default(self, workspace: Path) -> None:
        _write_config(workspace, {"dream_cycle": {"auto_interval_seconds": "every-half-hour"}})
        _, tasks = load_daemon_config(str(workspace))
        # Default for dream_cycle is 0
        assert next(t for t in tasks if t.name == "dream_cycle").interval_seconds == 0

    def test_malformed_json_returns_defaults(self, workspace: Path) -> None:
        (workspace / "mind-mem.json").write_text("{ this is not json }")
        enabled, tasks = load_daemon_config(str(workspace))
        assert enabled is True
        assert {t.name for t in tasks} == set(DEFAULT_INTERVALS.keys())


# ---------------------------------------------------------------------------
# Daemon class — start/stop/dry-run lifecycle (no real workspace mutations)
# ---------------------------------------------------------------------------


class TestDaemonLifecycle:
    def test_empty_workspace_rejected(self) -> None:
        with pytest.raises(ValueError, match="workspace"):
            Daemon("")

    def test_start_with_no_enabled_tasks_raises(self, workspace: Path) -> None:
        d = Daemon(str(workspace))
        with pytest.raises(ValueError, match="no enabled tasks"):
            d.start([TaskConfig("dream_cycle", 0, {})])

    def test_dry_run_records_last_run(self, workspace: Path) -> None:
        d = Daemon(str(workspace), dry_run=True)
        # Use the minimum interval (60s) but stop immediately after one tick by
        # invoking _tick directly — keeps the test fast and deterministic.
        task = TaskConfig("dream_cycle", 60, {})
        d._tick(task, lambda ws, extras: {"called": True})  # type: ignore[arg-type]
        ts = d.last_run("dream_cycle")
        assert ts is not None
        assert ts <= time.time() + 0.001

    def test_runner_exception_does_not_propagate(self, workspace: Path) -> None:
        d = Daemon(str(workspace))
        task = TaskConfig("dream_cycle", 60, {})

        def boom(_ws: str, _extras: dict[str, Any]) -> dict[str, Any]:
            raise RuntimeError("simulated failure")

        d._tick(task, boom)  # must not raise
        assert d.last_run("dream_cycle") is not None

    def test_stop_event_short_circuits_sleep(self, workspace: Path) -> None:
        """Calling stop() exits the loop within a second tick."""
        d = Daemon(str(workspace), dry_run=True)
        task = TaskConfig("dream_cycle", 5, {})  # below MIN, but bypassed by direct constructor
        d._stop_event.set()  # already stopped
        # _run_loop must return immediately; if it didn't, the test would hang.
        d._run_loop(task)


# ---------------------------------------------------------------------------
# run_daemon convenience entry — once mode
# ---------------------------------------------------------------------------


class TestRunDaemonOnce:
    def test_once_mode_runs_each_task_once(self, workspace: Path, monkeypatch) -> None:
        from mind_mem import daemon as daemon_mod

        calls: list[str] = []

        def fake_dream(ws: str, extras: dict[str, Any]) -> dict[str, Any]:
            calls.append("dream_cycle")
            return {"ok": True}

        def fake_other(ws: str, extras: dict[str, Any]) -> dict[str, Any]:
            calls.append("other")
            return {"ok": True}

        monkeypatch.setitem(daemon_mod._TASK_RUNNERS, "dream_cycle", fake_dream)
        monkeypatch.setitem(daemon_mod._TASK_RUNNERS, "intel_scan", fake_other)

        _write_config(
            workspace,
            {
                "dream_cycle": {"auto_interval_seconds": 60},
                "intel_scan": {"auto_interval_seconds": 60},
            },
        )

        rc = daemon_mod.run_daemon(str(workspace), once=True)
        assert rc == 0
        assert "dream_cycle" in calls
        assert "other" in calls

    def test_disabled_daemon_returns_0_without_running(self, workspace: Path) -> None:
        from mind_mem import daemon as daemon_mod

        _write_config(workspace, {"enabled": False, "dream_cycle": {"auto_interval_seconds": 60}})
        rc = daemon_mod.run_daemon(str(workspace), once=True)
        assert rc == 0

    def test_no_enabled_tasks_returns_1(self, workspace: Path) -> None:
        from mind_mem import daemon as daemon_mod

        _write_config(workspace, {"enabled": True})  # no per-task intervals
        rc = daemon_mod.run_daemon(str(workspace), once=True)
        assert rc == 1
