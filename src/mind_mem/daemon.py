"""Background daemon — `mm daemon` (v3.9.0 candidate).

A long-lived process that schedules mind-mem's periodic tasks
internally without an external cron / systemd timer. Runs each
configured job on its own interval thread and survives transient
errors. SIGINT / SIGTERM shuts down cleanly.

Configurable intervals live under the ``daemon`` block of
``mind-mem.json``::

    {
      "daemon": {
        "enabled": true,
        "dream_cycle":     { "auto_interval_seconds": 1800,  "dry_run": false },
        "intel_scan":      { "auto_interval_seconds": 21600 },
        "entity_ingest":   { "auto_interval_seconds": 21600 },
        "transcript_scan": { "auto_interval_seconds": 3600  }
      }
    }

Any task whose ``auto_interval_seconds`` is ``0`` or unset is
skipped. The daemon refuses to start if every task is disabled.

Usage::

    mm daemon                 # blocks, runs internally scheduled jobs
    mm daemon --dry-run       # log what would run, do not execute
    mm daemon --once          # run every enabled task once and exit
"""

from __future__ import annotations

import json
import logging
import os
import signal
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

__all__ = [
    "DEFAULT_INTERVALS",
    "Daemon",
    "load_daemon_config",
    "run_daemon",
]

_log = logging.getLogger("mind_mem.daemon")

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# Conservative defaults — jobs that hit disk every interval so the
# operator should opt in by editing config explicitly. Setting them all
# to 0 makes the daemon a no-op (which is the right default for tests
# and CI workspaces).
DEFAULT_INTERVALS: dict[str, int] = {
    "dream_cycle": 0,
    "intel_scan": 0,
    "entity_ingest": 0,
    "transcript_scan": 0,
}

_MIN_INTERVAL_SECONDS = 60  # Prevent foot-gun "every second" misconfig.


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskConfig:
    """Per-task daemon configuration."""

    name: str
    interval_seconds: int
    extras: dict[str, Any]

    @property
    def enabled(self) -> bool:
        return self.interval_seconds > 0


def load_daemon_config(workspace: str) -> tuple[bool, list[TaskConfig]]:
    """Load ``daemon`` block from ``mind-mem.json``.

    Returns ``(enabled, [TaskConfig...])``. ``enabled`` is the global
    ``daemon.enabled`` flag (defaults to ``True`` when the daemon block
    is present at all). Tasks default to disabled (interval=0).
    """
    config_path = os.path.join(os.path.abspath(workspace), "mind-mem.json")
    raw: dict[str, Any] = {}
    if os.path.isfile(config_path):
        try:
            with open(config_path, encoding="utf-8") as fh:
                raw = json.load(fh) or {}
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            raw = {}

    daemon_cfg = raw.get("daemon", {}) if isinstance(raw, dict) else {}
    if not isinstance(daemon_cfg, dict):
        daemon_cfg = {}

    enabled = bool(daemon_cfg.get("enabled", True))

    tasks: list[TaskConfig] = []
    for name, default_interval in DEFAULT_INTERVALS.items():
        task_cfg = daemon_cfg.get(name, {}) if isinstance(daemon_cfg, dict) else {}
        if not isinstance(task_cfg, dict):
            task_cfg = {}
        try:
            interval = int(task_cfg.get("auto_interval_seconds", default_interval))
        except (TypeError, ValueError):
            interval = default_interval
        if interval < 0:
            interval = 0
        if 0 < interval < _MIN_INTERVAL_SECONDS:
            interval = _MIN_INTERVAL_SECONDS
        extras = {k: v for k, v in task_cfg.items() if k != "auto_interval_seconds"}
        tasks.append(TaskConfig(name=name, interval_seconds=interval, extras=extras))

    return (enabled, tasks)


# ---------------------------------------------------------------------------
# Task runners — each takes (workspace, extras) and returns a summary
# ---------------------------------------------------------------------------


def _run_dream_cycle(workspace: str, extras: dict[str, Any]) -> dict[str, Any]:
    from .dream_cycle import run_dream_cycle

    dry_run = bool(extras.get("dry_run", False))
    auto_repair = bool(extras.get("auto_repair", False))
    report = run_dream_cycle(workspace, dry_run=dry_run, auto_repair=auto_repair)
    return {"ok": True, "dry_run": dry_run, "errors": len(getattr(report, "errors", []) or [])}


def _run_via_cron_runner(job_name: str, workspace: str, _extras: dict[str, Any]) -> dict[str, Any]:
    from .cron_runner import run_job

    return run_job(job_name, workspace)


_TASK_RUNNERS: dict[str, Callable[[str, dict[str, Any]], dict[str, Any]]] = {
    "dream_cycle": _run_dream_cycle,
    "intel_scan": lambda ws, extras: _run_via_cron_runner("intel_scan", ws, extras),
    "entity_ingest": lambda ws, extras: _run_via_cron_runner("entity_ingest", ws, extras),
    "transcript_scan": lambda ws, extras: _run_via_cron_runner("transcript_scan", ws, extras),
}


# ---------------------------------------------------------------------------
# Daemon
# ---------------------------------------------------------------------------


class Daemon:
    """Coordinates one background thread per enabled task.

    Each thread:
      1. Sleeps ``interval_seconds`` (in 1-second tick chunks so SIGINT
         is responsive).
      2. Calls the task runner.
      3. Logs success or failure but never propagates.
      4. Loops until ``stop()`` is called.

    Daemon takes no parameters that change per-call so it can be
    constructed once and started / stopped multiple times for tests.
    """

    def __init__(self, workspace: str, *, dry_run: bool = False) -> None:
        if not workspace:
            raise ValueError("workspace must be a non-empty path")
        self.workspace = workspace
        self.dry_run = dry_run
        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []
        self._last_runs: dict[str, float] = {}
        self._lock = threading.Lock()

    # ----- lifecycle ---------------------------------------------------

    def start(self, tasks: list[TaskConfig]) -> int:
        """Spawn one thread per enabled task. Returns the count started."""
        enabled = [t for t in tasks if t.enabled]
        if not enabled:
            raise ValueError("no enabled tasks (every auto_interval_seconds is 0)")
        for task in enabled:
            t = threading.Thread(
                target=self._run_loop,
                args=(task,),
                name=f"mm-daemon-{task.name}",
                daemon=True,
            )
            t.start()
            self._threads.append(t)
        return len(self._threads)

    def stop(self) -> None:
        """Signal every loop to exit. Threads are daemon=True so we don't join."""
        self._stop_event.set()

    def join(self, timeout: float | None = None) -> None:
        for t in self._threads:
            t.join(timeout=timeout)

    # ----- loop --------------------------------------------------------

    def _run_loop(self, task: TaskConfig) -> None:
        runner = _TASK_RUNNERS.get(task.name)
        if runner is None:
            _log.warning("daemon_unknown_task", extra={"task": task.name})
            return
        _log.info(
            "daemon_task_started",
            extra={"task": task.name, "interval_seconds": task.interval_seconds},
        )
        # First tick fires after the interval, not immediately, so a
        # bouncing daemon doesn't hammer the workspace at start.
        while not self._stop_event.is_set():
            for _ in range(task.interval_seconds):
                if self._stop_event.is_set():
                    return
                time.sleep(1)
            if self._stop_event.is_set():
                return
            self._tick(task, runner)

    def _tick(self, task: TaskConfig, runner: Callable[[str, dict[str, Any]], dict[str, Any]]) -> None:
        if self.dry_run:
            _log.info("daemon_tick_dryrun", extra={"task": task.name})
            with self._lock:
                self._last_runs[task.name] = time.time()
            return
        try:
            summary = runner(self.workspace, task.extras)
            _log.info("daemon_tick_complete", extra={"task": task.name, "summary": summary})
        except Exception as exc:  # task failures must not kill the loop
            _log.error("daemon_tick_failed", extra={"task": task.name, "error": str(exc)})
        finally:
            with self._lock:
                self._last_runs[task.name] = time.time()

    # ----- introspection ----------------------------------------------

    def last_run(self, task_name: str) -> float | None:
        with self._lock:
            return self._last_runs.get(task_name)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_daemon(workspace: str, *, dry_run: bool = False, once: bool = False) -> int:
    """Block on the daemon until SIGINT / SIGTERM. Returns process exit code.

    ``once=True`` runs every enabled task one time and exits. Useful
    for test harnesses and one-shot operator runs.
    """
    enabled, tasks = load_daemon_config(workspace)
    if not enabled:
        _log.warning("daemon_disabled", extra={"workspace": workspace})
        print("mind-mem daemon: disabled in config (set daemon.enabled = true)")
        return 0

    enabled_tasks = [t for t in tasks if t.enabled]
    if not enabled_tasks:
        print(
            "mind-mem daemon: no tasks enabled — set "
            "daemon.<task>.auto_interval_seconds (>= 60) for at least one of: "
            f"{', '.join(DEFAULT_INTERVALS.keys())}"
        )
        return 1

    print(
        f"mind-mem daemon: starting {len(enabled_tasks)} task(s) — " + ", ".join(f"{t.name}({t.interval_seconds}s)" for t in enabled_tasks)
    )

    daemon = Daemon(workspace, dry_run=dry_run)

    if once:
        # Run each task synchronously, exactly once, then exit.
        for task in enabled_tasks:
            runner = _TASK_RUNNERS.get(task.name)
            if runner is None:
                _log.warning("daemon_unknown_task_once", extra={"task": task.name})
                continue
            try:
                summary = runner(workspace, task.extras)
                print(f"  [once] {task.name}: {summary}")
            except Exception as exc:
                print(f"  [once] {task.name}: FAILED — {exc}")
                _log.error("daemon_once_failed", extra={"task": task.name, "error": str(exc)})
        return 0

    daemon.start(enabled_tasks)

    # Translate SIGINT / SIGTERM into a clean stop.
    def _on_signal(signum: int, _frame: Any) -> None:
        print(f"\nmind-mem daemon: signal {signum} received, shutting down ...")
        daemon.stop()

    signal.signal(signal.SIGINT, _on_signal)
    try:
        signal.signal(signal.SIGTERM, _on_signal)
    except (AttributeError, ValueError):
        pass  # Windows / non-main-thread

    try:
        # Block until stop_event fires; join threads with a small grace.
        while not daemon._stop_event.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        daemon.stop()
    finally:
        daemon.join(timeout=5.0)

    return 0
