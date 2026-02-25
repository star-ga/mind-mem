#!/usr/bin/env python3
"""Mind-Mem File Watcher — auto-reindex on workspace changes. Zero external deps.

Polls workspace .md files for mtime changes on a background thread.
When changes are detected, fires the callback (typically incremental reindex).

Usage:
    from .watcher import FileWatcher
    w = FileWatcher("/path/to/workspace", callback=my_reindex_fn, interval=5.0)
    w.start()
    # ... later ...
    w.stop()

Uses ONLY threading, time, os — all stdlib.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Callable

from .observability import get_logger

_log = get_logger("watcher")


class FileWatcher:
    """Watch workspace for .md file changes, trigger callback on detected changes."""

    def __init__(
        self,
        workspace: str,
        callback: Callable[[set[str]], None],
        interval: float = 5.0,
    ):
        self.workspace = os.path.abspath(workspace)
        self.callback = callback
        self.interval = interval
        self._mtimes: dict[str, float] = {}
        self._running = False
        self._thread: threading.Thread | None = None

    def _scan(self) -> set[str]:
        """Return set of changed file paths since last scan."""
        changed: set[str] = set()
        current: dict[str, float] = {}

        for root, _dirs, files in os.walk(self.workspace):
            # Skip hidden dirs and index dirs
            basename = os.path.basename(root)
            if basename.startswith(".") or basename == "__pycache__":
                continue
            for f in files:
                if not f.endswith(".md"):
                    continue
                path = os.path.join(root, f)
                try:
                    mtime = os.path.getmtime(path)
                except OSError:
                    continue
                current[path] = mtime
                if path not in self._mtimes or self._mtimes[path] != mtime:
                    changed.add(path)

        # Detect deletions
        deleted = set(self._mtimes.keys()) - set(current.keys())
        self._mtimes = current
        return changed | deleted

    def start(self) -> None:
        """Start watching in a background daemon thread."""
        if self._running:
            return
        self._running = True
        # Initial scan to populate mtimes (no callback on first scan)
        self._scan()

        def _loop() -> None:
            while self._running:
                time.sleep(self.interval)
                try:
                    changes = self._scan()
                    if changes:
                        _log.info(
                            "changes_detected",
                            count=len(changes),
                            files=[os.path.basename(f) for f in list(changes)[:5]],
                        )
                        self.callback(changes)
                except Exception as e:
                    _log.warning("watcher_callback_error", error=str(e))

        self._thread = threading.Thread(target=_loop, daemon=True, name="mind-mem-watcher")
        self._thread.start()
        _log.info("watcher_started", workspace=self.workspace, interval=self.interval)

    def stop(self) -> None:
        """Stop the watcher."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=self.interval + 1)
            self._thread = None
        _log.info("watcher_stopped")

    @property
    def is_running(self) -> bool:
        return self._running
