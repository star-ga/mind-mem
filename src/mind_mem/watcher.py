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
from collections.abc import Callable

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
        """Start watching in a background daemon thread.

        Refuses to start while a previous loop is still finishing (a
        :meth:`stop` whose join timed out): a second loop over the same
        workspace would double-fire the reindex callback.
        """
        if self._running:
            return
        previous = self._thread
        if previous is not None and previous.is_alive():
            _log.warning(
                "watcher_start_refused_previous_still_running",
                workspace=self.workspace,
            )
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

    def stop(self, timeout: float | None = None) -> bool:
        """Stop the watcher and wait for its loop to finish.

        Returns ``True`` when the loop actually finished, ``False`` when
        it was still running at the deadline. The join result is the
        whole point: the callback is typically an incremental reindex,
        which on a large workspace routinely outlasts ``interval + 1``
        seconds — dropping the thread handle there reported "stopped"
        while a writer was still active in the index, with nothing left
        on the object able to say so.

        The handle is kept on timeout, so :attr:`is_running` stays
        ``True`` and a later ``stop()`` can wait again.
        """
        self._running = False
        thread = self._thread
        if thread is None:
            return True
        deadline = self.interval + 1 if timeout is None else timeout
        thread.join(timeout=deadline)
        if thread.is_alive():
            _log.warning(
                "watcher_stop_timeout",
                workspace=self.workspace,
                timeout=deadline,
                msg="Watch loop still running after join timeout; callback may still be writing.",
            )
            return False
        self._thread = None
        _log.info("watcher_stopped")
        return True

    @property
    def is_running(self) -> bool:
        """True while the watch loop may still be executing.

        Reads the thread as well as the flag: after a ``stop()`` whose
        join timed out the loop is still inside the callback, and a
        caller trusting the flag alone would believe the workspace was
        released.
        """
        if self._running:
            return True
        thread = self._thread
        return thread is not None and thread.is_alive()
