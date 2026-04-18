# Copyright 2026 STARGA, Inc.
"""Event-driven ingestion + WAL + webhook endpoint (v2.5.0).

Three pieces:

1. :class:`IngestionQueue` — bounded event queue with backpressure.
2. :class:`WriteAheadLog` — JSONL WAL so crashes don't lose un-indexed
   writes.
3. :func:`serve_webhook` — stdlib-only HTTP endpoint that POSTs to the
   queue; no aiohttp dependency.

The ingestion path is deliberately optional: callers use ``recall``
synchronously today, and this module just lets them turn on
asynchronous writes + external ingestion without adding a new
runtime dependency.
"""

from __future__ import annotations

import json
import os
import queue
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from typing import Any, Callable, Mapping, Optional


@dataclass
class IngestionStats:
    accepted: int = 0
    rejected: int = 0
    backpressure_drops: int = 0
    applied: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "accepted": self.accepted,
            "rejected": self.rejected,
            "backpressure_drops": self.backpressure_drops,
            "applied": self.applied,
        }


class IngestionQueue:
    """Bounded event queue with explicit backpressure semantics."""

    def __init__(self, *, capacity: int = 1024) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self._q: queue.Queue = queue.Queue(maxsize=capacity)
        self._stats = IngestionStats()
        self._lock = threading.RLock()

    @property
    def depth(self) -> int:
        return self._q.qsize()

    @property
    def capacity(self) -> int:
        return self._q.maxsize

    def offer(self, event: Mapping[str, Any]) -> bool:
        """Non-blocking enqueue. Returns False when backpressure engages."""
        try:
            self._q.put_nowait(dict(event))
        except queue.Full:
            with self._lock:
                self._stats.backpressure_drops += 1
            return False
        with self._lock:
            self._stats.accepted += 1
        return True

    def drain(self, max_items: int = 64) -> list[dict]:
        drained: list[dict] = []
        for _ in range(max_items):
            try:
                drained.append(self._q.get_nowait())
            except queue.Empty:
                break
        return drained

    def stats(self) -> IngestionStats:
        with self._lock:
            return IngestionStats(
                accepted=self._stats.accepted,
                rejected=self._stats.rejected,
                backpressure_drops=self._stats.backpressure_drops,
                applied=self._stats.applied,
            )

    def mark_applied(self, count: int) -> None:
        with self._lock:
            self._stats.applied += int(count)


# ---------------------------------------------------------------------------
# Write-ahead log
# ---------------------------------------------------------------------------


class WriteAheadLog:
    """Append-only JSONL WAL for ingestion events.

    Every ``append`` fsyncs the file so a crash before indexing
    doesn't lose pending writes. ``replay`` yields un-applied records
    so the indexer can catch up after restart.
    """

    def __init__(self, path: str) -> None:
        self._path = os.path.abspath(path)
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        self._lock = threading.RLock()

    @property
    def path(self) -> str:
        return self._path

    def append(self, event: Mapping[str, Any]) -> None:
        with self._lock:
            with open(self._path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(event, separators=(",", ":"), default=str) + "\n")
                fh.flush()
                os.fsync(fh.fileno())

    def replay(self) -> list[dict]:
        if not os.path.isfile(self._path):
            return []
        out: list[dict] = []
        with open(self._path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    obj = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    out.append(obj)
        return out

    def truncate(self) -> None:
        with self._lock:
            with open(self._path, "w", encoding="utf-8"):
                pass


# ---------------------------------------------------------------------------
# HTTP webhook endpoint (stdlib-only)
# ---------------------------------------------------------------------------


class _ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def serve_webhook(
    port: int,
    ingestion: IngestionQueue,
    *,
    wal: Optional[WriteAheadLog] = None,
    host: str = "127.0.0.1",
) -> tuple[threading.Thread, Callable[[], None]]:
    """Start a stdlib HTTP server accepting POST /ingest with a JSON body.

    Returns ``(server_thread, stop_fn)``. Callers invoke ``stop_fn()``
    to shut the server down cleanly. Requests that exceed 1 MiB are
    refused with HTTP 413 so the endpoint cannot be used as a memory
    DoS vector.
    """

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # silence default
            return

        def do_POST(self) -> None:
            if self.path != "/ingest":
                self.send_response(404)
                self.end_headers()
                return
            length = int(self.headers.get("Content-Length", "0") or 0)
            if length > 1_048_576:
                self.send_response(413)
                self.end_headers()
                return
            raw = self.rfile.read(length) if length > 0 else b""
            try:
                event = json.loads(raw.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                self.send_response(400)
                self.end_headers()
                return
            if not isinstance(event, dict):
                self.send_response(400)
                self.end_headers()
                return
            if wal is not None:
                wal.append(event)
            ok = ingestion.offer(event)
            self.send_response(202 if ok else 503)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"accepted": ok}).encode("utf-8"))

    httpd = _ThreadingHTTPServer((host, port), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    def _stop() -> None:
        httpd.shutdown()
        httpd.server_close()

    return thread, _stop


__all__ = [
    "IngestionQueue",
    "IngestionStats",
    "WriteAheadLog",
    "serve_webhook",
]
