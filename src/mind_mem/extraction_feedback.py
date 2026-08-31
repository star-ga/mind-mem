#!/usr/bin/env python3
"""mind-mem Extraction Quality Feedback Tracker.

Tracks extraction outcomes (entities found, facts extracted, empty results)
per model and input type. Over time, identifies:
  - Which models produce better extractions for which content types
  - When extraction is consistently empty (wasted inference)
  - Quality trends (extraction improving or degrading)

Lightweight: JSON file, no dependencies, optional.

Copyright STARGA, Inc.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
import time
from typing import Any

from .observability import get_logger

_log = get_logger("extraction_feedback")


def default_feedback_path(workspace: str | None = None) -> str:
    """Resolve the feedback-file path anchored to *workspace*.

    Falls back to ``MIND_MEM_WORKSPACE`` and then the current
    directory — the old CWD-relative default meant the file landed
    wherever the process happened to start, scattering feedback state
    across unrelated directories.
    """
    ws = workspace or os.environ.get("MIND_MEM_WORKSPACE") or "."
    return os.path.join(ws, ".mind-mem", "extraction-feedback.json")


class ExtractionFeedback:
    """Track extraction quality per model and content type."""

    def __init__(self, path: str | None = None, workspace: str | None = None):
        self.path = path or default_feedback_path(workspace)
        self.records: list[dict[str, Any]] = []
        self._stats: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if not os.path.isfile(self.path):
            return
        try:
            # encoding= is explicit: on Windows the locale codepage is the
            # default, and one non-cp1252 byte would raise straight out of
            # __init__.
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._quarantine(f"{type(exc).__name__}: {exc}")
            return
        if not isinstance(data, dict):
            self._quarantine("feedback file does not hold a JSON object")
            return
        records = data.get("records", [])
        stats = data.get("stats", {})
        self.records = records if isinstance(records, list) else []
        self._stats = stats if isinstance(stats, dict) else {}

    def _quarantine(self, reason: str) -> None:
        """Move an unreadable feedback file aside instead of overwriting it.

        The next :meth:`_save` truncate-writes this path, so leaving a
        damaged file in place destroys the only copy of the history with
        no trace at all — and ``should_skip_extraction`` then answers
        ``False`` forever because ``total`` restarts below its threshold.
        Renaming keeps the remains and makes the loss visible.
        """
        backup = f"{self.path}.corrupt"
        try:
            os.replace(self.path, backup)
        except OSError as exc:
            _log.warning(
                "extraction_feedback_quarantine_failed",
                path=self.path,
                reason=reason,
                error=str(exc),
            )
        else:
            _log.warning(
                "extraction_feedback_unreadable",
                path=self.path,
                reason=reason,
                moved_to=backup,
            )
        self.records = []
        self._stats = {}

    def _save(self) -> None:
        directory = os.path.dirname(self.path) or "."
        os.makedirs(directory, exist_ok=True)
        data = {
            "version": 1,
            "records": self.records[-500:],  # keep last 500
            "stats": self._stats,
            "last_updated": time.time(),
        }
        # Write beside the target and rename over it. A truncate-write
        # here fires every 10 records; a process killed mid-dump would
        # leave a half-written file that no longer parses, and the whole
        # measured history would be dropped on the next load.
        fd, tmp = tempfile.mkstemp(dir=directory, prefix=".extraction-feedback-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise

    def record(
        self,
        model: str,
        operation: str,  # "entities" | "facts" | "enrich"
        input_length: int,
        output_count: int,
        latency_ms: float,
        content_type: str = "general",
    ) -> None:
        """Record an extraction outcome."""
        entry = {
            "model": model,
            "operation": operation,
            "input_length": input_length,
            "output_count": output_count,
            "latency_ms": latency_ms,
            "content_type": content_type,
            "timestamp": time.time(),
            "empty": output_count == 0,
        }
        self.records.append(entry)

        # Update running stats
        key = f"{model}:{operation}:{content_type}"
        if key not in self._stats:
            self._stats[key] = {
                "total": 0,
                "empty": 0,
                "total_output": 0,
                "total_latency_ms": 0.0,
            }
        s = self._stats[key]
        s["total"] += 1
        s["empty"] += 1 if output_count == 0 else 0
        s["total_output"] += output_count
        s["total_latency_ms"] += latency_ms

        # Auto-save every 10 records
        if len(self.records) % 10 == 0:
            self._save()

    def get_empty_rate(self, model: str, operation: str = "entities") -> float:
        """Get the empty extraction rate for a model+operation."""
        key = f"{model}:{operation}:general"
        s = self._stats.get(key)
        if not s or s["total"] == 0:
            return 0.0
        return float(s["empty"]) / float(s["total"])

    def get_avg_output(self, model: str, operation: str = "entities") -> float:
        """Average number of items extracted per call."""
        key = f"{model}:{operation}:general"
        s = self._stats.get(key)
        if not s or s["total"] == 0:
            return 0.0
        return float(s["total_output"]) / float(s["total"])

    def should_skip_extraction(self, model: str, operation: str = "entities") -> bool:
        """
        If a model consistently produces empty results (>80% empty rate
        after 10+ attempts), suggest skipping extraction to save inference.
        """
        key = f"{model}:{operation}:general"
        s = self._stats.get(key)
        if not s or s["total"] < 10:
            return False
        return bool((s["empty"] / s["total"]) > 0.8)

    def summary(self) -> dict[str, Any]:
        """Get summary stats for logging/debugging."""
        result = {}
        for key, s in self._stats.items():
            total = s["total"]
            if total == 0:
                continue
            result[key] = {
                "total": total,
                "empty_rate": round(s["empty"] / total, 3),
                "avg_output": round(s["total_output"] / total, 2),
                "avg_latency_ms": round(s["total_latency_ms"] / total, 1),
            }
        return result

    def flush(self) -> None:
        """Force save to disk."""
        self._save()
