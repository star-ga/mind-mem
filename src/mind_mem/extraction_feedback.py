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

import json
import os
import time
from typing import Any

_DEFAULT_PATH = os.path.join(".", ".mind-mem", "extraction-feedback.json")


class ExtractionFeedback:
    """Track extraction quality per model and content type."""

    def __init__(self, path: str | None = None):
        self.path = path or _DEFAULT_PATH
        self.records: list[dict[str, Any]] = []
        self._stats: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if os.path.isfile(self.path):
            try:
                with open(self.path) as f:
                    data = json.load(f)
                self.records = data.get("records", [])
                self._stats = data.get("stats", {})
            except (OSError, json.JSONDecodeError):
                self.records = []
                self._stats = {}

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        data = {
            "version": 1,
            "records": self.records[-500:],  # keep last 500
            "stats": self._stats,
            "last_updated": time.time(),
        }
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2)

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
        return s["empty"] / s["total"]

    def get_avg_output(self, model: str, operation: str = "entities") -> float:
        """Average number of items extracted per call."""
        key = f"{model}:{operation}:general"
        s = self._stats.get(key)
        if not s or s["total"] == 0:
            return 0.0
        return s["total_output"] / s["total"]

    def should_skip_extraction(self, model: str, operation: str = "entities") -> bool:
        """
        If a model consistently produces empty results (>80% empty rate
        after 10+ attempts), suggest skipping extraction to save inference.
        """
        key = f"{model}:{operation}:general"
        s = self._stats.get(key)
        if not s or s["total"] < 10:
            return False
        return (s["empty"] / s["total"]) > 0.8

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
