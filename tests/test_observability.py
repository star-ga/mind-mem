#!/usr/bin/env python3
"""Tests for observability.py — structured logging and metrics."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from observability import get_logger, Metrics, timed


class TestStructuredLogger(unittest.TestCase):
    def test_get_logger_returns_logger(self):
        log = get_logger("test-component")
        self.assertEqual(log.name, "test-component")

    def test_logging_doesnt_raise(self):
        log = get_logger("test")
        log.debug("test_event", key="value")
        log.info("test_info", count=5)
        log.warning("test_warn")
        log.error("test_error", detail="something failed")


class TestMetrics(unittest.TestCase):
    def test_inc_default(self):
        m = Metrics()
        m.inc("requests")
        self.assertEqual(m.get("requests"), 1)

    def test_inc_custom_value(self):
        m = Metrics()
        m.inc("bytes", 1024)
        self.assertEqual(m.get("bytes"), 1024)

    def test_inc_accumulates(self):
        m = Metrics()
        m.inc("count")
        m.inc("count")
        m.inc("count", 3)
        self.assertEqual(m.get("count"), 5)

    def test_get_missing_returns_zero(self):
        m = Metrics()
        self.assertEqual(m.get("nonexistent"), 0)

    def test_observe_records_values(self):
        m = Metrics()
        m.observe("latency_ms", 10.5)
        m.observe("latency_ms", 20.3)
        m.observe("latency_ms", 15.0)
        summary = m.summary()
        self.assertIn("observations", summary)
        obs = summary["observations"]["latency_ms"]
        self.assertEqual(obs["count"], 3)
        self.assertAlmostEqual(obs["min"], 10.5)
        self.assertAlmostEqual(obs["max"], 20.3)
        self.assertAlmostEqual(obs["avg"], (10.5 + 20.3 + 15.0) / 3)

    def test_summary_counters(self):
        m = Metrics()
        m.inc("a", 10)
        m.inc("b", 20)
        summary = m.summary()
        self.assertEqual(summary["counters"]["a"], 10)
        self.assertEqual(summary["counters"]["b"], 20)

    def test_summary_no_observations(self):
        m = Metrics()
        m.inc("x")
        summary = m.summary()
        self.assertNotIn("observations", summary)

    def test_reset_clears_all(self):
        m = Metrics()
        m.inc("count", 5)
        m.observe("lat", 10.0)
        m.reset()
        self.assertEqual(m.get("count"), 0)
        self.assertEqual(m.summary(), {"counters": {}})


class TestTimed(unittest.TestCase):
    def test_timed_records_metric(self):
        m = Metrics()
        # Replace global metrics temporarily
        import observability
        original = observability.metrics
        observability.metrics = m
        try:
            with timed("test_op"):
                pass  # instant
            summary = m.summary()
            self.assertIn("test_op_ms", summary.get("observations", {}))
            self.assertEqual(summary["observations"]["test_op_ms"]["count"], 1)
        finally:
            observability.metrics = original

    def test_timed_with_logger(self):
        log = get_logger("test")
        with timed("fast_op", logger=log):
            pass  # trivial


if __name__ == "__main__":
    unittest.main()
