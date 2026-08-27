#!/usr/bin/env python3
"""Tests for `mm usage` — per-workspace usage/cost rollup (Group G).

Acceptance gate covered here:
  1. counters increment across an integration test that exercises REAL
     operations (recall + contradiction detection on a real workspace);
  2. a quota breach exits non-zero and emits a clear alert line;
  3. NOTHING leaves the host — every socket entry point is trapped and the
     attempt list must stay empty across record/rollup/CLI.
Plus: flag-off (`MIND_MEM_USAGE_METER` unset) is byte-identical to before.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import socket
import tempfile
import unittest
from contextlib import contextmanager, redirect_stderr, redirect_stdout

from mind_mem import usage_meter
from mind_mem.observability import Metrics

# ---------------------------------------------------------------------------
# No-egress guard: record (and refuse) every outbound attempt.
# ---------------------------------------------------------------------------


@contextmanager
def no_egress():
    """Trap every stdlib egress entry point; yields the attempt list."""
    attempts: list[str] = []
    saved = {
        "connect": socket.socket.connect,
        "connect_ex": socket.socket.connect_ex,
        "create_connection": socket.create_connection,
        "getaddrinfo": socket.getaddrinfo,
        "sendto": socket.socket.sendto,
    }

    def _trap(name):
        def _blocked(*args, **kwargs):
            attempts.append(name)
            raise AssertionError(f"network egress attempted via {name}")

        return _blocked

    socket.socket.connect = _trap("connect")  # type: ignore[method-assign]
    socket.socket.connect_ex = _trap("connect_ex")  # type: ignore[method-assign]
    socket.create_connection = _trap("create_connection")  # type: ignore[assignment]
    socket.getaddrinfo = _trap("getaddrinfo")  # type: ignore[assignment]
    socket.socket.sendto = _trap("sendto")  # type: ignore[method-assign]
    try:
        yield attempts
    finally:
        socket.socket.connect = saved["connect"]  # type: ignore[method-assign]
        socket.socket.connect_ex = saved["connect_ex"]  # type: ignore[method-assign]
        socket.create_connection = saved["create_connection"]  # type: ignore[assignment]
        socket.getaddrinfo = saved["getaddrinfo"]  # type: ignore[assignment]
        socket.socket.sendto = saved["sendto"]  # type: ignore[method-assign]


def _make_workspace() -> str:
    ws = tempfile.mkdtemp()
    os.makedirs(os.path.join(ws, "decisions"))
    with open(os.path.join(ws, "decisions", "DECISIONS.md"), "w", encoding="utf-8") as fh:
        fh.write(
            "[D-20260101-001]\nStatement: Use PostgreSQL for the database\n"
            "Status: active\nDate: 2026-01-01\nTags: database\n\n---\n\n"
            "[D-20260102-001]\nStatement: Use Redis for the caching layer\n"
            "Status: active\nDate: 2026-01-02\nTags: caching\n"
        )
    return ws


class UsageMeterBase(unittest.TestCase):
    def setUp(self) -> None:
        self.ws = _make_workspace()
        usage_meter.reset_process_high_water()
        self.addCleanup(shutil.rmtree, self.ws, ignore_errors=True)
        self.addCleanup(usage_meter.reset_process_high_water)


class TestRollupCore(UsageMeterBase):
    def test_empty_workspace_is_zero(self) -> None:
        r = usage_meter.rollup(self.ws)
        self.assertEqual(r.total_operations, 0)
        self.assertEqual(r.total_cost_usd, 0.0)
        self.assertFalse(r.quota_breached)
        self.assertIsNone(r.ledger_error)

    def test_record_is_cumulative_and_priced(self) -> None:
        usage_meter.record(self.ws, {"recall_queries": 10})
        usage_meter.record(self.ws, {"recall_queries": 5, "vector_searches": 2})
        r = usage_meter.rollup(self.ws)
        self.assertEqual(r.counters["recall_queries"], 15)
        self.assertEqual(r.counters["vector_searches"], 2)
        expected = 15 * usage_meter.DEFAULT_UNIT_COSTS["recall_queries"] + 2 * usage_meter.DEFAULT_UNIT_COSTS["vector_searches"]
        self.assertAlmostEqual(r.total_cost_usd, round(expected, 8), places=8)
        self.assertEqual(r.sessions, 2)

    def test_process_counters_recorded_as_delta_not_double_counted(self) -> None:
        m = Metrics()
        m.inc("recall_queries", 3)
        usage_meter.record(self.ws, source=m)
        usage_meter.record(self.ws, source=m)  # nothing new happened
        m.inc("recall_queries", 2)
        usage_meter.record(self.ws, source=m)
        self.assertEqual(usage_meter.rollup(self.ws).counters["recall_queries"], 5)

    def test_unpriced_counter_counts_but_costs_nothing(self) -> None:
        usage_meter.record(self.ws, {"recall_results": 40})
        r = usage_meter.rollup(self.ws)
        self.assertEqual(r.total_operations, 40)
        self.assertEqual(r.total_cost_usd, 0.0)
        self.assertNotIn("recall_results", r.costs)

    def test_rate_card_override_from_workspace_config(self) -> None:
        with open(os.path.join(self.ws, "mind-mem.json"), "w", encoding="utf-8") as fh:
            json.dump({"usage": {"unit_costs": {"recall_queries": 1.0}}}, fh)
        usage_meter.record(self.ws, {"recall_queries": 3})
        self.assertAlmostEqual(usage_meter.rollup(self.ws).total_cost_usd, 3.0, places=8)

    def test_corrupt_ledger_degrades_instead_of_raising(self) -> None:
        path = usage_meter.ledger_path(self.ws)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("{not json")
        r = usage_meter.rollup(self.ws)
        self.assertEqual(r.total_operations, 0)
        self.assertIsNotNone(r.ledger_error)

    def test_input_validation_at_boundaries(self) -> None:
        with self.assertRaises(ValueError):
            usage_meter.rollup("")
        with self.assertRaises(ValueError):
            usage_meter.rollup(self.ws, quota_usd=-1.0)
        with self.assertRaises(ValueError):
            usage_meter.rollup(self.ws, quota_usd="1.0")  # type: ignore[arg-type]

    def test_record_with_no_new_counters_does_not_churn_the_ledger(self) -> None:
        usage_meter.record(self.ws, {"recall_queries": 2})
        path = usage_meter.ledger_path(self.ws)
        with open(path, "rb") as fh:
            before = fh.read()
        usage_meter.record(self.ws, {})
        with open(path, "rb") as fh:
            self.assertEqual(fh.read(), before, "a no-delta record must not rewrite the ledger")

    def test_reset_clears_ledger(self) -> None:
        usage_meter.record(self.ws, {"recall_queries": 7})
        before = usage_meter.reset(self.ws)
        self.assertEqual(before.counters["recall_queries"], 7)
        self.assertEqual(usage_meter.rollup(self.ws).total_operations, 0)


class TestRealOperationsIntegration(UsageMeterBase):
    """GATE 1 — counters increment across REAL mind-mem operations."""

    def test_counters_increment_across_real_operations(self) -> None:
        from mind_mem.contradiction_detector import detect_contradictions
        from mind_mem.observability import metrics as global_metrics
        from mind_mem.recall import recall

        global_metrics.reset()
        usage_meter.reset_process_high_water()
        self.assertEqual(usage_meter.rollup(self.ws).total_operations, 0)

        with no_egress() as attempts:
            for query in ("database", "caching", "postgres"):
                recall(self.ws, query, limit=5)
            detect_contradictions(
                self.ws,
                {
                    "ProposalId": "P-1",
                    "Title": "Switch the database",
                    "Description": "Use MySQL for the database instead of PostgreSQL",
                },
                use_bm25=False,
            )
            usage_meter.record(self.ws)
            r = usage_meter.rollup(self.ws)

        self.assertEqual(attempts, [], "real operations must not touch the network")
        self.assertGreaterEqual(r.counters.get("recall_queries", 0), 3)
        self.assertGreaterEqual(r.counters.get("contradiction_checks", 0), 1)
        self.assertGreater(r.total_operations, 0)
        self.assertGreater(r.total_cost_usd, 0.0)

        # A second real operation keeps moving the ledger forward.
        recall(self.ws, "redis", limit=5)
        usage_meter.record(self.ws)
        self.assertGreaterEqual(usage_meter.rollup(self.ws).counters["recall_queries"], 4)


class TestQuotaGate(UsageMeterBase):
    """GATE 2 — a quota breach exits non-zero with a clear alert line."""

    def _run_cli(self, argv: list[str]) -> tuple[int, str, str]:
        from mind_mem.mm_cli import main

        out, err = io.StringIO(), io.StringIO()
        env = dict(os.environ)
        os.environ["MIND_MEM_WORKSPACE"] = self.ws
        try:
            with redirect_stdout(out), redirect_stderr(err):
                code = main(argv)
        finally:
            os.environ.clear()
            os.environ.update(env)
        return code, out.getvalue(), err.getvalue()

    def test_under_quota_exits_zero(self) -> None:
        usage_meter.record(self.ws, {"recall_queries": 1})
        code, out, err = self._run_cli(["usage", "--quota", "100"])
        self.assertEqual(code, 0)
        self.assertIn("mind-mem usage", out)
        self.assertNotIn("QUOTA BREACH", err)

    def test_breach_exits_nonzero_with_alert_line(self) -> None:
        usage_meter.record(self.ws, {"recall_queries": 1000})
        with no_egress() as attempts:
            code, out, err = self._run_cli(["usage", "--quota", "0.000001"])
        self.assertEqual(attempts, [])
        self.assertNotEqual(code, 0)
        self.assertEqual(code, usage_meter.QUOTA_EXIT_CODE)
        alert = [ln for ln in err.splitlines() if ln.startswith("QUOTA BREACH:")]
        self.assertEqual(len(alert), 1, f"expected exactly one alert line, got: {err!r}")
        self.assertIn("quota=$0.000001", alert[0])
        self.assertIn(self.ws, alert[0])

    def test_json_output_reports_breach(self) -> None:
        usage_meter.record(self.ws, {"recall_queries": 1000})
        code, out, _ = self._run_cli(["usage", "--quota", "0.0", "--json"])
        payload = json.loads(out)
        self.assertTrue(payload["quota_breached"])
        self.assertEqual(payload["egress"], "none")
        self.assertEqual(code, usage_meter.QUOTA_EXIT_CODE)

    def test_bad_quota_is_a_usage_error_not_a_crash(self) -> None:
        code, _, err = self._run_cli(["usage", "--quota", "-5"])
        self.assertEqual(code, 64)
        self.assertIn("quota_usd", err)


class TestNoEgressAndOptIn(UsageMeterBase):
    """GATE 3 — no egress anywhere; and the CLI hook is default-OFF."""

    def test_module_has_no_network_imports(self) -> None:
        import inspect

        src = inspect.getsource(usage_meter)
        for banned in ("import socket", "import http", "urllib", "requests", "httpx", "OTLPSpanExporter"):
            self.assertNotIn(banned, src, f"usage_meter must not reference {banned}")

    def test_full_cycle_makes_zero_network_attempts(self) -> None:
        with no_egress() as attempts:
            usage_meter.record(self.ws, {"recall_queries": 5})
            usage_meter.rollup(self.ws, quota_usd=1.0, include_process=True)
            usage_meter.format_report(usage_meter.rollup(self.ws))
            usage_meter.reset(self.ws)
        self.assertEqual(attempts, [])

    def test_meter_flush_is_default_off_and_writes_nothing(self) -> None:
        from mind_mem.mm_cli import main

        env = dict(os.environ)
        os.environ["MIND_MEM_WORKSPACE"] = self.ws
        os.environ.pop(usage_meter.ENV_ENABLE, None)
        try:
            out_off = io.StringIO()
            with redirect_stdout(out_off), redirect_stderr(io.StringIO()):
                code_off = main(["status"])
        finally:
            os.environ.clear()
            os.environ.update(env)

        self.assertEqual(code_off, 0)
        self.assertFalse(
            os.path.exists(usage_meter.ledger_path(self.ws)),
            "flag-off must not create a usage ledger",
        )

        # Flag ON: same stdout bytes, but the ledger now exists.
        env = dict(os.environ)
        os.environ["MIND_MEM_WORKSPACE"] = self.ws
        os.environ[usage_meter.ENV_ENABLE] = "1"
        try:
            out_on = io.StringIO()
            with redirect_stdout(out_on), redirect_stderr(io.StringIO()):
                code_on = main(["status"])
        finally:
            os.environ.clear()
            os.environ.update(env)

        self.assertEqual(code_on, 0)
        self.assertEqual(
            out_off.getvalue().encode("utf-8"),
            out_on.getvalue().encode("utf-8"),
            "enabling the meter must not change command output",
        )
        self.assertTrue(os.path.exists(usage_meter.ledger_path(self.ws)))

    def test_meter_enabled_parses_only_explicit_truthy(self) -> None:
        for value in ("1", "true", "YES", "on"):
            self.assertTrue(usage_meter.meter_enabled({usage_meter.ENV_ENABLE: value}))
        for value in ("", "0", "false", "no", "maybe"):
            self.assertFalse(usage_meter.meter_enabled({usage_meter.ENV_ENABLE: value}))
        self.assertFalse(usage_meter.meter_enabled({}))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
