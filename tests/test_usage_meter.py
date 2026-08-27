#!/usr/bin/env python3
"""Tests for `mm usage` — local per-day model-call token counter (Group G).

Acceptance gate covered here:
  1. token counters increment across a REAL recompaction run whose model call
     is an injected stub (the proven injected-callable pattern);
  2. reaching the daily token cap is reported and exits non-zero — both at the
     CLI and at the call site, which refuses before the stub is invoked;
  3. NOTHING leaves the host — every socket entry point is trapped and the
     attempt list must stay empty across record / report / CLI;
  4. metering is opt-in by construction: an unwrapped compressor writes no
     ledger and returns byte-identical recompaction output.
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
from typing import Any

from mind_mem import usage_meter

DAY = "2026-08-27"
NEXT_DAY = "2026-08-28"

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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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


CLUSTER: list[dict[str, Any]] = [
    {"_id": "D-20260101-001", "body": "Use PostgreSQL for the primary store; it is the system of record."},
    {"_id": "D-20260102-001", "body": "Use Redis for the caching layer in front of the primary store."},
]


class StubModel:
    """Stand-in for the injected model call: deterministic, offline, counted."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.calls = 0

    def __call__(self, current_text: str, blocks: list[dict[str, Any]]) -> str:
        index = min(self.calls, len(self._replies) - 1)
        self.calls += 1
        return self._replies[index]


def _stub() -> StubModel:
    # Second reply repeats the first, so the recompaction loop reaches its
    # fixed point after exactly two model calls.
    summary = "PostgreSQL is the system of record; Redis caches in front of it."
    return StubModel([summary, summary])


class UsageMeterBase(unittest.TestCase):
    def setUp(self) -> None:
        self.ws = _make_workspace()
        self.addCleanup(shutil.rmtree, self.ws, ignore_errors=True)

    def write_config(self, payload: dict[str, Any]) -> None:
        with open(os.path.join(self.ws, "mind-mem.json"), "w", encoding="utf-8") as fh:
            json.dump(payload, fh)


# ---------------------------------------------------------------------------
# Core ledger behaviour
# ---------------------------------------------------------------------------


class TestTokenLedger(UsageMeterBase):
    def test_empty_workspace_is_zero(self) -> None:
        r = usage_meter.report(self.ws, day=DAY)
        self.assertEqual(r.total_tokens, 0)
        self.assertEqual(r.today_tokens, 0)
        self.assertFalse(r.cap_exceeded)
        self.assertIsNone(r.ledger_error)
        self.assertIsNone(r.daily_cap)

    def test_record_call_is_cumulative_per_day(self) -> None:
        usage_meter.record_call(self.ws, operation="recompaction", prompt_tokens=100, completion_tokens=20, day=DAY)
        usage_meter.record_call(self.ws, operation="extraction", prompt_tokens=40, completion_tokens=5, day=DAY)
        usage_meter.record_call(self.ws, operation="recompaction", prompt_tokens=7, completion_tokens=3, day=NEXT_DAY)

        r = usage_meter.report(self.ws, day=DAY)
        self.assertEqual(r.today_tokens, 165)
        self.assertEqual(r.today_calls, 2)
        self.assertEqual(r.total_tokens, 175)
        self.assertEqual(dict(r.days[DAY].operations), {"recompaction": 120, "extraction": 45})
        self.assertEqual(usage_meter.report(self.ws, day=NEXT_DAY).today_tokens, 10)

    def test_no_currency_or_quota_surface_remains(self) -> None:
        gone = (
            "rollup",
            "price",
            "quota_alert_line",
            "DEFAULT_UNIT_COSTS",
            "QUOTA_EXIT_CODE",
            "flush_if_enabled",
            "meter_enabled",
            "ENV_ENABLE",
        )
        for name in gone:
            self.assertFalse(hasattr(usage_meter, name), f"{name} must not survive the token-meter rework")
        payload = usage_meter.report(self.ws, day=DAY).as_dict()
        self.assertNotIn("total_cost_usd", payload)
        self.assertNotIn("quota_usd", payload)

    def test_input_validation_at_boundaries(self) -> None:
        with self.assertRaises(ValueError):
            usage_meter.report("")
        with self.assertRaises(ValueError):
            usage_meter.report(self.ws, daily_cap=-1)
        with self.assertRaises(ValueError):
            usage_meter.report(self.ws, day="27-08-2026")
        with self.assertRaises(ValueError):
            usage_meter.record_call(self.ws, operation="", prompt_tokens=1, completion_tokens=1, day=DAY)
        with self.assertRaises(ValueError):
            usage_meter.record_call(self.ws, operation="x", prompt_tokens=-1, completion_tokens=0, day=DAY)
        with self.assertRaises(ValueError):
            usage_meter.record_call(self.ws, operation="x", prompt_tokens=1.5, completion_tokens=0, day=DAY)  # type: ignore[arg-type]

    def test_corrupt_ledger_degrades_instead_of_raising(self) -> None:
        path = usage_meter.ledger_path(self.ws)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("{not json")
        r = usage_meter.report(self.ws, day=DAY)
        self.assertEqual(r.total_tokens, 0)
        self.assertIsNotNone(r.ledger_error)

    def test_legacy_cost_ledger_is_refused_not_priced(self) -> None:
        """A v1 (cost/quota) ledger left on disk is reported as empty, not read."""
        path = usage_meter.ledger_path(self.ws)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"version": 1, "counters": {"recall_queries": 9999}, "sessions": 3}, fh)
        r = usage_meter.report(self.ws, day=DAY)
        self.assertEqual(r.total_tokens, 0)
        self.assertEqual(r.ledger_error, "UnsupportedLedgerVersion")

    def test_retention_prunes_old_days_and_keeps_the_newest(self) -> None:
        for i in range(usage_meter.RETENTION_DAYS + 5):
            usage_meter.record_call(
                self.ws,
                operation="recompaction",
                prompt_tokens=1,
                completion_tokens=0,
                day=f"2026-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}",
            )
        days, err = usage_meter.load_ledger(self.ws)
        self.assertIsNone(err)
        self.assertEqual(len(days), usage_meter.RETENTION_DAYS)
        last = usage_meter.RETENTION_DAYS + 4
        self.assertEqual(max(days), f"2026-{(last // 28) + 1:02d}-{(last % 28) + 1:02d}")

    def test_reset_clears_ledger(self) -> None:
        usage_meter.record_call(self.ws, operation="recompaction", prompt_tokens=7, completion_tokens=1, day=DAY)
        before = usage_meter.reset(self.ws, day=DAY)
        self.assertEqual(before.today_tokens, 8)
        self.assertEqual(usage_meter.report(self.ws, day=DAY).total_tokens, 0)

    def test_cap_comes_from_workspace_config_when_unset(self) -> None:
        self.write_config({"usage": {"daily_token_cap": 50}})
        usage_meter.record_call(self.ws, operation="recompaction", prompt_tokens=49, completion_tokens=0, day=DAY)
        self.assertFalse(usage_meter.report(self.ws, day=DAY).cap_exceeded)
        usage_meter.record_call(self.ws, operation="recompaction", prompt_tokens=1, completion_tokens=0, day=DAY)
        r = usage_meter.report(self.ws, day=DAY)
        self.assertEqual(r.daily_cap, 50)
        self.assertTrue(r.cap_exceeded)

    def test_bad_config_cap_is_ignored_not_fatal(self) -> None:
        self.write_config({"usage": {"daily_token_cap": "lots"}})
        self.assertIsNone(usage_meter.load_daily_cap(self.ws))


# ---------------------------------------------------------------------------
# GATE 1 — counters increment across a real recompaction run
# ---------------------------------------------------------------------------


class TestMeteredModelCallIntegration(UsageMeterBase):
    def test_tokens_increment_across_a_real_recompaction(self) -> None:
        from mind_mem.recompaction import recompact_cluster

        stub = _stub()
        compressor = usage_meter.metered_compressor(stub, self.ws, day=DAY)

        with no_egress() as attempts:
            result = recompact_cluster(CLUSTER, compressor=compressor)
            r = usage_meter.report(self.ws, day=DAY)

        self.assertEqual(attempts, [], "a metered model call must not touch the network")
        self.assertTrue(result.converged)
        self.assertGreaterEqual(stub.calls, 2)
        self.assertEqual(r.today_calls, stub.calls)
        self.assertGreater(r.today_tokens, 0)
        self.assertGreater(r.days[DAY].prompt_tokens, 0)
        self.assertGreater(r.days[DAY].completion_tokens, 0)
        self.assertEqual(set(r.days[DAY].operations), {usage_meter.OP_RECOMPACTION})

        # A second run keeps moving the ledger forward.
        before = r.today_tokens
        recompact_cluster(CLUSTER, compressor=usage_meter.metered_compressor(_stub(), self.ws, day=DAY))
        self.assertGreater(usage_meter.report(self.ws, day=DAY).today_tokens, before)

    def test_operation_tag_separates_call_sites(self) -> None:
        from mind_mem.recompaction import recompact_cluster

        recompact_cluster(CLUSTER, compressor=usage_meter.metered_compressor(_stub(), self.ws, day=DAY))
        usage_meter.record_call(self.ws, operation="extraction", prompt_tokens=11, completion_tokens=2, day=DAY)
        ops = usage_meter.report(self.ws, day=DAY).days[DAY].operations
        self.assertEqual(set(ops), {"recompaction", "extraction"})
        self.assertEqual(ops["extraction"], 13)


# ---------------------------------------------------------------------------
# GATE 2 — the daily cap is reported and exits non-zero
# ---------------------------------------------------------------------------


class TestDailyCapGate(UsageMeterBase):
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

    def test_under_cap_exits_zero(self) -> None:
        usage_meter.record_call(self.ws, operation="recompaction", prompt_tokens=10, completion_tokens=1)
        code, out, err = self._run_cli(["usage", "--daily-cap", "100000"])
        self.assertEqual(code, 0)
        self.assertIn("mind-mem model-call tokens", out)
        self.assertNotIn("DAILY TOKEN CAP", err)

    def test_cap_reached_exits_nonzero_with_report_line(self) -> None:
        usage_meter.record_call(self.ws, operation="recompaction", prompt_tokens=500, completion_tokens=100)
        with no_egress() as attempts:
            code, _, err = self._run_cli(["usage", "--daily-cap", "10"])
        self.assertEqual(attempts, [])
        self.assertEqual(code, usage_meter.CAP_EXIT_CODE)
        self.assertNotEqual(code, 0)
        lines = [ln for ln in err.splitlines() if ln.startswith("DAILY TOKEN CAP:")]
        self.assertEqual(len(lines), 1, f"expected exactly one cap line, got: {err!r}")
        self.assertIn("cap=10", lines[0])
        self.assertIn(self.ws, lines[0])

    def test_json_output_reports_the_cap(self) -> None:
        usage_meter.record_call(self.ws, operation="recompaction", prompt_tokens=500, completion_tokens=0)
        code, out, _ = self._run_cli(["usage", "--daily-cap", "1", "--json"])
        payload = json.loads(out)
        self.assertTrue(payload["cap_exceeded"])
        self.assertEqual(payload["daily_token_cap"], 1)
        self.assertEqual(payload["egress"], "none")
        self.assertEqual(code, usage_meter.CAP_EXIT_CODE)

    def test_bad_cap_is_a_usage_error_not_a_crash(self) -> None:
        code, _, err = self._run_cli(["usage", "--daily-cap", "-5"])
        self.assertEqual(code, 64)
        self.assertIn("daily_token_cap", err)

    def test_metered_call_refuses_once_the_cap_is_reached(self) -> None:
        from mind_mem.recompaction import recompact_cluster

        usage_meter.record_call(self.ws, operation="recompaction", prompt_tokens=100, completion_tokens=0, day=DAY)
        stub = _stub()
        compressor = usage_meter.metered_compressor(stub, self.ws, daily_cap=100, day=DAY)
        with self.assertRaises(usage_meter.DailyTokenCapExceeded):
            recompact_cluster(CLUSTER, compressor=compressor)
        self.assertEqual(stub.calls, 0, "the cap must be checked BEFORE the model is called")

    def test_cli_surface_is_strictly_smaller(self) -> None:
        """`mm usage` lost --quota and --record; it must not have grown back."""
        from mind_mem.mm_cli import build_parser

        actions = {a for a in build_parser().parse_args(["usage"]).__dict__ if a not in {"func", "cmd"}}
        self.assertEqual(actions, {"daily_cap", "json", "reset"})
        for dead in (["usage", "--quota", "5"], ["usage", "--record"]):
            with self.assertRaises(SystemExit):
                with redirect_stderr(io.StringIO()):
                    build_parser().parse_args(dead)


# ---------------------------------------------------------------------------
# GATE 3 / 4 — no egress anywhere; metering is opt-in by construction
# ---------------------------------------------------------------------------


class TestNoEgressAndOptIn(UsageMeterBase):
    def test_module_has_no_network_imports(self) -> None:
        import inspect

        src = inspect.getsource(usage_meter)
        for banned in ("import socket", "import http", "urllib", "requests", "httpx", "OTLPSpanExporter"):
            self.assertNotIn(banned, src, f"usage_meter must not reference {banned}")

    def test_full_cycle_makes_zero_network_attempts(self) -> None:
        with no_egress() as attempts:
            usage_meter.record_call(self.ws, operation="recompaction", prompt_tokens=5, completion_tokens=1, day=DAY)
            r = usage_meter.report(self.ws, daily_cap=10, day=DAY)
            usage_meter.format_report(r)
            usage_meter.cap_line(r)
            usage_meter.reset(self.ws, day=DAY)
        self.assertEqual(attempts, [])

    def test_unwrapped_compressor_writes_nothing_and_is_byte_identical(self) -> None:
        """Default-OFF proof: metering only happens where a caller wraps."""
        from mind_mem.recompaction import recompact_cluster

        plain = recompact_cluster(CLUSTER, compressor=_stub())
        self.assertFalse(
            os.path.exists(usage_meter.ledger_path(self.ws)),
            "an unwrapped compressor must not create a token ledger",
        )

        metered = recompact_cluster(CLUSTER, compressor=usage_meter.metered_compressor(_stub(), self.ws, day=DAY))
        self.assertEqual(
            plain.text.encode("utf-8"),
            metered.text.encode("utf-8"),
            "metering must not change a single output byte",
        )
        self.assertEqual(plain.output_digest, metered.output_digest)
        self.assertEqual(plain.iterations, metered.iterations)
        self.assertTrue(os.path.exists(usage_meter.ledger_path(self.ws)))

    def test_other_mm_commands_write_no_ledger(self) -> None:
        from mind_mem.mm_cli import main

        env = dict(os.environ)
        os.environ["MIND_MEM_WORKSPACE"] = self.ws
        try:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                code = main(["status"])
        finally:
            os.environ.clear()
            os.environ.update(env)
        self.assertEqual(code, 0)
        self.assertFalse(
            os.path.exists(usage_meter.ledger_path(self.ws)),
            "a normal mm command must not create a usage ledger",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
