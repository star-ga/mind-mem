#!/usr/bin/env python3
"""Model-call token metering AT THE CALL SITES (Group G — `mm usage` wiring).

`tests/test_usage_meter.py` proves the ledger primitive. This file proves the
part that makes it a product feature: the places mind-mem actually talks to a
model are counted, and the optional daily cap refuses them.

Covered here:
  1. the shared extraction backend (`llm_extractor._query_llm`, i.e. the
     ollama / OpenAI-compatible / llama-cpp / transformers dispatch) counts
     every completed call into the workspace ledger;
  2. provider-reported token counts are preferred over the char estimator;
  3. the daily cap refuses the call BEFORE the backend is touched, and the
     refusal escapes the backend fall-through loop instead of degrading into
     an empty answer;
  4. with no cap configured the answer is byte-identical to the unmetered one;
  5. real entry points — `mm graph-backfill` (extraction) and the recall
     rerank stage — are metered, and `mm usage` reports what they spent.

Every model call in this file is a local HTTP stub. Nothing leaves the host.
"""

from __future__ import annotations

import io
import json
import os
import threading
from collections.abc import Callable, Iterator
from contextlib import redirect_stderr, redirect_stdout
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest

from mind_mem import usage_meter
from mind_mem.usage_meter import DailyTokenCapExceeded

BLOCK_A = "D-20260101-001"
BLOCK_B = "D-20260102-001"

# ---------------------------------------------------------------------------
# Local model stub — a real socket to 127.0.0.1, never off-host.
# ---------------------------------------------------------------------------


class ModelStub:
    """Local HTTP server standing in for ollama / an OpenAI-compatible host."""

    def __init__(self, reply: Callable[[int], dict[str, Any]]) -> None:
        self.requests: list[dict[str, Any]] = []
        outer = self

        class _Handler(BaseHTTPRequestHandler):
            def _send(self, payload: dict[str, Any]) -> None:
                body = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # /api/tags, /v1/models availability probes
                self._send({"models": [], "data": []})

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", 0))
                outer.requests.append(json.loads(self.rfile.read(length) or b"{}"))
                self._send(reply(len(outer.requests)))

            def log_message(self, *args: Any) -> None:
                pass

        self._server = HTTPServer(("127.0.0.1", 0), _Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    @property
    def calls(self) -> int:
        return len(self.requests)

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


def _ollama_reply(
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    text: str = "[]",
) -> Callable[[int], dict[str, Any]]:
    def _reply(_n: int) -> dict[str, Any]:
        payload: dict[str, Any] = {"response": text}
        if prompt_tokens is not None:
            payload["prompt_eval_count"] = prompt_tokens
        if completion_tokens is not None:
            payload["eval_count"] = completion_tokens
        return payload

    return _reply


def _oai_reply(
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    text: str = "[]",
) -> Callable[[int], dict[str, Any]]:
    def _reply(_n: int) -> dict[str, Any]:
        payload: dict[str, Any] = {"choices": [{"message": {"content": text}}]}
        if prompt_tokens is not None or completion_tokens is not None:
            payload["usage"] = {
                "prompt_tokens": prompt_tokens or 0,
                "completion_tokens": completion_tokens or 0,
            }
        return payload

    return _reply


@pytest.fixture
def stub() -> Iterator[ModelStub]:
    server = ModelStub(_ollama_reply())
    try:
        yield server
    finally:
        server.close()


@pytest.fixture
def ws(tmp_path: Any) -> str:
    """A workspace with a real two-block markdown corpus."""
    root = str(tmp_path)
    os.makedirs(os.path.join(root, "decisions"), exist_ok=True)
    with open(os.path.join(root, "decisions", "DECISIONS.md"), "w", encoding="utf-8") as fh:
        fh.write(
            f"[{BLOCK_A}]\nStatement: STARGA depends on mindc for builds\n"
            "Status: active\nDate: 2026-01-01\nTags: build\n\n---\n\n"
            f"[{BLOCK_B}]\nStatement: Redis caches in front of PostgreSQL\n"
            "Status: active\nDate: 2026-01-02\nTags: caching\n"
        )
    return root


def _write_config(workspace: str, payload: dict[str, Any]) -> None:
    with open(os.path.join(workspace, "mind-mem.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh)


def _ledger_totals(workspace: str) -> tuple[int, int, dict[str, int]]:
    """(total tokens, total calls, merged operation histogram) across all days."""
    days, err = usage_meter.load_ledger(workspace)
    assert err is None, err
    ops: dict[str, int] = {}
    for usage in days.values():
        for name, count in usage.operations.items():
            ops[name] = ops.get(name, 0) + count
    return (
        sum(u.total_tokens for u in days.values()),
        sum(u.calls for u in days.values()),
        ops,
    )


# ---------------------------------------------------------------------------
# 1 + 4 — the extraction backend counts, and is inert without a workspace
# ---------------------------------------------------------------------------


class TestExtractionBackendMetering:
    def test_no_workspace_means_no_ledger_at_all(self, stub: ModelStub, ws: str) -> None:
        """Default OFF: an unmetered call writes nothing and touches no config."""
        from mind_mem.llm_extractor import _query_llm

        out = _query_llm("prompt text", "m", "ollama", ollama_url=stub.base_url)
        assert out == "[]"
        assert stub.calls == 1
        assert not os.path.exists(usage_meter.ledger_path(ws))

    def test_counts_accumulate_across_calls(self, stub: ModelStub, ws: str) -> None:
        from mind_mem.llm_extractor import _query_llm

        for _ in range(3):
            _query_llm("prompt text", "m", "ollama", ollama_url=stub.base_url, workspace=ws)

        total, calls, ops = _ledger_totals(ws)
        assert calls == 3
        assert total > 0
        assert set(ops) == {usage_meter.OP_EXTRACTION}
        assert ops[usage_meter.OP_EXTRACTION] == total

    def test_ollama_reported_counts_are_preferred_over_the_estimate(self, ws: str) -> None:
        from mind_mem.llm_extractor import _query_llm

        server = ModelStub(_ollama_reply(prompt_tokens=123, completion_tokens=7))
        try:
            _query_llm("x" * 4000, "m", "ollama", ollama_url=server.base_url, workspace=ws)
        finally:
            server.close()

        days, _ = usage_meter.load_ledger(ws)
        usage = next(iter(days.values()))
        assert (usage.prompt_tokens, usage.completion_tokens) == (123, 7)

    def test_openai_compatible_usage_block_is_recorded(self, ws: str, monkeypatch: pytest.MonkeyPatch) -> None:
        from mind_mem.llm_extractor import _query_llm

        server = ModelStub(_oai_reply(prompt_tokens=41, completion_tokens=9))
        try:
            monkeypatch.setenv("MIND_MEM_LLM_BASE_URL", server.base_url)
            _query_llm("prompt", "m", "openai-compatible", workspace=ws)
        finally:
            server.close()

        days, _ = usage_meter.load_ledger(ws)
        usage = next(iter(days.values()))
        assert (usage.prompt_tokens, usage.completion_tokens) == (41, 9)

    def test_estimator_is_the_fallback_when_the_provider_reports_nothing(self, ws: str) -> None:
        from mind_mem.cognitive_forget import estimate_tokens
        from mind_mem.llm_extractor import _query_llm

        server = ModelStub(_ollama_reply(text="a reply with no usage block"))
        prompt = "y" * 800
        try:
            out = _query_llm(prompt, "m", "ollama", ollama_url=server.base_url, workspace=ws)
        finally:
            server.close()

        days, _ = usage_meter.load_ledger(ws)
        usage = next(iter(days.values()))
        assert usage.prompt_tokens == estimate_tokens(prompt)
        assert usage.completion_tokens == estimate_tokens(out)

    def test_no_cap_configured_returns_the_identical_answer(self, ws: str) -> None:
        """Metering must not change one byte of what the model said."""
        from mind_mem.llm_extractor import _query_llm

        server = ModelStub(_ollama_reply(text='[{"subject": "a"}]'))
        try:
            plain = _query_llm("prompt", "m", "ollama", ollama_url=server.base_url)
            metered = _query_llm("prompt", "m", "ollama", ollama_url=server.base_url, workspace=ws)
        finally:
            server.close()

        assert plain.encode("utf-8") == metered.encode("utf-8")
        assert usage_meter.report(ws).daily_cap is None
        assert _ledger_totals(ws)[1] == 1


# ---------------------------------------------------------------------------
# 3 — the cap fails closed
# ---------------------------------------------------------------------------


class TestDailyCapFailsClosed:
    def test_cap_refuses_before_the_backend_is_called(self, stub: ModelStub, ws: str) -> None:
        from mind_mem.llm_extractor import _query_llm

        _write_config(ws, {"usage": {"daily_token_cap": 50}})
        usage_meter.record_call(ws, operation=usage_meter.OP_EXTRACTION, prompt_tokens=50, completion_tokens=0)

        with pytest.raises(DailyTokenCapExceeded) as exc:
            _query_llm("prompt", "m", "ollama", ollama_url=stub.base_url, workspace=ws)

        assert "DAILY TOKEN CAP" in str(exc.value)
        assert stub.calls == 0, "the cap must refuse BEFORE the model is called"

    def test_cap_error_escapes_the_auto_backend_fallback_loop(self, stub: ModelStub, ws: str) -> None:
        """DailyTokenCapExceeded is a RuntimeError; the loop must not eat it."""
        from mind_mem.llm_extractor import _query_llm

        _write_config(ws, {"usage": {"daily_token_cap": 1}})
        usage_meter.record_call(ws, operation=usage_meter.OP_EXTRACTION, prompt_tokens=1, completion_tokens=0)

        with pytest.raises(DailyTokenCapExceeded):
            _query_llm("prompt", "m", "auto", ollama_url=stub.base_url, workspace=ws)
        assert stub.calls == 0

    def test_under_the_cap_the_call_goes_through(self, stub: ModelStub, ws: str) -> None:
        from mind_mem.llm_extractor import _query_llm

        _write_config(ws, {"usage": {"daily_token_cap": 1_000_000}})
        assert _query_llm("prompt", "m", "ollama", ollama_url=stub.base_url, workspace=ws) == "[]"
        assert stub.calls == 1
        assert _ledger_totals(ws)[1] == 1


# ---------------------------------------------------------------------------
# 5 — real entry points
# ---------------------------------------------------------------------------


def _run_cli(argv: list[str], workspace: str) -> tuple[int, str, str]:
    from mind_mem.mm_cli import main

    out, err = io.StringIO(), io.StringIO()
    saved = os.environ.get("MIND_MEM_WORKSPACE")
    os.environ["MIND_MEM_WORKSPACE"] = workspace
    try:
        with redirect_stdout(out), redirect_stderr(err):
            code = main(argv)
    finally:
        if saved is None:
            os.environ.pop("MIND_MEM_WORKSPACE", None)
        else:
            os.environ["MIND_MEM_WORKSPACE"] = saved
    return code, out.getvalue(), err.getvalue()


class TestGraphBackfillEntryPoint:
    """mm graph-backfill -> graph_ingest.backfill -> extract_relations -> _query_llm."""

    @staticmethod
    def _enable(workspace: str, base_url: str, cap: int | None = None) -> None:
        payload: dict[str, Any] = {
            "extraction": {"enabled": True, "model": "m", "backend": "ollama", "ollama_url": base_url},
        }
        if cap is not None:
            payload["usage"] = {"daily_token_cap": cap}
        _write_config(workspace, payload)

    def test_backfill_meters_and_mm_usage_reports_it(self, ws: str) -> None:
        triples = json.dumps([{"subject": "starga", "predicate": "depends_on", "object": "mindc", "confidence": 0.9}])
        server = ModelStub(_ollama_reply(prompt_tokens=200, completion_tokens=20, text=triples))
        try:
            self._enable(ws, server.base_url)
            code, out, _ = _run_cli(["graph-backfill", "--limit", "1"], ws)
        finally:
            server.close()

        assert code == 0, out
        assert server.calls == 1
        total, calls, ops = _ledger_totals(ws)
        assert calls == 1
        assert total == 220
        assert ops == {usage_meter.OP_EXTRACTION: 220}

        code, report, _ = _run_cli(["usage"], ws)
        assert code == 0
        assert usage_meter.OP_EXTRACTION in report
        assert "220" in report

    def test_backfill_fails_closed_on_the_cap(self, ws: str) -> None:
        server = ModelStub(_ollama_reply(text="[]"))
        try:
            self._enable(ws, server.base_url, cap=10)
            usage_meter.record_call(ws, operation=usage_meter.OP_EXTRACTION, prompt_tokens=10, completion_tokens=0)
            code, out, err = _run_cli(["graph-backfill", "--limit", "1"], ws)
        finally:
            server.close()

        assert code == usage_meter.CAP_EXIT_CODE, out
        assert "DAILY TOKEN CAP" in err
        assert server.calls == 0, "no model call may happen once the cap is spent"


class TestRerankMetering:
    """recall stage 2.7 -> _recall_reranking.llm_rerank -> ollama /api/generate."""

    @staticmethod
    def _hits() -> list[dict[str, Any]]:
        return [
            {"_id": "A", "score": 5.0, "excerpt": "first candidate"},
            {"_id": "B", "score": 3.0, "excerpt": "second candidate"},
        ]

    def test_rerank_is_counted_under_its_own_operation(self, ws: str) -> None:
        from mind_mem._recall_reranking import llm_rerank

        server = ModelStub(_ollama_reply(prompt_tokens=80, completion_tokens=12, text="[0.9, 0.1]"))
        try:
            out = llm_rerank("q", self._hits(), url=f"{server.base_url}/api/generate", workspace=ws)
        finally:
            server.close()

        assert [h["_id"] for h in out] == ["A", "B"]
        total, calls, ops = _ledger_totals(ws)
        assert (total, calls) == (92, 1)
        assert ops == {usage_meter.OP_RERANK: 92}

    def test_rerank_without_workspace_writes_no_ledger(self, ws: str) -> None:
        from mind_mem._recall_reranking import llm_rerank

        server = ModelStub(_ollama_reply(text="[0.9, 0.1]"))
        try:
            llm_rerank("q", self._hits(), url=f"{server.base_url}/api/generate")
        finally:
            server.close()
        assert not os.path.exists(usage_meter.ledger_path(ws))

    def test_rerank_refuses_over_the_cap_instead_of_degrading_silently(self, ws: str) -> None:
        from mind_mem._recall_reranking import llm_rerank

        _write_config(ws, {"usage": {"daily_token_cap": 5}})
        usage_meter.record_call(ws, operation=usage_meter.OP_RERANK, prompt_tokens=5, completion_tokens=0)

        server = ModelStub(_ollama_reply(text="[0.9, 0.1]"))
        try:
            with pytest.raises(DailyTokenCapExceeded):
                llm_rerank("q", self._hits(), url=f"{server.base_url}/api/generate", workspace=ws)
            assert server.calls == 0
        finally:
            server.close()


class TestQueryExpansionMetering:
    """hybrid recall -> query_expansion.LLMQueryExpander -> /chat/completions.

    The one wired call site that can reach a *paid* provider, so the one the
    cap exists for: the roadmap forbids implicit paid-API calls without a
    budget ceiling.
    """

    @staticmethod
    def _config(base_url: str) -> dict[str, Any]:
        return {
            "provider": "openai",
            "model": "m",
            "api_key_env": "MIND_MEM_TEST_EXPANSION_KEY",
            "base_url": base_url,
        }

    def test_expansion_is_counted_under_its_own_operation(self, ws: str, monkeypatch: pytest.MonkeyPatch) -> None:
        from mind_mem.query_expansion import LLMQueryExpander

        monkeypatch.setenv("MIND_MEM_TEST_EXPANSION_KEY", "test-key")
        server = ModelStub(_oai_reply(text="find failures\nlocate errors"))
        try:
            expander = LLMQueryExpander(config=self._config(server.base_url), workspace=ws)
            out = expander.expand("find errors", max_expansions=3)
        finally:
            server.close()

        assert out[0] == "find errors"
        assert len(out) == 3
        total, calls, ops = _ledger_totals(ws)
        assert calls == 1
        assert total > 0
        assert ops == {usage_meter.OP_QUERY_EXPANSION: total}

    def test_expansion_refuses_over_the_cap_instead_of_falling_back_quietly(self, ws: str, monkeypatch: pytest.MonkeyPatch) -> None:
        from mind_mem.query_expansion import LLMQueryExpander

        monkeypatch.setenv("MIND_MEM_TEST_EXPANSION_KEY", "test-key")
        _write_config(ws, {"usage": {"daily_token_cap": 3}})
        usage_meter.record_call(ws, operation=usage_meter.OP_QUERY_EXPANSION, prompt_tokens=3, completion_tokens=0)

        server = ModelStub(_oai_reply(text="alt one"))
        try:
            expander = LLMQueryExpander(config=self._config(server.base_url), workspace=ws)
            with pytest.raises(DailyTokenCapExceeded):
                expander.expand("find errors", max_expansions=3)
            assert server.calls == 0
        finally:
            server.close()

    def test_expansion_without_workspace_writes_no_ledger(self, ws: str, monkeypatch: pytest.MonkeyPatch) -> None:
        from mind_mem.query_expansion import LLMQueryExpander

        monkeypatch.setenv("MIND_MEM_TEST_EXPANSION_KEY", "test-key")
        server = ModelStub(_oai_reply(text="alt one"))
        try:
            LLMQueryExpander(config=self._config(server.base_url)).expand("find errors", max_expansions=2)
        finally:
            server.close()
        assert not os.path.exists(usage_meter.ledger_path(ws))


# ---------------------------------------------------------------------------
# Report surface — the new operations have to be legible in `mm usage`
# ---------------------------------------------------------------------------


class TestUsageReportSurface:
    def test_report_breaks_the_day_down_by_operation(self, ws: str) -> None:
        usage_meter.record_call(ws, operation=usage_meter.OP_EXTRACTION, prompt_tokens=10, completion_tokens=2)
        usage_meter.record_call(ws, operation=usage_meter.OP_RERANK, prompt_tokens=4, completion_tokens=1)
        text = usage_meter.format_report(usage_meter.report(ws))
        assert usage_meter.OP_EXTRACTION in text
        assert usage_meter.OP_RERANK in text
        assert "12" in text and "5" in text
