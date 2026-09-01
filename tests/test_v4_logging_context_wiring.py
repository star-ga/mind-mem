"""``v4.logging_context`` is WIRED — 5.1.0 restoration slice.

The 5.0.0 sweep deleted this module because nothing imported it. It is back,
and this file is the evidence that it is *reached*, not merely importable:

* :mod:`mind_mem.observability` installs :class:`StructuredLogFilter` on the
  handler ``StructuredLogger`` owns, so every mind-mem log record gains the
  active context, and :class:`JSONFormatter` serialises it.
* :func:`mind_mem.mcp.infra.observability.mcp_tool_observe` runs each tool
  call inside a fresh ``with_correlation_id`` scope.

Working definition, asserted below: **two concurrent recalls emit distinct
``correlation_id`` values on every log line.**

Both halves are gated on ``v4.logging_context`` and default OFF; the flag-OFF
tests pin that the emitted JSON is byte-for-byte what 5.0.0 emitted.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import threading
from pathlib import Path

import pytest

from mind_mem import observability
from mind_mem.mcp.tools import recall as recall_tools

_TIMEOUT = 30.0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _build_workspace(root: Path) -> None:
    """Minimal but real corpus — enough for a genuine BM25 recall."""
    for sub in ("decisions", "tasks", "entities", "intelligence", "memory"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    (root / "decisions" / "DECISIONS.md").write_text(
        "[D-20260101-001]\n"
        "Statement: Use PostgreSQL for the user database\n"
        "Status: active\n"
        "Date: 2026-01-01\n"
        "\n---\n\n"
        "[D-20260102-001]\n"
        "Statement: Use Redis for caching hot recall results\n"
        "Status: active\n"
        "Date: 2026-01-02\n",
        encoding="utf-8",
    )
    (root / "tasks" / "TASKS.md").write_text(
        "[T-20260101-001]\nDescription: Set up PostgreSQL in staging\nStatus: open\n",
        encoding="utf-8",
    )
    (root / "entities" / "projects.md").write_text(
        "[PRJ-mind-mem]\nName: mind-mem\nStatus: active\n",
        encoding="utf-8",
    )


def _write_config(root: Path, *, logging_context: bool) -> Path:
    cfg = root / "mind-mem.json"
    body: dict = {"version": "5.1.0", "cache": {"enabled": False}}
    if logging_context:
        body["v4"] = {"logging_context": {"enabled": True}}
    cfg.write_text(json.dumps(body), encoding="utf-8")
    return cfg


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "ws"
    root.mkdir()
    _build_workspace(root)
    monkeypatch.setenv("MIND_MEM_WORKSPACE", str(root))
    monkeypatch.setenv("MIND_MEM_SCOPE", "user")
    return root


@pytest.fixture
def rate_limit_budget():
    """``mcp_tool_observe``'s 120-call/60s window is process-global."""
    from mind_mem.mcp.infra import rate_limit

    def _clear() -> None:
        with rate_limit._rate_limiters_lock:
            rate_limit._rate_limiters.clear()

    _clear()
    yield
    _clear()


def _point_config_at_nothing(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve the active config to a file that does not exist."""
    monkeypatch.setenv("MIND_MEM_CONFIG", str(workspace / "absent-mind-mem.json"))


@pytest.fixture
def disarmed(workspace: Path, monkeypatch: pytest.MonkeyPatch, rate_limit_budget):
    """Flag OFF — the 5.0.0 baseline."""
    cfg = _write_config(workspace, logging_context=False)
    monkeypatch.setenv("MIND_MEM_CONFIG", str(cfg))
    observability.sync_log_context()
    try:
        yield workspace
    finally:
        _point_config_at_nothing(workspace, monkeypatch)
        assert observability.sync_log_context() is False


@pytest.fixture
def armed(workspace: Path, monkeypatch: pytest.MonkeyPatch, rate_limit_budget):
    """Flag ON — the same call ``sync_log_context`` makes at logger build time."""
    cfg = _write_config(workspace, logging_context=True)
    monkeypatch.setenv("MIND_MEM_CONFIG", str(cfg))
    assert observability.sync_log_context() is True
    try:
        yield workspace
    finally:
        # Disarm so the filter cannot leak `ctx` into any later test.
        # Unsetting MIND_MEM_CONFIG is not enough: _config_path() then falls
        # back to $MIND_MEM_WORKSPACE/mind-mem.json, which is the armed file.
        _point_config_at_nothing(workspace, monkeypatch)
        assert observability.sync_log_context() is False


@contextlib.contextmanager
def _capture_logs():
    """Redirect every mind-mem handler onto one buffer.

    The stream is swapped on the *real* handler, so the captured bytes went
    through the real filter chain and the real ``JSONFormatter``.
    """
    buf = io.StringIO()
    originals = {name: h.stream for name, h in observability._owned_handlers.items()}
    for name in originals:
        observability._owned_handlers[name].setStream(buf)
    try:
        yield buf
    finally:
        for name, stream in originals.items():
            observability._owned_handlers[name].setStream(stream)


def _lines(buf: io.StringIO) -> list[dict]:
    out = []
    for raw in buf.getvalue().splitlines():
        raw = raw.strip()
        if raw:
            out.append(json.loads(raw))
    return out


def _tool_calls(lines: list[dict], tool: str) -> list[dict]:
    return [entry for entry in lines if entry.get("event") == "mcp_tool_call" and entry.get("data", {}).get("tool_name") == tool]


# ---------------------------------------------------------------------------
# The working definition
# ---------------------------------------------------------------------------


class TestConcurrentRecallsAreCorrelated:
    def test_two_concurrent_recalls_emit_distinct_correlation_ids(self, armed: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The slice's definition of working, with the overlap forced.

        A barrier inside ``_recall_impl`` holds both calls open at the same
        instant, so this is genuine concurrency rather than two calls that
        merely happened to be issued from two threads. Everything the ids flow
        through — the decorator, the contextvar stack, the filter, the
        formatter — is the production object.
        """
        original_impl = recall_tools._recall_impl
        barrier = threading.Barrier(2, timeout=_TIMEOUT)
        seen_inside: dict[str, str | None] = {}

        def _synchronised_impl(query, *args, **kwargs):
            from mind_mem.v4.logging_context import current_context

            seen_inside[query] = current_context().get("correlation_id")
            barrier.wait()  # both calls are now inside the tool body
            return original_impl(query, *args, **kwargs)

        monkeypatch.setattr(recall_tools, "_recall_impl", _synchronised_impl)

        queries = ["PostgreSQL user database", "Redis caching hot results"]
        errors: list[BaseException] = []
        results: dict[str, str] = {}

        def _run(q: str) -> None:
            try:
                results[q] = recall_tools.recall(q, limit=3, backend="bm25")
            except BaseException as exc:  # noqa: BLE001 - re-raised below
                errors.append(exc)

        with _capture_logs() as buf:
            threads = [threading.Thread(target=_run, args=(q,), name=f"recall-{i}") for i, q in enumerate(queries)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=_TIMEOUT)
                assert not t.is_alive(), "recall thread did not finish — the barrier never released"

        assert not errors, errors
        assert set(results) == set(queries)
        for payload in results.values():
            assert "error" not in json.loads(payload), payload

        lines = _lines(buf)
        assert lines, "no log lines captured — the recall path emitted nothing to assert on"

        # 1. EVERY log line carries a correlation id.
        uncorrelated = [entry for entry in lines if not entry.get("ctx", {}).get("correlation_id")]
        assert not uncorrelated, f"log lines with no correlation_id: {uncorrelated}"

        # 2. The two concurrent calls got DIFFERENT ids.
        calls = _tool_calls(lines, "recall")
        assert len(calls) == 2, f"expected one mcp_tool_call per recall, got {calls}"
        ids = {entry["ctx"]["correlation_id"] for entry in calls}
        assert len(ids) == 2, f"the two concurrent recalls shared a correlation id: {ids}"

        # 3. The id on the log line is the id that was live INSIDE that call,
        #    i.e. it identifies the call rather than being a per-line uuid.
        assert set(seen_inside.values()) == ids, (seen_inside, ids)
        assert None not in seen_inside.values()

    def test_real_end_to_end_recall_is_correlated_with_nothing_patched(self, armed: Path) -> None:
        """No monkeypatching at all: the shipped ``recall`` tool, as called."""
        with _capture_logs() as buf:
            payload = recall_tools.recall("PostgreSQL", limit=3, backend="bm25")

        assert "error" not in json.loads(payload), payload
        calls = _tool_calls(_lines(buf), "recall")
        assert len(calls) == 1
        assert calls[0]["ctx"]["correlation_id"]

    def test_nested_calls_inherit_the_outer_correlation_id(self, armed: Path) -> None:
        """One request, one id — an inner tool call does not mint a second."""
        from mind_mem.v4.logging_context import with_context

        with _capture_logs() as buf, with_context(correlation_id="outer-fixed-id"):
            recall_tools.recall("Redis", limit=3, backend="bm25")

        calls = _tool_calls(_lines(buf), "recall")
        assert len(calls) == 1
        assert calls[0]["ctx"]["correlation_id"] == "outer-fixed-id"


# ---------------------------------------------------------------------------
# The wiring is what produces the id — remove it and the assertions above die
# ---------------------------------------------------------------------------


class TestTheWiringIsLoadBearing:
    def test_the_id_comes_from_the_decorator_not_from_ambient_state(self, armed: Path) -> None:
        """Same body, called around ``mcp_tool_observe``: no correlation id.

        ``recall.__wrapped__`` is the undecorated function. If the ids came
        from anywhere but the decorator, this would still be correlated.
        """
        with _capture_logs() as buf:
            recall_tools.recall.__wrapped__("PostgreSQL", limit=3, backend="bm25")

        lines = _lines(buf)
        assert not _tool_calls(lines, "recall"), "the undecorated body should not emit mcp_tool_call"
        assert not any(entry.get("ctx") for entry in lines), f"unexpected context outside the decorator: {lines}"

    def test_a_root_logger_install_would_be_a_no_op(self, armed: Path) -> None:
        """Why the filter goes on our OWN handler, not the root logger.

        ``StructuredLogger`` sets ``propagate = False``, so the arrangement the
        v4 module's docstring suggests — ``logging.getLogger().addFilter(...)``
        — never sees a single mind-mem record. This test pins that, so nobody
        "simplifies" the plug point back to the root logger.
        """
        import logging

        from mind_mem.v4.logging_context import StructuredLogFilter

        log = observability.get_logger("mcp_server")
        handler = observability._owned_handlers[log._logger.name]
        installed = [f for f in handler.filters if isinstance(f, StructuredLogFilter)]
        assert installed, "the filter is not on the handler mind-mem actually logs through"
        assert log._logger.propagate is False

        root_only = StructuredLogFilter()
        logging.getLogger().addFilter(root_only)
        try:
            for f in installed:
                handler.removeFilter(f)
            with _capture_logs() as buf:
                log.info("root_filter_probe")
            lines = _lines(buf)
        finally:
            logging.getLogger().removeFilter(root_only)
            for f in installed:
                handler.addFilter(f)

        assert lines and lines[0]["event"] == "root_filter_probe"
        assert "ctx" not in lines[0], "a root-logger filter appeared to work — propagate must have been left True"


# ---------------------------------------------------------------------------
# Flag OFF is the 5.0.0 baseline
# ---------------------------------------------------------------------------


class TestFlagOffIsUnchanged:
    def test_surface_is_off_by_default(self, disarmed: Path) -> None:
        assert observability.log_context_active() is False

    def test_no_filter_is_installed_on_any_mind_mem_handler(self, disarmed: Path) -> None:
        from mind_mem.v4.logging_context import StructuredLogFilter

        armed_handlers = [
            name for name, h in observability._owned_handlers.items() if any(isinstance(f, StructuredLogFilter) for f in h.filters)
        ]
        assert not armed_handlers, armed_handlers

    def test_recall_log_lines_have_the_5_0_0_shape(self, disarmed: Path) -> None:
        with _capture_logs() as buf:
            payload = recall_tools.recall("PostgreSQL", limit=3, backend="bm25")

        assert "error" not in json.loads(payload), payload
        raw = buf.getvalue()
        assert "ctx" not in raw, raw
        assert "correlation_id" not in raw, raw

        calls = _tool_calls(_lines(buf), "recall")
        assert len(calls) == 1
        assert set(calls[0]) == {"ts", "level", "component", "event", "data"}
        assert set(calls[0]["data"]) == {"tool_name", "duration_ms", "success", "error_type", "result_size"}

    def test_two_concurrent_recalls_still_work_with_the_flag_off(self, disarmed: Path) -> None:
        """Flag OFF must not merely be quiet — it must still serve recalls."""
        results: dict[str, str] = {}

        def _run(q: str) -> None:
            results[q] = recall_tools.recall(q, limit=3, backend="bm25")

        with _capture_logs() as buf:
            threads = [threading.Thread(target=_run, args=(q,)) for q in ("PostgreSQL", "Redis")]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=_TIMEOUT)
                assert not t.is_alive()

        assert len(results) == 2
        assert "ctx" not in buf.getvalue()


# ---------------------------------------------------------------------------
# Registry + determinism
# ---------------------------------------------------------------------------


class TestFlagRegistration:
    def test_flag_is_registered(self) -> None:
        from mind_mem.v4 import feature_flags

        assert "logging_context" in feature_flags.ALL_V4_FLAGS

    def test_unparseable_config_leaves_the_surface_off(self, workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        cfg = workspace / "mind-mem.json"
        cfg.write_text("{ not json", encoding="utf-8")
        monkeypatch.setenv("MIND_MEM_CONFIG", str(cfg))
        try:
            assert observability.sync_log_context() is False
        finally:
            monkeypatch.delenv("MIND_MEM_CONFIG", raising=False)
            observability.sync_log_context()

    def test_correlation_ids_never_touch_the_scored_path(self, armed: Path) -> None:
        """Determinism: the ids live in the log envelope, never in the answer.

        recall is documented as a pure function of (corpus, config,
        scoring_instant). A uuid4 minted per call must not reach the payload.
        """
        # One warm-up call: the first recall against a fresh workspace builds
        # the FTS index, and a cold-vs-warm leg is not what is under test here.
        recall_tools.recall("PostgreSQL", limit=3, backend="bm25", scoring_instant="2026-01-15")
        first = recall_tools.recall("PostgreSQL", limit=3, backend="bm25", scoring_instant="2026-01-15")
        second = recall_tools.recall("PostgreSQL", limit=3, backend="bm25", scoring_instant="2026-01-15")
        assert "correlation_id" not in first
        assert json.loads(first)["results"] == json.loads(second)["results"]


def test_no_write_path_was_added(armed: Path) -> None:
    """The governed write path is untouched: this slice only reads + logs."""
    before = sorted(p.name for p in (armed / "decisions").iterdir())
    recall_tools.recall("PostgreSQL", limit=3, backend="bm25")
    after = sorted(p.name for p in (armed / "decisions").iterdir())
    assert before == after
    assert not os.path.exists(armed / "proposals")
