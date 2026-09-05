# Copyright 2026 STARGA, Inc.
"""Auto-enabled expansion / decomposition is a DEPTH-1 decision.

The defect this file pins, shipped on the default recall path from v3.3.0
(decomposition) and v4.0.2 (expansion) through 5.0.1:

``HybridBackend._search_expanded`` fans its variants out and calls
``search(..., _skip_auto_features=True)`` for each one. That flag exists for
exactly one purpose -- stop the nested call re-entering the feature that
started the fan-out -- and its docstring says so. But the *auto-enable*
branches under it read only ``expansion_active`` / ``decomp_active``, never
the flag, so they flipped the feature back on inside the nested call. Since
**variant 0 of every expansion is the original query**, a temporal or
multi-hop query re-expanded into an identical list at every level, each level
holding a pool blocked in ``Future.result`` on the next. One process reached
30,935 OS threads at roughly 600 threads/second.

``auto_enable`` defaults to ``True`` and a workspace with no
``query_expansion`` section gets it, so this was on in production, and on the
REST / gRPC surfaces it was remotely triggerable.

Why this file is thread-free
----------------------------
Reproducing the bug by letting it leak threads would be a test that damages
the machine that runs it -- and under the suite's thread ceiling it does not
even report a failure, it reports a SIGKILL. So the fan-out's executor is
replaced with a **serial** one and the recursion is counted instead of felt.
The counter is the evidence; no OS thread is created, and each test asserts
that (``threading.active_count()`` is unchanged across the call).

Every assertion here is paired with something that can make it fail:

* ``entered == 1`` is not a constant -- :func:`test_the_depth_probe_counts_
  every_entry` drives two entries through the same probe and sees 2.
* ``entered == 0`` under ``_skip_auto_features=True`` is paired with the same
  query entering once WITHOUT the flag, so the zero is about the flag rather
  than about a query that never expanded.
* the serial executor's instance list is asserted non-empty, so a test cannot
  pass because the monkeypatch missed and the fan-out never ran.
* the log-record tests assert the *distinguishing* field on both branches, so
  neither passes by everything being labelled the same way.
"""

from __future__ import annotations

import logging
import os
import threading
from concurrent.futures import Future
from typing import Any

import pytest

from mind_mem.hybrid_recall import HybridBackend
from mind_mem.init_workspace import init

#: Classified ``temporal`` by ``_recall_detection.detect_query_type`` and
#: expanded to more than one variant, which is what makes the fan-out run.
_TEMPORAL_QUERY = "what happened before the migration in March"

#: Classified ``multi-hop``. Expands to two variants and decomposes to four,
#: so it reaches BOTH auto-enable branches depending on config.
_MULTI_HOP_QUERY = "why did the team choose the parser and how does it compare to the previous one and what changed"

#: A plain lookup: neither branch's query type, so neither fires.
_SINGLE_HOP_QUERY = "deterministic compiler evidence chain"

#: Depth at which the probe stops the recursion instead of letting it run.
#: Without this a mutated (pre-fix) source recurses until CPython's own limit,
#: doing a full BM25 search at every level. The probe records the entry first,
#: so the assertion still sees the re-entry it aborted.
_ABORT_DEPTH = 2

#: The fixed path's whole thread budget for one top-level search: the fan-out
#: pool (<=4 workers) plus, when the dense leg is on, one 2-worker leg pool per
#: variant -- 4 + 4*2 = 12. Recursion is what made that number unbounded.
_WORKER_BUDGET = 12


class _ReEntered(RuntimeError):
    """The fan-out was entered again from inside itself."""


class _SerialExecutor:
    """A ``ThreadPoolExecutor`` that creates no threads.

    A full drop-in: ``map``, ``submit`` and ``shutdown``, so replacing it
    changes only *where* the work runs, never *whether* it runs. A partial
    stand-in would silently disable whatever leg it could not serve, and the
    test would then be measuring a pipeline the product never runs.

    Instances are recorded so a test can assert the fan-out actually ran
    through here -- otherwise a monkeypatch that missed would make every count
    look correct.
    """

    def __init__(self, max_workers: int | None = None, *args: Any, **kwargs: Any) -> None:
        self.max_workers = max_workers
        self.mapped = 0

    def __enter__(self) -> "_SerialExecutor":
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False

    def map(self, fn: Any, *iterables: Any) -> list[Any]:
        self.mapped += 1
        return [fn(*args) for args in zip(*iterables)]

    def submit(self, fn: Any, *args: Any, **kwargs: Any) -> Future:
        future: Future = Future()
        future.set_running_or_notify_cancel()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as exc:  # noqa: BLE001 - mirrors the pool contract
            future.set_exception(exc)
        return future

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        return None


class _DepthProbe:
    """Counts entries into the fan-out and how deep they nested."""

    def __init__(self) -> None:
        self.entered = 0
        self.max_depth = 0
        self._depth = 0

    def wrap(self, original: Any) -> Any:
        def wrapper(inner_self: Any, *args: Any, **kwargs: Any) -> Any:
            self.entered += 1
            self._depth += 1
            self.max_depth = max(self.max_depth, self._depth)
            try:
                if self._depth > _ABORT_DEPTH:
                    raise _ReEntered(f"fan-out re-entered at depth {self._depth}")
                return original(inner_self, *args, **kwargs)
            finally:
                self._depth -= 1

        return wrapper


@pytest.fixture
def probe(monkeypatch: pytest.MonkeyPatch) -> _DepthProbe:
    """Serial executor + a depth counter on the fan-out. No threads."""
    executors: list[_SerialExecutor] = []

    class _Recorded(_SerialExecutor):
        def __init__(self, max_workers: int | None = None, *args: Any, **kwargs: Any) -> None:
            super().__init__(max_workers, *args, **kwargs)
            executors.append(self)

    # ``_search_expanded`` imports ThreadPoolExecutor from concurrent.futures
    # at call time, so the patch has to land on the source module. The
    # hybrid_recall global (the vector leg's own 2-worker pool) is patched too
    # so no test in this file can create a thread by any route.
    monkeypatch.setattr("concurrent.futures.ThreadPoolExecutor", _Recorded)
    monkeypatch.setattr("mind_mem.hybrid_recall.ThreadPoolExecutor", _Recorded)

    p = _DepthProbe()
    p.executors = executors  # type: ignore[attr-defined]
    monkeypatch.setattr(HybridBackend, "_search_expanded", p.wrap(HybridBackend._search_expanded))
    return p


def _write_blocks(ws: str, n: int) -> None:
    decisions = os.path.join(ws, "decisions")
    os.makedirs(decisions, exist_ok=True)
    body = ["# DECISIONS\n\n---\n"]
    for i in range(1, n + 1):
        body.append(
            f"\n[D-20260101-{i:06d}]\n"
            f"Date: 2026-06-20\n"
            f"Status: active\n"
            f"Scope: global\n"
            f"Statement: the migration in March moved the parser and the team "
            f"compared it to the previous deterministic compiler evidence chain {i}\n"
            f"Tags: compiler, recall\n"
        )
    with open(os.path.join(decisions, "DECISIONS.md"), "w", encoding="utf-8") as handle:
        handle.write("".join(body))


@pytest.fixture
def workspace(tmp_path) -> str:
    ws = str(tmp_path / "ws")
    os.makedirs(ws, exist_ok=True)
    init(ws)
    _write_blocks(ws, 8)
    return ws


def _backend(**recall: Any) -> HybridBackend:
    """A BM25-only hybrid backend. The dense leg is irrelevant to re-entry."""
    # The cross-encoder auto-enables on exactly these query types too. It is a
    # different feature with its own decision, and letting it load a model here
    # would make the test measure the reranker's latency instead of the
    # fan-out's depth -- so it is off, explicitly.
    cfg: dict[str, Any] = {
        "vector_enabled": False,
        "knee_cutoff": False,
        "cross_encoder": {"auto_enable": False},
    }
    cfg.update(recall)
    return HybridBackend(config=cfg)


def _assert_fanout_ran_serially(probe: _DepthProbe) -> None:
    """The fan-out really went through the serial executor, within budget."""
    executors = probe.executors  # type: ignore[attr-defined]
    assert executors, "the fan-out never constructed an executor -- the monkeypatch missed"
    requested = sum((ex.max_workers or 0) for ex in executors)
    assert requested <= _WORKER_BUDGET, (
        f"one top-level search asked for {requested} workers across {len(executors)} "
        f"pool(s); the documented budget for the depth-1 path is {_WORKER_BUDGET}"
    )


# ---------------------------------------------------------------------------
# 1. The fan-out is entered exactly once
# ---------------------------------------------------------------------------


def test_temporal_query_enters_the_fanout_exactly_once(workspace: str, probe: _DepthProbe) -> None:
    """The v4.0.2 half: a temporal query auto-enables expansion once."""
    before = threading.active_count()
    hits = _backend().search(_TEMPORAL_QUERY, workspace, limit=5)

    assert probe.entered == 1, (
        f"the expansion fan-out was entered {probe.entered} times for one temporal query "
        f"(max depth {probe.max_depth}); the auto-enable branch is re-entering itself"
    )
    assert probe.max_depth == 1
    _assert_fanout_ran_serially(probe)
    assert threading.active_count() == before, "this test must not create OS threads"
    assert list(hits), "the query retrieved nothing, so the count is about a search that did not happen"


def test_multi_hop_query_enters_the_fanout_exactly_once(workspace: str, probe: _DepthProbe) -> None:
    """A multi-hop query reaches the same expansion branch, once."""
    before = threading.active_count()
    hits = _backend().search(_MULTI_HOP_QUERY, workspace, limit=5)

    assert probe.entered == 1, (
        f"the expansion fan-out was entered {probe.entered} times for one multi-hop query (max depth {probe.max_depth})"
    )
    assert probe.max_depth == 1
    _assert_fanout_ran_serially(probe)
    assert threading.active_count() == before, "this test must not create OS threads"
    assert list(hits)


def test_multi_hop_decomposition_enters_the_fanout_exactly_once(workspace: str, probe: _DepthProbe) -> None:
    """The v3.3.0 half.

    Expansion runs first and would return before decomposition is reached, so
    its auto-enable is switched off here -- that is what routes a multi-hop
    query to the decomposition branch, which has the identical defect and its
    own shipped version.
    """
    before = threading.active_count()
    backend = _backend(query_expansion={"auto_enable": False})
    hits = backend.search(_MULTI_HOP_QUERY, workspace, limit=5)

    assert probe.entered == 1, (
        f"the decomposition fan-out was entered {probe.entered} times for one multi-hop query (max depth {probe.max_depth})"
    )
    assert probe.max_depth == 1
    _assert_fanout_ran_serially(probe)
    assert threading.active_count() == before, "this test must not create OS threads"
    assert list(hits)


# ---------------------------------------------------------------------------
# 2. Controls: the counts above are about the flag, and can move
# ---------------------------------------------------------------------------


def test_skip_auto_features_takes_the_plain_single_query_pipeline(workspace: str, probe: _DepthProbe) -> None:
    """The nested call's contract, asserted directly.

    Paired with the temporal test above: the same query, same backend, enters
    the fan-out once WITHOUT the flag and zero times WITH it. Without that
    pairing a zero here would also be what a query that never expands looks
    like.
    """
    backend = _backend()
    hits = backend.search(_TEMPORAL_QUERY, workspace, limit=5, _skip_auto_features=True)

    assert probe.entered == 0, "the nested call re-entered the fan-out; _skip_auto_features is not being honoured"
    assert list(hits), "the flagged call served nothing, so the zero is not evidence about the flag"

    probe.entered = 0
    backend.search(_TEMPORAL_QUERY, workspace, limit=5)
    assert probe.entered == 1, "the same query without the flag must enter once -- otherwise the zero above is vacuous"


def test_single_hop_query_never_enters_the_fanout(workspace: str, probe: _DepthProbe) -> None:
    """Off-path queries are untouched by either branch."""
    hits = _backend().search(_SINGLE_HOP_QUERY, workspace, limit=5)
    assert probe.entered == 0
    assert list(hits)


def test_the_depth_probe_counts_every_entry(workspace: str, probe: _DepthProbe) -> None:
    """Positive control on the instrument itself.

    ``entered == 1`` above is only evidence if the counter is capable of
    reporting more than one. Two top-level searches, two entries.
    """
    backend = _backend()
    backend.search(_TEMPORAL_QUERY, workspace, limit=5)
    backend.search(_MULTI_HOP_QUERY, workspace, limit=5)
    assert probe.entered == 2, "the probe is not counting entries; every count in this file would be meaningless"


# ---------------------------------------------------------------------------
# 3. The swallow that hid this for three releases
# ---------------------------------------------------------------------------


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def captured_log() -> Any:
    logger = logging.getLogger("mind-mem.hybrid_recall")
    handler = _Capture()
    previous = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        yield handler
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous)


def _failure_record(handler: _Capture) -> logging.LogRecord:
    matches = [r for r in handler.records if r.msg == "query_expansion_failed"]
    assert matches, f"no query_expansion_failed record was emitted; saw {[r.msg for r in handler.records]}"
    return matches[-1]


def test_thread_exhaustion_is_named_in_the_log_record(workspace: str, captured_log: _Capture, monkeypatch: pytest.MonkeyPatch) -> None:
    """The terminating error is no longer indistinguishable from a miss.

    ``RuntimeError: can't start new thread`` is what the runaway recursion
    finally raised, and it was logged as an ordinary
    ``query_expansion_failed / fallback=single_query`` -- a warning that reads
    like "this query did not expand". That is why v4.0.2 through 5.0.1 shipped
    the defect with nothing in any log to find.
    """

    def _out_of_threads(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr("mind_mem.query_expansion.expand_queries", _out_of_threads)

    hits = _backend().search(_TEMPORAL_QUERY, workspace, limit=5)

    record = _failure_record(captured_log)
    assert record.data["failure_kind"] == "thread_exhaustion"
    assert record.levelno == logging.ERROR, "thread exhaustion must not be logged at the same level as a miss"
    # And the serving contract is unchanged: a recall still answers.
    assert list(hits), "the fallback must still serve the single-query result"


def test_an_ordinary_expansion_miss_is_not_named_thread_exhaustion(
    workspace: str, captured_log: _Capture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other half. Without this, labelling everything ``thread_exhaustion``
    would pass the test above and distinguish nothing."""

    def _ordinary_failure(*args: Any, **kwargs: Any) -> Any:
        raise ValueError("synonym table unavailable")

    monkeypatch.setattr("mind_mem.query_expansion.expand_queries", _ordinary_failure)

    hits = _backend().search(_TEMPORAL_QUERY, workspace, limit=5)

    record = _failure_record(captured_log)
    assert record.data["failure_kind"] == "expansion_error"
    assert record.levelno == logging.WARNING
    assert list(hits)
