# Copyright 2026 STARGA, Inc.
"""Deterministic labelled fixture for the computed-abstention tests.

Forty blocks, each with one topic-specific *rare* word and a handful of words
drawn from a shared *generic* vocabulary, so that:

* a **present** query (the block's rare word plus generics) has exactly one
  block that is the right answer;
* an **absent** query (generic words only) matches MANY blocks weakly -- the
  flat, no-clear-winner shape that "none of these is the answer" produces on a
  real corpus, and that an empty result list alone would never exercise.

Everything is derived from fixed LCG seeds: no clock, no ``random`` import, the
same workspace and the same query lists on every host. This is a FIXTURE, not
evidence about any real workspace -- it is what lets the falsification
condition in ROADMAP be run in CI at all, and it is deliberately the kind of
corpus on which the unindexed full-scan fallback does badly (see
``tests/test_abstention_separation.py``).
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import time
from collections.abc import Callable, Iterator
from typing import Any

#: Modules whose structured loggers a capture-fixture test would otherwise create.
_SERVING_STACK = (
    "mind_mem.mm_cli",
    "mind_mem.mcp_server",
    "mind_mem.mcp.tools.recall",
    "mind_mem.recall",
    "mind_mem.hybrid_recall",
    "mind_mem.http_transport",
    "mind_mem.recall_attestation",
    "mind_mem.recall_cache",
    "mind_mem.retrieval_trace",
    "mind_mem.vector_inertness",
    "mind_mem.ffi",
    "mind_mem.abstention_store",
    "mind_mem.abstention_eval",
)


def preload_serving_stack() -> None:
    """Import the serving stack at collection time, before any ``capsys`` swaps stderr.

    Every mind-mem structured logger binds a handler to whatever ``sys.stderr``
    is when its module is first imported. Under ``capsys`` that is a per-test
    buffer closed when the test ends, and every later test in the same process
    that re-points the handlers (``test_v4_logging_context_wiring``) then dies
    with ``ValueError: I/O operation on closed file`` -- which of two runs fails
    depends only on which test a worker happened to receive first. A test module
    that drives the CLI under ``capsys`` calls this at import so the loggers are
    created against the session's stderr instead.
    """
    import importlib

    for name in _SERVING_STACK:
        with contextlib.suppress(ImportError):  # best effort: ``ffi`` is optional
            importlib.import_module(name)


GENERIC = (
    "team policy process review weekly update owner decision status plan "
    "release change risk budget customer support quality standard approval schedule"
).split()

RARE = (
    "zephyr quokka tundra mosaic falcon ember glacier harbor indigo juniper "
    "kestrel lantern meridian nimbus obsidian paladin quartz raven sable talon "
    "umber vortex willow xenon yarrow zircon aurora basalt cobalt dune "
    "eclipse fjord garnet helix ivory jasper krypton lumen magnet nectar"
).split()


def _lcg(seed: int) -> Iterator[int]:
    state = seed
    while True:
        state = (state * 1103515245 + 12345) % (2**31)
        yield state


def blocks() -> list[tuple[str, str, str]]:
    """``(block_id, rare_word, statement)`` for every fixture block."""
    rng = _lcg(7)
    out: list[tuple[str, str, str]] = []
    for index, rare in enumerate(RARE):
        gens = [GENERIC[next(rng) % len(GENERIC)] for _ in range(7)]
        statement = (
            f"The {rare} initiative: the {gens[0]} {gens[1]} requires {gens[2]} {gens[3]} "
            f"and {gens[4]} {gens[5]} {gens[6]} for {rare} work."
        )
        out.append((f"D-20260801-{index + 1:03d}", rare, statement))
    return out


def queries() -> tuple[list[str], list[str]]:
    """``(present, absent)`` -- forty queries each."""
    rng = _lcg(99)
    present: list[str] = []
    absent: list[str] = []
    for rare in RARE:
        g = [GENERIC[next(rng) % len(GENERIC)] for _ in range(2)]
        present.append(f"what is the {rare} {g[0]} {g[1]}")
    for _ in range(40):
        g = [GENERIC[next(rng) % len(GENERIC)] for _ in range(4)]
        absent.append(f"what is our {g[0]} {g[1]} {g[2]} {g[3]}")
    return present, absent


def make_workspace(root: str, *, indexed: bool = True, config: dict[str, object] | None = None) -> str:
    """Write the fixture corpus under ``root/ws`` and return the workspace path.

    ``indexed=True`` builds the FTS5 index (the product's normal operating
    mode); ``False`` leaves the unindexed full-scan fallback. Files are aged
    past the recall cache's racy-timestamp window so a cache-enabled test is
    not silently bypassed.
    """
    ws = os.path.join(root, "ws")
    for name in ("decisions", "tasks", "entities", "intelligence", "memory"):
        os.makedirs(os.path.join(ws, name), exist_ok=True)
    body = "\n".join(
        f"[{bid}]\nStatement: {statement}\nDate: 2026-08-01\nStatus: active\nTags: {rare}\n\n---\n" for bid, rare, statement in blocks()
    )
    with open(os.path.join(ws, "decisions", "DECISIONS.md"), "w", encoding="utf-8") as handle:
        handle.write(body)
    cfg: dict[str, object] = {"recall": {"vector_enabled": False, "provider": "local"}}
    if config:
        recall = dict(cfg["recall"])  # type: ignore[call-overload]
        recall.update(config)
        cfg["recall"] = recall
    with open(os.path.join(ws, "mind-mem.json"), "w", encoding="utf-8") as handle:
        json.dump(cfg, handle)
    if indexed:
        from mind_mem.sqlite_index import build_index

        build_index(ws)
    # Age EVERYTHING (the index included) past the recall cache's racy-timestamp
    # window, after it is built: a file modified inside the window makes the
    # corpus stamp unnameable and the request bypasses the cache.
    past = time.time() - 3600
    for base, _dirs, files in os.walk(ws):
        for name in files:
            os.utime(os.path.join(base, name), (past, past))
    return ws


def fake_engine(scores: list[float], *, backend: str = "sqlite", extra: dict[str, Any] | None = None) -> Callable[..., str]:
    """A stand-in for the cache-miss branch returning hits with exactly *scores*."""
    from mind_mem.mcp.infra.constants import MCP_SCHEMA_VERSION
    from mind_mem.scoring_instant import format_scoring_instant, resolve_scoring_instant

    def fn(
        query: str, limit: int = 10, active_only: bool = False, backend_arg: str = "auto", scoring_instant: Any = None, **_kw: Any
    ) -> str:
        hits = [
            {
                "_id": f"D-20260801-{index:03d}",
                "type": "decision",
                "score": score,
                "excerpt": f"hit {index}",
                "file": "decisions/DECISIONS.md",
                "line": index,
                "status": "active",
                "Date": "2026-08-01",
                **(extra or {}),
            }
            for index, score in enumerate(scores, 1)
        ]
        return json.dumps(
            {
                "_schema_version": MCP_SCHEMA_VERSION,
                "backend": backend,
                "query": query,
                "query_id": "",
                "count": len(hits),
                "scoring_instant": format_scoring_instant(resolve_scoring_instant(scoring_instant)),
                "results": hits,
            }
        )

    return fn


@contextlib.contextmanager
def log_events() -> Iterator[list[str]]:
    """Record every ``component:event`` the mind-mem structured loggers emit.

    The loggers write through handlers bound to whatever ``sys.stderr`` was at
    first use, so ``capsys`` / ``capfd`` never see them; a handler on the
    logger objects themselves does. Loggers are created at import, so warm the
    code path once before relying on a recording (an empty recording is
    otherwise indistinguishable from a silent one -- assert a positive control).
    """
    events: list[str] = []

    class _Recorder(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            events.append(f"{record.name}:{record.msg}")

    handler = _Recorder(level=logging.DEBUG)
    targets = [
        lg for name, lg in logging.Logger.manager.loggerDict.items() if name.startswith("mind-mem.") and isinstance(lg, logging.Logger)
    ]
    for target in targets:
        target.addHandler(handler)
    try:
        yield events
    finally:
        for target in targets:
            target.removeHandler(handler)
