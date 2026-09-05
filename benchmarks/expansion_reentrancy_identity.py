# Copyright 2026 STARGA, Inc.
"""Served-ranking evidence for the 5.0.2 expansion / decomposition depth guard.

The guard stops ``search(..., _skip_auto_features=True)`` from re-enabling the
very feature that started its fan-out. Two claims have to be measured rather
than argued, and this script produces both as re-runnable artifacts:

**Off-path: byte-identical.** For queries that are neither ``temporal`` nor
``multi-hop`` neither auto-enable branch ever fired, so the guard cannot have
changed anything. Run this on the pre-fix and post-fix trees; the digests must
match exactly. That is the positive proof the fix is inert where it does not
apply.

**On-path: the pre-fix ranking was not a function of the corpus.** The
pre-fix code recursed until the OS refused a thread, and the resulting
``RuntimeError: can't start new thread`` was swallowed into a single-query
fallback -- so the answer it served depended on *how deep it got before the
box ran out*, which is a property of the machine, not of the query. Proving
that by actually exhausting the machine is not acceptable (one process reached
30,935 threads), so ``--abort-depth`` reproduces it exactly and safely: the
fan-out's executor is replaced with a serial one, and the same
``RuntimeError`` the OS would have raised is raised at the chosen depth. Run
the pre-fix tree at two different depths: different digests from the same
corpus and the same query is the non-determinism, demonstrated.

Nothing here reads a clock: ``--instant`` pins every recency term, so a digest
is reproducible on any day.

Usage::

    python3 benchmarks/expansion_reentrancy_identity.py --battery off-path
    python3 benchmarks/expansion_reentrancy_identity.py --battery auto-enabled
    python3 benchmarks/expansion_reentrancy_identity.py --battery auto-enabled --abort-depth 2

Output is JSON on stdout, one digest per query, so two runs diff directly.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import date
from typing import Any

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from benchmarks.ranking_identity import fingerprint_battery, fingerprint_digest  # noqa: E402
from mind_mem.hybrid_recall import HybridBackend  # noqa: E402
from mind_mem.init_workspace import init  # noqa: E402

#: Neither ``temporal`` nor ``multi-hop``: the auto-enable branches never fire,
#: so the guard is not on this path at all.
OFF_PATH_BATTERY = (
    "deterministic compiler evidence chain",
    "parser tiles dishwasher",
    "recall index block",
    "the team and the release",
)

#: The two query classes the branches auto-enable for.
AUTO_ENABLED_BATTERY = (
    "what happened before the migration in March",
    "when did I last update the config",
    "why did the team choose the parser and how does it compare to the previous one and what changed",
    "compare the retrieval backend and the storage backend and explain why they differ",
)

#: Deliberately dense and overlapping: every query in both batteries has to
#: match ten-odd blocks, not one. A corpus that returns a single hit per query
#: cannot reorder, so a fingerprint over it would report "identical" having
#: been unable to report anything else.
_CORPUS = [
    "the migration in March moved the parser and the team compared it to the previous release",
    "the migration in March was reverted in April after the parser regressed",
    "before the migration the team ran the old parser on every release candidate",
    "deterministic compiler evidence chain anchored on canonical bytes",
    "the evidence chain records the compiler decision that preceded the migration",
    "the recall index block layout changed when the compiler evidence chain landed",
    "the retrieval backend fuses BM25 and vector legs with reciprocal rank fusion",
    "the retrieval backend was compared to the storage backend before the team chose one",
    "the storage backend keeps blocks in SQLite with write-ahead logging",
    "the storage backend and the retrieval backend differ in how they handle a release",
    "the config was last updated when the release was cut in June",
    "the config update in June followed the parser migration by two weeks",
    "I last updated the config after the team compared the two backends",
    "the kitchen renovation needs new tiles and a dishwasher installed by June",
    "the dishwasher and the tiles arrived before the parser migration was scheduled",
    "the scheduler writes to the cache that the router reads on every request",
    "the router compares the cache entry to the previous one on every request",
    "the parser was chosen after the team compared it to two alternatives",
    "the parser choice changed what the compiler emits for a recall index block",
    "why the team chose the parser is recorded in the evidence chain",
    "what changed in March was the parser, the config and the recall index block",
    "the release was cut in June and the config was updated the same day",
    "the team compared the retrieval backend to the storage backend and explained why they differ",
    "a recall index block carries the statement, the tags and the compiler evidence",
]


class _SerialExecutor:
    """A thread-free stand-in for ``ThreadPoolExecutor``.

    Full drop-in (``map``/``submit``/``shutdown``) so swapping it changes only
    where the work runs. Used so the pre-fix recursion can be reproduced
    without creating a single OS thread.
    """

    def __init__(self, max_workers: int | None = None, *args: Any, **kwargs: Any) -> None:
        self.max_workers = max_workers

    def __enter__(self) -> "_SerialExecutor":
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False

    def map(self, fn: Any, *iterables: Any) -> list[Any]:
        return [fn(*args) for args in zip(*iterables)]

    def submit(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        from concurrent.futures import Future

        future: Future = Future()
        future.set_running_or_notify_cancel()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as exc:  # noqa: BLE001 - mirrors the pool contract
            future.set_exception(exc)
        return future

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        return None


def install_thread_free_fanout(abort_depth: int) -> dict[str, int]:
    """Serialize the fan-out and raise the OS's own error at *abort_depth*.

    Returns a live counter dict (``entered``/``max_depth``) so the caller can
    report how deep the run actually went -- a run that never recursed and a
    run that was stopped at depth 1 look the same in the digest alone.
    """
    import concurrent.futures

    import mind_mem.hybrid_recall as hr

    concurrent.futures.ThreadPoolExecutor = _SerialExecutor  # type: ignore[misc,assignment]
    hr.ThreadPoolExecutor = _SerialExecutor  # type: ignore[assignment]

    counter = {"entered": 0, "max_depth": 0, "_depth": 0}
    original = HybridBackend._search_expanded

    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        counter["entered"] += 1
        counter["_depth"] += 1
        counter["max_depth"] = max(counter["max_depth"], counter["_depth"])
        try:
            if abort_depth and counter["_depth"] > abort_depth:
                # Verbatim the message CPython raises when the OS refuses a
                # thread; the product's own except-clause then swallows it into
                # the single-query fallback, exactly as it did in production.
                raise RuntimeError("can't start new thread")
            return original(self, *args, **kwargs)
        finally:
            counter["_depth"] -= 1

    HybridBackend._search_expanded = wrapper  # type: ignore[method-assign]
    return counter


def build_workspace(root: str) -> str:
    ws = os.path.join(root, "ws")
    os.makedirs(ws, exist_ok=True)
    init(ws)
    decisions = os.path.join(ws, "decisions")
    os.makedirs(decisions, exist_ok=True)
    body = ["# DECISIONS\n\n---\n"]
    for i, statement in enumerate(_CORPUS, start=1):
        body.append(
            f"\n[D-20260101-{i:06d}]\nDate: 2026-06-20\nStatus: active\nScope: global\nStatement: {statement}\nTags: compiler, recall\n"
        )
    with open(os.path.join(decisions, "DECISIONS.md"), "w", encoding="utf-8") as handle:
        handle.write("".join(body))
    return ws


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--battery", choices=("off-path", "auto-enabled"), default="off-path")
    parser.add_argument(
        "--abort-depth",
        type=int,
        default=0,
        help="serialize the fan-out and raise the OS's thread error below this depth (0: leave the fan-out alone)",
    )
    parser.add_argument("--instant", default="2026-09-03", help="UTC date every recency term scores against")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args(argv)

    counter: dict[str, int] | None = None
    if args.abort_depth:
        counter = install_thread_free_fanout(args.abort_depth)

    instant = date.fromisoformat(args.instant)
    battery = OFF_PATH_BATTERY if args.battery == "off-path" else AUTO_ENABLED_BATTERY

    with tempfile.TemporaryDirectory() as root:
        workspace = build_workspace(root)
        backend = HybridBackend(
            config={
                "vector_enabled": False,
                "knee_cutoff": False,
                # The cross-encoder auto-enables on the same query types and is
                # a different decision; holding it off keeps the digest about
                # the fan-out.
                "cross_encoder": {"auto_enable": False},
            }
        )

        def recall(query: str) -> list[dict]:
            return list(backend.search(query, workspace, limit=args.limit, scoring_instant=instant))

        prints = fingerprint_battery(recall, battery)

    report = {
        "battery": args.battery,
        "abort_depth": args.abort_depth,
        "instant": args.instant,
        "queries": {q: {"n": len(fp), "digest": fingerprint_digest(fp)} for q, fp in prints.items()},
    }
    if counter is not None:
        report["fanout"] = {"entered": counter["entered"], "max_depth": counter["max_depth"]}
    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
