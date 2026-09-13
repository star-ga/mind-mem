"""The context must not outlive its request, and must not leak between them.

WHY THIS IS A SEPARATE CONCERN FROM BINDING. The binding tests prove the engine reads the context
that was bound. These prove the context STOPS. A ContextVar left set after a request would make the
next request rank under the previous one's policy — a bug strictly worse than the one the binding
fixes, because it would be invisible: both requests return plausible answers, and the recorded rows
would each name their own captured hash while one of them ranked under the other's config.

The exception path is the one that actually breaks in practice. ``bind_request_context`` resets its
token in a ``finally``, so this is a regression guard rather than a discovery — but a guard that only
covers the happy path would let a later refactor turn the reset into an early ``return``.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from mind_mem.request_context import (
    RequestContext,
    active_request_context,
    bind_current,
    bind_request_context,
    context_config_for,
)


def _ws(root: Path) -> str:
    root.mkdir(parents=True, exist_ok=True)
    (root / "mind-mem.json").write_text(json.dumps({"recall": {}}), encoding="utf-8", newline="\n")
    return str(root)


def test_sequential_requests_each_see_their_own_context(tmp_path: Path) -> None:
    """Two requests in one thread must not share policy."""
    workspace = _ws(tmp_path / "seq")
    seen = []
    for tag in ("first", "second"):
        with bind_request_context(RequestContext(workspace=workspace, config={"recall": {"tag": tag}})):
            cfg = context_config_for(workspace)
            assert cfg is not None
            seen.append(cfg["recall"]["tag"])
        assert active_request_context() is None, f"the {tag} context outlived its request"
    assert seen == ["first", "second"], seen


def test_the_context_is_cleared_after_an_exception(tmp_path: Path) -> None:
    """An exception inside the request must not leave the next one bound to it."""
    workspace = _ws(tmp_path / "boom")
    with pytest.raises(RuntimeError, match="leg failed"):
        with bind_request_context(RequestContext(workspace=workspace, config={"recall": {}})):
            assert active_request_context() is not None
            raise RuntimeError("leg failed")
    assert active_request_context() is None, "a failed request left its context bound — the next recall would rank under it"
    assert context_config_for(workspace) is None


def test_nesting_restores_the_outer_context(tmp_path: Path) -> None:
    """An inner bind must not destroy the outer one — legs bind inside a door's bind."""
    workspace = _ws(tmp_path / "nest")
    outer = RequestContext(workspace=workspace, config={"recall": {"which": "outer"}})
    inner = RequestContext(workspace=workspace, config={"recall": {"which": "inner"}})
    with bind_request_context(outer):
        with bind_request_context(inner):
            cfg = context_config_for(workspace)
            assert cfg is not None and cfg["recall"]["which"] == "inner"
        cfg = context_config_for(workspace)
        assert cfg is not None and cfg["recall"]["which"] == "outer", "the inner bind clobbered the outer context instead of restoring it"
    assert active_request_context() is None


def test_a_worker_binding_does_not_escape_into_the_pool(tmp_path: Path) -> None:
    """A pooled thread is reused — a worker's bind must not survive into the next task.

    This is the one that a naive implementation gets wrong: setting the ContextVar in the worker
    without resetting it leaves the POOL THREAD bound, so an unrelated later task submitted bare
    would silently read the earlier request's policy.
    """
    workspace = _ws(tmp_path / "pool")
    context = RequestContext(workspace=workspace, config={"recall": {"from_context": True}})

    def probe() -> bool:
        return active_request_context() is not None

    with ThreadPoolExecutor(max_workers=1) as pool:  # ONE worker, so the thread is definitely reused
        with bind_request_context(context):
            assert pool.submit(bind_current(probe)).result() is True
        leaked = pool.submit(probe).result()

    assert leaked is False, (
        "the pooled thread stayed bound after the wrapped task finished, so a later bare task would read a previous request's policy"
    )
