"""The request context must be READ by the engine, and by its thread workers.

WHY THESE EXIST AS A SEPARATE FILE. The binding's whole value is that the ranking consumed it.
A context that is bound and never read is indistinguishable from no context at all, and the first
version of the thread wiring here had exactly that defect: it submitted a helper that read the
active context when the worker INVOKED it, inside a fresh worker context, so it bound nothing and
reported success. Nothing in the door-level tests could see the difference — the config on disk and
the config in the context are the same value in the ordinary case, so an unbound worker returns the
right answer for the wrong reason. These tests make the two distinguishable by putting a value in
the context that is NOT on disk.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from mind_mem.request_context import (
    RequestContext,
    active_request_context,
    bind_current,
    bind_request_context,
    context_config_for,
    context_was_consumed_for,
)


def _ws(tmp_path: Path) -> str:
    (tmp_path / "mind-mem.json").write_text(json.dumps({"recall": {"from_disk": True}}), encoding="utf-8", newline="\n")
    return str(tmp_path)


def test_a_bound_read_returns_the_context_not_the_file(tmp_path: Path) -> None:
    """The discriminating case: a context value that does not exist on disk."""
    workspace = _ws(tmp_path)
    context = RequestContext(workspace=workspace, config={"recall": {"from_context": True}})
    with bind_request_context(context):
        seen = context_config_for(workspace)
    assert seen is not None
    assert seen["recall"] == {"from_context": True}, "the read was served from disk, not the context"


def test_an_unbound_read_is_none_and_records_nothing(tmp_path: Path) -> None:
    """No context bound means no claim — and no receipt entry to claim with."""
    workspace = _ws(tmp_path)
    assert active_request_context() is None
    assert context_config_for(workspace) is None
    assert context_was_consumed_for(workspace) is False


def test_a_context_bound_for_another_workspace_does_not_answer(tmp_path: Path) -> None:
    """Workspace is part of the key, or one workspace's policy describes another's ranking."""
    a = _ws(tmp_path / "a") if (tmp_path / "a").mkdir() or True else ""
    b = _ws(tmp_path / "b") if (tmp_path / "b").mkdir() or True else ""
    context = RequestContext(workspace=a, config={"recall": {"which": "a"}})
    with bind_request_context(context):
        assert context_config_for(a) is not None
        assert context_config_for(b) is None, "a context bound for A answered a read for B"
        assert context_was_consumed_for(b) is False


def test_bound_but_unread_is_not_consumed(tmp_path: Path) -> None:
    """A bound context nobody read proves nothing, and must not report that it does."""
    workspace = _ws(tmp_path)
    context = RequestContext(workspace=workspace, config={"recall": {}})
    with bind_request_context(context):
        assert context.reads == 0
        assert context_was_consumed_for(workspace) is False
        context_config_for(workspace)
        assert context.reads == 1
        assert context_was_consumed_for(workspace) is True


def test_a_thread_worker_reads_the_context_only_when_bound_at_submit_time(tmp_path: Path) -> None:
    """The defect this file exists for: capture must happen in the SUBMITTING thread.

    The negative half is the point. Submitting the worker bare is the natural thing to write and it
    silently binds nothing — so this asserts BOTH that the bare form sees no context and that the
    wrapped form does, which is the only pair that can tell a working wrapper from a no-op one.
    """
    workspace = _ws(tmp_path)
    context = RequestContext(workspace=workspace, config={"recall": {"from_context": True}})

    def worker() -> object:
        cfg = context_config_for(workspace)
        return None if cfg is None else dict(cfg)

    with bind_request_context(context):
        with ThreadPoolExecutor(max_workers=2) as pool:
            bare = pool.submit(worker).result()
            wrapped = pool.submit(
                bind_current(worker),
            ).result()

    assert bare is None, (
        "a bare submit inherited a context it should not have — then this test cannot tell a working wrapper from a no-op one"
    )
    assert wrapped == {"recall": {"from_context": True}}, "the wrapped worker did not read the submitting thread's context"
    # And the parent's receipt saw the worker's read, which is what lets the row claim the fan-out.
    assert context.reads >= 1, "the worker's read was not recorded against the parent's context"


def test_bind_current_is_a_passthrough_when_nothing_is_bound() -> None:
    """An unbound path must pay no wrapper — and must not be reported as bound."""

    def worker() -> int:
        return 7

    assert bind_current(worker) is worker


def test_the_context_config_cannot_be_rebound_through_the_value_handed_out(tmp_path: Path) -> None:
    """Top-level sections are read-only, so the engine cannot edit the policy it was given."""
    workspace = _ws(tmp_path)
    context = RequestContext(workspace=workspace, config={"recall": {"a": 1}})
    with bind_request_context(context):
        cfg = context_config_for(workspace)
    assert cfg is not None
    try:
        cfg["recall"] = {"a": 2}  # type: ignore[index]
    except TypeError:
        pass
    else:  # pragma: no cover — a writable mapping means the immutability claim is false
        raise AssertionError("the bound config accepted a top-level rebind")
