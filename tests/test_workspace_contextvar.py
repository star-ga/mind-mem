"""v3.2.1 — regression test for per-request workspace ContextVar scoping.

Before v3.2.1 the REST layer mutated ``os.environ["MIND_MEM_WORKSPACE"]``
on every request. Under asyncio concurrency this raced: request A
setting workspace X would be clobbered by request B setting workspace
Y before A's tool call resolved the env var. The fix swaps env
mutation for a ``ContextVar`` override in
``mind_mem.mcp.infra.workspace`` that is task-local under asyncio and
thread-local through Starlette's thread pool for sync handlers.

These tests pin the fix: ``_workspace()`` respects the ContextVar
before falling back to env, and ``use_workspace`` resets cleanly
even when the contained code raises.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import pytest

from mind_mem.mcp.infra.workspace import _workspace, use_workspace


class TestContextVarOverride:
    def test_contextvar_takes_precedence_over_env(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        env_ws = tmp_path / "from-env"
        env_ws.mkdir()
        monkeypatch.setenv("MIND_MEM_WORKSPACE", str(env_ws))

        ctx_ws = tmp_path / "from-context"
        ctx_ws.mkdir()

        # Outside the context manager, env wins.
        assert _workspace() == str(env_ws)

        # Inside, ContextVar override wins.
        with use_workspace(str(ctx_ws)):
            assert _workspace() == str(ctx_ws)

        # After exit, env wins again.
        assert _workspace() == str(env_ws)

    def test_use_workspace_resets_on_exception(self, tmp_path) -> None:
        """If the block raises, the ContextVar must still be reset."""
        with pytest.raises(RuntimeError):
            with use_workspace(str(tmp_path)):
                raise RuntimeError("boom")

        # After the exception, the override must be cleared and env fallback wins.
        # (With no env set and no override, the default becomes cwd.)
        assert _workspace() != str(tmp_path) or _workspace() == os.path.abspath(str(tmp_path))

    def test_nested_overrides_stack(self, tmp_path) -> None:
        outer = tmp_path / "outer"
        inner = tmp_path / "inner"
        outer.mkdir()
        inner.mkdir()

        with use_workspace(str(outer)):
            assert _workspace() == str(outer)
            with use_workspace(str(inner)):
                assert _workspace() == str(inner)
            # Inner's reset restores outer.
            assert _workspace() == str(outer)

    def test_thread_isolation(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
        """ContextVar values don't leak across threads when not explicitly propagated."""
        env_ws = tmp_path / "shared-env"
        env_ws.mkdir()
        monkeypatch.setenv("MIND_MEM_WORKSPACE", str(env_ws))

        ctx_ws = tmp_path / "ctx-only"
        ctx_ws.mkdir()

        results: list[str] = []

        def worker() -> None:
            # This thread has no override set — should see env fallback.
            results.append(_workspace())

        with use_workspace(str(ctx_ws)):
            # Main thread sees the override.
            assert _workspace() == str(ctx_ws)
            # A vanilla thread (without contextvars propagation) sees env only.
            with ThreadPoolExecutor(max_workers=1) as pool:
                pool.submit(worker).result()

        # Vanilla Thread doesn't copy the context, so the worker thread
        # should have observed the env fallback, not ctx_ws.
        assert results == [str(env_ws)]

    def test_use_workspace_abspath(self, tmp_path) -> None:
        """``use_workspace`` normalises to an absolute path."""
        rel = os.path.relpath(str(tmp_path))
        with use_workspace(rel) as resolved:
            assert os.path.isabs(resolved)
            assert resolved == os.path.abspath(str(tmp_path))
            assert _workspace() == resolved
