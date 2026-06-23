# Copyright 2026 STARGA, Inc.
"""CLI workspace-resolution + silent-empty-store tests for recall.

Covers the two bugs fixed in ``_recall_workspace`` + ``_recall_core.main`` /
``recall_vector.main``:

1. The recall CLI never resolved ``MIND_MEM_WORKSPACE`` or discovered the
   nearest ``mind-mem.json`` upward — it hardcoded ``--workspace`` default
   ``"."``, so a recall from a deep subdir (or with the env var set but a
   different cwd) silently returned ``[]``.
2. ``main()`` printed one fixed ``No results found.`` whether the store was
   empty/unbuilt/missing or the query genuinely matched 0-of-N — the operator
   could not tell the two apart.

The library ``recall()`` contract (returns ``[]`` on an empty workspace) is a
hard regression guard here — the fix must not touch it.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

from mind_mem._recall_core import recall
from mind_mem._recall_workspace import (
    WorkspaceHealth,
    empty_workspace_warning,
    probe_block_count,
    resolve_workspace,
)


def _write_decision_block(ws: str) -> None:
    """Write one active decision block into ``ws/decisions/DECISIONS.md``."""
    dec_dir = os.path.join(ws, "decisions")
    os.makedirs(dec_dir, exist_ok=True)
    with open(os.path.join(dec_dir, "DECISIONS.md"), "w", encoding="utf-8") as f:
        f.write(
            "# DECISIONS\n\n---\n\n"
            "[D-20260620-001]\n"
            "Date: 2026-06-20\n"
            "Status: active\n"
            "Scope: global\n"
            "Statement: We will use PostgreSQL pgvector for hybrid retrieval indexing.\n"
            "Tags: database, retrieval, pgvector\n"
        )


def _run_main(monkeypatch, argv: list[str]) -> None:
    """Invoke ``_recall_core.main`` with a synthetic argv."""
    from mind_mem import _recall_core

    monkeypatch.setattr(sys, "argv", ["recall", *argv])
    _recall_core.main()


# ---------------------------------------------------------------------------
# resolve_workspace — pure unit tests (resolution order)
# ---------------------------------------------------------------------------


def test_resolve_explicit_arg_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("MIND_MEM_WORKSPACE", str(tmp_path / "env_ws"))
    explicit = str(tmp_path / "explicit_ws")
    assert resolve_workspace(explicit) == os.path.abspath(explicit)


def test_resolve_explicit_dot_distinguishable_from_unset(monkeypatch, tmp_path):
    # Passing "." must resolve to cwd-as-abspath, NOT fall through to the env
    # var — argparse default is None so "." is a real explicit value.
    monkeypatch.setenv("MIND_MEM_WORKSPACE", str(tmp_path / "env_ws"))
    monkeypatch.chdir(tmp_path)
    assert resolve_workspace(".") == os.path.abspath(str(tmp_path))


def test_resolve_env_var_when_no_arg(monkeypatch, tmp_path):
    env_ws = str(tmp_path / "env_ws")
    os.makedirs(env_ws)
    monkeypatch.setenv("MIND_MEM_WORKSPACE", env_ws)
    assert resolve_workspace(None) == os.path.abspath(env_ws)


def test_resolve_upward_discovery_from_subdir(monkeypatch, tmp_path):
    # The genuinely-new rung: walk up to the nearest mind-mem.json.
    ws = tmp_path / "ws"
    deep = ws / "a" / "b" / "c"
    deep.mkdir(parents=True)
    (ws / "mind-mem.json").write_text("{}", encoding="utf-8")
    monkeypatch.delenv("MIND_MEM_WORKSPACE", raising=False)
    monkeypatch.chdir(deep)
    assert resolve_workspace(None) == os.path.abspath(str(ws))


def test_resolve_falls_back_to_cwd(monkeypatch, tmp_path):
    nowhere = tmp_path / "nowhere"
    nowhere.mkdir()
    monkeypatch.delenv("MIND_MEM_WORKSPACE", raising=False)
    monkeypatch.chdir(nowhere)
    # No mind-mem.json anywhere up the tmp tree root => cwd.
    resolved = resolve_workspace(None)
    assert resolved == os.path.abspath(str(nowhere)) or os.path.isfile(os.path.join(resolved, "mind-mem.json"))


# ---------------------------------------------------------------------------
# Bug 1: main() honors MIND_MEM_WORKSPACE + upward discovery
# ---------------------------------------------------------------------------


def test_main_honors_env_workspace(workspace, monkeypatch, capsys):
    _write_decision_block(workspace)
    monkeypatch.setenv("MIND_MEM_WORKSPACE", workspace)
    monkeypatch.chdir(os.path.dirname(workspace))  # cwd is NOT the workspace

    _run_main(monkeypatch, ["--query", "PostgreSQL pgvector retrieval", "--backend", "scan", "--json"])
    out = capsys.readouterr().out
    results = json.loads(out)
    assert any(r["_id"] == "D-20260620-001" for r in results), out


def test_main_upward_discovery_from_subdir(workspace, monkeypatch, capsys):
    _write_decision_block(workspace)
    subdir = os.path.join(workspace, "decisions")  # a real subdir of the ws
    monkeypatch.delenv("MIND_MEM_WORKSPACE", raising=False)
    monkeypatch.chdir(subdir)

    _run_main(monkeypatch, ["--query", "PostgreSQL pgvector retrieval", "--backend", "scan", "--json"])
    out = capsys.readouterr().out
    results = json.loads(out)
    assert any(r["_id"] == "D-20260620-001" for r in results), out


# ---------------------------------------------------------------------------
# Bug 2: empty store -> LOUD stderr warning; genuine miss -> quiet stdout
# ---------------------------------------------------------------------------


def test_empty_store_warns_loudly_with_resolved_path(workspace, monkeypatch, capsys):
    # `workspace` fixture is initialized but has 0 blocks.
    monkeypatch.delenv("MIND_MEM_WORKSPACE", raising=False)
    _run_main(monkeypatch, ["--query", "anything at all", "--workspace", workspace, "--backend", "scan"])
    captured = capsys.readouterr()
    # Quiet stdout line unchanged (scripts parsing stdout don't break).
    assert "No results found." in captured.out
    # Loud, actionable stderr warning naming the resolved path + fix hint.
    assert "WARNING" in captured.err
    assert os.path.abspath(workspace) in captured.err
    assert "mind-mem-init" in captured.err
    assert "MIND_MEM_WORKSPACE" in captured.err


def test_unconfigured_workspace_warns(tmp_path, monkeypatch, capsys):
    ws = str(tmp_path / "no_config")  # no mind-mem.json at all
    os.makedirs(ws)
    monkeypatch.delenv("MIND_MEM_WORKSPACE", raising=False)
    _run_main(monkeypatch, ["--query", "anything", "--workspace", ws, "--backend", "scan"])
    captured = capsys.readouterr()
    assert "No results found." in captured.out
    assert "WARNING" in captured.err
    assert "mind-mem.json" in captured.err


def test_populated_store_genuine_miss_is_quiet(workspace, monkeypatch, capsys):
    _write_decision_block(workspace)  # 1 block present
    monkeypatch.delenv("MIND_MEM_WORKSPACE", raising=False)
    # Query that matches the one block on nothing.
    _run_main(
        monkeypatch,
        ["--query", "zzz quantum teleportation banana xyzzy", "--workspace", workspace, "--backend", "scan"],
    )
    captured = capsys.readouterr()
    assert "No results found." in captured.out
    # No loud warning — this is a genuine 0-of-N over a non-empty store.
    assert "WARNING" not in captured.err


# ---------------------------------------------------------------------------
# probe_block_count — backend-aware emptiness probe
# ---------------------------------------------------------------------------


def test_probe_reports_empty_for_initialized_blank_workspace(workspace):
    health = probe_block_count(workspace)
    assert health.configured is True
    assert health.blocks == 0
    assert health.is_empty_or_unbuilt is True
    assert health.probe_error is None


def test_probe_reports_populated_after_block_written(workspace):
    _write_decision_block(workspace)
    health = probe_block_count(workspace)
    assert health.configured is True
    assert health.blocks >= 1
    assert health.is_empty_or_unbuilt is False


def test_probe_unconfigured_is_empty(tmp_path):
    ws = str(tmp_path / "blank")
    os.makedirs(ws)
    health = probe_block_count(ws)
    assert health.configured is False
    assert health.is_empty_or_unbuilt is True


def test_probe_failure_is_not_treated_as_empty():
    # A degraded probe (blocks == -1) must NOT be flagged empty — otherwise a
    # flaky DB becomes a false "empty store" warning.
    health = WorkspaceHealth("/x", "postgres", configured=True, blocks=-1, probe_error="boom")
    assert health.is_empty_or_unbuilt is False


def test_warning_message_names_path_and_backend():
    health = WorkspaceHealth("/tmp/ws", "markdown", configured=True, blocks=0)
    msg = empty_workspace_warning(health)
    assert "/tmp/ws" in msg
    assert "markdown" in msg
    assert "mind-mem-init" in msg


# ---------------------------------------------------------------------------
# Regression guard: library recall() still returns [] on an empty workspace.
# ---------------------------------------------------------------------------


def test_library_recall_still_returns_empty_list(workspace):
    results = recall(workspace, "nothing here", limit=5)
    assert results == []


@pytest.mark.parametrize("query", ["test query", "anything", "memory"])
def test_library_recall_contract_unchanged(workspace, query):
    # Mirrors the existing tests/test_recall_empty_workspace.py contract.
    results = recall(workspace, query, limit=10)
    assert isinstance(results, list)
    assert results == []
