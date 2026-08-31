# Copyright 2026 STARGA, Inc.
"""Contract tests for the zero-count branch of ``probe_block_count``.

``probe_block_count`` used to guard its early return with
``if blocks > 0 or status.get("schema_built", True) is not False:``, whose
body was only ``if blocks > 0: return`` — the ``schema_built`` disjunct
could not change any outcome and was read only to be discarded, while the
comment above it described a trust distinction ("trust the index count when
the FTS schema is built") that the code never implemented.

These tests pin what the code actually does, so the next edit written
against that comment has to argue with a test rather than with a comment:
a zero index count is never trusted, *whatever* ``schema_built`` says, and
a populated workspace is never reported empty because its index is stale.
There is no behavioural change here — these pass before and after the dead
disjunct was removed; their job is to stop it coming back as a live one.
"""

from __future__ import annotations

import os

import pytest

from mind_mem._recall_workspace import probe_block_count
from mind_mem.init_workspace import init


def _write_decision_block(ws: str) -> None:
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


@pytest.fixture
def populated_workspace(tmp_path) -> str:
    ws = str(tmp_path / "ws")
    os.makedirs(ws, exist_ok=True)
    init(ws)
    _write_decision_block(ws)
    return ws


def _index_status(blocks: int, schema_built: bool) -> dict:
    return {"blocks": blocks, "schema_built": schema_built}


@pytest.mark.parametrize("schema_built", [True, False])
def test_zero_index_count_always_falls_through_to_the_corpus(populated_workspace: str, monkeypatch, schema_built: bool) -> None:
    """A stale index reporting 0 must not label a populated workspace empty."""
    import mind_mem.sqlite_index as sqlite_index

    monkeypatch.setattr(sqlite_index, "index_status", lambda ws: _index_status(0, schema_built))

    health = probe_block_count(populated_workspace)

    assert health.blocks >= 1
    assert health.is_empty_or_unbuilt is False


@pytest.mark.parametrize("schema_built", [True, False])
def test_positive_index_count_is_trusted_without_a_corpus_walk(populated_workspace: str, monkeypatch, schema_built: bool) -> None:
    import mind_mem.sqlite_index as sqlite_index
    import mind_mem.storage as storage

    monkeypatch.setattr(sqlite_index, "index_status", lambda ws: _index_status(7, schema_built))

    def _must_not_be_called(ws):  # pragma: no cover - the assertion is the point
        raise AssertionError("corpus walk must be skipped when the index reports rows")

    monkeypatch.setattr(storage, "iter_active_blocks", _must_not_be_called)

    assert probe_block_count(populated_workspace).blocks == 7


def test_probe_error_still_degrades_rather_than_raising(populated_workspace: str, monkeypatch) -> None:
    import mind_mem.sqlite_index as sqlite_index

    def _boom(ws):
        raise RuntimeError("index unreachable")

    monkeypatch.setattr(sqlite_index, "index_status", _boom)

    health = probe_block_count(populated_workspace)

    assert health.blocks == -1
    assert health.probe_error is not None
    assert health.is_empty_or_unbuilt is False
