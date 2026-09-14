"""Namespace ACLs must guard indexed recall before any result processing."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import pytest

import mind_mem._recall_core as recall_core
import mind_mem.sqlite_index as sqlite_index
from mind_mem._recall_core import PostgresRecallBackend, RecallBackend, recall
from mind_mem.hybrid_recall import RecallResults
from mind_mem.init_workspace import init


def _block(path: Path, block_id: str, statement: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"[{block_id}]\nType: Decision\nStatement: {statement}\nStatus: active\n\n",
        encoding="utf-8",
    )


def _hit(block_id: str, source: str | None, *, score: float = 10.0, source_file: Any = None) -> dict[str, Any]:
    hit: dict[str, Any] = {
        "_id": block_id,
        "type": "Decision",
        "score": score,
        "excerpt": block_id,
        "speaker": "",
        "tags": "",
        "line": 1,
        "status": "active",
    }
    if source is not None:
        hit["file"] = source
    if source_file is not None:
        hit["_source_file"] = source_file
    return hit


def _workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    init(str(workspace))
    config = json.loads((workspace / "mind-mem.json").read_text(encoding="utf-8"))
    config.setdefault("recall", {})["backend"] = "sqlite"
    (workspace / "mind-mem.json").write_text(json.dumps(config), encoding="utf-8")
    (workspace / "mind-mem-acl.json").write_text(
        json.dumps(
            {
                "default_policy": "read",
                "agents": {
                    "alice": {
                        "namespaces": ["shared", "agents/alice"],
                        "read": ["shared", "agents/alice"],
                        "write": [],
                    },
                    "bob": {
                        "namespaces": ["shared", "agents/bob"],
                        "read": ["shared", "agents/bob"],
                        "write": [],
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return workspace


def test_real_sqlite_index_is_filtered_before_return(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    relative = "agents/bob/decisions/DECISIONS.md"
    _block(workspace / relative, "D-BOB-1", "private bob indexed fact")

    manager = sqlite_index._get_conn_manager(str(workspace))
    with manager.write_lock:
        connection = manager.get_write_connection()
        connection.row_factory = sqlite_index.sqlite3.Row
        sqlite_index._init_schema(connection)
        sqlite_index._index_file(
            connection,
            str(workspace),
            "bob-private",
            relative,
            {"D-BOB-1"},
            force=True,
        )
        connection.commit()

    denied = recall(str(workspace), "private bob indexed", agent_id="alice", rerank=False)
    allowed = recall(str(workspace), "private bob indexed", agent_id="bob", rerank=False)
    assert denied == []
    assert allowed and allowed[0]["_id"] == "D-BOB-1"
    assert math.isfinite(float(allowed[0]["score"]))


def test_indexed_acl_filters_private_and_malformed_sources_before_processing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    raw = [
        _hit("SHARED-1", "shared/decisions/DECISIONS.md", score=9.0),
        _hit("ALICE-1", "agents/alice/decisions/DECISIONS.md", score=8.0),
        _hit("BOB-1", "agents/bob/decisions/DECISIONS.md", score=7.0),
        _hit("ROOT-1", "decisions/DECISIONS.md", score=6.5),
        _hit("NO-SOURCE", None, score=6.0),
        _hit("ABSOLUTE", "/tmp/private.md", score=5.0),
        _hit("TRAVERSAL", "agents/alice/../bob/decisions/DECISIONS.md", score=4.0),
        _hit(
            "CONFLICTING-SOURCES",
            "agents/bob/decisions/DECISIONS.md",
            source_file="agents/alice/decisions/DECISIONS.md",
            score=3.0,
        ),
    ]
    monkeypatch.setattr(sqlite_index, "query_index", lambda *args, **kwargs: list(raw))

    alice = recall(str(workspace), "indexed", agent_id="alice", rerank=False)
    bob = recall(str(workspace), "indexed", agent_id="bob", rerank=False)
    unknown = recall(str(workspace), "indexed", agent_id="unlisted", rerank=False)
    legacy = recall(str(workspace), "indexed", rerank=False)

    assert [hit["_id"] for hit in alice] == ["SHARED-1", "ALICE-1"]
    assert [hit["_id"] for hit in bob] == ["SHARED-1", "BOB-1"]
    assert [hit["_id"] for hit in unknown] == ["SHARED-1"]
    # Existing namespace-properties filtering still owns malformed source
    # claims on the legacy path; the new ACL filter adds no agent-less gate.
    assert [hit["_id"] for hit in legacy] == [
        "SHARED-1",
        "ALICE-1",
        "BOB-1",
        "ROOT-1",
        "CONFLICTING-SOURCES",
    ]
    assert [hit["score"] for hit in legacy] == [9.0, 8.0, 7.0, 6.5, 3.0]


def test_indexed_acl_rejects_shared_symlink_into_private_namespace(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    shared_decisions = workspace / "shared/decisions"
    private_decisions = workspace / "agents/bob/decisions"
    private_decisions.mkdir(parents=True, exist_ok=True)
    shared_decisions.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(private_decisions, shared_decisions, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"directory symlink unavailable: {exc}")

    monkeypatch.setattr(
        sqlite_index,
        "query_index",
        lambda *args, **kwargs: [_hit("BOB-VIA-SHARED-LINK", "shared/decisions/DECISIONS.md")],
    )
    assert recall(str(workspace), "indexed", agent_id="alice", rerank=False) == []
    assert recall(str(workspace), "indexed", agent_id="unlisted", rerank=False) == []


class _DeclaredBackend(RecallBackend):
    def __init__(self, hits: list[dict[str, Any]]) -> None:
        self.hits = hits

    def search(self, workspace: str, query: str, limit: int = 10, active_only: bool = False) -> list[dict[str, Any]]:
        return list(self.hits)

    def index(self, workspace: str) -> None:
        return None


def test_declared_backend_source_less_hits_fail_closed_for_agent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    backend = _DeclaredBackend([_hit("STORE-UNKNOWN", None)])
    monkeypatch.setattr(recall_core, "_load_backend", lambda _workspace: backend)

    assert recall(str(workspace), "remote", agent_id="alice", rerank=False) == []


def test_declared_backend_preserves_unbound_order_and_scores(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    raw = [_hit("SHARED-1", "shared/decisions/DECISIONS.md", score=0.25), _hit("SHARED-2", "shared/tasks/TASKS.md", score=0.125)]
    monkeypatch.setattr(recall_core, "_load_backend", lambda _workspace: _DeclaredBackend(raw))

    result = recall(str(workspace), "remote", rerank=False)
    assert [hit["_id"] for hit in result] == ["SHARED-1", "SHARED-2"]
    assert [hit["score"] for hit in result] == [0.25, 0.125]


def test_indexed_acl_preserves_backend_trace_and_degraded_marker(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    result = RecallResults([_hit("SHARED-TRACE", "shared/decisions/DECISIONS.md")])
    result.trace = {"backend": "fixture"}
    result.degraded = {"leg": "vector", "reason": "fixture"}

    class Backend(_DeclaredBackend):
        def search(self, *args, **kwargs):
            return result

    monkeypatch.setattr(recall_core, "_load_backend", lambda _workspace: Backend([]))
    returned = recall(str(workspace), "trace query", agent_id="alice", rerank=False)
    assert returned and returned[0]["_id"] == "SHARED-TRACE"
    assert getattr(returned, "trace", None) == {"backend": "fixture"}
    assert getattr(returned, "degraded", None) == {"leg": "vector", "reason": "fixture"}


def test_indexed_acl_normalizes_source_separators_but_rejects_dot_segments(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    monkeypatch.setattr(
        sqlite_index,
        "query_index",
        lambda *args, **kwargs: [
            _hit("SLASH-EQUIV", "shared/decisions/DECISIONS.md", source_file="shared\\decisions\\DECISIONS.md"),
            _hit("DOT-SEGMENT", "shared/./decisions/DECISIONS.md"),
            _hit("EMPTY-SEGMENT", "shared//decisions/DECISIONS.md"),
        ],
    )
    returned = recall(str(workspace), "source query", agent_id="alice", rerank=False)
    assert [hit["_id"] for hit in returned] == ["SLASH-EQUIV"]


def test_postgres_indexed_acl_does_not_consult_local_shadow_realpath(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    shadow = workspace / "agents/bob/decisions"
    shadow.mkdir(parents=True, exist_ok=True)
    shared = workspace / "shared/virtual"
    shared.parent.mkdir(parents=True, exist_ok=True)
    try:
        shared.symlink_to(shadow, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlink unavailable: {exc}")

    result = [_hit("PG-SHARED", "shared/virtual/DB-ROW.md")]
    monkeypatch.setattr(PostgresRecallBackend, "search", lambda self, *args, **kwargs: list(result))
    monkeypatch.setattr(recall_core, "_load_backend", lambda _workspace: PostgresRecallBackend(str(workspace)))
    returned = recall(str(workspace), "postgres query", agent_id="alice", rerank=False)
    assert [hit["_id"] for hit in returned] == ["PG-SHARED"]
