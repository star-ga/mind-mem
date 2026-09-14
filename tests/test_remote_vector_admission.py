"""Remote vector hits must be checked against the current corpus authority."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from mind_mem._recall_core import recall
from mind_mem.block_parser import parse_file
from mind_mem.recall_vector import source_content_digest


def _workspace(
    tmp_path: Path, *, status: str = "active", credential: bool = False, provider: str = "qdrant"
) -> tuple[Path, dict]:
    decisions = tmp_path / "decisions"
    decisions.mkdir()
    category = "\nContentCategory: credential" if credential else ""
    source = decisions / "DECISIONS.md"
    source.write_text(
        f"[D-REMOTE-1]\nType: Decision\nStatement: canonical source\nStatus: {status}{category}\n\n",
        encoding="utf-8",
    )
    config = {"recall": {"backend": "vector", "provider": provider}}
    if credential:
        config["recall"]["validity_gate"] = {
            "enabled": True,
            "content_categories": {"enabled": True, "ttl_days": {"infra": 30, "status": 30}},
        }
    (tmp_path / "mind-mem.json").write_text(json.dumps(config), encoding="utf-8")
    row = parse_file(str(source))[0]
    return tmp_path, {
        "_id": row["_id"],
        "type": "decision",
        "score": 0.99,
        "excerpt": "canonical source",
        "file": "decisions/DECISIONS.md",
        "line": 1,
        "status": "active",
        "source_digest": source_content_digest(row),
    }


def _install_qdrant(monkeypatch, payload: dict) -> None:
    class Hit:
        score = 0.99

        def __init__(self, row):
            self.payload = row

    class Client:
        def __init__(self, *, url):
            self.url = url

        def search(self, **kwargs):
            return [Hit(payload)]

    qdrant = types.ModuleType("qdrant_client")
    qdrant.QdrantClient = Client
    models = types.ModuleType("qdrant_client.models")
    models.FieldCondition = object
    models.Filter = object
    models.MatchValue = object
    monkeypatch.setitem(sys.modules, "qdrant_client", qdrant)
    monkeypatch.setitem(sys.modules, "qdrant_client.models", models)


def _install_pinecone(monkeypatch, payload: dict) -> None:
    class Index:
        def search_records(self, *, namespace, query):
            return {
                "result": {
                    "hits": [
                        {
                            "_id": payload.get("_id"),
                            "_score": payload.get("score", 0.99),
                            "fields": {
                                "block_type": payload.get("type"),
                                "excerpt": payload.get("excerpt"),
                                "file": payload.get("file"),
                                "line": payload.get("line"),
                                "status": payload.get("status"),
                                "source_digest": payload.get("source_digest"),
                            },
                        }
                    ]
                }
            }

    class Client:
        def __init__(self, *, api_key):
            self.api_key = api_key

        def Index(self, _name):
            return Index()

    pinecone = types.ModuleType("pinecone")
    pinecone.Pinecone = Client
    monkeypatch.setitem(sys.modules, "pinecone", pinecone)
    monkeypatch.setenv("PINECONE_API_KEY", "fixture-key")


def _run(workspace: Path, monkeypatch, payload: dict, provider: str = "qdrant") -> list[dict]:
    if provider == "qdrant":
        _install_qdrant(monkeypatch, payload)
    else:
        _install_pinecone(monkeypatch, payload)
    from mind_mem.recall_vector import VectorBackend

    monkeypatch.setattr(VectorBackend, "embed", lambda self, texts: [[0.1, 0.2] for _ in texts])
    return recall(str(workspace), "canonical source", limit=10, rerank=False)


def test_remote_active_hit_with_matching_source_is_served_without_local_index(tmp_path, monkeypatch):
    workspace, payload = _workspace(tmp_path)
    result = _run(workspace, monkeypatch, payload)
    assert [row["_id"] for row in result] == ["D-REMOTE-1"]
    assert not (workspace / ".mind-mem-vectors/index.json").exists()


def test_pinecone_active_hit_with_matching_source_is_served_without_local_index(tmp_path, monkeypatch):
    workspace, payload = _workspace(tmp_path, provider="pinecone")
    result = _run(workspace, monkeypatch, payload, provider="pinecone")
    assert [row["_id"] for row in result] == ["D-REMOTE-1"]
    assert not (workspace / ".mind-mem-vectors/index.json").exists()


@pytest.mark.parametrize("provider", ["qdrant", "pinecone"])
def test_remote_payload_cannot_override_quarantined_current_source(tmp_path, monkeypatch, provider):
    workspace, payload = _workspace(tmp_path, status="quarantined", provider=provider)
    result = _run(workspace, monkeypatch, payload, provider=provider)
    assert result == []


@pytest.mark.parametrize("provider", ["qdrant", "pinecone"])
def test_remote_payload_cannot_serve_revoked_credential(tmp_path, monkeypatch, provider):
    workspace, payload = _workspace(tmp_path, status="revoked", credential=True, provider=provider)
    result = _run(workspace, monkeypatch, payload, provider=provider)
    assert result == []


def test_remote_same_id_with_foreign_source_or_digest_is_refused(tmp_path, monkeypatch):
    workspace, payload = _workspace(tmp_path)
    payload["file"] = "agents/other/DECISIONS.md"
    payload["source_digest"] = "0" * 64
    result = _run(workspace, monkeypatch, payload)
    assert result == []


def test_remote_source_status_is_not_replaced_by_foreign_id_status(tmp_path, monkeypatch):
    workspace, payload = _workspace(tmp_path)
    # Simulate a workspace-wide legacy status map containing a duplicate id
    # from another namespace.  Source-bound admission must remain authoritative
    # for the remote row's declared source.
    monkeypatch.setattr("mind_mem._recall_core.live_statuses", lambda _workspace: {"D-REMOTE-1": "quarantined"})
    result = _run(workspace, monkeypatch, payload)
    assert [row["_id"] for row in result] == ["D-REMOTE-1"]


def test_remote_missing_source_proof_is_refused(tmp_path, monkeypatch):
    workspace, payload = _workspace(tmp_path)
    payload.pop("source_digest")
    result = _run(workspace, monkeypatch, payload)
    assert result == []
