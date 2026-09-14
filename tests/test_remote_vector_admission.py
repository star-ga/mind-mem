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


def _workspace(tmp_path: Path, *, status: str = "active", credential: bool = False, provider: str = "qdrant") -> tuple[Path, dict]:
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


def _install_qdrant(monkeypatch, payload: dict | None = None):
    stored_points = []

    class Hit:
        score = 0.99

        def __init__(self, row):
            self.payload = row

    class Client:
        def __init__(self, *, url):
            self.url = url

        def delete_collection(self, _collection):
            stored_points.clear()

        def create_collection(self, **_kwargs):
            return None

        def upsert(self, *, collection_name, points: list):
            del collection_name
            stored_points[:] = points

        def search(self, **kwargs):
            del kwargs
            rows = [point.payload for point in stored_points]
            if payload is not None:
                rows = [payload]
            return [Hit(row) for row in rows]

    qdrant = types.ModuleType("qdrant_client")
    qdrant.QdrantClient = Client
    models = types.ModuleType("qdrant_client.models")

    class PointStruct:
        def __init__(self, *, id, vector, payload):
            self.id, self.vector, self.payload = id, vector, payload

    class Filter:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FieldCondition:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class MatchValue:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class VectorParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    models.PointStruct = PointStruct
    models.VectorParams = VectorParams
    models.Distance = types.SimpleNamespace(COSINE="cosine")
    models.FieldCondition = FieldCondition
    models.Filter = Filter
    models.MatchValue = MatchValue
    monkeypatch.setitem(sys.modules, "qdrant_client", qdrant)
    monkeypatch.setitem(sys.modules, "qdrant_client.models", models)
    return stored_points


def _install_pinecone(monkeypatch, payload: dict | None = None):
    records = []

    class Index:
        def upsert_records(self, namespace, rows):
            del namespace
            records[:] = rows

        def search_records(self, *, namespace, query):
            del namespace, query
            rows = records or ([payload] if payload is not None else [])
            return {
                "result": {
                    "hits": [
                        {
                            "_id": row.get("_id"),
                            "_score": row.get("score", 0.99),
                            "fields": row
                            if payload is not None
                            else {
                                **row,
                                "block_type": row.get("block_type", row.get("type")),
                            },
                        }
                        for row in rows
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
    return records


def _run(
    workspace: Path,
    monkeypatch,
    payload: dict,
    provider: str = "qdrant",
    *,
    include_pending: bool = False,
) -> list[dict]:
    if provider == "qdrant":
        _install_qdrant(monkeypatch, payload)
    else:
        _install_pinecone(monkeypatch, payload)
    from mind_mem.recall_vector import VectorBackend

    monkeypatch.setattr(VectorBackend, "embed", lambda self, texts: [[0.1, 0.2] for _ in texts])
    return recall(str(workspace), "canonical source", limit=10, rerank=False, include_pending=include_pending)


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
def test_index_transport_search_retain_source_digest(tmp_path, monkeypatch, provider):
    workspace, expected = _workspace(tmp_path, provider=provider)
    from mind_mem.recall_vector import VectorBackend

    if provider == "qdrant":
        captured = _install_qdrant(monkeypatch)
    else:
        captured = _install_pinecone(monkeypatch)
    monkeypatch.setattr(VectorBackend, "_embed_for_provider", lambda self, texts: [[0.1, 0.2] for _ in texts])

    backend = VectorBackend({"provider": provider, "dimension": 2})
    backend.index(str(workspace))
    assert captured, "index() must emit a transport record"
    indexed = captured[0].payload if provider == "qdrant" else captured[0]
    assert indexed["source_digest"] == expected["source_digest"]

    result = recall(str(workspace), "canonical source", limit=10, rerank=False)
    assert [row["_id"] for row in result] == [expected["_id"]]
    assert result[0]["excerpt"] == "canonical source"


@pytest.mark.parametrize("provider", ["qdrant", "pinecone"])
def test_remote_payload_cannot_override_quarantined_current_source(tmp_path, monkeypatch, provider):
    workspace, payload = _workspace(tmp_path, status="quarantined", provider=provider)
    result = _run(workspace, monkeypatch, payload, provider=provider)
    assert result == []


@pytest.mark.parametrize("provider", ["qdrant", "pinecone"])
def test_remote_payload_content_is_rebuilt_from_current_source(tmp_path, monkeypatch, provider):
    workspace, payload = _workspace(tmp_path, provider=provider)
    payload["excerpt"] = "forged remote prose"
    payload["status"] = "quarantined"
    result = _run(workspace, monkeypatch, payload, provider=provider)
    assert len(result) == 1
    assert result[0]["excerpt"] == "canonical source"
    assert result[0]["status"] == "active"


@pytest.mark.parametrize("provider", ["qdrant", "pinecone"])
def test_remote_payload_cannot_serve_revoked_credential(tmp_path, monkeypatch, provider):
    workspace, payload = _workspace(tmp_path, status="revoked", credential=True, provider=provider)
    result = _run(workspace, monkeypatch, payload, provider=provider)
    assert result == []


@pytest.mark.parametrize("provider", ["qdrant", "pinecone"])
def test_pending_remote_hit_requires_explicit_include_pending(tmp_path, monkeypatch, provider):
    workspace, payload = _workspace(tmp_path, status="pending", provider=provider)
    assert _run(workspace, monkeypatch, payload, provider=provider) == []
    result = _run(workspace, monkeypatch, payload, provider=provider, include_pending=True)
    assert [row["_id"] for row in result] == ["D-REMOTE-1"]
    assert result[0]["status"] == "pending"


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
