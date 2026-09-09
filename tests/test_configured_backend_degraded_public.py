# Copyright 2026 STARGA, Inc.
"""Configured-vector degradation reaches the real public recall envelope."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from mind_mem import recall_cache


def _deterministic_embed(self, texts):
    return [[byte / 255.0 for byte in hashlib.sha256(text.encode("utf-8")).digest()[:8]] for text in texts]


@pytest.fixture(autouse=True)
def _isolated_cache():
    recall_cache.reset_singleton()
    yield
    recall_cache.reset_singleton()


@pytest.fixture
def deterministic_model(monkeypatch):
    from mind_mem import recall_vector

    monkeypatch.setattr(recall_vector.VectorBackend, "embed", _deterministic_embed)


def _workspace(root: Path, records: list[dict], *, canonical: bool) -> Path:
    for directory in ("decisions", "tasks", "entities", "intelligence"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    (root / "decisions" / "DECISIONS.md").write_text(
        "[D-1]\nStatement: retrieval carrier protocol shipped\nStatus: active\nDate: 2026-09-09\n\n",
        encoding="utf-8",
    )
    (root / "mind-mem.json").write_text(
        json.dumps({"recall": {"backend": "vector", "vector": {"provider": "local"}}}),
        encoding="utf-8",
    )
    vector_dir = root / ".mind-mem-vectors"
    vector_dir.mkdir()
    payload = {"blocks": records, "embeddings": [[0.1] * 8] * len(records)} if canonical else records
    (vector_dir / "index.json").write_text(json.dumps(payload), encoding="utf-8")
    return root


def _public_recall(workspace: Path, monkeypatch, query: str, *, mode: str = "auto", lifecycle: str = "") -> dict:
    from mind_mem.mcp.tools import public

    monkeypatch.setenv("MIND_MEM_WORKSPACE", str(workspace))
    monkeypatch.setenv("MIND_MEM_SCOPE", "admin")
    return json.loads(
        public.recall(
            query=query,
            mode=mode,
            limit=5,
            active_only=True,
            scoring_instant="2026-09-09",
            lifecycle=lifecycle,
        )
    )


LEGACY = [{"_id": "V-1", "embedding": [0.1] * 8, "excerpt": "retrieval carrier protocol"}]


@pytest.mark.parametrize("provider_rows", [[], [{"_id": "D-1", "status": "active", "score": 1.0}]])
def test_configured_backend_capture_handles_marked_empty_and_nonempty(provider_rows, tmp_path: Path, monkeypatch) -> None:
    """Direct engine seam: fallback and early-return paths both retain cause."""
    from mind_mem import _recall_core
    from mind_mem.hybrid_recall import RecallResults

    workspace = _workspace(tmp_path / ("empty" if not provider_rows else "nonempty"), [], canonical=True)

    class MarkedBackend(_recall_core.RecallBackend):
        def search(self, workspace, query, limit=10, active_only=False):
            result = RecallResults(provider_rows)
            result.degraded = {"leg": "vector", "reason": "deadline_exceeded"}
            return result

        def index(self, workspace):
            return None

    monkeypatch.setattr(_recall_core, "_load_backend", lambda unused: MarkedBackend())
    result = _recall_core.recall(str(workspace), "retrieval carrier protocol", limit=5, active_only=True)
    assert result, "positive control: provider or lexical fallback must answer"
    assert result.degraded == {"leg": "vector", "reason": "deadline_exceeded"}


def test_real_legacy_provider_marks_empty_fallback_in_envelope_and_attestation(tmp_path: Path, monkeypatch, deterministic_model) -> None:
    workspace = _workspace(tmp_path / "legacy", LEGACY, canonical=False)
    envelope = _public_recall(workspace, monkeypatch, "retrieval carrier protocol")
    assert envelope["count"] > 0, "positive control: lexical fallback must answer"
    assert envelope["degraded"]["leg"] == "vector"
    assert envelope["degraded"]["reason"] == "status_absent"
    assert envelope["degraded"]["index_shape"] == "legacy_list_shape"
    assert envelope["attestation"]["degraded"] == envelope["degraded"]


def test_real_marker_replays_unchanged_on_cache_hit(tmp_path: Path, monkeypatch, deterministic_model) -> None:
    workspace = _workspace(tmp_path / "cache", LEGACY, canonical=False)
    first = _public_recall(workspace, monkeypatch, "retrieval carrier protocol cache")
    second = _public_recall(workspace, monkeypatch, "retrieval carrier protocol cache")
    assert first["degraded"] == second["degraded"]
    assert second["degraded"]["reason"] == "status_absent"


def test_direct_bm25_surface_keeps_configured_backend_marker_through_filter(tmp_path: Path, monkeypatch, deterministic_model) -> None:
    workspace = _workspace(tmp_path / "direct", LEGACY, canonical=False)
    envelope = _public_recall(
        workspace,
        monkeypatch,
        "retrieval carrier protocol direct",
        mode="bm25",
        lifecycle="durable",
    )
    assert envelope["count"] > 0
    assert envelope["degraded"]["reason"] == "status_absent"
    assert envelope["attestation"]["degraded"] == envelope["degraded"]


def test_canonical_known_active_status_does_not_mark(tmp_path: Path, monkeypatch, deterministic_model) -> None:
    records = [{"_id": "D-1", "status": "active", "excerpt": "retrieval carrier protocol shipped"}]
    workspace = _workspace(tmp_path / "canonical", records, canonical=True)
    envelope = _public_recall(workspace, monkeypatch, "retrieval carrier protocol canonical")
    assert envelope["count"] > 0
    assert "degraded" not in envelope


def test_canonical_inactive_filtering_does_not_claim_degradation(tmp_path: Path, monkeypatch, deterministic_model) -> None:
    records = [{"_id": "D-9", "status": "archived", "excerpt": "retrieval carrier protocol archived"}]
    workspace = _workspace(tmp_path / "inactive", records, canonical=True)
    envelope = _public_recall(workspace, monkeypatch, "retrieval carrier protocol inactive")
    assert "degraded" not in envelope


def test_public_recall_does_not_rewrite_the_legacy_index(tmp_path: Path, monkeypatch, deterministic_model) -> None:
    workspace = _workspace(tmp_path / "frozen", LEGACY, canonical=False)
    index = workspace / ".mind-mem-vectors" / "index.json"
    before = hashlib.sha256(index.read_bytes()).hexdigest()
    _public_recall(workspace, monkeypatch, "retrieval carrier protocol frozen")
    assert hashlib.sha256(index.read_bytes()).hexdigest() == before
