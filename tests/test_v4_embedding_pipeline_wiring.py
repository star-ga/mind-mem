# Copyright 2026 STARGA, Inc.
"""``v4.embedding_pipeline`` is WIRED — 5.1.0 restoration slice.

The module exists to close the "caller-supplied embeddings" gap: every v4
surface that wants a vector took one as an argument, so nothing could derive
one. Its consumer is step 3 of the ``mm kinds backfill`` pass
(:func:`mind_mem.v4.kind_backfill.backfill`), which runs after ``block_kinds``
has written the block text it embeds.

Two things are asserted, because the module has two halves:

* :func:`~mind_mem.v4.embedding_pipeline.derive_embeddings` is CALLED and its
  vectors are what get registered for kind-filtered kNN;
* :func:`~mind_mem.v4.embedding_pipeline.set_embedder` is pointed at
  ``recall_vector``'s real provider chain, with the stdlib hashed-trigram
  embedder as an explicit, logged fallback — never silently passed off as a
  semantic embedding.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mind_mem import mm_cli
from mind_mem.v4 import embedding_pipeline, kind_backfill


class _StubBackend:
    """Stands in for ``recall_vector.VectorBackend``. No model, no download."""

    calls: list[str] = []

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def _embed_for_provider(self, texts: list[str]) -> list[list[float]]:
        _StubBackend.calls.extend(texts)
        # Deterministic and 4-dimensional, so it is trivially distinguishable
        # from the 128-dim hashed-trigram default.
        return [[1.0, 0.0, 0.0, float(len(t) % 7)] for t in texts]


def _build_workspace(root: Path) -> None:
    for sub in ("decisions", "tasks", "entities", "intelligence", "memory"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    (root / "decisions" / "DECISIONS.md").write_text(
        "[D-20260101-001]\nStatement: Use PostgreSQL for the user database\nStatus: active\n",
        encoding="utf-8",
    )
    (root / "entities" / "projects.md").write_text(
        "[PRJ-mind-mem]\nName: mind-mem\nStatus: active\n",
        encoding="utf-8",
    )


def _write_config(root: Path, *, pipeline: bool) -> Path:
    cfg = root / "mind-mem.json"
    v4: dict = {"block_kinds": {"enabled": True}, "hnsw_kind_index": {"enabled": True}}
    if pipeline:
        v4["embedding_pipeline"] = {"enabled": True}
    cfg.write_text(json.dumps({"version": "5.1.0", "recall": {"backend": "scan"}, "v4": v4}), encoding="utf-8")
    return cfg


@pytest.fixture(autouse=True)
def _restore_embedder():
    """The active embedder is module-global; never leak one between tests."""
    original = embedding_pipeline._active_embedder
    _StubBackend.calls = []
    yield
    embedding_pipeline._active_embedder = original


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "ws"
    root.mkdir()
    _build_workspace(root)
    monkeypatch.setenv("MIND_MEM_WORKSPACE", str(root))
    monkeypatch.setenv("MIND_MEM_SCOPE", "user")
    return root


@pytest.fixture
def armed(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("MIND_MEM_CONFIG", str(_write_config(workspace, pipeline=True)))
    return workspace


@pytest.fixture
def disarmed(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("MIND_MEM_CONFIG", str(_write_config(workspace, pipeline=False)))
    return workspace


@pytest.fixture
def no_provider(monkeypatch: pytest.MonkeyPatch):
    """``recall_vector`` unusable — the documented fallback path."""
    import mind_mem.recall_vector as rv

    def _boom(config):
        raise ImportError("sentence-transformers not installed")

    monkeypatch.setattr(rv, "VectorBackend", _boom)


@pytest.fixture
def stub_provider(monkeypatch: pytest.MonkeyPatch):
    import mind_mem.recall_vector as rv

    monkeypatch.setattr(rv, "VectorBackend", _StubBackend)


# ---------------------------------------------------------------------------
# The call site
# ---------------------------------------------------------------------------


class TestTheBackfillDerivesEmbeddings:
    def test_derive_embeddings_is_called_and_its_vectors_are_stored(self, armed: Path, no_provider) -> None:
        result = kind_backfill.backfill(armed)
        assert result.embeddings_derived == 2
        assert result.embeddings_registered == 2

        from mind_mem.v4.hnsw_kind_index import get_block_embedding

        vec = get_block_embedding(armed, "PRJ-mind-mem")
        assert len(vec) == 128, "the registered vector is not the pipeline's"

    def test_the_call_site_is_load_bearing(self, armed: Path, no_provider, monkeypatch: pytest.MonkeyPatch) -> None:
        """Remove the ``derive_embeddings`` call and this records nothing."""
        seen: list[list[str]] = []
        real = embedding_pipeline.derive_embeddings

        def _spy(ws, ids, **kw):
            ids = list(ids)
            seen.append(sorted(ids))
            return real(ws, ids, **kw)

        monkeypatch.setattr(embedding_pipeline, "derive_embeddings", _spy)
        assert mm_cli.main(["kinds", "backfill"]) == 0
        assert seen == [["D-20260101-001", "PRJ-mind-mem"]]

    def test_the_pipeline_reads_the_text_block_kinds_wrote(self, armed: Path, no_provider) -> None:
        """Dependency order, made observable.

        ``derive_embeddings`` looks content up in ``index.db(blocks)`` and
        only then falls back to the v3 recall index. Step 1 is what puts it
        there, so an empty-content store would give every block the
        zero-vector the module returns for empty input.
        """
        kind_backfill.backfill(armed)
        from mind_mem.v4.hnsw_kind_index import get_block_embedding

        vec = get_block_embedding(armed, "D-20260101-001")
        assert any(x != 0.0 for x in vec), "embedded an empty string — the content was not written first"

    def test_derivation_is_deterministic(self, armed: Path, no_provider) -> None:
        first = embedding_pipeline.derive_embedding("Use PostgreSQL for the user database")
        second = embedding_pipeline.derive_embedding("Use PostgreSQL for the user database")
        assert first == second


# ---------------------------------------------------------------------------
# The recall_vector bridge
# ---------------------------------------------------------------------------


class TestTheEmbedderBridge:
    def test_the_real_provider_chain_is_installed_when_it_answers(self, armed: Path, stub_provider) -> None:
        assert kind_backfill._install_recall_vector_embedder(str(armed)) == "recall_vector"
        assert _StubBackend.calls, "the provider chain was installed but never probed"
        assert embedding_pipeline.derive_embedding("hello") == [1.0, 0.0, 0.0, 5.0]

    def test_backfill_registers_the_provider_vectors(self, armed: Path, stub_provider) -> None:
        kind_backfill.backfill(armed)
        from mind_mem.v4.hnsw_kind_index import get_block_embedding

        vec = get_block_embedding(armed, "PRJ-mind-mem")
        assert len(vec) == 4, "the stdlib default ran even though a provider was available"

    def test_an_unusable_provider_falls_back_instead_of_crashing(self, armed: Path, no_provider) -> None:
        assert kind_backfill._install_recall_vector_embedder(str(armed)) == "hashed_trigram"
        assert len(embedding_pipeline.derive_embedding("hello")) == 128

    def test_a_provider_that_returns_nothing_is_not_installed(self, armed: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The probe exists because a chain can import fine and answer badly."""
        import mind_mem.recall_vector as rv

        class _Empty(_StubBackend):
            def _embed_for_provider(self, texts):
                return []

        monkeypatch.setattr(rv, "VectorBackend", _Empty)
        assert kind_backfill._install_recall_vector_embedder(str(armed)) == "hashed_trigram"
        assert len(embedding_pipeline.derive_embedding("hello")) == 128


# ---------------------------------------------------------------------------
# Flag OFF
# ---------------------------------------------------------------------------


class TestFlagOffIsUnchanged:
    def test_flag_off_never_calls_the_module(self, disarmed: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        def _explode(*a, **kw):
            raise AssertionError("embedding_pipeline ran with the flag OFF")

        monkeypatch.setattr(embedding_pipeline, "derive_embeddings", _explode)
        monkeypatch.setattr(embedding_pipeline, "set_embedder", _explode)
        result = kind_backfill.backfill(disarmed)
        assert result.embeddings_derived == 0
        assert result.steps_enabled["embedding_pipeline"] is False

    def test_flag_is_registered(self) -> None:
        from mind_mem.v4 import feature_flags

        assert "embedding_pipeline" in feature_flags.ALL_V4_FLAGS
