# Copyright 2026 STARGA, Inc.
"""``index()`` must embed through the same provider chain as everything else.

``VectorBackend`` resolves an embedder through ``_embed_for_provider``: ollama,
then llama_cpp, then fastembed, then sentence-transformers as an unguarded last
resort. Five call sites go through it — ``search``, the sqlite-vec builder, the
module-level ``rebuild_index``, and the recall core.

``index()`` did not. It called ``self.embed`` directly, which *is* the raw
sentence-transformers path, so the local-index builder skipped every earlier
provider.

The visible failure, measured 2026-09-07: a workspace configured for ollama
with ``model = "mxbai-embed-large"`` built an EMPTY index. That name is an
ollama tag, not a HuggingFace repo id, so ``SentenceTransformer`` cannot
resolve it and raises ``OSError`` — while ``embed_ollama`` answers the same
config correctly with 1024-dimensional vectors. The LongMemEval adapter
reported it honestly (``vector_index_blocks: 0``, ``vector_leg_inert: True``),
which is the only reason it was noticed rather than shipping as a hybrid number
whose dense leg contributed nothing.

Both tests below pair the assertion with proof it could have failed: the
ollama stub records that it was called, and the sentence-transformers stub
raises the real ``OSError`` rather than being merely absent.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from mind_mem.recall_vector import VectorBackend


def _workspace(tmp_path: Path) -> str:
    """A real workspace with two admitted blocks, rendered by the store.

    Hand-written markdown does not produce admitted blocks: ``index()`` finds
    nothing, returns before embedding, and the provider assertions below pass
    vacuously. The first draft of this file did exactly that and its positive
    control caught it.
    """
    from mind_mem.block_store import _render_block
    from mind_mem.init_workspace import init

    root = str(tmp_path / "ws")
    os.makedirs(root)
    init(root)
    decisions = Path(root) / "decisions" / "DECISIONS.md"
    decisions.write_text(
        _render_block(
            {
                "_id": "DEC-20260101-001",
                "Statement": "the sky over the harbour was grey",
                "Date": "2026-01-01",
                "Status": "active",
                "Type": "decision",
            }
        )
        + "\n"
        + _render_block(
            {
                "_id": "DEC-20260101-002",
                "Statement": "the ferry left at noon",
                "Date": "2026-01-02",
                "Status": "active",
                "Type": "decision",
            }
        ),
        encoding="utf-8",
    )
    return root


class TestIndexUsesTheProviderChain:
    def test_an_ollama_model_name_does_not_reach_sentence_transformers(self, tmp_path, monkeypatch):
        ws = _workspace(tmp_path)
        vb = VectorBackend({"provider": "local", "model": "mxbai-embed-large"})

        called = {"ollama": 0, "st": 0}

        def fake_ollama(texts):
            called["ollama"] += 1
            return [[0.1] * 1024 for _ in texts]

        def fake_st(texts):
            # The real failure this test exists for: SentenceTransformer cannot
            # resolve an ollama tag as a HuggingFace repo id.
            called["st"] += 1
            raise OSError("mxbai-embed-large is not a local folder and is not a valid model identifier")

        monkeypatch.setattr(vb, "embed_ollama", fake_ollama)
        monkeypatch.setattr(vb, "embed", fake_st)

        vb.index(ws)

        assert called["ollama"] >= 1, (
            "positive control failed: the ollama provider was never called, so this "
            "test would pass for a build that embedded nothing at all"
        )
        assert called["st"] == 0, "index() fell through to sentence-transformers with an ollama model name"

        index = vb._load_local_index(ws)
        assert index, "no index file was produced"
        assert len(index.get("blocks") or []) >= 2, (
            f"the index is empty ({len(index.get('blocks') or [])} blocks) — the "
            "dense leg would be inert while still reporting a hybrid backend"
        )

    def test_the_index_is_read_back_through_the_reader(self, tmp_path, monkeypatch):
        """Counting what the writer claims it wrote is how a shape mismatch hides."""
        ws = _workspace(tmp_path)
        vb = VectorBackend({"provider": "local", "model": "mxbai-embed-large"})
        monkeypatch.setattr(vb, "embed_ollama", lambda texts: [[0.2] * 1024 for _ in texts])
        monkeypatch.setattr(vb, "embed", lambda texts: pytest.fail("index() used the raw embedder"))

        vb.index(ws)
        index = vb._load_local_index(ws)

        blocks = index.get("blocks") or []
        embeddings = index.get("embeddings") or []
        assert len(blocks) == len(embeddings) and blocks, (
            f"reader round trip disagrees: {len(blocks)} blocks vs {len(embeddings)} embeddings"
        )
        assert len(embeddings[0]) == 1024, "the stored vector lost its dimension"


class TestAnExhaustedChainSaysWhatHappened:
    """The last resort cannot apply, so its error must not be the one shown.

    When every provider declines, the chain falls through to
    sentence-transformers. For an ollama-configured workspace that can never
    succeed — the model name is a provider tag, not a HuggingFace repo id — so
    the escaping error reads "... is not a valid model identifier" and points an
    operator at HuggingFace, which is not where the problem is.

    Measured 2026-09-07: across a 470-question hybrid benchmark, 36 runs (7.7%)
    surfaced exactly that OSError and built an EMPTY index, while ollama was
    healthy before and after. The benchmark reported it honestly as
    ``vector_leg_inert: True``; the error text is what misdirected.
    """

    def test_the_error_names_the_provider_not_huggingface(self, monkeypatch):
        from mind_mem.recall_vector import EmbeddingProvidersExhausted

        vb = VectorBackend({"provider": "local", "model": "mxbai-embed-large"})
        monkeypatch.setattr(vb, "embed_ollama", lambda t: (_ for _ in ()).throw(RuntimeError("connection refused")))
        monkeypatch.setattr(vb, "embed_fastembed", lambda t: (_ for _ in ()).throw(RuntimeError("absent")))
        monkeypatch.setattr(
            vb,
            "embed",
            lambda t: (_ for _ in ()).throw(OSError("mxbai-embed-large is not a local folder and is not a valid model identifier")),
        )

        with pytest.raises(EmbeddingProvidersExhausted) as caught:
            vb._embed_for_provider(["anything"])

        msg = str(caught.value)
        assert "ollama" in msg, "the message does not name the provider that actually failed"
        assert "provider tag" in msg, "the message does not explain why the last resort cannot apply"
        # Positive control: the underlying error is preserved, not swallowed.
        assert isinstance(caught.value.__cause__, OSError)

    def test_a_real_huggingface_id_still_raises_its_own_error(self, monkeypatch):
        """The rewrite must be narrow: a genuine repo id keeps its own message.

        Without this control the first test passes for a build that replaces
        EVERY last-resort failure, hiding real HuggingFace problems behind a
        message about ollama.
        """
        vb = VectorBackend({"provider": "local", "model": "mixedbread-ai/mxbai-embed-large-v1"})
        monkeypatch.setattr(vb, "embed_ollama", lambda t: (_ for _ in ()).throw(RuntimeError("down")))
        monkeypatch.setattr(vb, "embed_fastembed", lambda t: (_ for _ in ()).throw(RuntimeError("absent")))
        monkeypatch.setattr(vb, "embed", lambda t: (_ for _ in ()).throw(OSError("gated repo")))

        with pytest.raises(OSError) as caught:
            vb._embed_for_provider(["anything"])
        assert "gated repo" in str(caught.value)
