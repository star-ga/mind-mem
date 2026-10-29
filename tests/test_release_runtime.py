"""Serving-boundary tests use invented questions and fake retrieval only."""

import contextlib
from types import SimpleNamespace

from benchmarks.release_runtime import runtime_answer


def test_runtime_does_not_receive_evaluation_labels(monkeypatch):
    import mind_mem.chat_memory
    import mind_mem.hybrid_recall
    import mind_mem.recall
    import mind_mem.recall_attestation

    observed = {}

    def chat(workspace, question, **kwargs):
        observed.update(kwargs)
        assert "category" not in kwargs
        assert "reference" not in kwargs and "gold_evidence" not in kwargs
        hits = kwargs["recall_fn"](workspace, question, kwargs["limit"])
        result = kwargs["generator"](SimpleNamespace(prompt="Evidence: " + hits[0]["excerpt"]))
        return SimpleNamespace(answer=result)

    class Backend:
        vector_available = True

        def search(self, question, workspace, **kwargs):
            assert question == "What color is the lighthouse?"
            return [{"_id": "example-source", "excerpt": "The lighthouse is orange."}]

    monkeypatch.setattr(mind_mem.chat_memory, "chat_with_memory", chat)
    monkeypatch.setattr(mind_mem.recall, "serving_scope", contextlib.nullcontext)
    monkeypatch.setattr(mind_mem.hybrid_recall, "resolve_rerank_depth", lambda cfg, limit: limit)
    monkeypatch.setattr(mind_mem.recall_attestation, "derive_legs", lambda *a, **kw: (["vector", "bm25"], []))
    config = {"system_prompt": "Use the provided evidence.", "limit": 3, "max_evidence_chars": 120}
    result = runtime_answer("What color is the lighthouse?", ".", {}, config, lambda messages: ("orange [[example-source]]", 12), Backend())
    assert result["prompt_tokens"] == 12
    assert result["generated"] == result["product_answer"]
    assert observed["max_evidence_chars"] == 120
