"""v3.3.0 Tier 1 #1 — query decomposition for multi-hop questions.

``decompose_query`` should always return [original, ...sub_queries]
so the caller can RRF-fuse the results. The NLP decomposer handles
common multi-hop patterns (temporal, causal, conjunction) with pure
regex; the LLM decomposer is opt-in and falls back to NLP.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from mind_mem.query_planner import (
    LLMQueryDecomposer,
    NLPQueryDecomposer,
    create_decomposer,
    decompose_query,
)


class TestNLPDecomposer:
    def test_temporal_after_splits(self) -> None:
        subs = NLPQueryDecomposer().decompose("What did Alice say after Bob left?")
        assert len(subs) >= 2
        assert subs[0] == "What did Alice say after Bob left?"
        assert any("alice" in s.lower() for s in subs[1:])
        assert any("bob" in s.lower() for s in subs[1:])

    def test_causal_splits(self) -> None:
        subs = NLPQueryDecomposer().decompose("Why did we choose PostgreSQL because of scaling?")
        assert len(subs) >= 2

    def test_conjunction_splits(self) -> None:
        subs = NLPQueryDecomposer().decompose("Where does Alice work and what project is she on?")
        assert len(subs) >= 2

    def test_single_hop_returns_original_only(self) -> None:
        """No split pattern → one-element list (just the original)."""
        subs = NLPQueryDecomposer().decompose("PostgreSQL deployment strategy")
        assert subs == ["PostgreSQL deployment strategy"]

    def test_multiple_question_marks_split(self) -> None:
        subs = NLPQueryDecomposer().decompose("When did she start? Where was she before?")
        assert len(subs) >= 2

    def test_empty_query_returns_empty(self) -> None:
        assert NLPQueryDecomposer().decompose("") == [""]
        assert NLPQueryDecomposer().decompose("   ") == ["   "]

    def test_original_always_first(self) -> None:
        q = "What did X say about Y after Z?"
        subs = NLPQueryDecomposer().decompose(q)
        assert subs[0] == q

    def test_max_subqueries_honored(self) -> None:
        q = "A after B and C because D and E"
        subs = NLPQueryDecomposer().decompose(q, max_subqueries=3)
        assert len(subs) <= 3

    def test_no_duplicate_subqueries(self) -> None:
        """Identical fragments from different patterns get deduped."""
        subs = NLPQueryDecomposer().decompose("foo and foo")
        # Only [original]; "foo" on both sides dedupes away
        assert len(subs) <= 2

    def test_too_short_subqueries_dropped(self) -> None:
        """Fragments like 'X' (< 2 words) are not useful retrieval queries."""
        subs = NLPQueryDecomposer().decompose("X and the full second clause here")
        # First fragment is 'X' — dropped for being < 2 words
        assert all(len(s.split()) >= 2 or s == subs[0] for s in subs)


class TestLLMDecomposer:
    def test_falls_back_to_nlp_on_http_error(self) -> None:
        dec = LLMQueryDecomposer({"base_url": "http://127.0.0.1:1/fail"})
        subs = dec.decompose("What did Alice say after Bob left?")
        # Should still produce ≥1 via NLP fallback
        assert subs[0] == "What did Alice say after Bob left?"
        assert len(subs) >= 2

    def test_parses_llm_response(self) -> None:
        dec = LLMQueryDecomposer()
        fake_body = {
            "choices": [
                {
                    "message": {
                        "content": "What did Alice say?\nWhen did Bob leave?\nDid anyone else comment?",
                    }
                }
            ]
        }
        with patch.object(
            dec,
            "_call_llm",
            return_value=[
                "What did Alice say?",
                "When did Bob leave?",
                "Did anyone else comment?",
            ],
        ):
            subs = dec.decompose("complex question")
        assert subs[0] == "complex question"
        assert len(subs) == 4

    def test_falls_back_when_llm_returns_only_original(self) -> None:
        dec = LLMQueryDecomposer()
        with patch.object(dec, "_call_llm", return_value=[]):
            subs = dec.decompose("What did Alice say after Bob left?")
        # LLM returned nothing useful → NLP kicked in
        assert len(subs) >= 2


class TestFactoryAndEntryPoint:
    def test_default_is_nlp(self) -> None:
        dec = create_decomposer({})
        assert isinstance(dec, NLPQueryDecomposer)

    def test_nlp_explicit(self) -> None:
        dec = create_decomposer({"retrieval": {"query_decomposition": {"provider": "nlp"}}})
        assert isinstance(dec, NLPQueryDecomposer)

    def test_llm_provider(self) -> None:
        dec = create_decomposer({"retrieval": {"query_decomposition": {"provider": "llm"}}})
        assert isinstance(dec, LLMQueryDecomposer)

    def test_unknown_provider_falls_back_to_nlp(self) -> None:
        dec = create_decomposer({"retrieval": {"query_decomposition": {"provider": "mystery"}}})
        assert isinstance(dec, NLPQueryDecomposer)

    def test_decompose_query_public_entry(self) -> None:
        subs = decompose_query("What did X say after Y?")
        assert subs[0] == "What did X say after Y?"
        assert len(subs) >= 2

    def test_decompose_query_empty(self) -> None:
        assert decompose_query("") == [""]

    def test_decompose_query_config_threading(self) -> None:
        """Config is passed through to the decomposer."""
        cfg = {"retrieval": {"query_decomposition": {"provider": "nlp"}}}
        subs = decompose_query("What did X do after Y?", config=cfg, max_subqueries=2)
        assert len(subs) <= 2
