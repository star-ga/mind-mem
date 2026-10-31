"""Tests for recall constants module."""

from __future__ import annotations

from mind_mem._recall_constants import (
    _STOPWORDS,
    _VALID_RECALL_KEYS,
    BM25_B,
    BM25_K1,
    FIELD_WEIGHTS,
    MAX_BLOCKS_PER_QUERY,
)


def test_bm25_k1_positive():
    """BM25 k1 parameter is positive."""
    assert BM25_K1 > 0


def test_bm25_b_range():
    """BM25 b parameter is between 0 and 1."""
    assert 0 <= BM25_B <= 1


def test_field_weights_not_empty():
    """Field weights dictionary is not empty."""
    assert len(FIELD_WEIGHTS) > 0


def test_field_weights_positive():
    """All field weights are positive."""
    for field, weight in FIELD_WEIGHTS.items():
        assert weight > 0, f"Field {field} has non-positive weight {weight}"


def test_max_blocks_positive():
    """Max blocks per query is positive."""
    assert MAX_BLOCKS_PER_QUERY > 0


def test_stopwords_not_empty():
    """Stopwords set is not empty."""
    assert len(_STOPWORDS) > 0


def test_stopwords_lowercase():
    """All stopwords are lowercase."""
    for word in _STOPWORDS:
        assert word == word.lower(), f"Stopword '{word}' is not lowercase"


def test_valid_recall_keys_not_empty():
    """Valid recall keys set is not empty."""
    assert len(_VALID_RECALL_KEYS) > 0


def test_common_stopwords_present():
    """Common English stopwords are in the set."""
    for word in ["the", "a", "is", "in", "of"]:
        assert word in _STOPWORDS, f"Common stopword '{word}' missing"


class TestBlockIdRegex:
    """``_BLOCK_ID_RE`` matches every declared generic block-id kind.

    Uses non-benchmark IDs throughout: synthetic prefixes and session
    labels that never appear in any evaluation dataset.
    """

    def test_dialogue_turn_id_matched(self):
        from mind_mem._recall_constants import _BLOCK_ID_RE

        assert _BLOCK_ID_RE.search("see DIA-X1-42 for context")
        assert _BLOCK_ID_RE.search("DIA-abc-1")
        assert _BLOCK_ID_RE.search("DIA-S9-100")

    def test_dialogue_turn_id_not_benchmark_specific(self):
        from mind_mem._recall_constants import _BLOCK_ID_RE

        # The generic mechanism must accept any alphanumeric session label,
        # not just the one layout a particular dataset happens to use.
        assert _BLOCK_ID_RE.search("ref DIA-Q7-3")
        assert _BLOCK_ID_RE.search("ref DIA-tt-99")

    def test_standard_prefixes_still_match(self):
        from mind_mem._recall_constants import _BLOCK_ID_RE

        assert _BLOCK_ID_RE.search("D-20260420-100")
        assert _BLOCK_ID_RE.search("FACT-001")
        assert _BLOCK_ID_RE.search("T-20260420-001")

    def test_non_id_text_not_matched(self):
        from mind_mem._recall_constants import _BLOCK_ID_RE

        assert _BLOCK_ID_RE.search("hello world") is None
        assert _BLOCK_ID_RE.search("DIA-") is None
        assert _BLOCK_ID_RE.search("FACT-") is None


class TestGenericMechanismsOnNonBenchmarkData:
    """The generalised mechanisms work on non-benchmark data.

    Each test exercises a mechanism whose comment was previously
    benchmark-specific, using synthetic inputs that appear in no dataset.
    """

    def test_conversational_synonym_expansion(self):
        from mind_mem._recall_expansion import _SYNONYMS

        # The conversational-term bridge works for any casual dialogue,
        # not just one dataset's speech patterns.
        assert "discuss" in _SYNONYMS.get("talk", [])
        assert "mention" in _SYNONYMS.get("said", [])

    def test_reranker_weights_are_generic(self):
        from mind_mem._recall_reranking import (
            _RERANK_W_BIGRAM,
            _RERANK_W_ENTITY,
            _RERANK_W_RECENCY,
            _RERANK_W_TIME,
        )

        for w in (_RERANK_W_ENTITY, _RERANK_W_TIME, _RERANK_W_BIGRAM, _RERANK_W_RECENCY):
            assert isinstance(w, float)
            assert 0.0 < w < 1.0

    def test_speaker_extraction_on_non_benchmark_names(self):
        from mind_mem.abstention_classifier import _extract_speaker_from_query

        result = _extract_speaker_from_query("Did Zara ever mention wanting a telescope?")
        assert result == "Zara"

    def test_verify_patterns_on_generic_shapes(self):
        from mind_mem.answer_quality import _VERIFY_PATTERNS

        # The shapes are date / time / number / proper-noun / yes-no --
        # general gold-answer categories, not any dataset's specific outputs.
        assert "date" in _VERIFY_PATTERNS
        assert "time" in _VERIFY_PATTERNS
        assert "number" in _VERIFY_PATTERNS
        assert "yes_no" in _VERIFY_PATTERNS
        assert _VERIFY_PATTERNS["date"].search("January 5, 2024")
        assert _VERIFY_PATTERNS["yes_no"].search("No, never mentioned it")
