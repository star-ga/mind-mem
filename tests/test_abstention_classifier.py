"""Tests for the adversarial abstention classifier."""

from __future__ import annotations

import os
import sys

# Ensure scripts/ is on path
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "scripts"))

from abstention_classifier import (  # noqa: E402
    ABSTENTION_ANSWER,
    DEFAULT_THRESHOLD,
    AbstentionResult,
    _extract_query_entities,
    _extract_speaker_from_query,
    _speaker_in_hit,
    _term_overlap,
    classify_abstention,
)

# ── Fixtures ─────────────────────────────────────────────────────────

def _make_hit(excerpt: str, score: float = 5.0, speaker: str = "Emma") -> dict:
    """Build a minimal recall hit for testing."""
    return {
        "excerpt": excerpt,
        "score": score,
        "speaker": speaker,
        "DiaID": "D1:1",
        "tags": f"speaker:{speaker}",
        "_id": "test-001",
    }


RELEVANT_HITS = [
    _make_hit("Emma mentioned she wanted to adopt a golden retriever puppy", score=8.5),
    _make_hit("Emma said she loves dogs and has been looking at shelters", score=7.2),
    _make_hit("During the conversation, Emma talked about her pet preferences", score=6.0),
    _make_hit("Emma also mentioned her neighbor has a dog she walks sometimes", score=5.5),
    _make_hit("Emma discussed her weekend plans involving the animal shelter", score=4.8),
]

IRRELEVANT_HITS = [
    _make_hit("John talked about his new car and the dealership experience", score=3.1, speaker="John"),
    _make_hit("The weather forecast showed rain for the entire week", score=2.5, speaker="John"),
    _make_hit("Sarah mentioned her vacation plans to Italy next summer", score=2.0, speaker="Sarah"),
    _make_hit("The meeting agenda covered quarterly budget reviews", score=1.8, speaker="Manager"),
    _make_hit("Technical discussion about API rate limiting strategies", score=1.2, speaker="Dev"),
]

MIXED_HITS = [
    _make_hit("Emma mentioned she wanted to adopt a golden retriever puppy", score=8.5),
    _make_hit("John talked about his new car and the dealership experience", score=3.1, speaker="John"),
    _make_hit("Sarah mentioned her vacation plans to Italy next summer", score=2.0, speaker="Sarah"),
    _make_hit("The meeting agenda covered quarterly budget reviews", score=1.8, speaker="Manager"),
    _make_hit("Technical discussion about API rate limiting strategies", score=1.2, speaker="Dev"),
]


# ── Unit tests: entity extraction ────────────────────────────────────

class TestExtractQueryEntities:
    def test_basic_extraction(self):
        entities = _extract_query_entities("Did Emma mention adopting a dog?")
        assert "emma" in entities
        assert "adopting" in entities
        assert "dog" in entities

    def test_stops_removed(self):
        entities = _extract_query_entities("Did she ever mention wanting to adopt?")
        assert "she" not in entities
        assert "ever" not in entities
        assert "adopt" in entities
        assert "wanting" in entities

    def test_empty_query(self):
        assert _extract_query_entities("") == set()


class TestExtractSpeaker:
    def test_finds_name(self):
        assert _extract_speaker_from_query("Did Emma ever mention dogs?") == "emma"

    def test_finds_full_name(self):
        result = _extract_speaker_from_query("Did Emma Watson talk about her career?")
        assert result in ("emma watson", "emma")

    def test_no_name(self):
        assert _extract_speaker_from_query("what was discussed about dogs?") is None

    def test_skips_question_words(self):
        # "Did" and "The" should not be extracted as names
        assert _extract_speaker_from_query("Did the group discuss plans?") is None


class TestTermOverlap:
    def test_full_overlap(self):
        overlap = _term_overlap("Emma loves dogs and wants to adopt one", {"emma", "dogs", "adopt"})
        assert overlap == 1.0

    def test_partial_overlap(self):
        overlap = _term_overlap("Emma loves cats", {"emma", "dogs", "adopt"})
        assert 0.0 < overlap < 1.0

    def test_no_overlap(self):
        overlap = _term_overlap("Technical budget review", {"emma", "dogs", "adopt"})
        assert overlap == 0.0

    def test_empty_entities(self):
        assert _term_overlap("some text", set()) == 0.0


class TestSpeakerInHit:
    def test_speaker_field_match(self):
        hit = _make_hit("some text", speaker="Emma")
        assert _speaker_in_hit(hit, "emma") is True

    def test_excerpt_match(self):
        hit = _make_hit("Emma said hello", speaker="Unknown")
        assert _speaker_in_hit(hit, "emma") is True

    def test_no_match(self):
        hit = _make_hit("John said hello", speaker="John")
        assert _speaker_in_hit(hit, "emma") is False

    def test_no_speaker_query(self):
        hit = _make_hit("some text")
        assert _speaker_in_hit(hit, "") is False


# ── Integration tests: classify_abstention ───────────────────────────

class TestClassifyAbstention:
    def test_no_hits_abstains(self):
        result = classify_abstention("Did Emma ever adopt a dog?", [])
        assert result.should_abstain is True
        assert result.confidence == 0.0
        assert result.forced_answer == ABSTENTION_ANSWER

    def test_relevant_hits_no_abstain(self):
        result = classify_abstention("Did Emma ever mention adopting a dog?", RELEVANT_HITS)
        assert result.should_abstain is False
        assert result.confidence > DEFAULT_THRESHOLD

    def test_irrelevant_hits_abstains(self):
        result = classify_abstention("Did Emma ever mention adopting a dog?", IRRELEVANT_HITS)
        assert result.should_abstain is True
        assert result.confidence < DEFAULT_THRESHOLD

    def test_mixed_hits_intermediate_confidence(self):
        result = classify_abstention("Did Emma ever mention adopting a dog?", MIXED_HITS)
        # With 1 relevant + 4 irrelevant, confidence should be moderate
        assert 0.0 < result.confidence < 0.8

    def test_forced_answer_on_abstention(self):
        result = classify_abstention("Did Emma ever mention quantum physics?", IRRELEVANT_HITS)
        assert result.should_abstain is True
        assert result.forced_answer == ABSTENTION_ANSWER

    def test_no_forced_answer_when_confident(self):
        result = classify_abstention("Did Emma ever mention adopting a dog?", RELEVANT_HITS)
        assert result.forced_answer == ""

    def test_features_populated(self):
        result = classify_abstention("Did Emma ever mention dogs?", RELEVANT_HITS)
        assert "entity_overlap" in result.features
        assert "top1_score_raw" in result.features
        assert "speaker_coverage" in result.features
        assert "evidence_density" in result.features
        assert "speaker_detected" in result.features

    def test_threshold_tuning(self):
        # Very high threshold should cause abstention even with good hits
        result = classify_abstention(
            "Did Emma ever mention adopting a dog?",
            RELEVANT_HITS,
            threshold=0.99,
        )
        assert result.should_abstain is True

        # Zero threshold should never abstain (unless no hits)
        result = classify_abstention(
            "Did Emma ever mention quantum physics?",
            IRRELEVANT_HITS,
            threshold=0.0,
        )
        assert result.should_abstain is False

    def test_ever_pattern_penalty(self):
        """'did X ever' pattern should penalize low-overlap results."""
        r1 = classify_abstention("Did Emma ever mention dogs?", IRRELEVANT_HITS)
        r2 = classify_abstention("What did Emma say about dogs?", IRRELEVANT_HITS)
        # "ever" variant should have lower confidence due to negation penalty
        assert r1.features["has_ever_pattern"] is True
        assert r2.features["has_ever_pattern"] is False
        assert r1.confidence <= r2.confidence

    def test_never_pattern_also_triggers_penalty(self):
        """'never' should trigger the same negation penalty as 'ever'."""
        r = classify_abstention("Did Emma never mention dogs?", IRRELEVANT_HITS)
        assert r.features["has_ever_pattern"] is True
        assert r.confidence <= classify_abstention(
            "What did Emma say about dogs?", IRRELEVANT_HITS
        ).confidence

    def test_result_is_dataclass(self):
        result = classify_abstention("test?", [])
        assert isinstance(result, AbstentionResult)

    def test_confidence_clamped(self):
        result = classify_abstention("test?", RELEVANT_HITS)
        assert 0.0 <= result.confidence <= 1.0

    def test_non_adversarial_question(self):
        """Normal factual questions with good hits should not abstain."""
        good_hits = [
            _make_hit("The project deadline is March 15th", score=9.0, speaker="Manager"),
            _make_hit("We agreed on March 15th for the final delivery", score=8.0, speaker="Manager"),
        ]
        result = classify_abstention("What is the project deadline?", good_hits)
        assert result.should_abstain is False


# ── Edge cases ───────────────────────────────────────────────────────

class TestEdgeCases:
    def test_single_hit(self):
        result = classify_abstention(
            "Did Emma mention dogs?",
            [_make_hit("Emma loves dogs", score=8.0)],
        )
        assert isinstance(result, AbstentionResult)

    def test_hit_with_zero_score(self):
        hits = [_make_hit("some text about Emma dogs", score=0.0)]
        result = classify_abstention("Did Emma mention dogs?", hits)
        # Zero BM25 score lowers confidence vs same hit with high score
        high_score_hits = [_make_hit("some text about Emma dogs", score=8.0)]
        result_high = classify_abstention("Did Emma mention dogs?", high_score_hits)
        assert result.confidence < result_high.confidence

    def test_hit_missing_fields(self):
        """Hits with missing optional fields should not crash."""
        hits = [{"excerpt": "Emma mentioned dogs", "score": 5.0}]
        result = classify_abstention("Did Emma mention dogs?", hits)
        assert isinstance(result, AbstentionResult)

    def test_unicode_query(self):
        result = classify_abstention("Did Emma mention cafe?", RELEVANT_HITS)
        assert isinstance(result, AbstentionResult)

    def test_none_field_values(self):
        """Hits where fields are present but explicitly None must not crash."""
        hits = [{"excerpt": None, "score": 5.0, "speaker": None, "_id": "x"}]
        result = classify_abstention("Did Emma mention dogs?", hits)
        assert isinstance(result, AbstentionResult)

    def test_top_k_zero_with_nonempty_hits(self):
        """top_k=0 with non-empty hits must not crash — treated as no hits."""
        result = classify_abstention("Did Emma mention dogs?", RELEVANT_HITS, top_k=0)
        assert result.should_abstain is True
        assert result.confidence == 0.0

    def test_top_k_larger_than_hits(self):
        result = classify_abstention("Did Emma mention dogs?", RELEVANT_HITS, top_k=100)
        assert result.features["top_k_examined"] == len(RELEVANT_HITS)
        assert isinstance(result, AbstentionResult)

    def test_top_k_limits_hits_examined(self):
        result_k1 = classify_abstention("Did Emma mention dogs?", RELEVANT_HITS, top_k=1)
        result_k5 = classify_abstention("Did Emma mention dogs?", RELEVANT_HITS, top_k=5)
        assert result_k1.features["top_k_examined"] == 1
        assert result_k5.features["top_k_examined"] == 5

    def test_bm25_score_above_10_clamped(self):
        """BM25 scores above 10 should clamp top1_norm to 1.0."""
        hits = [_make_hit("Emma mentioned dogs", score=25.0)]
        result = classify_abstention("Did Emma mention dogs?", hits)
        assert result.features["top1_score_norm"] == 1.0


# ── Exact numerical verification of confidence formula ───────────────

class TestConfidenceFormula:
    def test_weighted_confidence_exact_all_match(self):
        """Manual reconstruction: all features maximal, no ever pattern."""
        # Query "Emma dogs" → entities after stops = {"emma", "dogs"}
        # Hit: excerpt has both, score=10, speaker=Emma
        # mean_overlap=1.0, top1_norm=1.0, speaker_cov=1.0,
        # evidence_density=1.0, negation_penalty=0.0
        # expected = 0.35*1.0 + 0.20*1.0 + 0.15*1.0 + 0.20*1.0 + (-0.10)*0.0
        #          = 0.90
        hit = {"excerpt": "Emma loves dogs", "score": 10.0, "speaker": "Emma"}
        result = classify_abstention("Emma dogs", [hit], top_k=1)
        assert abs(result.confidence - 0.90) < 1e-3
        assert abs(result.features["entity_overlap"] - 1.0) < 1e-4
        assert abs(result.features["top1_score_norm"] - 1.0) < 1e-4
        assert abs(result.features["speaker_coverage"] - 1.0) < 1e-4
        assert abs(result.features["evidence_density"] - 1.0) < 1e-4
        assert abs(result.features["negation_penalty"] - 0.0) < 1e-4

    def test_weighted_confidence_with_ever_penalty_clamped(self):
        """When overlap=0 and ever pattern fires, penalty clamps to 0.0."""
        # Query: "Did Emma ever adopt" → has_ever_pattern=True
        # Hit: excerpt="budget review", score=1.0, speaker="Finance"
        # entities after stops: {"emma", "adopt"}
        # overlap("budget review", {"emma","adopt"}) = 0/2 = 0.0
        # mean_overlap=0.0, top1_norm=0.1, speaker_cov=0.0 (emma not in finance)
        # evidence_density=0.0, negation_penalty=1.0-0.0=1.0
        # expected = 0.35*0.0 + 0.20*0.1 + 0.15*0.0 + 0.20*0.0 + (-0.10)*1.0
        #          = 0.02 - 0.10 = -0.08 → clamped to 0.0
        hit = {"excerpt": "budget review", "score": 1.0, "speaker": "Finance"}
        result = classify_abstention("Did Emma ever adopt", [hit], top_k=1)
        assert result.confidence == 0.0  # clamped from -0.08
        assert result.should_abstain is True
        assert abs(result.features["negation_penalty"] - 1.0) < 1e-4

    def test_no_speaker_neutral_coverage(self):
        """Queries without a speaker name produce speaker_coverage=0.5."""
        hits = [_make_hit("The deadline is March 15th", score=8.0, speaker="Manager")]
        result = classify_abstention("What is the project deadline?", hits)
        assert result.features["speaker_detected"] is None
        assert abs(result.features["speaker_coverage"] - 0.5) < 1e-4

    def test_threshold_exact_boundary(self):
        """confidence == threshold exactly → should NOT abstain (strict <)."""
        hit = {"excerpt": "Emma loves dogs", "score": 10.0, "speaker": "Emma"}
        result = classify_abstention("Emma dogs", [hit], top_k=1)
        exact = result.confidence

        # At exact threshold: must NOT abstain
        result2 = classify_abstention("Emma dogs", [hit], top_k=1, threshold=exact)
        assert result2.should_abstain is False

        # Just above: must abstain
        result3 = classify_abstention("Emma dogs", [hit], top_k=1, threshold=exact + 0.001)
        assert result3.should_abstain is True


# ── Term overlap substring behavior ─────────────────────────────────

class TestTermOverlapSubstring:
    def test_empty_excerpt(self):
        assert _term_overlap("", {"emma", "dogs"}) == 0.0

    def test_substring_match_is_intentional(self):
        """'dog' matches inside 'hotdogs' — substring, not word boundary.
        This is the current behavior and is pinned here."""
        assert _term_overlap("hotdogs for sale", {"dog"}) == 1.0
