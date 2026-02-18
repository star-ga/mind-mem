"""Tests for the evidence packer module."""

from __future__ import annotations

import os
import sys

# Ensure scripts/ is on path
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "scripts"))

from evidence_packer import (  # noqa: E402
    check_abstention,
    is_true_adversarial,
    pack_evidence,
    strip_semantic_prefix,
)
from abstention_classifier import ABSTENTION_ANSWER  # noqa: E402


# ── Fixtures ─────────────────────────────────────────────────────────

def _hit(excerpt="text", score=5.0, speaker="Emma", dia_id="D1:1", date="2024-01-01"):
    return {
        "excerpt": excerpt,
        "score": score,
        "speaker": speaker,
        "DiaID": dia_id,
        "Date": date,
    }


# ── strip_semantic_prefix ────────────────────────────────────────────

class TestStripSemanticPrefix:
    def test_removes_prefix(self):
        assert strip_semantic_prefix("(identity description) Emma is a teacher") == "Emma is a teacher"

    def test_no_prefix(self):
        assert strip_semantic_prefix("Emma is a teacher") == "Emma is a teacher"

    def test_empty_string(self):
        assert strip_semantic_prefix("") == ""

    def test_long_prefix_over_80_chars(self):
        """Prefixes over 80 chars are NOT stripped (safety bound)."""
        long_label = "(" + "x" * 85 + ") real text"
        assert strip_semantic_prefix(long_label) == long_label

    def test_prefix_with_no_trailing_space(self):
        """Regex uses \\s* so (label)text with no space still strips."""
        result = strip_semantic_prefix("(label)text")
        assert result == "text"


# ── is_true_adversarial ──────────────────────────────────────────────

class TestIsTrueAdversarial:
    def test_ever_pattern(self):
        assert is_true_adversarial("Did Emma ever mention dogs?") is True

    def test_never_pattern(self):
        assert is_true_adversarial("Did Emma never own a dog?") is True

    def test_deny_pattern(self):
        assert is_true_adversarial("Did she deny the claim?") is True

    def test_at_any_point(self):
        assert is_true_adversarial("At any point did he say that?") is True

    def test_contradict(self):
        assert is_true_adversarial("Does this contradict what was said?") is True

    def test_normal_question_not_adversarial(self):
        assert is_true_adversarial("What did Emma say about her schedule?") is False

    def test_simple_factual(self):
        assert is_true_adversarial("When is the project deadline?") is False

    def test_empty_string(self):
        assert is_true_adversarial("") is False


# ── check_abstention (production MCP path) ───────────────────────────

class TestCheckAbstention:
    def test_returns_tuple(self):
        should_abstain, answer, confidence = check_abstention("test?", [])
        assert should_abstain is True
        assert answer == ABSTENTION_ANSWER
        assert confidence == 0.0

    def test_with_good_hits(self):
        hits = [_hit("Emma mentioned dogs", score=8.0)]
        should_abstain, answer, confidence = check_abstention("Did Emma mention dogs?", hits)
        assert isinstance(should_abstain, bool)
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0

    def test_custom_threshold(self):
        hits = [_hit("Emma mentioned dogs", score=8.0)]
        should_abstain, _, _ = check_abstention("Did Emma mention dogs?", hits, threshold=0.99)
        assert should_abstain is True

    def test_zero_threshold_never_abstains(self):
        hits = [_hit("unrelated text", score=1.0, speaker="Other")]
        should_abstain, _, _ = check_abstention("Did Emma mention dogs?", hits, threshold=0.0)
        assert should_abstain is False


# ── pack_evidence dispatch routing ───────────────────────────────────

class TestPackEvidenceRouting:
    def test_empty_hits(self):
        assert pack_evidence([]) == ""

    def test_default_structured(self):
        hits = [_hit("Emma said hello")]
        result = pack_evidence(hits)
        assert "[SPEAKER=Emma]" in result
        assert "Emma said hello" in result

    def test_routes_temporal(self):
        hits = [
            _hit("Event B", dia_id="D2:1"),
            _hit("Event A", dia_id="D1:1"),
        ]
        result = pack_evidence(hits, query_type="temporal")
        # Chronological: D1:1 should appear before D2:1
        assert result.index("[DiaID=D1:1]") < result.index("[DiaID=D2:1]")

    def test_routes_multihop(self):
        hits = [
            _hit("Alice said X", speaker="Alice"),
            _hit("Bob said Y", speaker="Bob"),
        ]
        result = pack_evidence(hits, query_type="multi-hop")
        assert "Alice" in result and "Bob" in result

    def test_routes_adversarial_true(self):
        hits = [_hit("Emma never mentioned dogs")]
        result = pack_evidence(hits, question="Did Emma ever mention dogs?", query_type="adversarial")
        assert "EVIDENCE_FOUND:" in result
        assert "DENIAL_EVIDENCE:" in result

    def test_adversarial_non_true_falls_to_structured(self):
        """query_type=adversarial but is_true_adversarial=False → _pack_structured."""
        hits = [_hit("Emma mentioned dogs")]
        result = pack_evidence(hits, question="What did Emma say?", query_type="adversarial")
        # Structured format: no EVIDENCE_FOUND header
        assert "EVIDENCE_FOUND:" not in result
        assert "[SPEAKER=Emma]" in result

    def test_unknown_query_type_uses_structured(self):
        hits = [_hit("some text")]
        result = pack_evidence(hits, query_type="unknown_type")
        assert "[SPEAKER=Emma]" in result


# ── _pack_structured ─────────────────────────────────────────────────

class TestPackStructured:
    def test_includes_metadata_tags(self):
        hits = [_hit("Hello world", speaker="Alice", dia_id="D3:7", date="2024-06-15")]
        result = pack_evidence(hits)
        assert "[SPEAKER=Alice]" in result
        assert "[DATE=2024-06-15]" in result
        assert "[DiaID=D3:7]" in result

    def test_strips_semantic_prefix(self):
        hits = [_hit("(identity desc) Real content")]
        result = pack_evidence(hits)
        assert "(identity desc)" not in result
        assert "Real content" in result

    def test_respects_max_chars(self):
        long_text = "x" * 3000
        hits = [_hit(long_text, score=9.0), _hit(long_text, score=8.0)]
        result = pack_evidence(hits, max_chars=100)
        # Only one hit should fit (or none if first line > 100)
        lines = [ln for ln in result.split("\n") if ln.strip()]
        assert len(lines) <= 1

    def test_empty_excerpt_skipped(self):
        hits = [_hit(""), _hit("real text")]
        result = pack_evidence(hits)
        assert "real text" in result
        lines = [ln for ln in result.split("\n") if ln.strip()]
        assert len(lines) == 1

    def test_missing_speaker_shows_unknown(self):
        hits = [{"excerpt": "text", "score": 5.0}]
        result = pack_evidence(hits)
        assert "[SPEAKER=UNKNOWN]" in result


# ── _pack_temporal ───────────────────────────────────────────────────

class TestPackTemporal:
    def test_sorts_by_dia_id(self):
        hits = [
            _hit("Third", dia_id="D3:1"),
            _hit("First", dia_id="D1:1"),
            _hit("Second", dia_id="D2:5"),
        ]
        result = pack_evidence(hits, query_type="temporal")
        lines = result.split("\n")
        assert "First" in lines[0]
        assert "Second" in lines[1]
        assert "Third" in lines[2]

    def test_missing_dia_id_sorted_last(self):
        hits = [
            _hit("No ID", dia_id=""),
            _hit("Has ID", dia_id="D1:1"),
        ]
        result = pack_evidence(hits, query_type="temporal")
        lines = result.split("\n")
        assert "Has ID" in lines[0]
        assert "No ID" in lines[1]

    def test_same_session_sorted_by_turn(self):
        hits = [
            _hit("Turn 10", dia_id="D1:10"),
            _hit("Turn 2", dia_id="D1:2"),
        ]
        result = pack_evidence(hits, query_type="temporal")
        lines = result.split("\n")
        assert "Turn 2" in lines[0]
        assert "Turn 10" in lines[1]


# ── _pack_multihop ───────────────────────────────────────────────────

class TestPackMultihop:
    def test_interleaves_speakers(self):
        hits = [
            _hit("A1", speaker="Alice"),
            _hit("A2", speaker="Alice"),
            _hit("B1", speaker="Bob"),
            _hit("B2", speaker="Bob"),
        ]
        result = pack_evidence(hits, query_type="multi-hop")
        lines = result.split("\n")
        # Round-robin: should alternate speakers
        speakers = []
        for line in lines:
            if "Alice" in line:
                speakers.append("Alice")
            elif "Bob" in line:
                speakers.append("Bob")
        # First two should be different speakers
        assert len(speakers) >= 2
        assert speakers[0] != speakers[1]

    def test_empty_hits(self):
        assert pack_evidence([], query_type="multi-hop") == ""

    def test_all_empty_excerpts(self):
        hits = [_hit("", speaker="Emma"), _hit("", speaker="Bob")]
        result = pack_evidence(hits, query_type="multi-hop")
        assert result == ""

    def test_single_speaker(self):
        hits = [_hit("A1", speaker="Alice"), _hit("A2", speaker="Alice")]
        result = pack_evidence(hits, query_type="multi-hop")
        assert result.count("Alice") == 2


# ── _pack_adversarial ────────────────────────────────────────────────

class TestPackAdversarial:
    def test_separates_denial_evidence(self):
        hits = [
            _hit("Emma never mentioned dogs"),
            _hit("Emma loves animals"),
        ]
        result = pack_evidence(hits, question="Did Emma ever mention dogs?", query_type="adversarial")
        denial_section = result.split("DENIAL_EVIDENCE:")[1]
        evidence_section = result.split("DENIAL_EVIDENCE:")[0].split("EVIDENCE:")[1]
        assert "never mentioned" in denial_section
        assert "loves animals" in evidence_section

    def test_no_hits_shows_no_evidence(self):
        result = pack_evidence([], question="Did Emma ever mention dogs?", query_type="adversarial")
        assert "EVIDENCE_FOUND: NO" in result
        assert "- (none)" in result

    def test_evidence_found_header(self):
        hits = [_hit("Emma mentioned dogs")]
        result = pack_evidence(hits, question="Did Emma ever mention dogs?", query_type="adversarial")
        assert "EVIDENCE_FOUND: YES" in result

    def test_all_denials_no_positive_evidence(self):
        hits = [
            _hit("Emma didn't mention dogs"),
            _hit("She never said anything about pets"),
        ]
        result = pack_evidence(hits, question="Did Emma ever mention dogs?", query_type="adversarial")
        evidence_section = result.split("DENIAL_EVIDENCE:")[0]
        assert "- (none)" in evidence_section

    def test_overlap_ordering(self):
        """Hits with more query term overlap should appear first."""
        hits = [
            _hit("Unrelated budget review", score=9.0),
            _hit("Emma mentioned adopting dogs from shelter", score=3.0),
        ]
        result = pack_evidence(
            hits, question="Did Emma ever mention adopting dogs?", query_type="adversarial"
        )
        lines = [ln for ln in result.split("\n") if ln.startswith("- ")]
        # The more relevant hit should appear first
        if len(lines) >= 2:
            assert "adopting" in lines[0] or "dogs" in lines[0]
