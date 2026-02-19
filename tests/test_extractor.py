#!/usr/bin/env python3
"""Tests for the regex NER-lite entity/fact extractor."""

import os
import sys

# Add scripts/ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from extractor import extract_facts, extract_from_conversation, format_as_blocks


class TestExtractFacts:
    """Test atomic fact extraction from text."""

    def test_identity_extraction(self):
        cards = extract_facts("I'm a transgender woman living in the city", speaker="Caroline")
        assert any("transgender woman" in c["content"] for c in cards)

    def test_identity_filters_emotions(self):
        cards = extract_facts("I'm so happy about this", speaker="Caroline")
        identity_cards = [c for c in cards if c["type"] == "FACT" and "is " in c["content"]]
        assert not identity_cards, "Emotional states should be filtered"

    def test_identity_as_pattern(self):
        cards = extract_facts("My journey as a transgender woman has been challenging", speaker="Caroline")
        assert any("transgender woman" in c["content"] for c in cards)

    def test_event_extraction(self):
        cards = extract_facts("I went to a LGBTQ support group yesterday", speaker="Caroline")
        events = [c for c in cards if c["type"] == "EVENT"]
        assert len(events) >= 1
        assert any("support group" in c["content"] for c in events)

    def test_gerund_event(self):
        cards = extract_facts("Researching adoption agencies right now", speaker="Caroline")
        events = [c for c in cards if c["type"] == "EVENT"]
        assert len(events) >= 1
        assert any("adoption" in c["content"] for c in events)

    def test_preference_extraction(self):
        cards = extract_facts("I love painting landscapes", speaker="Melanie")
        prefs = [c for c in cards if c["type"] == "PREFERENCE"]
        assert len(prefs) >= 1
        assert any("painting" in c["content"] for c in prefs)

    def test_favorite_extraction(self):
        cards = extract_facts("My favorite color is blue", speaker="Melanie")
        prefs = [c for c in cards if c["type"] == "PREFERENCE"]
        assert any("blue" in c["content"] for c in prefs)

    def test_dislike_extraction(self):
        cards = extract_facts("I hate cold weather", speaker="Melanie")
        prefs = [c for c in cards if c["type"] == "PREFERENCE"]
        assert any("cold weather" in c["content"] for c in prefs)

    def test_negation_extraction(self):
        cards = extract_facts("I never learned to swim", speaker="Caroline")
        negations = [c for c in cards if c["type"] == "NEGATION"]
        assert len(negations) >= 1
        assert any("swim" in c["content"] for c in negations)

    def test_plan_extraction(self):
        cards = extract_facts("I want to become a counselor someday", speaker="Caroline")
        plans = [c for c in cards if c["type"] == "PLAN"]
        assert len(plans) >= 1
        assert any("counselor" in c["content"] for c in plans)

    def test_relation_extraction(self):
        cards = extract_facts("Sarah is my sister", speaker="Caroline")
        rels = [c for c in cards if c["type"] == "RELATION"]
        assert len(rels) >= 1
        assert any("sister" in c["content"] for c in rels)

    def test_speaker_prefix(self):
        cards = extract_facts("I went to the park", speaker="Caroline")
        assert all(c["content"].startswith("Caroline ") for c in cards if c["content"])

    def test_speaker_from_bracket(self):
        cards = extract_facts("[Caroline] I went to the park")
        assert all(c["speaker"] == "Caroline" for c in cards)

    def test_date_passthrough(self):
        cards = extract_facts("I went to the park", speaker="X", date="2023-05-07")
        assert all(c["date"] == "2023-05-07" for c in cards)

    def test_source_id_passthrough(self):
        cards = extract_facts("I went to the park", speaker="X", source_id="DIA-D1-3")
        assert all(c["source_id"] == "DIA-D1-3" for c in cards)

    def test_deduplication(self):
        # Same fact repeated shouldn't produce duplicates
        cards = extract_facts("I love painting. I really love painting.", speaker="M")
        contents = [c["content"].lower() for c in cards]
        # Count "painting"-related cards
        painting_cards = [c for c in contents if "painting" in c]
        assert len(painting_cards) <= 2  # At most one preference + one event/etc

    def test_content_truncation(self):
        long_text = "I visited the really incredibly amazingly beautiful " + "wonderful " * 30 + "park"
        cards = extract_facts(long_text, speaker="X")
        for c in cards:
            assert len(c["content"]) <= 140  # 120 + speaker prefix

    def test_activity_list(self):
        cards = extract_facts("I enjoy running, reading, or playing my violin", speaker="Melanie")
        prefs = [c for c in cards if c["type"] == "PREFERENCE"]
        contents = " ".join(c["content"] for c in prefs)
        assert "violin" in contents or "playing" in contents

    def test_empty_text(self):
        cards = extract_facts("")
        assert cards == []

    def test_no_speaker(self):
        cards = extract_facts("I went to the park")
        events = [c for c in cards if c["type"] == "EVENT"]
        assert len(events) >= 1
        assert events[0]["speaker"] == ""


class TestFormatAsBlocks:
    """Test block formatting for DECISIONS.md output."""

    def test_basic_format(self):
        cards = [{"type": "FACT", "content": "Caroline is a counselor",
                  "speaker": "Caroline", "date": "2023-05-07",
                  "source_id": "DIA-D1-3", "confidence": 0.85,
                  "dia_id": "D1:3"}]
        text = format_as_blocks(cards)
        assert "[FACT-001]" in text
        # Find the Statement line containing our content (not just first one)
        stmt_lines = [ln for ln in text.splitlines()
                      if ln.startswith("Statement:") and "Caroline is a counselor" in ln]
        assert len(stmt_lines) == 1, f"Expected 1 match, got {len(stmt_lines)}"
        stmt = stmt_lines[0]
        # Semantic prefix: starts with "(" and content follows after ") "
        after_key = stmt.split("Statement: ", 1)[1]
        assert after_key.startswith("("), "Missing semantic prefix"
        assert ") " in after_key, "Malformed prefix (no closing paren+space)"
        # Content is preserved after the prefix
        prefix_end = after_key.index(") ") + 2
        assert after_key[prefix_end:] == "Caroline is a counselor"
        # Standard fields
        assert "Date: 2023-05-07" in text
        assert "Status: active" in text
        assert "DiaID: D1:3" in text
        assert "Sources: DIA-D1-3" in text

    def test_custom_prefix(self):
        cards = [{"type": "EVENT", "content": "X visited Y",
                  "speaker": "", "date": "", "source_id": "",
                  "confidence": 0.8}]
        text = format_as_blocks(cards, id_prefix="EVT")
        assert "[EVT-001]" in text

    def test_counter_start(self):
        cards = [{"type": "FACT", "content": "test",
                  "speaker": "", "date": "", "source_id": "",
                  "confidence": 0.8}]
        text = format_as_blocks(cards, counter_start=42)
        assert "[FACT-042]" in text

    def test_no_dia_id(self):
        cards = [{"type": "FACT", "content": "test",
                  "speaker": "", "date": "", "source_id": "",
                  "confidence": 0.8}]
        text = format_as_blocks(cards)
        assert "DiaID:" not in text

    def test_semantic_prefix_all_types(self):
        """Each card type gets a distinct semantic prefix."""
        for card_type in ("FACT", "EVENT", "PREFERENCE", "RELATION"):
            cards = [{"type": card_type, "content": "X",
                      "speaker": "", "date": "", "source_id": "",
                      "confidence": 0.8}]
            text = format_as_blocks(cards)
            stmt = [ln for ln in text.splitlines() if ln.startswith("Statement:")][0]
            assert stmt.startswith("Statement: ("), f"{card_type} missing prefix"
            assert "X" in stmt

    def test_semantic_prefix_idempotent(self):
        """Re-formatting already-prefixed content must not double-prepend."""
        cards = [{"type": "FACT",
                  "content": "(identity description who is) Already prefixed",
                  "speaker": "", "date": "", "source_id": "",
                  "confidence": 0.8}]
        text = format_as_blocks(cards)
        # Select by content match, not index
        stmt_lines = [ln for ln in text.splitlines()
                      if ln.startswith("Statement:") and "Already prefixed" in ln]
        assert len(stmt_lines) == 1
        after_key = stmt_lines[0].split("Statement: ", 1)[1]
        # Structural check: exactly one leading prefix, not label-specific
        assert after_key.startswith("("), "Missing prefix"
        prefix_end = after_key.index(") ") + 2
        remainder = after_key[prefix_end:]
        # After the first prefix closes, content must NOT start with another prefix
        assert not remainder.startswith("("), \
            f"Double-prepend detected: {stmt_lines[0]}"

    def test_double_format_stability(self):
        """Formatting output twice should produce identical statements."""
        cards = [{"type": "EVENT", "content": "John visited the park",
                  "speaker": "John", "date": "2023-06-01",
                  "source_id": "DIA-D2-5", "confidence": 0.9}]
        text1 = format_as_blocks(cards)
        # Select by content match
        stmt1 = [ln for ln in text1.splitlines()
                 if ln.startswith("Statement:") and "visited the park" in ln][0]
        content_with_prefix = stmt1.split("Statement: ", 1)[1]
        # Feed it back through as content
        cards2 = [{"type": "EVENT", "content": content_with_prefix,
                   "speaker": "John", "date": "2023-06-01",
                   "source_id": "DIA-D2-5", "confidence": 0.9}]
        text2 = format_as_blocks(cards2)
        stmt2 = [ln for ln in text2.splitlines()
                 if ln.startswith("Statement:") and "visited the park" in ln][0]
        assert stmt1 == stmt2, f"Not idempotent:\n  pass1: {stmt1}\n  pass2: {stmt2}"


class TestExtractFromConversation:
    """Test conversation-level extraction."""

    def test_basic_conversation(self):
        turns = [
            {"speaker": "speaker_a", "dia_id": "D1:1", "text": "I went to the park today."},
            {"speaker": "speaker_b", "dia_id": "D1:2", "text": "I love hiking in the mountains."},
        ]
        cards = extract_from_conversation(turns, speaker_a="Alice", speaker_b="Bob")
        assert len(cards) >= 2
        assert any(c["speaker"] == "Alice" for c in cards)
        assert any(c["speaker"] == "Bob" for c in cards)

    def test_dia_id_carried(self):
        turns = [
            {"speaker": "speaker_a", "dia_id": "D1:5", "text": "I'm a software engineer."},
        ]
        cards = extract_from_conversation(turns, speaker_a="Alice", speaker_b="Bob")
        assert all(c.get("dia_id") == "D1:5" for c in cards)

    def test_source_id_format(self):
        turns = [
            {"speaker": "speaker_a", "dia_id": "D1:5", "text": "I went to the store."},
        ]
        cards = extract_from_conversation(turns, speaker_a="Alice", speaker_b="Bob")
        assert all(c["source_id"] == "DIA-D1-5" for c in cards)

    def test_empty_turns(self):
        cards = extract_from_conversation([], speaker_a="A", speaker_b="B")
        assert cards == []

    def test_coreference_resolution(self):
        """D.3: He/she → other speaker within conversation."""
        turns = [
            {"speaker": "speaker_a", "dia_id": "D1:1", "text": "He is a great doctor."},
        ]
        cards = extract_from_conversation(turns, speaker_a="Alice", speaker_b="Bob")
        # "He" should resolve to Bob (the other speaker)
        assert any("Bob" in c["content"] for c in cards)


class TestD3Patterns:
    """Test Phase D.3 extractor patterns."""

    def test_possessive_relation(self):
        cards = extract_facts("Tim's brother is really cool", speaker="John")
        rels = [c for c in cards if c["type"] == "RELATION"]
        assert any("Tim's brother" in c["content"] for c in rels)

    def test_possessive_fact(self):
        cards = extract_facts("Tim's car broke down yesterday", speaker="John")
        facts = [c for c in cards if "Tim's car" in c["content"]]
        assert len(facts) >= 1

    def test_habitual_preference(self):
        cards = extract_facts("I usually go surfing on weekends", speaker="Tim")
        prefs = [c for c in cards if c["type"] == "PREFERENCE"]
        assert any("surfing" in c["content"] for c in prefs)

    def test_habitual_often(self):
        cards = extract_facts("I often play basketball after work", speaker="Tim")
        prefs = [c for c in cards if c["type"] == "PREFERENCE"]
        assert any("basketball" in c["content"] for c in prefs)

    def test_temporal_normalization(self):
        cards = extract_facts("I visited Paris in March 2023", speaker="Tim")
        events = [c for c in cards if c["type"] == "EVENT"]
        assert any(c.get("date", "").startswith("2023-03") for c in events)

    def test_third_person_fact(self):
        cards = extract_facts("He is a software engineer at Google", speaker="Tim")
        facts = [c for c in cards if "software engineer" in c.get("content", "")]
        assert len(facts) >= 1
