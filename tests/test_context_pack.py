#!/usr/bin/env python3
"""Tests for context_pack rules: adjacency, diversity, pronoun rescue."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from recall import (
    context_pack,
)


def _make_block(dia_id, speaker, text, block_id=None):
    """Create a minimal parsed block for testing."""
    if block_id is None:
        block_id = f"DIA-{dia_id.replace(':', '-')}"
    return {
        "_id": block_id,
        "Statement": f"[{speaker}] {text}",
        "Tags": f"session-1, {speaker}",
        "Status": "active",
        "DiaID": dia_id,
        "_source_file": "decisions/DECISIONS.md",
        "_line": 10,
    }


def _make_result(dia_id, speaker, text, score=10.0, block_id=None):
    """Create a minimal result dict for testing."""
    if block_id is None:
        block_id = f"DIA-{dia_id.replace(':', '-')}"
    return {
        "_id": block_id,
        "type": "unknown",
        "score": score,
        "excerpt": f"[{speaker}] {text}",
        "speaker": speaker,
        "tags": f"session-1, {speaker}",
        "file": "decisions/DECISIONS.md",
        "line": 10,
        "status": "active",
        "DiaID": dia_id,
    }


class TestRule1DialogAdjacency:
    """Rule 1: Question turns pull next 1-2 answer turns."""

    def _dialog(self):
        """3-turn dialog: question → answer → followup."""
        return [
            _make_block("D1:1", "Alice", "What's your favorite color?"),
            _make_block("D1:2", "Bob", "I really love blue, it reminds me of the ocean."),
            _make_block("D1:3", "Alice", "That's nice! I prefer green myself."),
        ]

    def test_question_mark_triggers_adjacency(self):
        blocks = self._dialog()
        top = [_make_result("D1:1", "Alice", "What's your favorite color?", score=20.0)]
        result = context_pack("favorite color", top, blocks, top, limit=10)
        dias = {r["DiaID"] for r in result}
        assert "D1:2" in dias, "Answer turn D1:2 should be pulled via adjacency"

    def test_adjacency_pulls_up_to_2_turns(self):
        blocks = self._dialog()
        top = [_make_result("D1:1", "Alice", "What's your favorite color?", score=20.0)]
        result = context_pack("favorite color", top, blocks, top, limit=10)
        dias = {r["DiaID"] for r in result}
        assert "D1:2" in dias
        assert "D1:3" in dias

    def test_non_question_no_adjacency(self):
        blocks = self._dialog()
        top = [_make_result("D1:2", "Bob", "I really love blue, it reminds me of the ocean.", score=20.0)]
        result = context_pack("blue ocean", top, blocks, top, limit=10)
        assert len(result) == 1, "Non-question turn should not trigger adjacency"

    def test_question_cue_triggers(self):
        blocks = [
            _make_block("D1:1", "Alice", "Any tips on studying or time management"),
            _make_block("D1:2", "Bob", "Break tasks into smaller pieces and set goals."),
        ]
        top = [_make_result("D1:1", "Alice", "Any tips on studying or time management", score=20.0)]
        result = context_pack("study tips", top, blocks, top, limit=10)
        dias = {r["DiaID"] for r in result}
        assert "D1:2" in dias

    def test_fact_cards_not_expanded(self):
        blocks = [
            _make_block("D1:1", "Alice", "What's your hobby?"),
            _make_block("D1:2", "Bob", "I love painting"),
        ]
        # Fact card result (not DIA- prefix)
        fact_result = _make_result("D1:1", "Alice", "What's your hobby?",
                                   score=20.0, block_id="FACT-001")
        result = context_pack("hobby", [fact_result], blocks, [fact_result], limit=10)
        assert len(result) == 1, "Fact cards should not trigger adjacency"

    def test_adjacency_marked(self):
        blocks = self._dialog()
        top = [_make_result("D1:1", "Alice", "What's your favorite color?", score=20.0)]
        result = context_pack("favorite color", top, blocks, top, limit=10)
        adjacency_hits = [r for r in result if r.get("via_adjacency")]
        assert len(adjacency_hits) >= 1
        assert adjacency_hits[0]["DiaID"] == "D1:2"


class TestRule2MultiEntityDiversity:
    """Rule 2: Multi-entity/plural questions enforce speaker/session diversity."""

    def _two_speaker_blocks(self):
        return [
            _make_block("D1:1", "Alice", "I had a health scare last week."),
            _make_block("D1:2", "Bob", "I also had a health scare recently."),
            _make_block("D2:1", "Alice", "My scare was gastritis."),
            _make_block("D2:2", "Bob", "Mine was a heart palpitation."),
        ]

    def test_multi_entity_detects_and_query(self):
        blocks = self._two_speaker_blocks()
        # Only Alice's results in top
        top = [_make_result("D1:1", "Alice", "I had a health scare last week.", score=20.0)]
        wider = [
            top[0],
            _make_result("D1:2", "Bob", "I also had a health scare recently.", score=15.0),
        ]
        result = context_pack(
            "What health scares did Alice and Bob experience?",
            top, blocks, wider, limit=10,
        )
        speakers = {r.get("speaker", "").lower() for r in result}
        assert "bob" in speakers, "Diversity should pull Bob's block"

    def test_plural_cue_triggers_diversity(self):
        blocks = self._two_speaker_blocks()
        top = [_make_result("D1:1", "Alice", "I had a health scare last week.", score=20.0)]
        wider = [
            top[0],
            _make_result("D2:2", "Bob", "Mine was a heart palpitation.", score=12.0),
        ]
        result = context_pack(
            "What health scares happened?",
            top, blocks, wider, limit=10,
        )
        # "scares" is a plural cue, should try to diversify
        assert len(result) >= 2

    def test_no_diversity_when_already_diverse(self):
        blocks = self._two_speaker_blocks()
        top = [
            _make_result("D1:1", "Alice", "I had a health scare.", score=20.0),
            _make_result("D1:2", "Bob", "I also had a scare.", score=18.0),
        ]
        result = context_pack(
            "What health scares did Alice and Bob experience?",
            top, blocks, top, limit=10,
        )
        # Already have both speakers, no extra blocks needed
        diversity_hits = [r for r in result if r.get("via_diversity")]
        assert len(diversity_hits) == 0

    def test_diversity_marked(self):
        blocks = self._two_speaker_blocks()
        top = [_make_result("D1:1", "Alice", "I had a health scare.", score=20.0)]
        wider = [
            top[0],
            _make_result("D1:2", "Bob", "I also had a scare.", score=15.0),
        ]
        result = context_pack(
            "What health scares did Alice and Bob experience?",
            top, blocks, wider, limit=10,
        )
        diversity_hits = [r for r in result if r.get("via_diversity")]
        assert len(diversity_hits) >= 1


class TestRule3PronounRescue:
    """Rule 3: Pronoun-heavy hits trigger neighbor search for explicit nouns."""

    def _pronoun_dialog(self):
        return [
            _make_block("D2:22", "Alice", "I have a pet snake named Seraphim."),
            _make_block("D2:23", "Bob", "That's cool! When did you get it?"),
            _make_block("D2:24", "Alice", "I bought it a year ago in Paris."),
            _make_block("D2:25", "Bob", "It must be fun having a pet like that."),
        ]

    def test_pronoun_rescue_finds_noun(self):
        blocks = self._pronoun_dialog()
        # Top hit uses "it" without "snake"
        top = [_make_result("D2:24", "Alice", "I bought it a year ago in Paris.", score=20.0)]
        result = context_pack(
            "When did Alice buy her pet snake?",
            top, blocks, top, limit=10,
        )
        # Should rescue D2:22 which has "snake"
        dias = {r["DiaID"] for r in result}
        assert "D2:22" in dias, "Should rescue the turn that mentions 'snake'"

    def test_no_rescue_when_noun_present(self):
        blocks = self._pronoun_dialog()
        # Top hit already has the noun
        top = [_make_result("D2:22", "Alice", "I have a pet snake named Seraphim.", score=20.0)]
        result = context_pack(
            "When did Alice buy her pet snake?",
            top, blocks, top, limit=10,
        )
        rescue_hits = [r for r in result if r.get("via_pronoun_rescue")]
        assert len(rescue_hits) == 0, "No rescue needed when noun is already in hit"

    def test_rescue_marked(self):
        blocks = self._pronoun_dialog()
        top = [_make_result("D2:24", "Alice", "I bought it a year ago in Paris.", score=20.0)]
        result = context_pack(
            "When did Alice buy her pet snake?",
            top, blocks, top, limit=10,
        )
        rescue_hits = [r for r in result if r.get("via_pronoun_rescue")]
        assert len(rescue_hits) >= 1


class TestContextPackEdgeCases:
    """Edge cases and empty inputs."""

    def test_empty_results(self):
        assert context_pack("test", [], [], [], 10) == []

    def test_no_blocks(self):
        top = [_make_result("D1:1", "Alice", "test", score=10.0)]
        result = context_pack("test", top, [], top, 10)
        assert len(result) == 1

    def test_combined_rules(self):
        """All three rules can fire on the same query."""
        blocks = [
            _make_block("D1:1", "Alice", "What's your favorite snake?"),
            _make_block("D1:2", "Bob", "I bought it a year ago in Paris."),
            _make_block("D1:3", "Bob", "It's a python named Monty."),
            _make_block("D2:1", "Alice", "I also have a pet snake!"),
        ]
        # Question turn + pronoun hit
        top = [_make_result("D1:1", "Alice", "What's your favorite snake?", score=20.0)]
        wider = [
            top[0],
            _make_result("D2:1", "Alice", "I also have a pet snake!", score=8.0),
        ]
        result = context_pack(
            "What snakes do Alice and Bob have?",
            top, blocks, wider, limit=10,
        )
        # Should have adjacency (D1:2, D1:3) + diversity attempts
        assert len(result) >= 3
