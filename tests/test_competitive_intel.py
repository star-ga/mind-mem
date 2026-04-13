# Copyright 2026 STARGA, Inc.
"""Tests for cascading staleness + project profiles (v2.6.0)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from mind_mem.project_profile import ProjectProfile, build_profile
from mind_mem.staleness import StalenessPlan, propagate_staleness


# ---------------------------------------------------------------------------
# Staleness propagation
# ---------------------------------------------------------------------------


class TestStaleness:
    def test_seed_scored_at_one(self) -> None:
        plan = propagate_staleness(["A"], {"A": ["B"], "B": []})
        assert plan.scores["A"] == 1.0
        assert plan.scores["B"] == 0.9

    def test_three_hop_decay(self) -> None:
        adjacency = {
            "A": ["B"],
            "B": ["C"],
            "C": ["D"],
            "D": [],
        }
        plan = propagate_staleness(["A"], adjacency)
        assert plan.scores["A"] == 1.0
        assert plan.scores["B"] == 0.9
        assert plan.scores["C"] == 0.5
        assert plan.scores["D"] == 0.2

    def test_beyond_decay_length_not_scored(self) -> None:
        adjacency = {
            "A": ["B"],
            "B": ["C"],
            "C": ["D"],
            "D": ["E"],
            "E": [],
        }
        plan = propagate_staleness(["A"], adjacency)
        # Default decay has length 4 → 3 hops max. E is at 4 hops.
        assert "E" not in plan.scores

    def test_closer_seed_wins_over_farther(self) -> None:
        # Diamond: A → B, C → B. If A is a distant seed but C is
        # closer, B should take the closer (higher) score.
        adjacency = {
            "A": ["X"],
            "X": ["B"],
            "C": ["B"],
            "B": [],
        }
        plan = propagate_staleness(["A", "C"], adjacency)
        # B can be reached from A in 2 hops (decay 0.5) or from C in
        # 1 hop (decay 0.9). Must keep the higher score.
        assert plan.scores["B"] == 0.9

    def test_cycle_does_not_loop(self) -> None:
        adjacency = {"A": ["B"], "B": ["A"]}
        plan = propagate_staleness(["A"], adjacency)
        assert set(plan.scores.keys()) == {"A", "B"}

    def test_empty_seeds_empty_scores(self) -> None:
        plan = propagate_staleness([], {"A": ["B"]})
        assert plan.scores == {}

    def test_disconnected_block_not_scored(self) -> None:
        adjacency = {"A": ["B"], "B": [], "X": []}
        plan = propagate_staleness(["A"], adjacency)
        assert "X" not in plan.scores

    def test_custom_decay(self) -> None:
        plan = propagate_staleness(
            ["A"], {"A": ["B"], "B": []}, decay=(1.0, 0.01)
        )
        assert plan.scores["B"] == pytest.approx(0.01)

    def test_invalid_decay_rejected(self) -> None:
        with pytest.raises(ValueError):
            propagate_staleness(["A"], {"A": []}, decay=(1.0, 2.0))

    def test_max_hops_caps_traversal(self) -> None:
        adjacency = {"A": ["B"], "B": ["C"], "C": []}
        plan = propagate_staleness(["A"], adjacency, max_hops=1)
        assert "C" not in plan.scores

    def test_flagged_by_threshold(self) -> None:
        plan = propagate_staleness(
            ["A"], {"A": ["B"], "B": ["C"], "C": []}
        )
        assert plan.flagged(0.6) == ["A", "B"]
        # Threshold at 1.0 only returns the seed.
        assert plan.flagged(1.0) == ["A"]


# ---------------------------------------------------------------------------
# Project profiles
# ---------------------------------------------------------------------------


def _now() -> datetime:
    return datetime(2026, 4, 13, 12, 0, 0, tzinfo=timezone.utc)


class TestBuildProfile:
    def test_empty_input_empty_profile(self) -> None:
        prof = build_profile([], name="demo", now=_now())
        assert prof.total_blocks == 0
        assert prof.block_types == {}
        assert prof.top_concepts == []

    def test_block_type_histogram(self) -> None:
        blocks = [
            {"type": "decision"},
            {"type": "decision"},
            {"type": "task"},
        ]
        prof = build_profile(blocks, name="x", now=_now())
        assert prof.block_types == {"decision": 2, "task": 1}

    def test_file_frequency(self) -> None:
        blocks = [
            {"file": "decisions/2026-04.md"},
            {"file": "decisions/2026-04.md"},
            {"file": "tasks/list.md"},
        ]
        prof = build_profile(blocks, name="x", top_k=5, now=_now())
        assert prof.top_files[0] == "decisions/2026-04.md"

    def test_entity_frequency_via_entities_field(self) -> None:
        blocks = [
            {"entities": ["Alice", "Bob"]},
            {"entities": ["Alice"]},
        ]
        prof = build_profile(blocks, name="x", now=_now())
        assert prof.top_entities[0] == "Alice"

    def test_entity_frequency_via_mentions_field(self) -> None:
        blocks = [{"mentions": ["starga"]}]
        prof = build_profile(blocks, name="x", now=_now())
        assert "starga" in prof.top_entities

    def test_concepts_stopwords_stripped(self) -> None:
        blocks = [
            {"text": "the JWT token is for authentication and authorization"},
        ]
        prof = build_profile(blocks, name="x", top_k=10, now=_now())
        # Stopwords "the", "is", "for", "and" must not show up.
        for stop in ("the", "is", "for", "and"):
            assert stop not in prof.top_concepts
        # Real words do.
        assert "jwt" in prof.top_concepts
        assert "token" in prof.top_concepts

    def test_short_tokens_filtered(self) -> None:
        # Tokens under 3 characters get dropped; signal-to-noise.
        blocks = [{"text": "a be on go no ok jwt"}]
        prof = build_profile(blocks, name="x", top_k=10, now=_now())
        assert "jwt" in prof.top_concepts
        assert "a" not in prof.top_concepts
        assert "be" not in prof.top_concepts

    def test_recent_count_honours_window(self) -> None:
        blocks = [
            {"created_at": "2026-04-12T00:00:00Z"},
            {"created_at": "2026-04-05T00:00:00Z"},  # within 14-day window
            {"created_at": "2025-01-01T00:00:00Z"},  # old
        ]
        prof = build_profile(blocks, name="x", recent_window_days=14, now=_now())
        assert prof.recent_block_count == 2

    def test_invalid_top_k_rejected(self) -> None:
        with pytest.raises(ValueError):
            build_profile([], name="x", top_k=-1)

    def test_invalid_window_rejected(self) -> None:
        with pytest.raises(ValueError):
            build_profile([], name="x", recent_window_days=-1)

    def test_top_k_zero_returns_everything(self) -> None:
        blocks = [{"type": t} for t in ("decision", "task", "note", "fact")]
        prof = build_profile(blocks, name="x", top_k=0, now=_now())
        assert len(prof.block_types) == 4

    def test_malformed_block_ignored(self) -> None:
        prof = build_profile(
            ["not a dict", 42, None, {"type": "decision"}],  # type: ignore[list-item]
            name="x",
            now=_now(),
        )
        assert prof.total_blocks == 1
        assert prof.block_types == {"decision": 1}

    def test_as_dict_contains_all_fields(self) -> None:
        prof = build_profile([{"type": "decision"}], name="x", now=_now())
        d = prof.as_dict()
        assert set(d.keys()) >= {
            "name", "total_blocks", "block_types", "top_concepts",
            "top_files", "top_entities", "recent_block_count", "generated_at",
        }
