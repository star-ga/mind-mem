"""Tests for granularity_align — named merge operation (Group H, v4.0.x).

Covers:
- find_merge_candidates(): threshold semantics, determinism, cap, empty text.
- GranularityMergeCandidate.to_dict() shape.
- merge_blocks(): all three strategies.
- merge_blocks(): ValueError on unknown strategy.
- Tag merging and _merged_from audit field.
- find_merge_candidates(): exact-duplicate detection (sim ≈ 1.0).
- find_merge_candidates(): empty blocks list.
- find_merge_candidates(): blocks with no text content skipped.
- Suggested strategy heuristics.
"""

from __future__ import annotations

import pytest
from mind_mem.granularity_align import (
    DEFAULT_MIN_SIMILARITY,
    MERGE_STRATEGIES,
    find_merge_candidates,
    merge_blocks,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _blk(
    block_id: str,
    content: str = "",
    tags: str = "",
    maturity: float | None = None,
    status: str = "active",
) -> dict:
    b: dict = {"_id": block_id, "content": content, "tags": tags, "Status": status}
    if maturity is not None:
        b["Maturity"] = maturity
    return b


# ---------------------------------------------------------------------------
# find_merge_candidates — basic detection
# ---------------------------------------------------------------------------


def test_identical_blocks_detected() -> None:
    """Two blocks with identical content should be flagged as candidates."""
    text = "use Q16.16 fixed-point arithmetic for deterministic scoring"
    a = _blk("A-001", content=text)
    b = _blk("B-001", content=text)
    candidates = find_merge_candidates([a, b], min_similarity=0.75)
    assert len(candidates) == 1
    c = candidates[0]
    assert c.similarity >= 0.99
    # Pair is always (a, b) since A-001 < B-001 lexicographically in tie-break
    ids = {c.block_a["_id"], c.block_b["_id"]}
    assert ids == {"A-001", "B-001"}


def test_dissimilar_blocks_not_detected() -> None:
    """Unrelated blocks should not be returned."""
    a = _blk("A-001", content="the cat sat on the mat")
    b = _blk("B-001", content="quantum entanglement in photon pairs")
    candidates = find_merge_candidates([a, b], min_similarity=0.5)
    assert candidates == []


def test_threshold_semantics() -> None:
    """Raising the threshold reduces the candidate set."""
    text_a = "deterministic scoring with fixed-point arithmetic"
    text_b = "use fixed-point scoring for determinism across substrates"
    a = _blk("A-001", content=text_a)
    b = _blk("B-001", content=text_b)

    # Should find at moderate threshold
    at_low = find_merge_candidates([a, b], min_similarity=0.3)
    # At very high threshold may or may not find (depends on actual sim)
    at_high = find_merge_candidates([a, b], min_similarity=0.99)

    # Low threshold finds same or more than high threshold
    assert len(at_low) >= len(at_high)


def test_default_threshold_is_constant() -> None:
    assert DEFAULT_MIN_SIMILARITY == 0.75


def test_empty_block_list() -> None:
    candidates = find_merge_candidates([], min_similarity=0.5)
    assert candidates == []


def test_single_block_no_candidates() -> None:
    a = _blk("A-001", content="only one block")
    candidates = find_merge_candidates([a])
    assert candidates == []


def test_blocks_with_no_text_skipped() -> None:
    """Blocks whose combined text is empty produce no tokens and are skipped."""
    a = _blk("A-001", content="")  # empty
    b = _blk("B-001", content="some actual content here worth indexing")
    c = _blk("C-001", content="")  # also empty
    candidates = find_merge_candidates([a, b, c], min_similarity=0.01)
    # a and c have no tokens → no pair involving them
    assert candidates == []


def test_max_candidates_cap() -> None:
    """Candidate list is capped at max_candidates."""
    # Build 6 identical blocks → 15 pairs
    blocks = [_blk(f"X-{i:03d}", content="same content every time for testing") for i in range(6)]
    candidates_all = find_merge_candidates(blocks, min_similarity=0.5, max_candidates=0)
    candidates_capped = find_merge_candidates(blocks, min_similarity=0.5, max_candidates=5)
    assert len(candidates_capped) == 5
    assert len(candidates_all) > 5


def test_candidates_sorted_descending_similarity() -> None:
    """Returned candidates are sorted descending by similarity."""
    a = _blk("A", content="fixed-point arithmetic for determinism")
    b = _blk("B", content="fixed-point arithmetic for determinism and cross-substrate")
    c = _blk("C", content="completely unrelated memory about oceans and tides")
    d = _blk("D", content="completely unrelated oceans tides waves")
    candidates = find_merge_candidates([a, b, c, d], min_similarity=0.0)
    sims = [c_.similarity for c_ in candidates]
    assert sims == sorted(sims, reverse=True)


def test_determinism() -> None:
    """Same input always produces same output."""
    blocks = [
        _blk("A", content="deterministic scoring in Q16.16"),
        _blk("B", content="Q16.16 fixed-point scoring deterministic"),
        _blk("C", content="totally different memory about cats"),
    ]
    r1 = find_merge_candidates(blocks, min_similarity=0.3)
    r2 = find_merge_candidates(blocks, min_similarity=0.3)
    assert r1 == r2


# ---------------------------------------------------------------------------
# GranularityMergeCandidate.to_dict()
# ---------------------------------------------------------------------------


def test_to_dict_shape() -> None:
    a = _blk("A-001", content="shared content")
    b = _blk("B-001", content="shared content")
    candidates = find_merge_candidates([a, b])
    assert len(candidates) == 1
    d = candidates[0].to_dict()
    assert set(d.keys()) == {"id_a", "id_b", "similarity", "reason", "suggested_strategy"}
    assert isinstance(d["similarity"], float)
    assert isinstance(d["reason"], str) and d["reason"]
    assert d["suggested_strategy"] in MERGE_STRATEGIES


def test_reason_describes_similarity_level() -> None:
    text = "fixed-point arithmetic scoring"
    a = _blk("A", content=text)
    b = _blk("B", content=text)
    cands = find_merge_candidates([a, b])
    assert len(cands) == 1
    # Near-identical → should mention near-identical or high overlap
    assert "similar" in cands[0].reason.lower() or "identical" in cands[0].reason.lower() or "overlap" in cands[0].reason.lower()


# ---------------------------------------------------------------------------
# suggested_strategy heuristics
# ---------------------------------------------------------------------------


def test_suggested_strategy_keep_longer_when_no_maturity() -> None:
    text = "deterministic scoring block"
    a = _blk("A", content=text)
    b = _blk("B", content=text)
    cands = find_merge_candidates([a, b])
    assert cands[0].suggested_strategy == "keep_longer"


def test_suggested_strategy_keep_higher_maturity_when_maturity_present() -> None:
    text = "deterministic scoring block"
    a = _blk("A", content=text, maturity=0.9)
    b = _blk("B", content=text)
    cands = find_merge_candidates([a, b])
    assert cands[0].suggested_strategy == "keep_higher_maturity"


# ---------------------------------------------------------------------------
# merge_blocks — keep_longer
# ---------------------------------------------------------------------------


def test_merge_keep_longer_picks_longer_content() -> None:
    short = _blk("A", content="short text")
    long_ = _blk("B", content="much longer text with more detail about the topic")
    merged = merge_blocks(short, long_, strategy="keep_longer")
    # The winner is the longer block
    assert merged["_id"] == "B"
    assert merged["content"] == long_["content"]


def test_merge_keep_longer_same_length_picks_a() -> None:
    a = _blk("A", content="identical")
    b = _blk("B", content="identical")
    merged = merge_blocks(a, b, strategy="keep_longer")
    assert merged["_id"] == "A"


def test_merge_keep_longer_merged_from_contains_both_ids() -> None:
    a = _blk("A", content="some content here")
    b = _blk("B", content="some content here longer version")
    merged = merge_blocks(a, b, strategy="keep_longer")
    assert "_merged_from" in merged
    ids = set(merged["_merged_from"])
    assert ids == {"A", "B"}


def test_merge_keep_longer_tags_unioned() -> None:
    a = _blk("A", content="content", tags="alpha, beta")
    b = _blk("B", content="content with more words", tags="beta, gamma")
    merged = merge_blocks(a, b, strategy="keep_longer")
    tags = merged["tags"]
    assert "alpha" in tags
    assert "beta" in tags
    assert "gamma" in tags
    # beta deduped
    assert tags.count("beta") == 1


# ---------------------------------------------------------------------------
# merge_blocks — keep_higher_maturity
# ---------------------------------------------------------------------------


def test_merge_keep_higher_maturity_picks_higher() -> None:
    a = _blk("A", content="content", maturity=0.9)
    b = _blk("B", content="content longer version here", maturity=0.3)
    merged = merge_blocks(a, b, strategy="keep_higher_maturity")
    assert merged["_id"] == "A"


def test_merge_keep_higher_maturity_tie_uses_a() -> None:
    a = _blk("A", content="content", maturity=0.5)
    b = _blk("B", content="content", maturity=0.5)
    merged = merge_blocks(a, b, strategy="keep_higher_maturity")
    assert merged["_id"] == "A"


def test_merge_keep_higher_maturity_no_explicit_maturity_neutral() -> None:
    """When neither block has an explicit Maturity field, both score 0.5
    (neutral), and the tie resolves to keep a (>=)."""
    a = _blk("A", content="some content")
    b = _blk("B", content="some content")
    merged = merge_blocks(a, b, strategy="keep_higher_maturity")
    assert merged["_id"] == "A"


# ---------------------------------------------------------------------------
# merge_blocks — concatenate
# ---------------------------------------------------------------------------


def test_merge_concatenate_joins_content() -> None:
    a = _blk("A", content="first half of the memory")
    b = _blk("B", content="second half of the memory")
    merged = merge_blocks(a, b, strategy="concatenate")
    assert "first half" in merged["content"]
    assert "second half" in merged["content"]


def test_merge_concatenate_id_is_a() -> None:
    a = _blk("A", content="part one")
    b = _blk("B", content="part two")
    merged = merge_blocks(a, b, strategy="concatenate")
    assert merged["_id"] == "A"


def test_merge_concatenate_merged_from_recorded() -> None:
    a = _blk("A", content="part one")
    b = _blk("B", content="part two")
    merged = merge_blocks(a, b, strategy="concatenate")
    assert set(merged["_merged_from"]) == {"A", "B"}


def test_merge_concatenate_long_content_truncates_excerpt() -> None:
    long_content = "word " * 200  # 1000 chars
    a = _blk("A", content=long_content)
    b = _blk("B", content=long_content)
    merged = merge_blocks(a, b, strategy="concatenate")
    assert len(merged["excerpt"]) <= 500


# ---------------------------------------------------------------------------
# merge_blocks — error handling
# ---------------------------------------------------------------------------


def test_merge_unknown_strategy_raises() -> None:
    a = _blk("A", content="content")
    b = _blk("B", content="content")
    with pytest.raises(ValueError, match="strategy must be one of"):
        merge_blocks(a, b, strategy="invalid_strategy")


# ---------------------------------------------------------------------------
# MERGE_STRATEGIES constant
# ---------------------------------------------------------------------------


def test_merge_strategies_constant() -> None:
    assert MERGE_STRATEGIES == frozenset({"keep_longer", "keep_higher_maturity", "concatenate"})


# ---------------------------------------------------------------------------
# End-to-end: detect candidate, merge, check audit field
# ---------------------------------------------------------------------------


def test_end_to_end_detect_and_merge() -> None:
    """Full pipeline: detect candidates → pick one → merge → audit trail."""
    blocks = [
        _blk("D-001", content="fixed-point arithmetic ensures determinism across substrates", tags="determinism"),
        _blk("D-002", content="use fixed-point arithmetic for cross-substrate determinism", tags="cross-substrate"),
        _blk("D-003", content="quarterly earnings report for FY2025 shows growth"),
    ]
    candidates = find_merge_candidates(blocks, min_similarity=0.4)
    # D-001 and D-002 should be flagged
    flagged_ids = {frozenset([c.block_a["_id"], c.block_b["_id"]]) for c in candidates}
    assert frozenset(["D-001", "D-002"]) in flagged_ids

    # Take the top candidate and merge
    top = candidates[0]
    merged = merge_blocks(top.block_a, top.block_b, strategy=top.suggested_strategy)
    # Audit trail present
    assert "_merged_from" in merged
    assert len(merged["_merged_from"]) == 2
    # Tags unioned
    assert "determinism" in merged["tags"]
    assert "cross-substrate" in merged["tags"]
