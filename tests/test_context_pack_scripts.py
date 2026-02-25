"""Tests for context packing via scripts._recall_context."""
from __future__ import annotations

from scripts._recall_context import context_pack


def _make_block(block_id, statement, dia_id="1:001"):
    """Create a minimal block dict."""
    return {
        "_id": block_id,
        "Statement": statement,
        "Tags": "test",
        "Status": "active",
        "DiaID": dia_id,
        "_source_file": "decisions/DECISIONS.md",
        "_line": 10,
    }


def _make_result(block, score=10.0):
    """Wrap a block into a result entry."""
    return {"block": block, "score": score, "_id": block["_id"]}


def test_context_pack_empty():
    """Empty inputs return empty list."""
    result = context_pack("test", [], [], [])
    assert isinstance(result, list)
    assert len(result) == 0


def test_context_pack_single_block():
    """Single block gets packed correctly."""
    block = _make_block("TEST-001", "Test statement")
    entry = _make_result(block)
    result = context_pack("test", [entry], [block], [entry])
    assert len(result) == 1
    assert result[0]["_id"] == "TEST-001"


def test_context_pack_multiple_blocks():
    """Multiple blocks get packed."""
    blocks = [_make_block(f"TEST-{i:03d}", f"Statement {i}", f"1:{i:03d}") for i in range(5)]
    entries = [_make_result(b, score=10.0 - i) for i, b in enumerate(blocks)]
    result = context_pack("statement", entries, blocks, entries)
    assert len(result) >= 1
    ids = {r["_id"] for r in result}
    assert "TEST-000" in ids


def test_context_pack_preserves_content():
    """Context pack preserves block content."""
    block = _make_block("X-001", "Important decision about API design")
    entry = _make_result(block)
    result = context_pack("API design", [entry], [block], [entry])
    assert len(result) == 1
    assert result[0]["block"]["Statement"] == "Important decision about API design"


def test_context_pack_score_ordering():
    """Higher-scored results appear in output."""
    blocks = [
        _make_block("HIGH-001", "High priority item", "3:001"),
        _make_block("LOW-001", "Low priority item", "3:002"),
    ]
    entries = [
        _make_result(blocks[0], score=20.0),
        _make_result(blocks[1], score=1.0),
    ]
    result = context_pack("priority", entries, blocks, entries)
    assert len(result) >= 1
    assert result[0]["_id"] == "HIGH-001"


def test_context_pack_returns_list():
    """Return type is always a list of dicts."""
    block = _make_block("T-001", "s")
    entry = _make_result(block)
    result = context_pack("s", [entry], [block], [entry])
    assert isinstance(result, list)
    for item in result:
        assert isinstance(item, dict)
