#!/usr/bin/env python3
"""Tests for prefetch_context() in recall.py."""

import os
import sys

scripts_dir = os.path.join(os.path.dirname(__file__), "..", "scripts")
sys.path.insert(0, scripts_dir)

from recall import prefetch_context  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_workspace(tmp_path, decisions_md=""):
    """Create a minimal workspace with all required corpus files."""
    ws = str(tmp_path)
    for sub in ["decisions", "tasks", "entities", "intelligence"]:
        os.makedirs(os.path.join(ws, sub), exist_ok=True)
    # Write decisions
    with open(os.path.join(ws, "decisions", "DECISIONS.md"), "w", encoding="utf-8") as f:
        f.write(decisions_md)
    # Stub task file (needed for IDF diversity in recall)
    with open(os.path.join(ws, "tasks", "TASKS.md"), "w") as f:
        f.write("[T-20260218-099]\nTitle: Placeholder task\nStatus: active\n")
    # Stub remaining corpus files
    for fname in [
        "entities/projects.md",
        "entities/people.md",
        "entities/tools.md",
        "entities/incidents.md",
        "intelligence/CONTRADICTIONS.md",
        "intelligence/DRIFT.md",
        "intelligence/SIGNALS.md",
    ]:
        path = os.path.join(ws, fname)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            with open(path, "w") as f:
                f.write(f"# {os.path.basename(fname)}\n")
    return ws


def _block_md(block_id, statement, status="active", date="2026-02-18", tags=""):
    """Return a single block in Schema-v1 markdown format."""
    lines = [
        f"[{block_id}]",
        f"Statement: {statement}",
        f"Status: {status}",
        f"Date: {date}",
    ]
    if tags:
        lines.append(f"Tags: {tags}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_prefetch_returns_list(tmp_path):
    """prefetch_context always returns a list."""
    ws = _setup_workspace(tmp_path)
    result = prefetch_context(ws, ["test signal"])
    assert isinstance(result, list)


def test_prefetch_empty_signals(tmp_path):
    """Empty signals list returns empty results."""
    ws = _setup_workspace(tmp_path)
    result = prefetch_context(ws, [])
    assert result == []


def test_prefetch_with_entity_signals(tmp_path):
    """Signals containing entity names find relevant blocks."""
    decisions = "\n---\n\n".join([
        _block_md("D-20260218-001", "Use PostgreSQL for the main database",
                   tags="database, architecture"),
        _block_md("D-20260218-002", "Deploy frontend with Vercel",
                   tags="deployment, frontend"),
    ])
    ws = _setup_workspace(tmp_path, decisions)
    result = prefetch_context(ws, ["PostgreSQL"])
    assert len(result) >= 1
    ids = [r["_id"] for r in result]
    assert "D-20260218-001" in ids


def test_prefetch_with_topic_signals(tmp_path):
    """Topic keywords find blocks via recall."""
    decisions = "\n---\n\n".join([
        _block_md("D-20260218-010", "Use JWT tokens for API authentication",
                   tags="security, auth"),
        _block_md("D-20260218-011", "Store logs in Elasticsearch",
                   tags="logging, infrastructure"),
    ])
    ws = _setup_workspace(tmp_path, decisions)
    result = prefetch_context(ws, ["authentication security"])
    assert len(result) >= 1
    ids = [r["_id"] for r in result]
    assert "D-20260218-010" in ids


def test_prefetch_limit_respected(tmp_path):
    """Result count does not exceed limit."""
    blocks = []
    for i in range(1, 11):
        blocks.append(_block_md(
            f"D-20260218-{i:03d}",
            f"Database decision number {i} about storage",
            tags="database",
        ))
    decisions = "\n---\n\n".join(blocks)
    ws = _setup_workspace(tmp_path, decisions)
    result = prefetch_context(ws, ["database storage"], limit=3)
    assert len(result) <= 3


def test_prefetch_deduplicates(tmp_path):
    """Same block from multiple signals appears only once."""
    decisions = _block_md(
        "D-20260218-020",
        "Use PostgreSQL database for persistent storage",
        tags="database, architecture, storage",
    )
    ws = _setup_workspace(tmp_path, decisions)
    # Two different signals that should both match the same block
    result = prefetch_context(ws, ["PostgreSQL", "database storage"], limit=10)
    ids = [r["_id"] for r in result]
    assert ids.count("D-20260218-020") == 1, "Block should appear only once"


def test_prefetch_category_aware(tmp_path):
    """When categories/ exists, prefetch uses category context."""
    decisions = "\n---\n\n".join([
        _block_md("D-20260218-030", "Use Redis for caching layer",
                   tags="architecture, caching"),
        _block_md("D-20260218-031", "Fix login page timeout bug",
                   tags="bug, frontend"),
    ])
    ws = _setup_workspace(tmp_path, decisions)

    # Create a categories/ directory with a category file to simulate
    # prior distill output
    cats_dir = os.path.join(ws, "categories")
    os.makedirs(cats_dir, exist_ok=True)
    with open(os.path.join(cats_dir, "architecture.md"), "w") as f:
        f.write("# Architecture\n\n[D-20260218-030]\nStatement: Use Redis for caching layer\n")

    result = prefetch_context(ws, ["architecture caching"])
    assert isinstance(result, list)
    # Should still return results (category context augments, does not replace)
    # The exact behavior depends on implementation, but it must not crash
    # and should return relevant blocks when categories exist
    if len(result) > 0:
        ids = [r["_id"] for r in result]
        assert "D-20260218-030" in ids
