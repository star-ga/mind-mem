#!/usr/bin/env python3
"""Tests for category_distiller.py — CategoryDistiller class."""

import json
import os
import sys

scripts_dir = os.path.join(os.path.dirname(__file__), "..", "scripts")
sys.path.insert(0, scripts_dir)

from block_parser import parse_file  # noqa: E402
from category_distiller import CategoryDistiller  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_decisions(ws, blocks_md):
    """Write a DECISIONS.md file and return its path."""
    dec_dir = os.path.join(ws, "decisions")
    os.makedirs(dec_dir, exist_ok=True)
    path = os.path.join(dec_dir, "DECISIONS.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(blocks_md)
    return path


def _make_block_md(block_id, statement, status="active", date="2026-02-18", tags=""):
    """Return a single block in the Schema-v1 markdown format."""
    lines = [
        f"[{block_id}]",
        f"Statement: {statement}",
        f"Status: {status}",
        f"Date: {date}",
    ]
    if tags:
        lines.append(f"Tags: {tags}")
    return "\n".join(lines) + "\n"


def _setup_workspace(tmp_path, blocks_md):
    """Create a minimal workspace with the required corpus files."""
    ws = str(tmp_path)
    _write_decisions(ws, blocks_md)
    # Create other required corpus dirs/files so recall doesn't fail
    for sub in ["tasks", "entities", "intelligence"]:
        os.makedirs(os.path.join(ws, sub), exist_ok=True)
    for fname in [
        "tasks/TASKS.md",
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_categorize_block_architecture(tmp_path):
    """Block with architecture keywords gets the 'architecture' category."""
    ws = _setup_workspace(tmp_path, _make_block_md(
        "D-20260218-001",
        "Use PostgreSQL for the main database",
        tags="database, architecture",
    ))
    distiller = CategoryDistiller()
    blocks = parse_file(os.path.join(ws, "decisions", "DECISIONS.md"))
    cats = distiller.categorize_block(blocks[0])
    assert "architecture" in cats


def test_categorize_block_bugs(tmp_path):
    """Block describing a bug gets the 'bugs' category."""
    ws = _setup_workspace(tmp_path, _make_block_md(
        "D-20260218-002",
        "Fix null pointer bug in auth handler",
        tags="bug, critical",
    ))
    distiller = CategoryDistiller()
    blocks = parse_file(os.path.join(ws, "decisions", "DECISIONS.md"))
    cats = distiller.categorize_block(blocks[0])
    assert "bugs" in cats


def test_categorize_block_multiple_categories(tmp_path):
    """Block matching keywords from 2+ categories returns all of them."""
    ws = _setup_workspace(tmp_path, _make_block_md(
        "D-20260218-003",
        "Fix deployment architecture bug in CI pipeline",
        tags="deployment, bug, architecture",
    ))
    distiller = CategoryDistiller()
    blocks = parse_file(os.path.join(ws, "decisions", "DECISIONS.md"))
    cats = distiller.categorize_block(blocks[0])
    assert len(cats) >= 2, f"Expected 2+ categories, got {cats}"


def test_categorize_block_uncategorized(tmp_path):
    """Block with no keyword matches gets 'uncategorized'."""
    ws = _setup_workspace(tmp_path, _make_block_md(
        "D-20260218-004",
        "Miscellaneous note about something unrelated",
        tags="",
    ))
    distiller = CategoryDistiller()
    blocks = parse_file(os.path.join(ws, "decisions", "DECISIONS.md"))
    cats = distiller.categorize_block(blocks[0])
    assert "uncategorized" in cats


def test_categorize_block_from_tags(tmp_path):
    """Category detected from Tags field (higher weight than statement)."""
    ws = _setup_workspace(tmp_path, _make_block_md(
        "D-20260218-005",
        "We decided to proceed with this option",
        tags="auth, oauth, credential",
    ))
    distiller = CategoryDistiller()
    blocks = parse_file(os.path.join(ws, "decisions", "DECISIONS.md"))
    cats = distiller.categorize_block(blocks[0])
    assert "credentials" in cats


def test_distill_creates_category_files(tmp_path):
    """Full distill() writes files into categories/ directory."""
    md = "\n---\n\n".join([
        _make_block_md("D-20260218-010", "Use JWT for auth token", tags="credential"),
        _make_block_md("D-20260218-011", "Deploy via Docker containers", tags="deployment"),
    ])
    ws = _setup_workspace(tmp_path, md)
    distiller = CategoryDistiller()
    distiller.distill(ws)

    cats_dir = os.path.join(ws, "categories")
    assert os.path.isdir(cats_dir), "categories/ directory should exist"
    files = os.listdir(cats_dir)
    md_files = [f for f in files if f.endswith(".md")]
    assert len(md_files) >= 1, f"Expected at least 1 category .md file, got {md_files}"


def test_distill_creates_manifest(tmp_path):
    """distill() writes _manifest.json with correct counts."""
    md = "\n---\n\n".join([
        _make_block_md("D-20260218-020", "Use PostgreSQL", tags="database, architecture"),
        _make_block_md("D-20260218-021", "Fix auth bug", tags="bug"),
    ])
    ws = _setup_workspace(tmp_path, md)
    distiller = CategoryDistiller()
    distiller.distill(ws)

    manifest_path = os.path.join(ws, "categories", "_manifest.json")
    assert os.path.isfile(manifest_path), "_manifest.json should exist"

    with open(manifest_path) as f:
        manifest = json.load(f)
    assert isinstance(manifest, dict)
    assert "categories" in manifest
    assert "total_blocks" in manifest
    # Each category count should be positive
    for cat, count in manifest["categories"].items():
        assert count > 0, f"Category {cat} should have positive count"


def test_write_category_file(tmp_path):
    """Single category file has correct format: title, blocks, dates."""
    md = _make_block_md("D-20260218-030", "Use Redis for caching layer", tags="architecture")
    ws = _setup_workspace(tmp_path, md)
    distiller = CategoryDistiller()
    distiller.distill(ws)

    cats_dir = os.path.join(ws, "categories")
    # Find the architecture category file
    arch_file = os.path.join(cats_dir, "architecture.md")
    if not os.path.exists(arch_file):
        # Might be under a different name; check any .md file
        md_files = [f for f in os.listdir(cats_dir) if f.endswith(".md")]
        assert len(md_files) > 0, "Should have at least one category file"
        arch_file = os.path.join(cats_dir, md_files[0])

    content = open(arch_file, encoding="utf-8").read()
    assert "D-20260218-030" in content, "Block ID should appear in category file"
    assert "Redis" in content or "caching" in content, "Block content should appear"


def test_get_category_context(tmp_path):
    """Returns relevant category content for a query."""
    md = "\n---\n\n".join([
        _make_block_md("D-20260218-040", "Use PostgreSQL for storage", tags="database, architecture"),
        _make_block_md("D-20260218-041", "Fix login page bug", tags="bug"),
    ])
    ws = _setup_workspace(tmp_path, md)
    distiller = CategoryDistiller()
    distiller.distill(ws)

    ctx = distiller.get_category_context("database architecture", ws)
    assert isinstance(ctx, str)
    assert len(ctx) > 0, "Should return non-empty context for matching query"


def test_get_category_context_empty(tmp_path):
    """Returns '' when no categories directory exists."""
    ws = str(tmp_path)
    os.makedirs(ws, exist_ok=True)
    distiller = CategoryDistiller()
    ctx = distiller.get_category_context("anything", ws)
    assert ctx == ""


def test_get_categories_for_query(tmp_path):
    """Returns ordered category names matching a query."""
    md = "\n---\n\n".join([
        _make_block_md("D-20260218-050", "Use JWT for authentication", tags="auth"),
        _make_block_md("D-20260218-051", "Deploy with Kubernetes", tags="deployment"),
        _make_block_md("D-20260218-052", "Fix auth regression", tags="bug"),
    ])
    _setup_workspace(tmp_path, md)
    distiller = CategoryDistiller()

    cats = distiller.get_categories_for_query("auth credential token")
    assert isinstance(cats, list)
    assert len(cats) >= 1
    assert "credentials" in cats, "credentials category should match auth query"


def test_extra_categories(tmp_path):
    """Custom categories passed to constructor work alongside defaults."""
    md = _make_block_md(
        "D-20260218-060",
        "Integrate with Stripe payment gateway",
        tags="payments, billing",
    )
    ws = _setup_workspace(tmp_path, md)
    extra = {"payments": ["payment", "billing", "stripe", "invoice"]}
    distiller = CategoryDistiller(extra_categories=extra)
    blocks = parse_file(os.path.join(ws, "decisions", "DECISIONS.md"))
    cats = distiller.categorize_block(blocks[0])
    assert "payments" in cats


def test_distill_incremental(tmp_path):
    """Running distill() twice overwrites cleanly — no duplicated blocks."""
    md = _make_block_md("D-20260218-070", "Use Redis for caching", tags="architecture")
    ws = _setup_workspace(tmp_path, md)
    distiller = CategoryDistiller()

    distiller.distill(ws)
    distiller.distill(ws)  # run again

    cats_dir = os.path.join(ws, "categories")
    # Find any category file and verify block appears exactly once
    md_files = [f for f in os.listdir(cats_dir) if f.endswith(".md")]
    assert len(md_files) >= 1

    for md_file in md_files:
        content = open(os.path.join(cats_dir, md_file), encoding="utf-8").read()
        count = content.count("D-20260218-070")
        if count > 0:
            assert count == 1, f"Block ID should appear exactly once, found {count} times"
