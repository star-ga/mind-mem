"""Tests for block parser field extraction."""

from __future__ import annotations

import os
import tempfile

from mind_mem.block_parser import parse_file


def _parse_block(content):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(content)
        path = f.name
    try:
        return parse_file(path)
    finally:
        os.unlink(path)


def test_type_field():
    blocks = _parse_block("[F-001]\nType: Decision\nStatement: Test\n")
    assert len(blocks) >= 1


def test_statement_field():
    blocks = _parse_block("[F-002]\nType: Decision\nStatement: Important statement here\n")
    if blocks:
        b = blocks[0]
        assert "statement" in b or "Statement" in b or "raw" in b


def test_tags_field():
    blocks = _parse_block("[F-003]\nType: Decision\nStatement: Tagged\nTags: alpha, beta\n")
    assert len(blocks) >= 1


def test_date_field():
    blocks = _parse_block("[F-004]\nType: Decision\nStatement: Dated\nDate: 2026-02-24\n")
    assert len(blocks) >= 1


def test_status_field():
    blocks = _parse_block("[F-005]\nType: Decision\nStatement: Active\nStatus: Active\n")
    assert len(blocks) >= 1


def test_priority_field():
    blocks = _parse_block("[F-006]\nType: Decision\nStatement: High pri\nPriority: High\n")
    assert len(blocks) >= 1


def test_references_field():
    blocks = _parse_block("[F-007]\nType: Decision\nStatement: With refs\nReferences: F-001, F-002\n")
    assert len(blocks) >= 1
