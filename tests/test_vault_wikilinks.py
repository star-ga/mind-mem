# Copyright 2026 STARGA, Inc.
"""Tests for Obsidian wikilink export on vault_sync (v3.3 feature)."""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from mind_mem.agent_bridge import VaultBlock, VaultBridge
from mind_mem.knowledge_graph import KnowledgeGraph, Predicate

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_vault(tmp: str) -> str:
    """Create a minimal vault directory and return its path."""
    root = os.path.join(tmp, "vault")
    os.makedirs(root)
    return root


def _make_kg(tmp: str) -> str:
    """Create a KG db with two outgoing edges from 'PRJ-mind-mem' and return path."""
    kg_path = os.path.join(tmp, "kg.db")
    with KnowledgeGraph(kg_path) as kg:
        kg.add_edge(
            "PRJ-mind-mem",
            Predicate.DEPENDS_ON,
            "PRJ-sqlite",
            source_block_id="PRJ-mind-mem",
        )
        kg.add_edge(
            "PRJ-mind-mem",
            Predicate.SUPERSEDES,
            "PRJ-mem-os",
            source_block_id="PRJ-mind-mem",
        )
    return kg_path


def _read_written(vault_root: str, relative_path: str) -> str:
    with open(os.path.join(vault_root, relative_path), encoding="utf-8") as fh:
        return fh.read()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp(tmp_path: Path) -> str:
    return str(tmp_path)


@pytest.fixture()
def vault(tmp: str) -> str:
    return _make_vault(tmp)


@pytest.fixture()
def block() -> VaultBlock:
    return VaultBlock(
        relative_path="projects/mind-mem.md",
        block_id="PRJ-mind-mem",
        block_type="project",
        title="mind-mem",
        body="The flagship memory product.",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestWriteWithoutKG:
    """VaultBridge.write without kg_path must produce no ## Links section."""

    def test_no_links_section_when_kg_path_is_none(self, vault: str, block: VaultBlock) -> None:
        bridge = VaultBridge(vault_root=vault)
        bridge.write(block, overwrite=True)
        content = _read_written(vault, block.relative_path)
        assert "## Links" not in content

    def test_block_body_present(self, vault: str, block: VaultBlock) -> None:
        bridge = VaultBridge(vault_root=vault)
        bridge.write(block, overwrite=True)
        content = _read_written(vault, block.relative_path)
        assert "The flagship memory product." in content


class TestWriteWithKG:
    """VaultBridge.write_with_links appends a ## Links section from KG edges."""

    def test_links_section_present_with_two_edges(self, vault: str, block: VaultBlock, tmp: str) -> None:
        kg_path = _make_kg(tmp)
        bridge = VaultBridge(vault_root=vault)
        bridge.write(block, kg_path=kg_path, overwrite=True)
        content = _read_written(vault, block.relative_path)
        assert "## Links" in content

    def test_links_contain_both_wikilinks(self, vault: str, block: VaultBlock, tmp: str) -> None:
        kg_path = _make_kg(tmp)
        bridge = VaultBridge(vault_root=vault)
        bridge.write(block, kg_path=kg_path, overwrite=True)
        content = _read_written(vault, block.relative_path)
        # KG canonicalises entity names to lowercase.
        assert "[[prj-sqlite]]" in content
        assert "[[prj-mem-os]]" in content

    def test_links_contain_predicate_labels(self, vault: str, block: VaultBlock, tmp: str) -> None:
        kg_path = _make_kg(tmp)
        bridge = VaultBridge(vault_root=vault)
        bridge.write(block, kg_path=kg_path, overwrite=True)
        content = _read_written(vault, block.relative_path)
        assert "depends_on" in content
        assert "supersedes" in content

    def test_body_still_present_with_links(self, vault: str, block: VaultBlock, tmp: str) -> None:
        kg_path = _make_kg(tmp)
        bridge = VaultBridge(vault_root=vault)
        bridge.write(block, kg_path=kg_path, overwrite=True)
        content = _read_written(vault, block.relative_path)
        assert "The flagship memory product." in content
        assert "## Links" in content


class TestSlugDeduplication:
    """When two edges point to the same slug, only one [[link]] is emitted."""

    def test_duplicate_object_slug_deduped(self, vault: str, block: VaultBlock, tmp: str) -> None:
        kg_path = os.path.join(tmp, "kg_dup.db")
        # Two edges, same object (different predicate — KG allows this because
        # PK is (subject, predicate, object, source_block_id)).
        with KnowledgeGraph(kg_path) as kg:
            kg.add_edge(
                "PRJ-mind-mem",
                Predicate.DEPENDS_ON,
                "PRJ-sqlite",
                source_block_id="PRJ-mind-mem",
            )
            kg.add_edge(
                "PRJ-mind-mem",
                Predicate.RELATED_TO,
                "PRJ-sqlite",
                source_block_id="PRJ-mind-mem",
            )
        bridge = VaultBridge(vault_root=vault)
        bridge.write(block, kg_path=kg_path, overwrite=True)
        content = _read_written(vault, block.relative_path)
        # The slug should appear exactly once as a wikilink target.
        # KG canonicalises entity names to lowercase, so the slug is prj-sqlite.
        matches = re.findall(r"\[\[prj-sqlite\]\]", content)
        assert len(matches) == 1


class TestMissingKGFile:
    """When kg_path points to a non-existent file, write proceeds without links."""

    def test_missing_kg_produces_block_without_links(self, vault: str, block: VaultBlock, tmp: str) -> None:
        missing = os.path.join(tmp, "does_not_exist.db")
        bridge = VaultBridge(vault_root=vault)
        bridge.write(block, kg_path=missing, overwrite=True)
        content = _read_written(vault, block.relative_path)
        assert "## Links" not in content
        assert "The flagship memory product." in content


class TestSlugFormatting:
    """Object slugs preserve the entity name as returned by the KG."""

    def test_plain_entity_name_used_as_slug(self, vault: str, tmp: str) -> None:
        kg_path = os.path.join(tmp, "kg_plain.db")
        with KnowledgeGraph(kg_path) as kg:
            kg.add_edge(
                "my-block",
                Predicate.RELATED_TO,
                "openai",
                source_block_id="my-block",
            )
        plain_block = VaultBlock(
            relative_path="notes/my-block.md",
            block_id="my-block",
            block_type="note",
            title="My Block",
            body="Body text.",
        )
        bridge = VaultBridge(vault_root=vault)
        bridge.write(plain_block, kg_path=kg_path, overwrite=True)
        content = _read_written(vault, plain_block.relative_path)
        assert "[[openai]]" in content
