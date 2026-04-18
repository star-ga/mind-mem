# Copyright 2026 STARGA, Inc.
"""Tests for MerkleTree — Merkle tree over the block store for verification proofs."""

from __future__ import annotations

import json

import pytest

# Import the module's own hash helpers so tests stay in lockstep with the
# implementation — including the domain-separation tags introduced to
# prevent Bitcoin-style second-preimage attacks.
from mind_mem.merkle_tree import (
    MerkleNode,
    MerkleTree,
    _leaf_hash,
    _node_hash,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def three_blocks() -> list[tuple[str, str]]:
    return [
        ("blk-001", "aaa111"),
        ("blk-002", "bbb222"),
        ("blk-003", "ccc333"),
    ]


@pytest.fixture()
def four_blocks() -> list[tuple[str, str]]:
    return [
        ("blk-001", "aaa111"),
        ("blk-002", "bbb222"),
        ("blk-003", "ccc333"),
        ("blk-004", "ddd444"),
    ]


# ---------------------------------------------------------------------------
# 1. Empty tree
# ---------------------------------------------------------------------------


class TestEmptyTree:
    def test_empty_tree_root_hash_is_empty_string(self) -> None:
        tree = MerkleTree()
        assert tree.root_hash == ""

    def test_empty_tree_size_is_zero(self) -> None:
        tree = MerkleTree()
        assert tree.size == 0

    def test_empty_tree_verify_is_true(self) -> None:
        tree = MerkleTree()
        assert tree.verify_tree() is True

    def test_empty_tree_get_proof_raises(self) -> None:
        tree = MerkleTree()
        with pytest.raises(KeyError):
            tree.get_proof("blk-001")


# ---------------------------------------------------------------------------
# 2. Single block
# ---------------------------------------------------------------------------


class TestSingleBlock:
    def test_single_block_root_hash_equals_leaf_hash(self) -> None:
        tree = MerkleTree()
        tree.build([("blk-001", "abc")])
        # root of a single-leaf tree is the SHA3-512 of block_id + content_hash
        assert tree.root_hash == _leaf_hash("abc", "blk-001")

    def test_single_block_size(self) -> None:
        tree = MerkleTree()
        tree.build([("blk-001", "abc")])
        assert tree.size == 1

    def test_single_block_proof_is_empty_list(self) -> None:
        tree = MerkleTree()
        tree.build([("blk-001", "abc")])
        proof = tree.get_proof("blk-001")
        assert proof == []

    def test_single_block_verify_proof(self) -> None:
        tree = MerkleTree()
        tree.build([("blk-001", "abc")])
        proof = tree.get_proof("blk-001")
        assert tree.verify_proof("blk-001", "abc", proof, tree.root_hash) is True


# ---------------------------------------------------------------------------
# 3. Build from multiple blocks — root correctness
# ---------------------------------------------------------------------------


class TestBuildFromBlocks:
    def test_even_leaf_count_root(self, four_blocks: list[tuple[str, str]]) -> None:
        tree = MerkleTree()
        tree.build(four_blocks)
        # Manually compute expected root (leaf hash includes block_id)
        h0 = _leaf_hash("aaa111", "blk-001")
        h1 = _leaf_hash("bbb222", "blk-002")
        h2 = _leaf_hash("ccc333", "blk-003")
        h3 = _leaf_hash("ddd444", "blk-004")
        parent_left = _node_hash(h0, h1)
        parent_right = _node_hash(h2, h3)
        expected_root = _node_hash(parent_left, parent_right)
        assert tree.root_hash == expected_root

    def test_odd_leaf_count_duplicates_last(self, three_blocks: list[tuple[str, str]]) -> None:
        tree = MerkleTree()
        tree.build(three_blocks)
        # Odd: last leaf duplicated (leaf hash includes block_id)
        h0 = _leaf_hash("aaa111", "blk-001")
        h1 = _leaf_hash("bbb222", "blk-002")
        h2 = _leaf_hash("ccc333", "blk-003")
        h3 = h2  # duplicate
        parent_left = _node_hash(h0, h1)
        parent_right = _node_hash(h2, h3)
        expected_root = _node_hash(parent_left, parent_right)
        assert tree.root_hash == expected_root

    def test_build_sets_correct_size(self, four_blocks: list[tuple[str, str]]) -> None:
        tree = MerkleTree()
        tree.build(four_blocks)
        assert tree.size == 4

    def test_build_replaces_previous_tree(self, four_blocks: list[tuple[str, str]]) -> None:
        tree = MerkleTree()
        tree.build([("blk-001", "aaa111")])
        old_root = tree.root_hash
        tree.build(four_blocks)
        assert tree.root_hash != old_root
        assert tree.size == 4


# ---------------------------------------------------------------------------
# 4. Proof generation and verification
# ---------------------------------------------------------------------------


class TestProofGenerationAndVerification:
    def test_proof_verifies_for_every_leaf(self, four_blocks: list[tuple[str, str]]) -> None:
        tree = MerkleTree()
        tree.build(four_blocks)
        root = tree.root_hash
        for block_id, content_hash in four_blocks:
            proof = tree.get_proof(block_id)
            assert tree.verify_proof(block_id, content_hash, proof, root) is True

    def test_proof_length_is_log2_leaves(self, four_blocks: list[tuple[str, str]]) -> None:
        tree = MerkleTree()
        tree.build(four_blocks)  # 4 leaves → depth 2 → proof length 1 per level
        proof = tree.get_proof("blk-001")
        assert len(proof) == 2  # depth is 2 for 4-leaf tree

    def test_proof_direction_labels(self, four_blocks: list[tuple[str, str]]) -> None:
        tree = MerkleTree()
        tree.build(four_blocks)
        proof = tree.get_proof("blk-001")
        for hash_val, direction in proof:
            assert isinstance(hash_val, str) and len(hash_val) > 0
            assert direction in ("left", "right")

    def test_proof_unknown_block_raises(self, four_blocks: list[tuple[str, str]]) -> None:
        tree = MerkleTree()
        tree.build(four_blocks)
        with pytest.raises(KeyError):
            tree.get_proof("nonexistent")


# ---------------------------------------------------------------------------
# 5. Tamper detection
# ---------------------------------------------------------------------------


class TestTamperDetection:
    def test_wrong_content_hash_fails_verify(self, four_blocks: list[tuple[str, str]]) -> None:
        tree = MerkleTree()
        tree.build(four_blocks)
        proof = tree.get_proof("blk-001")
        # Use the correct proof but a tampered content hash
        assert tree.verify_proof("blk-001", "tampered_hash", proof, tree.root_hash) is False

    def test_wrong_root_hash_fails_verify(self, four_blocks: list[tuple[str, str]]) -> None:
        tree = MerkleTree()
        tree.build(four_blocks)
        proof = tree.get_proof("blk-001")
        assert tree.verify_proof("blk-001", "aaa111", proof, "bad_root_hash") is False

    def test_corrupted_proof_fails_verify(self, four_blocks: list[tuple[str, str]]) -> None:
        tree = MerkleTree()
        tree.build(four_blocks)
        proof = tree.get_proof("blk-001")
        corrupted = [(proof[0][0] + "x", proof[0][1])] + proof[1:]
        assert tree.verify_proof("blk-001", "aaa111", corrupted, tree.root_hash) is False

    def test_verify_tree_detects_root_corruption(self, four_blocks: list[tuple[str, str]]) -> None:
        tree = MerkleTree()
        tree.build(four_blocks)
        # Manually corrupt the root node hash
        assert tree._root is not None
        tree._root.hash = "corrupted"
        assert tree.verify_tree() is False


# ---------------------------------------------------------------------------
# 6. Add and remove blocks
# ---------------------------------------------------------------------------


class TestAddRemoveBlocks:
    def test_add_block_increases_size(self, four_blocks: list[tuple[str, str]]) -> None:
        tree = MerkleTree()
        tree.build(four_blocks)
        tree.add_block("blk-005", "eee555")
        assert tree.size == 5

    def test_add_block_changes_root(self, four_blocks: list[tuple[str, str]]) -> None:
        tree = MerkleTree()
        tree.build(four_blocks)
        old_root = tree.root_hash
        tree.add_block("blk-005", "eee555")
        assert tree.root_hash != old_root

    def test_add_block_proof_verifies(self, four_blocks: list[tuple[str, str]]) -> None:
        tree = MerkleTree()
        tree.build(four_blocks)
        tree.add_block("blk-005", "eee555")
        proof = tree.get_proof("blk-005")
        assert tree.verify_proof("blk-005", "eee555", proof, tree.root_hash) is True

    def test_remove_block_decreases_size(self, four_blocks: list[tuple[str, str]]) -> None:
        tree = MerkleTree()
        tree.build(four_blocks)
        tree.remove_block("blk-002")
        assert tree.size == 3

    def test_remove_block_changes_root(self, four_blocks: list[tuple[str, str]]) -> None:
        tree = MerkleTree()
        tree.build(four_blocks)
        old_root = tree.root_hash
        tree.remove_block("blk-002")
        assert tree.root_hash != old_root

    def test_remove_unknown_block_raises(self, four_blocks: list[tuple[str, str]]) -> None:
        tree = MerkleTree()
        tree.build(four_blocks)
        with pytest.raises(KeyError):
            tree.remove_block("nonexistent")

    def test_remove_all_blocks_gives_empty_tree(self, three_blocks: list[tuple[str, str]]) -> None:
        tree = MerkleTree()
        tree.build(three_blocks)
        for block_id, _ in three_blocks:
            tree.remove_block(block_id)
        assert tree.size == 0
        assert tree.root_hash == ""


# ---------------------------------------------------------------------------
# 7. verify_tree — full integrity check
# ---------------------------------------------------------------------------


class TestVerifyTree:
    def test_freshly_built_tree_verifies(self, four_blocks: list[tuple[str, str]]) -> None:
        tree = MerkleTree()
        tree.build(four_blocks)
        assert tree.verify_tree() is True

    def test_tree_after_add_verifies(self, four_blocks: list[tuple[str, str]]) -> None:
        tree = MerkleTree()
        tree.build(four_blocks)
        tree.add_block("blk-005", "eee555")
        assert tree.verify_tree() is True


# ---------------------------------------------------------------------------
# 8. MerkleNode dataclass
# ---------------------------------------------------------------------------


class TestMerkleNode:
    def test_leaf_node_has_block_id(self) -> None:
        node = MerkleNode(hash="abc", left=None, right=None, block_id="blk-001")
        assert node.block_id == "blk-001"
        assert node.left is None
        assert node.right is None

    def test_internal_node_has_no_block_id(self) -> None:
        left = MerkleNode(hash="lll", left=None, right=None, block_id="blk-001")
        right = MerkleNode(hash="rrr", left=None, right=None, block_id="blk-002")
        parent = MerkleNode(hash="ppp", left=left, right=right, block_id=None)
        assert parent.block_id is None
        assert parent.left is left
        assert parent.right is right


# ---------------------------------------------------------------------------
# 9. Serialization round-trip
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_export_import_round_trip(self, four_blocks: list[tuple[str, str]]) -> None:
        tree = MerkleTree()
        tree.build(four_blocks)
        original_root = tree.root_hash
        serialized = tree.export_json()
        assert isinstance(serialized, str)

        restored = MerkleTree()
        restored.import_json(serialized)
        assert restored.root_hash == original_root
        assert restored.size == 4

    def test_imported_tree_proofs_verify(self, four_blocks: list[tuple[str, str]]) -> None:
        tree = MerkleTree()
        tree.build(four_blocks)
        serialized = tree.export_json()

        restored = MerkleTree()
        restored.import_json(serialized)
        root = restored.root_hash
        for block_id, content_hash in four_blocks:
            proof = restored.get_proof(block_id)
            assert restored.verify_proof(block_id, content_hash, proof, root) is True

    def test_export_json_is_valid_json(self, four_blocks: list[tuple[str, str]]) -> None:
        tree = MerkleTree()
        tree.build(four_blocks)
        data = json.loads(tree.export_json())
        assert "root_hash" in data
        assert "leaves" in data
        assert len(data["leaves"]) == 4
