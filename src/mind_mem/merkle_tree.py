# Copyright 2026 STARGA, Inc.
"""Merkle tree over the block store for verification proofs.

Any block can be verified against the tree root without reading all other
blocks. Internal nodes use SHA3-512(left_hash + right_hash). Leaves use
SHA3-512(content_hash). Odd numbers of leaves are handled by duplicating
the last leaf (standard Merkle convention).

Usage:
    from .merkle_tree import MerkleTree

    tree = MerkleTree()
    tree.build([("blk-001", "abc123"), ("blk-002", "def456")])
    proof = tree.get_proof("blk-001")
    ok = tree.verify_proof("blk-001", "abc123", proof, tree.root_hash)

Zero external deps — hashlib, json (all stdlib).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Optional

from .observability import get_logger

_log = get_logger("merkle_tree")


# ---------------------------------------------------------------------------
# Internal hash helpers
# ---------------------------------------------------------------------------


def _sha3(data: str) -> str:
    """SHA3-512 hex digest of UTF-8-encoded data."""
    return hashlib.sha3_512(data.encode("utf-8")).hexdigest()


# Domain separation tags prevent second-preimage attacks where a crafted
# leaf-pair concatenation would hash to the same value as an internal node.
# This is the well-known Bitcoin Merkle tree CVE pattern.
_LEAF_TAG = "L:"
_NODE_TAG = "N:"


def _leaf_hash(content_hash: str, block_id: str = "") -> str:
    """Hash a leaf node from its block_id and content hash.

    block_id is included so two blocks with identical content produce
    distinct leaf hashes and cannot verify each other's proofs. A leaf
    domain tag ensures leaf hashes cannot be confused with internal nodes.
    """
    return _sha3(f"{_LEAF_TAG}{block_id}|{content_hash}")


def _node_hash(left: str, right: str) -> str:
    """Hash an internal node from its children's hashes.

    Domain tag distinguishes internal nodes from leaves so the tree is
    immune to the Bitcoin-style second-preimage attack.
    """
    return _sha3(f"{_NODE_TAG}{left}|{right}")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class MerkleNode:
    """A node in the Merkle tree.

    Leaf nodes have a non-None block_id and no children.
    Internal nodes have children and block_id=None.
    ``content_hash`` is retained on leaves so ``verify_tree`` can recompute
    the leaf hash from its inputs instead of blindly trusting it.
    """

    hash: str
    left: Optional[MerkleNode]
    right: Optional[MerkleNode]
    block_id: Optional[str]
    content_hash: Optional[str] = None


# ---------------------------------------------------------------------------
# MerkleTree
# ---------------------------------------------------------------------------


class MerkleTree:
    """Merkle tree providing membership proofs for block-store blocks.

    The tree is rebuilt from scratch whenever its leaf set changes
    (add_block / remove_block / build). This is O(n) but keeps the
    implementation simple and correct. For a store with millions of blocks
    an incremental approach would be warranted; at typical mind-mem scales
    (thousands of blocks) a full rebuild is fast enough.

    Internal nodes: SHA3-512(left_hash + right_hash)
    Leaf nodes:     SHA3-512(content_hash)
    Odd leaves:     last leaf duplicated (standard convention)
    """

    def __init__(self) -> None:
        self._root: Optional[MerkleNode] = None
        # Ordered list of (block_id, content_hash) — preserved insertion order
        self._leaves: list[tuple[str, str]] = []
        # Fast index: block_id → position in _leaves
        self._leaf_index: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def root_hash(self) -> str:
        """Current root hash of the tree, or '' for an empty tree."""
        return self._root.hash if self._root is not None else ""

    @property
    def size(self) -> int:
        """Number of leaf blocks in the tree."""
        return len(self._leaves)

    def build(self, block_hashes: list[tuple[str, str]]) -> MerkleNode:
        """Build (or rebuild) the tree from a list of (block_id, content_hash) pairs.

        Replaces any previously built tree. Returns the root node.
        """
        self._leaves = list(block_hashes)
        self._leaf_index = {bid: i for i, (bid, _) in enumerate(self._leaves)}
        self._root = self._build_tree(self._leaves)
        return self._root  # type: ignore[return-value]

    def get_proof(self, block_id: str) -> list[tuple[str, str]]:
        """Return a Merkle proof for the given block.

        Returns a list of (sibling_hash, direction) pairs where direction
        is "left" or "right" — indicating which side the sibling sits on.

        Raises KeyError if block_id is not in the tree.
        """
        if block_id not in self._leaf_index:
            raise KeyError(f"Block not found in tree: {block_id!r}")
        if not self._leaves:
            return []

        # Single-leaf tree: no siblings exist, proof is empty
        if len(self._leaves) == 1:
            return []

        leaf_nodes = self._build_leaf_nodes(self._leaves)
        # Pad to even count (duplicate last)
        padded = _pad_to_even(leaf_nodes)

        proof: list[tuple[str, str]] = []
        index = self._leaf_index[block_id]
        _collect_proof(padded, index, proof)
        return proof

    def verify_proof(
        self,
        block_id: str,
        content_hash: str,
        proof: list[tuple[str, str]],
        root_hash: str,
    ) -> bool:
        """Verify a block's membership against a known root hash.

        Parameters
        ----------
        block_id:
            Identifier of the block (not used in hash computation, kept for
            API symmetry and future extension).
        content_hash:
            The claimed content hash of the block.
        proof:
            Proof path as returned by get_proof().
        root_hash:
            The trusted root hash to verify against.
        """
        current = _leaf_hash(content_hash, block_id)
        for sibling_hash, direction in proof:
            if direction == "right":
                current = _node_hash(current, sibling_hash)
            else:
                current = _node_hash(sibling_hash, current)
        return current == root_hash

    def verify_tree(self) -> bool:
        """Verify the full internal integrity of the tree.

        Recomputes every node hash bottom-up and compares to the stored
        values. Returns True if the tree is intact.
        """
        if self._root is None:
            return True
        return _verify_node(self._root)

    def add_block(self, block_id: str, content_hash: str) -> None:
        """Add a new block and rebuild the tree."""
        if block_id in self._leaf_index:
            raise KeyError(f"Block already exists: {block_id!r}")
        self._leaves.append((block_id, content_hash))
        self._leaf_index[block_id] = len(self._leaves) - 1
        self._root = self._build_tree(self._leaves)
        _log.debug("add_block", block_id=block_id, size=self.size)

    def remove_block(self, block_id: str) -> None:
        """Remove a block and rebuild the tree.

        Raises KeyError if block_id is not present.
        """
        if block_id not in self._leaf_index:
            raise KeyError(f"Block not found: {block_id!r}")
        idx = self._leaf_index.pop(block_id)
        self._leaves.pop(idx)
        # Reindex everything after the removed position
        for i in range(idx, len(self._leaves)):
            self._leaf_index[self._leaves[i][0]] = i
        self._root = self._build_tree(self._leaves)
        _log.debug("remove_block", block_id=block_id, size=self.size)

    def export_json(self) -> str:
        """Serialize the tree state to a JSON string.

        The serialized form is minimal: the ordered leaf list plus the
        root hash. The tree can be fully reconstructed from the leaf list.
        """
        payload = {
            "root_hash": self.root_hash,
            "leaves": [{"block_id": bid, "content_hash": ch} for bid, ch in self._leaves],
        }
        return json.dumps(payload, separators=(",", ":"))

    def import_json(self, data: str) -> None:
        """Restore tree state from a JSON string produced by export_json.

        Verifies that the stored root hash matches the hash recomputed from
        the imported leaves. A mismatch indicates tampering or a corrupted
        export; callers should treat the import as failed.
        """
        payload = json.loads(data)
        leaves = [(entry["block_id"], entry["content_hash"]) for entry in payload["leaves"]]
        self.build(leaves)
        expected_root = payload.get("root_hash", "")
        if expected_root and expected_root != self.root_hash:
            raise ValueError(f"Merkle import root mismatch: stored={expected_root[:16]}… recomputed={self.root_hash[:16]}…")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_leaf_nodes(leaves: list[tuple[str, str]]) -> list[MerkleNode]:
        return [
            MerkleNode(
                hash=_leaf_hash(ch, bid),
                left=None,
                right=None,
                block_id=bid,
                content_hash=ch,
            )
            for bid, ch in leaves
        ]

    @staticmethod
    def _build_tree(leaves: list[tuple[str, str]]) -> Optional[MerkleNode]:
        """Build and return the root node, or None for an empty leaf list."""
        if not leaves:
            return None
        nodes = MerkleTree._build_leaf_nodes(leaves)
        if len(nodes) == 1:
            return nodes[0]
        return _build_from_nodes(_pad_to_even(nodes))


# ---------------------------------------------------------------------------
# Tree construction helpers (module-level, not methods, for clarity)
# ---------------------------------------------------------------------------


def _pad_to_even(nodes: list[MerkleNode]) -> list[MerkleNode]:
    """Duplicate the last node if the list is odd-length."""
    if len(nodes) % 2 == 1:
        last = nodes[-1]
        # Duplicate: same hash, no block_id (it's a phantom copy)
        nodes = nodes + [MerkleNode(hash=last.hash, left=None, right=None, block_id=None)]
    return nodes


def _build_from_nodes(nodes: list[MerkleNode]) -> MerkleNode:
    """Recursively build tree levels until a single root remains."""
    while len(nodes) > 1:
        nodes = _pad_to_even(nodes)
        next_level: list[MerkleNode] = []
        for i in range(0, len(nodes), 2):
            left = nodes[i]
            right = nodes[i + 1]
            parent = MerkleNode(
                hash=_node_hash(left.hash, right.hash),
                left=left,
                right=right,
                block_id=None,
            )
            next_level.append(parent)
        nodes = next_level
    return nodes[0]


def _collect_proof(
    leaves: list[MerkleNode],
    index: int,
    proof: list[tuple[str, str]],
) -> None:
    """Walk up the tree collecting sibling hashes for the proof path."""
    if len(leaves) == 1:
        return

    # Pad this level to even
    padded = _pad_to_even(leaves)

    # Determine sibling index and direction label
    if index % 2 == 0:
        sibling_idx = index + 1
        direction = "right"
    else:
        sibling_idx = index - 1
        direction = "left"

    proof.append((padded[sibling_idx].hash, direction))

    # Build the parent level and recurse
    next_level: list[MerkleNode] = []
    for i in range(0, len(padded), 2):
        left = padded[i]
        right = padded[i + 1]
        next_level.append(
            MerkleNode(
                hash=_node_hash(left.hash, right.hash),
                left=left,
                right=right,
                block_id=None,
            )
        )
    _collect_proof(next_level, index // 2, proof)


def _verify_node(node: MerkleNode) -> bool:
    """Recursively verify that every node hash matches its inputs.

    Leaf verification recomputes ``_leaf_hash(content_hash, block_id)`` from
    the stored inputs so a tampered leaf hash is caught. Internal nodes
    recompute ``_node_hash(left, right)``.
    """
    if node.left is None and node.right is None:
        if node.content_hash is None:
            # Phantom padding leaf duplicated from its sibling — hash was
            # copied from the real leaf and has nothing independent to
            # verify against. Trust is bootstrapped from the sibling check.
            return True
        expected_leaf = _leaf_hash(node.content_hash, node.block_id or "")
        return node.hash == expected_leaf

    if node.left is None or node.right is None:
        # Malformed tree
        return False

    if not _verify_node(node.left) or not _verify_node(node.right):
        return False

    expected = _node_hash(node.left.hash, node.right.hash)
    return node.hash == expected
