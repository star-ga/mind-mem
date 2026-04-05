# Copyright 2026 STARGA, Inc.
"""Tests for HashChainV2 — SHA3-512 per-block hash chain with SQLite backend."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import pytest

from mind_mem.hash_chain_v2 import (
    GENESIS_HASH,
    HashChainV2,
    HashEntry,
    MigrationError,
    convert_from_v1,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha3(data: str) -> str:
    return hashlib.sha3_512(data.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db_path(tmp_path: Path) -> str:
    return str(tmp_path / "chain_v2.db")


@pytest.fixture
def chain(db_path: str) -> HashChainV2:
    return HashChainV2(db_path)


@pytest.fixture
def populated_chain(chain: HashChainV2) -> HashChainV2:
    """Chain with 5 entries across 3 blocks."""
    chain.append("block-a", "create", "first content")
    chain.append("block-b", "create", "second content")
    chain.append("block-a", "update", "updated content")
    chain.append("block-c", "create", "third content")
    chain.append("block-b", "delete", "")
    return chain


# ---------------------------------------------------------------------------
# 1. HashEntry dataclass
# ---------------------------------------------------------------------------


class TestHashEntry:
    def test_is_frozen(self, chain: HashChainV2) -> None:
        """HashEntry must be immutable — frozen dataclass."""
        entry = chain.append("blk", "create", "data")
        with pytest.raises((AttributeError, TypeError)):
            entry.action = "tamper"  # type: ignore[misc]

    def test_fields_present(self, chain: HashChainV2) -> None:
        """All seven required fields must be present."""
        entry = chain.append("blk", "create", "data")
        for field in ("entry_id", "timestamp", "block_id", "action", "content_hash", "previous_hash", "entry_hash"):
            assert hasattr(entry, field), f"Missing field: {field}"

    def test_content_hash_is_sha3_512(self, chain: HashChainV2) -> None:
        """content_hash must be a 128-hex-char SHA3-512 digest."""
        entry = chain.append("blk", "create", "hello")
        expected = _sha3("hello")
        assert entry.content_hash == expected
        assert len(entry.content_hash) == 128

    def test_entry_hash_length(self, chain: HashChainV2) -> None:
        """entry_hash must be 128 hex chars (SHA3-512)."""
        entry = chain.append("blk", "create", "x")
        assert len(entry.entry_hash) == 128

    def test_serialization_round_trip(self, chain: HashChainV2) -> None:
        """asdict round-trip must preserve all fields."""
        entry = chain.append("blk", "create", "payload")
        d = asdict(entry)
        reconstructed = HashEntry(**d)
        assert reconstructed == entry


# ---------------------------------------------------------------------------
# 2. append
# ---------------------------------------------------------------------------


class TestAppend:
    def test_first_entry_previous_hash_is_genesis(self, chain: HashChainV2) -> None:
        entry = chain.append("b1", "create", "data")
        assert entry.previous_hash == GENESIS_HASH

    def test_second_entry_previous_hash_links_to_first(self, chain: HashChainV2) -> None:
        first = chain.append("b1", "create", "data")
        second = chain.append("b1", "update", "new data")
        assert second.previous_hash == first.entry_hash

    def test_append_increments_length(self, chain: HashChainV2) -> None:
        assert chain.length == 0
        chain.append("b1", "create", "x")
        assert chain.length == 1
        chain.append("b1", "update", "y")
        assert chain.length == 2

    def test_entry_id_is_unique(self, chain: HashChainV2) -> None:
        ids = {chain.append(f"b{i}", "create", str(i)).entry_id for i in range(10)}
        assert len(ids) == 10

    def test_returns_hash_entry(self, chain: HashChainV2) -> None:
        result = chain.append("blk", "create", "data")
        assert isinstance(result, HashEntry)

    def test_empty_content_is_accepted(self, chain: HashChainV2) -> None:
        entry = chain.append("blk", "delete", "")
        assert entry.content_hash == _sha3("")


# ---------------------------------------------------------------------------
# 3. verify_entry
# ---------------------------------------------------------------------------


class TestVerifyEntry:
    def test_valid_entry_passes(self, chain: HashChainV2) -> None:
        entry = chain.append("b1", "create", "hello")
        assert chain.verify_entry(entry) is True

    def test_tampered_content_hash_fails(self, chain: HashChainV2) -> None:
        entry = chain.append("b1", "create", "hello")
        bad = HashEntry(
            entry_id=entry.entry_id,
            timestamp=entry.timestamp,
            block_id=entry.block_id,
            action=entry.action,
            content_hash="a" * 128,  # tampered
            previous_hash=entry.previous_hash,
            entry_hash=entry.entry_hash,
        )
        assert chain.verify_entry(bad) is False

    def test_tampered_entry_hash_fails(self, chain: HashChainV2) -> None:
        entry = chain.append("b1", "create", "hello")
        bad = HashEntry(
            entry_id=entry.entry_id,
            timestamp=entry.timestamp,
            block_id=entry.block_id,
            action=entry.action,
            content_hash=entry.content_hash,
            previous_hash=entry.previous_hash,
            entry_hash="b" * 128,  # tampered
        )
        assert chain.verify_entry(bad) is False


# ---------------------------------------------------------------------------
# 4. verify_chain
# ---------------------------------------------------------------------------


class TestVerifyChain:
    def test_empty_chain_is_valid(self, chain: HashChainV2) -> None:
        valid, broken_at = chain.verify_chain()
        assert valid is True
        assert broken_at == -1

    def test_intact_chain_is_valid(self, populated_chain: HashChainV2) -> None:
        valid, broken_at = populated_chain.verify_chain()
        assert valid is True
        assert broken_at == -1

    def test_returns_tuple_bool_int(self, chain: HashChainV2) -> None:
        result = chain.verify_chain()
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], int)

    def test_tampered_db_detected(self, db_path: str, chain: HashChainV2) -> None:
        """Direct SQLite write that breaks the chain must be detected."""
        chain.append("b1", "create", "original")
        chain.append("b1", "update", "second")

        # Corrupt the first entry's content_hash in the database directly
        conn = sqlite3.connect(db_path)
        conn.execute(
            "UPDATE hash_chain SET content_hash = ? WHERE rowid = 1",
            ("c" * 128,),
        )
        conn.commit()
        conn.close()

        # Reload chain from the same db
        reloaded = HashChainV2(db_path)
        valid, broken_at = reloaded.verify_chain()
        assert valid is False
        assert broken_at == 0  # first entry (0-indexed)


# ---------------------------------------------------------------------------
# 5. get_block_chain
# ---------------------------------------------------------------------------


class TestGetBlockChain:
    def test_returns_only_matching_block(self, populated_chain: HashChainV2) -> None:
        entries = populated_chain.get_block_chain("block-a")
        assert all(e.block_id == "block-a" for e in entries)

    def test_block_a_has_two_entries(self, populated_chain: HashChainV2) -> None:
        entries = populated_chain.get_block_chain("block-a")
        assert len(entries) == 2

    def test_unknown_block_returns_empty(self, chain: HashChainV2) -> None:
        assert chain.get_block_chain("nonexistent") == []

    def test_entries_ordered_by_insertion(self, chain: HashChainV2) -> None:
        chain.append("blk", "create", "first")
        chain.append("blk", "update", "second")
        chain.append("blk", "update", "third")
        entries = chain.get_block_chain("blk")
        actions = [e.action for e in entries]
        assert actions == ["create", "update", "update"]


# ---------------------------------------------------------------------------
# 6. get_latest
# ---------------------------------------------------------------------------


class TestGetLatest:
    def test_returns_last_n(self, populated_chain: HashChainV2) -> None:
        latest = populated_chain.get_latest(3)
        assert len(latest) == 3

    def test_default_is_ten(self, chain: HashChainV2) -> None:
        for i in range(15):
            chain.append("b", "create", str(i))
        assert len(chain.get_latest()) == 10

    def test_empty_chain_returns_empty(self, chain: HashChainV2) -> None:
        assert chain.get_latest() == []

    def test_n_larger_than_chain_returns_all(self, chain: HashChainV2) -> None:
        chain.append("b", "create", "x")
        chain.append("b", "create", "y")
        result = chain.get_latest(100)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# 7. export_jsonl / import_jsonl
# ---------------------------------------------------------------------------


class TestExportImport:
    def test_export_produces_valid_jsonl(self, populated_chain: HashChainV2, tmp_path: Path) -> None:
        out = str(tmp_path / "export.jsonl")
        populated_chain.export_jsonl(out)
        lines = Path(out).read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 5
        for line in lines:
            obj = json.loads(line)
            assert "entry_id" in obj
            assert "entry_hash" in obj

    def test_import_restores_all_entries(self, populated_chain: HashChainV2, tmp_path: Path) -> None:
        out = str(tmp_path / "export.jsonl")
        populated_chain.export_jsonl(out)

        new_db = str(tmp_path / "restored.db")
        new_chain = HashChainV2(new_db)
        new_chain.import_jsonl(out)

        assert new_chain.length == populated_chain.length

    def test_imported_chain_verifies(self, populated_chain: HashChainV2, tmp_path: Path) -> None:
        out = str(tmp_path / "export.jsonl")
        populated_chain.export_jsonl(out)

        new_db = str(tmp_path / "restored.db")
        new_chain = HashChainV2(new_db)
        new_chain.import_jsonl(out)

        valid, broken_at = new_chain.verify_chain()
        assert valid is True

    def test_import_rejects_tampered_jsonl(self, populated_chain: HashChainV2, tmp_path: Path) -> None:
        out = str(tmp_path / "export.jsonl")
        populated_chain.export_jsonl(out)

        # Corrupt one line
        lines = Path(out).read_text(encoding="utf-8").splitlines()
        obj = json.loads(lines[0])
        obj["content_hash"] = "d" * 128
        lines[0] = json.dumps(obj)
        Path(out).write_text("\n".join(lines), encoding="utf-8")

        new_db = str(tmp_path / "bad.db")
        new_chain = HashChainV2(new_db)
        with pytest.raises(ValueError, match="tampered|invalid|corrupt"):
            new_chain.import_jsonl(out)


# ---------------------------------------------------------------------------
# 8. Migration from v1 (SHA256)
# ---------------------------------------------------------------------------


class TestMigrateFromV1:
    def _make_v1_jsonl(self, path: str) -> None:
        """Write a minimal v1-style JSONL chain file (SHA256 hashes)."""
        genesis = "0" * 64
        entries = [
            {
                "seq": 1,
                "timestamp": "2026-01-01T00:00:00+00:00",
                "operation": "create_block",
                "target": "block-one",
                "agent": "test",
                "reason": "init",
                "payload_hash": hashlib.sha256(b"some data").hexdigest(),
                "prev_hash": genesis,
                "entry_hash": "a" * 64,
            },
            {
                "seq": 2,
                "timestamp": "2026-01-01T00:01:00+00:00",
                "operation": "update_field",
                "target": "block-one",
                "agent": "test",
                "reason": "edit",
                "payload_hash": hashlib.sha256(b"other data").hexdigest(),
                "prev_hash": "a" * 64,
                "entry_hash": "b" * 64,
            },
        ]
        with open(path, "w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")

    def test_convert_produces_v2_chain(self, tmp_path: Path) -> None:
        v1_path = str(tmp_path / "v1_chain.jsonl")
        self._make_v1_jsonl(v1_path)
        new_db = str(tmp_path / "migrated.db")

        new_chain = convert_from_v1(v1_path, new_db)
        assert new_chain.length == 2

    def test_convert_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises((FileNotFoundError, MigrationError)):
            convert_from_v1(str(tmp_path / "nope.jsonl"), str(tmp_path / "out.db"))

    def test_converted_chain_has_sha3_hashes(self, tmp_path: Path) -> None:
        v1_path = str(tmp_path / "v1_chain.jsonl")
        self._make_v1_jsonl(v1_path)
        new_db = str(tmp_path / "migrated.db")

        new_chain = convert_from_v1(v1_path, new_db)
        for entry in new_chain.get_latest(100):
            # SHA3-512 produces 128 hex chars; SHA256 produces 64
            assert len(entry.entry_hash) == 128
