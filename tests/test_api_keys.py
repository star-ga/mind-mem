"""Tests for APIKeyStore in src/mind_mem/api/api_keys.py."""

from __future__ import annotations

import hashlib
import sqlite3
from typing import Any

import pytest

from mind_mem.api.api_keys import APIKeyStore, _sha256

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def store(tmp_path: Any) -> APIKeyStore:
    """Production-mode store backed by a tmp SQLite database."""
    return APIKeyStore(str(tmp_path / "api_keys.db"), production=True)


@pytest.fixture()
def test_store(tmp_path: Any) -> APIKeyStore:
    """Non-production (test) mode store."""
    return APIKeyStore(str(tmp_path / "api_keys_test.db"), production=False)


# ---------------------------------------------------------------------------
# Key creation
# ---------------------------------------------------------------------------


class TestCreate:
    def test_returns_raw_key(self, store: APIKeyStore) -> None:
        key = store.create("agent-1", ["user"])
        assert isinstance(key, str)
        assert len(key) > 10

    def test_live_prefix_in_production(self, store: APIKeyStore) -> None:
        key = store.create("agent-1", ["user"])
        assert key.startswith("mmk_live_")

    def test_test_prefix_in_non_production(self, test_store: APIKeyStore) -> None:
        key = test_store.create("agent-1", ["user"])
        assert key.startswith("mmk_test_")

    def test_raw_key_not_stored_in_db(self, store: APIKeyStore, tmp_path: Any) -> None:
        """Verify the SQLite database does NOT contain the raw key — only hash."""
        raw_key = store.create("agent-x", ["user"])
        conn = sqlite3.connect(str(tmp_path / "api_keys.db"))
        rows = conn.execute("SELECT key_hash FROM api_keys").fetchall()
        conn.close()

        stored_hashes = [r[0] for r in rows]
        assert raw_key not in stored_hashes
        expected_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        assert expected_hash in stored_hashes

    def test_sha256_hash_stored(self, store: APIKeyStore, tmp_path: Any) -> None:
        raw_key = store.create("agent-hash-check", ["admin"])
        expected = _sha256(raw_key)
        conn = sqlite3.connect(str(tmp_path / "api_keys.db"))
        row = conn.execute("SELECT key_hash FROM api_keys WHERE agent_id = ?", ("agent-hash-check",)).fetchone()
        conn.close()
        assert row is not None
        assert row[0] == expected

    def test_each_key_is_unique(self, store: APIKeyStore) -> None:
        k1 = store.create("agent-1", ["user"])
        k2 = store.create("agent-1", ["user"])
        assert k1 != k2

    def test_custom_expiry_stored(self, store: APIKeyStore, tmp_path: Any) -> None:
        store.create("agent-exp", ["user"], expires_in_days=7)
        conn = sqlite3.connect(str(tmp_path / "api_keys.db"))
        row = conn.execute(
            "SELECT expires_at, created_at FROM api_keys WHERE agent_id = ?", ("agent-exp",)
        ).fetchone()
        conn.close()
        assert row is not None
        # Rough check: expires_at should be after created_at
        assert row[0] > row[1]


# ---------------------------------------------------------------------------
# Verify round-trip
# ---------------------------------------------------------------------------


class TestVerify:
    def test_verify_returns_record_for_valid_key(self, store: APIKeyStore) -> None:
        raw_key = store.create("agent-v", ["user", "admin"])
        record = store.verify(raw_key)
        assert record is not None
        assert record["agent_id"] == "agent-v"
        assert "admin" in record["scopes"]

    def test_verify_returns_none_for_unknown_key(self, store: APIKeyStore) -> None:
        assert store.verify("mmk_live_" + "x" * 64) is None

    def test_verify_returns_none_after_revoke(self, store: APIKeyStore) -> None:
        raw_key = store.create("agent-r", ["user"])
        record = store.verify(raw_key)
        assert record is not None
        store.revoke(record["key_id"])
        assert store.verify(raw_key) is None

    def test_verify_returns_none_for_expired_key(self, store: APIKeyStore, tmp_path: Any) -> None:
        """Write a key with expires_at in the past directly into the DB."""
        raw_key = store.create("agent-expired", ["user"])
        key_hash = _sha256(raw_key)
        conn = sqlite3.connect(str(tmp_path / "api_keys.db"))
        conn.execute(
            "UPDATE api_keys SET expires_at = ? WHERE key_hash = ?",
            ("2000-01-01T00:00:00+00:00", key_hash),
        )
        conn.commit()
        conn.close()
        assert store.verify(raw_key) is None


# ---------------------------------------------------------------------------
# Revoke
# ---------------------------------------------------------------------------


class TestRevoke:
    def test_revoke_existing_key_returns_true(self, store: APIKeyStore) -> None:
        raw_key = store.create("agent-rev", ["user"])
        key_id = store.verify(raw_key)["key_id"]  # type: ignore[index]
        assert store.revoke(key_id) is True

    def test_revoke_non_existent_returns_false(self, store: APIKeyStore) -> None:
        assert store.revoke("nonexistent-key-id") is False

    def test_revoke_already_revoked_returns_false(self, store: APIKeyStore) -> None:
        raw_key = store.create("agent-dbl", ["user"])
        key_id = store.verify(raw_key)["key_id"]  # type: ignore[index]
        store.revoke(key_id)
        assert store.revoke(key_id) is False


# ---------------------------------------------------------------------------
# Rotate
# ---------------------------------------------------------------------------


class TestRotate:
    def test_rotate_returns_new_raw_key(self, store: APIKeyStore) -> None:
        old_key = store.create("agent-rot", ["user"])
        old_key_id = store.verify(old_key)["key_id"]  # type: ignore[index]
        new_key = store.rotate(old_key_id)
        assert new_key != old_key
        assert new_key.startswith("mmk_live_")

    def test_old_key_revoked_after_rotate(self, store: APIKeyStore) -> None:
        old_key = store.create("agent-rot2", ["user"])
        old_key_id = store.verify(old_key)["key_id"]  # type: ignore[index]
        store.rotate(old_key_id)
        assert store.verify(old_key) is None

    def test_new_key_valid_after_rotate(self, store: APIKeyStore) -> None:
        old_key = store.create("agent-rot3", ["admin"])
        old_key_id = store.verify(old_key)["key_id"]  # type: ignore[index]
        new_key = store.rotate(old_key_id)
        record = store.verify(new_key)
        assert record is not None
        assert record["agent_id"] == "agent-rot3"
        assert "admin" in record["scopes"]

    def test_rotate_nonexistent_raises_key_error(self, store: APIKeyStore) -> None:
        with pytest.raises(KeyError):
            store.rotate("no-such-id")


# ---------------------------------------------------------------------------
# List keys
# ---------------------------------------------------------------------------


class TestListKeys:
    def test_list_all_returns_all_keys(self, store: APIKeyStore) -> None:
        store.create("a1", ["user"])
        store.create("a2", ["admin"])
        keys = store.list_keys()
        assert len(keys) >= 2

    def test_list_filtered_by_agent_id(self, store: APIKeyStore) -> None:
        store.create("agent-list-1", ["user"])
        store.create("agent-list-1", ["user"])
        store.create("agent-list-2", ["user"])
        keys = store.list_keys(agent_id="agent-list-1")
        assert all(k["agent_id"] == "agent-list-1" for k in keys)
        assert len(keys) == 2

    def test_list_empty_for_unknown_agent(self, store: APIKeyStore) -> None:
        store.create("real-agent", ["user"])
        keys = store.list_keys(agent_id="ghost-agent")
        assert keys == []

    def test_key_hash_not_exposed_in_list(self, store: APIKeyStore) -> None:
        store.create("agent-sec", ["user"])
        keys = store.list_keys()
        for k in keys:
            assert "key_hash" not in k

    def test_revoked_field_present(self, store: APIKeyStore) -> None:
        raw_key = store.create("agent-rfl", ["user"])
        key_id = store.verify(raw_key)["key_id"]  # type: ignore[index]
        store.revoke(key_id)
        keys = store.list_keys(agent_id="agent-rfl")
        revoked_key = next(k for k in keys if k["key_id"] == key_id)
        assert revoked_key["revoked"] is True
