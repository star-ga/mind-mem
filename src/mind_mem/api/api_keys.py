"""Per-agent API key store for the mind-mem REST API.

Keys are prefixed with ``mmk_live_`` (production) or ``mmk_test_``
(non-production). Only SHA-256 hashes are persisted — the raw key is
shown once at creation time and never stored.

Storage backend: SQLite (stdlib only, zero extra deps).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LIVE_PREFIX = "mmk_live_"
_TEST_PREFIX = "mmk_test_"
_RAW_KEY_BYTES = 32  # 256-bit entropy → 64 hex chars after prefix


# ---------------------------------------------------------------------------
# APIKeyStore
# ---------------------------------------------------------------------------


class APIKeyStore:
    """SQLite-backed store for per-agent API keys.

    Only the SHA-256 hash of each key is persisted. The raw key is returned
    once by :meth:`create` and never retrievable again.

    Args:
        db_path: Filesystem path for the SQLite database file.
                 The parent directory is created if it does not exist.
        production: When *True* raw keys carry the ``mmk_live_`` prefix;
                    when *False* they carry ``mmk_test_``.  Defaults to
                    ``True`` so production deployments get the right prefix
                    without explicit configuration.
    """

    def __init__(self, db_path: str, production: bool = True) -> None:
        self._db_path = db_path
        self._production = production
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._init_db()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create(
        self,
        agent_id: str,
        scopes: list[str],
        expires_in_days: int = 90,
    ) -> str:
        """Create a new API key for *agent_id*.

        Args:
            agent_id:        Identifier for the agent that owns this key.
            scopes:          Access scopes for this key (e.g. ``["user"]``
                             or ``["user", "admin"]``).
            expires_in_days: Validity period in calendar days (default 90).

        Returns:
            The raw API key string (shown once — store it securely).
        """
        prefix = _LIVE_PREFIX if self._production else _TEST_PREFIX
        raw_key = prefix + secrets.token_hex(_RAW_KEY_BYTES)
        key_id = secrets.token_hex(16)
        key_hash = _sha256(raw_key)
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(days=expires_in_days)

        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO api_keys
                        (key_id, key_hash, agent_id, scopes, created_at, expires_at, revoked)
                    VALUES (?, ?, ?, ?, ?, ?, 0)
                    """,
                    (
                        key_id,
                        key_hash,
                        agent_id,
                        json.dumps(scopes),
                        now.isoformat(),
                        expires_at.isoformat(),
                    ),
                )
                conn.commit()
            finally:
                conn.close()
        return raw_key

    def verify(self, raw_key: str) -> dict[str, Any] | None:
        """Return the agent record for *raw_key*, or ``None`` if invalid.

        Returns ``None`` when the key does not exist, has been revoked,
        or has passed its expiry date.
        """
        key_hash = _sha256(raw_key)
        now = datetime.now(timezone.utc).isoformat()

        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    """
                    SELECT key_id, agent_id, scopes, created_at, expires_at, revoked
                    FROM api_keys
                    WHERE key_hash = ? AND revoked = 0 AND expires_at > ?
                    """,
                    (key_hash, now),
                ).fetchone()
            finally:
                conn.close()

        if row is None:
            return None
        return {
            "key_id": row[0],
            "agent_id": row[1],
            "scopes": json.loads(row[2]),
            "created_at": row[3],
            "expires_at": row[4],
        }

    def revoke(self, key_id: str) -> bool:
        """Revoke the key with *key_id*.

        Returns:
            ``True`` if a non-revoked key was found and revoked,
            ``False`` if the key did not exist or was already revoked.
        """
        with self._lock:
            conn = self._connect()
            try:
                cursor = conn.execute(
                    "UPDATE api_keys SET revoked = 1 WHERE key_id = ? AND revoked = 0",
                    (key_id,),
                )
                conn.commit()
                return cursor.rowcount > 0
            finally:
                conn.close()

    def list_keys(self, agent_id: str = "") -> list[dict[str, Any]]:
        """List API keys, optionally filtered by *agent_id*.

        Args:
            agent_id: When non-empty, only keys belonging to this agent are
                      returned.  Pass an empty string to list all keys.

        Returns:
            List of dicts with ``key_id``, ``agent_id``, ``scopes``,
            ``created_at``, ``expires_at``, and ``revoked`` fields.
            The ``key_hash`` column is never included.
        """
        with self._lock:
            conn = self._connect()
            try:
                if agent_id:
                    rows = conn.execute(
                        """
                        SELECT key_id, agent_id, scopes, created_at, expires_at, revoked
                        FROM api_keys WHERE agent_id = ?
                        ORDER BY created_at DESC
                        """,
                        (agent_id,),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        """
                        SELECT key_id, agent_id, scopes, created_at, expires_at, revoked
                        FROM api_keys
                        ORDER BY created_at DESC
                        """
                    ).fetchall()
            finally:
                conn.close()

        return [
            {
                "key_id": r[0],
                "agent_id": r[1],
                "scopes": json.loads(r[2]),
                "created_at": r[3],
                "expires_at": r[4],
                "revoked": bool(r[5]),
            }
            for r in rows
        ]

    def rotate(self, old_key_id: str) -> str:
        """Rotate a key: revoke the old one and issue a new key for the same agent.

        Both operations execute inside a single SQLite transaction so a crash
        between them can never leave both the old and new key simultaneously
        valid (which would constitute a privilege-elevation window).

        Args:
            old_key_id: The ``key_id`` of the key to replace.

        Returns:
            Raw value of the new key (shown once).

        Raises:
            KeyError: When *old_key_id* does not exist.
        """
        prefix = _LIVE_PREFIX if self._production else _TEST_PREFIX
        new_raw_key = prefix + secrets.token_hex(_RAW_KEY_BYTES)
        new_key_id = secrets.token_hex(16)
        new_key_hash = _sha256(new_raw_key)
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(days=90)

        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT agent_id, scopes FROM api_keys WHERE key_id = ?",
                    (old_key_id,),
                ).fetchone()
                if row is None:
                    raise KeyError(f"API key not found: {old_key_id}")

                agent_id, scopes_json = row

                # Insert new key and revoke old key atomically.
                conn.execute(
                    """
                    INSERT INTO api_keys
                        (key_id, key_hash, agent_id, scopes, created_at, expires_at, revoked)
                    VALUES (?, ?, ?, ?, ?, ?, 0)
                    """,
                    (
                        new_key_id,
                        new_key_hash,
                        agent_id,
                        scopes_json,
                        now.isoformat(),
                        expires_at.isoformat(),
                    ),
                )
                conn.execute(
                    "UPDATE api_keys SET revoked = 1 WHERE key_id = ? AND revoked = 0",
                    (old_key_id,),
                )
                conn.commit()
            finally:
                conn.close()

        return new_raw_key

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS api_keys (
                        key_id     TEXT PRIMARY KEY,
                        key_hash   TEXT NOT NULL UNIQUE,
                        agent_id   TEXT NOT NULL,
                        scopes     TEXT NOT NULL DEFAULT '[]',
                        created_at TEXT NOT NULL,
                        expires_at TEXT NOT NULL,
                        revoked    INTEGER NOT NULL DEFAULT 0
                    )
                    """
                )
                conn.commit()
            finally:
                conn.close()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sha256(value: str) -> str:
    """Return the hex SHA-256 digest of a UTF-8-encoded string."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _constant_time_equal(a: str, b: str) -> bool:
    """Constant-time string equality to prevent timing attacks."""
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))
