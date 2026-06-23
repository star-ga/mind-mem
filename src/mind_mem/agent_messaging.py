# Copyright 2026 STARGA, Inc.
"""Agent-to-agent messaging over the shared mind-mem block store (v4.0.19).

The sanctioned cross-agent / cross-node comm channel *is* the block store.
One agent "sends" by writing an ``MSG-`` block; another "receives" by
recalling over the ``memory/MESSAGES.md`` corpus (indexed for both the
SQLite default and the Postgres federation hub). There is no separate
message-bus daemon — delivery is durability + recall, which is why it
works identically from any CLI (claude / codex / gemini / grok …) against
the shared ``.193`` Postgres store.

This module is the pure, transport-free core behind ``mm send`` /
``mm inbox``. It mirrors :mod:`mind_mem.inbox`'s block-shape +
``stamp_transform_hash`` write pattern. Blocks are built immutably (a
fresh ``dict`` per send), so callers never mutate shared state.
"""

from __future__ import annotations

import os
import secrets
from datetime import datetime, timezone
from typing import Optional

from .observability import get_logger

_log = get_logger("agent_messaging")

__all__ = [
    "MESSAGE_TYPE",
    "build_message_block",
    "read_inbox",
    "send_message",
]

# Block ``type`` field used for every agent message. ``mm inbox`` recalls
# against this token so messages are separable from other corpus blocks.
MESSAGE_TYPE = "AgentMessage"


def _now_iso() -> str:
    """UTC timestamp with second precision (matches inbox.py's format)."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def build_message_block(
    text: str,
    *,
    to: Optional[str] = None,
    sender: Optional[str] = None,
    subject: Optional[str] = None,
    timestamp: Optional[str] = None,
    nonce: Optional[str] = None,
) -> dict:
    """Build an immutable ``MSG-`` block for *text*.

    The block id is ``MSG-<ts>-<rand>`` so it routes through the
    ``MSG`` entry in ``_BLOCK_PREFIX_MAP`` to ``memory/MESSAGES.md``.
    ``timestamp`` / ``nonce`` are injectable for deterministic tests;
    in production they default to a UTC stamp + 8 hex chars.

    Returns a new dict every call — no shared mutable state.

    Raises:
        ValueError: *text* is empty / whitespace-only.
    """
    if not text or not text.strip():
        raise ValueError("message text must be a non-empty string")

    ts = timestamp or _now_iso()
    rand = nonce or secrets.token_hex(4)
    block_id = f"MSG-{ts}-{rand}"

    block: dict[str, object] = {
        "_id": block_id,
        "type": MESSAGE_TYPE,
        "Statement": text,
        "Timestamp": ts,
        "Status": "active",
    }
    # Optional routing fields, only set when provided (keeps blocks tidy).
    if to:
        block["To"] = to
    if sender:
        block["From"] = sender
    if subject:
        block["Subject"] = subject
    return block


def send_message(
    workspace: str,
    text: str,
    *,
    to: Optional[str] = None,
    sender: Optional[str] = None,
    subject: Optional[str] = None,
    reindex: bool = True,
) -> str:
    """Write an agent message and return its block id.

    The message is written to the configured block store (SQLite/markdown
    default or the Postgres federation hub) and — on the markdown default,
    where recall reads a pre-built index — the SQLite index is rebuilt so
    the message is immediately visible to ``mm inbox`` / ``recall``. On
    Postgres the write is itself the index, so ``reindex`` is a no-op cost.

    Raises:
        ValueError: *text* is empty (from :func:`build_message_block`).
    """
    # Lazy imports — storage/index factories are heavy; keep module import
    # cheap so test collection that only touches block-building stays fast.
    from .pipeline_hash import stamp_transform_hash
    from .storage import get_block_store

    block = build_message_block(text, to=to, sender=sender, subject=subject)
    store = get_block_store(workspace)
    written_id = store.write_block(stamp_transform_hash(workspace, block))

    if reindex:
        # Only the markdown corpus needs an explicit index rebuild for the
        # new block to be recallable; build_index is a no-op / cheap when
        # the file is unchanged, and read-only-safe on PG-backed workspaces.
        try:
            from .sqlite_index import build_index

            build_index(workspace)
        except Exception as exc:  # pragma: no cover - index rebuild is best-effort
            # A send must not fail just because the local index couldn't be
            # refreshed; the block is durably written either way — log it
            # rather than swallow it silently (avoids B110 / our no-silent-swallow rule).
            _log.debug("send_message: best-effort index rebuild skipped: %s", exc)
    return written_id


def read_inbox(
    workspace: str,
    *,
    to: Optional[str] = None,
    since: Optional[str] = None,
    limit: int = 20,
) -> list[dict]:
    """Return agent messages addressed to *to* (plus broadcasts), newest-first.

    Receiving mail is *enumeration*, not BM25 search: it routes through the
    backend-aware ``storage.iter_active_blocks`` so the full block fields
    (``To`` / ``From`` / ``Subject`` / ``type``) are preserved — recall's
    lean projection drops them — and so it works identically on the SQLite
    markdown default and the Postgres federation hub.

    Filtering:
      * keep only ``AgentMessage`` blocks (``MSG-`` ids);
      * when *to* is set, keep messages addressed to *to* plus broadcasts
        (no ``To`` field), so a recipient sees its own mail + broadcasts;
      * when *since* is set, keep messages whose ``Timestamp`` (or ``Date``)
        is >= *since* (ISO-8601 string compare — works for both the
        ``YYYYMMDDTHHMMSSZ`` stamp and ``YYYY-MM-DD`` dates).

    Returns at most *limit* blocks, newest-first.
    """
    from .storage import iter_active_blocks

    blocks = iter_active_blocks(workspace)

    def _is_message(b: dict) -> bool:
        if str(b.get("type", "")).strip() == MESSAGE_TYPE:
            return True
        # Fall back to the id prefix in case a backend strips/renames type.
        return str(b.get("_id", "")).startswith("MSG-")

    messages = [b for b in blocks if _is_message(b)]

    if to:
        messages = [b for b in messages if not b.get("To") or str(b.get("To")) == to]

    if since:

        def _stamp(b: dict) -> str:
            return str(b.get("Timestamp") or b.get("Date") or "")

        messages = [b for b in messages if _stamp(b) >= since]

    # Newest-first by Timestamp (falls back to id, which is timestamp-prefixed).
    messages.sort(
        key=lambda b: str(b.get("Timestamp") or b.get("_id") or ""),
        reverse=True,
    )
    return messages[:limit]


# Re-exported helper so callers don't depend on os import details.
def messages_file(workspace: str) -> str:
    """Absolute path of the markdown messages corpus for *workspace*."""
    return os.path.join(workspace, "memory", "MESSAGES.md")
