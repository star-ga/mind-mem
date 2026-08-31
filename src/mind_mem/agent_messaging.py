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

**A message arrives withheld.** ``IngestTier.AGENT_MESSAGE`` maps to
``Status.QUARANTINED`` in :data:`~mind_mem.enums.INITIAL_STATUS`,
reversing what earlier versions shipped. A peer agent is the standard
prompt-injection carrier, and a single-operator fleet makes the *sender*
accountable without making its *input* trusted — the text an agent
relays is frequently not its own. So the sanctioned channel keeps
working, and its content stops being silently retrievable as memory:

* ``read_inbox`` / ``mm inbox`` still show every message. Opening a
  named mailbox is an explicit act by the recipient, not a retrieval
  into its context, and a quarantine that hides mail from the addressee
  is a broken mailbox rather than a safe one.
* ``recall`` no longer returns messages. To make one part of memory,
  release it through a governance proposal, exactly as an imported
  corpus is released.
"""

from __future__ import annotations

import os
import secrets
from datetime import datetime, timezone
from typing import Optional

from .enums import INITIAL_STATUS, IngestTier, Status
from .observability import get_logger

_log = get_logger("agent_messaging")

__all__ = [
    "MESSAGE_STATUS",
    "MESSAGE_TIER",
    "MESSAGE_TYPE",
    "build_message_block",
    "read_inbox",
    "send_message",
]

# Block ``type`` field used for every agent message. ``mm inbox`` recalls
# against this token so messages are separable from other corpus blocks.
MESSAGE_TYPE = "AgentMessage"

#: The ingest tier every message is admitted under.
MESSAGE_TIER: IngestTier = IngestTier.AGENT_MESSAGE

#: The status that tier mints. Resolved from the table, not restated.
MESSAGE_STATUS: Status = INITIAL_STATUS[MESSAGE_TIER] or Status.QUARANTINED

#: Block field naming the ingest tier (same spelling as the importer's).
TIER_FIELD = "IngestTier"


def _now_iso() -> str:
    """UTC timestamp with second precision (matches inbox.py's format)."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


#: Width of the canonical stamp, ``YYYYMMDDHHMMSS``.
_STAMP_WIDTH = 14


def _normalise_stamp(raw: object) -> str:
    """Canonicalise a stamp to ``YYYYMMDDHHMMSS`` so compares are ordered.

    Two spellings meet here: :func:`build_message_block` writes the compact
    ``YYYYMMDDTHHMMSSZ`` ``Timestamp``, while the wider corpus convention
    is a dashed ``YYYY-MM-DD`` ``Date``. A raw string compare across them
    is not merely imprecise, it is inverted — at index 4 the compact stamp
    holds a digit (0x30+) and the dashed date holds ``-`` (0x2D), so every
    compact stamp sorts above every dashed date of the same year, and a
    ``since`` in one spelling silently mis-filters blocks in the other.
    Dropping the separators and right-padding puts both on one scale.

    A date-only value pads to midnight, i.e. ``since="2026-12-31"`` means
    "from the start of that day". Any timezone suffix is truncated rather
    than applied: the corpus writes UTC and dated blocks carry no zone.

    A value that does not begin with an 8-digit date is returned stripped
    but otherwise unchanged, so an unrecognised format still compares
    against itself exactly as before.
    """
    text = str(raw).strip()
    if not text or not text[:4].isdigit():
        return text
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) < 8:
        return text
    return digits[:_STAMP_WIDTH].ljust(_STAMP_WIDTH, "0")


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
        # Read from the table, never spelled here: INITIAL_STATUS is the
        # only place an initial status is decided, so this door cannot
        # drift from the tier it is admitted under.
        "Status": MESSAGE_STATUS.value,
        TIER_FIELD: MESSAGE_TIER.value,
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
    from .governance_gate import get_gate
    from .pipeline_hash import stamp_transform_hash
    from .storage import get_block_store

    block = build_message_block(text, to=to, sender=sender, subject=subject)
    store = get_block_store(workspace)
    stamped = stamp_transform_hash(workspace, block)
    block_id = str(stamped["_id"])
    # An agent message arrives QUARANTINED. This reverses the earlier
    # reasoning ("the sender is a known local actor, not the untrusted drop
    # folder"): accountability for who *sent* a message says nothing about
    # who *wrote* the text it carries, and a peer agent is the standard
    # prompt-injection carrier. The mailbox still shows it; recall does not.
    with get_gate(workspace).admit_block(
        action="MESSAGE",
        block_id=block_id,
        content=text,
        tier=MESSAGE_TIER,
        actor=f"agent:{sender}" if sender else "agent",
        metadata={"to": to, "subject": subject or ""},
    ):
        written_id = store.write_block(stamped)

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
            _log.debug("send_message_index_rebuild_skipped", error=str(exc))
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
    backend-aware ``storage.iter_blocks`` so the full block fields
    (``To`` / ``From`` / ``Subject`` / ``type``) are preserved — recall's
    lean projection drops them — and so it works identically on the SQLite
    markdown default and the Postgres federation hub.

    It enumerates with ``active_only=False`` because messages now arrive
    quarantined (see the module docstring). That is not a hole in the
    quarantine: the recipient asked for this mailbox by name, so nothing
    is being retrieved into a context that did not request it, and the
    alternative — a mailbox that hides the mail — makes the channel
    useless without making it safer. The withheld property that matters
    is enforced where it matters, on the recall path.

    Filtering:
      * keep only ``AgentMessage`` blocks (``MSG-`` ids);
      * when *to* is set, keep messages addressed to *to* plus broadcasts
        (no ``To`` field), so a recipient sees its own mail + broadcasts;
      * when *since* is set, keep messages whose ``Timestamp`` (or ``Date``)
        is >= *since*. Both sides are canonicalised to ``YYYYMMDDHHMMSS``
        first (see :func:`_normalise_stamp`), so the compact
        ``YYYYMMDDTHHMMSSZ`` stamp and a dashed ``YYYY-MM-DD`` date can be
        mixed freely in either position — a raw string compare across the
        two spellings orders them backwards. A date-only *since* means
        midnight of that day.

    Returns at most *limit* blocks, newest-first.
    """
    from .storage import iter_blocks

    blocks = iter_blocks(workspace, active_only=False)

    def _is_message(b: dict) -> bool:
        if str(b.get("type", "")).strip() == MESSAGE_TYPE:
            return True
        # Fall back to the id prefix in case a backend strips/renames type.
        return str(b.get("_id", "")).startswith("MSG-")

    messages = [b for b in blocks if _is_message(b)]

    if to:
        messages = [b for b in messages if not b.get("To") or str(b.get("To")) == to]

    if since:
        bound = _normalise_stamp(since)

        def _stamp(b: dict) -> str:
            return _normalise_stamp(b.get("Timestamp") or b.get("Date") or "")

        messages = [b for b in messages if _stamp(b) >= bound]

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
