"""Tests for v4.0.19 agent-to-agent messaging (`mm send` / `mm inbox`).

The sanctioned cross-agent comm channel is the shared block store: a
sender writes an ``MSG-`` block, a recipient receives by enumerating the
``memory/MESSAGES.md`` corpus. These tests pin the wiring that makes that
work on the SQLite/markdown default:

  * the ``MSG`` prefix is mapped (so ``write_block`` accepts it) and the
    two duplicate prefix maps stay in lockstep;
  * ``memory/MESSAGES.md`` is in ``CORPUS_FILES`` (so messages are indexed
    / recallable, at parity with Postgres);
  * a full send -> inbox round-trip with recipient scoping + broadcasts.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mind_mem._recall_constants import CORPUS_FILES
from mind_mem.agent_messaging import (
    MESSAGE_TYPE,
    build_message_block,
    read_inbox,
    send_message,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def workspace(tmp_path: Path) -> str:
    ws = tmp_path / "ws"
    (ws / "memory").mkdir(parents=True)
    config = {
        "version": "4.0.19",
        "workspace_path": str(ws),
        "block_store": {"backend": "markdown"},
    }
    (ws / "mind-mem.json").write_text(json.dumps(config))
    return str(ws)


# ---------------------------------------------------------------------------
# Wiring invariants (regression guards for the two bugs)
# ---------------------------------------------------------------------------


def test_msg_prefix_is_mapped() -> None:
    """BUG A guard: ``write_block`` must accept an MSG- block."""
    from mind_mem.block_store import _BLOCK_PREFIX_MAP

    assert _BLOCK_PREFIX_MAP["MSG"] == ("memory", "MESSAGES.md")


def test_prefix_maps_in_lockstep() -> None:
    """The two duplicate prefix maps must stay identical (comment requires it)."""
    from mind_mem.block_store import _BLOCK_PREFIX_MAP as store_map
    from mind_mem.mcp.tools.memory_ops import _BLOCK_PREFIX_MAP as mcp_map

    assert store_map == mcp_map


def test_messages_corpus_is_indexed() -> None:
    """BUG B guard: the messages file must be in CORPUS_FILES (so it indexes)."""
    assert CORPUS_FILES.get("messages") == "memory/MESSAGES.md"


def test_inbox_corpus_is_indexed() -> None:
    """Pre-existing inbox-invisible-on-SQLite bug: INBOX file must be indexed."""
    assert CORPUS_FILES.get("inbox") == "memory/INBOX.md"


# ---------------------------------------------------------------------------
# build_message_block
# ---------------------------------------------------------------------------


def test_build_message_block_shape() -> None:
    b = build_message_block("hello", to="S1", sender="U1", subject="hi", timestamp="20260623T000000Z", nonce="dead")
    assert b["_id"] == "MSG-20260623T000000Z-dead"
    assert b["type"] == MESSAGE_TYPE
    assert b["Statement"] == "hello"
    assert b["To"] == "S1"
    assert b["From"] == "U1"
    assert b["Subject"] == "hi"
    assert b["Status"] == "active"


def test_build_message_block_omits_empty_routing_fields() -> None:
    b = build_message_block("broadcast", timestamp="20260623T000000Z", nonce="beef")
    assert "To" not in b
    assert "From" not in b
    assert "Subject" not in b


def test_build_message_block_rejects_empty_text() -> None:
    with pytest.raises(ValueError):
        build_message_block("   ")


# ---------------------------------------------------------------------------
# send -> inbox round-trip
# ---------------------------------------------------------------------------


def test_send_then_inbox_roundtrip(workspace: str) -> None:
    mid = send_message(workspace, "deploy the patch", to="S1", sender="U1", subject="patch")
    assert mid.startswith("MSG-")

    inbox = read_inbox(workspace, to="S1")
    assert len(inbox) == 1
    msg = inbox[0]
    assert msg["_id"] == mid
    # Full fields are preserved (recall's lean projection would drop these).
    assert msg["Statement"] == "deploy the patch"
    assert msg["To"] == "S1"
    assert msg["From"] == "U1"
    assert msg["Subject"] == "patch"
    # NB: the markdown parser drops the lowercase ``type:`` field on
    # re-parse (same as INBOX_DOCUMENT blocks), so read_inbox identifies
    # messages by the ``MSG-`` id prefix rather than the type token.
    assert msg["_id"].startswith("MSG-")


def test_inbox_recipient_scoping_and_broadcast(workspace: str) -> None:
    send_message(workspace, "for S1 only", to="S1", sender="U1")
    send_message(workspace, "fleet broadcast", sender="U1")  # no To = broadcast

    s1 = read_inbox(workspace, to="S1")
    g1 = read_inbox(workspace, to="G1")

    s1_texts = {m["Statement"] for m in s1}
    g1_texts = {m["Statement"] for m in g1}

    # S1 sees its mail plus the broadcast.
    assert s1_texts == {"for S1 only", "fleet broadcast"}
    # G1 sees only the broadcast, never S1's addressed mail.
    assert g1_texts == {"fleet broadcast"}


def test_inbox_unfiltered_returns_all(workspace: str) -> None:
    send_message(workspace, "m1", to="S1", sender="U1")
    send_message(workspace, "m2", to="G1", sender="U1")
    all_msgs = read_inbox(workspace)
    assert len(all_msgs) == 2


def test_inbox_since_filter(workspace: str) -> None:
    send_message(workspace, "old", to="S1", sender="U1")
    # A clearly-future lower bound excludes the just-sent message.
    assert read_inbox(workspace, to="S1", since="20990101T000000Z") == []
    # A past lower bound includes it.
    assert len(read_inbox(workspace, to="S1", since="20000101T000000Z")) == 1


def test_message_is_recallable(workspace: str) -> None:
    """The search path: a sent message is findable via recall (index rebuilt)."""
    from mind_mem.recall import recall

    send_message(workspace, "zephyrine-unique-marker-9931 ping", to="S1", sender="U1")
    hits = recall(workspace, "zephyrine-unique-marker-9931", limit=5, active_only=False)
    assert isinstance(hits, list)
    assert any(h.get("_id", "").startswith("MSG-") for h in hits)
