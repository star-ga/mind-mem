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
from mind_mem.enums import INITIAL_STATUS, IngestTier

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
    # A message arrives WITHHELD (see the module docstring): a peer agent is
    # the standard prompt-injection carrier. The status is not spelled in
    # this module -- it is read from INITIAL_STATUS, the single place an
    # initial status is decided -- so this assertion is against the table.
    assert b["Status"] == INITIAL_STATUS[IngestTier.AGENT_MESSAGE].value == "quarantined"
    assert b["IngestTier"] == IngestTier.AGENT_MESSAGE.value


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


def test_message_is_withheld_from_recall_until_released(workspace: str) -> None:
    """The search path: a sent message is NOT retrievable as memory.

    This inverts the pre-quarantine assertion, so it has to prove the
    absence is the *status* and not a broken index or an unmatched query.
    Three legs, in order:

      1. the mailbox shows it   -> it was durably written and is parseable;
      2. recall does not        -> the withheld property;
      3. flip the same block to ``active`` through a proposal admission,
         reindex, and recall the same query -> it comes back.

    Leg 3 is the control. Without it this test would pass just as happily
    against a corpus that recall cannot read at all.
    """
    from mind_mem.governance_gate import get_gate
    from mind_mem.recall import recall
    from mind_mem.sqlite_index import build_index
    from mind_mem.storage import get_block_store

    marker = "zephyrine-unique-marker-9931"
    mid = send_message(workspace, f"{marker} ping", to="S1", sender="U1")

    # 1. present in the mailbox
    assert [m["_id"] for m in read_inbox(workspace)] == [mid]

    # 2. absent from recall
    withheld_hits = recall(workspace, marker, limit=5, active_only=False)
    assert isinstance(withheld_hits, list)
    assert not [h for h in withheld_hits if h.get("_id", "").startswith("MSG-")], "a quarantined agent message was served by recall"

    # 3. control — release it the sanctioned way and it comes straight back.
    # The block comes from the mailbox read above (the markdown store's
    # get_by_id does not resolve memory/ blocks); private ``_`` keys the
    # enumeration tags on are dropped so only real fields are re-rendered.
    store = get_block_store(workspace)
    stored = next(m for m in read_inbox(workspace) if m["_id"] == mid)
    released = {k: v for k, v in stored.items() if not k.startswith("_") or k == "_id"}
    released["Status"] = "active"
    with get_gate(workspace).admit_proposal(proposal_id="P-RELEASE", content="[]"):
        store.write_block(released)
    build_index(workspace)
    released_hits = recall(workspace, marker, limit=5, active_only=False)
    assert [h for h in released_hits if h.get("_id", "").startswith("MSG-")], (
        "control leg failed: recall cannot find the message even when active, so leg 2 proves nothing about the quarantine"
    )


# ---------------------------------------------------------------------------
# ``since`` across the two stamp spellings
# ---------------------------------------------------------------------------


def _fake_corpus(monkeypatch: pytest.MonkeyPatch, blocks: list[dict]) -> None:
    """Serve *blocks* to ``read_inbox`` without touching a real workspace."""
    import mind_mem.storage as storage

    monkeypatch.setattr(storage, "iter_blocks", lambda ws, active_only=False: list(blocks))


def test_since_dashed_date_excludes_earlier_compact_stamp(monkeypatch: pytest.MonkeyPatch) -> None:
    """A raw string compare orders these backwards.

    At index 4 a compact stamp holds a digit (0x30+) and a dashed date
    holds ``-`` (0x2D), so ``'20260101T000000Z' >= '2026-12-31'`` is True
    and a January message survived a December lower bound.
    """
    _fake_corpus(
        monkeypatch,
        [{"_id": "MSG-20260101T000000Z-aa", "type": MESSAGE_TYPE, "Statement": "january", "Timestamp": "20260101T000000Z"}],
    )
    assert read_inbox("/nonexistent", since="2026-12-31") == []


def test_since_compact_stamp_keeps_later_dashed_date_block(monkeypatch: pytest.MonkeyPatch) -> None:
    """The mirror case: a dated block must not be dropped by a compact bound."""
    _fake_corpus(
        monkeypatch,
        [{"_id": "MSG-1", "type": MESSAGE_TYPE, "Statement": "dated", "Date": "2026-01-05"}],
    )
    kept = read_inbox("/nonexistent", since="20260101T000000Z")
    assert [m["Statement"] for m in kept] == ["dated"]


def test_since_date_only_bound_means_midnight(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_corpus(
        monkeypatch,
        [
            {"_id": "MSG-a", "type": MESSAGE_TYPE, "Statement": "same day", "Timestamp": "20260315T093000Z"},
            {"_id": "MSG-b", "type": MESSAGE_TYPE, "Statement": "day before", "Timestamp": "20260314T235959Z"},
        ],
    )
    kept = read_inbox("/nonexistent", since="2026-03-15")
    assert [m["Statement"] for m in kept] == ["same day"]


def test_since_same_spelling_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """Compact-vs-compact keeps behaving exactly as before."""
    _fake_corpus(
        monkeypatch,
        [
            {"_id": "MSG-a", "type": MESSAGE_TYPE, "Statement": "after", "Timestamp": "20260315T000001Z"},
            {"_id": "MSG-b", "type": MESSAGE_TYPE, "Statement": "before", "Timestamp": "20260314T000000Z"},
        ],
    )
    kept = read_inbox("/nonexistent", since="20260315T000000Z")
    assert [m["Statement"] for m in kept] == ["after"]
