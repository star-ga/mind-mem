# Copyright 2026 STARGA, Inc.
"""A same-day signal never takes an id the block store already holds.

``capture.append_signals`` picks the next ``SIG-YYYYMMDD-NNN`` id from the
ids in ``intelligence/SIGNALS.md``. That file is authoritative only for the
Markdown backend. On a Postgres-backed workspace the file stops changing, so
every signal of a day was given ``-001`` and ``write_block`` (persist *or
replace*) overwrote the one written before it: measured on a live workspace,
one SIG block survived per day while every call returned ``written: 1``.

These tests drive the real door against a real workspace whose block store is
replaced by a dict-backed store that the file does not mirror, which is the
Postgres shape without a database.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from typing import Any
from unittest import mock

from mind_mem import capture
from mind_mem.block_store import MarkdownBlockStore
from mind_mem.init_workspace import init


def _signal(text: str) -> dict:
    return {
        "line": 1,
        "type": "decision",
        "text": text,
        "pattern": ".*",
        "confidence": "high",
        "priority": "P1",
        "structure": {"subject": "we", "object": "cache", "tags": ["database"]},
    }


class _DetachedStore:
    """Holds blocks in a dict; nothing it writes reaches SIGNALS.md."""

    def __init__(self) -> None:
        self.blocks: dict[str, dict[str, Any]] = {}

    def get_by_id(self, block_id: str) -> dict[str, Any] | None:
        block = self.blocks.get(block_id)
        return dict(block) if block is not None else None

    def write_block(self, block: dict[str, Any]) -> str:
        block_id = str(block["_id"])
        self.blocks[block_id] = dict(block)
        return block_id


class TestSameDaySignalsOnADetachedStore(unittest.TestCase):
    def setUp(self) -> None:
        self.ws = tempfile.mkdtemp(prefix="mm-sigid-")
        self.addCleanup(shutil.rmtree, self.ws, ignore_errors=True)
        init(self.ws)
        from mind_mem.governance_gate import evict_gate

        self.addCleanup(evict_gate, self.ws)
        evict_gate(self.ws)
        self.store = _DetachedStore()
        patcher = mock.patch("mind_mem.storage.get_block_store", return_value=self.store)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_two_signals_on_one_day_both_survive(self) -> None:
        first = capture.append_signals(self.ws, [_signal("We decided to move the cache to Redis")], "2026-11-15")
        second = capture.append_signals(self.ws, [_signal("We decided to pin the tie-break to the lowest index")], "2026-11-15")

        # Positive control: each call reports one landed block.
        self.assertEqual((first, second), (1, 1))
        ids = sorted(self.store.blocks)
        self.assertEqual(ids, ["SIG-20261115-001", "SIG-20261115-002"], f"same-day ids collided: {ids}")
        excerpts = [str(self.store.blocks[i].get("Excerpt", "")) for i in ids]
        self.assertIn("Redis", excerpts[0], "the first signal was overwritten by the second")
        self.assertIn("tie-break", excerpts[1])

    def test_an_id_held_only_by_the_store_is_skipped(self) -> None:
        self.store.blocks["SIG-20261115-001"] = {"_id": "SIG-20261115-001", "Excerpt": "held before this call"}

        written = capture.append_signals(self.ws, [_signal("We decided to keep the old schema")], "2026-11-15")

        self.assertEqual(written, 1)
        self.assertEqual(self.store.blocks["SIG-20261115-001"]["Excerpt"], "held before this call")
        self.assertIn("SIG-20261115-002", self.store.blocks)

    def test_a_same_day_repeat_is_skipped_not_given_a_fresh_id(self) -> None:
        text = "We decided to move the cache to Redis"
        first = capture.append_signals(self.ws, [_signal(text)], "2026-11-15")
        repeat = capture.append_signals(self.ws, [_signal(text)], "2026-11-15")

        # Positive control: the first write landed, so the 0 below is the dedup.
        self.assertEqual(first, 1)
        self.assertEqual(repeat, 0)
        self.assertEqual(sorted(self.store.blocks), ["SIG-20261115-001"])

    def test_a_batch_skips_a_taken_id_in_the_middle(self) -> None:
        self.store.blocks["SIG-20261115-002"] = {"_id": "SIG-20261115-002", "Excerpt": "held"}

        written = capture.append_signals(
            self.ws,
            [_signal("We decided on plan A"), _signal("We decided on plan B"), _signal("We decided on plan C")],
            "2026-11-15",
        )

        self.assertEqual(written, 3)
        self.assertEqual(self.store.blocks["SIG-20261115-002"]["Excerpt"], "held")
        self.assertEqual(sorted(self.store.blocks), ["SIG-20261115-001", "SIG-20261115-002", "SIG-20261115-003", "SIG-20261115-004"])

    def test_a_nearly_full_day_truncates_the_batch(self) -> None:
        for n in range(1, 998):
            sid = f"SIG-20261115-{n:03d}"
            self.store.blocks[sid] = {"_id": sid, "Excerpt": "taken"}

        written = capture.append_signals(
            self.ws,
            [_signal("We decided on plan A"), _signal("We decided on plan B"), _signal("We decided on plan C")],
            "2026-11-15",
        )

        self.assertEqual(written, 2)
        self.assertIn("SIG-20261115-998", self.store.blocks)
        self.assertIn("SIG-20261115-999", self.store.blocks)
        self.assertEqual(len(self.store.blocks), 999)

    def test_a_full_day_writes_nothing_rather_than_overwriting(self) -> None:
        for n in range(1, 1000):
            sid = f"SIG-20261115-{n:03d}"
            self.store.blocks[sid] = {"_id": sid, "Excerpt": "taken"}
        before = dict(self.store.blocks)

        written = capture.append_signals(self.ws, [_signal("We decided one signal too many")], "2026-11-15")

        self.assertEqual(written, 0)
        self.assertEqual(self.store.blocks, before)


class TestTheMarkdownBackendNeverProbes(unittest.TestCase):
    """The file is the Markdown store, so its ids are used without a store read."""

    def test_free_ids_come_from_the_counter_without_a_store_call(self) -> None:
        store = mock.create_autospec(MarkdownBlockStore, instance=True)

        ids = capture._free_signal_ids(store, "20261115", 4, 3)

        self.assertEqual(ids, ["SIG-20261115-004", "SIG-20261115-005", "SIG-20261115-006"])
        store.get_by_id.assert_not_called()

    def test_the_day_run_is_empty_without_a_store_call(self) -> None:
        store = mock.create_autospec(MarkdownBlockStore, instance=True)

        self.assertEqual(capture._store_day_run(store, "20261115", 4), (set(), 4))
        store.get_by_id.assert_not_called()

    def test_a_non_markdown_store_is_probed(self) -> None:
        # Positive control for the two tests above: the same calls on a
        # store that is not Markdown do read it.
        store = _DetachedStore()
        store.blocks["SIG-20261115-004"] = {"_id": "SIG-20261115-004", "ContentHash": "abc"}

        self.assertEqual(capture._store_day_run(store, "20261115", 4), ({"abc"}, 5))
        self.assertEqual(capture._free_signal_ids(store, "20261115", 4, 1), ["SIG-20261115-005"])


if __name__ == "__main__":
    unittest.main()
