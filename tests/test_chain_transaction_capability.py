# Copyright 2026 STARGA, Inc.
"""A transaction handle is a capability scoped to one lock, one thread, one block.

The first version was none of those. Root's independent review showed the token
appending AFTER its context exited, appending from a DIFFERENT thread while the
context still held the lock, and handing out live record metadata that a caller
could mutate into the chain -- and a directly constructed handle authorised
appends with no lock at all. That is an unlocked append door, which is worse
than the race the transaction was written to close.

Every negative below asserts exact BYTE preservation by comparing content, not
size: a same-length rewrite passes a size check.
"""

import threading

import pytest

from mind_mem.evidence_objects import ChainTransaction, EvidenceAction, EvidenceChain


@pytest.fixture
def chain(tmp_path):
    store = tmp_path / "evidence_chain.jsonl"
    ch = EvidenceChain(store_path=str(store))
    ch.create(
        action=EvidenceAction.VERIFY, actor="operator", target_block_id="seed", target_file="", metadata={"nested": {"value": "ORIGINAL"}}
    )
    return ch


def _bytes(chain):
    with open(chain._store_path, "rb") as h:
        return h.read()


def _append(txn, tag="x"):
    return txn.create(action=EvidenceAction.VERIFY, actor="operator", target_block_id=tag, target_file="")


class TestTheCapabilityIsActiveOnlyInsideItsScope:
    def test_create_inside_the_active_context_still_writes(self, chain):
        """Positive control. Without it every refusal below proves nothing."""
        before = _bytes(chain)
        with chain.transaction() as txn:
            assert isinstance(txn.tail, str)
            assert txn.entries >= 1
            assert len(txn.records()) >= 1
            _append(txn, "inside")
        assert _bytes(chain) != before

    def test_create_after_normal_exit_is_refused(self, chain):
        escaped = None
        with chain.transaction() as txn:
            escaped = txn
        before = _bytes(chain)
        with pytest.raises(Exception):
            _append(escaped, "after-exit")
        assert _bytes(chain) == before, "an append landed after the scope closed"

    def test_create_after_an_EXCEPTION_exit_is_refused(self, chain):
        escaped = None
        with pytest.raises(RuntimeError):
            with chain.transaction() as txn:
                escaped = txn
                raise RuntimeError("boom")
        before = _bytes(chain)
        with pytest.raises(Exception):
            _append(escaped, "after-exception")
        assert _bytes(chain) == before

    def test_reads_after_exit_are_refused_too(self, chain):
        """A stale tail is a stale premise for the caller's own decision."""
        escaped = None
        with chain.transaction() as txn:
            escaped = txn
        for probe in (lambda: escaped.tail, lambda: escaped.entries, escaped.records):
            with pytest.raises(Exception):
                probe()


class TestTheCapabilityBelongsToOneThread:
    def test_another_thread_cannot_append_through_it(self, chain):
        result = {}
        with chain.transaction() as txn:
            before = _bytes(chain)

            def other():
                try:
                    _append(txn, "cross-thread")
                    result["r"] = "accepted"
                except Exception as exc:
                    result["r"] = type(exc).__name__

            t = threading.Thread(target=other)
            t.start()
            t.join(10)
            assert not t.is_alive(), "the cross-thread append hung"
            assert result.get("r") != "accepted", "another thread appended through the handle"
            assert _bytes(chain) == before


class TestDirectConstructionAuthorizesNothing:
    def test_a_hand_built_handle_refuses(self, chain):
        before = _bytes(chain)
        handle = ChainTransaction(chain)
        with pytest.raises(Exception):
            _append(handle, "direct")
        assert _bytes(chain) == before, "a hand-built handle appended without a lock"


class TestRecordsAreCopies:
    def test_mutating_returned_metadata_does_not_touch_the_chain(self, chain):
        with chain.transaction() as txn:
            rec = txn.records()[0]
            rec.metadata["nested"]["value"] = "MUTATED"
            rec.metadata["injected"] = True
        assert chain._entries[0].metadata["nested"]["value"] == "ORIGINAL", "a returned record shared live nested metadata with the chain"
        assert "injected" not in chain._entries[0].metadata


class TestOrdinaryCreateIsUnchanged:
    def test_create_outside_a_transaction_still_works(self, chain):
        before = _bytes(chain)
        chain.create(action=EvidenceAction.VERIFY, actor="operator", target_block_id="ordinary", target_file="")
        assert _bytes(chain) != before
