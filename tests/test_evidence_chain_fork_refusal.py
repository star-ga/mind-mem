# Copyright 2026 STARGA, Inc.
"""A chain whose stored history did not load intact must refuse to be written.

``EvidenceChain._load_from_file`` stops at the first record it cannot
trust and leaves ``_entries`` empty, setting ``_integrity_compromised``.
That flag gated ``verify_chain()`` and nothing else — the write path
never consulted it. So the next ``create()`` read an *empty* in-memory
chain, took ``_GENESIS_HASH`` as ``previous_hash``, and appended a
second genesis-rooted record behind the untrusted tail. One tampered
byte therefore did not damage a tail, it forked the ledger permanently:
every later governed write re-rooted at genesis and the entire audit
history stopped verifying. ``governance_gate.evict_gate`` documents that
exact fork as the thing that must not happen.

No attacker is required, either: the loader's entry/line caps raise the
same flag on a merely oversized chain, so a large honest chain would
have started forking too.

These tests pin the refusal, and pin that a healthy chain is untouched
by it.
"""

from __future__ import annotations

import json
import os

import pytest

from mind_mem.admission import GovernanceBypassError
from mind_mem.evidence_objects import (
    _GENESIS_HASH,
    EvidenceAction,
    EvidenceChain,
    EvidenceChainCompromisedError,
)


def _seed(store: str, n: int = 3) -> EvidenceChain:
    """Write *n* linked records to *store* and return the live chain."""
    chain = EvidenceChain(store_path=store)
    for i in range(n):
        chain.create(
            action=EvidenceAction.APPLY,
            actor="seed",
            target_block_id=f"B-{i:03d}",
            target_file="decisions/DECISIONS.md",
            payload=b"payload",
        )
    return chain


def _tamper_actor(store: str, line_index: int = 1) -> None:
    """Rewrite one record's actor, leaving its evidence_hash stale."""
    lines = open(store, encoding="utf-8").read().splitlines()
    record = json.loads(lines[line_index])
    record["actor"] = "tampered"
    lines[line_index] = json.dumps(record, separators=(",", ":"))
    with open(store, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def _genesis_rooted(store: str) -> int:
    """How many records in the file claim the genesis hash as parent."""
    return sum(
        1
        for line in open(store, encoding="utf-8").read().splitlines()
        if line.strip() and json.loads(line)["previous_hash"] == _GENESIS_HASH
    )


class TestATamperedChainRefusesToBeAppendedTo:
    def test_create_raises_instead_of_forking(self, tmp_path):
        store = str(tmp_path / "evidence.jsonl")
        _seed(store)
        _tamper_actor(store)

        reopened = EvidenceChain(store_path=store)
        assert reopened.verify_chain() == (False, ["load_integrity_compromised"])

        with pytest.raises(EvidenceChainCompromisedError) as caught:
            reopened.create(
                action=EvidenceAction.APPLY,
                actor="later-writer",
                target_block_id="B-FORK",
                target_file="decisions/DECISIONS.md",
                payload=b"payload",
            )
        assert "did not load intact" in str(caught.value)

    def test_nothing_reaches_the_store_file(self, tmp_path):
        """The refusal must land before the append, not after it."""
        store = str(tmp_path / "evidence.jsonl")
        _seed(store)
        _tamper_actor(store)
        before = open(store, "rb").read()

        reopened = EvidenceChain(store_path=store)
        with pytest.raises(EvidenceChainCompromisedError):
            reopened.create(
                action=EvidenceAction.APPLY,
                actor="later-writer",
                target_block_id="B-FORK",
                target_file="decisions/DECISIONS.md",
                payload=b"payload",
            )

        assert open(store, "rb").read() == before
        assert _genesis_rooted(store) == 1, "a second genesis-rooted record forked the ledger"

    def test_the_refusal_survives_an_invalid_confidence(self, tmp_path):
        """Argument validation must not preempt the integrity refusal.

        A caller passing junk confidence to a compromised chain should
        still be told the chain is compromised — that is the finding, and
        a ValueError would send them off fixing the wrong thing.
        """
        store = str(tmp_path / "evidence.jsonl")
        _seed(store)
        _tamper_actor(store)

        reopened = EvidenceChain(store_path=store)
        with pytest.raises(EvidenceChainCompromisedError):
            reopened.create(
                action=EvidenceAction.APPLY,
                actor="later-writer",
                target_block_id="B-FORK",
                target_file="decisions/DECISIONS.md",
                payload=b"payload",
                confidence=7.5,
            )

    def test_the_refusal_is_a_governance_bypass_error(self, tmp_path):
        """So the gate's existing callers abort on it like any refused write."""
        store = str(tmp_path / "evidence.jsonl")
        _seed(store)
        _tamper_actor(store)

        reopened = EvidenceChain(store_path=store)
        with pytest.raises(GovernanceBypassError):
            reopened.create(
                action=EvidenceAction.APPLY,
                actor="later-writer",
                target_block_id="B-FORK",
                target_file="decisions/DECISIONS.md",
                payload=b"payload",
            )


class TestEveryLoadFailureFreezesTheChain:
    """Tamper is not the only way to lose the tail — caps do it too."""

    def test_an_entry_cap_hit_freezes_the_chain(self, tmp_path, monkeypatch):
        store = str(tmp_path / "evidence.jsonl")
        _seed(store, n=4)
        monkeypatch.setattr(EvidenceChain, "_MAX_LOAD_ENTRIES", 2)

        reopened = EvidenceChain(store_path=store)
        assert reopened.integrity_compromised is True
        assert "entry cap" in (reopened.load_failure or "")
        with pytest.raises(EvidenceChainCompromisedError):
            reopened.create(
                action=EvidenceAction.APPLY,
                actor="later-writer",
                target_block_id="B-FORK",
                target_file="decisions/DECISIONS.md",
                payload=b"payload",
            )
        assert _genesis_rooted(store) == 1

    def test_an_unparseable_line_freezes_the_chain(self, tmp_path):
        store = str(tmp_path / "evidence.jsonl")
        _seed(store, n=2)
        with open(store, "a", encoding="utf-8") as fh:
            fh.write("{not json\n")

        reopened = EvidenceChain(store_path=store)
        assert reopened.integrity_compromised is True
        with pytest.raises(EvidenceChainCompromisedError):
            reopened.create(
                action=EvidenceAction.APPLY,
                actor="later-writer",
                target_block_id="B-FORK",
                target_file="decisions/DECISIONS.md",
                payload=b"payload",
            )

    def test_a_broken_linkage_freezes_the_chain(self, tmp_path):
        """Records that each self-verify but do not link to one another."""
        store = str(tmp_path / "evidence.jsonl")
        _seed(store, n=2)
        other = str(tmp_path / "other.jsonl")
        _seed(other, n=1)
        with open(store, "a", encoding="utf-8") as fh:
            fh.write(open(other, encoding="utf-8").read())

        reopened = EvidenceChain(store_path=store)
        assert reopened.integrity_compromised is True
        assert "linkage" in (reopened.load_failure or "")
        with pytest.raises(EvidenceChainCompromisedError):
            reopened.create(
                action=EvidenceAction.APPLY,
                actor="later-writer",
                target_block_id="B-FORK",
                target_file="decisions/DECISIONS.md",
                payload=b"payload",
            )


class TestExportDoesNotPublishAnEmptyHistory:
    def test_export_refuses_on_a_compromised_chain(self, tmp_path):
        store = str(tmp_path / "evidence.jsonl")
        _seed(store)
        _tamper_actor(store)

        reopened = EvidenceChain(store_path=store)
        out = str(tmp_path / "exported.jsonl")
        with pytest.raises(EvidenceChainCompromisedError):
            reopened.export_jsonl(out)
        assert not os.path.exists(out), "an empty file was published as the history"

    def test_export_over_the_store_cannot_truncate_it(self, tmp_path):
        store = str(tmp_path / "evidence.jsonl")
        _seed(store)
        _tamper_actor(store)
        before = open(store, "rb").read()

        reopened = EvidenceChain(store_path=store)
        with pytest.raises(EvidenceChainCompromisedError):
            reopened.export_jsonl(store)
        assert open(store, "rb").read() == before


class TestAHealthyChainIsUnaffected:
    """The guard must refuse a broken chain, not freeze a working one."""

    def test_reload_and_append_still_works(self, tmp_path):
        store = str(tmp_path / "evidence.jsonl")
        _seed(store, n=2)

        reopened = EvidenceChain(store_path=store)
        assert reopened.integrity_compromised is False
        assert reopened.load_failure is None
        ev = reopened.create(
            action=EvidenceAction.APPLY,
            actor="later-writer",
            target_block_id="B-NEXT",
            target_file="decisions/DECISIONS.md",
            payload=b"payload",
        )
        assert ev.previous_hash != _GENESIS_HASH
        assert len(reopened) == 3
        assert reopened.verify_chain() == (True, [])
        assert _genesis_rooted(store) == 1

    def test_export_still_works(self, tmp_path):
        store = str(tmp_path / "evidence.jsonl")
        _seed(store, n=2)
        out = str(tmp_path / "exported.jsonl")
        EvidenceChain(store_path=store).export_jsonl(out)
        assert len(open(out, encoding="utf-8").read().splitlines()) == 2


class TestTheGateRefusesTheWriteEndToEnd:
    """The path that actually matters: a governed write on a forked store."""

    def test_admit_refuses_when_the_evidence_store_is_tampered(self, tmp_path):
        from mind_mem.governance_gate import GovernanceBypassError as GateBypassError
        from mind_mem.governance_gate import evict_gate, get_gate

        ws = str(tmp_path / "ws")
        os.makedirs(ws)
        gate = get_gate(ws)
        gate.admit(action="WRITE", block_id="B-1", content="payload", actor="test")
        gate.admit(action="WRITE", block_id="B-2", content="payload", actor="test")
        assert evict_gate(ws) is True

        store = os.path.join(ws, "memory", "evidence_chain.jsonl")
        _tamper_actor(store, line_index=0)
        before = open(store, "rb").read()

        reopened_gate = get_gate(ws)
        try:
            with pytest.raises(GateBypassError) as caught:
                reopened_gate.admit(action="WRITE", block_id="B-3", content="payload", actor="test")
            # Name the mechanism: the gate must refuse *for this reason*, not
            # because it was retired or its spec drifted.
            assert isinstance(caught.value, EvidenceChainCompromisedError)
            assert open(store, "rb").read() == before
            assert _genesis_rooted(store) == 1
            assert reopened_gate.chain.verify_chain()[0] is True, "the hash chain took a write the evidence chain refused"
        finally:
            evict_gate(ws)
