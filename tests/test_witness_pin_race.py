# Copyright 2026 STARGA, Inc.
"""The pin must be re-read inside the same transaction that appends.

Root's independent review reproduced an invalid acceptance at the public CLI:
mutate the governed pin between ``governed_witness_pin()`` and
``EvidenceChain.create()`` and the writer returns rc=0 with ``written=true``,
while ``reconcile`` and ``verify_workspace`` both reject the result. Twenty-two
existing controls pass straight through that gap, which is what makes this test
load-bearing rather than redundant.

LIMIT, stated because an advisory lock cannot deliver more. This closes the
window for COOPERATING writers that take the evidence store lock. It does not
bind a process that rewrites ``mind-mem.json`` while ignoring the protocol --
and measured today, NO current pin writer takes that lock (baseline_snapshot,
accountability_views, accountability_dashboard, event_fanout all write the file
lock-free). The refusal below is therefore last-check detection, not mutual
exclusion, and the code says so where it matters.
"""

import contextlib
import hashlib
import io
import json
import os

import pytest

from mind_mem.cross_ledger import reconcile
from mind_mem.evidence_objects import EvidenceChain
from mind_mem.evidence_recovery import RECOVERY_ACTION, RECOVERY_VERB, RECOVERY_VERB_KEY
from mind_mem.hash_chain_v2 import HashChainV2
from mind_mem.mm_cli import main as mm_main
from mind_mem.verify_cli import verify_workspace

ARCHIVE = "evidence_chain.jsonl.damaged-20260101T000000Z"


@pytest.fixture
def ws(tmp_path):
    root = tmp_path / "ws"
    mem = root / "memory"
    mem.mkdir(parents=True)
    arch = mem / ARCHIVE
    arch.write_bytes(b'{"r": 1}\n')
    chain = HashChainV2(str(mem / "hash_chain_v2.db"))
    for i in range(4):
        chain.append(block_id=f"B{i}", action="create_block", content=f"r{i}", timestamp=f"2026-01-01T0{i}:00:00+00:00")
    data = arch.read_bytes()
    EvidenceChain(store_path=str(mem / "evidence_chain.jsonl")).create(
        action=RECOVERY_ACTION,
        actor="operator",
        target_block_id="recovery",
        target_file="",
        metadata={
            RECOVERY_VERB_KEY: RECOVERY_VERB,
            "archived_chain": ARCHIVE,
            "archived_sha256": hashlib.sha256(data).hexdigest(),
            "archived_bytes": len(data),
        },
    )
    return root


def _ledger(ws):
    return os.path.join(str(ws), "memory", "evidence_chain.jsonl")


def _content(ws):
    """The ledger's actual BYTES.

    getsize proves LENGTH only -- a same-length rewrite passes it. Every
    preservation claim here compares content.
    """
    with open(_ledger(ws), "rb") as h:
        return h.read()


def _anchor(ws):
    with open(_ledger(ws), encoding="utf-8") as h:
        rec = json.loads([ln for ln in h if ln.strip()][0])
    return rec["evidence_id"], rec["evidence_hash"]


def _run(ws, *extra):
    aid, ahash = _anchor(ws)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = mm_main(["chain", "witness", str(ws), "--anchor-id", aid, "--anchor-hash", ahash, "--json", *extra])
    out = buf.getvalue()
    return rc, (json.loads(out[out.index("{") :]) if "{" in out else {})


def _pin(ws, anchor_id, digest):
    path = os.path.join(str(ws), "mind-mem.json")
    cfg = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    cfg.setdefault("recovery", {}).setdefault("boundary_witness_pins", {})[anchor_id] = digest
    with open(path, "w", encoding="utf-8") as h:
        json.dump(cfg, h)


class TestThePinIsRevalidatedInsideTheTransaction:
    def test_a_pin_mutated_after_its_read_is_refused(self, ws, monkeypatch):
        aid, _ = _anchor(ws)
        _, prep = _run(ws)
        _pin(ws, aid, prep["digest"])

        # The competing writer lands BETWEEN the outer pin check and the
        # revalidation inside the transaction. Patching governed_witness_pin
        # itself would defeat the test: the revalidation calls that same
        # function, so a stub returning the correct value hides the mutation it
        # is supposed to expose. Hooking the transaction places the write in the
        # real interval and leaves the pin reader honest.
        calls = {"n": 0}
        real_txn = EvidenceChain.transaction

        def _mutating_txn(self):
            calls["n"] += 1
            _pin(ws, aid, "0" * 128)  # the competing writer, mid-flight
            return real_txn(self)

        monkeypatch.setattr(EvidenceChain, "transaction", _mutating_txn)

        before = _content(ws)
        rc, payload = _run(ws, "--confirm")

        assert calls["n"] >= 1, "the interposition never fired; the test proves nothing"
        assert rc != 0, "the CLI accepted an attestation the governed pin no longer authorises"
        assert payload.get("written") is not True, "a stale-pin append reported success"
        assert _content(ws) == before, "the ledger BYTES changed on a refusal"

    def test_the_verifiers_and_the_cli_agree_after_the_fix(self, ws, monkeypatch):
        """The CLI must not disagree with reconcile / verify_workspace."""
        aid, _ = _anchor(ws)
        _, prep = _run(ws)
        _pin(ws, aid, prep["digest"])
        real_txn = EvidenceChain.transaction

        def _mutating_txn(self):
            _pin(ws, aid, "0" * 128)
            return real_txn(self)

        monkeypatch.setattr(EvidenceChain, "transaction", _mutating_txn)
        rc, _ = _run(ws, "--confirm")
        monkeypatch.undo()

        cli_said_ok = rc == 0
        assert cli_said_ok is (reconcile(str(ws)).ok is True), "the CLI and reconcile disagree about the same ledger"
        assert cli_said_ok is (verify_workspace(str(ws)).ok is True), "the CLI and the public verifier disagree about the same ledger"

    def test_the_unmutated_path_still_writes(self, ws):
        """Positive control: the refusal above is about the mutation."""
        aid, _ = _anchor(ws)
        _, prep = _run(ws)
        _pin(ws, aid, prep["digest"])
        before = _content(ws)
        rc, payload = _run(ws, "--confirm")
        assert rc == 0 and payload.get("written") is True
        assert _content(ws) != before
        assert reconcile(str(ws)).ok is True


class TestArchiveAndCompanionAreRevalidatedToo:
    """Content, not just recorded digests.

    The anchor's archive fields were compared field-by-field before the append,
    but the archive's actual BYTES were never re-hashed at that point -- so a
    same-length rewrite of the archive between validation and append went
    unnoticed. Root's static finding, tested here against content.
    """

    def _prepare(self, ws):
        aid, _ = _anchor(ws)
        _, prep = _run(ws)
        _pin(ws, aid, prep["digest"])
        return aid

    def test_archive_content_rewritten_mid_transaction_is_refused(self, ws, monkeypatch):
        self._prepare(ws)
        arch = os.path.join(str(ws), "memory", ARCHIVE)
        original = open(arch, "rb").read()
        assert len(original) == 9, "fixture: keep the replacement the SAME LENGTH"

        real_txn = EvidenceChain.transaction

        def _mutating_txn(self):
            with open(arch, "wb") as h:
                h.write(b'{"r": 9}\n')  # same length, different bytes
            return real_txn(self)

        monkeypatch.setattr(EvidenceChain, "transaction", _mutating_txn)
        before = _content(ws)
        rc, payload = _run(ws, "--confirm")

        assert rc != 0, "a same-length archive rewrite was not detected"
        assert payload.get("written") is not True
        assert _content(ws) == before

    def test_a_companion_ADVANCE_is_accepted_and_does_not_move_the_boundary(self, ws, monkeypatch):
        """A valid competing append is not globally invalid.

        My first version of this test asserted a refusal. That was wrong, and
        the code was right: appending to the companion leaves the attested
        PREFIX untouched, so the binding still holds and refusing would make the
        writer fail on ordinary concurrent activity. Prefixes are immutable --
        that is the whole reason the boundary is expressed as one.
        """
        self._prepare(ws)
        db = os.path.join(str(ws), "memory", "hash_chain_v2.db")
        real_txn = EvidenceChain.transaction

        def _advancing_txn(self):
            HashChainV2(db).append(block_id="RACE", action="create_block", content="mid-transaction", timestamp="2026-03-01T00:00:00+00:00")
            return real_txn(self)

        monkeypatch.setattr(EvidenceChain, "transaction", _advancing_txn)
        rc, payload = _run(ws, "--confirm")
        assert rc == 0, "an honest concurrent companion append must not fail the writer"
        assert payload.get("written") is True
        assert payload["witness"]["prefix_count"] == 4, "the boundary absorbed the new row"

    def test_a_companion_PREFIX_change_is_refused(self, ws, monkeypatch):
        """What must refuse: the attested prefix itself no longer matching."""
        self._prepare(ws)
        mem = os.path.join(str(ws), "memory")
        db = os.path.join(mem, "hash_chain_v2.db")
        real_txn = EvidenceChain.transaction

        def _rebuilding_txn(self):
            # Rebuild the companion so the attested prefix tail no longer holds.
            os.remove(db)
            fresh = HashChainV2(db)
            for i in range(4):
                fresh.append(block_id=f"X{i}", action="create_block", content=f"different {i}", timestamp=f"2026-01-01T0{i}:00:00+00:00")
            return real_txn(self)

        monkeypatch.setattr(EvidenceChain, "transaction", _rebuilding_txn)
        before = _content(ws)
        rc, payload = _run(ws, "--confirm")
        assert rc != 0, "the attested companion prefix changed and was not detected"
        assert payload.get("written") is not True
        assert _content(ws) == before

    def test_the_undisturbed_path_still_writes(self, ws):
        """Positive control for both refusals above."""
        self._prepare(ws)
        before = _content(ws)
        rc, payload = _run(ws, "--confirm")
        assert rc == 0 and payload.get("written") is True
        assert _content(ws) != before
