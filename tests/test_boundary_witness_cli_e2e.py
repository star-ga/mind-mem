# Copyright 2026 STARGA, Inc.
"""The witness path, exercised through the PUBLIC CLI and the ordinary verifier.

The unit tests call ``resolve_supplemental_baseline`` directly and hand it a
digest. That proves the helper, not the product: the real caller in ``reconcile``
supplied no digest at all for one revision, so every supplemental baseline
refused on the empty default while the helper's own tests stayed green. Nothing
below constructs a digest by hand or calls the resolver directly -- the witness
is produced by ``mm chain witness``, pinned in ``mind-mem.json`` the way an
operator would, and the verdict is read from ``reconcile``.
"""

import json
import os

import pytest

from mind_mem import boundary_witness as bw
from mind_mem.cross_ledger import reconcile
from mind_mem.evidence_objects import EvidenceChain
from mind_mem.evidence_recovery import RECOVERY_ACTION, RECOVERY_VERB, RECOVERY_VERB_KEY
from mind_mem.hash_chain_v2 import HashChainV2
from mind_mem.mm_cli import main as mm_main
from mind_mem.verify_cli import verify_workspace

ANCHOR_ARCHIVE = "evidence_chain.jsonl.damaged-20260101T000000Z"


@pytest.fixture
def workspace(tmp_path):
    """A workspace whose anchor is LEGACY: minted with no companion baseline."""
    ws = tmp_path / "ws"
    mem = ws / "memory"
    mem.mkdir(parents=True)

    archive = mem / ANCHOR_ARCHIVE
    archive.write_bytes(b'{"row": 1}\n{"row": 2}\n')

    chain = HashChainV2(str(mem / "hash_chain_v2.db"))
    for i in range(4):
        chain.append(
            block_id=f"B{i}",
            action="create_block",
            content=f"row {i}",
            timestamp=f"2026-01-01T0{i}:00:00+00:00",
        )

    import hashlib

    data = archive.read_bytes()
    ledger = EvidenceChain(store_path=str(mem / "evidence_chain.jsonl"))
    ledger.create(
        action=RECOVERY_ACTION,
        actor="operator",
        target_block_id="recovery",
        target_file="",
        metadata={
            RECOVERY_VERB_KEY: RECOVERY_VERB,
            "archived_chain": ANCHOR_ARCHIVE,
            "archived_sha256": hashlib.sha256(data).hexdigest(),
            "archived_bytes": len(data),
            # NO companion_hash_chain_* keys -- this is the legacy shape.
        },
    )
    return ws


def _anchor(workspace):
    store = os.path.join(str(workspace), "memory", "evidence_chain.jsonl")
    with open(store, encoding="utf-8") as handle:
        rec = json.loads([ln for ln in handle if ln.strip()][0])
    return rec["evidence_id"], rec["evidence_hash"]


def _run(workspace, *extra):
    aid, ahash = _anchor(workspace)
    argv = ["chain", "witness", str(workspace), "--anchor-id", aid, "--anchor-hash", ahash, "--json", *extra]
    return mm_main(argv)


def _witness_from_cli(workspace, capsys, *extra):
    rc = _run(workspace, *extra)
    out = capsys.readouterr().out
    return rc, json.loads(out[out.index("{") :]) if "{" in out else {}


def _pin(workspace, anchor_id, digest):
    path = os.path.join(str(workspace), "mind-mem.json")
    config = {}
    if os.path.exists(path):
        config = json.load(open(path, encoding="utf-8"))
    config.setdefault("recovery", {}).setdefault("boundary_witness_pins", {})[anchor_id] = digest
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(config, handle)


class TestThePublicPathClosesEndToEnd:
    def test_cli_witness_then_pin_then_confirm_then_verify_passes(self, workspace, capsys):
        aid, _ = _anchor(workspace)

        # Before anything: the legacy anchor is unbound, so reconcile refuses.
        assert reconcile(str(workspace)).ok is False

        # 1. The writer prints the witness and its digest, and writes nothing.
        rc, payload = _witness_from_cli(workspace, capsys)
        assert rc == 0
        assert payload["written"] is False
        digest = payload["digest"]
        assert payload["witness"]["prefix_count"] == 4

        # 2. Confirming WITHOUT a pin is refused: the writer must never supply
        #    its own trust root.
        rc, _ = _witness_from_cli(workspace, capsys, "--confirm")
        assert rc != 0
        assert reconcile(str(workspace)).ok is False

        # 3. The operator pins it under governance. Now the write is allowed.
        _pin(workspace, aid, digest)
        rc, payload = _witness_from_cli(workspace, capsys, "--confirm")
        assert rc == 0, payload
        assert payload["written"] is True

        # 4. The ORDINARY PUBLIC verifier passes -- verify_workspace, which is
        #    what `mm verify` runs, not just reconcile. Nothing is handed to it.
        result = reconcile(str(workspace))
        assert result.ok is True, result
        assert result.recovery_baseline_entries == 4

        report = verify_workspace(str(workspace))
        assert "cross_ledger" in report.checks, "verify_workspace ran no cross_ledger check"
        assert report.checks["cross_ledger"] is True, report.messages
        assert report.ok is True, report.messages

    def test_a_legitimate_new_admission_does_not_break_the_binding(self, workspace, capsys):
        aid, _ = _anchor(workspace)
        _, payload = _witness_from_cli(workspace, capsys)
        _pin(workspace, aid, payload["digest"])
        _witness_from_cli(workspace, capsys, "--confirm")
        assert reconcile(str(workspace)).ok is True

        # Real post-recovery admission, honestly stamped.
        HashChainV2(os.path.join(str(workspace), "memory", "hash_chain_v2.db")).append(
            block_id="NEW",
            action="create_block",
            content="after recovery",
            timestamp="2026-02-01T00:00:00+00:00",
        )
        result = reconcile(str(workspace))
        assert result.ok is True, "an honest later admission must not invalidate the witness"
        assert result.recovery_baseline_entries == 4, "the boundary must stay at 4, not absorb the new row"


class TestThePublicPathRefuses:
    def _bind(self, workspace, capsys):
        aid, _ = _anchor(workspace)
        _, payload = _witness_from_cli(workspace, capsys)
        _pin(workspace, aid, payload["digest"])
        _witness_from_cli(workspace, capsys, "--confirm")
        return aid

    def test_an_altered_count_with_a_recomputed_self_digest_is_refused(self, workspace, capsys):
        """The attestation cannot supply its own trust root."""
        aid = self._bind(workspace, capsys)
        assert reconcile(str(workspace)).ok is True

        store = os.path.join(str(workspace), "memory", "evidence_chain.jsonl")
        lines = [ln for ln in open(store, encoding="utf-8") if ln.strip()]
        rec = json.loads(lines[-1])
        content = rec["metadata"][bw.WITNESS_CONTENT_KEY]
        content["prefix_count"] = 3
        content["prefix_tail"] = bw._prefix_tail(store, 3)
        rec["metadata"][bw.WITNESS_CONTENT_KEY] = content
        rec["metadata"]["companion_hash_chain_entries"] = 3
        rec["metadata"]["companion_hash_chain_tail"] = content["prefix_tail"]
        lines[-1] = json.dumps(rec) + "\n"
        with open(store, "w", encoding="utf-8") as handle:
            handle.writelines(lines)

        # Even though the attacker can recompute a digest OVER their own content,
        # the pinned digest is the one that must be reproduced.
        assert bw.witness_digest(content) != bw.governed_witness_pin(store, aid)
        assert reconcile(str(workspace)).ok is False

    def test_an_absent_pin_is_refused(self, workspace, capsys):
        self._bind(workspace, capsys)
        assert reconcile(str(workspace)).ok is True
        os.remove(os.path.join(str(workspace), "mind-mem.json"))
        assert reconcile(str(workspace)).ok is False, "no pin means no authority"

    def test_a_corrupt_companion_is_refused_and_distinguished_from_absent(self, workspace, capsys):
        self._bind(workspace, capsys)
        assert reconcile(str(workspace)).ok is True

        db = os.path.join(str(workspace), "memory", "hash_chain_v2.db")
        with open(db, "r+b") as handle:
            handle.seek(0)
            handle.write(b"NOTASQLITEFILE!!")
        store = os.path.join(str(workspace), "memory", "evidence_chain.jsonl")
        status, _, _ = bw.companion_status(store)
        assert status == bw.COMPANION_CORRUPT, "corrupt must not read as absent"

        os.remove(db)
        status, _, _ = bw.companion_status(store)
        assert status == bw.COMPANION_ABSENT
        assert bw.COMPANION_CORRUPT != bw.COMPANION_ABSENT

    def test_the_writer_refuses_a_count_past_the_chain(self, workspace, capsys):
        rc, _ = _witness_from_cli(workspace, capsys, "--entries", "99")
        assert rc != 0


class TestTheWitnessGuardIsLoadBearing:
    """A validly-recreated altered attestation, so only the witness can refuse.

    The tamper test above edits the stored JSON in place, which invalidates the
    record's own evidence_hash -- so the chain's integrity check could be what
    refuses, and the witness guard would look load-bearing without being tested
    at all. Here the altered attestation is APPENDED THROUGH THE REAL WRITER, so
    its record hash is valid and its chain links are intact. The governed pin is
    left untouched. Nothing but the witness binding can reject it.
    """

    def _bound(self, workspace, capsys):
        aid, _ = _anchor(workspace)
        _, payload = _witness_from_cli(workspace, capsys)
        _pin(workspace, aid, payload["digest"])
        _witness_from_cli(workspace, capsys, "--confirm")
        assert reconcile(str(workspace)).ok is True
        return aid

    def test_a_validly_recreated_altered_attestation_is_refused(self, workspace, capsys):
        aid, ahash = _anchor(workspace)
        self._bound(workspace, capsys)

        store = os.path.join(str(workspace), "memory", "evidence_chain.jsonl")
        pin_before = bw.governed_witness_pin(store, aid)
        assert pin_before, "fixture: the pin must exist for this test to mean anything"

        # A second attestation over a DIFFERENT prefix, appended properly so the
        # record hash and chain links are valid.
        short_tail = bw._prefix_tail(store, 3)
        altered = {
            "anchor_id": aid,
            "anchor_hash": ahash,
            "prefix_count": 3,
            "prefix_tail": short_tail,
            "authority": "operator",
            "trust_assumption": "fixes the boundary under operator authority; restores no historic trust",
        }
        from mind_mem.evidence_objects import EvidenceAction
        from mind_mem.evidence_recovery import (
            ATTESTS_ANCHOR_HASH_KEY,
            ATTESTS_ANCHOR_ID_KEY,
            BASELINE_VERB,
            COMPANION_ENTRIES_KEY,
            COMPANION_PRESENT_KEY,
            COMPANION_TAIL_KEY,
            DENIES_PREDECESSOR_KEY,
            RECOVERY_VERB_KEY,
            TRUST_RESTORED_KEY,
        )

        rec0 = json.loads(open(store, encoding="utf-8").readline())
        arch = {k: (rec0.get("metadata") or {}).get(k) for k in ("archived_chain", "archived_sha256", "archived_bytes")}
        EvidenceChain(store_path=store).create(
            action=EvidenceAction.ROLLBACK,
            actor="operator",
            target_block_id=aid,
            target_file="",
            metadata={
                RECOVERY_VERB_KEY: BASELINE_VERB,
                ATTESTS_ANCHOR_ID_KEY: aid,
                ATTESTS_ANCHOR_HASH_KEY: ahash,
                COMPANION_PRESENT_KEY: True,
                COMPANION_ENTRIES_KEY: 3,
                COMPANION_TAIL_KEY: short_tail,
                DENIES_PREDECESSOR_KEY: True,
                TRUST_RESTORED_KEY: False,
                bw.WITNESS_CONTENT_KEY: altered,
                **arch,
            },
        )

        # The chain itself still verifies -- so the refusal cannot be the record
        # hash or a broken link.
        ok, _ = EvidenceChain(store_path=store).verify_chain()
        assert ok is True, "the altered attestation must be a VALID chain record"
        assert bw.governed_witness_pin(store, aid) == pin_before, "the pin must be untouched"
        assert bw.witness_digest(altered) != pin_before

        assert reconcile(str(workspace)).ok is False, (
            "a validly-recorded attestation over the wrong prefix was accepted; the witness guard is not load-bearing"
        )
        assert verify_workspace(str(workspace)).checks.get("cross_ledger") is not True


class TestCorruptCompanionRefusesThroughThePublicPath:
    """Not just companion_status -- the writer and the public verifier."""

    def test_confirm_and_the_public_verifier_both_refuse(self, workspace, capsys):
        aid, _ = _anchor(workspace)
        _, payload = _witness_from_cli(workspace, capsys)
        _pin(workspace, aid, payload["digest"])

        db = os.path.join(str(workspace), "memory", "hash_chain_v2.db")
        with open(db, "r+b") as handle:
            handle.seek(0)
            handle.write(b"NOTASQLITEFILE!!")

        store = os.path.join(str(workspace), "memory", "evidence_chain.jsonl")
        before = sum(1 for ln in open(store, encoding="utf-8") if ln.strip())

        rc, _ = _witness_from_cli(workspace, capsys, "--confirm")
        assert rc != 0, "the writer confirmed a boundary over a corrupt companion chain"
        after = sum(1 for ln in open(store, encoding="utf-8") if ln.strip())
        assert after == before, "a record was appended despite the corrupt companion"

        assert verify_workspace(str(workspace)).checks.get("cross_ledger") is not True


class TestAdmissionOrderingIsObserved:
    def test_the_attestation_is_admitted_strictly_after_the_bound_prefix(self, workspace, capsys):
        """Ordering proves sequence, not exactness -- observed explicitly."""
        aid, _ = _anchor(workspace)
        _, payload = _witness_from_cli(workspace, capsys)
        _pin(workspace, aid, payload["digest"])
        _witness_from_cli(workspace, capsys, "--confirm")

        store = os.path.join(str(workspace), "memory", "evidence_chain.jsonl")
        rows = [json.loads(ln) for ln in open(store, encoding="utf-8") if ln.strip()]
        anchor_idx = next(i for i, r in enumerate(rows) if r["evidence_id"] == aid)
        att_idx = next(i for i, r in enumerate(rows) if bw.WITNESS_CONTENT_KEY in (r.get("metadata") or {}))
        assert att_idx > anchor_idx, "the attestation must be admitted after its anchor"
        assert rows[att_idx]["metadata"][bw.WITNESS_CONTENT_KEY]["prefix_count"] == 4


class TestClosureAtTheCommandLevel:
    """Through `mind-mem-verify` itself, not the function it calls.

    Root asked for closure at `mm verify`. That subcommand does not exist -- the
    public entry is the `mind-mem-verify` console script, declared in
    pyproject.toml as mind_mem.verify_cli:main. This drives that main() with
    real argv and reads its exit code, so the closure is proven at the layer an
    operator and a CI gate actually use.
    """

    def test_the_verify_command_exits_zero_once_the_witness_is_bound(self, workspace, capsys):
        from mind_mem.verify_cli import main as verify_main

        aid, _ = _anchor(workspace)

        # Before binding, the command must FAIL -- otherwise a later pass proves
        # nothing about the witness.
        rc_before = verify_main([str(workspace), "--json"])
        capsys.readouterr()
        assert rc_before != 0, "the verify command passed on an unbound legacy anchor"

        _, payload = _witness_from_cli(workspace, capsys)
        _pin(workspace, aid, payload["digest"])
        _witness_from_cli(workspace, capsys, "--confirm")

        rc_after = verify_main([str(workspace), "--json"])
        out = capsys.readouterr().out
        assert rc_after == 0, out
        report = json.loads(out[out.index("{") :])
        assert report["checks"]["cross_ledger"] is True, report
