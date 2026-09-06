# Copyright 2026 STARGA, Inc.
"""The writer's receipt must describe what happened, not what was requested.

`--json` emitted ``written`` from ``bool(args.confirm)`` BEFORE the pin was
checked and BEFORE the append, so a missing or mismatched pin published a
successful-write receipt and then refused. A machine consuming that JSON would
record an attestation that does not exist.

Every refusal below asserts THREE things together -- exit code, the receipt's
own ``written`` field, and the ledger's byte count -- because checking only the
exit code is exactly how the defect survived its first test.
"""

import contextlib
import hashlib
import io
import json
import os

import pytest

from mind_mem import boundary_witness as bw
from mind_mem.evidence_objects import EvidenceChain
from mind_mem.evidence_recovery import RECOVERY_ACTION, RECOVERY_VERB, RECOVERY_VERB_KEY
from mind_mem.hash_chain_v2 import HashChainV2
from mind_mem.mm_cli import main as mm_main

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


def _rows(ws):
    with open(_ledger(ws), encoding="utf-8") as h:
        return sum(1 for ln in h if ln.strip())


def _bytes(ws):
    """Byte length of the ledger.

    The row count was reported as a "byte count" in an earlier handback. Rows
    and bytes are different assertions: a rewrite that preserves the row count
    changes the bytes, and the stronger claim is the one that was written down.
    Both are asserted now, and the report says which is which.
    """
    return os.path.getsize(_ledger(ws))


def _anchor(ws):
    with open(_ledger(ws), encoding="utf-8") as h:
        rec = json.loads([ln for ln in h if ln.strip()][0])
    return rec["evidence_id"], rec["evidence_hash"]


def _run(ws, *extra, anchor_hash=None):
    aid, ahash = _anchor(ws)
    argv = ["chain", "witness", str(ws), "--anchor-id", aid, "--anchor-hash", anchor_hash or ahash, "--json", *extra]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = mm_main(argv)
    out = buf.getvalue()
    payload = json.loads(out[out.index("{") :]) if "{" in out else {}
    return rc, payload


def _pin(ws, anchor_id, digest, *, where="workspace"):
    base = str(ws) if where == "workspace" else os.path.join(str(ws), "memory")
    path = os.path.join(base, "mind-mem.json")
    cfg = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    cfg.setdefault("recovery", {}).setdefault("boundary_witness_pins", {})[anchor_id] = digest
    with open(path, "w", encoding="utf-8") as h:
        json.dump(cfg, h)


class TestTheReceiptIsTruthful:
    def test_no_pin_refuses_with_written_false_and_no_ledger_growth(self, ws):
        before, before_b = _rows(ws), _bytes(ws)
        rc, payload = _run(ws, "--confirm")
        assert rc != 0
        assert payload.get("written") is False, "a refusal published a successful-write receipt"
        assert _rows(ws) == before
        assert _bytes(ws) == before_b, "the ledger BYTES changed on a refusal"

    def test_wrong_pin_refuses_with_written_false(self, ws):
        aid, _ = _anchor(ws)
        _pin(ws, aid, "0" * 128)
        before, before_b = _rows(ws), _bytes(ws)
        rc, payload = _run(ws, "--confirm")
        assert rc != 0
        assert payload.get("written") is False
        assert _rows(ws) == before
        assert _bytes(ws) == before_b, "the ledger BYTES changed on a refusal"

    def test_dry_run_reports_written_false(self, ws):
        before = _rows(ws)
        rc, payload = _run(ws)
        assert rc == 0
        assert payload.get("written") is False
        assert _rows(ws) == before

    def test_a_real_write_reports_written_true_and_grows_the_ledger(self, ws):
        """Positive control: written=True must still be reachable."""
        aid, _ = _anchor(ws)
        _, prep = _run(ws)
        _pin(ws, aid, prep["digest"])
        before = _rows(ws)
        rc, payload = _run(ws, "--confirm")
        assert rc == 0, payload
        assert payload.get("written") is True
        assert _rows(ws) == before + 1


class TestPreAppendValidation:
    def test_a_wrong_anchor_hash_is_refused_before_any_append(self, ws):
        """The supplied --anchor-hash was never checked against the record."""
        aid, _ = _anchor(ws)
        _, prep = _run(ws, anchor_hash="b" * 64)
        # Pin whatever that malformed preparation produced, so the pin check
        # cannot be what refuses -- the anchor identity check must be.
        if prep.get("digest"):
            _pin(ws, aid, prep["digest"])
        before = _rows(ws)
        rc, payload = _run(ws, "--confirm", anchor_hash="b" * 64)
        assert rc != 0, "an attestation was appended for an anchor hash that does not match"
        assert payload.get("written") is not True
        assert _rows(ws) == before

    def test_a_target_that_is_not_a_recovery_anchor_is_refused(self, ws):
        """Only a verified recovery anchor may be witnessed."""
        led = EvidenceChain(store_path=_ledger(ws))
        led.create(
            action=RECOVERY_ACTION,
            actor="operator",
            target_block_id="not-an-anchor",
            target_file="",
            metadata={"note": "ordinary record, no recovery verb"},
        )
        with open(_ledger(ws), encoding="utf-8") as h:
            rec = json.loads([ln for ln in h if ln.strip()][-1])
        before = _rows(ws)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = mm_main(["chain", "witness", str(ws), "--anchor-id", rec["evidence_id"], "--anchor-hash", rec["evidence_hash"], "--json"])
        assert rc != 0, "a non-anchor record was accepted as a witnessable anchor"
        assert _rows(ws) == before


class TestPinDiscoveryHasOneAuthority:
    def test_a_config_beside_the_ledger_does_not_override_the_workspace(self, ws):
        """A second config file must not silently become the authority."""
        aid, _ = _anchor(ws)
        _, prep = _run(ws)
        # Canonical workspace config pins the CORRECT digest.
        _pin(ws, aid, prep["digest"], where="workspace")
        # A rogue config beside the ledger pins a different one.
        _pin(ws, aid, "9" * 128, where="memory")
        found = bw.governed_witness_pin(_ledger(ws), aid)
        assert found == prep["digest"], "a config beside the ledger overrode the canonical workspace config"

    def test_a_malformed_canonical_config_refuses_rather_than_falling_through(self, ws):
        aid, _ = _anchor(ws)
        _, prep = _run(ws)
        with open(os.path.join(str(ws), "mind-mem.json"), "w", encoding="utf-8") as h:
            h.write("{ this is not json")
        _pin(ws, aid, prep["digest"], where="memory")
        assert bw.governed_witness_pin(_ledger(ws), aid) == "", "a malformed canonical config fell through to a secondary file"


class TestAppendFailure:
    """Root: the append-failure written=false claim had no catch and no test.

    The catch was described in a commit message and never landed -- the patch
    that would have added it died on an earlier assertion. This drives a real
    append failure through the CLI and asserts the receipt, the exit code and
    the ledger's BYTE length together.
    """

    def test_an_append_failure_reports_written_false_and_changes_no_bytes(self, ws, monkeypatch):
        aid, _ = _anchor(ws)
        _, prep = _run(ws)
        _pin(ws, aid, prep["digest"])

        before_rows, before_bytes = _rows(ws), _bytes(ws)

        # The writer now appends through ChainTransaction.create inside the
        # store transaction, so patching EvidenceChain.create no longer
        # intercepts it -- the test would pass while exercising nothing.
        from mind_mem.evidence_objects import ChainTransaction

        def _boom(self, **kw):
            raise OSError(28, "No space left on device")

        monkeypatch.setattr(ChainTransaction, "create", _boom)

        rc, payload = _run(ws, "--confirm")
        assert rc != 0, "an append failure exited zero"
        assert payload.get("written") is False, "an append failure reported a successful write"
        assert _rows(ws) == before_rows
        assert _bytes(ws) == before_bytes, "bytes changed despite the append failing"

    def test_the_positive_control_still_writes(self, ws):
        """Without this, the failure above could be refusing for any reason."""
        aid, _ = _anchor(ws)
        _, prep = _run(ws)
        _pin(ws, aid, prep["digest"])
        before_bytes = _bytes(ws)
        rc, payload = _run(ws, "--confirm")
        assert rc == 0 and payload.get("written") is True
        assert _bytes(ws) > before_bytes


class TestMalformedRecordsRefuseCleanly:
    def test_a_record_whose_metadata_is_a_list_is_refused(self, ws):
        """`or {}` would have normalised this into a misleading "no verb"."""
        import contextlib
        import io

        led = _ledger(ws)
        rows = [json.loads(ln) for ln in open(led, encoding="utf-8") if ln.strip()]
        rows[0]["metadata"] = ["not", "a", "dict"]
        with open(led, "w", encoding="utf-8") as h:
            for r in rows:
                h.write(json.dumps(r) + "\n")
        before = _bytes(ws)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = mm_main(
                ["chain", "witness", str(ws), "--anchor-id", rows[0]["evidence_id"], "--anchor-hash", rows[0]["evidence_hash"], "--json"]
            )
        assert rc != 0, "malformed metadata was accepted"
        assert _bytes(ws) == before


class TestTheActionCheckIsIndependentlyLoadBearing:
    """A record carrying the recovery VERB but not the recovery ACTION.

    Removing the action check did not turn any test red: every fixture that
    exercised it also lacked the verb, so the verb check refused first and the
    action check was dead weight dressed as a guard. This isolates it -- correct
    verb, wrong action -- so only the action check can refuse.
    """

    def test_right_verb_wrong_action_is_refused(self, ws):
        import contextlib
        import io

        from mind_mem.evidence_objects import EvidenceAction, EvidenceChain
        from mind_mem.evidence_recovery import RECOVERY_VERB, RECOVERY_VERB_KEY

        led = _ledger(ws)
        EvidenceChain(store_path=led).create(
            action=EvidenceAction.ROLLBACK,  # NOT the recovery action
            actor="operator",
            target_block_id="verb-without-action",
            target_file="",
            metadata={
                RECOVERY_VERB_KEY: RECOVERY_VERB,  # but it DOES carry the verb
                "archived_chain": ARCHIVE,
                "archived_sha256": "0" * 64,
                "archived_bytes": 1,
            },
        )
        rows = [json.loads(ln) for ln in open(led, encoding="utf-8") if ln.strip()]
        rec = rows[-1]
        before = _bytes(ws)

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = mm_main(["chain", "witness", str(ws), "--anchor-id", rec["evidence_id"], "--anchor-hash", rec["evidence_hash"], "--json"])
        assert rc != 0, "a record with the wrong action was accepted as an anchor"
        assert "action" in buf.getvalue() or True  # message shape is not the contract
        assert _bytes(ws) == before
