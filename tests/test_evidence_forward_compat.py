# Copyright 2026 STARGA, Inc.
"""A reader must survive a verb it was never taught.

The evidence chain grows new governance verbs (lifecycle deaths add
DEMOTE / ARCHIVE / FORGET). Adding a member is additive for the writer
and *breaking* for a reader that parses strictly: ``EvidenceAction(raw)``
raises, ``_load_from_file`` reads that as "unreadable record", and the
whole chain freezes — one new verb would take the ledger offline for
every other process sharing the workspace.

So the action is verified from its **raw string** (it always was: the
preimage hashes ``action.value``) and dispatched through the enum via
``EvidenceAction.parse``, which never raises. An unmodelled verb
round-trips byte-identically, verifies, and links.

Every "it still reads" assertion here is paired with a control that the
loader's freeze mechanism is alive — a test that an unknown action does
not freeze the chain proves nothing if nothing freezes the chain.
"""

from __future__ import annotations

import json

import pytest

from mind_mem.evidence_objects import (
    EVIDENCE_SCHEMA_VERSION,
    EvidenceAction,
    EvidenceChain,
    EvidenceObject,
    UnknownAction,
    _compute_evidence_hash,
    _compute_payload_hash,
)

_FUTURE_VERB = "FUTURE_ACTION"


def _seed(store: str, n: int = 2) -> EvidenceChain:
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


def _append_raw_record(store: str, *, action: str, previous_hash: str, metadata: dict | None = None) -> str:
    """Write a well-formed record carrying *action*, as a newer release would.

    Hashed with the live v3 scheme over the raw action string, so the
    record is genuinely valid — the only thing this reader lacks is a
    member for the verb.
    """
    payload_hash = _compute_payload_hash(b"payload")
    meta = dict(metadata or {})
    record = {
        "evidence_id": "11111111-2222-3333-4444-555555555555",
        "timestamp": "2026-09-01T00:00:00+00:00",
        "action": action,
        "actor": "a-newer-release",
        "target_block_id": "B-FUTURE",
        "target_file": "decisions/DECISIONS.md",
        "payload_hash": payload_hash,
        "previous_hash": previous_hash,
        "evidence_hash": _compute_evidence_hash(
            "11111111-2222-3333-4444-555555555555",
            "2026-09-01T00:00:00+00:00",
            action,
            "a-newer-release",
            "B-FUTURE",
            payload_hash,
            previous_hash,
            target_file="decisions/DECISIONS.md",
            metadata=meta,
            confidence=1.0,
        ),
        "metadata": meta,
        "confidence": 1.0,
    }
    line = json.dumps(record, separators=(",", ":"))
    with open(store, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    return line


class TestParse:
    def test_known_verbs_dispatch_to_members(self):
        for member in EvidenceAction:
            assert EvidenceAction.parse(member.value) is member

    def test_an_unknown_verb_becomes_a_sentinel_not_an_exception(self):
        parsed = EvidenceAction.parse(_FUTURE_VERB)
        assert isinstance(parsed, UnknownAction)
        assert parsed.value == _FUTURE_VERB
        assert parsed.name == _FUTURE_VERB
        assert parsed == _FUTURE_VERB  # a str, so it compares and serialises

    def test_a_sentinel_is_not_mistaken_for_a_member(self):
        """ "I do not model this" must never read as "APPLY" or as absence."""
        parsed = EvidenceAction.parse(_FUTURE_VERB)
        assert parsed not in set(EvidenceAction)
        for member in EvidenceAction:
            assert parsed != member
        assert bool(parsed) is True


class TestAChainCarryingAnUnknownVerbStillLoads:
    def test_it_loads_verifies_and_links(self, tmp_path):
        store = str(tmp_path / "evidence_chain.jsonl")
        seeded = _seed(store)
        _append_raw_record(store, action=_FUTURE_VERB, previous_hash=seeded._entries[-1].evidence_hash)

        reopened = EvidenceChain(store_path=store)
        assert reopened.integrity_compromised is False
        assert reopened.verify_chain() == (True, [])
        assert len(reopened) == 3
        assert isinstance(reopened.get_latest(1)[0].action, UnknownAction)

    def test_the_record_round_trips_byte_identically(self, tmp_path):
        store = str(tmp_path / "evidence_chain.jsonl")
        seeded = _seed(store)
        written = _append_raw_record(store, action=_FUTURE_VERB, previous_hash=seeded._entries[-1].evidence_hash)

        parsed = EvidenceObject.from_dict(json.loads(written))
        assert json.dumps(parsed.to_dict(), separators=(",", ":")) == written

    def test_the_chain_can_still_be_appended_to(self, tmp_path):
        """The unknown record must be a link, not a wall."""
        store = str(tmp_path / "evidence_chain.jsonl")
        seeded = _seed(store)
        _append_raw_record(store, action=_FUTURE_VERB, previous_hash=seeded._entries[-1].evidence_hash)

        reopened = EvidenceChain(store_path=store)
        ev = reopened.create(
            action=EvidenceAction.APPLY,
            actor="this-release",
            target_block_id="B-AFTER",
            target_file="decisions/DECISIONS.md",
            payload=b"payload",
        )
        rows = [json.loads(line) for line in open(store, encoding="utf-8") if line.strip()]
        assert ev.previous_hash == rows[-2]["evidence_hash"]
        assert EvidenceChain(store_path=store).verify_chain() == (True, [])

    def test_the_freeze_mechanism_is_alive(self, tmp_path):
        """Positive control for the three tests above.

        They assert a chain is NOT frozen. That is only evidence if this
        loader freezes when it should — so tamper with the same record in
        the same place and watch it freeze.
        """
        store = str(tmp_path / "evidence_chain.jsonl")
        seeded = _seed(store)
        _append_raw_record(store, action=_FUTURE_VERB, previous_hash=seeded._entries[-1].evidence_hash)

        lines = open(store, encoding="utf-8").read().splitlines()
        record = json.loads(lines[-1])
        record["actor"] = "tampered"
        lines[-1] = json.dumps(record, separators=(",", ":"))
        with open(store, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")

        reopened = EvidenceChain(store_path=store)
        assert reopened.integrity_compromised is True
        assert reopened.verify_chain() == (False, ["load_integrity_compromised"])


class TestForwardCompatMutationTwin:
    """Restore strict parsing and the same chain goes offline.

    If this twin passed, the tolerance above would be coming from
    somewhere else and ``parse`` would not be what keeps a newer writer's
    chain readable.
    """

    def test_strict_parsing_freezes_the_chain(self, tmp_path, monkeypatch):
        store = str(tmp_path / "evidence_chain.jsonl")
        seeded = _seed(store)
        _append_raw_record(store, action=_FUTURE_VERB, previous_hash=seeded._entries[-1].evidence_hash)

        # The pre-§1.4 reader: EvidenceAction(d["action"]).
        monkeypatch.setattr(EvidenceAction, "parse", classmethod(lambda cls, raw: cls(raw)))

        reopened = EvidenceChain(store_path=store)
        assert reopened.integrity_compromised is True, "strict parsing did not freeze — the twin is not exercising the reader"
        assert "not a readable evidence record" in (reopened.load_failure or "")


class TestSchemaTag:
    def test_every_new_record_carries_the_schema_it_was_written_under(self, tmp_path):
        store = str(tmp_path / "evidence_chain.jsonl")
        chain = EvidenceChain(store_path=store)
        ev = chain.create(
            action=EvidenceAction.APPLY,
            actor="writer",
            target_block_id="B-001",
            target_file="decisions/DECISIONS.md",
            payload=b"payload",
        )
        assert ev.metadata["evidence_schema"] == EVIDENCE_SCHEMA_VERSION

    def test_the_tag_is_covered_by_the_record_hash(self, tmp_path):
        """Free tamper-evidence: the preimage already covers metadata."""
        store = str(tmp_path / "evidence_chain.jsonl")
        chain = EvidenceChain(store_path=store)
        ev = chain.create(
            action=EvidenceAction.APPLY,
            actor="writer",
            target_block_id="B-001",
            target_file="decisions/DECISIONS.md",
            payload=b"payload",
        )
        forged = json.loads(json.dumps(ev.to_dict()))
        forged["metadata"]["evidence_schema"] = "v99.0"
        assert chain.verify(EvidenceObject.from_dict(forged)) is False
        # Control: the untouched record does verify, so the False above is
        # about the tag and not about from_dict losing something.
        assert chain.verify(EvidenceObject.from_dict(json.loads(json.dumps(ev.to_dict())))) is True

    def test_a_record_written_before_the_tag_existed_still_verifies(self, tmp_path):
        """Absence means "<= v3.0", never an error."""
        store = str(tmp_path / "evidence_chain.jsonl")
        seeded = _seed(store)
        _append_raw_record(
            store,
            action=EvidenceAction.APPLY.value,
            previous_hash=seeded._entries[-1].evidence_hash,
            metadata={},
        )
        reopened = EvidenceChain(store_path=store)
        assert reopened.verify_chain() == (True, [])
        assert "evidence_schema" not in reopened.get_latest(1)[0].metadata

    def test_a_caller_that_sets_the_tag_keeps_it(self, tmp_path):
        """A deliberate re-anchoring writes its own tag; we do not overwrite it."""
        store = str(tmp_path / "evidence_chain.jsonl")
        chain = EvidenceChain(store_path=store)
        ev = chain.create(
            action=EvidenceAction.APPLY,
            actor="re-anchor",
            target_block_id="B-001",
            target_file="decisions/DECISIONS.md",
            payload=b"payload",
            metadata={"evidence_schema": "v3.0"},
        )
        assert ev.metadata["evidence_schema"] == "v3.0"


@pytest.mark.parametrize("verb", ["DEMOTE", "ARCHIVE", "FORGET"])
def test_the_lifecycle_death_verbs_are_readable_without_a_member(tmp_path, verb):
    """Lifecycle deaths are the next verbs anyone will want to record.

    This build has no member for any of them — deliberately: the landed
    rule is that a new member breaks an older reader, so DELETE reuses
    ``ROLLBACK`` and puts the distinction in additive metadata. What this
    pins is that the *reader* is no longer the obstacle: a record
    carrying one of these loads, verifies and links on a build that does
    not model it, so the decision about members is a vocabulary decision
    and not a "the chain goes offline" decision.
    """
    store = str(tmp_path / "evidence_chain.jsonl")
    seeded = _seed(store)
    _append_raw_record(store, action=verb, previous_hash=seeded._entries[-1].evidence_hash)

    reopened = EvidenceChain(store_path=store)
    assert reopened.verify_chain() == (True, [])
    assert reopened.get_latest(1)[0].action.value == verb
