# Copyright 2026 STARGA, Inc.
"""Tests for mind-mem Evidence Objects — structured tamper-evident governance records."""

from __future__ import annotations

import json
import os

import pytest

from mind_mem.evidence_objects import (
    EvidenceAction,
    EvidenceChain,
    EvidenceObject,
    _compute_evidence_hash,
    _compute_payload_hash,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def chain() -> EvidenceChain:
    """In-memory EvidenceChain with no persistence."""
    return EvidenceChain()


@pytest.fixture
def persisted_chain(tmp_path) -> EvidenceChain:
    """EvidenceChain backed by a temp JSONL file."""
    store = str(tmp_path / "evidence.jsonl")
    return EvidenceChain(store_path=store)


def _make_evidence(chain: EvidenceChain, **kwargs) -> EvidenceObject:
    """Helper to create an evidence object with sensible defaults."""
    defaults = dict(
        action=EvidenceAction.PROPOSE,
        actor="test_actor",
        target_block_id="B-001",
        target_file="decisions/DECISIONS.md",
        payload=b"some content",
        metadata={"proposal": "test proposal"},
        confidence=0.9,
    )
    defaults.update(kwargs)
    return chain.create(**defaults)


# ---------------------------------------------------------------------------
# EvidenceAction enum
# ---------------------------------------------------------------------------


class TestEvidenceAction:
    def test_all_required_actions_exist(self):
        names = {a.name for a in EvidenceAction}
        required = {"PROPOSE", "APPLY", "ROLLBACK", "CONTRADICT", "DRIFT", "RESOLVE", "VERIFY"}
        assert required == names

    def test_action_values_are_strings(self):
        for action in EvidenceAction:
            assert isinstance(action.value, str)
            assert action.value  # non-empty


# ---------------------------------------------------------------------------
# Hash helpers
# ---------------------------------------------------------------------------


class TestHashHelpers:
    def test_payload_hash_bytes(self):
        h = _compute_payload_hash(b"hello")
        assert len(h) == 64  # SHA256 hex digest
        assert h == _compute_payload_hash(b"hello")  # deterministic

    def test_payload_hash_str(self):
        h = _compute_payload_hash("hello")
        assert len(h) == 64
        assert h == _compute_payload_hash("hello")

    def test_payload_hash_dict(self):
        h = _compute_payload_hash({"key": "value"})
        assert len(h) == 64
        # Same dict always produces same hash (sorted keys)
        assert h == _compute_payload_hash({"key": "value"})

    def test_payload_hash_dict_key_order_invariant(self):
        h1 = _compute_payload_hash({"b": 2, "a": 1})
        h2 = _compute_payload_hash({"a": 1, "b": 2})
        assert h1 == h2

    def test_evidence_hash_deterministic(self):
        h1 = _compute_evidence_hash(
            "ev-001",
            "2026-01-01T00:00:00+00:00",
            "PROPOSE",
            "actor1",
            "B-001",
            "ph1",
            "prev1",
        )
        h2 = _compute_evidence_hash(
            "ev-001",
            "2026-01-01T00:00:00+00:00",
            "PROPOSE",
            "actor1",
            "B-001",
            "ph1",
            "prev1",
        )
        assert h1 == h2
        assert len(h1) == 64

    def test_evidence_hash_changes_with_any_field(self):
        base_args = (
            "ev-001",
            "2026-01-01T00:00:00+00:00",
            "PROPOSE",
            "actor1",
            "B-001",
            "ph1",
            "prev1",
        )
        base = _compute_evidence_hash(*base_args)
        # Change each field individually and confirm hash changes
        assert _compute_evidence_hash("ev-CHANGED", *base_args[1:]) != base
        assert _compute_evidence_hash(base_args[0], "2026-01-02T00:00:00+00:00", *base_args[2:]) != base
        assert _compute_evidence_hash(*base_args[:2], "APPLY", *base_args[3:]) != base
        assert _compute_evidence_hash(*base_args[:3], "other_actor", *base_args[4:]) != base


# ---------------------------------------------------------------------------
# EvidenceObject creation
# ---------------------------------------------------------------------------


class TestEvidenceObjectCreation:
    def test_create_returns_evidence_object(self, chain):
        ev = _make_evidence(chain)
        assert isinstance(ev, EvidenceObject)

    def test_evidence_id_is_uuid4(self, chain):
        import uuid

        ev = _make_evidence(chain)
        parsed = uuid.UUID(ev.evidence_id, version=4)
        assert str(parsed) == ev.evidence_id

    def test_timestamp_is_utc(self, chain):
        ev = _make_evidence(chain)
        assert ev.timestamp.tzinfo is not None
        assert ev.timestamp.utcoffset().total_seconds() == 0.0

    def test_action_stored_correctly(self, chain):
        ev = _make_evidence(chain, action=EvidenceAction.APPLY)
        assert ev.action == EvidenceAction.APPLY

    def test_actor_stored(self, chain):
        ev = _make_evidence(chain, actor="drift_detector")
        assert ev.actor == "drift_detector"

    def test_target_fields_stored(self, chain):
        ev = _make_evidence(chain, target_block_id="D-20260401-007", target_file="mem/notes.md")
        assert ev.target_block_id == "D-20260401-007"
        assert ev.target_file == "mem/notes.md"

    def test_confidence_stored(self, chain):
        ev = _make_evidence(chain, confidence=0.75)
        assert ev.confidence == 0.75

    def test_metadata_stored(self, chain):
        meta = {"detail": "some info", "count": 3}
        ev = _make_evidence(chain, metadata=meta)
        assert ev.metadata == meta

    def test_payload_hash_is_sha256(self, chain):
        ev = _make_evidence(chain, payload=b"test payload")
        assert len(ev.payload_hash) == 64
        assert all(c in "0123456789abcdef" for c in ev.payload_hash)

    def test_first_evidence_previous_hash_is_genesis(self, chain):
        ev = _make_evidence(chain)
        # Genesis hash is 64 zeros
        assert ev.previous_hash == "0" * 64

    def test_evidence_is_frozen(self, chain):
        ev = _make_evidence(chain)
        with pytest.raises((AttributeError, TypeError)):
            ev.actor = "tampered"  # type: ignore[misc]

    def test_confidence_bounds_validated(self, chain):
        with pytest.raises(ValueError, match="confidence"):
            _make_evidence(chain, confidence=1.5)
        with pytest.raises(ValueError, match="confidence"):
            _make_evidence(chain, confidence=-0.1)

    def test_confidence_boundary_values_accepted(self, chain):
        ev0 = _make_evidence(chain, confidence=0.0)
        assert ev0.confidence == 0.0
        ev1 = _make_evidence(chain, confidence=1.0)
        assert ev1.confidence == 1.0


# ---------------------------------------------------------------------------
# Self-hash verification
# ---------------------------------------------------------------------------


class TestEvidenceHashVerification:
    def test_verify_valid_evidence(self, chain):
        ev = _make_evidence(chain)
        assert chain.verify(ev) is True

    def test_tamper_actor_fails_verify(self, chain):
        ev = _make_evidence(chain)
        # Bypass frozen dataclass via object.__setattr__
        import dataclasses

        tampered = dataclasses.replace(ev, actor="hacker")
        assert chain.verify(tampered) is False

    def test_tamper_payload_hash_fails_verify(self, chain):
        import dataclasses

        ev = _make_evidence(chain)
        tampered = dataclasses.replace(ev, payload_hash="a" * 64)
        assert chain.verify(tampered) is False

    def test_tamper_previous_hash_fails_verify(self, chain):
        import dataclasses

        ev = _make_evidence(chain)
        tampered = dataclasses.replace(ev, previous_hash="b" * 64)
        assert chain.verify(tampered) is False

    def test_tamper_action_fails_verify(self, chain):
        import dataclasses

        ev = _make_evidence(chain)
        tampered = dataclasses.replace(ev, action=EvidenceAction.ROLLBACK, evidence_hash=ev.evidence_hash)
        # evidence_hash was computed for PROPOSE, but now action is ROLLBACK
        assert chain.verify(tampered) is False


# ---------------------------------------------------------------------------
# Chain linkage (previous_hash)
# ---------------------------------------------------------------------------


class TestChainLinkage:
    def test_second_evidence_links_to_first(self, chain):
        ev1 = _make_evidence(chain)
        ev2 = _make_evidence(chain)
        assert ev2.previous_hash == ev1.evidence_hash

    def test_third_evidence_links_to_second(self, chain):
        ev1 = _make_evidence(chain)
        ev2 = _make_evidence(chain)
        ev3 = _make_evidence(chain)
        assert ev3.previous_hash == ev2.evidence_hash
        assert ev2.previous_hash == ev1.evidence_hash

    def test_verify_chain_empty_is_valid(self, chain):
        valid, broken = chain.verify_chain()
        assert valid is True
        assert broken == []

    def test_verify_chain_single_entry(self, chain):
        _make_evidence(chain)
        valid, broken = chain.verify_chain()
        assert valid is True
        assert broken == []

    def test_verify_chain_multiple_entries(self, chain):
        for _ in range(5):
            _make_evidence(chain)
        valid, broken = chain.verify_chain()
        assert valid is True
        assert broken == []

    def test_verify_chain_detects_tampered_entry(self, chain):
        import dataclasses

        ev1 = _make_evidence(chain, target_block_id="B-001")
        _make_evidence(chain, target_block_id="B-002")

        # Replace ev1 in the chain with a tampered version
        tampered = dataclasses.replace(ev1, actor="hacker")
        chain._entries[0] = tampered

        valid, broken = chain.verify_chain()
        assert valid is False
        assert len(broken) > 0

    def test_verify_chain_detects_broken_linkage(self, chain):
        import dataclasses

        _make_evidence(chain, target_block_id="B-001")
        ev2 = _make_evidence(chain, target_block_id="B-002")
        # Corrupt ev2's previous_hash without touching evidence_hash (broken link)
        tampered = dataclasses.replace(ev2, previous_hash="c" * 64)
        chain._entries[1] = tampered

        valid, broken = chain.verify_chain()
        assert valid is False
        assert ev2.evidence_id in broken


# ---------------------------------------------------------------------------
# Query methods
# ---------------------------------------------------------------------------


class TestQueryMethods:
    def test_get_evidence_for_block_returns_matching(self, chain):
        _make_evidence(chain, target_block_id="B-001")
        _make_evidence(chain, target_block_id="B-002")
        _make_evidence(chain, target_block_id="B-001")

        results = chain.get_evidence_for_block("B-001")
        assert len(results) == 2
        assert all(e.target_block_id == "B-001" for e in results)

    def test_get_evidence_for_block_empty(self, chain):
        assert chain.get_evidence_for_block("nonexistent") == []

    def test_get_evidence_by_action(self, chain):
        _make_evidence(chain, action=EvidenceAction.PROPOSE)
        _make_evidence(chain, action=EvidenceAction.APPLY)
        _make_evidence(chain, action=EvidenceAction.PROPOSE)

        results = chain.get_evidence_by_action(EvidenceAction.PROPOSE)
        assert len(results) == 2
        assert all(e.action == EvidenceAction.PROPOSE for e in results)

    def test_get_evidence_by_action_no_matches(self, chain):
        _make_evidence(chain, action=EvidenceAction.PROPOSE)
        assert chain.get_evidence_by_action(EvidenceAction.ROLLBACK) == []

    def test_get_latest_returns_most_recent(self, chain):
        for i in range(7):
            _make_evidence(chain, target_block_id=f"B-{i:03d}")

        latest = chain.get_latest(3)
        assert len(latest) == 3
        # Most recent last entries
        assert latest[-1].target_block_id == "B-006"
        assert latest[-2].target_block_id == "B-005"
        assert latest[-3].target_block_id == "B-004"

    def test_get_latest_default_is_ten(self, chain):
        for i in range(15):
            _make_evidence(chain)
        assert len(chain.get_latest()) == 10

    def test_get_latest_on_empty_chain(self, chain):
        assert chain.get_latest() == []

    def test_get_latest_fewer_than_n(self, chain):
        _make_evidence(chain)
        _make_evidence(chain)
        results = chain.get_latest(10)
        assert len(results) == 2


# ---------------------------------------------------------------------------
# JSONL export / import round-trip
# ---------------------------------------------------------------------------


class TestJsonlRoundTrip:
    def test_export_creates_file(self, chain, tmp_path):
        _make_evidence(chain)
        out = str(tmp_path / "out.jsonl")
        chain.export_jsonl(out)
        assert os.path.isfile(out)

    def test_export_import_round_trip(self, chain, tmp_path):
        for i in range(4):
            _make_evidence(chain, target_block_id=f"B-{i:03d}", confidence=float(i) / 4)

        out = str(tmp_path / "chain.jsonl")
        chain.export_jsonl(out)

        restored = EvidenceChain()
        restored.import_jsonl(out)

        assert len(restored._entries) == 4
        for orig, rest in zip(chain._entries, restored._entries):
            assert orig.evidence_id == rest.evidence_id
            assert orig.action == rest.action
            assert orig.actor == rest.actor
            assert orig.payload_hash == rest.payload_hash
            assert orig.evidence_hash == rest.evidence_hash

    def test_import_verifies_chain_integrity(self, chain, tmp_path):
        _make_evidence(chain)
        out = str(tmp_path / "chain.jsonl")
        chain.export_jsonl(out)

        # Corrupt the file
        with open(out, "r") as f:
            lines = f.readlines()
        data = json.loads(lines[0])
        data["actor"] = "tampered"
        with open(out, "w") as f:
            f.write(json.dumps(data) + "\n")

        fresh = EvidenceChain()
        with pytest.raises(ValueError, match="tamper"):
            fresh.import_jsonl(out)

    def test_export_empty_chain(self, chain, tmp_path):
        out = str(tmp_path / "empty.jsonl")
        chain.export_jsonl(out)
        assert os.path.isfile(out)
        with open(out) as f:
            content = f.read().strip()
        assert content == ""

    def test_import_empty_file(self, tmp_path):
        out = str(tmp_path / "empty.jsonl")
        with open(out, "w") as f:
            f.write("")
        fresh = EvidenceChain()
        fresh.import_jsonl(out)
        assert fresh._entries == []


# ---------------------------------------------------------------------------
# Persistence (store_path)
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_persisted_chain_writes_file(self, persisted_chain):
        _make_evidence(persisted_chain)
        assert os.path.isfile(persisted_chain._store_path)

    def test_persisted_chain_survives_reload(self, tmp_path):
        store = str(tmp_path / "evidence.jsonl")
        c1 = EvidenceChain(store_path=store)
        _make_evidence(c1, target_block_id="B-PERSIST")

        c2 = EvidenceChain(store_path=store)
        assert len(c2._entries) == 1
        assert c2._entries[0].target_block_id == "B-PERSIST"

    def test_persisted_chain_append_integrity(self, tmp_path):
        store = str(tmp_path / "evidence.jsonl")
        c1 = EvidenceChain(store_path=store)
        _make_evidence(c1)
        _make_evidence(c1)

        c2 = EvidenceChain(store_path=store)
        valid, broken = c2.verify_chain()
        assert valid is True
        assert broken == []
