#!/usr/bin/env python3
"""Tests for fold_attestation.py — evidence-anchored folded==ran attestation.

The compressor is injected and pure, so every property below is proven
deterministically with zero API calls. The audit chain is real (a temp
workspace), so the anchoring + tamper-evidence are exercised end to end.

Hard rail under test: the chain is *tamper-evident, not signed*. The forged-
but-self-consistent test makes that explicit — a hand-recomputed attestation
hash passes the record check, and only the re-run-and-diff catches the lie.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from mind_mem.audit_chain import AuditChain, _payload_hash
from mind_mem.compressors import EchoCompressor, OllamaCompressor
from mind_mem.fold_attestation import (
    FOLD_ATTEST_OP,
    FOLD_ATTEST_TAG,
    FoldAttestation,
    attest_fold,
    build_attestation,
    compressor_identity,
    verify_fold,
)
from mind_mem.recompaction import recompact_cluster

# --- pure, deterministic stand-in compressors ------------------------------


def _upper(text: str, blocks: list[dict]) -> str:
    """Pure + idempotent: ``.upper()`` reaches a byte fixed point on pass 2."""
    return text.upper()


_upper.fold_identity = ("test/upper", "1")  # type: ignore[attr-defined]


def _upper_v2(text: str, blocks: list[dict]) -> str:
    """Byte-identical OUTPUT to ``_upper`` but a different version fingerprint.

    Load-bearing for the version-binding test: the fold it produces is
    indistinguishable by output, yet a verifier must still refuse it because the
    attestation bound version ``1``.
    """
    return text.upper()


_upper_v2.fold_identity = ("test/upper", "2")  # type: ignore[attr-defined]


def _blocks(*bodies: str) -> list[dict]:
    return [{"_id": f"DEC-{i:03d}", "body": b} for i, b in enumerate(bodies, 1)]


@pytest.fixture
def workspace(tmp_path):
    return str(tmp_path)


@pytest.fixture
def chain(workspace):
    return AuditChain(workspace)


@pytest.fixture
def cluster():
    return _blocks("first fact block", "second fact block")


@pytest.fixture
def result(cluster):
    """A converged fold over ``cluster`` with the pure ``_upper`` compressor."""
    return recompact_cluster(cluster, compressor=_upper)


# --- emission + chaining ----------------------------------------------------


def test_attestation_is_emitted_and_chained(chain, result, cluster):
    """attest_fold appends a fold_attest entry and the chain stays valid."""
    before = chain.entry_count()
    att = attest_fold(chain, result, _upper, agent="dream_cycle", reason="unit test")

    assert chain.entry_count() == before + 1
    ok, errors = chain.verify()
    assert ok, errors

    entry = chain.entries(last_n=1)[0]
    assert entry.operation == FOLD_ATTEST_OP
    assert entry.agent == "dream_cycle"
    # The anchored entry binds the FULL attestation via its payload_hash.
    assert entry.payload_hash == _payload_hash(att.to_dict())
    assert att.is_internally_consistent()


def test_fold_attest_is_a_valid_operation():
    from mind_mem.audit_chain import VALID_OPERATIONS

    assert FOLD_ATTEST_OP in VALID_OPERATIONS


def test_attestation_schema_shape(result):
    att = build_attestation(result, compressor_id="test/upper", compressor_version="1")
    d = att.to_dict()
    assert d["schema"] == FOLD_ATTEST_TAG
    assert set(d) == {
        "schema",
        "compressor_id",
        "compressor_version",
        "input_digest",
        "output_digest",
        "iterations",
        "trajectory_digest",
        "source_digest",
        "source_ids",
        "attestation_hash",
    }
    assert len(d["attestation_hash"]) == 64
    assert d["source_ids"] == ["DEC-001", "DEC-002"]


# --- folded == ran ----------------------------------------------------------


def test_folded_equals_ran_on_rerun(result, cluster):
    """Re-running the same pure compressor reproduces the attested output_digest."""
    att = build_attestation(result, compressor_id="test/upper", compressor_version="1")
    outcome = verify_fold(att, cluster, _upper)
    assert outcome.ok, outcome.reason
    assert outcome.reason == "folded == ran"
    assert outcome.recomputed_output_digest == att.output_digest


def test_echo_compressor_round_trips(chain):
    """The trivially-converging control attests + verifies (identity from the hook)."""
    blocks = _blocks("alpha block", "beta block")
    res = recompact_cluster(blocks, compressor=EchoCompressor())
    att = attest_fold(chain, res, EchoCompressor())
    assert att.compressor_id == "mind-mem/echo"
    assert att.compressor_version == "1"
    assert verify_fold(att, blocks, EchoCompressor()).ok


# --- tampered output_digest fails verification ------------------------------


def test_tampered_output_digest_fails_record_consistency(result, cluster):
    """A record whose output_digest is edited without recomputing the hash is caught."""
    att = build_attestation(result, compressor_id="test/upper", compressor_version="1")
    tampered = FoldAttestation.from_dict({**att.to_dict(), "output_digest": "0" * 64})
    assert not tampered.is_internally_consistent()

    outcome = verify_fold(tampered, cluster, _upper)
    assert not outcome.ok
    assert "tampered" in outcome.reason


def test_forged_but_self_consistent_attestation_fails_on_rerun(result, cluster):
    """Tamper-evident, NOT signed: a re-hashed forgery passes the record check.

    Anyone can recompute a SHA-256 (there is no signature), so an attacker can
    forge a fully self-consistent attestation over a fake output_digest. The
    re-run-and-diff is what catches it — folded != ran.
    """
    forged_result = dataclasses.replace(result, output_digest="deadbeef" * 8)
    forged = build_attestation(forged_result, compressor_id="test/upper", compressor_version="1")
    assert forged.is_internally_consistent()  # attacker recomputed the hash

    outcome = verify_fold(forged, cluster, _upper)
    assert not outcome.ok
    assert "folded != ran" in outcome.reason
    # The verifier still surfaces what the honest re-run produced.
    assert outcome.recomputed_output_digest == result.output_digest


def test_editing_the_anchored_entry_breaks_the_chain(chain, result, tmp_path):
    """Chain-level tamper-evidence: editing an anchored entry fails chain.verify()."""
    attest_fold(chain, result, _upper)
    assert chain.verify()[0]

    chain_path = tmp_path / ".mind-mem-audit" / "chain.jsonl"
    lines = chain_path.read_text(encoding="utf-8").splitlines()
    entry = json.loads(lines[-1])
    entry["payload_hash"] = "f" * 64  # forge the bound payload hash
    lines[-1] = json.dumps(entry, separators=(",", ":"))
    chain_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    ok, errors = chain.verify()
    assert not ok
    assert any("entry_hash tampered" in e for e in errors)


# --- compressor identity + version binding ----------------------------------


def test_compressor_version_binding_is_present(result):
    """The version is carried in the record AND bound into the preimage."""
    att = build_attestation(result, compressor_id="test/upper", compressor_version="1")
    assert att.compressor_version == "1"
    assert att.to_dict()["compressor_version"] == "1"

    # Changing only the version yields a different attestation hash — it is bound,
    # not decorative.
    other = build_attestation(result, compressor_id="test/upper", compressor_version="2")
    assert other.attestation_hash != att.attestation_hash


def test_tampered_compressor_version_fails_record_consistency(result):
    att = build_attestation(result, compressor_id="test/upper", compressor_version="1")
    tampered = FoldAttestation.from_dict({**att.to_dict(), "compressor_version": "999"})
    assert not tampered.is_internally_consistent()


def test_wrong_compressor_version_is_refused_on_verify(result, cluster):
    """Same output, different version -> the verifier refuses (gate 2), never diffs."""
    att = build_attestation(result, compressor_id="test/upper", compressor_version="1")
    outcome = verify_fold(att, cluster, _upper_v2)  # identical output, version "2"
    assert not outcome.ok
    assert "compressor identity mismatch" in outcome.reason
    # It short-circuited before the re-run.
    assert outcome.recomputed_output_digest is None


def test_wrong_compressor_id_is_refused_on_verify(result, cluster):
    att = build_attestation(result, compressor_id="test/upper", compressor_version="1")
    outcome = verify_fold(att, cluster, EchoCompressor())
    assert not outcome.ok
    assert "compressor identity mismatch" in outcome.reason


def test_compressor_without_fold_identity_is_refused():
    def _anon(text, blocks):
        return text

    with pytest.raises(ValueError, match="fold_identity"):
        compressor_identity(_anon)


def test_ollama_fold_identity_binds_the_deterministic_knobs():
    c = OllamaCompressor(model="mind-mem:4b", seed=7)
    cid, cver = compressor_identity(c)
    assert cid == "mind-mem/ollama:mind-mem:4b"
    assert "seed7" in cver and "temp0.0" in cver and "pipeline1" in cver

    # A different seed is a different function -> a different version.
    _, cver_other = compressor_identity(OllamaCompressor(model="mind-mem:4b", seed=8))
    assert cver_other != cver


# --- input-cluster binding --------------------------------------------------


def test_wrong_input_cluster_is_refused(result):
    """Verifying against a different cluster is refused at the input gate."""
    att = build_attestation(result, compressor_id="test/upper", compressor_version="1")
    outcome = verify_fold(att, _blocks("totally", "different"), _upper)
    assert not outcome.ok
    assert "input cluster digest" in outcome.reason


# --- determinism (no wall-clock / random in the preimage) -------------------


def test_attestation_hash_is_deterministic(result):
    """Building twice from the same fold yields byte-identical hashes — no clock."""
    a = build_attestation(result, compressor_id="test/upper", compressor_version="1")
    b = build_attestation(result, compressor_id="test/upper", compressor_version="1")
    assert a.attestation_hash == b.attestation_hash
    assert a == b


def test_trajectory_binding_is_load_bearing(result):
    """The fold path (trajectory) is bound: editing its digest breaks consistency."""
    att = build_attestation(result, compressor_id="test/upper", compressor_version="1")
    tampered = FoldAttestation.from_dict({**att.to_dict(), "trajectory_digest": "a" * 64})
    assert not tampered.is_internally_consistent()


def test_attestation_is_frozen(result):
    att = build_attestation(result, compressor_id="test/upper", compressor_version="1")
    with pytest.raises(Exception):
        att.output_digest = "mutated"  # type: ignore[misc]
