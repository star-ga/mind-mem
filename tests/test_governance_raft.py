"""v4.0 prep — Raft-style consensus wrapper for governance writes."""

from __future__ import annotations

import secrets

import pytest

from mind_mem.governance_raft import (
    CommitResult,
    LocalConsensusLog,
    Proposal,
    create_consensus_log,
    register_consensus_log,
    replicate,
    sign_proposal,
    verify_proposal,
)


@pytest.fixture
def log() -> LocalConsensusLog:
    return LocalConsensusLog()


@pytest.fixture(autouse=True)
def _reset_registry() -> None:
    # Undo any register_consensus_log() calls a test might make.
    from mind_mem import governance_raft

    yield
    governance_raft._factory = None  # type: ignore[attr-defined]


class TestProposal:
    def test_digest_stable_across_calls(self) -> None:
        p = Proposal(operation="APPEND", payload={"x": 1}, client_id="a")
        assert p.digest() == p.digest()

    def test_different_payload_different_digest(self) -> None:
        a = Proposal(operation="APPEND", payload={"x": 1}, client_id="c")
        b = Proposal(operation="APPEND", payload={"x": 2}, client_id="c")
        assert a.digest() != b.digest()


class TestLocalConsensusLog:
    def test_submit_commits_immediately(self, log: LocalConsensusLog) -> None:
        r = log.submit(Proposal(operation="APPEND", payload={}, client_id="c"))
        assert r.committed is True
        assert r.index == 1
        assert r.term == 1
        assert r.reason == "local_immediate"

    def test_indexes_monotonic(self, log: LocalConsensusLog) -> None:
        a = log.submit(Proposal(operation="APPEND", payload={}, client_id="c"))
        b = log.submit(Proposal(operation="APPEND", payload={}, client_id="c"))
        assert b.index == a.index + 1

    def test_is_leader_true(self, log: LocalConsensusLog) -> None:
        assert log.is_leader() is True

    def test_subscribers_notified_on_commit(self, log: LocalConsensusLog) -> None:
        seen: list[tuple[Proposal, CommitResult]] = []
        log.subscribe(lambda p, r: seen.append((p, r)))
        log.submit(Proposal(operation="APPEND", payload={"a": 1}, client_id="c"))
        assert len(seen) == 1
        assert seen[0][0].payload == {"a": 1}

    def test_failing_handler_doesnt_block_commit(self, log: LocalConsensusLog) -> None:
        log.subscribe(lambda p, r: (_ for _ in ()).throw(RuntimeError("boom")))
        # submit must still succeed despite the handler raising
        r = log.submit(Proposal(operation="APPEND", payload={}, client_id="c"))
        assert r.committed is True

    def test_log_entries_accumulate(self, log: LocalConsensusLog) -> None:
        for i in range(3):
            log.submit(Proposal(operation="APPEND", payload={"i": i}, client_id="c"))
        entries = log.log_entries()
        assert [p.payload["i"] for p, _ in entries] == [0, 1, 2]


class TestFactoryRegistry:
    def test_default_is_local(self) -> None:
        log = create_consensus_log({})
        assert isinstance(log, LocalConsensusLog)

    def test_register_override(self) -> None:
        captured: dict = {}

        class _StubLog:
            def __init__(self, cfg: dict) -> None:
                captured["cfg"] = cfg

            def submit(self, proposal, *, timeout_seconds=5.0):
                return CommitResult(committed=True, term=7, index=42, digest=b"d")

            def subscribe(self, handler):
                pass

            def current_term(self):
                return 7

            def is_leader(self):
                return True

            def close(self):
                pass

        register_consensus_log(lambda cfg: _StubLog(cfg))
        log = create_consensus_log({"hosts": ["a", "b"]})
        r = log.submit(Proposal(operation="APPEND", payload={}, client_id="c"))
        assert r.term == 7
        assert captured["cfg"]["hosts"] == ["a", "b"]


class TestReplicate:
    def test_returns_per_proposal_results(self, log: LocalConsensusLog) -> None:
        proposals = [Proposal(operation="APPEND", payload={"n": i}, client_id="c") for i in range(3)]
        results = replicate(log, proposals)
        assert len(results) == 3
        assert [r.committed for r in results] == [True, True, True]
        assert [r.index for r in results] == [1, 2, 3]


class TestProposalSigning:
    def test_sign_verify_roundtrip(self) -> None:
        secret = secrets.token_bytes(32)
        p = Proposal(operation="APPEND", payload={"x": 1}, client_id="c")
        sig = sign_proposal(p, secret)
        assert verify_proposal(p, sig, secret) is True

    def test_wrong_secret_fails(self) -> None:
        a = secrets.token_bytes(32)
        b = secrets.token_bytes(32)
        p = Proposal(operation="APPEND", payload={}, client_id="c")
        sig = sign_proposal(p, a)
        assert verify_proposal(p, sig, b) is False

    def test_tampered_payload_fails(self) -> None:
        secret = secrets.token_bytes(32)
        original = Proposal(operation="APPEND", payload={"x": 1}, client_id="c")
        tampered = Proposal(operation="APPEND", payload={"x": 2}, client_id="c")
        sig = sign_proposal(original, secret)
        assert verify_proposal(tampered, sig, secret) is False

    def test_short_secret_rejected(self) -> None:
        p = Proposal(operation="APPEND", payload={}, client_id="c")
        with pytest.raises(ValueError):
            sign_proposal(p, b"too-short")
        # verify_proposal returns False rather than raising on a
        # short secret — attacker-supplied secrets shouldn't reveal
        # length-check signals.
        assert verify_proposal(p, b"any", b"too-short") is False
