"""Restore-44 slice 5 — ``consensus_vote`` wired into ``conflict_resolver``.

The module shipped in v3.3.0 with 19 unit tests and **no caller**: nothing in
the product ever asked a quorum anything. These tests cover the wiring, not the
tallying — that a MANUAL-strategy contradiction actually consults the quorum,
that the quorum's winner is *staged* rather than applied, that a vote that has
not passed the governance gate cannot swing it, and that with the flag off the
whole leg is invisible.
"""

from __future__ import annotations

import json
import os

import pytest

from mind_mem.conflict_resolver import (
    ResolutionStrategy,
    analyze_contradiction,
    generate_resolution_proposals,
    resolve_contradictions,
)
from mind_mem.consensus_vote import Vote

# Two decisions with the SAME date, no ConstraintSignatures: no priority
# delta, no scope delta, no timestamp delta — the exact shape that falls
# through every deterministic strategy to MANUAL.
_DECISIONS = (
    "[D-20260201-001]\n"
    "Statement: Use PostgreSQL\n"
    "Date: 2026-02-01\n"
    "Status: active\n\n---\n\n"
    "[D-20260201-002]\n"
    "Statement: Use MySQL\n"
    "Date: 2026-02-01\n"
    "Status: active\n"
)

# "CONTRA-001" deliberately does not match conflict_resolver's
# [A-Z]+-\d{8}-\d{3} id pattern, so the first two regex hits in the body are
# the two decision ids.
_CONTRADICTION = "[CONTRA-001]\nType: contradiction\nBlocks: D-20260201-001 vs D-20260201-002\nStatus: active\n"

_A = "D-20260201-001"
_B = "D-20260201-002"


def _vote_block(bid: str, agent: str, choice: str, status: str = "active") -> str:
    return f"[{bid}]\nContradiction: CONTRA-001\nAgent: {agent}\nChoice: {choice}\nStatus: {status}\n\n---\n\n"


def _workspace(
    tmp_path,
    *,
    votes: str | None = None,
    config: dict | None = None,
    decisions: str = _DECISIONS,
) -> str:
    ws = tmp_path / "ws"
    (ws / "intelligence").mkdir(parents=True)
    (ws / "decisions").mkdir(parents=True)
    (ws / "decisions" / "DECISIONS.md").write_text(decisions, encoding="utf-8")
    (ws / "intelligence" / "CONTRADICTIONS.md").write_text(_CONTRADICTION, encoding="utf-8")
    if votes is not None:
        (ws / "intelligence" / "VOTES.md").write_text(votes, encoding="utf-8")
    if config is not None:
        (ws / "mind-mem.json").write_text(json.dumps(config), encoding="utf-8")
    return str(ws)


def _enabled(**overrides) -> dict:
    consensus = {"enabled": True, "quorum_threshold": 0.66, "min_votes": 2}
    consensus.update(overrides.pop("consensus", {}))
    cfg: dict = {"governance": {"consensus": consensus}}
    cfg.update(overrides)
    return cfg


def _only(resolutions: list[dict]) -> dict:
    assert len(resolutions) == 1, resolutions
    return resolutions[0]


# ---------------------------------------------------------------------------
# Flag OFF — the leg must not exist
# ---------------------------------------------------------------------------


class TestFlagOff:
    def test_votes_present_but_flag_off_stays_manual(self, tmp_path) -> None:
        """A full two-agent quorum sitting on disk changes nothing by default."""
        ws = _workspace(
            tmp_path,
            votes=_vote_block("V-20260201-001", "agent-alice", _A) + _vote_block("V-20260201-002", "agent-bob", _A),
        )
        res = _only(resolve_contradictions(ws))
        assert res["strategy"] == ResolutionStrategy.MANUAL
        assert res["winner_id"] is None
        assert "consensus" not in res

    def test_flag_off_result_identical_to_no_votes_at_all(self, tmp_path) -> None:
        """Byte-identical: the votes file and an explicit ``enabled: false``
        produce exactly the resolution a workspace with neither produces."""
        votes = _vote_block("V-20260201-001", "agent-alice", _A) + _vote_block("V-20260201-002", "agent-bob", _A)
        with_votes = _workspace(
            tmp_path / "a",
            votes=votes,
            config={"governance": {"consensus": {"enabled": False}}},
        )
        bare = _workspace(tmp_path / "b")
        assert resolve_contradictions(with_votes) == resolve_contradictions(bare)

    def test_flag_off_never_opens_the_votes_file(self, monkeypatch, tmp_path) -> None:
        """Not merely "produces the same answer" — the probe does not run.

        A flag that stats/reads a corpus file to decide it is off is
        observable (an audit log, an access time, a lock). This pins that the
        loader is not reached at all on the default path.
        """
        import mind_mem.conflict_resolver as cr

        def _explode(_ws: str) -> dict:
            raise AssertionError("flag-off path must not load the votes corpus")

        monkeypatch.setattr(cr, "_load_consensus_votes", _explode)
        ws = _workspace(tmp_path, votes=_vote_block("V-20260201-001", "agent-alice", _A))
        assert _only(resolve_contradictions(ws))["strategy"] == ResolutionStrategy.MANUAL

    def test_analyze_contradiction_ignores_votes_without_the_config(self) -> None:
        """The pure function's flag is the ``consensus`` argument, not the votes."""
        block_a = {"_id": _A, "Date": "2026-02-01"}
        block_b = {"_id": _B, "Date": "2026-02-01"}
        res = analyze_contradiction(
            block_a,
            block_b,
            votes=[Vote("agent-alice", _A), Vote("agent-bob", _A)],
            consensus=None,
        )
        assert res["strategy"] == ResolutionStrategy.MANUAL
        assert res["winner_id"] is None


# ---------------------------------------------------------------------------
# Flag ON — the quorum is actually consulted
# ---------------------------------------------------------------------------


class TestQuorumConsulted:
    def test_two_agreeing_agents_reach_quorum(self, tmp_path) -> None:
        ws = _workspace(
            tmp_path,
            votes=_vote_block("V-20260201-001", "agent-alice", _A) + _vote_block("V-20260201-002", "agent-bob", _A),
            config=_enabled(),
        )
        res = _only(resolve_contradictions(ws))
        assert res["strategy"] == ResolutionStrategy.CONSENSUS
        assert res["winner_id"] == _A
        assert res["loser_id"] == _B
        assert res["consensus"]["margin"] == pytest.approx(1.0)
        assert res["consensus"]["agents"] == ["agent-alice", "agent-bob"]

    def test_single_operator_falls_back_to_manual(self, tmp_path) -> None:
        """One voter is below ``min_votes`` — resolves exactly as it did before."""
        ws = _workspace(
            tmp_path,
            votes=_vote_block("V-20260201-001", "agent-alice", _A),
            config=_enabled(),
        )
        res = _only(resolve_contradictions(ws))
        assert res["strategy"] == ResolutionStrategy.MANUAL
        assert res["winner_id"] is None

    def test_split_vote_below_threshold_stays_manual(self, tmp_path) -> None:
        ws = _workspace(
            tmp_path,
            votes=_vote_block("V-20260201-001", "agent-alice", _A) + _vote_block("V-20260201-002", "agent-bob", _B),
            config=_enabled(),
        )
        assert _only(resolve_contradictions(ws))["strategy"] == ResolutionStrategy.MANUAL

    def test_namespace_trust_weight_reaches_the_tally(self, tmp_path) -> None:
        """A configured ``trust_weight`` decides a vote the raw count cannot.

        1 vs 1 is a 0.5 margin and no quorum; alice at weight 3 makes it 0.75.
        """
        ws = _workspace(
            tmp_path,
            votes=_vote_block("V-20260201-001", "agent-alice", _A) + _vote_block("V-20260201-002", "agent-bob", _B),
            config=_enabled(namespaces={"agent-alice": {"trust_weight": 3.0}}),
        )
        res = _only(resolve_contradictions(ws))
        assert res["strategy"] == ResolutionStrategy.CONSENSUS
        assert res["winner_id"] == _A
        assert res["consensus"]["margin"] == pytest.approx(0.75)

    def test_vote_cannot_award_itself_trust(self, tmp_path) -> None:
        """A ``TrustWeight`` field in the corpus is not read — weight is config.

        Otherwise anything that could write a vote block could also write its
        own authority to decide the outcome.
        """
        crafted = (
            f"[V-20260201-001]\nContradiction: CONTRA-001\nAgent: agent-mallory\nChoice: {_A}\nTrustWeight: 99\nStatus: active\n\n---\n\n"
        )
        ws = _workspace(
            tmp_path,
            votes=crafted + _vote_block("V-20260201-002", "agent-bob", _B),
            config=_enabled(),
        )
        assert _only(resolve_contradictions(ws))["strategy"] == ResolutionStrategy.MANUAL

    def test_quorum_for_an_unrelated_block_is_refused(self, tmp_path) -> None:
        """A unanimous quorum for an id outside the pair never becomes a winner.

        Without the guard this id would land on the ``Winner:`` line of a
        supersede proposal for a contradiction it has nothing to do with.
        """
        outsider = "D-20260301-009"
        ws = _workspace(
            tmp_path,
            votes=_vote_block("V-20260201-001", "agent-alice", outsider) + _vote_block("V-20260201-002", "agent-bob", outsider),
            config=_enabled(),
        )
        res = _only(resolve_contradictions(ws))
        assert res["strategy"] == ResolutionStrategy.MANUAL
        assert res["winner_id"] is None

    def test_consensus_never_overrides_a_deterministic_strategy(self, tmp_path) -> None:
        """Votes are consulted only where MANUAL would have been the verdict."""
        decisions = (
            "[D-20260201-001]\n"
            "Statement: Use PostgreSQL\n"
            "Date: 2026-02-01\n"
            "Status: active\n\n---\n\n"
            "[D-20260201-002]\n"
            "Statement: Use MySQL\n"
            "Date: 2026-02-09\n"
            "Status: active\n"
        )
        ws = _workspace(
            tmp_path,
            decisions=decisions,
            votes=_vote_block("V-20260201-001", "agent-alice", _A) + _vote_block("V-20260201-002", "agent-bob", _A),
            config=_enabled(),
        )
        res = _only(resolve_contradictions(ws))
        assert res["strategy"] == ResolutionStrategy.TIMESTAMP
        assert res["winner_id"] == _B  # the newer block, not the voted one

    def test_deterministic_across_repeated_runs(self, tmp_path) -> None:
        """No clock and no randomness on this path."""
        ws = _workspace(
            tmp_path,
            votes=_vote_block("V-20260201-001", "agent-alice", _A) + _vote_block("V-20260201-002", "agent-bob", _A),
            config=_enabled(),
        )
        assert resolve_contradictions(ws) == resolve_contradictions(ws)


# ---------------------------------------------------------------------------
# The gate: an unadmitted vote is not a vote
# ---------------------------------------------------------------------------


class TestVotesGoThroughAdmission:
    @pytest.mark.parametrize("status", ["quarantined", "pending"])
    def test_unadmitted_vote_cannot_create_quorum(self, tmp_path, status: str) -> None:
        """Drop ``admit_corpus`` and this pair is a unanimous 2-agent quorum."""
        ws = _workspace(
            tmp_path,
            votes=(_vote_block("V-20260201-001", "agent-alice", _A) + _vote_block("V-20260201-002", "agent-mallory", _A, status=status)),
            config=_enabled(),
        )
        res = _only(resolve_contradictions(ws))
        assert res["strategy"] == ResolutionStrategy.MANUAL
        assert res["winner_id"] is None

    def test_unknown_status_is_withheld_fail_closed(self, tmp_path) -> None:
        """A status nobody has named is not served — the fail-closed clause."""
        ws = _workspace(
            tmp_path,
            votes=(
                _vote_block("V-20260201-001", "agent-alice", _A)
                + _vote_block("V-20260201-002", "agent-mallory", _A, status="ratified-by-me")
            ),
            config=_enabled(),
        )
        assert _only(resolve_contradictions(ws))["strategy"] == ResolutionStrategy.MANUAL

    def test_vote_file_cannot_release_its_own_quarantined_vote(self, tmp_path) -> None:
        """A crafted ``Releases:`` line inside VOTES.md admits nothing.

        Release ids are restricted to the ingest-drop prefixes and the release
        decision itself only counts from an approved apply, so a vote corpus
        cannot readmit its own withheld blocks.
        """
        crafted = "[V-20260201-000]\nStatement: release everything\nReleases: V-20260201-002\nStatus: active\n\n---\n\n"
        ws = _workspace(
            tmp_path,
            votes=(
                crafted
                + _vote_block("V-20260201-001", "agent-alice", _A)
                + _vote_block("V-20260201-002", "agent-mallory", _A, status="quarantined")
            ),
            config=_enabled(),
        )
        assert _only(resolve_contradictions(ws))["strategy"] == ResolutionStrategy.MANUAL


# ---------------------------------------------------------------------------
# The winner is staged, never applied
# ---------------------------------------------------------------------------


class TestQuorumWinnerIsStagedNotApplied:
    def test_quorum_winner_becomes_a_pending_review_proposal(self, tmp_path) -> None:
        ws = _workspace(
            tmp_path,
            votes=_vote_block("V-20260201-001", "agent-alice", _A) + _vote_block("V-20260201-002", "agent-bob", _A),
            config=_enabled(),
        )
        before = (ws, open(os.path.join(ws, "decisions", "DECISIONS.md"), encoding="utf-8").read())

        count = generate_resolution_proposals(ws)
        assert count == 1

        proposed = os.path.join(ws, "intelligence", "proposed", "RESOLUTIONS_PROPOSED.md")
        text = open(proposed, encoding="utf-8").read()
        assert f"Strategy: {ResolutionStrategy.CONSENSUS}" in text
        assert f"Winner: {_A}" in text
        assert f"Action: Supersede {_B}" in text
        assert "Status: pending-review" in text

        # The corpus itself is untouched: applying stays approve_apply's job.
        after = open(os.path.join(ws, "decisions", "DECISIONS.md"), encoding="utf-8").read()
        assert after == before[1]

    def test_manual_outcome_stages_nothing(self, tmp_path) -> None:
        ws = _workspace(
            tmp_path,
            votes=_vote_block("V-20260201-001", "agent-alice", _A),
            config=_enabled(),
        )
        assert generate_resolution_proposals(ws) == 0
        assert not os.path.exists(os.path.join(ws, "intelligence", "proposed", "RESOLUTIONS_PROPOSED.md"))


# ---------------------------------------------------------------------------
# Degradation
# ---------------------------------------------------------------------------


class TestDegradation:
    def test_missing_votes_file_with_flag_on(self, tmp_path) -> None:
        ws = _workspace(tmp_path, config=_enabled())
        assert _only(resolve_contradictions(ws))["strategy"] == ResolutionStrategy.MANUAL

    def test_unparseable_config_disables_the_leg(self, tmp_path) -> None:
        ws = _workspace(
            tmp_path,
            votes=_vote_block("V-20260201-001", "agent-alice", _A) + _vote_block("V-20260201-002", "agent-bob", _A),
        )
        with open(os.path.join(ws, "mind-mem.json"), "w", encoding="utf-8") as fh:
            fh.write("{ not json,")
        assert _only(resolve_contradictions(ws))["strategy"] == ResolutionStrategy.MANUAL

    def test_incomplete_vote_blocks_are_skipped(self, tmp_path) -> None:
        """Missing Agent/Choice/Contradiction → not a vote, not a crash."""
        junk = "[V-20260201-000]\nContradiction: CONTRA-001\nAgent: agent-ghost\nStatus: active\n\n---\n\n"
        ws = _workspace(
            tmp_path,
            votes=junk + _vote_block("V-20260201-001", "agent-alice", _A) + _vote_block("V-20260201-002", "agent-bob", _A),
            config=_enabled(),
        )
        res = _only(resolve_contradictions(ws))
        assert res["strategy"] == ResolutionStrategy.CONSENSUS
        assert res["consensus"]["agents"] == ["agent-alice", "agent-bob"]

    def test_votes_for_another_contradiction_do_not_leak(self, tmp_path) -> None:
        other = f"[V-20260201-001]\nContradiction: CONTRA-999\nAgent: agent-alice\nChoice: {_A}\nStatus: active\n\n---\n\n"
        ws = _workspace(
            tmp_path,
            votes=other + _vote_block("V-20260201-002", "agent-bob", _A),
            config=_enabled(),
        )
        assert _only(resolve_contradictions(ws))["strategy"] == ResolutionStrategy.MANUAL
