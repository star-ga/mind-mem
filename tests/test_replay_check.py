"""Replay — does the ledger corroborate what one recall attestation claims?

Each half of RA's evidence already checks itself: the attestation hashes its
own bound fields, and the served-set ledger's row chain proves no row was
altered or removed. Neither checks the other, and "a recall can be replayed
and shown to have served exactly what it says it served" is a claim about the
JOIN. These tests are about the join and about the three ways it can lie.

R1  **The count check is the one thing neither half can do alone.**
    ``run_id`` binds the answer *digest*, never its cardinality, so an
    attestation can claim ``result_count = 9`` over a two-block answer and
    stay internally consistent forever. The ledger row holds the ids, and the
    chain walk has already proven they hash to the row's digest — so
    ``len(row.ids)`` is an independent witness. Tested with a forged record
    that passes ``verify_recall_attestation`` and fails here.

R2  **A pass is never granted on evidence that failed.** A broken chain means
    no row in the ledger is evidence, even a row that matches perfectly.

R3  **Silence is not refutation.** The ledger records by default (5.0.2) and proves
    nothing about completeness, so "no row" is reported as a missing record
    with the reason named, never as a run that did not happen.

R4  **The same answer served twice is not a mismatch.** ``run_id`` excludes
    the index anchor and the scoring instant by design, so a second serving
    under a different anchor is a legitimate second row.

R5  **Hostile input is answered, not raised on.** A verifier's whole job is
    to be handed malformed values.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import pytest

import mind_mem
from mind_mem.recall_attestation import (
    DERIVATION_ASSERTED,
    LEG_BM25,
    RECALL_ATTEST_TAG,
    build_recall_attestation,
    verify_recall_attestation,
)
from mind_mem.recall_digests import query_hash, served_set_digest
from mind_mem.replay_check import (
    PASSING_VERDICTS,
    VERDICT_ANSWER_RECORDED,
    VERDICT_CHAIN_BROKEN,
    VERDICT_MISMATCH,
    VERDICT_NOT_RECORDED,
    VERDICT_REPLAYED,
    VERDICT_UNVERIFIABLE,
    replay_check,
)
from mind_mem.served_ledger import append_served_run, ledger_path, read_served_runs, verify_served_chain

SRC = pathlib.Path(mind_mem.__file__).parent

CONFIG = "b" * 64
ANCHOR = "c" * 64
OTHER_ANCHOR = "d" * 64
INSTANT = "2026-09-01"
OTHER_INSTANT = "2026-09-02"
QUERY = "why did the retrieval rollout land"
SERVED = ("D-20260901-201", "D-20260901-202")


def _enable_ledger(workspace: str) -> None:
    with open(os.path.join(workspace, "mind-mem.json"), "w", encoding="utf-8") as handle:
        json.dump({"served_ledger": {"enabled": True}}, handle)


def _attest(*, result_count: int | None = None, served=SERVED, anchor: str = ANCHOR, instant: str = INSTANT):
    """A well-formed v2 attestation for the run below.

    ``result_count`` defaults to the truth. Passing a different one mints the
    forgery R1 exists for: internally consistent, and a lie about how much was
    served.
    """
    return build_recall_attestation(
        legs_ran=(LEG_BM25,),
        legs_degraded=(),
        config_hash=CONFIG,
        degraded=None,
        index_anchor=anchor,
        result_count=len(served) if result_count is None else result_count,
        query=QUERY,
        served_ids=tuple(served),
        derivation=DERIVATION_ASSERTED,
        scoring_instant=instant,
    )


def _append(workspace: str, *, served=SERVED, anchor: str = ANCHOR, instant: str = INSTANT):
    row = append_served_run(
        workspace,
        query_hash=query_hash(QUERY),
        served_digest=served_set_digest(tuple(served)),
        ids=tuple(served),
        pipeline_hash=CONFIG,
        index_anchor=anchor,
        scoring_instant=instant,
    )
    assert row is not None, "positive control: the ledger must be enabled or this test means nothing"
    return row


@pytest.fixture
def workspace(tmp_path) -> str:
    ws = str(tmp_path / "ws")
    os.makedirs(ws)
    _enable_ledger(ws)
    return ws


def _cli(workspace: str, payload: str, *argv: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": str(SRC.parent), "MIND_MEM_WORKSPACE": workspace}
    return subprocess.run(
        [sys.executable, "-m", "mind_mem.mm_cli", *argv],
        input=payload,
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )


# ---------------------------------------------------------------------------
# The happy path, and its positive controls
# ---------------------------------------------------------------------------


def test_a_recorded_run_replays(workspace: str) -> None:
    row = _append(workspace)
    attestation = _attest()
    assert attestation.query_id == row.run_id, "positive control: the two halves must share the join key"

    verdict = replay_check(workspace, attestation.to_dict())
    assert verdict.verdict == VERDICT_REPLAYED
    assert verdict.replayable is True
    assert verdict.matched_seqs == (0,)
    assert verdict.exact_seqs == (0,)
    assert verdict.recorded_counts == (2,)
    assert verdict.run_id == row.run_id


def test_replay_accepts_a_whole_envelope_shape(workspace: str) -> None:
    """A caller holds an envelope, not a bare record — both must work."""
    _append(workspace)
    envelope = {"results": [], "attestation": _attest().to_dict()}
    assert replay_check(workspace, envelope["attestation"]).replayable is True


# ---------------------------------------------------------------------------
# R1 — the count check, the one neither half can make alone
# ---------------------------------------------------------------------------


def test_r1_an_overstated_result_count_is_caught_by_the_ledger(workspace: str) -> None:
    """The gate. Nothing outside the ledger can witness ``result_count``."""
    _append(workspace)
    forged = _attest(result_count=9)

    # Positive control, and it is the whole point: this record is a valid,
    # internally consistent attestation. The attestation layer cannot fault it.
    assert verify_recall_attestation(forged.to_dict()) is True
    assert forged.query_id == read_served_runs(workspace)[0].run_id, "it still names the same run"

    verdict = replay_check(workspace, forged.to_dict())
    assert verdict.verdict == VERDICT_MISMATCH
    assert verdict.replayable is False
    assert verdict.attested_count == 9
    assert verdict.recorded_counts == (2,)
    assert any("recorded 2 served ids" in f and "result_count=9" in f for f in verdict.findings), verdict.findings


def test_r1_an_understated_result_count_is_caught_too(workspace: str) -> None:
    """Both directions — a count check that only fires upward is half a check."""
    _append(workspace)
    verdict = replay_check(workspace, _attest(result_count=1).to_dict())
    assert verdict.verdict == VERDICT_MISMATCH
    assert verdict.recorded_counts == (2,)


def test_r1_the_honest_count_passes(workspace: str) -> None:
    """Positive control for both tests above: the check does not fire on truth."""
    _append(workspace)
    assert replay_check(workspace, _attest().to_dict()).verdict == VERDICT_REPLAYED


# ---------------------------------------------------------------------------
# R2 — a pass is never granted on evidence that failed
# ---------------------------------------------------------------------------


def test_r2_a_broken_chain_refuses_even_a_matching_row(workspace: str) -> None:
    _append(workspace)
    _append(workspace, served=("D-20260901-203",), anchor=ANCHOR, instant=INSTANT)
    assert replay_check(workspace, _attest().to_dict()).verdict == VERDICT_REPLAYED, "positive control: clean first"

    path = ledger_path(workspace)
    rows = [json.loads(line) for line in open(path, encoding="utf-8").read().splitlines() if line.strip()]
    rows[1]["scoring_instant"] = "2020-01-01"
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    assert verify_served_chain(workspace).ok is False, "positive control: the edit really did break the chain"

    verdict = replay_check(workspace, _attest().to_dict())
    assert verdict.verdict == VERDICT_CHAIN_BROKEN
    assert verdict.replayable is False
    assert verdict.chain_ok is False


# ---------------------------------------------------------------------------
# R3 — silence is not refutation
# ---------------------------------------------------------------------------


def test_r3_no_ledger_at_all_is_a_missing_record_not_a_refuted_run(tmp_path) -> None:
    ws = str(tmp_path / "bare")
    os.makedirs(ws)
    verdict = replay_check(ws, _attest().to_dict())
    assert verdict.verdict == VERDICT_NOT_RECORDED
    assert verdict.replayable is False
    assert "records by default" in verdict.reason
    assert "not evidence the run did not happen" in verdict.reason


def test_r3_an_enabled_ledger_with_no_matching_row_says_so(workspace: str) -> None:
    _append(workspace, served=("D-20260901-299",))
    verdict = replay_check(workspace, _attest().to_dict())
    assert verdict.verdict == VERDICT_NOT_RECORDED
    assert verdict.rows_examined == 1, "positive control: there IS a row, it just is not this run"
    assert "proves nothing about completeness" in verdict.reason


def test_r3_a_disabled_but_populated_ledger_names_that_specifically(workspace: str) -> None:
    _append(workspace, served=("D-20260901-299",))
    with open(os.path.join(workspace, "mind-mem.json"), "w", encoding="utf-8") as handle:
        json.dump({"served_ledger": {"enabled": False}}, handle)
    verdict = replay_check(workspace, _attest().to_dict())
    assert verdict.verdict == VERDICT_NOT_RECORDED
    assert "opted out of recording" in verdict.reason


# ---------------------------------------------------------------------------
# R4 — the same answer served twice is not a disagreement
# ---------------------------------------------------------------------------


def test_r4_the_same_answer_under_a_different_anchor_is_a_pass(workspace: str) -> None:
    """``run_id`` names THE ANSWER, not one serving of it."""
    row = _append(workspace, anchor=OTHER_ANCHOR, instant=OTHER_INSTANT)
    attestation = _attest(anchor=ANCHOR, instant=INSTANT)
    assert attestation.query_id == row.run_id, "positive control: anchor/instant are excluded from the key"

    verdict = replay_check(workspace, attestation.to_dict())
    assert verdict.verdict == VERDICT_ANSWER_RECORDED
    assert verdict.replayable is True
    assert verdict.matched_seqs == (0,)
    assert verdict.exact_seqs == ()
    assert verdict.verdict in PASSING_VERDICTS


def test_r4_an_exact_serving_outranks_a_merely_equal_answer(workspace: str) -> None:
    _append(workspace, anchor=OTHER_ANCHOR, instant=OTHER_INSTANT)
    _append(workspace, anchor=ANCHOR, instant=INSTANT)
    verdict = replay_check(workspace, _attest().to_dict())
    assert verdict.verdict == VERDICT_REPLAYED
    assert verdict.matched_seqs == (0, 1)
    assert verdict.exact_seqs == (1,)


# ---------------------------------------------------------------------------
# R5 — hostile input is answered, never raised on
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "hostile",
    [
        None,
        "not a mapping",
        [],
        {},
        {"schema": "RECALL_ATTEST_v1"},
        {"schema": RECALL_ATTEST_TAG},
        {"schema": RECALL_ATTEST_TAG, "attestation_hash": "0" * 64},
    ],
)
def test_r5_malformed_input_is_unverifiable_not_an_exception(workspace: str, hostile) -> None:
    _append(workspace)
    verdict = replay_check(workspace, hostile)
    assert verdict.verdict == VERDICT_UNVERIFIABLE
    assert verdict.replayable is False


def test_r5_a_tampered_field_makes_the_record_unverifiable(workspace: str) -> None:
    _append(workspace)
    record = _attest().to_dict()
    assert replay_check(workspace, dict(record)).replayable is True, "positive control: untouched, it replays"
    record["index_anchor"] = OTHER_ANCHOR
    verdict = replay_check(workspace, record)
    assert verdict.verdict == VERDICT_UNVERIFIABLE
    assert verdict.record_consistent is False
    assert "does not hash to its own bound fields" in verdict.reason


# ---------------------------------------------------------------------------
# Rails — writes nothing, names no id, exits with its verdict
# ---------------------------------------------------------------------------


def test_replay_check_creates_nothing(tmp_path) -> None:
    ws = str(tmp_path / "bare")
    os.makedirs(ws)
    before = sorted(os.listdir(ws))
    replay_check(ws, _attest().to_dict())
    assert sorted(os.listdir(ws)) == before == []


def test_the_verdict_names_no_block_id(workspace: str) -> None:
    """The ids on a row were admitted when served and may not be admitted now."""
    row = _append(workspace)
    assert row.ids == SERVED, "positive control: the ids really are on the row"
    payload = json.dumps(replay_check(workspace, _attest().to_dict()).to_dict())
    for block_id in SERVED:
        assert block_id not in payload, f"the replay verdict published {block_id}"


def test_the_cli_exit_code_carries_the_verdict(workspace: str) -> None:
    """A verb whose failure is only visible in its JSON is a verb no gate can call."""
    _append(workspace)
    good = _cli(workspace, json.dumps(_attest().to_dict()), "replay-check", "--attestation", "-")
    assert good.returncode == 0, good.stderr
    assert json.loads(good.stdout)["verdict"] == VERDICT_REPLAYED

    forged = _cli(workspace, json.dumps(_attest(result_count=9).to_dict()), "replay-check", "--attestation", "-")
    assert forged.returncode == 1
    assert json.loads(forged.stdout)["verdict"] == VERDICT_MISMATCH


def test_the_cli_accepts_a_whole_envelope_on_stdin(workspace: str) -> None:
    _append(workspace)
    envelope = json.dumps({"results": [], "attestation": _attest().to_dict()})
    result = _cli(workspace, envelope, "replay-check", "--attestation", "-")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["verdict"] == VERDICT_REPLAYED


def test_unreadable_input_is_unverifiable_and_exits_nonzero(workspace: str) -> None:
    result = _cli(workspace, "{not json", "replay-check", "--attestation", "-")
    assert result.returncode == 1
    assert json.loads(result.stdout)["verdict"] == VERDICT_UNVERIFIABLE


def test_the_cli_accepts_an_attestation_file(workspace: str, tmp_path) -> None:
    """The file form the docs give as the first example, gated rather than asserted."""
    _append(workspace)
    path = tmp_path / "envelope.json"
    path.write_text(json.dumps({"results": [], "attestation": _attest().to_dict()}), encoding="utf-8")
    result = _cli(workspace, "", "replay-check", "--attestation", str(path))
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["verdict"] == VERDICT_REPLAYED


def test_a_missing_attestation_file_is_unverifiable_not_a_traceback(workspace: str, tmp_path) -> None:
    result = _cli(workspace, "", "replay-check", "--attestation", str(tmp_path / "nope.json"))
    assert result.returncode == 1
    assert json.loads(result.stdout)["verdict"] == VERDICT_UNVERIFIABLE
    assert "Traceback" not in result.stderr


def test_both_documented_verbs_are_registered_on_the_cli(workspace: str) -> None:
    """``docs/configuration.md`` names two verbs; a doc naming a verb that does
    not exist is the false green this repo keeps paying for."""
    help_text = _cli(workspace, "", "--help").stdout
    assert "dashboard" in help_text
    assert "replay-check" in help_text


def test_an_unreadable_ledger_row_is_chain_broken_not_a_traceback(workspace: str) -> None:
    """``read_served_runs`` raises on a malformed row; the chain check runs first."""
    _append(workspace)
    assert replay_check(workspace, _attest().to_dict()).replayable is True, "positive control: readable first"
    with open(ledger_path(workspace), "a", encoding="utf-8") as handle:
        handle.write("{not json\n")
    verdict = replay_check(workspace, _attest().to_dict())
    assert verdict.verdict == VERDICT_CHAIN_BROKEN
    assert verdict.replayable is False
    assert "unreadable row" in verdict.reason
