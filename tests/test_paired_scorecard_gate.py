# Copyright 2026 STARGA, Inc.
"""The ranking gate: every other ranking fix has to pass through this.

Two halves, one purpose -- make "it did not move ranking" a measurement.

1. **Artifact level.** The paired scorecard over two per-question NDJSON runs.
   Its numbers are pinned against the committed 2026-09-03 LongMemEval-S
   artifacts, which an independent hand-computation produced first: any@5
   4/18 discordant at p=0.0043, all@5 22/28 at p=0.4799, MRR 24/53 at p=0.0013.
   If this file and that hand-computation ever disagree, the code is wrong --
   the expected values are not the thing to adjust.
2. **Call level.** :mod:`benchmarks.ranking_identity` over live ``recall``:
   the served ``(id, score)`` list, byte for byte, across a query battery.

Every assertion here is paired with a case that makes it go RED -- a moved
run for the no-diff check, a real ranking knob for the byte-identity check, a
different seed for the bootstrap. A gate whose green cannot be turned red is
not a gate, and this project has shipped one silent recall regression already.
"""

from __future__ import annotations

import json
import os
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

from benchmarks.compare_runs import main as compare_runs_main
from benchmarks.paired_scorecard import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    PINNED_BASELINE,
    PairingError,
    bootstrap_mean_diff_ci,
    build_scorecard,
    load_run,
    pair_runs,
)
from benchmarks.ranking_identity import (
    RankingMoved,
    VacuousComparison,
    assert_battery_unchanged,
    assert_ranking_unchanged,
    compare_rankings,
    fingerprint_battery,
    fingerprint_digest,
    ranking_fingerprint,
)

_ARTIFACTS = Path(__file__).resolve().parents[1] / "docs" / "benchmarks"
_MIND_MEM_REP1 = _ARTIFACTS / "2026-09-03-longmemeval-s-full-mind_mem-rep1.ndjson"
_MIND_MEM_REP2 = _ARTIFACTS / "2026-09-03-longmemeval-s-full-mind_mem-rep2.ndjson"
_BM25_REP1 = _ARTIFACTS / "2026-09-03-longmemeval-s-full-bm25_baseline-rep1.ndjson"

#: Small enough to keep the pairing/verdict tests quick. The bootstrap count
#: does not touch a split, a p-value or a verdict -- only the interval -- so
#: the CI itself gets its own test at the committed default instead.
_FAST_RESAMPLES = 200


# ---------------------------------------------------------------------------
# 1. The pinned baseline and the reference numbers
# ---------------------------------------------------------------------------


def test_the_pinned_baseline_is_the_committed_2026_09_03_rep1() -> None:
    """The default before-artifact exists, is the named one, and is 470 rows.

    A default baseline pointing at a path that does not exist would make every
    ``paired`` invocation fail loudly, which is survivable. One silently
    pointing at a *different* file would make every comparison answer a
    question nobody asked, which is not.
    """
    assert PINNED_BASELINE.name == "2026-09-03-longmemeval-s-full-mind_mem-rep1.ndjson"
    assert PINNED_BASELINE.is_file()
    assert len(load_run(PINNED_BASELINE)) == 470


@pytest.fixture(scope="module")
def published() -> Any:
    """The published comparison: pinned baseline (mind_mem) vs the BM25 arm."""
    return build_scorecard(_MIND_MEM_REP1, _BM25_REP1, k=5, resamples=_FAST_RESAMPLES)


def _by_label(card: Any, label: str) -> Any:
    for comparison in card.comparisons:
        if comparison.label == label:
            return comparison
    raise AssertionError(f"no comparison labelled {label!r} in {[c.label for c in card.comparisons]}")


def test_pairs_all_470_questions(published: Any) -> None:
    assert published.n_pairs == 470
    assert published.dropped_non_ok == ()


def test_reproduces_the_published_recall_any_split(published: Any) -> None:
    """4 mind_mem-only / 18 bm25-only, 22 discordant, p=0.0043."""
    c = _by_label(published, "recall_any_at_k@5")
    assert (c.baseline_only, c.candidate_only, c.n_discordant) == (4, 18, 22)
    assert round(float(c.p_value), 4) == 0.0043
    assert c.verdict == "candidate_better"


def test_reproduces_the_published_recall_all_split(published: Any) -> None:
    """22 mind_mem-only / 28 bm25-only, 50 discordant, p=0.4799."""
    c = _by_label(published, "recall_all_at_k@5")
    assert (c.baseline_only, c.candidate_only, c.n_discordant) == (22, 28, 50)
    assert round(float(c.p_value), 4) == 0.4799
    assert c.verdict == "not_significant"


def test_reproduces_the_published_mrr_sign_test(published: Any) -> None:
    """24 mind_mem-better / 53 bm25-better / 393 tied, p=0.0013."""
    c = _by_label(published, "reciprocal_rank")
    assert (c.baseline_only, c.candidate_only, c.concordant) == (24, 53, 393)
    assert c.n_discordant == 77
    assert round(float(c.p_value), 4) == 0.0013
    assert c.verdict == "candidate_better"


def test_reproduces_the_published_means(published: Any) -> None:
    """The headline means the paired tests sit underneath."""
    assert round(_by_label(published, "recall_any_at_k@5").baseline_mean, 4) == 0.9404
    assert round(_by_label(published, "recall_any_at_k@5").candidate_mean, 4) == 0.9702
    assert round(_by_label(published, "recall_all_at_k@5").baseline_mean, 4) == 0.8170
    assert round(_by_label(published, "recall_all_at_k@5").candidate_mean, 4) == 0.8298
    assert round(_by_label(published, "reciprocal_rank").baseline_mean, 4) == 0.8776
    assert round(_by_label(published, "reciprocal_rank").candidate_mean, 4) == 0.9081


def test_both_discordant_directions_are_reported_never_a_net(published: Any) -> None:
    """A net would erase the difference between 4-vs-18 and 11-vs-11.

    Both are 22 discordant; one is significant and one could not be. The
    serialised form must carry both directions so a reader cannot be handed
    the ambiguous number.
    """
    row = _by_label(published, "recall_any_at_k@5").as_dict()
    assert row["baseline_only"] == 4
    assert row["candidate_only"] == 18
    assert row["n_discordant"] == 22
    assert "net" not in json.dumps(row)


# ---------------------------------------------------------------------------
# 2. The bootstrap is seeded, reproducible, and actually uses its seed
# ---------------------------------------------------------------------------


def _mrr_diffs() -> list[float]:
    runs = pair_runs(load_run(_MIND_MEM_REP1), load_run(_BM25_REP1))
    return [runs.candidate[q]["reciprocal_rank"] - runs.baseline[q]["reciprocal_rank"] for q in runs.question_ids]


def test_bootstrap_ci_is_pinned_at_the_committed_seed() -> None:
    """The published MRR interval, at the committed seed and resample count.

    Pinned rather than merely "reproducible": an interval that reproduces
    only against itself would still drift silently if the draw sequence
    changed. These bounds were verified identical on CPython 3.12 and 3.14.
    """
    ci = bootstrap_mean_diff_ci(_mrr_diffs(), seed=BOOTSTRAP_SEED, resamples=BOOTSTRAP_RESAMPLES)
    assert round(ci.mean_diff, 6) == 0.030498
    assert round(ci.low, 6) == 0.012418
    assert round(ci.high, 6) == 0.048749
    assert ci.seed == BOOTSTRAP_SEED == 20260903
    assert ci.resamples == BOOTSTRAP_RESAMPLES == 10_000


def test_bootstrap_repeats_exactly_and_changes_with_the_seed() -> None:
    """Same seed, same interval; different seed, different interval.

    The second half is the mutation: if the seed were ignored -- hard-coded,
    or shadowed by a module-level RNG -- the first assertion would still pass
    and mean nothing.
    """
    diffs = _mrr_diffs()
    first = bootstrap_mean_diff_ci(diffs, seed=BOOTSTRAP_SEED, resamples=_FAST_RESAMPLES)
    again = bootstrap_mean_diff_ci(diffs, seed=BOOTSTRAP_SEED, resamples=_FAST_RESAMPLES)
    other = bootstrap_mean_diff_ci(diffs, seed=BOOTSTRAP_SEED + 1, resamples=_FAST_RESAMPLES)
    assert (first.low, first.high) == (again.low, again.high)
    assert (first.low, first.high) != (other.low, other.high)


def test_the_seed_travels_in_the_serialised_scorecard(published: Any) -> None:
    """An unrecorded seed makes an interval decoration, not a measurement."""
    for row in published.as_dict()["comparisons"]:
        assert row["bootstrap_seed"] == BOOTSTRAP_SEED
        assert row["bootstrap_resamples"] == _FAST_RESAMPLES
        assert row["confidence"] == 0.95


def test_bootstrap_ci_brackets_the_mean_and_flags_zero() -> None:
    diffs = _mrr_diffs()
    ci = bootstrap_mean_diff_ci(diffs, seed=BOOTSTRAP_SEED, resamples=_FAST_RESAMPLES)
    assert ci.low <= ci.mean_diff <= ci.high
    assert ci.as_dict()["ci_excludes_zero"] is True
    flat = bootstrap_mean_diff_ci([0.0] * 20, seed=BOOTSTRAP_SEED, resamples=_FAST_RESAMPLES)
    assert flat.as_dict()["ci_excludes_zero"] is False


def test_bootstrap_refuses_a_degenerate_request() -> None:
    with pytest.raises(ValueError):
        bootstrap_mean_diff_ci([], seed=1, resamples=10)
    with pytest.raises(ValueError):
        bootstrap_mean_diff_ci([0.1], seed=1, resamples=0)
    with pytest.raises(ValueError):
        bootstrap_mean_diff_ci([0.1], seed=1, resamples=10, confidence=1.0)


# ---------------------------------------------------------------------------
# 3. Pairing refuses everything it cannot honestly compare
# ---------------------------------------------------------------------------


def _row(qid: str, *, any5: int = 1, all5: int = 1, rr: float = 1.0, status: str = "ok") -> dict[str, Any]:
    return {
        "question_id": qid,
        "unit_status": status,
        "recall_any_at_k": {"1": any5, "3": any5, "5": any5, "10": any5},
        "recall_all_at_k": {"1": all5, "3": all5, "5": all5, "10": all5},
        "reciprocal_rank": rr,
    }


def build_scorecard_from_rows(baseline: list[dict], candidate: list[dict], **kwargs: Any) -> Any:
    """Run the scorecard over in-memory rows, via the same public path."""
    from benchmarks.paired_scorecard import compare_binary

    runs = pair_runs(baseline, candidate)
    return compare_binary(runs, "recall_any_at_k", 5, resamples=8, **kwargs)


def test_a_duplicate_question_id_is_refused() -> None:
    with pytest.raises(PairingError, match="more than once"):
        pair_runs([_row("q1"), _row("q1")], [_row("q1"), _row("q2")])


def test_different_question_sets_are_refused_not_intersected() -> None:
    """Intersecting would silently answer a question about a lucky subset."""
    with pytest.raises(PairingError, match="different questions"):
        pair_runs([_row("q1"), _row("q2")], [_row("q1"), _row("q3")])


def test_an_incomplete_unit_is_refused_unless_dropping_is_asked_for() -> None:
    baseline = [_row("q1"), _row("q2", status="error")]
    candidate = [_row("q1"), _row("q2")]
    with pytest.raises(PairingError, match="did not complete"):
        pair_runs(baseline, candidate)
    runs = pair_runs(baseline, candidate, drop_non_ok=True)
    assert runs.question_ids == ("q1",)
    assert runs.dropped_non_ok == ("q2",)


def test_dropping_every_question_is_refused() -> None:
    with pytest.raises(PairingError, match="no completed question"):
        pair_runs([_row("q1", status="error")], [_row("q1")], drop_non_ok=True)


def test_a_missing_question_id_is_refused() -> None:
    with pytest.raises(PairingError, match="no usable question_id"):
        pair_runs([{"unit_status": "ok"}], [_row("q1")])


def test_a_non_binary_recall_value_is_refused() -> None:
    """0.5 is not an outcome McNemar can be run on; it must not be rounded."""
    bad = _row("q1")
    bad["recall_any_at_k"]["5"] = 0.5
    with pytest.raises(PairingError, match="not a 0/1 outcome"):
        build_scorecard_from_rows([bad], [_row("q1")])


def test_a_missing_k_is_refused() -> None:
    bad = _row("q1")
    del bad["recall_any_at_k"]["5"]
    with pytest.raises(PairingError, match="has no k=5"):
        build_scorecard_from_rows([bad], [_row("q1")])


def test_an_empty_or_malformed_artifact_is_refused(tmp_path: Path) -> None:
    empty = tmp_path / "empty.ndjson"
    empty.write_text("\n\n", encoding="utf-8")
    with pytest.raises(PairingError, match="no rows"):
        load_run(empty)
    broken = tmp_path / "broken.ndjson"
    broken.write_text('{"question_id": "q1"}\nnot json\n', encoding="utf-8")
    with pytest.raises(PairingError, match="not valid JSON"):
        load_run(broken)
    scalar = tmp_path / "scalar.ndjson"
    scalar.write_text("42\n", encoding="utf-8")
    with pytest.raises(PairingError, match="expected a JSON object"):
        load_run(scalar)


# ---------------------------------------------------------------------------
# 4. Verdicts: an underpowered split is never headlined as a result
# ---------------------------------------------------------------------------


def test_a_one_question_difference_is_named_underpowered() -> None:
    """Below the floor, no split can reach alpha -- so it is not evidence."""
    baseline = [_row(f"q{i}", any5=1) for i in range(20)]
    candidate = [_row(f"q{i}", any5=1) for i in range(20)]
    candidate[0]["recall_any_at_k"]["5"] = 0
    comparison = build_scorecard_from_rows(baseline, candidate)
    assert comparison.n_discordant == 1
    assert comparison.verdict == "underpowered"
    assert comparison.min_discordant_for_significance == 6


def test_a_perfectly_agreeing_pair_is_named_no_evidence() -> None:
    rows = [_row(f"q{i}") for i in range(20)]
    comparison = build_scorecard_from_rows(rows, [dict(r) for r in rows])
    assert comparison.n_discordant == 0
    assert comparison.verdict == "no_evidence"


def test_an_even_split_and_a_lopsided_split_are_told_apart() -> None:
    """Both are 22 discordant. Only one of them is a result."""
    even = build_scorecard_from_rows(*_split(11, 11))
    lopsided = build_scorecard_from_rows(*_split(0, 22))
    assert even.n_discordant == lopsided.n_discordant == 22
    assert even.verdict == "not_significant"
    assert lopsided.verdict == "candidate_better"
    assert float(even.p_value) > float(lopsided.p_value)


def _split(baseline_only: int, candidate_only: int, total: int = 60) -> tuple[list[dict], list[dict]]:
    baseline = [_row(f"q{i}", any5=1) for i in range(total)]
    candidate = [_row(f"q{i}", any5=1) for i in range(total)]
    for i in range(baseline_only):
        candidate[i]["recall_any_at_k"]["5"] = 0
    for i in range(baseline_only, baseline_only + candidate_only):
        baseline[i]["recall_any_at_k"]["5"] = 0
    return baseline, candidate


def test_alpha_is_reported_alongside_every_p_value(published: Any) -> None:
    for comparison in published.comparisons:
        assert comparison.alpha == Fraction(1, 20)
        assert comparison.as_dict()["alpha"] == 0.05


# ---------------------------------------------------------------------------
# 5. The no-diff assertion, at run-artifact level -- and its positive control
# ---------------------------------------------------------------------------


def test_two_reps_of_one_arm_are_identical() -> None:
    """The bit-identical property the committed reps were run to demonstrate."""
    card = build_scorecard(_MIND_MEM_REP1, _MIND_MEM_REP2, k=5, resamples=_FAST_RESAMPLES)
    assert card.identical is True
    assert card.moved() == ()
    assert all(c.n_discordant == 0 for c in card.comparisons)


def test_the_no_diff_assertion_goes_red_on_a_run_that_moved(published: Any) -> None:
    """The positive control. Without it, ``identical`` could be a constant."""
    assert published.identical is False
    assert {c.label for c in published.moved()} == {"recall_any_at_k@5", "recall_all_at_k@5", "reciprocal_rank"}


def test_identical_is_not_satisfied_by_a_net_of_zero() -> None:
    """11 up and 11 down nets to zero and is 22 questions that moved."""
    comparison = build_scorecard_from_rows(*_split(11, 11))
    assert comparison.candidate_only == comparison.baseline_only == 11
    assert comparison.n_discordant == 22


def test_the_cli_exits_zero_on_identical_and_one_on_moved(tmp_path: Path) -> None:
    """The gate as CI would invoke it, both ways round."""
    out = tmp_path / "card.json"
    identical = compare_runs_main(
        ["paired", str(_MIND_MEM_REP2), "--baseline", str(_MIND_MEM_REP1), "--require-identical", "--resamples", "8", "--json", str(out)]
    )
    assert identical == 0
    assert json.loads(out.read_text(encoding="utf-8"))["identical"] is True
    moved = compare_runs_main(["paired", str(_BM25_REP1), "--baseline", str(_MIND_MEM_REP1), "--require-identical", "--resamples", "8"])
    assert moved == 1


def test_the_cli_defaults_its_baseline_to_the_pinned_artifact(tmp_path: Path) -> None:
    out = tmp_path / "card.json"
    assert compare_runs_main(["paired", str(_MIND_MEM_REP2), "--resamples", "8", "--json", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["baseline_path"] == str(PINNED_BASELINE)


def test_the_cli_names_the_artifact_it_could_not_read(tmp_path: Path, capsys: Any) -> None:
    """A mistyped path is the likeliest wrong invocation; name which one."""
    missing = tmp_path / "not-there.ndjson"
    assert compare_runs_main(["paired", str(missing), "--resamples", "8"]) == 2
    assert str(missing) in capsys.readouterr().err


def test_the_cli_reports_an_unpairable_comparison_instead_of_a_number(tmp_path: Path) -> None:
    lonely = tmp_path / "one.ndjson"
    lonely.write_text(json.dumps(_row("q1")) + "\n", encoding="utf-8")
    assert compare_runs_main(["paired", str(lonely), "--baseline", str(_MIND_MEM_REP1), "--resamples", "8"]) == 2


def test_the_original_locomo_mode_still_answers_two_bare_paths(tmp_path: Path) -> None:
    """Backward compatibility: the documented two-positional-path invocation."""
    for name in ("a.json", "b.json"):
        (tmp_path / name).write_text(json.dumps({"per_question": [{"judge_score": 80, "category": "x"}]}), encoding="utf-8")
    assert compare_runs_main([str(tmp_path / "a.json"), str(tmp_path / "b.json")]) == 0


# ---------------------------------------------------------------------------
# 6. The byte-identity gate, at served (id, score) level
# ---------------------------------------------------------------------------


def test_a_fingerprint_commits_to_order_not_just_membership() -> None:
    a = ranking_fingerprint([{"_id": "A", "score": 2.0}, {"_id": "B", "score": 1.0}])
    b = ranking_fingerprint([{"_id": "B", "score": 1.0}, {"_id": "A", "score": 2.0}])
    assert a != b
    assert fingerprint_digest(a) != fingerprint_digest(b)


def test_a_one_ulp_score_change_is_a_difference() -> None:
    """ "Byte-identical" that tolerated rounding would not be byte-identical."""
    import math

    base = 12.5
    nudged = math.nextafter(base, math.inf)
    assert round(base, 12) == round(nudged, 12)
    a = ranking_fingerprint([{"_id": "A", "score": base}])
    b = ranking_fingerprint([{"_id": "A", "score": nudged}])
    assert a != b
    assert compare_rankings(a, b).moved is True


def test_signed_zero_is_not_erased() -> None:
    a = ranking_fingerprint([{"_id": "A", "score": 0.0}])
    b = ranking_fingerprint([{"_id": "A", "score": -0.0}])
    assert 0.0 == -0.0
    assert a != b


def test_a_missing_field_raises_instead_of_emptying_every_row() -> None:
    """The rename trap: ``.get(_id, "")`` would make every comparison pass."""
    with pytest.raises(KeyError, match="_id"):
        ranking_fingerprint([{"id": "A", "score": 1.0}])
    with pytest.raises(KeyError, match="score"):
        ranking_fingerprint([{"_id": "A", "relevance": 1.0}])


def test_a_non_numeric_score_raises() -> None:
    with pytest.raises(TypeError, match="bool"):
        ranking_fingerprint([{"_id": "A", "score": True}])
    with pytest.raises(TypeError, match="str"):
        ranking_fingerprint([{"_id": "A", "score": "1.0"}])


def test_the_digest_framing_cannot_be_forged_by_concatenation() -> None:
    ab_c = fingerprint_digest((("AB", "i:1"), ("C", "i:2")))
    a_bc = fingerprint_digest((("A", "i:1"), ("BC", "i:2")))
    assert ab_c != a_bc


def test_an_empty_comparison_is_refused_not_passed() -> None:
    """``() == ()`` is the most common way a gate proves nothing."""
    with pytest.raises(VacuousComparison, match="able to fail"):
        assert_ranking_unchanged((), ())


def test_a_battery_with_no_queries_is_refused() -> None:
    with pytest.raises(VacuousComparison, match="no queries"):
        fingerprint_battery(lambda q: [], [])


def test_a_battery_that_changed_its_questions_is_refused() -> None:
    """A query that stopped being asked must not read as a pass."""
    before = {"a": (("X", "i:1"),)}
    after = {"b": (("X", "i:1"),)}
    with pytest.raises(VacuousComparison, match="different questions"):
        assert_battery_unchanged(before, after)


def test_a_moved_ranking_names_where_it_moved() -> None:
    before = ranking_fingerprint([{"_id": "A", "score": 2.0}, {"_id": "B", "score": 1.0}])
    after = ranking_fingerprint([{"_id": "A", "score": 2.0}, {"_id": "C", "score": 1.0}])
    with pytest.raises(RankingMoved, match="rank 2"):
        assert_ranking_unchanged(before, after)
    assert compare_rankings(before, after).first_divergence == 1


def test_a_truncated_ranking_is_a_difference() -> None:
    before = ranking_fingerprint([{"_id": "A", "score": 2.0}, {"_id": "B", "score": 1.0}])
    after = ranking_fingerprint([{"_id": "A", "score": 2.0}])
    diff = compare_rankings(before, after)
    assert diff.moved is True
    assert (diff.n_before, diff.n_after) == (2, 1)
    assert diff.first_divergence == 1


# ---------------------------------------------------------------------------
# 7. The byte-identity gate over LIVE recall -- and its positive control
# ---------------------------------------------------------------------------

_QUERIES = (
    "deterministic compiler",
    "evidence chain",
    "recall index block 7",
    "ranking gate",
)


@pytest.fixture
def workspace(tmp_path: Path) -> str:
    from mind_mem.init_workspace import init

    ws = str(tmp_path / "ws")
    os.makedirs(ws, exist_ok=True)
    init(ws)
    decisions = os.path.join(ws, "decisions")
    os.makedirs(decisions, exist_ok=True)
    body = ["# DECISIONS\n\n---\n"]
    for i in range(1, 41):
        body.append(
            f"\n[D-20260101-{i:06d}]\n"
            f"Date: 2026-06-20\n"
            f"Status: active\n"
            f"Scope: global\n"
            f"Statement: deterministic compiler evidence chain recall index block {i} ranking gate\n"
            f"Tags: compiler, recall\n"
        )
    with open(os.path.join(decisions, "DECISIONS.md"), "w", encoding="utf-8") as handle:
        handle.write("".join(body))
    return ws


def _battery(workspace: str, *, limit: int = 10, **kwargs: Any) -> dict[str, Any]:
    from mind_mem.recall import recall

    return fingerprint_battery(lambda query: recall(workspace, query, limit=limit, **kwargs), _QUERIES)


def test_the_battery_actually_retrieves_something(workspace: str) -> None:
    """The positive control for the gate itself.

    If recall returned nothing, every identity assertion below would pass over
    four empty lists. ``min_results`` guards that at assert time; this pins it
    at probe time too, so a failure says "retrieval broke", not "ranking moved".
    """
    battery = _battery(workspace)
    assert set(battery) == set(_QUERIES)
    for query, fingerprint in battery.items():
        assert len(fingerprint) == 10, f"{query!r} served {len(fingerprint)} hit(s)"


def test_an_unchanged_path_serves_a_byte_identical_ranking(workspace: str) -> None:
    """The reusable form of "latency only, no ranking movement"."""
    before = _battery(workspace)
    after = _battery(workspace)
    diffs = assert_battery_unchanged(before, after, label="unchanged path")
    assert all(not diff.moved for diff in diffs.values())
    assert all(diff.n_before == 10 for diff in diffs.values())


def test_the_live_gate_goes_red_on_a_real_ranking_change(workspace: str) -> None:
    """Mutation: turn a ranking stage off and the gate must see it.

    ``rerank=False`` is a genuine ranking change on the same corpus, same
    query, same k. If this passed, the assertion above would be measuring
    nothing but its own repetition.
    """
    before = _battery(workspace)
    after = _battery(workspace, rerank=False)
    with pytest.raises(RankingMoved, match="quer"):
        assert_battery_unchanged(before, after, label="rerank off")


def test_the_gate_names_every_query_that_moved_not_just_the_first(workspace: str) -> None:
    before = _battery(workspace)
    after = _battery(workspace, rerank=False)
    with pytest.raises(RankingMoved) as excinfo:
        assert_battery_unchanged(before, after, label="rerank off")
    message = str(excinfo.value)
    assert f"of {len(_QUERIES)} quer" in message
    assert sum(1 for query in _QUERIES if repr(query) in message) >= 2


def test_a_narrower_k_is_detected_as_movement(workspace: str) -> None:
    """A second, independent knob -- so the red is not specific to reranking."""
    before = _battery(workspace)
    after = _battery(workspace, limit=5)
    with pytest.raises(RankingMoved):
        assert_battery_unchanged(before, after, label="limit 5")
