"""The full-run driver must score the same pool the canonical harness does.

``benchmarks/longmemeval_full_run.py`` re-implements the *loop* so each
question can run in a killable child process. It deliberately does NOT
re-implement anything that decides a number -- the eligibility filter is the
one place a duplicate could drift, and a drifting filter is a drifting
denominator, which is exactly how a published recall figure stops meaning what
its N says it means.

So the filter is pinned here against the canonical harness by running both
over the same synthetic dataset and comparing the question ids scored.
"""

from __future__ import annotations

from typing import Any

import pytest

from benchmarks.longmemeval_full_run import _load_done, eligible_questions, run
from mind_mem.bench.longmemeval_suite import run_suite


def _question(qid: str, gold: list[str] | None, n_sessions: int = 3) -> dict[str, Any]:
    return {
        "question_id": qid,
        "question": f"where did we decide {qid}",
        "question_type": "single-session-user",
        "answer_session_ids": gold if gold is not None else [],
        "haystack_session_ids": [f"s{i}" for i in range(n_sessions)],
        "haystack_sessions": [[{"role": "user", "content": f"session {i} about {qid}"}] for i in range(n_sessions)],
    }


@pytest.fixture
def dataset() -> list[dict[str, Any]]:
    """A pool exercising every exclusion the canonical filter applies."""
    return [
        _question("q-normal-1", ["s0"]),
        _question("q-abs", ["s1"]),  # not excluded by id...
        _question("q-real_abs", ["s1"]),  # ...this one is: id ends with _abs
        _question("q-no-gold", []),
        _question("q-null-gold", None),
        _question("q-normal-2", ["s2"]),
    ]


def test_eligibility_matches_the_canonical_harness(dataset: list[dict[str, Any]]) -> None:
    """Same pool, same exclusions, same counts -- proven by running both."""
    pool, excl_abs, excl_no_gold = eligible_questions(dataset)

    canonical = run_suite("bm25_baseline", dataset, k=5, turns="all")

    assert [q["question_id"] for q in pool] == [s.question_id for s in canonical.scores]
    assert len(pool) == canonical.eligible
    assert excl_abs == canonical.excluded_abstention
    assert excl_no_gold == canonical.excluded_no_gold


def test_the_filter_actually_excludes_something(dataset: list[dict[str, Any]]) -> None:
    """Positive control: an all-pass filter would satisfy the test above too."""
    pool, excl_abs, excl_no_gold = eligible_questions(dataset)
    assert excl_abs == 1, "the `_abs` exclusion never fired; the parity test is vacuous"
    assert excl_no_gold == 2, "the no-gold exclusion never fired; the parity test is vacuous"
    assert len(pool) == 3
    assert len(pool) < len(dataset)


def test_run_is_resumable_and_appends(tmp_path: Any, dataset: list[dict[str, Any]]) -> None:
    """A second pass scores nothing already present, and does not duplicate."""
    out = str(tmp_path / "run.ndjson")

    first = run("bm25_baseline", dataset, out, k=5, qtimeout_s=120.0, progress_every=0)
    assert first["attempted_this_pass"] == 3
    assert len(_load_done(out)) == 3

    second = run("bm25_baseline", dataset, out, k=5, qtimeout_s=120.0, progress_every=0)
    assert second["attempted_this_pass"] == 0, "a resumed run re-scored questions it had already scored"
    assert len(_load_done(out)) == 3

    with open(out, encoding="utf-8") as handle:
        rows = [line for line in handle if line.strip()]
    assert len(rows) == 3


def test_every_row_records_the_unit_status(tmp_path: Any, dataset: list[dict[str, Any]]) -> None:
    """A row with no status could not be told apart from a killed unit."""
    import json

    out = str(tmp_path / "status.ndjson")
    run("bm25_baseline", dataset, out, k=5, qtimeout_s=120.0, progress_every=0)
    with open(out, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    assert rows
    for row in rows:
        assert "unit_status" in row
        assert "unit_elapsed_s" in row
        assert "pipeline" in row


def test_a_killed_unit_is_scored_as_a_miss_and_labelled(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """The timeout policy, executed: a hung question is a miss, not a gap.

    Drives the real hard-kill path by pointing the per-question worker at a
    genuine native hang, so this measures the policy rather than restating it.
    """
    import json

    import benchmarks.longmemeval_full_run as driver
    from benchmarks.hard_timeout import TIMEOUT, native_sqlite_hang

    monkeypatch.setattr(driver, "score_one_question", native_sqlite_hang)
    out = str(tmp_path / "killed.ndjson")
    driver.run("bm25_baseline", [_question("q-hangs", ["s0"])], out, k=5, qtimeout_s=2.0, progress_every=0)

    with open(out, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    assert len(rows) == 1
    row = rows[0]
    assert row["unit_status"] == TIMEOUT, row
    assert row["n_retrieved"] == 0
    assert row["hit"] is False
    assert row["recall_any_at_k"]["5"] == 0
