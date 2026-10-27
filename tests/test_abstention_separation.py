# Copyright 2026 STARGA, Inc.
"""The falsification condition, executable.

The roadmap entry that proposed computed abstention carried its own kill
switch: *run recall over a labelled set where the right answer is known to be
absent; if the derived statistic does not separate those queries from ones
where the answer is present, there is no signal here and no amount of
threshold tuning will create one.*

:mod:`mind_mem.abstention_eval` is that experiment as a function an operator
can point at their own workspace (``mm abstention-eval``). These tests run it
on a deterministic labelled fixture -- forty present and forty absent queries
over an FTS-indexed corpus whose absent queries match many blocks weakly --
and, just as importantly, prove the experiment CAN come out negative: a
measurement whose verdict is always "separates" is a rubber stamp.

The fixture is evidence that the statistic carries signal on a ranker that
finds the right block. It is not evidence about any real workspace, and the
shipped thresholds are provisional for exactly that reason.
"""

from __future__ import annotations

import json
import math
import random
from typing import Any

import pytest

from mind_mem.abstention_eval import auc, evaluate, load_labelled
from tests._abstention_fixture import make_workspace, preload_serving_stack, queries

preload_serving_stack()  # before any capsys test can bind a logger to a per-test stream


@pytest.fixture(scope="module")
def labelled() -> list[dict[str, Any]]:
    present, absent = queries()
    return [{"query": q, "present": True} for q in present] + [{"query": q, "present": False} for q in absent]


@pytest.fixture(scope="module")
def indexed_ws(tmp_path_factory: pytest.TempPathFactory) -> str:
    # recall.backend=sqlite so the library door serves from the FTS5 index --
    # the product's normal operating mode once a workspace has been indexed.
    return make_workspace(str(tmp_path_factory.mktemp("sep")), config={"backend": "sqlite"})


@pytest.fixture(scope="module")
def report(indexed_ws: str, labelled: list[dict[str, Any]]) -> dict[str, Any]:
    return evaluate(indexed_ws, labelled, limit=10, intended_use="answer", scoring_instant="2026-10-09")


# ---------------------------------------------------------------------------
# The measurement on the fixture
# ---------------------------------------------------------------------------


def test_the_statistic_separates_present_from_absent_queries(report: dict[str, Any]) -> None:
    assert report["present"] == report["absent"] == 40
    assert report["auc"] >= 0.85, report
    assert report["verdict"] == "separates"
    assert report["separation"]["present"]["median"] > report["separation"]["absent"]["median"]


def test_at_the_answer_thresholds_absent_queries_are_not_acted_on(report: dict[str, Any]) -> None:
    """The costly error for a hallucination guardrail is ACTING on an absent
    answer. The shipped answer band must essentially never do so here, while
    still acting on a useful share of present queries."""
    rates = report["at_thresholds"]
    assert rates["act_rate_absent"] <= 0.05, rates
    assert rates["act_rate_present"] >= 0.5, rates
    assert rates["abstain_rate_absent"] > rates["abstain_rate_present"], rates


def test_the_report_is_deterministic(indexed_ws: str, labelled: list[dict[str, Any]], report: dict[str, Any]) -> None:
    again = evaluate(indexed_ws, labelled, limit=10, intended_use="answer", scoring_instant="2026-10-09")
    assert again == report


def test_the_evaluation_leaves_the_served_ledger_alone(tmp_path: Any, labelled: list[dict[str, Any]]) -> None:
    """A measurement run is not a serve: it must not append the 80 candidate
    sets nobody was ever handed to the ledger the outcome join reads."""
    import os

    ws = make_workspace(str(tmp_path), config={"backend": "sqlite"})
    evaluate(ws, labelled[:6] + labelled[-6:], limit=10, intended_use="summary", scoring_instant="2026-10-09")
    ledger = os.path.join(ws, ".mind-mem-ledger", "served.jsonl")
    assert not os.path.exists(ledger) or os.path.getsize(ledger) == 0
    # Positive control: a real serve through the same door DOES write a row.
    from mind_mem.recall import recall

    recall(ws, labelled[0]["query"], limit=5, scoring_instant="2026-10-09")
    assert os.path.getsize(ledger) > 0


# ---------------------------------------------------------------------------
# The experiment can come out negative
# ---------------------------------------------------------------------------


def _fixed(scores: list[float]) -> Any:
    return lambda query: [{"_id": f"b{i}", "score": s} for i, s in enumerate(scores)]  # noqa: E731


def test_flipped_labels_flip_the_verdict(indexed_ws: str, labelled: list[dict[str, Any]], report: dict[str, Any]) -> None:
    flipped = [{**row, "present": not row["present"]} for row in labelled]
    wrong = evaluate(indexed_ws, flipped, limit=10, intended_use="answer", scoring_instant="2026-10-09")
    assert wrong["auc"] == pytest.approx(1.0 - report["auc"], abs=2e-4)  # both are rounded to 4 places
    assert wrong["verdict"] == "no_signal"


def test_an_uninformative_ranker_is_reported_as_no_signal(labelled: list[dict[str, Any]]) -> None:
    flat = evaluate("unused", labelled, serve=_fixed([5.0, 4.9, 4.8, 4.7]))
    assert flat["auc"] == pytest.approx(0.5)
    assert flat["verdict"] == "no_signal"
    assert "declined" in flat["detail"]


def test_a_random_ranker_is_not_called_separating() -> None:
    rng = random.Random(3)
    rows = [{"query": f"q{i}", "present": i % 2 == 0} for i in range(200)]
    scores: dict[str, list[float]] = {r["query"]: [rng.random() * 10 for _ in range(6)] for r in rows}
    out = evaluate("unused", rows, serve=lambda q: [{"_id": f"b{i}", "score": s} for i, s in enumerate(scores[q])])
    assert out["verdict"] in ("no_signal", "weak"), out
    assert out["auc"] < 0.75


def test_too_few_labels_is_insufficient_not_a_verdict() -> None:
    rows = [{"query": "a", "present": True}, {"query": "b", "present": False}]
    out = evaluate("unused", rows, serve=_fixed([9.0, 1.0]))
    assert out["verdict"] == "insufficient"
    assert out["auc"] is None or isinstance(out["auc"], float)


def test_one_class_only_has_no_auc() -> None:
    rows = [{"query": f"q{i}", "present": True} for i in range(10)]
    out = evaluate("unused", rows, serve=_fixed([9.0, 1.0]))
    assert out["auc"] is None
    assert out["verdict"] == "insufficient"


# ---------------------------------------------------------------------------
# auc()
# ---------------------------------------------------------------------------


def test_auc_known_answers() -> None:
    assert auc([3, 4, 5], [0, 1, 2]) == 1.0
    assert auc([0, 1, 2], [3, 4, 5]) == 0.0
    assert auc([1, 1], [1, 1]) == 0.5
    assert auc([2, 0], [1]) == pytest.approx(0.5)
    assert auc([], [1.0]) is None
    assert auc([1.0], []) is None


def test_auc_is_symmetric_under_label_swap() -> None:
    rng = random.Random(11)
    pos = [rng.random() for _ in range(30)]
    neg = [rng.random() * 0.8 for _ in range(25)]
    forward = auc(pos, neg)
    backward = auc(neg, pos)
    assert forward is not None and backward is not None
    assert math.isclose(forward + backward, 1.0, abs_tol=1e-12)


# ---------------------------------------------------------------------------
# Input boundary
# ---------------------------------------------------------------------------


def test_load_labelled_accepts_jsonl_and_rejects_bad_rows(tmp_path: Any) -> None:
    good = tmp_path / "ok.jsonl"
    good.write_text(
        json.dumps({"query": "what is x", "present": True}) + "\n\n" + json.dumps({"query": "what is y", "present": False}) + "\n",
        encoding="utf-8",
    )
    assert load_labelled(str(good)) == [{"query": "what is x", "present": True}, {"query": "what is y", "present": False}]

    for bad_line in (
        "not json",
        json.dumps({"query": "q"}),
        json.dumps({"present": True}),
        json.dumps({"query": "", "present": True}),
        json.dumps({"query": "q", "present": "yes"}),
        json.dumps(["q", True]),
    ):
        path = tmp_path / "bad.jsonl"
        path.write_text(bad_line + "\n", encoding="utf-8")
        with pytest.raises(ValueError):
            load_labelled(str(path))

    with pytest.raises(ValueError):
        load_labelled(str(tmp_path / "missing.jsonl"))


def test_the_cli_runs_the_experiment_and_exits_non_zero_on_no_signal(
    indexed_ws: str, labelled: list[dict[str, Any]], tmp_path: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from mind_mem import mm_cli

    monkeypatch.setenv("MIND_MEM_WORKSPACE", indexed_ws)
    path = tmp_path / "labels.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in labelled) + "\n", encoding="utf-8")
    assert mm_cli.main(["abstention-eval", "--labelled", str(path), "--scoring-instant", "2026-10-09"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["verdict"] == "separates"

    flipped = tmp_path / "flipped.jsonl"
    flipped.write_text("\n".join(json.dumps({**row, "present": not row["present"]}) for row in labelled) + "\n", encoding="utf-8")
    assert mm_cli.main(["abstention-eval", "--labelled", str(flipped), "--scoring-instant", "2026-10-09"]) == 3
    assert json.loads(capsys.readouterr().out)["verdict"] == "no_signal"

    assert mm_cli.main(["abstention-eval", "--labelled", str(tmp_path / "nope.jsonl")]) == 2
